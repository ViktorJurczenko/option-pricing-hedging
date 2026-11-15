"""Seeded bank of hedging scenarios for the Deep Hedging dataset: put contracts and their daily price paths.

A scenario is one put contract (days to expiry, volatility, rate and moneyness `S_0 / K`) and one simulated
share-price path for it. Contracts are drawn from a family, so one policy is trained over many contracts
rather than one, and each contract gets `paths_per_contract` paths. The paths are simulated by `GBM` under the
risk-neutral drift `r`, with each contract's own `sigma` and `r`.

Maturities are whole numbers of days, so every scenario lives on the same daily grid: step `k` is time
`k / DAYS_PER_YEAR`, and a contract expires exactly at step `days`. Steps after a scenario's expiry are NaN, so
reading a price the contract never sees fails loudly instead of silently.

Prices are in dollars at a fixed strike (100 by default), the units of the hedging results.
"""

from dataclasses import dataclass

import numpy as np

from optionpricing.core.gbm import GBM

DAYS_PER_YEAR = 365  # calendar days: the hedger rebalances daily, and coarser grids are subsets of these dates


@dataclass(frozen=True)
class ScenarioBank:
    """Simulated scenarios: the drawn contracts and one price path per scenario.

    The contract arrays have one entry per contract. `contract` maps each scenario to its contract, so a value
    per scenario is `bank.sigma[bank.contract]`, and work done once per contract (a tree price, an exercise
    boundary) is shared by all its paths.

    Attributes:
        spot: share prices, shape `(n_scenarios, max(days) + 1)`, one row per scenario and time last. Step 0
            is `S_0`, step `days` of the scenario's contract is its expiry, and later steps are NaN.
        contract: index of each scenario's contract, shape `(n_scenarios,)`. Scenarios are grouped by
            contract: the first `paths_per_contract` belong to contract 0, and so on.
        days: days to expiry of each contract, shape `(n_contracts,)`, ints.
        sigma: volatility of each contract, shape `(n_contracts,)`.
        risk_free_rate: rate of each contract, shape `(n_contracts,)`.
        moneyness: `S_0 / K` of each contract, shape `(n_contracts,)`.
        strike: the strike shared by every contract, in dollars.
    """

    spot: np.ndarray
    contract: np.ndarray
    days: np.ndarray
    sigma: np.ndarray
    risk_free_rate: np.ndarray
    moneyness: np.ndarray
    strike: float

    @property
    def maturity(self) -> np.ndarray:
        """Maturity of each contract in years, `days / DAYS_PER_YEAR`."""
        return self.days / DAYS_PER_YEAR

    @property
    def times(self) -> np.ndarray:
        """Time in years of each step (column) of `spot`, shape `(max(days) + 1,)`."""
        return np.arange(self.spot.shape[1]) / DAYS_PER_YEAR


def _check_range(name: str, low_high: tuple[float, float], positive: bool) -> None:
    """Raise ValueError unless `low_high` is an ordered pair of finite values, > 0 if `positive`, else >= 0."""
    low, high = low_high
    in_domain = low > 0 if positive else low >= 0
    if not (np.isfinite(low) and np.isfinite(high) and in_domain and low <= high):
        bound = "0 < low" if positive else "0 <= low"
        raise ValueError(f"{name} must be finite with {bound} <= high, got {low_high}")


def simulate_scenarios(
    n_contracts: int,
    paths_per_contract: int,
    seed: int,
    days_range: tuple[int, int],
    sigma_range: tuple[float, float],
    rate_range: tuple[float, float],
    moneyness_range: tuple[float, float],
    strike: float = 100.0,
) -> ScenarioBank:
    """Draw `n_contracts` put contracts and simulate `paths_per_contract` daily price paths for each.

    Each contract is drawn uniformly and independently: its days to expiry among the whole numbers in
    `days_range` (both ends included), and `sigma`, `r` and `S_0 / K` in their ranges. The draws and the paths
    come from one generator seeded by `seed`, so the same arguments give the same bank, and the global
    `np.random` state is left alone.

    Args:
        n_contracts: number of contracts drawn, >= 1.
        paths_per_contract: price paths simulated for each contract, >= 1.
        seed: seed of the generator.
        days_range: shortest and longest days to expiry, whole numbers with 1 <= low <= high.
        sigma_range: lowest and highest volatility, > 0.
        rate_range: lowest and highest risk-free rate, >= 0.
        moneyness_range: lowest and highest `S_0 / K`, > 0.
        strike: strike of every contract, in dollars, > 0.

    Returns:
        The `ScenarioBank`, with `n_contracts * paths_per_contract` scenarios.

    Raises:
        ValueError: if a count or `strike` is out of range, `days_range` is not whole numbers with
            1 <= low <= high, or another range is not an ordered pair of finite values in its domain.
    """
    if n_contracts < 1:
        raise ValueError(f"n_contracts must be >= 1, got {n_contracts}")
    if paths_per_contract < 1:
        raise ValueError(f"paths_per_contract must be >= 1, got {paths_per_contract}")
    if not (strike > 0 and np.isfinite(strike)):
        raise ValueError(f"strike must be positive and finite, got {strike}")
    low_days, high_days = days_range
    if not (float(low_days).is_integer() and float(high_days).is_integer() and 1 <= low_days <= high_days):
        raise ValueError(f"days_range must be whole numbers with 1 <= low <= high, got {days_range}")
    _check_range("sigma_range", sigma_range, positive=True)
    _check_range("rate_range", rate_range, positive=False)
    _check_range("moneyness_range", moneyness_range, positive=True)

    rng = np.random.default_rng(seed)
    days = rng.integers(int(low_days), int(high_days), endpoint=True, size=n_contracts)
    sigma = rng.uniform(*sigma_range, size=n_contracts)
    rate = rng.uniform(*rate_range, size=n_contracts)
    moneyness = rng.uniform(*moneyness_range, size=n_contracts)

    horizon = int(days.max())
    gbm = GBM(sigma=sigma, start_price=moneyness * strike, risk_free_rate=rate, seed=rng)
    paths = gbm.simulate_paths(paths_per_contract, maturity=horizon / DAYS_PER_YEAR, sampling=DAYS_PER_YEAR)
    # (n_contracts, paths_per_contract, steps) -> one row per scenario, each contract's paths next to each other
    spot = paths.reshape(n_contracts * paths_per_contract, horizon + 1)
    contract = np.repeat(np.arange(n_contracts), paths_per_contract)
    spot[np.arange(horizon + 1) > days[contract][:, None]] = np.nan
    return ScenarioBank(spot, contract, days, sigma, rate, moneyness, float(strike))
