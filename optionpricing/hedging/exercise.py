"""Exercise dates of the American puts in a scenario bank: the day `τ` each holder ends the contract.

The hedger is short the put, so it does not choose `τ`: the holder does, and the hedge must live with it. How the
holder chooses is a setting of the dataset, so suboptimal holders are a test scenario:
- `tree`: the optimal holder. It exercises the first day the price is at or below the early-exercise boundary
  `S*` of the binomial tree.
- `late`: a holder who waits too long. It uses the tree's boundary shifted down by a fraction `shift`, so it
  exercises later than optimal, or not at all.
- `lsm`: the Longstaff-Schwartz rule, fitted on each contract's own paths, as the legacy hedger did. The rule is
  fitted on the paths it is then applied to, so it sees their future: it is kept to compare with the legacy
  results, not as a realistic holder.
- `never`: never exercises early.

The holder checks once a day, on the bank's daily grid. A put not exercised before expiry gets `τ` = its days to
expiry, whether it ends in or out of the money: settlement (`settlement.py`) pays it if it is in the money and
lets it lapse otherwise. Day 0 counts: a put whose `S_0` is already at or below the boundary is exercised at once.

Per unit of strike the boundary depends only on the time left, `r` and `σ`, so one tree per contract gives it
for every day of every path of that contract.
"""

import numpy as np

from optionpricing.hedging.simulator import DAYS_PER_YEAR, ScenarioBank
from optionpricing.pricing.american.batch_binomial_tree import exercise_boundary
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz

METHODS = ("tree", "late", "lsm", "never")


def daily_boundary(bank: ScenarioBank, nb_steps: int = 1000, chunk_size: int | None = None) -> np.ndarray:
    """Early-exercise boundary `S*` of each contract on every day to its expiry, in dollars.

    One tree per contract (`exercise_boundary`) gives `S*/K` at its `nb_steps + 1` levels. Day `k` of a contract
    falls between two levels unless the steps divide its days evenly, so the boundary is read off by linear
    interpolation in time. Levels are `days / nb_steps` apart, so the nearest one is at most 0.18 day from a
    daily date at 1000 steps, for contracts up to a year. Trees whose levels land exactly on days are no more
    accurate: the error comes from the tree's step size, not from the interpolation
    (`docs/results/hedging-training.md`, "The exercise boundary").

    Args:
        bank: the scenario bank whose contracts are used.
        nb_steps: tree steps per contract, as in `exercise_boundary`.
        chunk_size: contracts stepped together per pass, as in `exercise_boundary`. Only affects speed and
            memory.

    Returns:
        An array of shape `(n_contracts, max(days) + 1)`, one row per contract on the columns of `bank.spot`.
        Entry `k` is `S*` on day `k`. The expiry entry is the strike, later entries are NaN, and so are all
        entries before expiry for a contract with `r <= 0`, whose put is never exercised early.

    Raises:
        ValueError: for an invalid `nb_steps` or `chunk_size`, as `exercise_boundary`.
    """
    per_strike = exercise_boundary(
        bank.maturity, bank.risk_free_rate, bank.sigma, nb_steps=nb_steps, chunk_size=chunk_size
    )
    # The bank's daily grid, 0, 1, ..., max(days): one entry per column of `bank.spot`, shared by every
    # contract. These are the days the boundary is wanted on.
    days = np.arange(bank.spot.shape[1])
    boundary = np.full((len(bank.days), len(days)), np.nan)
    for i, contract_days in enumerate(bank.days):
        # `contract_days`: this contract's days to expiry, so its tree spans days 0 to `contract_days`.
        # `level_days`: the day each of its tree's levels falls on, level j at `j * contract_days / nb_steps`.
        level_days = np.linspace(0, contract_days, nb_steps + 1)
        # The days of the grid on which this contract is still alive, expiry included. Later days stay NaN.
        alive = days <= contract_days
        boundary[i, alive] = np.interp(days[alive], level_days, per_strike[i])
    return boundary * bank.strike


def check_method(method: str, shift: float | None = None) -> None:
    """Check a holder's settings as `exercise_days` does, so a caller can refuse bad ones before any work starts.

    Args:
        method: one of `METHODS`.
        shift: a fraction in (0, 1) for `late`, None for every other method.

    Raises:
        ValueError: for an unknown `method`, or a `shift` missing or outside (0, 1) with `late` or given with
            another method.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if method == "late":
        if shift is None or not 0 < shift < 1:
            raise ValueError(f"method 'late' needs a shift in (0, 1), got {shift}")
    elif shift is not None:
        raise ValueError(f"shift is only used by method 'late', got shift={shift} with {method!r}")


def exercise_days(
    bank: ScenarioBank,
    method: str,
    boundary: np.ndarray | None = None,
    shift: float | None = None,
) -> np.ndarray:
    """Day `τ` on which each scenario's put is exercised, or its expiry if it never is early.

    Args:
        bank: the scenario bank.
        method: how the holder decides, one of `METHODS` (see the module docstring).
        boundary: `daily_boundary(bank)`, for `tree` and `late`. Computed with its defaults when None, so a
            caller that also stores the boundary passes it to avoid a second set of trees.
        shift: for `late` only, the fraction in (0, 1) the boundary is lowered by: the holder exercises at
            `(1 - shift) S*`.

    Returns:
        An int array of shape `(n_scenarios,)`, each entry between 0 and its contract's days to expiry.

    Raises:
        ValueError: for an unknown `method`, a `shift` missing or outside (0, 1) with `late` or given with
            another method, a `boundary` given with `lsm` or `never`, or a `boundary` of the wrong shape.
    """
    check_method(method, shift)
    if boundary is not None:
        if method not in ("tree", "late"):
            raise ValueError(f"boundary is only used by methods 'tree' and 'late', got {method!r}")
        expected = (len(bank.days), bank.spot.shape[1])
        if boundary.shape != expected:
            raise ValueError(f"boundary must have shape {expected} (contracts, days), got {boundary.shape}")

    expiry = bank.days[bank.contract]
    if method == "never":
        return expiry.copy()
    if method == "lsm":
        return _lsm_exercise_days(bank)
    if boundary is None:
        boundary = daily_boundary(bank)
    threshold = boundary if method == "tree" else (1 - shift) * boundary
    return _first_day_at_or_below(bank.spot, threshold[bank.contract], expiry)


def _first_day_at_or_below(spot: np.ndarray, threshold: np.ndarray, expiry: np.ndarray) -> np.ndarray:
    """First day with `spot <= threshold`, per row, and `expiry` where there is none.

    NaN entries never count as a crossing, so the days past expiry (NaN spot) and a missing boundary (NaN
    threshold, `r <= 0`) are skipped. A first crossing on the expiry day itself gives `expiry`, the same as no
    crossing: the put ends then anyway, and settlement decides whether it pays.
    """
    with np.errstate(invalid="ignore"):
        crossed = spot <= threshold
    return np.where(crossed.any(axis=1), crossed.argmax(axis=1), expiry)


def _lsm_exercise_days(bank: ScenarioBank) -> np.ndarray:
    """Exercise days of the plain Longstaff-Schwartz rule, fitted on each contract's own paths up to its expiry.

    The defaults of `LongstaffSchwartz` are kept (cubic `simple` basis, no control variate), as the legacy
    hedger used. With few paths per contract the regression has little to fit, so the rule is poor: it
    reproduces the legacy setup, it is not a good holder.
    """
    tau = np.empty(len(bank.contract), dtype=int)
    for i, contract_days in enumerate(bank.days):
        # `contract_days`: this contract's days to expiry. Its paths are cut there, so LSM's last step is its
        # expiry and its `exercise_time` counts in days. `rows`: the scenarios that belong to this contract.
        rows = np.flatnonzero(bank.contract == i)
        lsm = LongstaffSchwartz(
            bank.spot[rows, : contract_days + 1],
            strike=bank.strike,
            risk_free_rate=bank.risk_free_rate[i],
            maturity=contract_days / DAYS_PER_YEAR,
            option_type="put",
            compute_greeks=False,
        )
        lsm.predict_payoff()
        tau[rows] = lsm.exercise_time
    return tau
