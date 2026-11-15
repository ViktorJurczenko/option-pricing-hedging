"""The baselines on simulated American puts, hedged at tree deltas against an optimal holder.

This checks the whole chain the hedging results rest on: the scenario bank, the tree's exercise days, tree
greeks along each path, `hedge_pnl` and the baselines. Under the risk-neutral drift, with the holder exercising at
the tree's boundary and the tree price as premium, the premium is the expected discounted payoff, and the
discounted share gains are a martingale. So every strategy's mean P&L is about 0 at zero cost, and only its
spread tells strategies apart.
"""

import math

import numpy as np
import pytest
import torch

from optionpricing.hedging import baselines as b
from optionpricing.hedging.exercise import exercise_days
from optionpricing.hedging.pnl import hedge_pnl
from optionpricing.hedging.risk import cvar
from optionpricing.hedging.simulator import DAYS_PER_YEAR, simulate_scenarios
from optionpricing.pricing.american.batch_binomial_tree import price_put_batch

# 200 tree steps for the greeks along the paths: 86k queries in about 12 s instead of a minute at 1000. The
# deltas move by less than the daily hedging error, which is what this test measures.
GREEK_STEPS = 200
N_PATHS = 2000


@pytest.fixture(scope="module")
def hedge_args():
    """`hedge_pnl` arguments for 4 contracts of 30 to 60 days, 500 paths each, with tree greeks as baseline inputs."""
    bank = simulate_scenarios(4, N_PATHS // 4, 7, (30, 60), (0.15, 0.3), (0.02, 0.06), (0.9, 1.1))
    tau = exercise_days(bank, "tree")
    strike, contract = bank.strike, bank.contract
    n_days = bank.spot.shape[1]
    days = np.arange(n_days)
    time_left = (bank.days[contract][:, None] - days) / DAYS_PER_YEAR
    rate = np.repeat(bank.risk_free_rate[contract][:, None], n_days, axis=1)
    sigma = np.repeat(bank.sigma[contract][:, None], n_days, axis=1)
    hedged = days < tau[:, None]  # the hedge dates; the policy never sees the other days
    delta, gamma = np.full((2, *bank.spot.shape), np.nan)
    res = price_put_batch(
        bank.spot[hedged] / strike, time_left[hedged], rate[hedged], sigma[hedged], nb_steps=GREEK_STEPS, chunk_size=32
    )
    delta[hedged], gamma[hedged] = res.delta, res.gamma / strike
    premium = strike * price_put_batch(bank.moneyness, bank.maturity, bank.risk_free_rate, bank.sigma).price
    return {
        "inputs": b.baseline_inputs(bank.spot, delta, gamma, time_left, rate),
        "spot": torch.tensor(bank.spot),
        "tau": torch.tensor(tau),
        "premium": torch.tensor(premium[contract]),
        "strike": strike,
        "rate": torch.tensor(bank.risk_free_rate[contract]),
    }


def standard_error(pnl: torch.Tensor) -> float:
    return pnl.std().item() / math.sqrt(len(pnl))


@pytest.mark.parametrize("policy", [b.no_hedge, b.delta_hedge], ids=["no_hedge", "delta"])
def test_mean_pnl_is_zero_at_zero_cost(hedge_args, policy):
    # The premium is the expected discounted payoff only up to the tree's error and to the holder checking daily
    # rather than continuously, both small next to the daily delta hedge's standard error (0.009 on these paths
    # (`docs/results/hedging-training.md`, "Baseline checks")). 4 SE is a failure for a real bias.
    pnl = hedge_pnl(policy, **hedge_args).pnl
    assert abs(pnl.mean().item()) < 4 * standard_error(pnl)


def test_delta_hedge_removes_most_of_the_risk_and_its_error_grows_like_sqrt_dt(hedge_args):
    unhedged = hedge_pnl(b.no_hedge, **hedge_args).pnl
    daily = hedge_pnl(b.delta_hedge, **hedge_args).pnl
    every4 = hedge_pnl(b.delta_hedge, **hedge_args, rebalance_every=4).pnl
    # Measured ratio 0.081 (`docs/results/hedging-training.md`, "Baseline checks"). 0.15 leaves room. A sign error
    # in delta would hold -delta, which adds the option's moves instead of cancelling them, so it would roughly
    # double the error of not hedging.
    assert daily.std() < 0.15 * unhedged.std()
    # The discrete hedging error scales like sqrt(dt), so rebalancing every 4 days doubles it (measured 1.96).
    # The two share their paths, so the ratio is tighter than either std. 15% allows the extra error near the
    # boundary, where the hedge ends at an exercise date that falls between two hedge dates.
    assert (every4.std() / daily.std()).item() == pytest.approx(2.0, rel=0.15)


@pytest.mark.parametrize(
    "band",
    # the Whalley-Wilmott band is built for the 1% cost it is charged
    [b.whalley_wilmott(1.0, 0.01), b.band_hedge(0.1)],
    ids=["whalley_wilmott", "fixed_band"],
)
def test_bands_beat_the_delta_hedge_under_proportional_costs(hedge_args, band):
    # Measured (`docs/results/hedging-training.md`, "Baseline checks"): the bands' CVaR 95% is ~2.0 above the
    # delta hedge's, with a paired bootstrap standard error of 0.09-0.11, so a margin of 1.0 is ~9
    # standard errors below the measured gap. They win by trading less: their costs are 0.49 and 0.54 of the
    # delta hedge's, against the bound of 0.75.
    costs = {"proportional": 0.01}
    delta = hedge_pnl(b.delta_hedge, **hedge_args, **costs)
    out = hedge_pnl(band, **hedge_args, **costs)
    assert cvar(out.pnl) > cvar(delta.pnl) + 1.0
    assert out.costs.mean() < 0.75 * delta.costs.mean()


def test_tune_finds_an_interior_band_width_under_costs(hedge_args):
    # Too narrow overtrades, too wide leaves the put unhedged, so the best CVaR lies inside the grid.
    def score(policy):
        return cvar(hedge_pnl(policy, **hedge_args, proportional=0.01).pnl).item()

    grid = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    best, scores = b.tune(b.band_hedge, grid, score)
    assert best not in (grid[0], grid[-1])
    assert scores[grid.index(best)] == max(scores)
