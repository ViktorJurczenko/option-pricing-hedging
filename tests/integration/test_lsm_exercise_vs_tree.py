import numpy as np
import pytest

from Hedging.option_value import binomial_value_fn
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from core.gbm import GBM

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
SAMPLING, N_PATHS = 52, 5000


def _tree_exercise_rows(simulation):
    """Tree-optimal exercise row per path: the first row before expiry where
    the put is in the money and the CRR value has collapsed to intrinsic."""
    n_rows = simulation.shape[0]
    value_fn = binomial_value_fn(S0, K, R, SIGMA, T, n_rows, "put", substeps=5)
    rows = np.full(simulation.shape[1], n_rows - 1)
    undecided = np.ones(simulation.shape[1], dtype=bool)
    for t in range(n_rows - 1):
        S = simulation[t]
        intrinsic = np.maximum(K - S, 0.0)
        stop = undecided & (intrinsic > 0) & (value_fn(t, S) <= intrinsic + 1e-6)
        rows[stop] = t
        undecided &= ~stop
    return rows


@pytest.mark.parametrize("seed", [1, 42])
def test_lsm_put_exercise_decisions_agree_with_tree_optimal_rule(seed):
    sim = GBM(sigma=SIGMA, start_price=S0, risk_free_rate=R, seed=seed).simulate_paths(N_PATHS, T, SAMPLING)
    lsm = LongstaffSchwartz(sim, K, R, T, "put", "simple", 3, sigma=SIGMA)
    lsm.predict_payoff()
    last = sim.shape[0] - 1

    lsm_early = lsm.exercise_time < last
    tree_early = _tree_exercise_rows(sim) < last

    # Measured agreement 95-97% here (seeds 1,2,3,42; 93-98% on 10000x365); 85% leaves margin
    # for this coarser 52-step grid while still failing on a wrong-horizon or
    # broken-regression LSM (chance agreement is ~50%).
    assert (lsm_early == tree_early).mean() > 0.85
    # Measured fraction gap here is 0.005-0.04 (LSM 40-43% vs tree 38-40%); 0.08 leaves margin.
    assert abs(lsm_early.mean() - tree_early.mean()) < 0.08
