import matplotlib
matplotlib.use("Agg")  # headless backend so plot_paths() never opens a window during tests

import numpy as np
import pytest

from optionpricing.core.gbm import GBM

def test_simulate_paths_shape_and_start_price():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1)
    simulation = gbm.simulate_paths(nb_predictions=50, maturity=1, sampling=252)

    assert simulation.shape == (253, 50)  # 252 intervals + the start row
    assert np.all(simulation[0] == 100.0)
    assert np.all(simulation > 0)  # GBM paths can never go negative

def test_same_seed_is_reproducible():
    sim1 = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=7).simulate_paths(100, 1, 50)
    sim2 = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=7).simulate_paths(100, 1, 50)

    np.testing.assert_array_equal(sim1, sim2)

def test_different_seeds_diverge():
    sim1 = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=1).simulate_paths(100, 1, 50)
    sim2 = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=2).simulate_paths(100, 1, 50)

    assert not np.array_equal(sim1, sim2)

def test_plot_paths_simulates_automatically_when_not_run_yet(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1)
    assert gbm.simulations is None

    gbm.plot_paths()  # must not raise AttributeError

    assert gbm.simulations is not None


def test_last_row_sits_exactly_at_maturity():
    # Regression: the grid used to have int(T*sampling) rows, so the last row
    # was at T*(rows-1)/rows and every consumer priced a too-short option.
    # With ~zero volatility the path is deterministic: S_T = S0 * exp(r*T).
    sim = GBM(sigma=1e-9, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(3, maturity=2.0, sampling=52)
    assert sim.shape == (2 * 52 + 1, 3)
    np.testing.assert_allclose(sim[-1], 100.0 * np.exp(0.05 * 2.0), rtol=1e-6)


def test_non_integer_interval_count_uses_truncated_intervals_and_matching_dt():
    # maturity*sampling = 15.6 -> 15 intervals of 0.3/15, still ending at T.
    sim = GBM(sigma=1e-9, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(2, maturity=0.3, sampling=52)
    assert sim.shape == (16, 2)
    np.testing.assert_allclose(sim[1], 100.0 * np.exp(0.05 * 0.3 / 15), rtol=1e-6)
    np.testing.assert_allclose(sim[-1], 100.0 * np.exp(0.05 * 0.3), rtol=1e-6)


def test_log_return_variance_scales_with_full_maturity():
    # Var[log(S_T/S0)] = sigma^2 * T only if the last row is at T.
    # 20000 paths -> sample variance rel. std ~ sqrt(2/n) = 1%; 5% tolerance.
    sim = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=3).simulate_paths(20000, maturity=0.25, sampling=8)
    assert np.var(np.log(sim[-1] / 100.0)) == pytest.approx(0.3**2 * 0.25, rel=0.05)


@pytest.mark.parametrize("maturity,sampling", [(0.25, 2), (1, 0), (0.01, 50)])
def test_fewer_than_one_interval_raises(maturity, sampling):
    with pytest.raises(ValueError, match=">= 1"):
        GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05).simulate_paths(5, maturity, sampling)
