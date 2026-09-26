import matplotlib
matplotlib.use("Agg")  # headless backend so plot_paths() never opens a window during tests

import numpy as np
import pytest

from core.gbm import GBM

def test_simulate_paths_shape_and_start_price():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1)
    simulation = gbm.simulate_paths(nb_predictions=50, maturity=1, sampling=252)

    assert simulation.shape == (252, 50)
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
