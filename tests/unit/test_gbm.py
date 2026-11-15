"""Tests for optionpricing.core.gbm: GBM paths, time-last layout, per-contract parameters and seeding."""

import re

import matplotlib

matplotlib.use("Agg")  # headless backend so plot_paths() never opens a window during tests

import numpy as np
import pytest

from optionpricing.core.gbm import GBM


def test_simulate_paths_shape_and_start_price():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1)
    simulation = gbm.simulate_paths(nb_predictions=50, maturity=1, sampling=252)

    assert simulation.shape == (50, 253)  # 50 paths, 252 intervals + the start step
    assert np.all(simulation[:, 0] == 100.0)
    assert np.all(simulation > 0)  # GBM paths can never go negative


def test_an_int_seed_reproduces_the_numbers_of_the_time_first_version():
    # End prices of the first 3 paths, from the committed code before the switch to time-last
    # (GBM(0.2, 100, 0.05, seed=42).simulate_paths(1000, 1.0, 365)[-1, :3]). Every seeded result in the docs
    # relies on these draws being unchanged.
    sim = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=42).simulate_paths(1000, 1.0, 365)
    np.testing.assert_array_equal(sim[:3, -1], [157.07997475459558, 124.7205830432862, 92.69736995813615])


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


def test_last_step_sits_exactly_at_maturity():
    # Regression: the grid used to have int(T*sampling) steps, so the last one
    # was at T*(steps-1)/steps and every consumer priced a too-short option.
    # With ~zero volatility the path is deterministic: S_T = S0 * exp(r*T).
    sim = GBM(sigma=1e-9, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(3, maturity=2.0, sampling=52)
    assert sim.shape == (3, 2 * 52 + 1)
    np.testing.assert_allclose(sim[:, -1], 100.0 * np.exp(0.05 * 2.0), rtol=1e-6)


def test_non_integer_interval_count_uses_truncated_intervals_and_matching_dt():
    # maturity*sampling = 15.6 -> 15 intervals of 0.3/15, still ending at T.
    sim = GBM(sigma=1e-9, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(2, maturity=0.3, sampling=52)
    assert sim.shape == (2, 16)
    np.testing.assert_allclose(sim[:, 1], 100.0 * np.exp(0.05 * 0.3 / 15), rtol=1e-6)
    np.testing.assert_allclose(sim[:, -1], 100.0 * np.exp(0.05 * 0.3), rtol=1e-6)


def test_log_return_variance_scales_with_full_maturity():
    # Var[log(S_T/S0)] = sigma^2 * T only if the last step is at T.
    # 20000 paths -> sample variance rel. std ~ sqrt(2/n) = 1%; 5% tolerance.
    sim = GBM(sigma=0.3, start_price=100.0, risk_free_rate=0.02, seed=3).simulate_paths(
        20000, maturity=0.25, sampling=8
    )
    assert np.var(np.log(sim[:, -1] / 100.0)) == pytest.approx(0.3**2 * 0.25, rel=0.05)


@pytest.mark.parametrize("maturity,sampling", [(0.25, 2), (1, 0), (0.01, 50)])
def test_fewer_than_one_interval_raises(maturity, sampling):
    with pytest.raises(ValueError, match=">= 1"):
        GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05).simulate_paths(5, maturity, sampling)


@pytest.mark.parametrize("days", [3, 53, 96, 191])
def test_a_whole_number_of_days_is_not_truncated_by_float_error(days):
    # Regression: days / 365 * 365 is just below `days` in floating point for these values, and int() used to
    # drop a step, so the grid had days - 1 intervals of the wrong length.
    sim = GBM(sigma=1e-9, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(2, days / 365, 365)
    assert sim.shape == (2, days + 1)
    # sigma = 1e-9 leaves noise of ~1e-9 relative; a wrong step length (1/364 vs 1/365 of 5%) is ~4e-7
    np.testing.assert_allclose(sim[:, 1], 100.0 * np.exp(0.05 / 365), rtol=1e-8)


# per-contract parameters


def test_per_contract_parameters_give_a_contract_axis_first():
    sim = GBM(sigma=np.array([0.1, 0.2, 0.3]), seed=1).simulate_paths(4, maturity=1.0, sampling=12)
    assert sim.shape == (3, 4, 13)  # contract, path, step


def test_each_contract_follows_its_own_start_price_and_rate():
    # with ~zero volatility every path is deterministic: S_T = S0 * exp(r * T), with its contract's S0 and r
    start, rate = np.array([90.0, 100.0, 110.0]), np.array([0.02, 0.04, 0.06])
    sim = GBM(sigma=1e-9, start_price=start, risk_free_rate=rate, seed=1).simulate_paths(2, maturity=0.5, sampling=12)
    np.testing.assert_array_equal(sim[:, :, 0], np.repeat(start[:, None], 2, axis=1))
    expected = np.repeat((start * np.exp(rate * 0.5))[:, None], 2, axis=1)
    np.testing.assert_allclose(sim[:, :, -1], expected, rtol=1e-7)  # sigma = 1e-9 noise is ~1e-9


def test_each_contract_follows_its_own_volatility():
    # 2 contracts at 10% and 40%, 10000 paths each -> sample variance rel. std ~ sqrt(2/n) = 1.4%; 6% tolerance.
    # With 2 contracts and 2 paths the shapes would also line up the wrong way, so this pins sigma to contracts.
    sim = GBM(sigma=np.array([0.1, 0.4]), start_price=100.0, risk_free_rate=0.03, seed=4).simulate_paths(10000, 1.0, 4)
    log_return = np.log(sim[:, :, -1] / 100.0)
    assert np.var(log_return[0]) == pytest.approx(0.1**2, rel=0.06)
    assert np.var(log_return[1]) == pytest.approx(0.4**2, rel=0.06)


def test_sigma_belongs_to_the_contract_when_contracts_and_paths_are_equal_in_number():
    # 2 contracts x 2 paths: pairing sigma with paths instead of contracts would broadcast silently. Contract 0
    # has ~zero volatility, so both its paths are deterministic and equal; contract 1's two paths differ.
    sim = GBM(sigma=np.array([1e-12, 0.3]), start_price=100.0, risk_free_rate=0.0, seed=2).simulate_paths(2, 1.0, 10)
    np.testing.assert_allclose(sim[0], 100.0, rtol=1e-9)
    assert not np.allclose(sim[1, 0], sim[1, 1])


def test_paths_of_one_contract_differ():
    sim = GBM(sigma=np.array([0.2, 0.2]), seed=3).simulate_paths(3, 1.0, 10)
    assert not np.array_equal(sim[0, 0], sim[0, 1]) and not np.array_equal(sim[0, 0], sim[1, 0])


def test_a_number_and_per_contract_arrays_mix():
    sim = GBM(sigma=0.2, start_price=np.array([90.0, 110.0]), risk_free_rate=0.05, seed=1).simulate_paths(3, 1.0, 4)
    np.testing.assert_array_equal(sim[:, :, 0], [[90.0] * 3, [110.0] * 3])


def test_per_contract_arrays_of_different_lengths_raise():
    with pytest.raises(ValueError, match="must all have the same length"):
        GBM(sigma=np.ones(3), start_price=np.ones(4)).simulate_paths(5, 1.0, 10)


def test_a_two_dimensional_parameter_raises():
    with pytest.raises(ValueError, match=re.escape("a number or a 1-D array")):
        GBM(sigma=np.ones((2, 3))).simulate_paths(5, 1.0, 10)


def test_plot_paths_handles_per_contract_paths(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    gbm = GBM(sigma=np.array([0.1, 0.2]), seed=1)
    gbm.simulate_paths(3, 1.0, 4)
    gbm.plot_paths()  # 3-D simulations must be flattened to one line per path


def test_a_generator_seed_is_reproducible_and_leaves_the_global_state_alone():
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    sim1 = GBM(seed=np.random.default_rng(9)).simulate_paths(20, 1.0, 10)
    assert np.random.random() == expected_next  # the global stream did not move
    sim2 = GBM(seed=np.random.default_rng(9)).simulate_paths(20, 1.0, 10)
    np.testing.assert_array_equal(sim1, sim2)
