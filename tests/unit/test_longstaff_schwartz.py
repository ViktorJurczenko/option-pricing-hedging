import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanOptionPricing.basis_functions import simple_poly

def test_calculate_option_price_is_mean_of_time_zero_payoff():
    simulation = np.ones((3, 5)) * 100.0
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0, option_type="put")
    lsm.payoff[0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    assert lsm.calculate_option_price() == pytest.approx(3.0)

def test_terminal_payoff_matches_vanilla_put_payoff():
    rng = np.random.default_rng(0)
    simulation = rng.uniform(80, 120, size=(4, 20))
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0, option_type="put", max_degree=2)
    lsm.predict_payoff()

    expected_terminal = np.maximum(100.0 - simulation[-1], 0)
    np.testing.assert_allclose(lsm.payoff[-1], expected_terminal)

def test_path_that_is_always_out_of_the_money_never_exercises_early():
    # Put, strike 100, price constant at 200 the whole time -> always OTM
    simulation = np.full((4, 3), 200.0)
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0, option_type="put", max_degree=2)
    lsm.predict_payoff()

    assert np.all(lsm.exercise_time == simulation.shape[0] - 1)
    assert np.all(lsm.payoff[0] == 0)

def test_deep_itm_put_exercises_immediately():
    # Put, strike 100, price constant at 1 the whole time -> always deep ITM,
    # exercising now is always at least as good as any regression-estimated continuation.
    simulation = np.full((4, 3), 1.0)
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0, option_type="put", max_degree=2)
    lsm.predict_payoff()

    assert lsm.calculate_option_price() == pytest.approx(99.0)

@pytest.mark.parametrize("option_type,reg_type,max_degree", [
    ("invalid", "simple", 2),
])
def test_invalid_option_type_raises(option_type, reg_type, max_degree):
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type=option_type, reg_type=reg_type, max_degree=max_degree)

def test_invalid_reg_type_raises():
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type="put", reg_type="invalid")

def test_negative_max_degree_raises():
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type="put", max_degree=-1)

def test_negative_ridge_lambda_raises():
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type="put", ridge_lambda=-0.01)

def test_moneyness_bins_below_one_raises():
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type="put", moneyness_bins=0)

@pytest.mark.parametrize("window", [-0.01, 1.01])
def test_moneyness_bins_window_out_of_unit_range_raises(window):
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), option_type="put", moneyness_bins_window=window)


# --- ridge_lambda / moneyness_bins regression coverage (BUGS.md: LSM
# delta/gamma estimates unreliable near maturity) ---

def test_fit_model_default_ridge_lambda_matches_plain_linear_regression():
    """_fit_model's ridge_lambda==0.0 branch must call plain LinearRegression,
    not Ridge(alpha=0) -- sklearn does not guarantee those are bit-identical,
    and the whole "default preserves prior behavior exactly" guarantee rests
    on this exact branch, not just on alpha numerically being 0."""
    rng = np.random.default_rng(0)
    X = rng.uniform(50, 150, size=(30, 4))
    y = rng.uniform(0, 20, size=30)

    lsm = LongstaffSchwartz(np.ones((2, 2)), option_type="put", ridge_lambda=0.0)
    model = lsm._fit_model(X, y)
    reference = LinearRegression(fit_intercept=False).fit(X, y)

    assert isinstance(model, LinearRegression)
    np.testing.assert_array_equal(model.coef_, reference.coef_)

def test_fit_model_positive_ridge_lambda_uses_ridge_and_changes_coefficients():
    """Regression test that the Ridge branch in _fit_model is actually
    reachable and does something: on the same (mildly collinear) data, a
    positive ridge_lambda must produce a different fit than the unregularized
    default, not silently fall through to LinearRegression."""
    rng = np.random.default_rng(1)
    x = rng.uniform(50, 150, size=40)
    X = simple_poly(x, 3)  # collinear polynomial features -> ridge should visibly shrink coefficients
    y = rng.uniform(0, 20, size=40)

    lsm_plain = LongstaffSchwartz(np.ones((2, 2)), option_type="put", ridge_lambda=0.0)
    lsm_ridge = LongstaffSchwartz(np.ones((2, 2)), option_type="put", ridge_lambda=5.0)

    model_plain = lsm_plain._fit_model(X, y)
    model_ridge = lsm_ridge._fit_model(X, y)

    assert isinstance(model_ridge, Ridge)
    assert not np.allclose(model_plain.coef_, model_ridge.coef_)

def test_ridge_lambda_changes_delta_end_to_end():
    """The ridge branch must actually change fit()'s output through the full
    predict_payoff() pipeline, not just in the isolated _fit_model unit test
    above."""
    rng = np.random.default_rng(2)
    simulation = rng.uniform(80, 120, size=(6, 200))

    lsm_plain = LongstaffSchwartz(simulation, strike=100.0, option_type="call", max_degree=3)
    lsm_plain.predict_payoff()

    lsm_ridge = LongstaffSchwartz(simulation, strike=100.0, option_type="call", max_degree=3, ridge_lambda=1.0)
    lsm_ridge.predict_payoff()

    assert not np.allclose(lsm_plain.delta, lsm_ridge.delta)

def test_new_params_at_default_reproduce_prior_behavior_bit_for_bit():
    """ridge_lambda=0.0, moneyness_bins=1, moneyness_bins_window=1.0 (all
    explicit) must reproduce the plain-default constructor's delta/gamma
    bit-for-bit and price to floating-point tolerance -- the backward
    compatibility guarantee the whole feature is opt-in around."""
    rng = np.random.default_rng(3)
    simulation = rng.uniform(80, 120, size=(6, 50))

    lsm_old = LongstaffSchwartz(simulation, strike=100.0, option_type="put", max_degree=2)
    lsm_old.predict_payoff()

    lsm_new = LongstaffSchwartz(
        simulation, strike=100.0, option_type="put", max_degree=2,
        ridge_lambda=0.0, moneyness_bins=1, moneyness_bins_window=1.0,
    )
    lsm_new.predict_payoff()

    np.testing.assert_array_equal(lsm_old.delta, lsm_new.delta)
    np.testing.assert_array_equal(lsm_old.gamma, lsm_new.gamma)
    assert lsm_old.calculate_option_price() == pytest.approx(lsm_new.calculate_option_price())

def test_digitize_with_no_interior_edges_maps_every_value_to_bin_zero():
    """Pins down the numpy behavior the moneyness_bins=1 backward-compat
    guarantee rests on: at moneyness_bins=1, bin_edges has exactly 2 entries
    so interior_edges = bin_edges[1:-1] is empty, and np.digitize(x, [])
    returns 0 for every x regardless of magnitude -- this is *why*
    moneyness_bins=1 needs no separate code branch to behave like a single
    global fit."""
    values = np.array([-1000.0, 0.0, 50.0, 100.0, 1e6])
    result = np.digitize(values, np.array([]))
    np.testing.assert_array_equal(result, np.zeros(5, dtype=result.dtype))

def test_moneyness_bins_one_matches_global_fit_regardless_of_window():
    """moneyness_bins=1 must reduce to the single-global-fit behavior no
    matter what moneyness_bins_window is set to, since n_bins is only ever
    read as 1 in that case (near_maturity gating never has anything to
    gate)."""
    rng = np.random.default_rng(4)
    simulation = rng.uniform(80, 120, size=(8, 80))

    lsm_wide_window = LongstaffSchwartz(
        simulation, strike=100.0, option_type="call", max_degree=2,
        moneyness_bins=1, moneyness_bins_window=1.0,
    )
    lsm_wide_window.predict_payoff()

    lsm_narrow_window = LongstaffSchwartz(
        simulation, strike=100.0, option_type="call", max_degree=2,
        moneyness_bins=1, moneyness_bins_window=0.1,
    )
    lsm_narrow_window.predict_payoff()

    np.testing.assert_array_equal(lsm_wide_window.delta, lsm_narrow_window.delta)
    np.testing.assert_array_equal(lsm_wide_window.gamma, lsm_narrow_window.gamma)

def test_moneyness_bins_window_gates_binning_to_near_maturity_timesteps_only():
    """moneyness_bins > 1 changes fit()'s single-timestep regression only
    when that timestep's t_frac_to_maturity falls inside moneyness_bins_window
    of maturity; outside the window, fit() must produce the exact same result
    as moneyness_bins=1 (single global fit) for identical inputs, since
    applying binning to the well-behaved middle of the timeline was found to
    make it worse (see BUGS.md).

    Calls fit() directly (not predict_payoff()) with a shared payoff_next
    array for an early and a late index, so the comparison isolates this
    single fit() call -- going through predict_payoff() would confound it,
    since binning at a near-maturity index changes that index's payoff,
    which then cascades backward into every earlier index's payoff_next
    through backward induction, even though those earlier fits themselves
    stay on the n_bins=1 code path.
    """
    n_steps = 20
    window = 0.3
    rng = np.random.default_rng(7)
    n_paths = 1000
    strike = 100.0
    S_t = rng.uniform(80, 150, size=n_paths)  # ~71% ITM for a call at K=100
    payoff_next = np.maximum(rng.uniform(80, 150, size=n_paths) - strike, 0)

    def make_lsm(n_bins):
        simulation = np.tile(S_t, (n_steps, 1))
        return LongstaffSchwartz(
            simulation, strike=strike, option_type="call", max_degree=2,
            moneyness_bins=n_bins, moneyness_bins_window=window,
        )

    early_index = 0            # t_frac = 1/20 = 0.05, well outside the 0.3 window
    late_index = n_steps - 2   # t_frac = 19/20 = 0.95, inside the 0.3 window

    lsm_binned_early = make_lsm(4)
    lsm_global_early = make_lsm(1)
    lsm_binned_early.fit(payoff_next, early_index)
    lsm_global_early.fit(payoff_next, early_index)
    np.testing.assert_array_equal(lsm_binned_early.delta[early_index], lsm_global_early.delta[early_index])
    np.testing.assert_array_equal(lsm_binned_early.gamma[early_index], lsm_global_early.gamma[early_index])

    lsm_binned_late = make_lsm(4)
    lsm_global_late = make_lsm(1)
    lsm_binned_late.fit(payoff_next, late_index)
    lsm_global_late.fit(payoff_next, late_index)
    # Inside the window, binning must actually change the fit -- otherwise
    # this test wouldn't distinguish "gating works" from "binning never does
    # anything."
    assert not np.allclose(lsm_binned_late.delta[late_index], lsm_global_late.delta[late_index])

def test_min_points_per_bin_fallback_matches_single_global_fit():
    """When a timestep has too few ITM paths to support the requested bin
    count reliably (fewer than moneyness_bins * min_points_per_bin), fit()
    must silently fall back to a single global fit rather than erroring or
    fitting an underdetermined regression per bin."""
    rng = np.random.default_rng(6)
    n_paths = 20
    simulation = np.zeros((2, n_paths))
    simulation[0] = rng.uniform(95, 140, size=n_paths)  # mostly ITM for a call at K=100
    simulation[1] = rng.uniform(95, 140, size=n_paths)

    strike = 100.0
    payoff_next = np.maximum(simulation[1] - strike, 0)

    # min_points_per_bin = max(max_degree + 2, 10) = 10; requesting 5 bins
    # needs >= 50 ITM paths, which this fixture deliberately doesn't have.
    n_itm = int(np.sum(simulation[0] > strike))
    assert n_itm < 5 * max(2 + 2, 10)

    lsm_many_bins = LongstaffSchwartz(
        simulation, strike=strike, option_type="call", max_degree=2,
        moneyness_bins=5, moneyness_bins_window=1.0,
    )
    lsm_many_bins.fit(payoff_next, 0)

    lsm_one_bin = LongstaffSchwartz(
        simulation, strike=strike, option_type="call", max_degree=2,
        moneyness_bins=1, moneyness_bins_window=1.0,
    )
    lsm_one_bin.fit(payoff_next, 0)

    np.testing.assert_array_equal(lsm_many_bins.continue_val[0], lsm_one_bin.continue_val[0])
    np.testing.assert_array_equal(lsm_many_bins.delta[0], lsm_one_bin.delta[0])
    np.testing.assert_array_equal(lsm_many_bins.gamma[0], lsm_one_bin.gamma[0])

def test_out_of_the_money_paths_get_continuation_value_via_clipped_bin_assignment():
    """OTM paths never enter the per-bin regression (only ITM paths do), but
    they still need a continuation-value/Greeks estimate for predict_payoff's
    exercise decision. They must be clipped into the nearest existing bin
    (np.clip(np.digitize(...), 0, n_bins-1)) rather than left at their
    zero-initialized default."""
    rng = np.random.default_rng(8)
    n_itm, n_otm = 200, 50
    strike = 100.0
    S_t_itm = rng.uniform(100.5, 150.0, size=n_itm)
    S_t_otm = rng.uniform(50.0, 99.5, size=n_otm)
    S_t = np.concatenate([S_t_itm, S_t_otm])
    S_next = S_t * (1 + rng.normal(0, 0.02, size=S_t.shape[0]))
    simulation = np.vstack([S_t, S_next])
    payoff_next = np.maximum(S_next - strike, 0)

    n_bins = 4
    lsm = LongstaffSchwartz(
        simulation, strike=strike, risk_free_rate=0.05, maturity=1.0,
        option_type="call", max_degree=2,
        moneyness_bins=n_bins, moneyness_bins_window=1.0,
    )
    lsm.fit(payoff_next, 0)

    otm_mask = S_t < strike
    itm_mask = S_t > strike
    # Would spuriously be all zeros if OTM paths were left at their
    # zero-initialized default instead of being assigned a real bin.
    assert not np.allclose(lsm.continue_val[0][otm_mask], 0.0)

    # Every OTM price is below the smallest ITM price, hence below every
    # interior bin edge (which is quantiles of ITM prices only) -> digitize
    # must map all of them to bin 0.
    S_t_norm = S_t / strike
    bin_edges = np.quantile(S_t_norm[itm_mask], np.linspace(0, 1, n_bins + 1))
    interior_edges = bin_edges[1:-1]
    bin_idx_otm = np.digitize(S_t_norm[otm_mask], interior_edges)
    assert np.all(bin_idx_otm == 0)

    # Refit bin 0's model independently and confirm lsm's OTM continuation
    # values match its predictions exactly -- i.e. OTM paths were actually
    # routed through the bin-0 fit, not just nonzero by coincidence.
    bin_idx_itm = np.digitize(S_t_norm[itm_mask], interior_edges)
    dt = lsm.maturity / simulation.shape[0]
    true_continue_itm = np.exp(-lsm.risk_free_rate * dt) * payoff_next[itm_mask]
    bin0_mask = bin_idx_itm == 0
    model_bin0 = lsm._fit_model(simple_poly(S_t_norm[itm_mask][bin0_mask], 2), true_continue_itm[bin0_mask])
    expected_otm = model_bin0.predict(simple_poly(S_t_norm[otm_mask], 2))
    np.testing.assert_allclose(lsm.continue_val[0][otm_mask], expected_otm)
