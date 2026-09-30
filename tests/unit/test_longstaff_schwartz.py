import numpy as np
import pytest

from AmericanHedging.option_value import binomial_value_fn
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from core.gbm import GBM
from EuropeanOptionPricing.BlackScholes import BlackScholes

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


@pytest.mark.parametrize("sigma", [0.0, -0.2])
def test_non_positive_sigma_raises(sigma):
    with pytest.raises(ValueError):
        LongstaffSchwartz(np.ones((2, 2)), sigma=sigma)


# --- Delta/gamma via pathwise-delta regression (BUGS.md: differentiating the
# ITM-only continuation fit extrapolated to OTM paths, giving deltas in the tens) ---

SIGMA, R, K, T = 0.2, 0.05, 100.0, 1.0
N_STEPS, N_PATHS = 50, 4000


def _fitted_lsm(option_type="call", sigma=SIGMA, seed=42):
    sim = GBM(sigma=SIGMA, start_price=100.0, risk_free_rate=R, seed=seed).simulate_paths(N_PATHS, T, N_STEPS)
    lsm = LongstaffSchwartz(sim, K, R, T, option_type, "simple", 3, sigma=sigma)
    lsm.predict_payoff()
    return lsm


def _bs(lsm, t, option_type):
    # Same grid convention as the implementation: last row is expiry.
    tau = (N_STEPS - 1 - t) * T / N_STEPS
    return BlackScholes(SIGMA, lsm.simulation[t], K, R, option_type), tau


@pytest.fixture(scope="module")
def call_lsm():
    return _fitted_lsm("call")


@pytest.fixture(scope="module")
def put_lsm():
    return _fitted_lsm("put")


@pytest.mark.parametrize("t", [10, 25, 40, 47])
@pytest.mark.parametrize("moneyness", ["itm", "otm"])
def test_call_delta_close_to_black_scholes_itm_and_otm(call_lsm, t, moneyness):
    # An American call without dividends equals the European one, so BS delta is exact.
    # Tolerance 0.05 mean abs error: measured 0.001-0.017 on a 10000x365 config;
    # margin covers the smaller sample here.
    bs, tau = _bs(call_lsm, t, "call")
    mask = call_lsm.simulation[t] > K if moneyness == "itm" else call_lsm.simulation[t] < K
    err = np.abs(call_lsm.delta[t][mask] - bs.calculate_delta(tau)[mask])
    assert mask.sum() > 20
    assert err.mean() < 0.05


@pytest.mark.parametrize("t", [10, 25, 40, 47])
def test_call_gamma_close_to_black_scholes(call_lsm, t):
    # Measured mean abs error <= 0.0015 on a larger config; 0.005 leaves margin.
    bs, tau = _bs(call_lsm, t, "call")
    err = np.abs(call_lsm.gamma[t] - bs.calculate_gamma(tau))
    assert err.mean() < 0.005


def test_otm_call_delta_near_expiry_is_near_zero_not_extrapolated(call_lsm):
    # Regression: the old continuation-derivative delta gave ~0.9 (and values in
    # the tens) for OTM paths near expiry via polynomial extrapolation.
    t = N_STEPS - 3
    otm = call_lsm.simulation[t] < 0.95 * K
    assert otm.sum() > 20
    assert np.abs(call_lsm.delta[t][otm]).mean() < 0.1


def test_call_delta_stays_bounded_everywhere(call_lsm):
    # Old approach produced |delta| of 13-48; measured max here <= 1.3.
    assert np.abs(call_lsm.delta).max() < 2.0


def test_put_delta_is_negative_and_bounded_on_average(put_lsm):
    for t in [10, 25, 40]:
        d = put_lsm.delta[t]
        assert np.abs(d).max() < 2.0
        deep_itm = put_lsm.simulation[t] < 0.9 * K
        deep_otm = put_lsm.simulation[t] > 1.1 * K
        assert d[deep_itm].mean() < -0.5
        assert d[deep_otm].mean() > -0.3


def test_put_delta_close_to_tree_reference(put_lsm):
    # Central difference of the American-put tree value. Measured mean abs
    # error 0.004-0.03 on a larger config; 0.1 leaves margin for 4000 paths.
    value_fn = binomial_value_fn(100.0, K, R, SIGMA, T, N_STEPS, "put")
    for t in [10, 25]:
        S = put_lsm.simulation[t]
        near = (S > 85) & (S < 115)
        h = 0.5
        ref = (value_fn(t, S[near] + h) - value_fn(t, S[near] - h)) / (2 * h)
        assert np.abs(put_lsm.delta[t][near] - ref).mean() < 0.1


def test_sigma_none_still_gives_finite_deltas_and_same_price():
    with_sigma = _fitted_lsm("call", sigma=SIGMA)
    without = _fitted_lsm("call", sigma=None)
    assert np.all(np.isfinite(without.delta)) and np.all(np.isfinite(without.gamma))
    # sigma only affects the delta regression, never the exercise decision.
    assert without.calculate_option_price() == with_sigma.calculate_option_price()
    assert np.array_equal(without.exercise_time, with_sigma.exercise_time)


def test_sigma_feature_improves_call_delta_near_expiry():
    # Without the BS feature a cubic cannot follow delta's steepening near
    # expiry (BUGS.md); the feature must therefore reduce the error there.
    t = N_STEPS - 3
    errs = []
    for sigma in (SIGMA, None):
        lsm = _fitted_lsm("call", sigma=sigma)
        bs, tau = _bs(lsm, t, "call")
        errs.append(np.abs(lsm.delta[t] - bs.calculate_delta(tau)).mean())
    assert errs[0] < errs[1]


def test_price_matches_known_value_independent_of_delta_change():
    # Prices/exercise decisions were left unchanged by the delta rewrite: the
    # continuation regression is the same unregularized least squares. An
    # American call without dividends must price close to European BS.
    lsm = _fitted_lsm("call")
    ref = BlackScholes(SIGMA, 100.0, K, R, "call").calculate_option_price(T)
    assert lsm.calculate_option_price() == pytest.approx(ref, abs=0.5)


# --- European control variate (BUGS.md: plain LSM spuriously early-exercises
# an American call near expiry because regression noise exceeds the tiny gap
# between continuation and intrinsic value) ---

def _lsm(option_type, **kwargs):
    sim = GBM(sigma=SIGMA, start_price=100.0, risk_free_rate=R, seed=42).simulate_paths(N_PATHS, T, N_STEPS)
    lsm = LongstaffSchwartz(sim, K, R, T, option_type, "simple", 3, **kwargs)
    lsm.predict_payoff()
    return lsm


def test_control_variate_without_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        LongstaffSchwartz(np.ones((2, 2)), control_variate=True)


def test_control_variate_defaults_to_off():
    assert LongstaffSchwartz(np.ones((2, 2))).control_variate is False


def test_plain_lsm_call_early_exercises_on_this_config():
    # Guards the premise of the tests below: without the control variate this
    # seeded config does spuriously exercise a call early, so "no early
    # exercise with it on" is a real difference and not a vacuous pass.
    lsm = _lsm("call", sigma=SIGMA)
    assert (lsm.exercise_time < N_STEPS - 1).mean() > 0.05


def test_call_with_control_variate_never_exercises_early():
    # Regression: early exercise of a non-dividend call is never optimal.
    lsm = _lsm("call", sigma=SIGMA, control_variate=True)
    assert np.all(lsm.exercise_time == N_STEPS - 1)


def test_call_with_control_variate_prices_exactly_at_european_bs_price():
    # With no early exercise, payoff[0] equals the discounted European payoff
    # path by path, so the estimator collapses to the exact BS price at
    # tau = (n-1)*T/n (the grid's expiry convention). Exact up to float error.
    lsm = _lsm("call", sigma=SIGMA, control_variate=True)
    tau = (N_STEPS - 1) * T / N_STEPS
    bs_price = BlackScholes(SIGMA, 100.0, K, R, "call").calculate_option_price(tau)
    assert lsm.calculate_option_price() == pytest.approx(float(bs_price), abs=1e-6)


def test_call_with_control_variate_delta_stays_close_to_black_scholes():
    # Measured mean abs error of the same regression is <0.02 without the
    # control variate; 0.05 matches the existing call-delta tolerance.
    lsm = _lsm("call", sigma=SIGMA, control_variate=True)
    for t in [10, 25, 40]:
        bs, tau = _bs(lsm, t, "call")
        assert np.abs(lsm.delta[t] - bs.calculate_delta(tau)).mean() < 0.05


def test_control_variate_false_matches_omitting_the_argument():
    a = _lsm("put", sigma=SIGMA, control_variate=False)
    b = _lsm("put", sigma=SIGMA)
    np.testing.assert_array_equal(a.exercise_time, b.exercise_time)
    assert a.calculate_option_price() == b.calculate_option_price()
    np.testing.assert_array_equal(a.delta, b.delta)


def test_put_with_control_variate_runs_and_prices_near_tree_price():
    # Put still gets a sane value: between the European price and a band
    # around the binomial reference (true ~6.08). The band (+-0.4) covers the
    # measured 5.92-6.08 spread across seeds plus the smaller sample here.
    lsm = _lsm("put", sigma=SIGMA, control_variate=True)
    tau = (N_STEPS - 1) * T / N_STEPS
    european = float(BlackScholes(SIGMA, 100.0, K, R, "put").calculate_option_price(tau))
    price = lsm.calculate_option_price()
    assert price > european - 0.2
    assert abs(price - 6.08) < 0.4
    assert lsm.exercise_time.min() >= 0 and lsm.exercise_time.max() == N_STEPS - 1
    assert (lsm.exercise_time < N_STEPS - 1).mean() > 0.2  # puts do exercise early
