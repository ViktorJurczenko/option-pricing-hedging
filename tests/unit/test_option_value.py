"""
Tests for AmericanHedging/option_value.py: the value functions used by
HedgingEnv's "accounting" reward mode (see tests/unit/test_hedge_env.py) to
know the option's value at every simulation row, not just at exercise.

Rather than pinning single hardcoded prices, these mostly check the
functions against the principled relationships they're built from:
black_scholes_value_fn against a directly-constructed BlackScholes instance
at the same reduced tau, binomial_value_fn against a directly-built
BinomialTree at the equivalent scaled maturity/step count, and the
call-never-exercised-early / American->=European invariants.
"""
import numpy as np
import pytest

from AmericanHedging.option_value import (
    black_scholes_value_fn,
    binomial_value_fn,
    make_option_value_fn,
)
from AmericanOptionPricing.BinomialTree import BinomialTree
from EuropeanOptionPricing.BlackScholes import BlackScholes

STRIKE = 100.0
R = 0.05
SIGMA = 0.2
MATURITY = 1.0
N_STEPS = 20  # small grid so binomial_value_fn's tree build stays fast


# --- black_scholes_value_fn --------------------------------------------------

def test_black_scholes_value_fn_raises_on_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="straddle")


def test_black_scholes_value_fn_raises_type_error_when_option_type_omitted():
    """Regression guard: option_type used to default to "call", so omitting
    it silently priced a call. It is now a required positional/keyword
    argument -- calling without it must raise TypeError, not fall back."""
    with pytest.raises(TypeError):
        black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS)


def test_black_scholes_value_fn_matches_direct_bs_at_remaining_time():
    """value_fn(t, S) must equal a fresh BlackScholes priced at the row's
    remaining time tau = (n_steps-1-t)*dt with dt = maturity/(n_steps-1)."""
    value_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call")
    t = 5
    S = np.array([80.0, 100.0, 120.0])
    tau = (N_STEPS - 1 - t) * MATURITY / (N_STEPS - 1)
    expected = BlackScholes(
        sigma=SIGMA, start_price=S, strike=STRIKE, risk_free_rate=R, option_type="call"
    ).calculate_option_price(T=tau)
    np.testing.assert_allclose(value_fn(t, S), expected)


def test_black_scholes_value_fn_last_row_is_exactly_intrinsic_for_call_and_put():
    S = np.array([80.0, 100.0, 120.0])
    call_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call")
    put_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")

    np.testing.assert_allclose(call_fn(N_STEPS - 1, S), np.maximum(S - STRIKE, 0.0))
    np.testing.assert_allclose(put_fn(N_STEPS - 1, S), np.maximum(STRIKE - S, 0.0))


def test_black_scholes_value_fn_accepts_vectorized_spot_input():
    value_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call")
    S = np.linspace(50, 150, 11)

    result = value_fn(0, S)

    assert result.shape == S.shape
    assert np.all(np.isfinite(result))
    # A call's value is strictly increasing in spot -- catches e.g. an
    # accidental sign flip in d1/d2 for the vectorized path.
    assert np.all(np.diff(result) > 0)


def test_black_scholes_value_fn_t_zero_uses_full_maturity():
    """Regression: row 0 used to have remaining time (n-1)/n * maturity
    (the grid ended short of maturity). It is now the full maturity."""
    value_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")
    expected = BlackScholes(
        sigma=SIGMA, start_price=STRIKE, strike=STRIKE, risk_free_rate=R, option_type="put"
    ).calculate_option_price(T=MATURITY)
    assert value_fn(0, np.array([STRIKE]))[0] == pytest.approx(float(expected))


# --- binomial_value_fn -------------------------------------------------------

def test_binomial_value_fn_raises_on_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="straddle")


def test_binomial_value_fn_raises_type_error_when_option_type_omitted():
    """Regression guard: option_type used to default to "put", so omitting
    it silently priced a put. It is now a required positional/keyword
    argument -- calling without it must raise TypeError, not fall back."""
    with pytest.raises(TypeError):
        binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS)


def test_binomial_value_fn_raises_on_substeps_below_one():
    with pytest.raises(ValueError, match="substeps"):
        binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put", substeps=0)


def test_binomial_value_fn_last_row_is_exactly_intrinsic():
    value_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")
    S = np.array([80.0, 100.0, 120.0])

    result = value_fn(N_STEPS - 1, S)

    np.testing.assert_allclose(result, np.maximum(STRIKE - S, 0.0))


def test_binomial_value_fn_at_t_zero_matches_binomial_tree_at_full_maturity():
    """value_fn(0, start_price) is the price of the same CRR tree built
    directly through BinomialTree with the full maturity and
    nb_steps=(n_steps-1)*substeps."""
    substeps = 4
    tree = BinomialTree(
        sigma=SIGMA, start_price=STRIKE, strike=STRIKE, risk_free_rate=R,
        maturity=MATURITY, nb_steps=(N_STEPS - 1) * substeps, option_type="put",
    )
    tree.predict_payoff()
    expected = tree.calculate_option_price()

    value_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put", substeps=substeps)
    result = value_fn(0, np.array([STRIKE]))[0]

    assert result == pytest.approx(expected, rel=1e-9)


def test_binomial_value_fn_put_at_t_zero_is_at_least_the_european_value():
    """American put value >= European value: the early-exercise premium is
    never negative."""
    value_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")
    american_value = value_fn(0, np.array([STRIKE]))[0]

    tau0 = (N_STEPS - 1) * MATURITY / N_STEPS
    european_value = BlackScholes(
        sigma=SIGMA, start_price=STRIKE, strike=STRIKE, risk_free_rate=R, option_type="put"
    ).calculate_option_price(T=tau0)

    assert american_value >= european_value - 1e-9


def test_binomial_value_fn_call_tracks_black_scholes_closely():
    """For a call on a non-dividend stock, early exercise is never optimal,
    so the American tree value should stay close to the closed-form European
    (Black-Scholes) value at every simulation row -- tolerance is loose
    (atol=0.5, on a ~100-strike underlying) because binomial_value_fn is a
    discrete-tree approximation, not the closed form itself. Row 0 is
    excluded here: the tree has only its single root node at that row (see
    node_log_moneyness(0)), so value_fn(0, S) is constant in S by
    construction and is covered separately by the t=0/start_price check in
    test_binomial_value_fn_at_t_zero_matches_binomial_tree_with_scaled_maturity_and_steps."""
    bs_fn = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call")
    tree_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call", substeps=4)
    S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    for t in [5, 10, 15]:
        np.testing.assert_allclose(tree_fn(t, S), bs_fn(t, S), atol=0.5, rtol=0.05)


def test_binomial_value_fn_floors_interpolated_value_at_intrinsic_outside_node_range():
    """A spot far outside a row's node range gets clamped by np.interp to
    the nearest edge node's (low) value; the function must still floor the
    result at the immediate-exercise intrinsic value, which for a deep-ITM
    put/near-zero spot is far larger than that clamped continuation value."""
    value_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put", substeps=2)
    S = np.array([0.01])

    result = value_fn(0, S)[0]

    intrinsic = STRIKE - 0.01
    assert result == pytest.approx(intrinsic)


def test_binomial_value_fn_accepts_vectorized_spot_input():
    value_fn = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")
    S = np.linspace(50, 150, 11)

    result = value_fn(3, S)

    assert result.shape == S.shape
    assert np.all(np.isfinite(result))


# --- make_option_value_fn ----------------------------------------------------

def test_make_option_value_fn_raises_on_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        make_option_value_fn("straddle", STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS)


def test_make_option_value_fn_call_delegates_to_black_scholes_value_fn():
    combined = make_option_value_fn("call", STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS)
    direct = black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="call")
    S = np.array([90.0, 100.0, 110.0])

    np.testing.assert_allclose(combined(2, S), direct(2, S))


def test_make_option_value_fn_put_delegates_to_binomial_value_fn():
    combined = make_option_value_fn("put", STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS)
    direct = binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, N_STEPS, option_type="put")
    S = np.array([90.0, 100.0, 110.0])

    np.testing.assert_allclose(combined(2, S), direct(2, S))


# --- n_steps (simulation row count) must leave at least one interval ---------

@pytest.mark.parametrize("n_steps", [0, 1])
def test_black_scholes_value_fn_raises_when_fewer_than_two_rows(n_steps):
    # Regression: dt = maturity / (n_steps - 1) divided by zero for one row.
    with pytest.raises(ValueError, match="n_steps"):
        black_scholes_value_fn(STRIKE, R, SIGMA, MATURITY, n_steps, option_type="call")


@pytest.mark.parametrize("n_steps", [0, 1])
def test_binomial_value_fn_raises_when_fewer_than_two_rows(n_steps):
    with pytest.raises(ValueError, match="n_steps"):
        binomial_value_fn(STRIKE, STRIKE, R, SIGMA, MATURITY, n_steps, option_type="put")
