import numpy as np
import pytest

from EuropeanOptionPricing.BlackScholes import BlackScholes

def test_put_call_parity():
    common = dict(sigma=0.25, start_price=105.0, strike=100.0, risk_free_rate=0.03)
    call = BlackScholes(option_type="call", **common)
    put = BlackScholes(option_type="put", **common)

    lhs = call.calculate_option_price(1.0) - put.calculate_option_price(1.0)
    rhs = common["start_price"] - common["strike"] * np.exp(-common["risk_free_rate"] * 1.0)

    assert lhs == pytest.approx(rhs, abs=1e-8)

def test_deep_itm_call_converges_to_discounted_forward_payoff():
    # As sigma -> 0 the stock is effectively riskless, so the option's value
    # collapses to the discounted deterministic payoff: max(S - K*exp(-rT), 0).
    bs = BlackScholes(sigma=1e-4, start_price=100.0, strike=90.0, risk_free_rate=0.05, option_type="call")
    expected = 100.0 - 90.0 * np.exp(-0.05 * 1.0)

    assert bs.calculate_option_price(1.0) == pytest.approx(expected, abs=1e-2)

def test_deep_otm_call_converges_to_zero():
    bs = BlackScholes(sigma=1e-4, start_price=80.0, strike=100.0, risk_free_rate=0.05, option_type="call")

    assert bs.calculate_option_price(1.0) == pytest.approx(0.0, abs=1e-2)

def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BlackScholes(option_type="invalid")


COMMON = dict(sigma=0.25, strike=100.0, risk_free_rate=0.03)
S_GRID = np.array([70.0, 90.0, 100.0, 110.0, 140.0])


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference_of_price(option_type):
    h = 1e-4
    up = BlackScholes(start_price=S_GRID + h, option_type=option_type, **COMMON).calculate_option_price(0.7)
    dn = BlackScholes(start_price=S_GRID - h, option_type=option_type, **COMMON).calculate_option_price(0.7)
    delta = BlackScholes(start_price=S_GRID, option_type=option_type, **COMMON).calculate_delta(0.7)
    np.testing.assert_allclose(delta, (up - dn) / (2 * h), atol=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_second_finite_difference_of_price(option_type):
    h = 1e-2
    price = lambda s: BlackScholes(start_price=s, option_type=option_type, **COMMON).calculate_option_price(0.7)
    fd = (price(S_GRID + h) - 2 * price(S_GRID) + price(S_GRID - h)) / h**2
    gamma = BlackScholes(start_price=S_GRID, option_type=option_type, **COMMON).calculate_gamma(0.7)
    np.testing.assert_allclose(gamma, fd, atol=1e-5)


def test_put_delta_equals_call_delta_minus_one_and_gamma_is_shared():
    call = BlackScholes(start_price=S_GRID, option_type="call", **COMMON)
    put = BlackScholes(start_price=S_GRID, option_type="put", **COMMON)
    np.testing.assert_allclose(put.calculate_delta(0.5), call.calculate_delta(0.5) - 1)
    np.testing.assert_allclose(put.calculate_gamma(0.5), call.calculate_gamma(0.5))


def test_delta_ranges_and_gamma_positive():
    call = BlackScholes(start_price=S_GRID, option_type="call", **COMMON)
    put = BlackScholes(start_price=S_GRID, option_type="put", **COMMON)
    assert np.all((call.calculate_delta() > 0) & (call.calculate_delta() < 1))
    assert np.all((put.calculate_delta() > -1) & (put.calculate_delta() < 0))
    assert np.all(call.calculate_gamma() > 0)


def test_delta_and_gamma_are_vectorized_over_start_price():
    bs = BlackScholes(start_price=S_GRID, **COMMON)
    assert bs.calculate_delta(1.0).shape == S_GRID.shape
    assert bs.calculate_gamma(1.0).shape == S_GRID.shape
