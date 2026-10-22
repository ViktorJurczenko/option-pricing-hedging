import numpy as np
import pytest

from EuropeanOptionPricing.BlackScholes import (BlackScholes, black_scholes_delta, black_scholes_gamma,
                                                black_scholes_price)

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


def test_known_textbook_values_at_the_money():
    # S=K=100, r=5%, sigma=20%, T=1: the standard reference call 10.4506 / put 5.5735.
    common = dict(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05)
    assert BlackScholes(option_type="call", **common).calculate_option_price(1.0) == pytest.approx(10.4506, abs=5e-5)
    assert BlackScholes(option_type="put", **common).calculate_option_price(1.0) == pytest.approx(5.5735, abs=5e-5)


def test_non_positive_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        BlackScholes(sigma=0.0)


@pytest.mark.parametrize("T", [0.0, -1.0])
def test_non_positive_maturity_raises(T):
    # One guard in _d1 covers price, delta and gamma alike.
    with pytest.raises(ValueError, match="T"):
        BlackScholes(**COMMON).calculate_option_price(T)


def test_any_non_positive_element_of_an_array_maturity_raises():
    bs = BlackScholes(start_price=S_GRID, **COMMON)
    with pytest.raises(ValueError, match="T"):
        bs.calculate_option_price(np.array([1.0, 1.0, 0.0, 1.0, 1.0]))


def test_functions_match_the_class_option_by_option_with_array_sigma():
    # The batch functions broadcast every argument; with array sigma each
    # element must equal the class built for that option alone.
    S = np.array([80.0, 100.0, 125.0])
    T = np.array([0.3, 1.0, 2.5])
    r = np.array([0.0, 0.03, 0.07])
    sigma = np.array([0.1, 0.25, 0.5])
    for option_type in ("call", "put"):
        price = black_scholes_price(S, 100.0, T, r, sigma, option_type)
        delta = black_scholes_delta(S, 100.0, T, r, sigma, option_type)
        gamma = black_scholes_gamma(S, 100.0, T, r, sigma)
        for k in range(3):
            bs = BlackScholes(sigma=sigma[k], start_price=S[k], strike=100.0, risk_free_rate=r[k],
                              option_type=option_type)
            assert price[k] == pytest.approx(bs.calculate_option_price(T[k]), rel=1e-14)
            assert delta[k] == pytest.approx(bs.calculate_delta(T[k]), rel=1e-14)
            assert gamma[k] == pytest.approx(bs.calculate_gamma(T[k]), rel=1e-14)


def test_delta_and_gamma_match_scipy_norm():
    from scipy.stats import norm

    bs = BlackScholes(sigma=0.3, start_price=np.linspace(60, 150, 7), strike=100.0, risk_free_rate=0.04)
    d1 = bs._d1(0.75)

    np.testing.assert_allclose(bs.calculate_delta(0.75), norm.cdf(d1), rtol=1e-14)
    np.testing.assert_allclose(bs.calculate_gamma(0.75), norm.pdf(d1) / (bs.start_price * 0.3 * np.sqrt(0.75)), rtol=1e-14)
