import pytest

from EuropeanOptionPricing.BlackScholes import BlackScholes
from AmericanOptionPricing.BinomialTree import BinomialTree

def test_call_converges_to_black_scholes():
    # Early exercise is never optimal for a call on a non-dividend-paying stock,
    # so American == European value and the tree should converge to Black-Scholes.
    common = dict(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05)
    bs = BlackScholes(option_type="call", **common)
    bin_tree = BinomialTree(maturity=1.0, nb_steps=500, option_type="call", **common)
    bin_tree.predict_payoff()

    assert bin_tree.calculate_option_price() == pytest.approx(bs.calculate_option_price(1.0), rel=1e-2)

def test_put_has_early_exercise_premium_over_black_scholes():
    common = dict(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05)
    bs = BlackScholes(option_type="put", **common)
    bin_tree = BinomialTree(maturity=1.0, nb_steps=500, option_type="put", **common)
    bin_tree.predict_payoff()

    assert bin_tree.calculate_option_price() >= bs.calculate_option_price(1.0) - 1e-9

def test_calling_price_before_predict_payoff_raises():
    bin_tree = BinomialTree(option_type="call")
    with pytest.raises(AttributeError):
        bin_tree.calculate_option_price()

def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BinomialTree(option_type="invalid")
