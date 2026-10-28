import numpy as np
import pytest

from optionpricing.pricing.european.black_scholes import BlackScholes
from optionpricing.pricing.american.binomial_tree import BinomialTree

def test_call_converges_to_black_scholes():
    # Early exercise is never optimal for a call on a non-dividend-paying stock,
    # so American == European value and the tree should converge to Black-Scholes.
    common = dict(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05)
    bs = BlackScholes(option_type="call", **common)
    bin_tree = BinomialTree(maturity=1.0, nb_steps=500, option_type="call", **common)
    bin_tree.predict_payoff()

    # Measured CRR(500) vs BS relative error: 3.8e-4; 1e-3 is ~2.6x that.
    assert bin_tree.calculate_option_price() == pytest.approx(bs.calculate_option_price(1.0), rel=1e-3)

def test_put_has_early_exercise_premium_over_black_scholes():
    common = dict(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05)
    bs = BlackScholes(option_type="put", **common)
    bin_tree = BinomialTree(maturity=1.0, nb_steps=500, option_type="put", **common)
    bin_tree.predict_payoff()

    # Measured: tree 6.0888 vs European 5.5735, a premium of ~0.51; require > 0.3 so a
    # tree that silently dropped the max(exercise, continuation) step (premium 0) fails.
    assert bin_tree.calculate_option_price() > bs.calculate_option_price(1.0) + 0.3

def test_calling_price_before_predict_payoff_raises():
    bin_tree = BinomialTree(option_type="call")
    with pytest.raises(AttributeError):
        bin_tree.calculate_option_price()

def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BinomialTree(option_type="invalid")

@pytest.mark.parametrize("kwargs", [dict(sigma=0.0), dict(maturity=0.0), dict(nb_steps=0)])
def test_invalid_numeric_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        BinomialTree(**kwargs)

def test_single_step_tree_is_accepted_and_matches_hand_computation():
    # nb_steps=1 is the boundary of the valid range. One-step CRR put: u=exp(sigma*sqrt(T)), d=1/u.
    sigma, r, S, K = 0.2, 0.05, 100.0, 100.0
    u = np.exp(sigma); d = 1 / u
    p = (np.exp(r) - d) / (u - d)
    expected = max(np.exp(-r) * (1 - p) * (K - S * d), K - S)
    tree = BinomialTree(sigma=sigma, start_price=S, strike=K, risk_free_rate=r, maturity=1.0, nb_steps=1,
                        option_type="put")
    tree.predict_payoff()
    assert tree.calculate_option_price() == pytest.approx(expected)
