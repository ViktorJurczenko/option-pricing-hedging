import pytest

from core.gbm import GBM
from EuropeanOptionPricing.MonteCarlo import MonteCarlo
from EuropeanOptionPricing.BlackScholes import BlackScholes
from AmericanOptionPricing.BinomialTree import BinomialTree
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz

def test_monte_carlo_converges_to_black_scholes():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=123)
    simulation = gbm.simulate_paths(nb_predictions=100_000, maturity=1, sampling=252)

    mc = MonteCarlo(simulation, strike=100.0, risk_free_rate=0.05, option_type="call")
    bs = BlackScholes(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05, option_type="call")

    assert mc.calculate_option_price(1) == pytest.approx(bs.calculate_option_price(1), rel=0.03)

def test_longstaff_schwartz_converges_to_binomial_tree_for_american_put():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7)
    simulation = gbm.simulate_paths(nb_predictions=50_000, maturity=1, sampling=50)

    lsm = LongstaffSchwartz(
        simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0,
        option_type="put", reg_type="laguerre", max_degree=3,
    )
    lsm.predict_payoff()

    bin_tree = BinomialTree(
        sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05,
        maturity=1.0, nb_steps=500, option_type="put",
    )
    bin_tree.predict_payoff()

    assert lsm.calculate_option_price() == pytest.approx(bin_tree.calculate_option_price(), rel=0.05)
