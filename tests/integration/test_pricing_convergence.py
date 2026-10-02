import numpy as np
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

    # MC discounts by the full T and the last row is now exactly at T, so the
    # estimator is unbiased for BS(T=1); std error ~0.045 on 10.45 (<0.5%) at
    # 100k paths, so 3% is conservative.
    assert mc.calculate_option_price(1) == pytest.approx(bs.calculate_option_price(1), rel=0.03)

def test_longstaff_schwartz_converges_to_binomial_tree_for_american_put():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7)
    simulation = gbm.simulate_paths(nb_predictions=50_000, maturity=1, sampling=50)

    lsm = LongstaffSchwartz(
        simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0,
        option_type="put", reg_type="simple", max_degree=3,
    )
    lsm.predict_payoff()

    bin_tree = BinomialTree(
        sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05,
        maturity=1.0, nb_steps=500, option_type="put",
    )
    bin_tree.predict_payoff()

    assert lsm.calculate_option_price() == pytest.approx(bin_tree.calculate_option_price(), rel=0.05)

def test_simple_and_laguerre_bases_agree_at_degree_three():
    # Both span the same cubic polynomials, so the least-squares fit (hence price
    # and exercise rule) is identical up to conditioning; measured ~1e-9.
    simulation = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7).simulate_paths(
        nb_predictions=3000, maturity=1, sampling=20)
    models = []
    for reg_type in ("simple", "laguerre"):
        lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0,
                                option_type="put", reg_type=reg_type, max_degree=3)
        lsm.predict_payoff()
        models.append(lsm)

    assert models[0].calculate_option_price() == pytest.approx(models[1].calculate_option_price(), abs=1e-6)
    assert np.array_equal(models[0].exercise_time, models[1].exercise_time)
