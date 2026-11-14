import numpy as np
import pytest

from optionpricing.core.gbm import GBM
from optionpricing.pricing.american.batch_binomial_tree import exercise_boundary, price_put_batch
from optionpricing.pricing.american.binomial_tree import BinomialTree
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz
from optionpricing.pricing.european.black_scholes import BlackScholes
from optionpricing.pricing.european.monte_carlo import MonteCarlo


def test_monte_carlo_converges_to_black_scholes():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=123)
    simulation = gbm.simulate_paths(nb_predictions=100_000, maturity=1, sampling=252)

    mc = MonteCarlo(simulation, strike=100.0, risk_free_rate=0.05, option_type="call")
    bs = BlackScholes(sigma=0.2, start_price=100.0, strike=100.0, risk_free_rate=0.05, option_type="call")

    # MC discounts by the full T and the last row is now exactly at T, so the
    # estimator is unbiased for BS(T=1); std error ~0.45% at 100k paths. Measured relative
    # error: 0.22% for this seed, 0.02-0.60% over seeds 1-4; 1% is ~1.7x the worst.
    assert mc.calculate_option_price(1) == pytest.approx(bs.calculate_option_price(1), rel=0.01)


def test_longstaff_schwartz_converges_to_binomial_tree_for_american_put():
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7)
    simulation = gbm.simulate_paths(nb_predictions=50_000, maturity=1, sampling=50)

    lsm = LongstaffSchwartz(
        simulation,
        strike=100.0,
        risk_free_rate=0.05,
        maturity=1.0,
        option_type="put",
        reg_type="simple",
        max_degree=3,
    )
    lsm.predict_payoff()

    bin_tree = BinomialTree(
        sigma=0.2,
        start_price=100.0,
        strike=100.0,
        risk_free_rate=0.05,
        maturity=1.0,
        nb_steps=500,
        option_type="put",
    )
    bin_tree.predict_payoff()

    # Measured relative error vs the tree: 0.19% for this seed, 0.2-1.3% over seeds 1-4
    # (50k paths, 50 steps); 3% is ~2.3x the worst.
    assert lsm.calculate_option_price() == pytest.approx(bin_tree.calculate_option_price(), rel=0.03)


def test_simple_and_laguerre_bases_agree_at_degree_three():
    # Both span the same cubic polynomials, so the least-squares fit (hence price
    # and exercise rule) is identical up to conditioning; measured ~1e-9.
    simulation = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7).simulate_paths(
        nb_predictions=3000, maturity=1, sampling=20
    )
    models = []
    for reg_type in ("simple", "laguerre"):
        lsm = LongstaffSchwartz(
            simulation,
            strike=100.0,
            risk_free_rate=0.05,
            maturity=1.0,
            option_type="put",
            reg_type=reg_type,
            max_degree=3,
        )
        lsm.predict_payoff()
        models.append(lsm)

    assert models[0].calculate_option_price() == pytest.approx(models[1].calculate_option_price(), abs=1e-6)
    assert np.array_equal(models[0].exercise_time, models[1].exercise_time)


def test_put_american_price_exceeds_european_by_the_early_exercise_premium():
    # Measured premium: CRR(500) 6.0888 vs BS 5.5735 = 0.515. Both LSM and tree must show it.
    # The 0.3 floor fails a pricer that never exercises early (premium 0).
    gbm = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=7)
    simulation = gbm.simulate_paths(nb_predictions=50_000, maturity=1, sampling=50)
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0, option_type="put")
    lsm.predict_payoff()
    european = BlackScholes(0.2, 100.0, 100.0, 0.05, "put").calculate_option_price(1.0)

    assert lsm.calculate_option_price() > european + 0.3


def test_exercise_boundary_converges_to_the_fine_tree_threshold():
    # Reference: the moneyness where an 8000-step BBSR tree switches from
    # exercising today to holding, by bisection on its `exercised` flag, at
    # five times left on one contract (T = 1, r = 5%, sigma = 20%), all
    # bisected together. The boundary is read along one tree's curve, as the
    # hedger reads it, so late levels have fewer steps left than a tree for
    # that maturity would. Measured max gaps: 2.1e-3 of K at 1000 steps (with
    # 3 months and 1 month left), 8.4e-4 at 4000. The nearest node alone was
    # off by up to 7.7e-3 at 1000 steps. The reference is itself still ~3e-4
    # above the 16000-step value (0.80962 against 0.80937 with 1 year left),
    # since smooth pasting makes any boundary location imprecise.
    left = np.array([1.0, 0.5, 0.25, 0.1, 1 / 52])
    lo, hi = np.full(len(left), 0.3), np.ones(len(left))
    for _ in range(30):  # 0.7 / 2^30 ~ 7e-10
        mid = (lo + hi) / 2
        exercised = price_put_batch(mid, left, 0.05, 0.2, nb_steps=8000).exercised
        lo, hi = np.where(exercised, mid, lo), np.where(exercised, hi, mid)
    reference = (lo + hi) / 2

    for nb_steps, tol in ((1000, 3e-3), (4000, 1.2e-3)):
        boundary = exercise_boundary(1.0, 0.05, 0.2, nb_steps=nb_steps)
        at_left = boundary[np.rint(nb_steps * (1 - left)).astype(int)]
        np.testing.assert_allclose(at_left, reference, rtol=0, atol=tol, err_msg=f"{nb_steps} steps")
