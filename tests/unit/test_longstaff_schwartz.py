import numpy as np
import pytest

from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz

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
