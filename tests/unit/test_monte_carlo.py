import numpy as np
import pytest

from optionpricing.pricing.european.monte_carlo import MonteCarlo

@pytest.fixture
def predictions():
    # 2 time steps, 4 paths; only the final row (maturity prices) matters
    return np.array([
        [100.0, 100.0, 100.0, 100.0],
        [110.0, 90.0, 105.0, 95.0],
    ])

def test_call_payoff_average(predictions):
    mc = MonteCarlo(predictions, strike=100.0, risk_free_rate=0.0, option_type="call")
    # payoffs: max(110-100,0)=10, max(90-100,0)=0, max(105-100,0)=5, max(95-100,0)=0 -> mean = 3.75
    assert mc.calculate_option_price(T=1) == pytest.approx(3.75)

def test_put_payoff_average(predictions):
    mc = MonteCarlo(predictions, strike=100.0, risk_free_rate=0.0, option_type="put")
    # payoffs: max(100-110,0)=0, max(100-90,0)=10, max(100-105,0)=0, max(100-95,0)=5 -> mean = 3.75
    assert mc.calculate_option_price(T=1) == pytest.approx(3.75)

def test_discounting_is_applied(predictions):
    mc = MonteCarlo(predictions, strike=100.0, risk_free_rate=0.05, option_type="call")
    undiscounted_mean = 3.75
    assert mc.calculate_option_price(T=2) == pytest.approx(undiscounted_mean * np.exp(-0.05 * 2))

def test_invalid_option_type_raises(predictions):
    with pytest.raises(ValueError):
        MonteCarlo(predictions, option_type="invalid")

@pytest.mark.parametrize("T", [0, 0.0, -1])
def test_non_positive_maturity_raises(predictions, T):
    mc = MonteCarlo(predictions, strike=100.0, risk_free_rate=0.05, option_type="call")
    with pytest.raises(ValueError, match="T"):
        mc.calculate_option_price(T=T)
