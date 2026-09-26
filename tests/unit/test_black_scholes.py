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
