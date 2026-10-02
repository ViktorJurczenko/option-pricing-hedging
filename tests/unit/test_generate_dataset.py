import numpy as np
import pytest

from AmericanOptionPricing.ML import smooth_intrinsic_value
from core.gbm import GBM
from datasets.generate_dataset import gen_data, process_sample
from EuropeanOptionPricing.BlackScholes import BlackScholes

NORM = dict(log_sigma_mean=np.log(0.2), log_sigma_std=0.5, r_mean=0.03, r_std=0.02,
            log_T_mean=0.0, log_T_std=0.5)


def _sample(option_types=("call", "put"), S0=100.0, K=100.0, sigma=0.2, r=0.05, T=1.0, **kw):
    return process_sample(S0, K, sigma, NORM["log_sigma_mean"], NORM["log_sigma_std"], r,
                          NORM["r_mean"], NORM["r_std"], T, NORM["log_T_mean"], NORM["log_T_std"],
                          500, 20, option_types, **kw)


@pytest.mark.parametrize("S0, K", [(100.0, 100.0), (120.0, 100.0), (80.0, 100.0)])
def test_call_label_reconstructs_black_scholes_price(S0, K):
    # Calls use the control variate, so label*K + smooth intrinsic is exactly BS
    # (an American call without dividends is never exercised early).
    sim = GBM(0.2, S0, 0.05, seed=1).simulate_paths(500, 1.0, 20)
    label = gen_data(sim, S0, K, 0.05, 1.0, 0.2, "call", 3, "simple")
    price = label * K + smooth_intrinsic_value(S0, K, "call")
    assert price == pytest.approx(BlackScholes(0.2, S0, K, 0.05, "call").calculate_option_price(1.0), abs=1e-4)

def test_put_label_is_not_the_black_scholes_european_price():
    # Puts use plain LSM (early exercise has value), so the label must differ
    # from the European price; a deep ITM put makes the gap large vs MC noise.
    S0, K = 70.0, 100.0
    sim = GBM(0.2, S0, 0.05, seed=1).simulate_paths(5000, 1.0, 20)
    label = gen_data(sim, S0, K, 0.05, 1.0, 0.2, "put", 3, "simple")
    price = label * K + smooth_intrinsic_value(S0, K, "put")
    eu = BlackScholes(0.2, S0, K, 0.05, "put").calculate_option_price(1.0)
    assert price > eu + 0.3  # early-exercise premium here is ~0.7+

def test_process_sample_returns_call_then_put_rows_with_is_call_feature():
    rows = _sample()
    assert len(rows) == 2
    (f_call, _), (f_put, _) = rows
    assert len(f_call) == len(f_put) == 5
    assert f_call[4] == 1.0 and f_put[4] == 0.0
    assert f_call[:4] == f_put[:4]  # same market inputs, one simulation
    assert f_call[0] == pytest.approx(1.0)
    assert f_call[2] == pytest.approx((0.05 - 0.03) / 0.02)
    assert f_call[3] == pytest.approx(0.0)  # sigma == exp(log_sigma_mean)

def test_process_sample_single_type_gives_one_row():
    rows = _sample(option_types=("put",))
    assert len(rows) == 1 and rows[0][0][4] == 0.0

def test_process_sample_returns_empty_list_on_failure(capsys):
    # Regression: failures used to return (None, None); main() flattens lists, so [] is required.
    assert _sample(sigma=-1.0) == []
    assert "Skipping sample" in capsys.readouterr().out
