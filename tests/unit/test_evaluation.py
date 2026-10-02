import numpy as np
import pytest
import torch

from AmericanOptionPricing.evaluation import test_option_price_lsm as price_vs_lsm
from AmericanOptionPricing.ML import MLP, smooth_intrinsic_value
from core.gbm import GBM
from EuropeanOptionPricing.BlackScholes import BlackScholes

K, R, T, SIGMA = 100.0, 0.05, 1.0, 0.2


class RecordingModel(torch.nn.Module):
    """Returns a fixed time-value output and records the feature rows it gets."""
    def __init__(self, output=0.0):
        super().__init__()
        self.output = output
        self.seen = []

    def forward(self, x):
        self.seen.append(x.clone())
        return torch.full((x.shape[0], 1), self.output)


@pytest.fixture(scope="module")
def simulation():
    return GBM(SIGMA, 100.0, R, seed=3).simulate_paths(2000, T, 20)


@pytest.mark.parametrize("option_type, flag", [("call", 1.0), ("put", 0.0)])
def test_model_gets_market_features_then_is_call(simulation, option_type, flag):
    model = RecordingModel()
    price_vs_lsm(model, simulation, 1.0, K, R, 0.1, T, 0.2, 0.3, option_type=option_type)
    assert model.seen[0][0].tolist() == pytest.approx([1.0, 0.2, 0.1, 0.3, flag])

def test_real_mlp_runs_through_the_benchmark(simulation):
    error, abs_error = price_vs_lsm(MLP(), simulation, 1.0, K, R, 0.1, T, 0.2, 0.3)
    assert np.isfinite(error) and abs_error >= 0

def test_call_with_control_variate_benchmark_matches_black_scholes_price(simulation):
    # With the control variate the LSM call equals the BS price exactly, so a model
    # whose time value reproduces BS - smooth_intrinsic has ~0 error. This proves
    # sigma/control_variate reach the LSM benchmark and that the smooth intrinsic
    # value is added back for calls.
    bs = BlackScholes(SIGMA, 100.0, K, R, "call").calculate_option_price(T)
    tv = (bs - smooth_intrinsic_value(100.0, K, "call")) / K
    model = RecordingModel(output=tv)

    error, abs_error = price_vs_lsm(model, simulation, 1.0, K, R, 0.1, T, 0.2, 0.3,
                                    option_type="call", sigma=SIGMA, control_variate=True)

    assert abs_error == pytest.approx(0.0, abs=1e-3)  # identity holds to ~1e-6; float32 model output adds ~1e-5
    assert error < 0.1

def test_control_variate_without_sigma_is_rejected_by_benchmark(simulation):
    # LongstaffSchwartz requires sigma for the control variate, so the flag must be forwarded.
    with pytest.raises(ValueError):
        price_vs_lsm(RecordingModel(), simulation, 1.0, K, R, 0.1, T, 0.2, 0.3,
                     option_type="call", control_variate=True)
