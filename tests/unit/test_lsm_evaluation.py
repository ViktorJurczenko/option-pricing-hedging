import numpy as np
import pytest
import torch

from optionpricing.core.gbm import GBM
from optionpricing.ml.surrogate.legacy.lsm_evaluation import test_option_price_lsm as price_vs_lsm
from optionpricing.ml.surrogate.legacy.lsm_model import LegacyMLP, smooth_intrinsic_value
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz
from optionpricing.pricing.european.black_scholes import BlackScholes

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


def test_call_with_control_variate_benchmark_matches_black_scholes_price(simulation):
    # With the control variate the LSM call equals the BS price exactly, so a model
    # whose time value reproduces BS - smooth_intrinsic has ~0 error. This proves
    # sigma/control_variate reach the LSM benchmark and that the smooth intrinsic
    # value is added back for calls.
    bs = BlackScholes(SIGMA, 100.0, K, R, "call").calculate_option_price(T)
    tv = (bs - smooth_intrinsic_value(100.0, K, "call")) / K
    model = RecordingModel(output=tv)

    error, abs_error = price_vs_lsm(
        model, simulation, 1.0, K, R, 0.1, T, 0.2, 0.3, option_type="call", sigma=SIGMA, control_variate=True
    )

    assert abs_error == pytest.approx(0.0, abs=1e-3)  # identity holds to ~1e-6; float32 model output adds ~1e-5
    assert error < 0.1


def test_control_variate_without_sigma_is_rejected_by_benchmark(simulation):
    # LongstaffSchwartz requires sigma for the control variate, so the flag must be forwarded.
    with pytest.raises(ValueError):
        price_vs_lsm(
            RecordingModel(), simulation, 1.0, K, R, 0.1, T, 0.2, 0.3, option_type="call", control_variate=True
        )


def _constant_output_mlp(output):
    """A real MLP whose output is `output` for every input (last layer zeroed, bias set)."""
    mlp = LegacyMLP()
    with torch.no_grad():
        mlp.layer4.weight.zero_()
        mlp.layer4.bias.fill_(output)
    return mlp


def test_real_mlp_error_matches_hand_computation(simulation):
    # price_ml = smooth_intrinsic(S0) + output*K, compared with an independently fitted LSM.
    output = 0.01
    lsm = LongstaffSchwartz(simulation, K, R, T, "put", "simple", 3)
    lsm.predict_payoff()
    expected_abs = abs(float(smooth_intrinsic_value(100.0, K, "put")) + output * K - lsm.calculate_option_price())

    error, abs_error = price_vs_lsm(_constant_output_mlp(output), simulation, 1.0, K, R, 0.1, T, 0.2, 0.3)

    assert abs_error == pytest.approx(expected_abs, abs=1e-5)  # float32 output
    assert error == pytest.approx(expected_abs / lsm.calculate_option_price() * 100, rel=1e-4)


def test_model_is_put_in_eval_mode_before_inference(simulation):
    # Dropout/BatchNorm would make single-row inference fail or be random in train mode.
    mlp = _constant_output_mlp(0.0).train()
    price_vs_lsm(mlp, simulation, 1.0, K, R, 0.1, T, 0.2, 0.3)
    assert not mlp.training


def test_error_is_percentage_of_lsm_price_when_price_is_large():
    # Constant deep-ITM put: LSM price is exactly the 99 intrinsic; model output 0 gives the
    # smooth intrinsic, so the error is |smooth - 99| / 99 * 100.
    sim = np.full((5, 4), 1.0)
    error, abs_error = price_vs_lsm(RecordingModel(0.0), sim, 0.01, K, R, 0.1, T, 0.2, 0.3)
    smooth = float(smooth_intrinsic_value(1.0, K, "put"))
    assert abs_error == pytest.approx(abs(smooth - 99.0))
    assert error == pytest.approx(abs(smooth - 99.0) / 99.0 * 100)


def test_error_falls_back_to_absolute_error_when_lsm_price_is_below_ten_cents():
    # Constant OTM put: LSM price is 0, so a percentage would divide by ~0. The
    # fallback reports the absolute error in price units instead.
    sim = np.full((5, 4), 200.0)
    error, abs_error = price_vs_lsm(RecordingModel(0.5), sim, 2.0, K, R, 0.1, T, 0.2, 0.3)
    assert abs_error == pytest.approx(0.5 * K + float(smooth_intrinsic_value(200.0, K, "put")), rel=1e-5)
    assert error == abs_error


# --- regression_tests ---


def test_regression_tests_prints_mse_mae_r2_for_a_known_case(capsys):
    from torch.utils.data import DataLoader, TensorDataset

    from optionpricing.ml.surrogate.legacy.lsm_evaluation import regression_tests

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    # pred [1,2,3] vs true [1,2,5]: MSE 4/3, MAE 2/3, R2 = 1 - 4 / (78/9). Batch size 2
    # forces the concatenation of two batches (one of them ragged).
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([1.0, 2.0, 5.0])
    model = Identity().train()
    regression_tests(model, DataLoader(TensorDataset(x, y), batch_size=2))

    out = capsys.readouterr().out
    assert f"MSE:  {4 / 3:.6f}" in out
    assert f"MAE:  {2 / 3:.6f}" in out
    assert f"R²:   {1 - 4 / (78 / 9):.4f}" in out
    assert not model.training
