import numpy as np
import pytest
import torch

from optionpricing.ml.surrogate.legacy.lsm_model import LegacyMLP, intrinsic_value, option_type_feature, smooth_intrinsic_value


def test_option_type_feature_encodes_call_as_one_and_put_as_zero():
    assert option_type_feature("call") == 1.0
    assert option_type_feature("put") == 0.0

@pytest.mark.parametrize("bad", ["both", "Call", "", None])
def test_option_type_feature_rejects_unknown_type(bad):
    with pytest.raises(ValueError):
        option_type_feature(bad)

def test_legacy_mlp_takes_five_inputs_and_returns_one_output_per_row():
    mlp = LegacyMLP().eval()
    assert mlp.layer1.weight.shape[1] == 5
    assert mlp(torch.zeros(3, 5)).shape == (3, 1)


# --- smooth_intrinsic_value ---

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_converges_to_exact_payoff_as_beta_frac_goes_to_zero(option_type):
    S = np.array([70.0, 99.0, 100.0, 101.0, 130.0])
    exact = intrinsic_value(S, 100.0, option_type)
    gaps = [np.abs(smooth_intrinsic_value(S, 100.0, option_type, beta_frac=b) - exact).max()
            for b in (0.06, 0.01, 1e-4)]
    # The max gap is at S == K and equals beta*log(2) = beta_frac*K*log(2).
    assert gaps[0] == pytest.approx(0.06 * 100.0 * np.log(2))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-2

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_is_above_exact_payoff_and_close_far_from_the_strike(option_type):
    S = np.linspace(20.0, 180.0, 161)
    smooth = smooth_intrinsic_value(S, 100.0, option_type)
    exact = intrinsic_value(S, 100.0, option_type)
    assert np.all(smooth >= exact)  # softplus(z) >= max(z, 0)
    far = np.abs(S - 100.0) > 60.0  # z >= 10 widths of beta=6: gap <= 6*exp(-10) ~ 3e-4
    assert np.abs(smooth - exact)[far].max() < 1e-3

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_has_no_overflow_or_nan_at_extreme_moneyness(option_type):
    # exp(z) with z = +-1e5/6 would overflow in a naive softplus; the stable form must not.
    S = np.array([1e-6, 1.0, 1e5, 1e8])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = smooth_intrinsic_value(S, 100.0, option_type)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    exact = intrinsic_value(S, 100.0, option_type)
    np.testing.assert_allclose(out, exact, atol=1e-6)

def test_smooth_intrinsic_at_the_money_is_beta_log_two_for_both_types():
    assert smooth_intrinsic_value(100.0, 100.0, "put") == pytest.approx(6.0 * np.log(2))
    assert smooth_intrinsic_value(100.0, 100.0, "call") == pytest.approx(6.0 * np.log(2))
