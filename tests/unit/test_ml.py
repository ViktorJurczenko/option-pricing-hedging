import pytest
import torch

from AmericanOptionPricing.ML import MLP, option_type_feature


def test_option_type_feature_encodes_call_as_one_and_put_as_zero():
    assert option_type_feature("call") == 1.0
    assert option_type_feature("put") == 0.0

@pytest.mark.parametrize("bad", ["both", "Call", "", None])
def test_option_type_feature_rejects_unknown_type(bad):
    with pytest.raises(ValueError):
        option_type_feature(bad)

def test_mlp_takes_five_inputs_and_returns_one_output_per_row():
    mlp = MLP().eval()
    assert mlp.layer1.weight.shape[1] == 5
    assert mlp(torch.zeros(3, 5)).shape == (3, 1)
