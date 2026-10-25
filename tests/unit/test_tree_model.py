import numpy as np
import pytest
import torch

from ML.surrogate.tree.tree_dataset import FEATURES
from ML.surrogate.tree.tree_model import MLP


# --- MLP (the tree-trained net) ---

def test_mlp_default_is_four_silu_layers_of_128_on_the_dataset_features():
    mlp = MLP()
    linears = [l for l in mlp.net if isinstance(l, torch.nn.Linear)]
    assert [l.weight.shape[1] for l in linears] == [len(FEATURES), 128, 128, 128, 128]
    assert linears[-1].out_features == 1
    assert sum(isinstance(l, torch.nn.SiLU) for l in mlp.net) == 4
    # (6*128 + 128) + 3 * (128*128 + 128) + (128 + 1): the ~50k weights of REMINDER.md
    assert sum(p.numel() for p in mlp.parameters()) == 50561

def test_mlp_rows_are_independent_and_train_mode_changes_nothing():
    # no BatchNorm/Dropout: a row's output must not depend on its batch or on
    # train()/eval(); the price helper's sum-then-grad Greeks rely on it
    torch.manual_seed(0)
    mlp = MLP()
    x = torch.randn(16, len(FEATURES))
    full = mlp.train()(x)
    torch.testing.assert_close(mlp.eval()(x), full)
    torch.testing.assert_close(mlp(x[3:4]), full[3:4])

@pytest.mark.parametrize("output", ["linear", "softplus"])
def test_mlp_softplus_output_stays_positive_where_linear_goes_negative(output):
    # the last layer's bias forces a negative pre-output, so the result
    # doesn't depend on how a seed happens to initialize the weights
    torch.manual_seed(0)
    mlp = MLP(output=output)
    last = [l for l in mlp.net if isinstance(l, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(-1.0)
    out = mlp(torch.randn(8, len(FEATURES)))
    if output == "linear":
        assert (out == -1.0).all()
    else:
        torch.testing.assert_close(out, torch.full_like(out, float(np.log1p(np.exp(-1.0)))))  # softplus(-1) > 0

def test_mlp_silu_has_curvature_where_relu_has_none():
    # why SiLU: a ReLU net is piecewise linear, its second input derivative is 0
    for activation, curved in (("silu", True), ("relu", False)):
        torch.manual_seed(0)
        mlp = MLP(n_inputs=1, depth=2, width=16, activation=activation).double()
        x = torch.linspace(-2, 2, 41, dtype=torch.float64, requires_grad=True).unsqueeze(-1)
        d, = torch.autograd.grad(mlp(x).sum(), x, create_graph=True)
        dd, = torch.autograd.grad(d.sum(), x, allow_unused=True)
        has_curvature = dd is not None and bool((dd.abs() > 1e-12).any())
        assert has_curvature == curved

def test_mlp_config_rebuilds_the_same_architecture():
    mlp = MLP(n_inputs=3, depth=2, width=8, activation="tanh", output="softplus")
    rebuilt = MLP(**mlp.config)
    rebuilt.load_state_dict(mlp.state_dict())  # strict: same layers, same shapes
    assert isinstance(rebuilt.net[1], torch.nn.Tanh) and isinstance(rebuilt.net[-1], torch.nn.Softplus)

@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"width": 0}, {"n_inputs": 0},
                                    {"activation": "sigmoid"}, {"output": "relu"}])
def test_mlp_rejects_bad_architecture(kwargs):
    with pytest.raises(ValueError):
        MLP(**kwargs)
