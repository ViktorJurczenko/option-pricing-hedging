import torch
import torch.nn as nn

from optionpricing.ml.surrogate.tree.tree_dataset import FEATURES

# Hidden-layer activations MLP accepts. The default is smooth: a ReLU net is
# piecewise linear, so its delta is a staircase and its gamma is 0, and the
# price helper reads both off the net by autograd. "relu" is listed for the
# ablation against the old architecture.
ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "softplus": nn.Softplus, "relu": nn.ReLU}
# Output layers: "linear" leaves the premium unconstrained, "softplus" forces
# it > 0, as the true premium is (an ablation decides between them).
OUTPUTS = ("linear", "softplus")


class MLP(nn.Module):
    """
    The surrogate net: standardized features (build_features, then
    FeatureStats.apply) -> the early-exercise premium of an American put
    per unit of strike, (American - European) / K, one value per row.

    `depth` hidden layers of `width` units, each followed by `activation`,
    then one linear layer to the output (passed through a softplus if
    `output="softplus"`). No BatchNorm or Dropout: every row's output
    depends on that row only, in training and evaluation alike, which the
    autograd Greeks rely on.

    Args:
        n_inputs: input features per row (len(FEATURES) for the tree dataset).
        depth: number of hidden layers (>= 1).
        width: units per hidden layer (>= 1).
        activation: a key of ACTIVATIONS.
        output: "linear" or "softplus".

    Raises:
        ValueError: for depth/width/n_inputs < 1 or an unknown activation/output.
    """

    def __init__(
        self,
        n_inputs: int = len(FEATURES),
        depth: int = 4,
        width: int = 128,
        activation: str = "silu",
        output: str = "linear",
    ):
        super().__init__()
        if min(n_inputs, depth, width) < 1:
            raise ValueError(f"n_inputs, depth and width must be >= 1, got {n_inputs}, {depth}, {width}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
        # the constructor's arguments, stored in the checkpoint to rebuild the same net
        self.config = {"n_inputs": n_inputs, "depth": depth, "width": width, "activation": activation, "output": output}

        layers, size = [], n_inputs
        for _ in range(depth):
            layers += [nn.Linear(size, width), ACTIVATIONS[activation]()]
            size = width
        layers.append(nn.Linear(size, 1))
        if output == "softplus":
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n, n_inputs) standardized features -> (n, 1) premium per unit of strike."""
        return self.net(x)
