"""The Deep Hedging policy: a no-transaction band network around the surrogate delta (Imaki et al. 2021).

At each hedge date `k` the network reads the day's features and the position held, and outputs two numbers
`a_k, b_k`, turned into a band around the surrogate delta `Δ_k`:

    l_k = -softplus(a_k) ≤ 0,   u_k = softplus(b_k) ≥ 0,   δ_k = clamp(δ_{k-1}, Δ_k + l_k, Δ_k + u_k)

Inside the band the position is exactly the one held, so `hedge_pnl` sees a zero trade and charges no cost.
The output layer starts with zero weights and a very negative bias, so the untrained band has the same tiny
width everywhere and the policy starts as the delta hedge (`docs/results/hedging-training.md`, "The band
network").

The network is a `pnl.Policy`: `model(inputs, position) -> position`, with `inputs` one hedge date of
`hedge_inputs`. It standardizes its features itself, with the training bank's statistics stored as buffers, so a
saved network prices raw features on any bank.
"""

import math
import os
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from optionpricing.ml.hedging.hedge_dataset import HedgeDataset, Hedger
from optionpricing.ml.surrogate.tree.tree_model import ACTIVATIONS

# The rows of the `inputs` tensor the network reads, in this order. Units:
# - log_moneyness: ln(S_k / K).
# - time_left: time to expiry `T - t_k`, in years.
# - sigma, rate: the contract's volatility and risk-free rate, per year.
# - delta: the surrogate's dV/dS of the put, in shares (in [-1, 0]). The band is centred on it.
# - gamma: the surrogate's d2V/dS2 of the put, in shares per dollar.
# - boundary_distance (optional): ln(S_k / S*(T - t_k)), how far the spot is above the exercise boundary.
FEATURES = ("log_moneyness", "time_left", "sigma", "rate", "delta", "gamma")
BOUNDARY_FEATURE = "boundary_distance"
DELTA = FEATURES.index("delta")
# Tag of the files `BandNetwork.save` writes, so `load` refuses anything else.
FORMAT = "hedge_policy"


def feature_names(boundary_input: bool) -> tuple[str, ...]:
    """The rows of `hedge_inputs`, with the boundary distance last when `boundary_input` is on."""
    return (*FEATURES, BOUNDARY_FEATURE) if boundary_input else FEATURES


def hedge_inputs(batch: dict[str, torch.Tensor], strike: float, boundary_input: bool = False) -> torch.Tensor:
    """Stack a batch of scenarios into the `inputs` tensor `hedge_pnl` slices for the network.

    Args:
        batch: a batch of `HedgeDataset` items (indexed with a tensor or list), holding `spot`, `delta`,
            `gamma`, `boundary` and `time_left` of shape `(nb_scenarios, n_days)` and `sigma`, `rate` of shape
            `(nb_scenarios,)`.
        strike: the bank's strike, in dollars, > 0.
        boundary_input: add the distance to the exercise boundary as a last row.

    Returns:
        A tensor of shape `(nb_scenarios, len(feature_names(boundary_input)), n_days)`, in the batch's dtype.
        Days from a scenario's `τ` on may be NaN: `hedge_pnl` never shows them to the policy.

    Raises:
        ValueError: if `strike` is not positive and finite, or `spot` is not 2-D.
    """
    if not (strike > 0 and math.isfinite(strike)):
        raise ValueError(f"strike must be positive and finite, got {strike}")
    spot = batch["spot"]
    if spot.ndim != 2:
        raise ValueError(f"spot must have shape (nb_scenarios, n_days), got {tuple(spot.shape)}")
    rows = [
        torch.log(spot / strike),
        batch["time_left"],
        batch["sigma"][:, None].expand(spot.shape),
        batch["rate"][:, None].expand(spot.shape),
        batch["delta"],
        batch["gamma"],
    ]
    if boundary_input:
        rows.append(torch.log(spot / batch["boundary"]))
    return torch.stack(rows, dim=1)


def fit_feature_stats(
    dataset: HedgeDataset, boundary_input: bool = False, chunk_size: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of each feature over the hedge dates of a bank, to standardize the inputs.

    Only days before each scenario's `τ` count, the days the policy is called on. The bank is read in chunks of
    scenarios, so a large bank never needs all its features in memory at once. A feature constant over the bank
    (e.g. one volatility for every contract) gets a standard deviation of 1, so standardizing only centres it.

    Args:
        dataset: a `HedgeDataset`, the training bank.
        boundary_input: whether the boundary distance is a feature.
        chunk_size: scenarios read at a time, >= 1.

    Returns:
        `(mean, std)`, float64 tensors of shape `(n_features,)`.

    Raises:
        ValueError: if `chunk_size` < 1 or the bank has no hedge date (every `τ` is 0).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    n_features = len(feature_names(boundary_input))
    total = torch.zeros(n_features, dtype=torch.float64)
    total_sq = torch.zeros(n_features, dtype=torch.float64)
    count = 0
    for start in range(0, len(dataset), chunk_size):
        batch = dataset[torch.arange(start, min(start + chunk_size, len(dataset)))]
        inputs = hedge_inputs(batch, dataset.strike, boundary_input).to(torch.float64)
        hedge_day = torch.arange(inputs.shape[2]) < batch["tau"][:, None]  # (nb_scenarios, n_days)
        values = inputs.permute(0, 2, 1)[hedge_day]  # (n_hedge_dates, n_features)
        total += values.sum(dim=0)
        total_sq += (values**2).sum(dim=0)
        count += len(values)
    if count == 0:
        raise ValueError("the bank has no hedge date (every scenario is exercised at once): nothing to fit")
    mean = total / count
    # E[x^2] - E[x]^2 can round below 0 for a constant feature
    std = (total_sq / count - mean**2).clamp_min(0).sqrt()
    std = torch.where(std > 1e-12 * mean.abs().clamp_min(1.0), std, torch.ones_like(std))
    return mean, std


class BandNetwork(nn.Module):
    """No-transaction band network: features and position held in, new position out (a `pnl.Policy`).

    `depth` hidden layers of `width` units, each followed by `activation`, then a linear layer to the band's two
    logits. The input is the standardized features plus the position held, which matters under costs: the best
    trade depends on where the hedge already is.

    Args:
        mean, std: the features' statistics (`fit_feature_stats`), shape `(n_features,)`. Their length sets the
            number of features, `len(feature_names(boundary_input))`.
        depth: number of hidden layers, >= 1.
        width: units per hidden layer, >= 1.
        activation: a key of `ACTIVATIONS`. Nothing differentiates the band with respect to the inputs (unlike
            the surrogate, whose Greeks are autograd derivatives), so any of them can train it. SiLU gives a
            smooth band, ReLU a piecewise-linear one.
        init_logit: bias of both band logits at initialization, finite. The untrained half-widths are
            `softplus(init_logit)` shares, 0.0067 at the default -5: the delta hedge to within a few thousandths
            of a share. More negative is closer to delta, but Adam moves the bias by about one learning rate per
            step, so the band would take longer to open.

    Raises:
        ValueError: if `mean` and `std` differ in shape or are not 1-D with at least `DELTA + 1` entries, a std
            is not positive, `depth` or `width` < 1, `activation` is unknown, or `init_logit` is not finite.
    """

    def __init__(
        self,
        mean: Sequence[float] | torch.Tensor,
        std: Sequence[float] | torch.Tensor,
        depth: int = 4,
        width: int = 32,
        init_logit: float = -5.0,
        activation: str = "silu",
    ):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.ndim != 1 or mean.shape != std.shape or len(mean) <= DELTA:
            raise ValueError(
                f"mean and std must be 1-D of the same length > {DELTA}, got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not torch.all(std > 0):
            raise ValueError("every std must be > 0")
        if min(depth, width) < 1:
            raise ValueError(f"depth and width must be >= 1, got {depth} and {width}")
        if not math.isfinite(init_logit):
            raise ValueError(f"init_logit must be finite, got {init_logit}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}, got {activation!r}")
        self.hparams = {"depth": depth, "width": width, "init_logit": init_logit, "activation": activation}
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        layers = []
        n_in = len(mean) + 1  # the features plus the position held
        for _ in range(depth):
            layers += [nn.Linear(n_in, width), ACTIVATIONS[activation]()]
            n_in = width
        out = nn.Linear(n_in, 2)
        with torch.no_grad():
            out.weight.zero_()  # the same band for every input at the start
            out.bias.fill_(init_logit)
        self.net = nn.Sequential(*layers, out)

    @property
    def n_features(self) -> int:
        """Number of feature rows the network reads (6, or 7 with the boundary distance)."""
        return len(self.mean)

    def band(self, inputs: torch.Tensor, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The band's edges `(Δ - l, Δ + u)` for one hedge date, each of shape `(nb_scenarios,)`.

        Args:
            inputs: one hedge date of `hedge_inputs`, shape `(nb_scenarios, n_features)`.
            position: shares held before the trade, shape `(nb_scenarios,)`.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features or position.shape != inputs.shape[:1]:
            raise ValueError(
                f"inputs must have shape (nb_scenarios, {self.n_features}) and position (nb_scenarios,), got "
                f"{tuple(inputs.shape)} and {tuple(position.shape)}"
            )
        dtype = self.mean.dtype
        x = torch.cat([(inputs.to(dtype) - self.mean) / self.std, position.to(dtype)[:, None]], dim=1)
        logits = self.net(x).to(inputs.dtype)
        delta = inputs[:, DELTA]
        return delta - F.softplus(logits[:, 0]), delta + F.softplus(logits[:, 1])

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Hold `position` if it is inside the band, else trade to the nearest edge. Shape `(nb_scenarios,)`."""
        lower, upper = self.band(inputs, position)
        # clamp returns its input unchanged between the bounds, so a held position is exact (no trade, no fee)
        return torch.clamp(position, lower, upper)

    def save(self, path: str | os.PathLike, **extra) -> None:
        """Save the network with everything `load` needs to rebuild it, plus `extra` (e.g. the training config)."""
        torch.save(
            {
                "format": FORMAT,
                "hparams": self.hparams,
                "n_features": self.n_features,
                "state_dict": self.state_dict(),
                "extra": extra,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> tuple["BandNetwork", dict]:
        """Rebuild a network saved by `save`. Returns `(network, extra)`.

        Raises:
            ValueError: if the file was not written by `BandNetwork.save`.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
            raise ValueError(f"'{path}' is not a hedge policy (BandNetwork.save, format '{FORMAT}')")
        n = checkpoint["n_features"]
        model = cls(torch.zeros(n), torch.ones(n), **checkpoint["hparams"])
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint["extra"]


def network_hedger(model: BandNetwork, boundary_input: bool = False) -> Hedger:
    """The band network with its own features (`hedge_inputs`), the boundary distance included if it reads it.

    Args:
        model: the network.
        boundary_input: whether the network was trained with the boundary distance, as its training config says.

    Returns:
        The `Hedger`, for `hedge_batch` and `hedge_bank`.

    Raises:
        ValueError: if `boundary_input` does not match the network's number of features.
    """
    expected = len(feature_names(boundary_input))
    if model.n_features != expected:
        raise ValueError(
            f"the network reads {model.n_features} features, but boundary_input={boundary_input} gives {expected}"
        )
    return Hedger(model, lambda batch, strike: hedge_inputs(batch, strike, boundary_input))
