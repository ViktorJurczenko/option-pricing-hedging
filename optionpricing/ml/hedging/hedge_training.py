"""Train the band network by Deep Hedging: minimize a risk measure of the terminal P&L by backpropagation.

Each step draws a batch of scenarios from the training bank, hedges them with the network through `hedge_pnl`
and lowers a risk measure of the batch's P&Ls (`risk.py`) by Adam. The gradient flows through every hedge date
of every path, so the objective is the metric the hedger is judged on, with no critic and no reward shaping
(`docs/results/hedging-training.md`, "Deep Hedging instead of SAC").

The objectives:
- `cvar`: the Rockafellar-Uryasev form `cvar_loss(pnl, w, alpha)`, minimized over the network and the threshold
  `w` together. `w` has its own learning rate, so an experiment can check that it needs one.
- `entropic` and `variance`: `RiskMeasure.loss` directly.

After every epoch the network hedges the validation bank (`hedge_bank`) and is scored by the same risk measure,
computed exactly (`RiskMeasure.loss`: for CVaR, `-cvar(pnl, alpha)`, not the training form with the current `w`). The best epoch is kept. The untrained
network, the delta hedge to within its initial band, is scored first as epoch 0, so training never returns a
network that hedges worse than delta on validation.
"""

import math
import os
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
from tqdm import tqdm

from optionpricing.hedging.pnl import HedgeSettings
from optionpricing.hedging.risk import RiskMeasure, cvar_loss
from optionpricing.ml.hedging.hedge_dataset import HedgeDataset, hedge_bank, hedge_batch
from optionpricing.ml.hedging.hedge_model import ACTIVATIONS, BandNetwork, fit_feature_stats, network_hedger

# The defaults of `HedgeTrainConfig`. The network size, activation and learning rates are starting points, each to
# be set by a check on the full banks.
DEPTH = 4
WIDTH = 32  # 4 x 32, the network size of Imaki et al.'s band network
INIT_LOGIT = -5.0  # untrained half-width softplus(-5) = 0.0067 shares
ACTIVATION = "silu"  # any key of ACTIVATIONS trains the band
EPOCHS = 50
# ~200 paths in a 95% tail (docs/results/hedging-training.md, "Deep Hedging instead of SAC")
BATCH_SIZE = 4096
# the best of 1e-3, 3e-3 and 1e-2 on the full banks (docs/results/hedging-results.md, "Learning rates")
LR = 1e-2
# w lives in dollars and must track the batch's VaR as the policy changes. Rates up to 30x faster scored the same
# at a 1% cost (docs/results/hedging-results.md, "Learning rates").
W_LR = 1e-2
SEED = 42  # the repo's usual seed


@dataclass(frozen=True)
class HedgeTrainConfig:
    """Everything a training run depends on: a run is reproduced from its config and its two banks.

    Network: `depth`, `width`, `init_logit`, `activation` (`BandNetwork`'s arguments) and `boundary_input` (the
    distance to the exercise boundary as an 8th input).
    `hedging`: how the training hedges are booked (`HedgeSettings`), the same type the test's `EvalConfig` holds.
    `risk`: the objective (`RiskMeasure`), which is also the validation score.
    Optimization: `epochs` (full passes over the training bank), `batch_size` (scenarios per step; CVaR is a
    property of the batch's tail, so batches are large), `lr` (the network's Adam rate) and `w_lr` (the CVaR
    threshold's).
    `seed`: seeds the weights' initialization and the batch order.

    Raises:
        ValueError: for an unknown activation, or a non-positive size, count or learning rate. `HedgeSettings` and
            `RiskMeasure` check their own fields.
    """

    depth: int = DEPTH
    width: int = WIDTH
    init_logit: float = INIT_LOGIT
    activation: str = ACTIVATION
    boundary_input: bool = False
    hedging: HedgeSettings = field(default_factory=HedgeSettings)
    risk: RiskMeasure = field(default_factory=RiskMeasure)
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    w_lr: float = W_LR
    seed: int = SEED

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}, got {self.activation!r}")
        for name in ("depth", "width", "epochs", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("lr", "w_lr"):
            value = getattr(self, name)
            if not (value > 0 and math.isfinite(value)):
                raise ValueError(f"{name} must be positive and finite, got {value}")

    @classmethod
    def from_dict(cls, values: dict) -> "HedgeTrainConfig":
        """Rebuild a config from `dataclasses.asdict(config)`, as a checkpoint stores it."""
        return cls(**{**values, "hedging": HedgeSettings(**values["hedging"]), "risk": RiskMeasure(**values["risk"])})


class TrainReport(NamedTuple):
    """What `train_hedging_model` returns.

    Attributes:
        model: the best network on validation, on the CPU.
        history: one dict per epoch, epoch 0 being the untrained network: `epoch`, `train_loss` (the mean
            training objective of the epoch's steps, NaN at epoch 0), `valid_score` (the validation risk, lower
            is better), `valid_mean_pnl`, `valid_mean_costs` and `w` (the CVaR threshold, NaN for the other
            objectives).
        best_epoch: the epoch whose network is returned, 0 if training never beat the untrained network.
    """

    model: BandNetwork
    history: list[dict]
    best_epoch: int


@torch.no_grad()
def _validate(model: BandNetwork, dataset: HedgeDataset, config: HedgeTrainConfig) -> dict[str, float]:
    """Score `model` on every scenario of `dataset`: `{"score": the exact risk, "mean_pnl", "mean_costs"}`."""
    outcome = hedge_bank(network_hedger(model, config.boundary_input), dataset, config.hedging, config.batch_size)
    return {
        "score": float(config.risk.loss(outcome.pnl)),
        "mean_pnl": float(outcome.pnl.mean()),
        "mean_costs": float(outcome.costs.mean()),
    }


def train_hedging_model(
    train_path: str | os.PathLike,
    valid_path: str | os.PathLike,
    config: HedgeTrainConfig = HedgeTrainConfig(),  # noqa: B008 (frozen dataclass, immutable)
    checkpoint_path: str | os.PathLike | None = None,
) -> TrainReport:
    """Train a band network on a training bank, keeping the epoch that scores best on a validation bank.

    The banks come from `generate_hedge_dataset` with the same settings and distinct seeds (`check_compatible`).
    The network standardizes its inputs with the training bank's statistics. For CVaR, `w` starts at the VaR of
    the untrained network's loss on the first batch, so it begins near its optimum. A step whose batch has no
    hedge date (every scenario exercised at once) has no gradient and is skipped.

    Args:
        train_path: the training bank.
        valid_path: the validation bank.
        config: the run's settings.
        checkpoint_path: where to save the best network (`BandNetwork.save`, with the config as `asdict` and the
            training bank's settings as extras), each time it improves. None saves nothing. The config is rebuilt
            with `HedgeTrainConfig.from_dict`.

    Returns:
        The `TrainReport`.

    Raises:
        ValueError: if a bank is not a hedge bank, the banks do not belong together, or the validation score is
            never finite.
    """
    train_data = HedgeDataset(train_path)
    valid_data = HedgeDataset(valid_path, reference=train_data.settings)
    mean, std = fit_feature_stats(train_data, config.boundary_input, config.batch_size)
    torch.manual_seed(config.seed)  # the weights' initialization
    model = BandNetwork(mean, std, config.depth, config.width, config.init_logit, config.activation)
    batch_order = torch.Generator().manual_seed(config.seed)
    hedger = network_hedger(model, config.boundary_input)

    w = None
    groups = [{"params": model.parameters(), "lr": config.lr}]
    if config.risk.name == "cvar":
        first = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(config.seed))
        with torch.no_grad():
            loss = -hedge_batch(hedger, train_data, first[: config.batch_size], config.hedging).pnl
        w = torch.nn.Parameter(torch.quantile(loss.to(torch.float64), config.risk.alpha).to(loss.dtype))
        groups.append({"params": [w], "lr": config.w_lr})
    optimizer = torch.optim.Adam(groups)

    def save():
        if checkpoint_path is not None:
            model.save(checkpoint_path, config=asdict(config), bank_settings=train_data.settings)

    def record(epoch, train_loss):
        metrics = _validate(model, valid_data, config)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"valid_{k}": v for k, v in metrics.items()},
                "w": math.nan if w is None else float(w.detach()),
            }
        )
        return metrics["score"]

    history = []
    best_score = record(0, math.nan)
    best_epoch, best_state = 0, {k: v.clone() for k, v in model.state_dict().items()}
    if math.isfinite(best_score):
        save()
    else:
        best_score = math.inf  # any finite epoch replaces the non-finite start
    steps_per_epoch = math.ceil(len(train_data) / config.batch_size)
    bar = tqdm(range(1, config.epochs + 1), desc="Training hedger", unit="epoch")
    for epoch in bar:
        model.train()
        order = torch.randperm(len(train_data), generator=batch_order)
        total, steps = 0.0, 0
        for step in range(steps_per_epoch):
            idx = order[step * config.batch_size : (step + 1) * config.batch_size]
            pnl = hedge_batch(hedger, train_data, idx, config.hedging).pnl
            loss = cvar_loss(pnl, w, config.risk.alpha) if w is not None else config.risk.loss(pnl)
            if not pnl.requires_grad:  # no hedge date in this batch: the network played no part
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss.detach()), steps + 1
        model.eval()
        score = record(epoch, total / steps if steps else math.nan)
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save()
        bar.set_postfix(train=f"{history[-1]['train_loss']:.4f}", valid=f"{score:.4f}", best=f"{best_epoch}")

    if not math.isfinite(best_score):
        raise ValueError("the validation score was never finite: training diverged (try a lower lr)")
    model.load_state_dict(best_state)
    return TrainReport(model, history, best_epoch)
