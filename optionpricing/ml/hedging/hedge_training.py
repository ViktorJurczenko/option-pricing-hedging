"""Train the band network by Deep Hedging: minimize a risk measure of the terminal P&L by backpropagation.

Each step draws a batch of scenarios from the training bank, hedges them with the network through `hedge_pnl`
and lowers a risk measure of the batch's P&Ls (`risk.py`) by Adam. The gradient flows through every hedge date
of every path, so the objective is the metric the hedger is judged on, with no critic and no reward shaping
(`docs/results/hedging-training.md`, "Deep Hedging instead of SAC").

The objectives:
- `cvar`: the Rockafellar-Uryasev form `cvar_loss(pnl, w, alpha)`, minimized over the network and the threshold
  `w` together. `w` has its own learning rate, so an experiment can check that it needs one.
- `entropic`: `entropic(pnl, lam)`.
- `variance`: `variance(pnl)`.

After every epoch the network is scored on a validation bank by the same risk measure, computed exactly (for
CVaR, `-cvar(pnl, alpha)`, not the training form with the current `w`). The best epoch is kept. The untrained
network, the delta hedge to within its initial band, is scored first as epoch 0, so training never returns a
network that hedges worse than delta on validation.
"""

import math
import os
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from tqdm import tqdm

from optionpricing.hedging.costs import transaction_cost
from optionpricing.hedging.pnl import SETTLEMENTS, HedgeOutcome, hedge_pnl
from optionpricing.hedging.risk import cvar, cvar_loss, entropic, variance
from optionpricing.ml.hedging.hedge_dataset import HedgeDataset
from optionpricing.ml.hedging.hedge_model import ACTIVATIONS, BandNetwork, fit_feature_stats, hedge_inputs

OBJECTIVES = ("cvar", "entropic", "variance")


@dataclass(frozen=True)
class HedgeTrainConfig:
    """Everything a training run depends on: a run is reproduced from its config and its two banks.

    Network: `depth`, `width`, `init_logit`, `activation` (`BandNetwork`'s arguments) and `boundary_input` (the
    distance to the exercise boundary as an 8th input).
    Objective: `objective` (one of `OBJECTIVES`), `alpha` (CVaR level), `lam` (entropic risk aversion, in
    1/dollars).
    Hedging: `rebalance_every`, `settlement`, `proportional`, `fixed`, `kappa`, as in `hedge_pnl`. `kappa` is
    needed when `fixed > 0`: it shapes the fee's straight-through gradient.
    Optimization: `epochs` (full passes over the training bank), `batch_size` (scenarios per step; CVaR is a
    property of the batch's tail, so batches are large), `lr` (the network's Adam rate) and `w_lr` (the CVaR
    threshold's).
    `seed`: seeds the weights' initialization and the batch order.

    Raises:
        ValueError: for an unknown objective, settlement or activation, `alpha` outside (0, 1), a non-positive size, count,
            learning rate or `lam`, a negative cost, or `fixed > 0` without `kappa`.
    """

    depth: int = 4
    width: int = 32  # 4 x 32, the network size of Imaki et al.'s band network; set by a check later
    init_logit: float = -5.0  # untrained half-width softplus(-5) = 0.0067 shares
    activation: str = "silu"  # any key of ACTIVATIONS trains the band; set by a check later
    boundary_input: bool = False
    objective: str = "cvar"
    alpha: float = 0.95  # the project's CVaR level (docs/results/hedging-training.md, "The defaults")
    lam: float = 1.0
    rebalance_every: int = 1  # daily
    settlement: str = "cash"
    proportional: float = 0.0
    fixed: float = 0.0
    kappa: float | None = None
    epochs: int = 50
    batch_size: int = 4096  # ~200 paths in a 95% tail (docs/results/hedging-training.md, "Deep Hedging")
    lr: float = 1e-3  # Adam's standard start
    # 10x the network's: w lives in dollars and must track the batch's VaR as the policy changes
    w_lr: float = 1e-2
    seed: int = 42  # the repo's usual seed

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(ACTIVATIONS)}, got {self.activation!r}")
        if self.settlement not in SETTLEMENTS:
            raise ValueError(f"settlement must be one of {tuple(SETTLEMENTS)}, got {self.settlement!r}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        for name in ("depth", "width", "rebalance_every", "epochs", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("lam", "lr", "w_lr"):
            value = getattr(self, name)
            if not (value > 0 and math.isfinite(value)):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        # transaction_cost checks the rates and kappa; a zero trade exercises the checks without a gradient
        transaction_cost(0.0, 1.0, self.proportional, self.fixed, self.kappa)
        if self.fixed > 0 and self.kappa is None:
            raise ValueError("kappa is needed when fixed > 0: it shapes the fee's gradient")


class TrainResult(NamedTuple):
    """What `train_hedger` returns.

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


def risk_score(pnl: torch.Tensor, config: HedgeTrainConfig) -> torch.Tensor:
    """The risk of a P&L sample under the config's objective, as a loss (lower is better), computed exactly.

    For CVaR this is `-cvar(pnl, alpha)`, the minimum of the training form over `w`, so it does not depend on the
    current threshold.
    """
    if config.objective == "cvar":
        return -cvar(pnl, config.alpha)
    if config.objective == "entropic":
        return entropic(pnl, config.lam)
    return variance(pnl)


def _hedge(model: BandNetwork, dataset: HedgeDataset, idx: torch.Tensor, config: HedgeTrainConfig) -> HedgeOutcome:
    """Hedge the scenarios `idx` of `dataset` with `model` and return their `HedgeOutcome`."""
    batch = dataset[idx]
    return hedge_pnl(
        model,
        hedge_inputs(batch, dataset.strike, config.boundary_input),
        batch["spot"],
        batch["tau"],
        batch["premium"],
        dataset.strike,
        batch["rate"],
        rebalance_every=config.rebalance_every,
        settlement=config.settlement,
        proportional=config.proportional,
        fixed=config.fixed,
        kappa=config.kappa,
    )


@torch.no_grad()
def evaluate(model: BandNetwork, dataset: HedgeDataset, config: HedgeTrainConfig) -> dict[str, float]:
    """Score `model` on every scenario of `dataset`, in batches of `config.batch_size`.

    Returns:
        `{"score": risk_score of all P&Ls, "mean_pnl": ..., "mean_costs": ...}`, in dollars.
    """
    pnl, costs = [], []
    for start in range(0, len(dataset), config.batch_size):
        outcome = _hedge(model, dataset, torch.arange(start, min(start + config.batch_size, len(dataset))), config)
        pnl.append(outcome.pnl)
        costs.append(outcome.costs)
    pnl, costs = torch.cat(pnl), torch.cat(costs)
    return {"score": float(risk_score(pnl, config)), "mean_pnl": float(pnl.mean()), "mean_costs": float(costs.mean())}


def train_hedger(
    train_path: str | os.PathLike,
    valid_path: str | os.PathLike,
    config: HedgeTrainConfig = HedgeTrainConfig(),  # noqa: B008 (frozen dataclass, immutable)
    checkpoint_path: str | os.PathLike | None = None,
) -> TrainResult:
    """Train a band network on a training bank, keeping the epoch that scores best on a validation bank.

    The banks come from `generate_hedge_dataset` with the same settings and distinct seeds (`check_compatible`).
    The network standardizes its inputs with the training bank's statistics. For CVaR, `w` starts at the VaR of
    the untrained network's loss on the first batch, so it begins near its optimum. A step whose batch has no
    hedge date (every scenario exercised at once) has no gradient and is skipped.

    Args:
        train_path: the training bank.
        valid_path: the validation bank.
        config: the run's settings.
        checkpoint_path: where to save the best network (`BandNetwork.save`, with the config and the training
            bank's settings as extras), each time it improves. None saves nothing.

    Returns:
        The `TrainResult`.

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

    w = None
    groups = [{"params": model.parameters(), "lr": config.lr}]
    if config.objective == "cvar":
        first = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(config.seed))
        with torch.no_grad():
            loss = -_hedge(model, train_data, first[: config.batch_size], config).pnl
        w = torch.nn.Parameter(torch.quantile(loss.to(torch.float64), config.alpha).to(loss.dtype))
        groups.append({"params": [w], "lr": config.w_lr})
    optimizer = torch.optim.Adam(groups)

    def save():
        if checkpoint_path is not None:
            model.save(checkpoint_path, config=asdict(config), bank_settings=train_data.settings)

    def record(epoch, train_loss):
        metrics = evaluate(model, valid_data, config)
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
            pnl = _hedge(
                model, train_data, order[step * config.batch_size : (step + 1) * config.batch_size], config
            ).pnl
            loss = cvar_loss(pnl, w, config.alpha) if w is not None else risk_score(pnl, config)
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
    return TrainResult(model, history, best_epoch)
