"""Score the trained band network and every baseline on the same test bank, with the risk measures of `risk.py`.

The test step of the Deep Hedging pipeline: `generate_hedge_dataset` writes the banks, `hedge_training` trains
the network, and this module books every hedger's P&L on one test bank with `hedge_bank`. All hedgers see the
same paths (common random numbers), so their differences come from the hedging, not from the draw of paths.

The hedgers:
- `no hedge`: never trades, the worst case a hedge must beat.
- `delta`: the surrogate delta stored in the bank, rebalanced at every hedge date.
- `band`: a no-trade band of fixed half-width around delta (`band_hedge`), its width tuned on a validation bank.
- `whalley-wilmott`: the Whalley-Wilmott band (`whalley_wilmott`), built for the proportional cost charged, its
  risk aversion tuned on the validation bank.
- `network`: the trained `BandNetwork`, if a checkpoint is given.

The hedging settings are chosen here and can differ from the training run's (`dataclasses.replace` on its
`HedgeSettings`), so a policy can be tested at costs it was not trained on. The bands are tuned at the settings
they are scored at, so each baseline is the best of its family for the test, not for the network's training.
"""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from tqdm import tqdm

from optionpricing.hedging.baselines import (
    BAND_TARGETS,
    band_hedge,
    baseline_inputs,
    delta_hedge,
    no_hedge,
    tune,
    whalley_wilmott,
)
from optionpricing.hedging.pnl import HedgeOutcome, HedgeSettings, Policy
from optionpricing.hedging.risk import RiskMeasure, cvar, entropic, variance
from optionpricing.ml.hedging.hedge_dataset import HedgeDataset, Hedger, check_compatible, hedge_bank
from optionpricing.ml.hedging.hedge_model import BandNetwork, network_hedger
from optionpricing.ml.hedging.hedge_training import BATCH_SIZE, HedgeTrainConfig

# The names of the hedgers in an `EvalReport`, in the order they are reported.
NO_HEDGE, DELTA, BAND, WHALLEY_WILMOTT, NETWORK = "no hedge", "delta", "band", "whalley-wilmott", "network"
# Half-widths tried for `band`, in shares. 0 is the delta hedge. The grid is fine (0.0125 apart) from 0.05 to 0.125,
# around the 0.075 that scored best at 1% cost in the first evaluation, where a grid doubling at each step had
# stepped over it, and coarse out to 0.4 (docs/results/hedging-results.md, "First evaluation").
BAND_WIDTHS = (0.0, 0.025, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.15, 0.175, 0.2, 0.4)
# Risk aversions tried for `whalley-wilmott`, in 1/dollars, from 0.1 to 10, around the 1.4 that scored best at 1% cost
# in the first evaluation, and finer between 0.3 and 3, where the score is nearly flat (same section).
RISK_AVERSIONS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 10.0)


@dataclass(frozen=True)
class EvalConfig:
    """The settings a test is run at. They need not be the training run's.

    Attributes:
        hedging: how every hedger is booked (`HedgeSettings`), the same type the training config holds.
        risk: the measure the bands are tuned on (`RiskMeasure`), the network's training objective for a fair
            comparison. Its `alpha` and `lam` also set the reported CVaR and entropic risk.
        band_widths: half-widths tried for `band`, in shares (`BAND_WIDTHS` by default).
        risk_aversions: `γ` tried for `whalley-wilmott`, in 1/dollars (`RISK_AVERSIONS` by default).
        band_target: `"edge"` or `"delta"`, where both bands trade to outside the band.
        batch_size: scenarios hedged at a time. It only affects speed and memory.

    Raises:
        ValueError: for an unknown band target, a `batch_size` < 1, or an empty grid or one with an invalid value.
            `HedgeSettings` and `RiskMeasure` check their own fields.
    """

    hedging: HedgeSettings = field(default_factory=HedgeSettings)
    risk: RiskMeasure = field(default_factory=RiskMeasure)
    band_widths: tuple[float, ...] = BAND_WIDTHS
    risk_aversions: tuple[float, ...] = RISK_AVERSIONS
    band_target: str = "edge"
    batch_size: int = BATCH_SIZE  # the training's default; here it only affects speed and memory

    def __post_init__(self):
        if self.band_target not in BAND_TARGETS:
            raise ValueError(f"band_target must be one of {BAND_TARGETS}, got {self.band_target!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not self.band_widths or not self.risk_aversions:
            raise ValueError("band_widths and risk_aversions must each hold at least one value")
        # build one policy per grid value, so a bad value fails here rather than after the first scores
        for width in self.band_widths:
            band_hedge(width, self.band_target)
        for gamma in self.risk_aversions:
            whalley_wilmott(gamma, self.hedging.proportional, self.band_target)


class EvalReport(NamedTuple):
    """What `evaluate_hedgers` returns.

    Attributes:
        metrics: one dict per hedger, in the order of the module docstring, from `summarize`.
        pnl: the test P&L of each hedger, per scenario, in the same order of scenarios for all, so two hedgers
            can be compared path by path.
        tuned: for `band` and `whalley-wilmott`, `{"best": the value chosen, "grid": the values tried, "scores":
            the validation score of each, higher is better}`. A best value on the edge of the grid means the grid
            is too narrow.
    """

    metrics: dict[str, dict[str, float]]
    pnl: dict[str, torch.Tensor]
    tuned: dict[str, dict]


def bank_baseline_inputs(batch: dict[str, torch.Tensor], strike: float) -> torch.Tensor:
    """The baselines' `inputs` (`baseline_inputs`) for a batch of a bank, from its stored surrogate Greeks.

    The bank stores gamma in shares per dollar already, so it is passed as is. The strike is not read: the
    baselines do not need it, and the argument is there to match `hedge_dataset.Features`.
    """
    return baseline_inputs(batch["spot"], batch["delta"], batch["gamma"], batch["time_left"], batch["rate"][:, None])


def baseline_hedger(policy: Policy) -> Hedger:
    """A baseline policy with the bank's baseline features."""
    return Hedger(policy, bank_baseline_inputs)


def summarize(outcome: HedgeOutcome, risk: RiskMeasure) -> dict[str, float]:
    """The reported statistics of one hedger's outcome, in dollars except `mean_trades`.

    Args:
        outcome: the hedger's outcome on a bank, from `hedge_bank`.
        risk: its `alpha` and `lam` set the reported CVaR and entropic risk.

    Returns:
        A dict with:
        - `mean_pnl` and `se_mean_pnl`, its standard error `std / sqrt(n)`.
        - `std_pnl`, the population standard deviation (`variance` of `risk.py`, square-rooted).
        - `cvar`: `cvar(pnl, alpha)`, on the P&L scale (higher is better).
        - `entropic`: `entropic(pnl, lam)`, a loss (lower is better).
        - `mean_costs` and `mean_trades` (`HedgeOutcome.trades`), per scenario.
    """
    pnl = outcome.pnl
    std = float(variance(pnl).sqrt())
    return {
        "mean_pnl": float(pnl.mean()),
        "se_mean_pnl": std / math.sqrt(len(pnl)),
        "std_pnl": std,
        "cvar": float(cvar(pnl, risk.alpha)),
        "entropic": float(entropic(pnl, risk.lam)),
        "mean_costs": float(outcome.costs.mean()),
        "mean_trades": float(outcome.trades.to(pnl.dtype).mean()),
    }


def tune_baseline(
    make_policy: Callable[[float], Policy],
    grid: Sequence[float],
    valid: HedgeDataset,
    config: EvalConfig,
    bar: tqdm | None = None,
) -> tuple[float, list[float]]:
    """Pick the parameter of a baseline family that scores best on a validation bank, by `config.risk`.

    The score is `-config.risk.loss` of the validation P&L, so higher is better, and for CVaR it is the CVaR.

    Args:
        make_policy: builds a policy from one parameter value, as in `baselines.tune`.
        grid: the parameter values tried, at least one.
        valid: the validation bank.
        config: the hedging settings and the risk measure.
        bar: a progress bar advanced once per value scored, or None.

    Returns:
        `(best, scores)`, as `baselines.tune`.

    Raises:
        ValueError: as `baselines.tune`, if `grid` is empty or a score is not finite.
    """

    def score(policy: Policy) -> float:
        pnl = hedge_bank(baseline_hedger(policy), valid, config.hedging, config.batch_size).pnl
        result = -float(config.risk.loss(pnl))
        if bar is not None:
            bar.update()
        return result

    return tune(make_policy, grid, score)


def evaluate_hedgers(
    test_path: str | os.PathLike,
    valid_path: str | os.PathLike,
    checkpoint_path: str | os.PathLike | None = None,
    config: EvalConfig = EvalConfig(),  # noqa: B008 (frozen dataclass, immutable)
) -> EvalReport:
    """Score every baseline, and the network if a checkpoint is given, on a test bank.

    The bands are tuned on the validation bank, then every hedger is scored on the test bank, in float64. The two
    banks must share their settings and differ in seed (`check_compatible`). A network is only tested on a bank
    from its training bank's settings, with another seed, so it is never scored on the paths it was trained on.
    Each hedger is summarized as soon as it is booked, so only one outcome's positions are held at a time.

    Args:
        test_path: the test bank (`generate_hedge_dataset`, e.g. `--split test`).
        valid_path: the bank the bands are tuned on. The network's validation bank can be reused.
        checkpoint_path: a network saved by `train_hedging_model` (`BandNetwork.save` with its config and training bank
            settings). None tests the baselines only.
        config: the test's settings.

    Returns:
        The `EvalReport`.

    Raises:
        ValueError: if a file is not a bank or a policy, the banks do not belong together, the test bank does not
            belong with the network's training bank, or a tuning score is not finite.
    """
    test = HedgeDataset(test_path, dtype=torch.float64)
    valid = HedgeDataset(valid_path, dtype=torch.float64, reference=test.settings)
    network = None
    if checkpoint_path is not None:
        model, extra = BandNetwork.load(checkpoint_path)
        check_compatible(extra["bank_settings"], test.settings)
        model.eval()
        network = network_hedger(model, HedgeTrainConfig.from_dict(extra["config"]).boundary_input)
    families = {
        BAND: (lambda h: band_hedge(h, config.band_target), config.band_widths),
        WHALLEY_WILMOTT: (
            lambda g: whalley_wilmott(g, config.hedging.proportional, config.band_target),
            config.risk_aversions,
        ),
    }
    # One pass over a bank per grid value, then one per hedger on the test bank
    n_hedgers = 2 + len(families) + (network is not None)
    total = len(config.band_widths) + len(config.risk_aversions) + n_hedgers
    with tqdm(total=total, desc="Testing hedgers", unit="pass") as bar:
        hedgers = {NO_HEDGE: baseline_hedger(no_hedge), DELTA: baseline_hedger(delta_hedge)}
        tuned = {}
        for name, (make_policy, grid) in families.items():
            bar.set_postfix(tuning=name)
            best, scores = tune_baseline(make_policy, grid, valid, config, bar)
            tuned[name] = {"best": best, "grid": list(grid), "scores": scores}
            hedgers[name] = baseline_hedger(make_policy(best))
        if network is not None:
            hedgers[NETWORK] = network

        metrics, pnl = {}, {}
        for name, hedger in hedgers.items():
            bar.set_postfix(testing=name)
            outcome = hedge_bank(hedger, test, config.hedging, config.batch_size)
            metrics[name], pnl[name] = summarize(outcome, config.risk), outcome.pnl
            bar.update()
    return EvalReport(metrics=metrics, pnl=pnl, tuned=tuned)
