"""The test-hedge command: score the trained band network and every baseline on a test bank.

The bands are tuned on a validation bank, then every hedger hedges the same test paths (`evaluate_hedgers`). The
hedging and risk flags default to the network's training settings, read from its checkpoint, so the network is
tested at the costs and on the objective it was trained for unless a flag changes them. `--baselines_only` scores
the baselines alone, at the library defaults unless a flag says otherwise.
"""

import json
from dataclasses import asdict
from pathlib import Path

from optionpricing.hedging.baselines import BAND_TARGETS
from optionpricing.hedging.risk import RiskMeasure
from optionpricing.ml.hedging.hedge_evaluation import (
    BAND,
    BAND_WIDTHS,
    RISK_AVERSIONS,
    WHALLEY_WILMOTT,
    EvalConfig,
    EvalReport,
    evaluate_hedgers,
)
from optionpricing.ml.hedging.hedge_model import BandNetwork
from optionpricing.ml.hedging.hedge_training import BATCH_SIZE, HedgeTrainConfig
from scripts.hedge_flags import add_settings_arguments, describe_measure, describe_settings, resolve_settings
from scripts.train_hedge import CHECKPOINT_PATH, VALID_PATH, check_banks

TEST_PATH = "artifacts/data/hedge_test.npz"
# The tuned parameter of each band, with its unit and its flag, for the report.
TUNED = {BAND: ("half-width", "shares", "--band_widths"), WHALLEY_WILMOTT: ("risk aversion", "1/$", "--risk_aversions")}


def add_arguments(parser):
    parser.add_argument("--test_path", default=TEST_PATH, type=str, help="The test bank (--split test).")
    parser.add_argument(
        "--valid_path",
        default=VALID_PATH,
        type=str,
        help="The bank the bands are tuned on (--split val). The network's validation bank can be reused.",
    )
    parser.add_argument(
        "--checkpoint_path", default=CHECKPOINT_PATH, type=str, help="The network saved by train-hedge."
    )
    parser.add_argument("--baselines_only", action="store_true", help="Score the baselines only, without a network.")
    parser.add_argument(
        "--report_path",
        default=None,
        type=str,
        help="A JSON file for the settings, the metrics and the tuned bands. Default: none written.",
    )
    add_settings_arguments(parser, inherit=True)
    group = parser.add_argument_group("baselines", "The bands tuned on the validation bank.")
    group.add_argument(
        "--band_widths",
        default=list(BAND_WIDTHS),
        type=float,
        nargs="+",
        help="Half-widths tried for the fixed band, in shares (0 is the delta hedge).",
    )
    group.add_argument(
        "--risk_aversions",
        default=list(RISK_AVERSIONS),
        type=float,
        nargs="+",
        help="Risk aversions tried for the Whalley-Wilmott band, in 1/dollars.",
    )
    group.add_argument(
        "--band_target",
        default="edge",
        choices=BAND_TARGETS,
        help="Where both bands trade to when the position leaves the band: its edge, or delta.",
    )
    parser.add_argument(
        "--batch_size", default=BATCH_SIZE, type=int, help="Scenarios hedged at a time (speed and memory only)."
    )


def eval_config(args, trained: HedgeTrainConfig | None) -> EvalConfig:
    """The `EvalConfig` of the parsed flags, with the training settings of `trained` where a flag is None.

    Args:
        args: the parsed flags.
        trained: the network's training config, or None for the library defaults.

    Raises:
        ValueError: as `EvalConfig`, `HedgeSettings` and `RiskMeasure`, for an invalid value.
    """
    hedging, risk = resolve_settings(args, *(() if trained is None else (trained.hedging, trained.risk)))
    return EvalConfig(
        hedging=hedging,
        risk=risk,
        band_widths=tuple(args.band_widths),
        risk_aversions=tuple(args.risk_aversions),
        band_target=args.band_target,
        batch_size=args.batch_size,
    )


def tuning_notes(tuned: dict, risk: RiskMeasure) -> list[str]:
    """One line per tuned band: its value, the measure it was tuned on, and a warning when the grid cannot tell
    where the best value is.

    A best value on an edge of the grid may lie beyond it, except a half-width of 0, the delta hedge, below which
    there is nothing. Scores that all tie (e.g. Whalley-Wilmott at zero cost, whose band is 0 for every `γ`) make
    the choice meaningless. A best score that ties only some other values is kept at its first value in the grid
    (`baselines.tune`), so the line says how many it ties.

    Args:
        tuned: `EvalReport.tuned`.
        risk: the measure the bands were tuned on (`EvalConfig.risk`).
    """
    lines = []
    for name, (what, unit, flag) in TUNED.items():
        entry = tuned[name]
        best, grid, scores = entry["best"], entry["grid"], entry["scores"]
        line = f"{name}: {what} {best:g} {unit}, the best of {len(grid)} by validation {describe_measure(risk)}"
        ties = sum(score == scores[grid.index(best)] for score in scores) - 1
        if len(grid) > 1 and ties == len(grid) - 1:
            line += ". Every value scores the same, so the choice means nothing."
        else:
            if ties:
                line += f". It ties with {ties} other value(s), and the first in the grid is kept"
            if len(grid) > 1 and (best == max(grid) or (best == min(grid) and not (name == BAND and best == 0))):
                line += f". It is on the edge of the grid: widen {flag}"
            line += "."
        lines.append(line)
    return lines


def format_report(report: EvalReport, risk: RiskMeasure) -> str:
    """The metrics as a markdown table, one row per hedger, the means in cents and the rest in dollars."""
    lines = [
        f"| Hedger | Mean P&L (cents) | SE (cents) | Std P&L ($) | CVaR {risk.alpha:.0%} ($) | Entropic, lam {risk.lam:g} ($) "
        "| Costs ($) | Trades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report.metrics.items():
        lines.append(
            f"| {name} | {100 * m['mean_pnl']:+.2f} | {100 * m['se_mean_pnl']:.2f} | {m['std_pnl']:.3f} | "
            f"{m['cvar']:+.3f} | {m['entropic']:.3f} | {m['mean_costs']:.3f} | {m['mean_trades']:.1f} |"
        )
    return "\n".join(lines)


def run(args):
    # before any work: a missing file or a bad flag fails here, not after the bands are tuned
    check_banks({"--test_path": (args.test_path, "test"), "--valid_path": (args.valid_path, "val")})
    trained, checkpoint = None, None
    if not args.baselines_only:
        if not Path(args.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"--checkpoint_path: no file at {args.checkpoint_path} (train the network with "
                "`python main.py train-hedge`, or pass --baselines_only)."
            )
        checkpoint = args.checkpoint_path
        trained = HedgeTrainConfig.from_dict(BandNetwork.load(checkpoint)[1]["config"])
    config = eval_config(args, trained)
    print(f"Testing on {describe_settings(config.hedging, config.risk)}")
    if trained is not None and (trained.hedging, trained.risk) != (config.hedging, config.risk):
        print(f"  the network was trained on {describe_settings(trained.hedging, trained.risk)}")

    report = evaluate_hedgers(args.test_path, args.valid_path, checkpoint, config)
    print(format_report(report, config.risk))
    for line in tuning_notes(report.tuned, config.risk):
        print(line)
    if args.report_path is not None:
        Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
        settings = {"hedging": asdict(config.hedging), "risk": asdict(config.risk), "checkpoint": checkpoint}
        with open(args.report_path, "w", encoding="utf-8") as file:
            json.dump({"settings": settings, "metrics": report.metrics, "tuned": report.tuned}, file, indent=2)
        print(f"Report was saved as : {args.report_path}")
