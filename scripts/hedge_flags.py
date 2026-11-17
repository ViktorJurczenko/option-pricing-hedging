"""The flags `train-hedge` and `test-hedge` share: how a hedge is booked (`HedgeSettings`) and the risk measure
(`RiskMeasure`), one flag per field.

`train-hedge` gives each flag its library default. `test-hedge` gives each the default None, which means "as the
network was trained", so a test is run at the training costs and on the training objective unless a flag says
otherwise. Without a network the library defaults apply.
"""

from dataclasses import fields, replace

from optionpricing.hedging.pnl import SETTLEMENTS, HedgeSettings
from optionpricing.hedging.risk import RISK_MEASURES, RiskMeasure

# The flags of the two settings, in `HedgeSettings` and `RiskMeasure` field order. The risk measure's `name` is
# `--risk` on the command line, the other flags are their field's name.
HEDGING_FLAGS = tuple(f.name for f in fields(HedgeSettings))
RISK_FLAGS = {"risk": "name", "alpha": "alpha", "lam": "lam"}


def add_settings_arguments(parser, inherit: bool) -> None:
    """Add one flag per field of `HedgeSettings` and `RiskMeasure` to `parser`.

    Args:
        parser: the command's parser.
        inherit: True gives every flag the default None, read by `resolve_settings` as "the network's training
            value". False gives the library defaults.
    """
    hedging, risk = HedgeSettings(), RiskMeasure()

    def default(value):
        return None if inherit else value

    def note(value) -> str:
        return f" Default: the network's training value, else {value}." if inherit else ""

    group = parser.add_argument_group("hedging", "How every hedge is booked (HedgeSettings).")
    group.add_argument(
        "--rebalance_every",
        default=default(hedging.rebalance_every),
        type=int,
        help="Days between hedge dates (1 is daily)." + note(hedging.rebalance_every),
    )
    group.add_argument(
        "--settlement",
        default=default(hedging.settlement),
        choices=sorted(SETTLEMENTS),
        help="How the exercised put is settled." + note(hedging.settlement),
    )
    group.add_argument(
        "--proportional",
        default=default(hedging.proportional),
        type=float,
        help="Cost rate on the dollars traded (0.01 is 1%%)." + note(hedging.proportional),
    )
    group.add_argument(
        "--fixed",
        default=default(hedging.fixed),
        type=float,
        help="Fee per trade, in dollars. Needs --kappa." + note(hedging.fixed),
    )
    group.add_argument(
        "--kappa",
        default=default(hedging.kappa),
        type=float,
        help="Width of the fee's straight-through gradient, in shares. It shapes the gradient only, so it changes "
        "no P&L." + note(hedging.kappa),
    )
    group = parser.add_argument_group("risk", "The risk measure (RiskMeasure).")
    group.add_argument(
        "--risk",
        default=default(risk.name),
        choices=RISK_MEASURES,
        help="The risk measure trained on, or the bands are tuned on." + note(risk.name),
    )
    group.add_argument(
        "--alpha", default=default(risk.alpha), type=float, help="CVaR level, in (0, 1)." + note(risk.alpha)
    )
    group.add_argument(
        "--lam",
        default=default(risk.lam),
        type=float,
        help="Entropic risk aversion, in 1/dollars." + note(risk.lam),
    )


def resolve_settings(
    args,
    base_hedging: HedgeSettings = HedgeSettings(),  # noqa: B008 (frozen dataclass, immutable)
    base_risk: RiskMeasure = RiskMeasure(),  # noqa: B008 (frozen dataclass, immutable)
) -> tuple[HedgeSettings, RiskMeasure]:
    """The settings of the parsed flags: each flag that is not None replaces its field of the base settings.

    The flags of `add_settings_arguments(inherit=False)` are read without a base: `--kappa` defaults to None
    there too, which would keep a base's `kappa`.

    Args:
        args: the parsed flags, from a parser with `add_settings_arguments`.
        base_hedging: the settings a None flag keeps, e.g. the network's training settings.
        base_risk: the same, for the risk measure.

    Returns:
        `(hedging, risk)`.

    Raises:
        ValueError: as `HedgeSettings` and `RiskMeasure`, if the result is invalid (e.g. `--fixed` without
            `--kappa`).
    """
    hedging = {name: getattr(args, name) for name in HEDGING_FLAGS if getattr(args, name) is not None}
    risk = {field: getattr(args, flag) for flag, field in RISK_FLAGS.items() if getattr(args, flag) is not None}
    return replace(base_hedging, **hedging), replace(base_risk, **risk)


def describe_measure(risk: RiskMeasure) -> str:
    """The risk measure with the level it reads: `CVaR 95%`, `entropic risk (lam 1/$)` or `variance`."""
    return {
        "cvar": f"CVaR {risk.alpha:.0%}",
        "entropic": f"entropic risk (lam {risk.lam:g}/$)",
        "variance": "variance",
    }[risk.name]


def describe_settings(hedging: HedgeSettings, risk: RiskMeasure) -> str:
    """The settings as one line, for a command's report."""
    fee = f", fee ${hedging.fixed:g} (kappa {hedging.kappa:g})" if hedging.fixed > 0 else ""
    return (
        f"hedge every {hedging.rebalance_every} day(s), {hedging.settlement} settlement, "
        f"cost {hedging.proportional:.2%}{fee}, risk {describe_measure(risk)}"
    )


def describe_score(risk: RiskMeasure, loss: float) -> str:
    """A risk measure's loss (`RiskMeasure.loss`) in its own units: CVaR on the P&L scale, the others as losses."""
    if risk.name == "cvar":
        return f"CVaR {risk.alpha:.0%} {-loss:+.4f} $"
    if risk.name == "entropic":
        return f"entropic risk {loss:.4f} $"
    return f"variance {loss:.4f} $^2"
