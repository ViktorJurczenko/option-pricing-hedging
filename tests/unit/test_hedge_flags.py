"""The flags train-hedge and test-hedge share (scripts/hedge_flags.py): defaults, inheritance and the report lines."""

import argparse

import pytest

from optionpricing.hedging.pnl import HedgeSettings
from optionpricing.hedging.risk import RiskMeasure
from scripts.hedge_flags import (
    add_settings_arguments,
    describe_measure,
    describe_score,
    describe_settings,
    resolve_settings,
)

TRAINED_HEDGING = HedgeSettings(rebalance_every=5, settlement="physical", proportional=0.01, fixed=0.5, kappa=0.02)
TRAINED_RISK = RiskMeasure("entropic", alpha=0.9, lam=2.0)


def parse(inherit, *argv):
    parser = argparse.ArgumentParser()
    add_settings_arguments(parser, inherit)
    return parser.parse_args(list(argv))


def test_without_inheritance_the_defaults_are_the_librarys():
    assert resolve_settings(parse(False)) == (HedgeSettings(), RiskMeasure())


def test_with_inheritance_an_unset_flag_keeps_the_base_settings():
    assert resolve_settings(parse(True), TRAINED_HEDGING, TRAINED_RISK) == (TRAINED_HEDGING, TRAINED_RISK)


def test_with_inheritance_and_no_base_the_defaults_are_the_librarys():
    assert resolve_settings(parse(True)) == (HedgeSettings(), RiskMeasure())


def test_a_set_flag_replaces_only_its_own_field():
    hedging, risk = resolve_settings(
        parse(True, "--proportional", "0.001", "--risk", "cvar"), TRAINED_HEDGING, TRAINED_RISK
    )
    assert hedging == HedgeSettings(5, "physical", 0.001, 0.5, 0.02)
    assert risk == RiskMeasure("cvar", alpha=0.9, lam=2.0)  # --risk sets the measure's name


def test_every_field_has_a_flag_that_reaches_it():
    argv = (
        "--rebalance_every", "5", "--settlement", "physical", "--proportional", "0.01", "--fixed", "0.5",
        "--kappa", "0.02", "--risk", "entropic", "--alpha", "0.9", "--lam", "2",
    )  # fmt: skip
    # the inherit=False defaults differ from these in every field, so a flag that does not reach its field fails
    assert resolve_settings(parse(False, *argv)) == (TRAINED_HEDGING, TRAINED_RISK)


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (("--fixed", "1"), "kappa is needed"),
        (("--alpha", "1"), "alpha"),
        (("--rebalance_every", "0"), "rebalance_every"),
    ],
)
def test_an_invalid_combination_raises_the_settings_error(argv, match):
    with pytest.raises(ValueError, match=match):
        resolve_settings(parse(False, *argv))


def test_an_unknown_choice_is_a_usage_error():
    with pytest.raises(SystemExit):
        parse(False, "--risk", "sharpe")


def test_describe_settings_names_the_cost_fee_and_level():
    line = describe_settings(TRAINED_HEDGING, RiskMeasure("cvar", alpha=0.99))
    assert "every 5 day(s)" in line and "physical" in line and "1.00%" in line
    assert "fee $0.5 (kappa 0.02)" in line and "risk CVaR 99%" in line
    assert "fee" not in describe_settings(HedgeSettings(), RiskMeasure())  # no fee, none shown


def test_describe_score_puts_cvar_back_on_the_pnl_scale():
    # RiskMeasure.loss is -CVaR, so a loss of 0.93 is a CVaR of -0.93 dollars
    assert describe_score(RiskMeasure("cvar"), 0.93) == "CVaR 95% -0.9300 $"
    assert describe_score(RiskMeasure("entropic"), 0.5) == "entropic risk 0.5000 $"
    assert describe_score(RiskMeasure("variance"), 0.25) == "variance 0.2500 $^2"


def test_describe_measure_names_the_level_each_measure_reads():
    assert describe_measure(RiskMeasure("cvar", alpha=0.9, lam=3.0)) == "CVaR 90%"
    assert describe_measure(RiskMeasure("entropic", alpha=0.9, lam=3.0)) == "entropic risk (lam 3/$)"
    assert describe_measure(RiskMeasure("variance", alpha=0.9, lam=3.0)) == "variance"
