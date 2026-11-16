"""Tests for optionpricing.ml.hedging.hedge_evaluation: the network and the baselines scored on one test bank."""

import io
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from tqdm import tqdm

from optionpricing.hedging.baselines import band_hedge, delta_hedge, no_hedge, whalley_wilmott
from optionpricing.hedging.pnl import HedgeOutcome, HedgeSettings
from optionpricing.hedging.risk import RiskMeasure, cvar, entropic, variance
from optionpricing.ml.hedging.hedge_dataset import HedgeDataset, hedge_bank
from optionpricing.ml.hedging.hedge_evaluation import (
    BAND,
    DELTA,
    NETWORK,
    NO_HEDGE,
    WHALLEY_WILMOTT,
    EvalConfig,
    baseline_hedger,
    evaluate_hedgers,
    summarize,
    tune_baseline,
)
from optionpricing.ml.hedging.hedge_model import BandNetwork, network_hedger
from optionpricing.ml.hedging.hedge_training import HedgeTrainConfig, train_hedging_model
from tests.helpers import write_hedge_bank, write_put_surrogate

# Small grids keep a full evaluation to a dozen passes over 8-scenario banks.
GRIDS = dict(band_widths=(0.0, 0.1, 0.4), risk_aversions=(0.1, 1.0), batch_size=4)
TRAIN = HedgeTrainConfig(depth=1, width=8, epochs=2, batch_size=4, lr=1e-2, hedging=HedgeSettings(proportional=0.01))


def config(proportional=0.0, objective="cvar", **overrides):
    return EvalConfig(
        **{**GRIDS, **overrides}, hedging=HedgeSettings(proportional=proportional), risk=RiskMeasure(objective)
    )


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """Paths of a train, a validation and a test bank, generated with the same settings and their split's seed."""
    out = tmp_path_factory.mktemp("banks")
    surrogate = write_put_surrogate(out / "surrogate.pth")
    return {split: write_hedge_bank(out, surrogate, "--split", split) for split in ("train", "val", "test")}


@pytest.fixture(scope="module")
def checkpoint(banks, tmp_path_factory):
    """A band network trained on the train bank at a 1% cost, saved by train_hedging_model."""
    path = tmp_path_factory.mktemp("policy") / "policy.pth"
    train_hedging_model(banks["train"], banks["val"], TRAIN, path)
    return path


def bank(path):
    return HedgeDataset(path, dtype=torch.float64)


def book(policy, dataset, cfg):
    return hedge_bank(baseline_hedger(policy), dataset, cfg.hedging, cfg.batch_size)


# the baselines' features


def test_no_hedge_pnl_is_the_premium_minus_the_discounted_payoff(banks):
    # An independent reference from the raw bank: with no shares and no costs, the hedger keeps the premium and
    # pays K - S at tau, discounted, if the put is in the money.
    test = bank(banks["test"])
    batch = test[torch.arange(len(test))]
    spot_tau = batch["spot"][torch.arange(len(test)), batch["tau"]]
    years = batch["tau"] / test.settings["days_per_year"]
    expected = batch["premium"] - torch.exp(-batch["rate"] * years) * (test.strike - spot_tau).clamp_min(0)
    outcome = book(no_hedge, test, config())
    torch.testing.assert_close(outcome.pnl, expected, rtol=0, atol=1e-9)
    assert torch.all(outcome.costs == 0) and torch.all(outcome.trades == 0)


def test_a_band_wider_than_any_delta_never_trades(banks):
    # A put's delta lies in [-1, 0], so a band of half-width 2 always contains the starting 0 shares: the hedger
    # holds nothing and books exactly the no-hedge P&L. A swapped delta or gamma row would break this.
    test = bank(banks["test"])
    cfg = config(proportional=0.01)
    wide, none = book(band_hedge(2.0), test, cfg), book(no_hedge, test, cfg)
    assert torch.all(wide.trades == 0)
    torch.testing.assert_close(wide.pnl, none.pnl, rtol=0, atol=0)


# summarize


def test_summarize_reports_the_risk_measures_of_risk_py():
    pnl = torch.tensor([-3.0, -1.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    # positions giving 1, 2, 0, 3 and 4 trades
    positions = torch.tensor(
        [[1, 1, 1, 1], [1, 2, 2, 2], [0, 0, 0, 0], [1, 2, 3, 3], [1, 2, 3, 4]], dtype=torch.float64
    )
    outcome = HedgeOutcome(pnl, torch.tensor([0.1, 0.2, 0.0, 0.3, 0.4], dtype=torch.float64), positions)
    metrics = summarize(outcome, RiskMeasure("cvar", alpha=0.6, lam=0.5))
    assert metrics["mean_pnl"] == pytest.approx(-0.1)
    std = math.sqrt(float(variance(pnl)))
    assert metrics["std_pnl"] == pytest.approx(std)
    assert metrics["se_mean_pnl"] == pytest.approx(std / math.sqrt(5))
    assert metrics["cvar"] == pytest.approx(-2.0)  # the worst 40% of 5 paths: -3 and -1
    assert metrics["cvar"] == pytest.approx(float(cvar(pnl, 0.6)))
    assert metrics["entropic"] == pytest.approx(float(entropic(pnl, 0.5)))
    assert metrics["mean_costs"] == pytest.approx(0.2)
    assert metrics["mean_trades"] == pytest.approx(2.0)


# tune_baseline


@pytest.mark.parametrize("objective", ["cvar", "variance"])
def test_tune_baseline_picks_the_width_that_scores_best_on_validation(banks, objective):
    valid = bank(banks["val"])
    cfg = config(proportional=0.01, objective=objective)
    bar = tqdm(total=len(cfg.band_widths), file=io.StringIO())  # a disabled bar would not count
    best, scores = tune_baseline(band_hedge, cfg.band_widths, valid, cfg, bar)
    # The reference scores each width directly: CVaR on the P&L scale, the variance negated (higher is better).
    expected = []
    for width in cfg.band_widths:
        pnl = book(band_hedge(width), valid, cfg).pnl
        expected.append(float(cvar(pnl, 0.95)) if objective == "cvar" else -float(variance(pnl)))
    assert scores == pytest.approx(expected)
    assert best == cfg.band_widths[int(np.argmax(expected))]
    assert bar.n == len(cfg.band_widths)


# evaluate_hedgers


def test_evaluate_hedgers_scores_every_baseline_on_the_test_bank(banks):
    cfg = config(proportional=0.01)
    report = evaluate_hedgers(banks["test"], banks["val"], config=cfg)
    assert list(report.metrics) == [NO_HEDGE, DELTA, BAND, WHALLEY_WILMOTT]
    assert list(report.pnl) == list(report.metrics)
    test = bank(banks["test"])
    # Each baseline is scored with the tuned parameter, on the test bank, as hedge_bank books it.
    policies = {
        NO_HEDGE: no_hedge,
        DELTA: delta_hedge,
        BAND: band_hedge(report.tuned[BAND]["best"]),
        WHALLEY_WILMOTT: whalley_wilmott(report.tuned[WHALLEY_WILMOTT]["best"], 0.01),
    }
    for name, policy in policies.items():
        outcome = book(policy, test, cfg)
        assert report.metrics[name] == pytest.approx(summarize(outcome, cfg.risk)), name
        torch.testing.assert_close(report.pnl[name], outcome.pnl, rtol=0, atol=0)


def test_the_bands_are_tuned_on_validation_not_on_test(banks):
    cfg = config(proportional=0.01)
    report = evaluate_hedgers(banks["test"], banks["val"], config=cfg)
    expected, scores = tune_baseline(band_hedge, cfg.band_widths, bank(banks["val"]), cfg)
    assert report.tuned[BAND] == {"best": expected, "grid": list(cfg.band_widths), "scores": scores}
    assert report.tuned[WHALLEY_WILMOTT]["grid"] == list(cfg.risk_aversions)


def test_the_test_costs_need_not_be_the_training_costs(banks, checkpoint):
    # The network was trained at a 1% cost; at zero cost every hedger, the network included, pays nothing.
    report = evaluate_hedgers(banks["test"], banks["val"], checkpoint, config())
    assert all(metrics["mean_costs"] == 0 for metrics in report.metrics.values())
    priced = evaluate_hedgers(banks["test"], banks["val"], checkpoint, config(proportional=0.01))
    assert priced.metrics[NETWORK]["mean_costs"] > 0


def test_the_network_is_scored_as_its_checkpoint_hedges(banks, checkpoint):
    cfg = config(proportional=0.01)
    report = evaluate_hedgers(banks["test"], banks["val"], checkpoint, cfg)
    assert list(report.metrics)[-1] == NETWORK
    model, _ = BandNetwork.load(checkpoint)
    outcome = hedge_bank(network_hedger(model), bank(banks["test"]), cfg.hedging, cfg.batch_size)
    assert report.metrics[NETWORK] == pytest.approx(summarize(outcome, cfg.risk))


def test_a_network_trained_with_the_boundary_input_reads_it_at_test(banks, tmp_path):
    # The stored config says the network reads 7 features; ignoring it would feed 6 and raise.
    path = tmp_path / "policy.pth"
    train_hedging_model(banks["train"], banks["val"], replace(TRAIN, boundary_input=True), path)
    report = evaluate_hedgers(banks["test"], banks["val"], path, config())
    assert math.isfinite(report.metrics[NETWORK]["cvar"])


def test_a_network_is_never_tested_on_its_training_paths(banks, checkpoint):
    with pytest.raises(ValueError, match="share their paths"):
        evaluate_hedgers(banks["train"], banks["val"], checkpoint, config())


def test_banks_that_share_their_paths_are_refused(banks):
    with pytest.raises(ValueError, match="share their paths"):
        evaluate_hedgers(banks["test"], banks["test"], config=config())


# EvalConfig


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"band_target": "middle"}, "band_target must be one of"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"band_widths": ()}, "at least one value"),
        ({"risk_aversions": ()}, "at least one value"),
        ({"band_widths": (0.1, -0.1)}, "half_width must be finite and >= 0"),
        ({"risk_aversions": (0.0,)}, "risk_aversion must be finite and > 0"),
    ],
)
def test_config_rejects_invalid_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        EvalConfig(**overrides)
