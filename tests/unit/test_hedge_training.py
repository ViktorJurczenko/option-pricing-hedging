"""Tests for optionpricing.ml.hedging.hedge_training: the Deep Hedging training loop and its validation checkpoint."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from optionpricing.hedging.pnl import HedgeSettings
from optionpricing.hedging.risk import RiskMeasure
from optionpricing.ml.hedging.hedge_dataset import HedgeDataset, hedge_bank
from optionpricing.ml.hedging.hedge_model import BandNetwork, network_hedger
from optionpricing.ml.hedging.hedge_training import HedgeTrainConfig, train_hedging_model
from tests.helpers import write_hedge_bank, write_put_surrogate

# Small and quick: 8 scenarios of 5 to 12 days, 4 per batch so an epoch has 2 steps. A 1% cost makes the band
# matter, so training has something to change.
QUICK = dict(depth=1, width=8, epochs=3, batch_size=4, lr=1e-2)


def config(proportional=0.01, objective="cvar", **overrides):
    return HedgeTrainConfig(
        **{**QUICK, **overrides}, hedging=HedgeSettings(proportional=proportional), risk=RiskMeasure(objective)
    )


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """Paths of a train bank and a validation bank generated with the same settings."""
    out = tmp_path_factory.mktemp("banks")
    surrogate = write_put_surrogate(out / "surrogate.pth")
    return write_hedge_bank(out, surrogate), write_hedge_bank(out, surrogate, "--split", "val")


def train(banks, checkpoint_path=None, **overrides):
    return train_hedging_model(*banks, config(**overrides), checkpoint_path)


def valid_score(model, dataset, cfg):
    """The validation risk of `model`, booked independently of the training loop's own validation."""
    outcome = hedge_bank(network_hedger(model, cfg.boundary_input), dataset, cfg.hedging, cfg.batch_size)
    return float(cfg.risk.loss(outcome.pnl))


# training


def test_a_run_is_reproduced_from_its_seed(banks):
    first, second = train(banks), train(banks)
    assert first.history == second.history
    for a, b in zip(first.model.state_dict().values(), second.model.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_epoch_zero_scores_the_untrained_network_and_the_best_epoch_is_returned(banks):
    result = train(banks)
    cfg = config()
    valid = HedgeDataset(banks[1])
    torch.manual_seed(cfg.seed)  # the initialization train_hedging_model draws
    weights = BandNetwork(result.model.mean, result.model.std, cfg.depth, cfg.width, cfg.init_logit)
    assert result.history[0]["epoch"] == 0 and math.isnan(result.history[0]["train_loss"])
    assert result.history[0]["valid_score"] == pytest.approx(valid_score(weights, valid, cfg))
    scores = [h["valid_score"] for h in result.history]
    assert len(scores) == cfg.epochs + 1
    assert result.best_epoch == int(np.argmin(scores))
    assert valid_score(result.model, valid, cfg) == pytest.approx(min(scores))


def test_training_lowers_the_objective_at_a_cost(banks):
    # At 1% cost, the untrained band trades back to delta every day. Widening it saves costs, so the training
    # loss falls; at lr 1e-2 the band opens within a few steps (docs/results/hedging-results.md, "Training checks").
    result = train(banks, epochs=10)
    losses = [h["train_loss"] for h in result.history[1:]]
    assert losses[-1] < losses[0]


def test_a_run_that_never_improves_returns_the_untrained_network(banks):
    # With no cost, a band can only hedge worse than delta, and a tiny rate cannot move it: epoch 0 must win, so
    # the run returns the delta hedge rather than its last epoch.
    result = train(banks, proportional=0.0, lr=1e-9, w_lr=1e-9)
    assert result.best_epoch == 0
    torch.manual_seed(42)
    untrained = BandNetwork(result.model.mean, result.model.std, 1, 8)
    for a, b in zip(result.model.state_dict().values(), untrained.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_the_cvar_threshold_moves_on_its_own_learning_rate(banks):
    frozen = train(banks, lr=1e-9, w_lr=1e-9)
    moving = train(banks, lr=1e-9, w_lr=1.0)
    w_frozen = [h["w"] for h in frozen.history]
    w_moving = [h["w"] for h in moving.history]
    assert w_frozen[0] == w_moving[0]  # both start at the untrained network's VaR on the first batch
    assert max(abs(w - w_frozen[0]) for w in w_frozen) < 1e-6  # ~1e-9 per step for 6 steps
    assert abs(w_moving[-1] - w_moving[0]) > 0.1  # Adam moves w by about w_lr per step


@pytest.mark.parametrize("objective", ["entropic", "variance"])
def test_other_objectives_train_without_a_threshold(banks, objective):
    result = train(banks, objective=objective, epochs=1)
    assert all(math.isnan(h["w"]) for h in result.history)
    assert math.isfinite(result.history[1]["train_loss"])


def test_the_configs_activation_builds_the_network(banks):
    model = train(banks, activation="relu", epochs=1).model
    assert model.hparams["activation"] == "relu"
    assert isinstance(model.net[1], torch.nn.ReLU)


def test_the_boundary_input_adds_a_feature(banks):
    assert train(banks, boundary_input=True, epochs=1).model.n_features == 7
    assert train(banks, epochs=1).model.n_features == 6


def test_the_checkpoint_holds_the_best_network_its_config_and_the_training_banks_settings(banks, tmp_path):
    result = train(banks, checkpoint_path=tmp_path / "policy.pth")
    loaded, extra = BandNetwork.load(tmp_path / "policy.pth")
    cfg = config()
    valid = HedgeDataset(banks[1])
    assert valid_score(loaded, valid, cfg) == pytest.approx(result.history[result.best_epoch]["valid_score"])
    assert HedgeTrainConfig.from_dict(extra["config"]) == cfg
    stored, settings = extra["bank_settings"], HedgeDataset(banks[0]).settings
    assert math.isnan(stored.pop("shift")) and math.isnan(settings.pop("shift"))  # no shift is stored as NaN
    assert stored == settings


def test_a_batch_with_no_hedge_date_is_skipped(banks, tmp_path):
    # A scenario exercised at once has no hedge date, so a batch of it alone has no gradient. The loop must step
    # over it rather than call backward on a loss without a graph.
    raw = dict(np.load(banks[0]))
    raw["tau"][0] = 0
    path = tmp_path / "hedge_train.npz"
    np.savez(path, **raw)
    result = train_hedging_model(path, banks[1], config(batch_size=1, epochs=1))
    assert math.isfinite(result.history[1]["train_loss"])


def test_a_validation_score_that_is_never_finite_is_refused(banks, tmp_path):
    # A NaN premium makes every validation P&L NaN, so no epoch can be chosen: the run must say it diverged
    # rather than return a network picked on NaN scores.
    raw = dict(np.load(banks[1]))
    raw["premium"][:] = np.nan
    path = tmp_path / "hedge_val.npz"
    np.savez(path, **raw)
    with pytest.raises(ValueError, match="never finite"):
        train_hedging_model(banks[0], path, config(epochs=1))


def test_training_refuses_banks_that_share_their_paths(banks):
    with pytest.raises(ValueError, match="share their paths"):
        train_hedging_model(banks[0], banks[0], config())


# HedgeTrainConfig.from_dict


def test_from_dict_rebuilds_a_config_with_its_nested_settings():
    cfg = HedgeTrainConfig(
        hedging=HedgeSettings(rebalance_every=2, fixed=0.05, kappa=0.01), risk=RiskMeasure("entropic", lam=0.5)
    )
    assert HedgeTrainConfig.from_dict(asdict(cfg)) == cfg


# HedgeTrainConfig


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"activation": "sigmoid"}, "activation must be one of"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"epochs": 0}, "epochs must be >= 1"),
        ({"depth": 0}, "depth must be >= 1"),
        ({"lr": 0.0}, "lr must be positive"),
        ({"w_lr": math.nan}, "w_lr must be positive"),
    ],
)
def test_config_rejects_invalid_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        HedgeTrainConfig(**overrides)
