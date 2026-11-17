"""The train-hedge and test-hedge CLI entry points (scripts/train_hedge.py, scripts/test_hedge.py): flags, up-front
validation, settings inherited from the checkpoint, and the report."""

import json
import math
from dataclasses import fields, replace

import pytest

from optionpricing.hedging.pnl import HedgeSettings
from optionpricing.hedging.risk import RiskMeasure
from optionpricing.ml.hedging.hedge_evaluation import BAND, NETWORK, WHALLEY_WILMOTT, EvalConfig
from optionpricing.ml.hedging.hedge_training import HedgeTrainConfig, TrainReport
from scripts import test_hedge, train_hedge
from tests.helpers import make_parser, write_hedge_bank, write_put_surrogate

# A tiny network and grids: these tests check the wiring, not the hedging.
TINY = ["--depth", "1", "--width", "8", "--epochs", "2", "--batch_size", "4", "--lr", "0.01"]
GRIDS = ["--band_widths", "0", "0.1", "--risk_aversions", "0.1", "1", "--batch_size", "4"]


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """Paths of a train, a validation and a test bank, generated with the same settings and their split's seed."""
    out = tmp_path_factory.mktemp("banks")
    surrogate = write_put_surrogate(out / "surrogate.pth")
    return {split: str(write_hedge_bank(out, surrogate, "--split", split)) for split in ("train", "val", "test")}


@pytest.fixture(scope="module")
def checkpoint(banks, tmp_path_factory):
    """A network trained by train-hedge at a 1% cost on CVaR at 90%, and its path."""
    path = str(tmp_path_factory.mktemp("policy") / "net.pth")
    argv = ["--train_path", banks["train"], "--valid_path", banks["val"], "--checkpoint_path", path, *TINY]
    train_hedge.run(make_parser(train_hedge).parse_args([*argv, "--proportional", "0.01", "--alpha", "0.9"]))
    return path


def hedge_test_args(banks, *argv):
    return make_parser(test_hedge).parse_args(["--test_path", banks["test"], "--valid_path", banks["val"], *argv])


def no_call(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be reached")

    return fail


# train-hedge

# every HedgeTrainConfig field, nested ones included, set away from its default as typed on the command line
NON_DEFAULT_FLAGS = {
    "depth": "2",
    "width": "16",
    "init_logit": "-3",
    "activation": "tanh",
    "epochs": "3",
    "batch_size": "64",
    "lr": "0.01",
    "w_lr": "0.1",
    "seed": "7",
}
NON_DEFAULT_CONFIG = HedgeTrainConfig(
    depth=2,
    width=16,
    init_logit=-3.0,
    activation="tanh",
    boundary_input=True,
    hedging=HedgeSettings(rebalance_every=5, settlement="physical", proportional=0.01, fixed=0.5, kappa=0.02),
    risk=RiskMeasure("entropic", alpha=0.9, lam=2.0),
    epochs=3,
    batch_size=64,
    lr=0.01,
    w_lr=0.1,
    seed=7,
)


def test_train_hedge_defaults_are_the_train_configs():
    # a default edited in one place only would make train-hedge train another run than HedgeTrainConfig() describes
    assert train_hedge.train_config(make_parser(train_hedge).parse_args([])) == HedgeTrainConfig()


def test_train_hedge_has_a_flag_for_every_config_field_that_reaches_the_config():
    for f in fields(HedgeTrainConfig):  # else the comparison below proves nothing for that field
        assert getattr(NON_DEFAULT_CONFIG, f.name) != getattr(HedgeTrainConfig(), f.name), f.name
    argv = [token for name, value in NON_DEFAULT_FLAGS.items() for token in (f"--{name}", value)]
    argv += ["--boundary_input", "--rebalance_every", "5", "--settlement", "physical", "--proportional", "0.01"]
    argv += ["--fixed", "0.5", "--kappa", "0.02", "--risk", "entropic", "--alpha", "0.9", "--lam", "2"]
    assert train_hedge.train_config(make_parser(train_hedge).parse_args(argv)) == NON_DEFAULT_CONFIG


def test_train_hedge_saves_where_test_hedge_loads_and_validates_on_its_tuning_bank():
    train_args, test_args = make_parser(train_hedge).parse_args([]), make_parser(test_hedge).parse_args([])
    for name in ("checkpoint_path", "valid_path"):
        assert getattr(train_args, name) == getattr(test_args, name), name


def test_train_hedge_rejects_a_bad_setting_before_reading_the_banks(tmp_path, monkeypatch):
    monkeypatch.setattr(train_hedge, "train_hedging_model", no_call("train_hedging_model"))
    # the banks do not exist either: the setting must be refused first
    args = make_parser(train_hedge).parse_args(["--train_path", str(tmp_path / "x.npz"), "--fixed", "1"])
    with pytest.raises(ValueError, match="kappa"):
        train_hedge.run(args)


@pytest.mark.parametrize(("flag", "split"), [("--train_path", "train"), ("--valid_path", "val")])
def test_train_hedge_names_a_missing_bank_and_how_to_generate_it(banks, tmp_path, monkeypatch, flag, split):
    monkeypatch.setattr(train_hedge, "train_hedging_model", no_call("train_hedging_model"))
    paths = {"--train_path": banks["train"], "--valid_path": banks["val"], flag: str(tmp_path / "missing.npz")}
    args = make_parser(train_hedge).parse_args([token for item in paths.items() for token in item])
    with pytest.raises(FileNotFoundError, match=rf"{flag}: no file at .*missing.npz .*--split {split}"):
        train_hedge.run(args)


def test_train_hedge_reports_the_best_epoch_and_writes_the_history(banks, tmp_path, monkeypatch, capsys):
    calls = []

    def fake_train(train_path, valid_path, config, checkpoint_path):
        calls.append((train_path, valid_path, config, checkpoint_path))
        entry = lambda epoch, score: {
            "epoch": epoch,
            "train_loss": math.nan if epoch == 0 else 1.0,
            "valid_score": score,
            "valid_mean_pnl": 0.01 * epoch,
            "valid_mean_costs": 0.1,
            "w": 0.5,
        }
        return TrainReport(None, [entry(0, 0.9), entry(1, 0.7), entry(2, 0.8)], best_epoch=1)

    monkeypatch.setattr(train_hedge, "train_hedging_model", fake_train)
    ckpt, history = str(tmp_path / "sub" / "net.pth"), str(tmp_path / "h.json")
    argv = ["--train_path", banks["train"], "--valid_path", banks["val"], "--checkpoint_path", ckpt]
    train_hedge.run(make_parser(train_hedge).parse_args([*argv, "--history_path", history, "--epochs", "2"]))

    assert calls == [(banks["train"], banks["val"], HedgeTrainConfig(epochs=2), ckpt)]
    assert (tmp_path / "sub").is_dir()  # the checkpoint's folder is made before training saves into it
    out = capsys.readouterr().out
    # the score is RiskMeasure.loss, -CVaR: the best epoch's 0.7 is a CVaR of -0.7
    assert "Best epoch 1 of 2: validation CVaR 95% -0.7000 $" in out and "CVaR 95% -0.9000 $" in out
    assert "never beat" not in out and f"Model was saved as : {ckpt}" in out
    saved = json.load(open(history, encoding="utf-8"))
    assert saved["best_epoch"] == 1 and HedgeTrainConfig.from_dict(saved["config"]) == HedgeTrainConfig(epochs=2)
    assert [h["valid_score"] for h in saved["history"]] == [0.9, 0.7, 0.8]
    assert math.isnan(saved["history"][0]["train_loss"])


def test_train_hedge_says_when_training_never_beat_the_untrained_network(banks, tmp_path, monkeypatch, capsys):
    entry = {"epoch": 0, "train_loss": math.nan, "valid_score": 0.9, "valid_mean_pnl": 0.0, "valid_mean_costs": 0.1}
    monkeypatch.setattr(train_hedge, "train_hedging_model", lambda *a: TrainReport(None, [entry], best_epoch=0))
    argv = ["--train_path", banks["train"], "--valid_path", banks["val"], "--checkpoint_path", str(tmp_path / "n.pth")]
    train_hedge.run(make_parser(train_hedge).parse_args(argv))
    assert "Training never beat the untrained network" in capsys.readouterr().out


def test_train_hedge_saves_a_network_with_the_flags_config(checkpoint):
    from optionpricing.ml.hedging.hedge_model import BandNetwork

    model, extra = BandNetwork.load(checkpoint)
    config = HedgeTrainConfig.from_dict(extra["config"])
    assert (config.depth, config.width, config.epochs) == (1, 8, 2)
    assert config.hedging.proportional == 0.01 and config.risk.alpha == 0.9


# test-hedge


def test_test_hedge_without_a_network_uses_the_library_defaults(banks):
    config = test_hedge.eval_config(hedge_test_args(banks), None)
    assert config == replace(EvalConfig(), batch_size=config.batch_size)


def test_test_hedge_inherits_the_training_settings_a_flag_does_not_set(banks):
    trained = HedgeTrainConfig(
        hedging=HedgeSettings(proportional=0.01, fixed=0.5, kappa=0.02), risk=RiskMeasure(alpha=0.9)
    )
    config = test_hedge.eval_config(hedge_test_args(banks, "--proportional", "0.001"), trained)
    # the cost is the flag's, the fee and the risk level the training's
    assert config.hedging == HedgeSettings(proportional=0.001, fixed=0.5, kappa=0.02)
    assert config.risk == RiskMeasure(alpha=0.9)


def test_test_hedge_names_a_missing_checkpoint_and_the_way_around_it(banks, tmp_path, monkeypatch):
    monkeypatch.setattr(test_hedge, "evaluate_hedgers", no_call("evaluate_hedgers"))
    args = hedge_test_args(banks, "--checkpoint_path", str(tmp_path / "none.pth"))
    with pytest.raises(FileNotFoundError, match=r"--checkpoint_path: .*none.pth.*train-hedge.*--baselines_only"):
        test_hedge.run(args)


@pytest.mark.parametrize(("flag", "split"), [("--test_path", "test"), ("--valid_path", "val")])
def test_test_hedge_names_a_missing_bank_and_how_to_generate_it(banks, tmp_path, monkeypatch, flag, split):
    monkeypatch.setattr(test_hedge, "evaluate_hedgers", no_call("evaluate_hedgers"))
    args = hedge_test_args(banks, flag, str(tmp_path / "missing.npz"), "--baselines_only")
    with pytest.raises(FileNotFoundError, match=rf"{flag}: no file at .*--split {split}"):
        test_hedge.run(args)


def test_test_hedge_rejects_a_bad_grid_before_any_work(banks, monkeypatch):
    monkeypatch.setattr(test_hedge, "evaluate_hedgers", no_call("evaluate_hedgers"))
    with pytest.raises(ValueError, match="half_width"):
        test_hedge.run(hedge_test_args(banks, "--baselines_only", "--band_widths", "-0.1"))


def test_test_hedge_scores_the_network_at_its_training_settings(banks, checkpoint, tmp_path, capsys):
    report = tmp_path / "out" / "report.json"
    test_hedge.run(hedge_test_args(banks, "--checkpoint_path", checkpoint, "--report_path", str(report), *GRIDS))
    out = capsys.readouterr().out
    assert "Testing on hedge every 1 day(s), cash settlement, cost 1.00%, risk CVaR 90%" in out
    assert "trained on" not in out  # same settings as training: nothing to point out
    rows = [line for line in out.splitlines() if line.startswith("| ") and not line.startswith("| Hedger")]
    assert [row.split(" | ")[0][2:] for row in rows] == ["no hedge", "delta", BAND, WHALLEY_WILMOTT, NETWORK]

    saved = json.load(open(report, encoding="utf-8"))
    assert saved["settings"]["hedging"]["proportional"] == 0.01 and saved["settings"]["risk"]["alpha"] == 0.9
    assert saved["settings"]["checkpoint"] == checkpoint
    # the table prints the saved metrics: the network's CVaR, to the table's 3 decimals
    network_cvar = float(rows[-1].split(" | ")[4])
    assert network_cvar == pytest.approx(saved["metrics"][NETWORK]["cvar"], abs=5e-4)
    assert saved["tuned"][BAND]["grid"] == [0.0, 0.1]


def test_test_hedge_points_out_settings_that_differ_from_training(banks, checkpoint, capsys):
    test_hedge.run(hedge_test_args(banks, "--checkpoint_path", checkpoint, "--proportional", "0", *GRIDS))
    out = capsys.readouterr().out
    assert "cost 0.00%" in out and "the network was trained on hedge every 1 day(s), cash settlement, cost 1.00%" in out


def test_test_hedge_baselines_only_never_reads_a_checkpoint(banks, tmp_path, capsys):
    args = hedge_test_args(banks, "--baselines_only", "--checkpoint_path", str(tmp_path / "none.pth"), *GRIDS)
    test_hedge.run(args)
    out = capsys.readouterr().out
    assert "| delta |" in out and f"| {NETWORK} |" not in out


RISK = RiskMeasure(alpha=0.9)


def tuned(band_best, band_grid, band_scores, ww_best=1.0, ww_grid=(0.1, 1.0, 10.0), ww_scores=(1.0, 2.0, 1.0)):
    return {
        BAND: {"best": band_best, "grid": list(band_grid), "scores": list(band_scores)},
        WHALLEY_WILMOTT: {"best": ww_best, "grid": list(ww_grid), "scores": list(ww_scores)},
    }


def test_tuning_notes_warn_when_the_best_value_is_on_the_edge_of_the_grid():
    notes = test_hedge.tuning_notes(tuned(0.4, (0.0, 0.1, 0.4), (1, 2, 3), ww_best=0.1, ww_scores=(3, 2, 1)), RISK)
    assert "half-width 0.4 shares" in notes[0] and "widen --band_widths" in notes[0]
    assert "risk aversion 0.1 1/$" in notes[1] and "widen --risk_aversions" in notes[1]  # the lower edge too


def test_tuning_notes_accept_the_delta_hedge_at_a_zero_half_width():
    # nothing lies below a half-width of 0, so 0 at the lower edge is a real optimum
    notes = test_hedge.tuning_notes(tuned(0.0, (0.0, 0.1, 0.4), (3, 2, 1)), RISK)
    assert "widen" not in notes[0] and "widen" not in notes[1]


def test_tuning_notes_say_when_every_value_ties():
    # Whalley-Wilmott at zero cost: its band is 0 for every risk aversion, so the choice is the grid's first value
    notes = test_hedge.tuning_notes(tuned(0.0, (0.0, 0.1), (3, 2), ww_best=0.1, ww_scores=(2, 2, 2)), RISK)
    assert "means nothing" in notes[1] and "widen" not in notes[1]


def test_tuning_notes_do_not_warn_about_a_one_value_grid():
    notes = test_hedge.tuning_notes(tuned(0.1, (0.1,), (1,), ww_best=1.0, ww_grid=(1.0,), ww_scores=(1,)), RISK)
    assert all("widen" not in n and "means nothing" not in n for n in notes)


def test_tuning_notes_name_the_measure_the_bands_were_tuned_on():
    notes = test_hedge.tuning_notes(tuned(0.1, (0.0, 0.1, 0.4), (1, 3, 2)), RISK)
    assert notes[0] == "band: half-width 0.1 shares, the best of 3 by validation CVaR 90%."


def test_tuning_notes_say_how_many_values_the_best_score_ties():
    # tune keeps the first of tied values, so a tie hides how flat the score is around it
    notes = test_hedge.tuning_notes(tuned(0.1, (0.0, 0.1, 0.2, 0.4), (1, 3, 3, 2)), RISK)
    assert "ties with 1 other value(s)" in notes[0] and "widen" not in notes[0]
    assert "ties" not in notes[1]  # Whalley-Wilmott's best, 2.0, is unique
