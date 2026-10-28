"""The train-ml and test-ml CLI entry points (scripts/train_ml.py, scripts/test_ml.py): flags, up-front validation."""
import argparse
from dataclasses import fields

import pytest

from optionpricing.ml.surrogate.tree.tree_training import TrainConfig, TrainResult
from optionpricing.ml.surrogate.tree.tree_evaluation import TEACHER_METHOD, TEACHER_STEPS
from scripts import test_ml, train_ml
from tests.helpers import make_parser, write_tree_dataset


def _touch(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"")  # not a valid npz/checkpoint: any work done on it would fail differently
    return str(path)

def _test_ml_args(**overrides):
    """test-ml's parser defaults plus overrides (parse_script_args adds simulation flags test-ml lacks)."""
    args = make_parser(test_ml).parse_args([])
    for name, value in overrides.items():
        assert hasattr(args, name), name
        setattr(args, name, value)
    return args


def test_train_ml_missing_dataset_raises_file_not_found_naming_the_path(tmp_path):
    missing = str(tmp_path / "nope.npz")
    args = argparse.Namespace(dataset_path=missing)
    with pytest.raises(FileNotFoundError, match="nope.npz"):
        train_ml.run(args)

def test_train_ml_checks_the_dataset_is_a_file_not_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_ml.run(argparse.Namespace(dataset_path=str(tmp_path)))

# every TrainConfig field set away from its default, as typed on the command line
NON_DEFAULT_FLAGS = {"depth": "2", "width": "16", "activation": "tanh", "output": "softplus", "split_ratio": "0.7",
                     "valid_ratio": "0.25", "batch_size": "64", "epochs": "3", "lr": "0.01", "lr_min": "0.001",
                     "patience": "2", "lbfgs_steps": "5", "lambda_delta": "0.5", "lambda_shape": "0.1",
                     "shape_points": "32", "seed": "7"}

def test_train_ml_defaults_are_train_configs():
    # a default edited in one place only would make train-ml train another run than TrainConfig() describes
    assert train_ml.train_config(make_parser(train_ml).parse_args([])) == TrainConfig()

def test_train_ml_has_a_flag_for_every_train_config_field_that_reaches_the_config():
    assert set(NON_DEFAULT_FLAGS) == {f.name for f in fields(TrainConfig)}  # a new field needs a flag (and a row here)
    argv = [token for name, value in NON_DEFAULT_FLAGS.items() for token in (f"--{name}", value)]
    config = train_ml.train_config(make_parser(train_ml).parse_args(argv))
    for f in fields(TrainConfig):
        value = getattr(config, f.name)
        assert value != getattr(TrainConfig(), f.name), f.name  # else the check below proves nothing
        assert value == type(value)(NON_DEFAULT_FLAGS[f.name]), f.name

def test_train_ml_saves_where_test_ml_loads_and_splits_like_it():
    train_args, test_args = make_parser(train_ml).parse_args([]), _test_ml_args()
    for name in ("dataset_path", "checkpoint_path", "split_ratio", "seed"):
        assert getattr(train_args, name) == getattr(test_args, name), name

def test_train_ml_rejects_a_bad_setting_before_training(tmp_path, monkeypatch):
    def no_training(*args, **kwargs):
        raise AssertionError("train_surrogate must not be reached")
    monkeypatch.setattr(train_ml, "train_surrogate", no_training)
    args = make_parser(train_ml).parse_args(["--dataset_path", _touch(tmp_path, "d.npz"), "--lr", "1e-3",
                                             "--lr_min", "1e-2"])
    with pytest.raises(ValueError, match="lr_min"):
        train_ml.run(args)

def test_train_ml_reports_the_checkpointed_entry_an_lbfgs_finish_included(tmp_path, monkeypatch, capsys):
    calls = []
    def fake_train(dataset_path, config, checkpoint_path, device):
        calls.append((dataset_path, config, checkpoint_path))
        entry = lambda epoch, loss: {"epoch": epoch, "valid_loss": loss, "valid_premium_term": 0.25 * loss,
                                     "valid_delta_term": 0.75 * loss, "valid_premium_rmse": 2 * loss,
                                     "valid_delta_rmse": 3 * loss}
        # epoch 2 is Adam's best, the kept L-BFGS finish beats it
        return TrainResult(None, [entry(1, 0.5), entry(2, 0.2), entry(3, 0.3), entry("lbfgs", 0.1)])
    monkeypatch.setattr(train_ml, "train_surrogate", fake_train)
    dataset, ckpt = _touch(tmp_path, "d.npz"), str(tmp_path / "s.pth")
    train_ml.run(make_parser(train_ml).parse_args(["--dataset_path", dataset, "--checkpoint_path", ckpt,
                                                   "--epochs", "3"]))
    assert calls == [(dataset, TrainConfig(epochs=3), ckpt)]
    out = capsys.readouterr().out
    assert "Best validation loss 1.0000e-01 (epoch lbfgs)" in out and "premium RMSE 2.000e-01" in out
    assert "premium term 2.5000e-02 + delta term 7.5000e-02" in out
    assert f"Model was saved as : {ckpt}" in out

@pytest.mark.parametrize("missing_flag", ["dataset_path", "checkpoint_path"])
def test_test_ml_missing_file_raises_file_not_found_naming_the_flag(tmp_path, missing_flag):
    paths = {flag: _touch(tmp_path, flag) for flag in ("dataset_path", "checkpoint_path")}
    paths[missing_flag] = str(tmp_path / "absent")
    args = _test_ml_args(**paths)

    # Raised before the (empty, unreadable) other files are touched, and names the flag.
    with pytest.raises(FileNotFoundError, match=f"--{missing_flag}"):
        test_ml.run(args)

def test_test_ml_refuses_a_reference_tree_with_the_teachers_settings_before_loading_the_model(tmp_path):
    dataset = write_tree_dataset(tmp_path)  # labelled by bbsr, 50 steps
    # the checkpoint is not readable: refusing must come first, before any work
    args = _test_ml_args(dataset_path=str(dataset), checkpoint_path=_touch(tmp_path, "ckpt.pth"),
                             ref_method="bbsr", ref_steps=50)
    with pytest.raises(ValueError, match="teacher"):
        test_ml.run(args)

def test_test_ml_defaults_match_the_training_runs_split():
    # the test split is only unseen if test-ml splits like train_surrogate (TrainConfig's defaults)
    args = _test_ml_args()
    assert (args.split_ratio, args.seed) == (TrainConfig().split_ratio, TrainConfig().seed)
    assert (args.ref_method, args.ref_steps) != (TEACHER_METHOD, TEACHER_STEPS)
