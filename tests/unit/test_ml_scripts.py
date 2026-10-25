"""Up-front file validation of the train-ml and test-ml CLI entry points (scripts/train_ml.py, scripts/test_ml.py)."""
import argparse

import pytest

from ML.surrogate.training import TrainConfig
from ML.surrogate.tree_evaluation import TEACHER_METHOD, TEACHER_STEPS
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
