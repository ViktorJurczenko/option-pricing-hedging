"""Up-front file validation of the train-ml and test-ml CLI entry points (scripts/train_ml.py, scripts/test_ml.py)."""
import argparse

import numpy as np
import pytest

from scripts import test_ml, train_ml


def _touch(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"")  # not a valid npz/checkpoint: any work done on it would fail differently
    return str(path)


def test_train_ml_missing_dataset_raises_file_not_found_naming_the_path(tmp_path):
    missing = str(tmp_path / "nope.npz")
    args = argparse.Namespace(dataset_path=missing)
    with pytest.raises(FileNotFoundError, match="nope.npz"):
        train_ml.run(args)

def test_train_ml_checks_the_dataset_is_a_file_not_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_ml.run(argparse.Namespace(dataset_path=str(tmp_path)))

@pytest.mark.parametrize("missing_flag", ["dataset_path", "features_norm_path", "checkpoint_path"])
def test_test_ml_missing_file_raises_file_not_found_naming_the_flag(tmp_path, missing_flag):
    paths = {flag: _touch(tmp_path, flag) for flag in ("dataset_path", "features_norm_path", "checkpoint_path")}
    paths[missing_flag] = str(tmp_path / "absent")
    args = argparse.Namespace(**paths, split_ratio=0.8, seed=0)

    # Raised before the (empty, unreadable) other files are touched, and names the flag.
    with pytest.raises(FileNotFoundError, match=f"--{missing_flag}"):
        test_ml.run(args)

def test_test_ml_no_longer_takes_lsm_flags():
    # The benchmark's LSM settings come from the dataset, so a flag would only let them drift.
    parser = argparse.ArgumentParser()
    test_ml.add_arguments(parser)
    defaults = vars(parser.parse_args([]))
    assert "reg_type" not in defaults and "max_degree" not in defaults

def test_load_lsm_settings_returns_what_the_dataset_stored(tmp_path):
    path = tmp_path / "ds.npz"
    np.savez(path, features=np.zeros((1, 5)), label=np.zeros(1),
             reg_type="laguerre", max_degree=4, path_per_sim=777, sampling=26)
    assert test_ml.load_lsm_settings(path) == {"reg_type": "laguerre", "max_degree": 4, "sampling": 26}

def test_load_lsm_settings_on_a_dataset_without_them_asks_to_regenerate(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, features=np.zeros((1, 5)), label=np.zeros(1))
    with pytest.raises(ValueError, match="regenerate"):
        test_ml.load_lsm_settings(path)
