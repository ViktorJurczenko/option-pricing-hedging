"""
generate_dataset -> train-ml -> test-ml on tiny inputs, in a temp dir, so a
shape change (feature count, checkpoint format) between the three stages is
caught without the ~1h default-size run.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from AmericanOptionPricing.ML import MLP

ROOT = Path(__file__).resolve().parents[2]


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          timeout=timeout, cwd=ROOT)

def _generate(out_dir, *extra):
    result = _run("-m", "datasets.generate_dataset", "--path_per_sim", "500", "--nb_param_comb", "40",
                  "--samples_per_param", "2", "--nb_jobs", "2", "--sampling", "20",
                  "--output_dir", str(out_dir), *extra)
    assert result.returncode == 0, result.stderr
    return np.load(out_dir / "american_option_dataset.npz")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("ml5")
    data = _generate(out)
    ckpt = out / "model.pth"
    train = _run("main.py", "train-ml", "--dataset_path", str(out / "american_option_dataset.npz"),
                 "--epochs", "2", "--batch_size", "16", "--checkpoint_path", str(ckpt))
    return out, data, ckpt, train


def test_generate_dataset_default_writes_five_columns_with_calls_and_puts(pipeline):
    _, data, _, _ = pipeline
    features = data["features"]
    assert features.shape == (160, 5)
    assert features[:, 4].sum() == 80  # half calls
    assert set(np.unique(features[:, 4])) == {0.0, 1.0}
    assert np.isfinite(data["label"]).all()

def test_generate_dataset_single_type_still_writes_five_columns(tmp_path):
    features = _generate(tmp_path, "--option_type", "put")["features"]
    assert features.shape == (80, 5)
    assert (features[:, 4] == 0.0).all()

def test_train_ml_saves_a_checkpoint_that_loads_into_the_mlp(pipeline):
    _, _, ckpt, train = pipeline
    assert train.returncode == 0, train.stderr
    MLP().load_state_dict(torch.load(ckpt, map_location="cpu"))  # strict: shapes must match

def test_test_ml_reports_put_and_call_blocks(pipeline):
    out, _, ckpt, train = pipeline
    assert train.returncode == 0, train.stderr
    result = _run("main.py", "test-ml", "--dataset_path", str(out / "american_option_dataset.npz"),
                  "--features_norm_path", str(out / "features_norm.npz"), "--checkpoint_path", str(ckpt))
    assert result.returncode == 0, result.stderr
    assert "MSE" in result.stdout
    assert result.stdout.index("Put:") < result.stdout.index("Call:")
    assert result.stdout.count("ITM avg error") == 2
