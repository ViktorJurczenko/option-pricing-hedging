"""
The legacy LSM pipeline (ML/surrogate/legacy/), generate_lsm_dataset ->
train_legacy_mlp, on tiny inputs in a temp dir, so a shape change (feature
count, checkpoint format) between the stages is caught without the ~1h
default-size run. The generator runs as `python -m`, as documented; no CLI
command trains or scores LegacyMLP any more (train-ml/test-ml run the tree
surrogate, tests/integration/test_cli_smoke.py).
"""
import numpy as np
import pytest
import torch

from ML.surrogate.legacy.lsm_model import LegacyMLP
from ML.surrogate.legacy.lsm_training import train_legacy_mlp
from tests.helpers import run_python

# Dataset generation and training take longer than the helpers' default timeout.
PIPELINE_TIMEOUT = 600

def _generate(out_dir, *extra):
    result = run_python("-m", "ML.surrogate.legacy.generate_lsm_dataset", "--path_per_sim", "500", "--nb_param_comb", "40",
                  "--samples_per_param", "2", "--nb_jobs", "2", "--sampling", "20",
                  "--output_dir", str(out_dir), *extra, timeout=PIPELINE_TIMEOUT)
    assert result.returncode == 0, result.stderr
    return np.load(out_dir / "american_option_dataset.npz")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("ml5")
    data = _generate(out)
    ckpt = out / "model.pth"
    torch.manual_seed(0)  # LegacyMLP's initialization and the batch shuffle
    mlp = train_legacy_mlp(out / "american_option_dataset.npz", epochs=2, batch_size=16, checkpoint_path=ckpt)
    return out, data, ckpt, mlp


def test_generate_lsm_dataset_default_writes_five_columns_with_calls_and_puts(pipeline):
    _, data, _, _ = pipeline
    features = data["features"]
    assert features.shape == (160, 5)
    assert features[:, 4].sum() == 80  # half calls
    assert set(np.unique(features[:, 4])) == {0.0, 1.0}
    assert np.isfinite(data["label"]).all()

def test_generate_lsm_dataset_stores_the_lsm_settings_it_labelled_with(pipeline, tmp_path):
    _, data, _, _ = pipeline
    assert (str(data["reg_type"]), int(data["max_degree"]), int(data["path_per_sim"]), int(data["sampling"])) \
        == ("simple", 3, 500, 20)
    custom = _generate(tmp_path, "--reg_type", "laguerre", "--max_degree", "2", "--option_type", "put")
    assert (str(custom["reg_type"]), int(custom["max_degree"])) == ("laguerre", 2)

def test_generate_lsm_dataset_is_seeded_chunked_and_cleans_up_its_chunks(pipeline, tmp_path):
    out, data, _, _ = pipeline
    assert not (out / "chunks").exists()  # removed once the dataset is assembled
    # Same seed, other chunking: the same dataset, features and labels (each sample has its own seed).
    again = _generate(tmp_path, "--chunk_size", "30")
    np.testing.assert_array_equal(again["features"], data["features"])
    np.testing.assert_array_equal(again["label"], data["label"])

def test_generate_lsm_dataset_single_type_still_writes_five_columns(tmp_path):
    features = _generate(tmp_path, "--option_type", "put")["features"]
    assert features.shape == (80, 5)
    assert (features[:, 4] == 0.0).all()

def test_train_legacy_mlp_saves_a_checkpoint_that_loads_into_the_legacy_mlp(pipeline):
    _, _, ckpt, mlp = pipeline
    assert isinstance(mlp, LegacyMLP)
    LegacyMLP().load_state_dict(torch.load(ckpt, map_location="cpu"))  # strict: shapes must match

def test_train_legacy_mlp_missing_dataset_names_the_generator(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_lsm_dataset"):
        train_legacy_mlp(tmp_path / "nope.npz", checkpoint_path=tmp_path / "m.pth")

def test_features_norm_keys_and_values_invert_the_generators_columns(pipeline):
    # CLAUDE.md requires generate_lsm_dataset and the dataset class to share one
    # normalization: sigma and T as z-scored logs, r as a raw z-score, (log(x) - mean) / std with
    # these exact keys. The generator z-scores over the same rows it
    # writes (each parameter set is repeated uniformly), so the stored columns must have mean 0 and
    # std 1, and inverting them with the stored stats must land inside the generator's
    # sampling ranges (sigma 0.05-0.6, T 0.25-5, r 0-0.08). A different log base, a missing log, or
    # swapped keys would break one of these.
    out, data, _, _ = pipeline
    norms = np.load(out / "features_norm.npz")
    assert set(norms.files) == {"log_sigma_mean", "log_sigma_std", "r_mean", "r_std", "log_T_mean", "log_T_std"}

    T_norm, r_norm, sigma_norm = (data["features"][:, i] for i in (1, 2, 3))
    for col in (T_norm, r_norm, sigma_norm):
        assert col.mean() == pytest.approx(0.0, abs=1e-9)
        assert col.std() == pytest.approx(1.0, abs=1e-9)

    sigma = np.exp(sigma_norm * norms["log_sigma_std"] + norms["log_sigma_mean"])
    T = np.exp(T_norm * norms["log_T_std"] + norms["log_T_mean"])
    r = r_norm * norms["r_std"] + norms["r_mean"]
    eps = 1e-9
    assert sigma.min() >= 0.05 - eps and sigma.max() <= 0.6 + eps
    assert T.min() >= 0.25 - eps and T.max() <= 5 + eps
    assert r.min() >= 0.0 - eps and r.max() <= 0.08 + eps

    # And the forward formula reproduces the stored column.
    np.testing.assert_allclose((np.log(sigma) - norms["log_sigma_mean"]) / norms["log_sigma_std"], sigma_norm, atol=1e-9)
    np.testing.assert_allclose((np.log(T) - norms["log_T_mean"]) / norms["log_T_std"], T_norm, atol=1e-9)
    np.testing.assert_allclose((r - norms["r_mean"]) / norms["r_std"], r_norm, atol=1e-9)
