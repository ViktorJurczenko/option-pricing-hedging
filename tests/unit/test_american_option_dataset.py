import numpy as np
import torch

from datasets.american_option_dataset import AmericanOptionDataset

def _write_toy_dataset(path, n=100, n_features=4):
    features = np.random.default_rng(0).random((n, n_features)).astype(np.float32)
    labels = np.random.default_rng(1).random(n).astype(np.float32)
    np.savez(path, features=features, label=labels)
    return features, labels

def test_train_test_split_is_disjoint_and_complete(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    _write_toy_dataset(path, n=100)

    train_ds = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.8, seed=42)
    test_ds = AmericanOptionDataset(str(path), is_train=False, split_ratio=0.8, seed=42)

    assert len(train_ds) == 80
    assert len(train_ds) + len(test_ds) == 100

    train_rows = {tuple(row) for row in train_ds.features}
    test_rows = {tuple(row) for row in test_ds.features}
    assert train_rows.isdisjoint(test_rows)

def test_same_seed_gives_same_split(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    _write_toy_dataset(path, n=50)

    ds1 = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.8, seed=7)
    ds2 = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.8, seed=7)

    np.testing.assert_array_equal(ds1.features, ds2.features)

def test_getitem_returns_float32_tensors(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    _write_toy_dataset(path, n=10)

    ds = AmericanOptionDataset(str(path), is_train=True, split_ratio=1.0, seed=0)
    features, label = ds[0]

    assert isinstance(features, torch.Tensor) and features.dtype == torch.float32
    assert isinstance(label, torch.Tensor) and label.dtype == torch.float32
    assert features.shape == (4,)
