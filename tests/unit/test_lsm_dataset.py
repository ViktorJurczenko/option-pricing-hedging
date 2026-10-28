import numpy as np
import pytest
import torch

from optionpricing.ml.surrogate.legacy.lsm_dataset import AmericanOptionDataset

def _write_toy_dataset(path, n=100, n_features=5):
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
    assert features.shape == (5,)

def test_split_ratio_one_gives_empty_test_split_and_zero_gives_empty_train_split(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    _write_toy_dataset(path, n=10)

    full_train = AmericanOptionDataset(str(path), is_train=True, split_ratio=1.0)
    empty_test = AmericanOptionDataset(str(path), is_train=False, split_ratio=1.0)
    empty_train = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.0)
    full_test = AmericanOptionDataset(str(path), is_train=False, split_ratio=0.0)

    assert (len(full_train), len(empty_test), len(empty_train), len(full_test)) == (10, 0, 0, 10)
    with pytest.raises(IndexError):
        empty_test[0]

def test_split_sizes_truncate_toward_zero_and_the_split_is_a_permutation(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    features, labels = _write_toy_dataset(path, n=7)

    train = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.5, seed=3)
    test = AmericanOptionDataset(str(path), is_train=False, split_ratio=0.5, seed=3)

    assert (len(train), len(test)) == (3, 4)  # int(7 * 0.5) == 3
    got = np.concatenate([train.label, test.label])
    np.testing.assert_array_equal(np.sort(got), np.sort(labels))

def test_features_and_labels_stay_aligned_after_shuffling(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    features = np.arange(20, dtype=np.float32).reshape(20, 1) * np.ones((1, 5), dtype=np.float32)
    np.savez(path, features=features, label=features[:, 0] * 10)

    ds = AmericanOptionDataset(str(path), is_train=True, split_ratio=0.5, seed=1)

    for i in range(len(ds)):
        x, y = ds[i]
        assert y.item() == pytest.approx(x[0].item() * 10)

def test_transforms_are_applied_to_features_and_labels_separately(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    _write_toy_dataset(path, n=10)
    plain = AmericanOptionDataset(str(path), split_ratio=1.0, seed=0)
    ds = AmericanOptionDataset(str(path), transform=lambda f: f + 1.0, label_transform=lambda l: l * 3.0,
                               split_ratio=1.0, seed=0)

    (fp, lp), (ft, lt) = plain[2], ds[2]

    assert torch.allclose(ft, fp + 1.0)
    assert lt.item() == pytest.approx(lp.item() * 3.0)
