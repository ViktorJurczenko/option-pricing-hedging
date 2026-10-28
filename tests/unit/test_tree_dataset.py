import numpy as np
import pytest
import torch

import optionpricing.ml.surrogate.tree.generate_tree_dataset as gen
from optionpricing.ml.surrogate.tree.tree_dataset import FEATURES, FeatureStats, TreePutDataset, build_features
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_price
from tests.helpers import write_tree_dataset

# --- TreePutDataset and its features -----------------------------------------


@pytest.fixture(scope="module")
def tree_file(tmp_path_factory):
    return write_tree_dataset(tmp_path_factory.mktemp("tree"))


def _raw(ds):
    return build_features(*(ds.params[c] for c in ("m", "T", "r", "sigma")))


def test_build_features_match_their_definitions():
    m, T, r, sigma = np.array([1.1, 0.7]), np.array([0.25, 4.0]), np.array([0.03, 0.0]), np.array([0.2, 0.5])
    x = build_features(m, T, r, sigma).numpy()
    v = sigma * np.sqrt(T)  # 0.1 and 1.0
    expected = np.stack([np.log(m), np.log(T), np.log(sigma), r, v, np.log(m) / v], axis=-1)
    assert x.shape == (2, len(FEATURES))
    np.testing.assert_allclose(x, expected, rtol=1e-15)


def test_build_features_keep_the_autograd_graph_of_tensor_inputs():
    m = torch.tensor([0.9, 1.2], dtype=torch.float64, requires_grad=True)
    T, r, sigma = (torch.tensor(v, dtype=torch.float64) for v in ([0.5, 2.0], [0.01, 0.05], [0.3, 0.1]))
    x = build_features(m, T, r, sigma)
    (dz_dm,) = torch.autograd.grad(x[:, FEATURES.index("z")].sum(), m)
    # d/dm [log m / (sigma sqrt T)] = 1 / (m sigma sqrt T)
    torch.testing.assert_close(dz_dm, 1 / (m.detach() * sigma * torch.sqrt(T)))


def test_feature_stats_standardize_their_own_rows_and_only_center_a_constant_feature():
    raw = torch.tensor([[1.0, 5.0], [3.0, 5.0], [8.0, 5.0]], dtype=torch.float64)
    stats = FeatureStats.fit(raw)
    x = stats.apply(raw)
    torch.testing.assert_close(x[:, 0].mean(), torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(x[:, 0].std(unbiased=False), torch.tensor(1.0, dtype=torch.float64))
    # the constant column would be 0/0 = NaN if divided by its std
    assert stats.std[1] == 1 and torch.all(x[:, 1] == 0)


def test_feature_stats_refuse_an_empty_training_split():
    with pytest.raises(ValueError, match="empty"):
        FeatureStats.fit(torch.zeros((0, len(FEATURES)), dtype=torch.float64))


def test_feature_stats_round_trip_through_a_dict_and_refuse_other_features():
    stats = FeatureStats(torch.arange(6.0, dtype=torch.float64), torch.arange(1.0, 7.0, dtype=torch.float64))
    d = stats.to_dict()
    back = FeatureStats.from_dict(d)
    torch.testing.assert_close(back.mean, stats.mean)
    torch.testing.assert_close(back.std, stats.std)
    # a model trained with another feature list must not be fed these inputs
    with pytest.raises(ValueError, match="feature statistics"):
        FeatureStats.from_dict({**d, "names": ["log_m", "log_T", "log_sigma", "r", "total_vol", "is_call"]})


def test_tree_dataset_split_is_disjoint_complete_and_fixed_by_the_seed(tree_file):
    train = TreePutDataset(tree_file, is_train=True, split_ratio=0.75, seed=1)
    test = TreePutDataset(tree_file, is_train=False, split_ratio=0.75, seed=1)
    assert (len(train), len(test)) == (96, 32)
    all_m = np.sort(np.load(tree_file)["m"])
    np.testing.assert_array_equal(np.sort(np.concatenate([train.params["m"], test.params["m"]])), all_m)
    np.testing.assert_array_equal(TreePutDataset(tree_file, split_ratio=0.75, seed=1).params["m"], train.params["m"])
    assert not np.array_equal(TreePutDataset(tree_file, split_ratio=0.75, seed=2).params["m"], train.params["m"])


def test_both_splits_are_standardized_with_the_training_split_stats(tree_file):
    train = TreePutDataset(tree_file, is_train=True, split_ratio=0.75, seed=1)
    test = TreePutDataset(tree_file, is_train=False, split_ratio=0.75, seed=1)
    torch.testing.assert_close(test.stats.mean, train.stats.mean)
    torch.testing.assert_close(test.stats.std, train.stats.std)
    # the training features come out standardized (stats fitted on all rows
    # instead, a leak, would leave them off 0 and 1)
    torch.testing.assert_close(train.features.mean(dim=0), torch.zeros(len(FEATURES)), atol=1e-5, rtol=0)
    torch.testing.assert_close(train.features.std(dim=0, unbiased=False), torch.ones(len(FEATURES)), atol=1e-5, rtol=0)
    torch.testing.assert_close(test.features, train.stats.apply(_raw(test)).float())


def test_given_stats_replace_the_fitted_ones(tree_file):
    stats = FeatureStats(torch.full((6,), 0.5, dtype=torch.float64), torch.full((6,), 2.0, dtype=torch.float64))
    ds = TreePutDataset(tree_file, split_ratio=1.0, stats=stats)
    assert ds.stats is stats
    torch.testing.assert_close(ds.features, ((_raw(ds) - 0.5) / 2.0).float())


def test_targets_are_the_early_exercise_premium_and_its_delta(tree_file):
    ds = TreePutDataset(tree_file, split_ratio=0.5, seed=4)
    raw = np.load(tree_file)
    row = {m: i for i, m in enumerate(raw["m"])}  # Sobol draws: every m is distinct
    i = np.array([row[m] for m in ds.params["m"]])
    m, T, r, sigma = (raw[c][i] for c in ("m", "T", "r", "sigma"))
    np.testing.assert_allclose(
        ds.premium.numpy(), raw["price"][i] - black_scholes_price(m, 1.0, T, r, sigma, "put"), rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        ds.premium_dm.numpy(), raw["delta"][i] - black_scholes_delta(m, 1.0, T, r, sigma, "put"), rtol=1e-6, atol=1e-8
    )
    # an exercised put is worth 1 - m with delta -1: its premium delta is -N(d1)
    ex = ds.exercised
    assert ex.any() and not ex.all()
    np.testing.assert_allclose(
        ds.premium_dm.numpy()[ex], -black_scholes_delta(m, 1.0, T, r, sigma, "call")[ex], rtol=1e-6, atol=1e-8
    )


def test_features_dm_matches_a_finite_difference_of_the_features_in_m(tree_file):
    ds = TreePutDataset(tree_file, split_ratio=1.0)
    p = ds.params
    h = 1e-6 * p["m"]
    up = ds.stats.apply(build_features(p["m"] + h, p["T"], p["r"], p["sigma"]))
    down = ds.stats.apply(build_features(p["m"] - h, p["T"], p["r"], p["sigma"]))
    fd = ((up - down) / torch.tensor(2 * h)[:, None]).float()
    torch.testing.assert_close(ds.features_dm, fd, rtol=1e-4, atol=1e-6)
    # only log m and z move with m
    still = [j for j, name in enumerate(FEATURES) if name not in ("log_m", "z")]
    assert torch.all(ds.features_dm[:, still] == 0)


def test_tree_dataset_items_are_aligned_float32_rows(tree_file):
    ds = TreePutDataset(tree_file, split_ratio=0.5, seed=0)
    x, x_dm, premium, premium_dm = ds[3]
    assert x.shape == x_dm.shape == (len(FEATURES),) and premium.shape == premium_dm.shape == ()
    assert all(t.dtype == torch.float32 for t in (x, x_dm, premium, premium_dm))
    torch.testing.assert_close(x, ds.stats.apply(_raw(ds)[3]).float())
    # a DataLoader batch keeps the four columns together
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
    assert [t.shape[0] for t in batch] == [8] * 4


def test_tree_dataset_reads_the_sampling_domain_from_the_file(tree_file):
    expected = {k: tuple(v) for k, v in gen.DOMAIN.items() if k.endswith("_range")}
    assert TreePutDataset(tree_file, split_ratio=1.0).domain == expected


def test_build_features_broadcast_scalars_against_arrays():
    # a delta curve over m at one (T, r, sigma), as the price helper will be asked
    m = np.linspace(0.8, 1.2, 5)
    full = build_features(m, np.full(5, 0.5), np.full(5, 0.03), np.full(5, 0.2))
    torch.testing.assert_close(build_features(m, 0.5, 0.03, 0.2), full)
    assert build_features(1.0, 0.5, 0.03, 0.2).shape == (len(FEATURES),)


@pytest.mark.parametrize("split_ratio", [-0.1, 1.1])
def test_tree_dataset_rejects_a_split_ratio_outside_0_1(tree_file, split_ratio):
    # slicing used to accept them silently: -0.1 gave a 90/10 split
    with pytest.raises(ValueError, match="split_ratio"):
        TreePutDataset(tree_file, split_ratio=split_ratio)


def test_tree_dataset_names_a_missing_column(tree_file, tmp_path):
    path = tmp_path / "no_delta.npz"
    np.savez(path, **{k: v for k, v in np.load(tree_file).items() if k != "delta"})
    with pytest.raises(ValueError, match=r"missing \['delta'\]"):
        TreePutDataset(path)


def test_tree_dataset_rejects_an_lsm_dataset(tmp_path):
    path = tmp_path / "lsm.npz"
    np.savez(path, features=np.zeros((10, 5)), label=np.zeros(10))  # the legacy file's layout
    with pytest.raises(ValueError, match="not a tree put dataset"):
        TreePutDataset(path)


def test_tree_dataset_rejects_another_teacher(tree_file, tmp_path):
    path = tmp_path / "other.npz"
    np.savez(path, **{**np.load(tree_file), "teacher": "lsm"})
    with pytest.raises(ValueError, match="teacher is 'lsm'"):
        TreePutDataset(path)


def test_an_empty_training_split_needs_given_stats(tree_file):
    with pytest.raises(ValueError, match="empty"):
        TreePutDataset(tree_file, is_train=False, split_ratio=0.0)
    stats = TreePutDataset(tree_file, split_ratio=0.5).stats
    assert len(TreePutDataset(tree_file, is_train=False, split_ratio=0.0, stats=stats)) == 128
    empty = TreePutDataset(tree_file, is_train=False, split_ratio=1.0)
    assert len(empty) == 0 and empty.features.shape == (0, len(FEATURES))
