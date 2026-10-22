import numpy as np
import pytest

import ML.surrogate.GenerateTreeDataset as gen
from AmericanOptionPricing.BatchBinomialTree import binomial_put_batch
from EuropeanOptionPricing.BlackScholes import black_scholes_price
from ML.surrogate.GenerateTreeDataset import (COLUMNS, DOMAIN, SAMPLED, generate_tree_chunks, label_puts, load_tree_chunks,
                                              map_to_domain, sobol_points)
from ML.surrogate.generate_dataset import chunk_file

# small trees keep these fast; the labels' accuracy is the pricer's tests' job
TINY = ["--nb_steps", "50", "--nb_jobs", "1"]
N_DIMS = len(SAMPLED)


@pytest.mark.parametrize("n", [0, 3, 100, 1000])
def test_sobol_points_reject_a_count_that_is_not_a_power_of_two(n):
    with pytest.raises(ValueError, match="power of 2"):
        sobol_points(n, N_DIMS, seed=0)


def test_sobol_points_are_fixed_by_the_seed_and_independent_of_the_count():
    # a run's first 2^k rows must be the same points whatever the total, so a
    # subset for the learning curve is the same balanced design
    full = sobol_points(256, N_DIMS, seed=7)
    assert full.shape == (256, N_DIMS) and np.all((full >= 0) & (full < 1))
    np.testing.assert_array_equal(sobol_points(64, N_DIMS, seed=7), full[:64])
    assert not np.array_equal(sobol_points(64, N_DIMS, seed=8), full[:64])


def test_sobol_points_fill_each_dimension_evenly():
    # a scrambled Sobol net with 2^k points puts exactly one point in each of
    # the 2^k equal intervals of every coordinate; independent draws would not
    u = sobol_points(1024, N_DIMS, seed=3)
    for d in range(N_DIMS):
        np.testing.assert_array_equal(np.bincount((u[:, d] * 1024).astype(int), minlength=1024), 1)


def test_map_to_domain_keeps_every_row_inside_the_domain():
    p = map_to_domain(sobol_points(4096, N_DIMS, seed=1))
    for name, col in (("T", p["T"]), ("sigma", p["sigma"]), ("r", p["r"]), ("z", p["z"]), ("m", p["m"])):
        low, high = DOMAIN[f"{name}_range"]
        assert np.all((col >= low) & (col <= high)), name
    # absolute: ln m is ~0 near the money, and log(exp(x)) only rounds (~1e-16)
    np.testing.assert_allclose(np.log(p["m"]), p["z"] * p["sigma"] * np.sqrt(p["T"]), rtol=0, atol=1e-14)


@pytest.mark.parametrize("n_dims", [N_DIMS - 1, N_DIMS + 1])
def test_map_to_domain_rejects_uniforms_with_the_wrong_number_of_columns(n_dims):
    with pytest.raises(ValueError, match="shape"):
        map_to_domain(sobol_points(16, n_dims, seed=0))


def test_sobol_points_take_their_dimension_from_the_caller():
    assert sobol_points(16, 6, seed=0).shape == (16, 6)


def test_map_to_domain_maps_t_sigma_r_from_their_own_uniform_only():
    # The z position must not change T, sigma or r: rejecting rows outside
    # the m cap instead would have skewed their distributions.
    u = sobol_points(256, N_DIMS, seed=2)
    col = dict(zip(SAMPLED, u.T))
    shuffled = u.copy()
    z = SAMPLED.index("z")
    shuffled[:, z] = shuffled[::-1, z]
    a, b = map_to_domain(u), map_to_domain(shuffled)
    for name in ("T", "sigma", "r"):
        np.testing.assert_array_equal(a[name], b[name])
    # log-uniform T and sigma, uniform r
    (t_low, t_high), (s_low, s_high), (r_low, r_high) = DOMAIN["T_range"], DOMAIN["sigma_range"], DOMAIN["r_range"]
    np.testing.assert_allclose(np.log(a["T"]), np.log(t_low) + col["T"] * np.log(t_high / t_low), rtol=1e-12)
    np.testing.assert_allclose(np.log(a["sigma"]), np.log(s_low) + col["sigma"] * np.log(s_high / s_low), rtol=1e-12)
    np.testing.assert_allclose(a["r"], r_low + col["r"] * (r_high - r_low), rtol=1e-12)


@pytest.mark.parametrize("u_t_sigma, z_ends, m_ends", [
    # 1 week, 5% vol: v = 0.0069, the z range binds on both sides
    ((0.0, 0.0), DOMAIN["z_range"], None),
    # 5 years, 60% vol: v = 1.34, the m cap binds on both sides
    ((1 - 1e-12, 1 - 1e-12), None, DOMAIN["m_range"]),
])
def test_z_interval_ends_where_the_tighter_of_the_z_range_and_m_cap_does(u_t_sigma, z_ends, m_ends):
    # two rows, same T, sigma and r, z at the bottom and the top of its interval
    u = np.full((2, N_DIMS), 0.5)
    u[:, SAMPLED.index("T")], u[:, SAMPLED.index("sigma")] = u_t_sigma
    u[:, SAMPLED.index("z")] = [0.0, 1.0]
    p = map_to_domain(u)
    if z_ends:
        np.testing.assert_allclose(p["z"], z_ends, rtol=1e-12)
    else:
        np.testing.assert_allclose(p["m"], m_ends, rtol=1e-9)


def test_label_puts_match_the_tree_and_black_scholes():
    p = map_to_domain(sobol_points(64, N_DIMS, seed=4))
    out = label_puts(p, nb_steps=50, method="bbsr")
    assert set(out) == set(COLUMNS)
    tree = binomial_put_batch(p["m"], p["T"], p["r"], p["sigma"], nb_steps=50)
    np.testing.assert_array_equal(out["price"], tree.price)
    np.testing.assert_array_equal(out["exercised"], tree.exercised)
    np.testing.assert_array_equal(out["european_price"], black_scholes_price(p["m"], 1.0, p["T"], p["r"], p["sigma"], "put"))
    assert 0 < out["exercised"].mean() < 1


def test_main_writes_raw_columns_and_settings_and_removes_the_chunks(tmp_path):
    gen.main(["--n_rows", "64", "--chunk_size", "16", "--seed", "5", "--output_dir", str(tmp_path), *TINY])
    data = np.load(tmp_path / "tree_put_dataset.npz")
    # the chunks reassemble in Sobol order: row i is labelled point i
    expected = label_puts(map_to_domain(sobol_points(64, N_DIMS, seed=5)), nb_steps=50, method="bbsr")
    for col in COLUMNS:
        np.testing.assert_array_equal(data[col], expected[col], err_msg=col)
    assert data["teacher"] == "tree" and data["method"] == "bbsr" and data["sampler"] == "sobol"
    assert data["nb_steps"] == 50 and data["seed"] == 5 and data["n_rows"] == 64
    for name, bounds in DOMAIN.items():
        np.testing.assert_array_equal(data[name], bounds)
    assert not (tmp_path / "tree_chunks").exists()


@pytest.mark.parametrize("flag, value", [("--n_rows", "100"), ("--chunk_size", "1000")])
def test_main_rejects_sizes_that_are_not_powers_of_two(tmp_path, flag, value):
    with pytest.raises(ValueError, match="power of 2"):
        gen.main([flag, value, "--output_dir", str(tmp_path), *TINY])


def test_generate_tree_chunks_resumes_by_redoing_only_the_missing_chunk(tmp_path, monkeypatch):
    u = sobol_points(64, N_DIMS, seed=6)
    generate_tree_chunks(u, tmp_path, chunk_size=16, nb_steps=50, method="bbsr", n_jobs=1)
    full = load_tree_chunks(tmp_path, 4)
    chunk_file(tmp_path, 2).unlink()
    # A redone chunk is bit-identical, so only a spy shows what was redone
    # (n_jobs=1 runs the jobs in this process, through the patched name).
    labelled = []
    original = gen.generate_chunk
    monkeypatch.setattr(gen, "generate_chunk", lambda path, *a: (labelled.append(path), original(path, *a)))
    generate_tree_chunks(u, tmp_path, chunk_size=16, nb_steps=50, method="bbsr", n_jobs=1)
    assert labelled == [chunk_file(tmp_path, 2)]
    resumed = load_tree_chunks(tmp_path, 4)
    for col in COLUMNS:
        np.testing.assert_array_equal(resumed[col], full[col])


@pytest.mark.parametrize("nb_steps, error", [
    ("51", "even"),  # "bbsr" halves the tree
    # 10-step coarse tree at T = 5: r dt > sigma sqrt(dt) at sigma = 0.05, r = 0.08
    ("20", "probability"),
])
def test_bad_tree_settings_fail_before_the_config_is_written(tmp_path, nb_steps, error):
    # They used to fail inside the first job, after config.json was saved, so
    # the corrected re-run was refused as "other settings".
    args = ["--n_rows", "32", "--chunk_size", "16", "--nb_jobs", "1", "--output_dir", str(tmp_path)]
    with pytest.raises(ValueError, match=error):
        gen.main([*args, "--nb_steps", nb_steps])
    assert not (tmp_path / "tree_chunks").exists()
    gen.main([*args, "--nb_steps", "50"])
    assert (tmp_path / "tree_put_dataset.npz").exists()


def test_interrupted_main_resumes_with_the_same_flags_and_refuses_other_ones(tmp_path, monkeypatch):
    args = ["--n_rows", "32", "--chunk_size", "16", "--output_dir", str(tmp_path), *TINY]

    def interrupted(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(gen, "generate_tree_chunks", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gen.main(args)
    monkeypatch.undo()
    # The stored config is JSON (lists): the domain's tuples must not make
    # the same flags look like different settings.
    with pytest.raises(ValueError, match="other settings"):
        gen.main([*args, "--nb_steps", "60"])  # argparse keeps the last value
    gen.main(args)
    assert np.load(tmp_path / "tree_put_dataset.npz")["price"].shape == (32,)
