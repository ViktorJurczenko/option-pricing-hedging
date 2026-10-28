import math

import numpy as np
import pytest
import torch

from optionpricing.pricing.american.batch_binomial_tree import binomial_put_batch
from optionpricing.pricing.european.black_scholes import black_scholes_price, black_scholes_vega
from optionpricing.ml.surrogate.tree import tree_evaluation as te
from optionpricing.ml.surrogate.tree.tree_dataset import FeatureStats, TreePutDataset, build_features
from optionpricing.ml.surrogate.tree.generate_tree_dataset import DOMAIN, map_to_domain, sobol_points
from optionpricing.ml.surrogate.tree.tree_model import MLP
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate
from optionpricing.ml.surrogate.tree.tree_training import SHAPE_CONDITIONS, ShapePoints, shape_violations
from tests.helpers import write_tree_dataset


def _surrogate(zero_net=False, domain=DOMAIN, stats=None):
    """A seeded, untrained float64 surrogate (float64 so finite differences are precise)."""
    if stats is None:
        p = map_to_domain(sobol_points(256, 4, seed=0))
        stats = FeatureStats.fit(build_features(p["m"], p["T"], p["r"], p["sigma"]))
    torch.manual_seed(0)
    model = MLP(depth=2, width=16).double()
    if zero_net:  # premium identically 0: the net route is plain Black-Scholes
        torch.nn.init.zeros_(model.net[-1].weight)
        torch.nn.init.zeros_(model.net[-1].bias)
    return PutSurrogate(model, stats, domain)

# --- the grid ---

def test_grid_is_the_full_factorial_when_the_moneyness_cap_does_not_bind():
    wide = {**DOMAIN, "m_range": (1e-6, 1e6)}
    grid = te.evaluation_grid(wide, n_z=5, n_sigma=3, n_T=4, rates=(0.01, 0.05))
    assert len(grid.m) == 5 * 3 * 4 * 2
    # every combination once: the pairs of distinct values are all present
    combos = set(zip(np.round(grid.z, 9), grid.sigma, grid.T, grid.r))
    assert len(combos) == len(grid.m)
    assert set(np.round(grid.sigma, 12)) == set(np.round(np.geomspace(*DOMAIN["sigma_range"], 3), 12))
    np.testing.assert_allclose(grid.m, np.exp(grid.z * grid.sigma * np.sqrt(grid.T)))

def test_grid_drops_exactly_the_points_outside_the_moneyness_cap():
    full = te.evaluation_grid({**DOMAIN, "m_range": (1e-6, 1e6)}, n_z=9, n_sigma=4, n_T=5)
    capped = te.evaluation_grid(DOMAIN, n_z=9, n_sigma=4, n_T=5)
    m_low, m_high = DOMAIN["m_range"]
    inside = (full.m >= m_low) & (full.m <= m_high)
    assert 0 < inside.sum() < len(full.m)  # the default domain's long, volatile corner reaches the cap
    np.testing.assert_array_equal(capped.m, full.m[inside])

def test_every_grid_point_is_priced_by_the_net_even_at_the_z_bounds():
    # z = -3.5 and 3 exactly would round across the bound once rebuilt from m and leave the net route
    grid = te.evaluation_grid(DOMAIN, n_z=2, n_sigma=3, n_T=3)
    assert grid.z.min() == pytest.approx(DOMAIN["z_range"][0]) and grid.z.max() == pytest.approx(DOMAIN["z_range"][1])
    route = _surrogate(zero_net=True).price(grid.m, grid.T, grid.r, grid.sigma).route
    assert set(route) <= {"net", "payoff_floor"}

@pytest.mark.parametrize("kwargs", [{"n_z": 0}, {"n_sigma": 0}, {"n_T": 0}, {"rates": ()},
                                    {"rates": (0.09,)}, {"rates": (-0.01,)}])
def test_grid_rejects_empty_axes_and_rates_outside_the_domain(kwargs):
    with pytest.raises(ValueError):
        te.evaluation_grid(DOMAIN, **kwargs)

# --- reference tree and vega ---

def test_tree_prices_in_blocks_equal_one_tree_call():
    m = np.linspace(0.7, 1.3, te.TREE_BLOCK + 3)  # crosses a block boundary
    T, r, s = np.full_like(m, 0.5), np.full_like(m, 0.04), np.full_like(m, 0.3)
    res = te.tree_prices(m, T, r, s, nb_steps=50, method="bbsr")
    ref = binomial_put_batch(m, T, r, s, nb_steps=50, method="bbsr")
    for got, want in zip(res, ref):
        np.testing.assert_array_equal(got, want)

def test_tree_vega_matches_black_scholes_when_early_exercise_is_worthless():
    # with r = 0 an American put is never exercised early, so it is the European put
    m = np.array([0.8, 1.0, 1.2])
    T, r, s = np.full(3, 0.5), np.zeros(3), np.full(3, 0.25)
    # tolerance: measured 4e-5 here (the tree's price error over the 2 * 1% * sigma wide
    # difference), 5x margin; still 50x below MIN_IV_VEGA, the scale vega is used at
    np.testing.assert_allclose(te.tree_vega(m, T, r, s), black_scholes_vega(m, 1.0, T, r, s), atol=2e-4)

# --- net diagnostics and the theta check ---

def test_net_diagnostics_match_the_pricer_finite_differences_and_shape_violations():
    sur = _surrogate()
    m, T, r, s = np.array([0.8, 1.0, 1.15]), np.array([0.3, 0.7, 1.5]), np.array([0.02, 0.04, 0.06]), np.full(3, 0.3)
    diag = te.net_diagnostics(sur, m, T, r, s)
    res = sur.price(m, T, r, s)
    raw = black_scholes_price(m, 1.0, T, r, s, "put") + sur._premium(m, T, r, s)[0]
    np.testing.assert_allclose(diag.price, raw, atol=1e-12)
    # theta = -dV/dT; central difference, step 1e-5: truncation ~1e-10, rounding ~1e-11
    h = 1e-5
    fd = -(sur.price(m, T + h, r, s).price - sur.price(m, T - h, r, s).price) / (2 * h)
    assert (res.route == "net").all()  # no floor: the pricer's price is the raw net's
    np.testing.assert_allclose(diag.theta, fd, atol=1e-7)
    np.testing.assert_allclose(diag.theta_pde, r * raw - r * m * res.delta - 0.5 * s ** 2 * m ** 2 * res.gamma,
                               atol=1e-10)
    pts = ShapePoints.from_params(m, T, r, s)
    np.testing.assert_allclose(diag.violations, shape_violations(sur.model, sur.stats, pts).detach().numpy())

def test_theta_gap_vanishes_for_black_scholes_and_is_theta_on_exercised_rows():
    # a zero-premium net is the European put, which satisfies the Black-Scholes PDE exactly
    sur = _surrogate(zero_net=True)
    grid = te.evaluation_grid(DOMAIN, n_z=7, n_sigma=3, n_T=3)
    diag = te.net_diagnostics(sur, grid.m, grid.T, grid.r, grid.sigma)
    held = np.zeros(len(grid.m), dtype=bool)
    gap = te.theta_gap(diag, held)
    # relative to |theta|: float64 rounding of the closed forms, far below the 1.8e-6 / 7.8e-3 the tree reaches
    assert np.max(gap / (np.abs(diag.theta) + 1e-3)) < 1e-8
    exercised = np.ones_like(held)
    np.testing.assert_array_equal(te.theta_gap(diag, exercised), np.abs(diag.theta))

def test_net_diagnostics_of_no_points_are_empty():
    diag = te.net_diagnostics(_surrogate(), *(np.empty(0) for _ in range(4)))
    assert diag.price.shape == (0,) and diag.violations.shape == (0, len(SHAPE_CONDITIONS))

# --- buckets and metrics ---

def test_bucket_masks_split_continuation_rows_at_inclusive_low_edges_and_keep_exercised_apart():
    values = np.array([0.0, 0.999, 1.0, 1.5, 2.0, 3.4, 0.5])
    exercised = np.array([False] * 6 + [True])
    groups = dict(te.bucket_masks(values, te.Z_BUCKETS, exercised))
    assert list(groups) == ["|z| < 1", "1 <= |z| < 2", "|z| >= 2", "exercised", "all"]
    np.testing.assert_array_equal(groups["|z| < 1"], [1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(groups["1 <= |z| < 2"], [0, 0, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(groups["|z| >= 2"], [0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(groups["exercised"], exercised)
    assert groups["all"].all()

def test_bucket_metrics_restrict_relative_and_iv_errors_to_where_they_mean_something():
    price_err = np.array([1e-4, -2e-4, 3e-4, -4e-4])
    ref_price = np.array([1e-2, 1e-4, 2e-2, 5e-1])           # row 1 below MIN_RELATIVE_PRICE
    vega = np.array([0.2, 0.3, 1e-3, 0.4])                    # row 2 below MIN_IV_VEGA
    exercised = np.array([False, False, False, True])         # row 3: no implied vol
    floored = np.array([False, True, False, True])
    groups = [("all", np.ones(4, dtype=bool)), ("none", np.zeros(4, dtype=bool))]
    rows = te.bucket_metrics(groups, price_err, ref_price, delta_err=np.array([0.1, -0.2, 0.0, 0.3]),
                             floored=floored, gamma_err=np.array([1.0, -1.0, 2.0, 0.0]), vega=vega,
                             exercised=exercised, gap=np.array([1.0, 2.0, 3.0, 10.0]))
    row = rows[0]
    assert row["n"] == 4
    assert row["price_mae"] == pytest.approx(2.5e-4) and row["price_max"] == pytest.approx(4e-4)
    assert row["n_rel"] == 3
    assert row["rel_mae"] == pytest.approx(np.mean([1e-4 / 1e-2, 3e-4 / 2e-2, 4e-4 / 5e-1]))
    assert row["n_iv"] == 2
    assert row["iv_mae"] == pytest.approx(np.mean([1e-4 / 0.2, 2e-4 / 0.3]))
    assert row["delta_mae"] == pytest.approx(0.15) and row["gamma_mae"] == pytest.approx(1.0)
    assert row["theta_gap"] == pytest.approx(2.5)  # the median, robust to the one large gap
    assert row["floor_rate"] == pytest.approx(0.5)
    empty = rows[1]
    assert empty["n"] == 0 and all(math.isnan(empty[k]) for k in ("price_mae", "price_max", "rel_mae", "theta_gap"))

def test_bucket_metrics_without_gamma_and_vega_leave_those_columns_empty():
    row = te.bucket_metrics([("all", np.ones(2, dtype=bool))], np.array([1e-3, 0.0]), np.array([0.1, 0.1]),
                            np.zeros(2), np.zeros(2, dtype=bool))[0]
    assert math.isnan(row["gamma_mae"]) and math.isnan(row["iv_mae"]) and row["n_iv"] == 0

def test_violation_rates_count_strictly_above_the_tolerance_per_region():
    tol = 1e-4
    violations = np.array([[0, 0, 0, 0], [tol, 0, 0, 0], [0, 2 * tol, 0, 0], [0, 0, 3 * tol, 5 * tol]], dtype=float)
    exercised = np.array([False, False, False, True])
    rates = te.violation_rates(violations, exercised, tol)
    cont, ex = rates["continuation"], rates["exercised"]
    assert cont["n"] == 3 and ex["n"] == 1
    assert cont["decreasing_in_m"] == 0.0  # exactly at the tolerance does not count
    assert cont["convex_in_m"] == pytest.approx(1 / 3) and cont["any"] == pytest.approx(1 / 3)
    assert ex["increasing_in_T"] == 1.0 and ex["increasing_in_sigma"] == 1.0 and ex["any"] == 1.0

def test_violation_rates_by_tol_gives_one_table_per_tolerance_in_order():
    violations = np.array([[0, 0, 0, 0], [5e-5, 0, 0, 0], [2e-4, 0, 0, 0]], dtype=float)
    exercised = np.zeros(3, dtype=bool)
    rates = te.violation_rates_by_tol(violations, exercised, (1e-4, 1e-5))
    assert list(rates) == [1e-4, 1e-5]
    for tol in (1e-4, 1e-5):
        assert rates[tol] == te.violation_rates(violations, exercised, tol)
    # 5e-5 counts only at the stricter tolerance
    assert rates[1e-4]["continuation"]["any"] == pytest.approx(1 / 3)
    assert rates[1e-5]["continuation"]["any"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="tolerance"):
        te.violation_rates_by_tol(violations, exercised, ())

# --- routing, reference checks ---

def test_routing_queries_leave_the_net_and_reach_every_other_route():
    q = te.routing_queries(DOMAIN)
    route = _surrogate().price(q.m, q.T, q.r, q.sigma).route
    assert "net" not in route and "payoff_floor" not in route
    assert {"black_scholes", "exercised", "tree"} <= set(route)

def test_check_reference_refuses_the_teachers_own_settings():
    with pytest.raises(ValueError, match="teacher"):
        te.check_reference(("bbsr", 1000), "bbsr", 1000)
    te.check_reference(("bbsr", 1000), "bbsr", 4000)
    te.check_reference(("bbsr", 1000), "bbs", 1000)

# --- whole evaluations ---

def test_grid_evaluation_of_black_scholes_at_zero_rate_matches_the_reference_tree():
    # r = 0: the American put is the European one, so a zero-premium net is exact and
    # every error left is the reference tree's own, which tests the alignment of the arrays
    sur = _surrogate(zero_net=True)
    out = te.evaluate_grid(sur, ref_steps=400, ref_method="bbsr", n_z=5, n_sigma=2, n_T=2, rates=(0.0,))
    all_row = out["by_z"][-1]
    assert all_row["name"] == "all" and all_row["n"] == out["n"] > 0
    # tolerance: BBSR's error at 400 steps, ~2.5x the 1000-step maximum of 3.2e-5 (error ~ 1/N)
    assert all_row["price_max"] < 1e-4
    assert out["by_z"][-2]["n"] == 0  # nothing exercised at r = 0
    for table in ("by_z", "by_T", "by_r"):
        assert sum(row["n"] for row in out[table][:-1]) == out["n"]  # the buckets partition the grid
    assert sum(out["routes"].values()) == out["n"]
    assert {row["route"] for row in out["routing"]} >= {"black_scholes", "exercised", "tree"}
    assert all(v > 0 for v in out["timing"].values())

def test_test_split_scores_against_the_files_labels(tmp_path):
    path = write_tree_dataset(tmp_path)
    data = TreePutDataset(path)
    sur = _surrogate(stats=data.stats, domain=data.domain)
    out = te.evaluate_test_split(sur, path)
    # independent of the module: the split TreePutDataset documents (seeded permutation, 80%)
    with np.load(path) as f:
        cols = {c: f[c] for c in ("m", "T", "r", "sigma", "price", "exercised")}
    idx = np.random.default_rng(42).permutation(len(cols["m"]))[int(len(cols["m"]) * 0.8):]
    pred = sur.price(*(cols[c][idx] for c in ("m", "T", "r", "sigma"))).price
    all_row = out["by_z"][-1]
    assert out["n"] == len(idx) == all_row["n"]
    # the labels are rebuilt from float32 premiums: ~1e-9 off the file's prices
    assert all_row["price_mae"] == pytest.approx(np.mean(np.abs(pred - cols["price"][idx])), abs=1e-8)
    assert out["by_z"][-2]["n"] == cols["exercised"][idx].sum()
    assert out["violations"][te.VIOLATION_TOL]["exercised"]["n"] == cols["exercised"][idx].sum()

def test_test_split_refuses_a_surrogate_trained_on_another_domain(tmp_path):
    path = write_tree_dataset(tmp_path)
    data = TreePutDataset(path)
    other = {**data.domain, "T_range": (0.25, 5.0)}
    with pytest.raises(ValueError, match="another dataset"):
        te.evaluate_test_split(_surrogate(stats=data.stats, domain=other), path)

def test_format_table_aligns_and_prints_nan_as_a_dash():
    rows = [{"name": "a", "n": 3, "price_mae": 1e-4}, {"name": "bb", "n": 10, "price_mae": math.nan}]
    text = te.format_table(rows, (("bucket", "name", None, ""), ("n", "n", None, "d"),
                                  ("price MAE $", "price_mae", te.STRIKE, ".4f")))
    lines = text.splitlines()
    assert len({len(line) for line in lines}) == 1  # aligned
    assert lines[2].endswith("0.0100") and lines[3].endswith("-")

def test_bucket_metrics_refuses_vega_without_the_exercised_flags():
    # the implied-vol error must skip exercised rows, so vega alone used to crash on ~None
    with pytest.raises(ValueError, match="exercised"):
        te.bucket_metrics([("all", np.ones(2, dtype=bool))], np.zeros(2), np.ones(2), np.zeros(2),
                          np.zeros(2, dtype=bool), vega=np.ones(2))

def test_grid_evaluation_with_rates_separates_exercised_rows_and_prices_them_exactly():
    # r > 0: some puts are exercised today. A zero-premium net prices them at the European
    # price, below the payoff, so the floor answers with exactly 1 - m, the reference's price
    sur = _surrogate(zero_net=True)
    rates = (0.02, 0.06)
    kw = dict(n_z=7, n_sigma=2, n_T=2, rates=rates)
    out = te.evaluate_grid(sur, ref_steps=200, ref_method="bbsr", **kw)

    grid = te.evaluation_grid(DOMAIN, **kw)
    exercised = binomial_put_batch(grid.m, grid.T, grid.r, grid.sigma, nb_steps=200, method="bbsr").exercised
    assert 0 < exercised.sum() < len(grid.m)  # both regions present, else this test checks nothing

    by_r = {row["name"]: row for row in out["by_r"]}
    for rate in rates:
        assert by_r[f"r = {rate:g}"]["n"] == int((~exercised & (grid.r == rate)).sum())
    ex = by_r["exercised"]
    assert ex["n"] == exercised.sum()
    assert ex["price_max"] == 0.0 and ex["floor_rate"] == 1.0
    assert ex["n_iv"] == 0  # no implied volatility on exercised rows
    assert out["violations"][te.VIOLATION_TOL]["exercised"]["n"] == exercised.sum()
    for table in ("by_z", "by_T", "by_r"):
        assert sum(row["n"] for row in out[table][:-1]) == out["n"]

def test_teacher_settings_reads_the_datasets_tree_method_and_steps(tmp_path):
    assert te.teacher_settings(write_tree_dataset(tmp_path)) == ("bbsr", 50)  # the helper's 50-step trees

def _report_results(tmp_path):
    """A tiny grid result and test-split result of one untrained surrogate."""
    path = write_tree_dataset(tmp_path)
    data = TreePutDataset(path)
    sur = _surrogate(stats=data.stats, domain=data.domain)
    grid = te.evaluate_grid(sur, ref_steps=100, ref_method="bbsr", n_z=3, n_sigma=1, n_T=1, rates=(0.04,))
    return grid, te.evaluate_test_split(sur, path)

def test_print_report_prints_only_the_sections_it_is_given(tmp_path, capsys):
    grid, test = _report_results(tmp_path)
    capsys.readouterr()  # drop the dataset writer's and progress bars' output

    te.print_report(grid=grid)
    out = capsys.readouterr().out
    assert "=== Grid" in out and "=== Test split" not in out and "Time per price" in out

    te.print_report(test=test)
    out = capsys.readouterr().out
    assert "=== Test split" in out and "=== Grid" not in out and "Time per price" not in out
    assert "IV err bp" not in out  # the test split's tables have no implied-vol column (the footnote still defines it)

def test_print_report_prints_one_violation_table_per_tolerance(tmp_path, capsys):
    path = write_tree_dataset(tmp_path)
    data = TreePutDataset(path)
    sur = _surrogate(stats=data.stats, domain=data.domain)
    grid = te.evaluate_grid(sur, ref_steps=100, ref_method="bbsr", n_z=3, n_sigma=1, n_T=1, rates=(0.04,),
                            tols=(1e-4, 1e-5))
    test = te.evaluate_test_split(sur, path, tols=(1e-4, 1e-5))
    capsys.readouterr()
    te.print_report(grid, test)
    out = capsys.readouterr().out
    # once for the test split, once for the grid, each tolerance
    assert out.count("violations > 0.0001 ") == 2 and out.count("violations > 1e-05 ") == 2


# --- the reference cache ---

SMALL = dict(ref_steps=100, ref_method="bbsr", n_z=3, n_sigma=1, n_T=2, rates=(0.04,))

def test_grid_reference_cache_is_written_then_loaded_without_recomputing(tmp_path, monkeypatch):
    cache = tmp_path / "sub" / "ref.npz"
    first = te.grid_reference(DOMAIN, cache_path=cache, **SMALL)
    assert cache.is_file() and not list(cache.parent.glob("*.tmp.npz"))

    def no_trees(*args, **kwargs):
        raise AssertionError("a cached reference must not price any tree")
    monkeypatch.setattr(te, "tree_prices", no_trees)
    again = te.grid_reference(DOMAIN, cache_path=cache, **SMALL)
    for a, b in zip((*first.grid, *first.tree, first.vega), (*again.grid, *again.tree, again.vega)):
        np.testing.assert_array_equal(a, b)
    assert again.tree.exercised.dtype == bool and again.seconds_per_option == first.seconds_per_option

@pytest.mark.parametrize("change, named", [({"ref_steps": 200}, "ref_steps"), ({"n_z": 4}, "n_z"),
                                           ({"rates": (0.05,)}, "rates"), ("domain", "T_range")])
def test_grid_reference_cache_refuses_other_settings(tmp_path, change, named):
    # a reference for another grid or tree would silently score every model against the wrong prices
    cache = tmp_path / "ref.npz"
    te.grid_reference(DOMAIN, cache_path=cache, **SMALL)
    domain, kw = DOMAIN, dict(SMALL)
    if change == "domain":
        domain = {**DOMAIN, "T_range": (0.25, 5.0)}
    else:
        kw.update(change)
    with pytest.raises(ValueError, match=named):
        te.grid_reference(domain, cache_path=cache, **kw)

def test_evaluate_grid_scores_the_same_with_a_cached_reference(tmp_path):
    sur = _surrogate()
    kw = dict(SMALL, tols=(1e-4, 1e-5))
    plain = te.evaluate_grid(sur, **kw)
    cache = tmp_path / "ref.npz"
    te.evaluate_grid(sur, reference_cache=cache, **kw)  # writes the cache
    cached = te.evaluate_grid(sur, reference_cache=cache, **kw)  # reads it
    for key in ("n", "by_z", "by_T", "by_r", "violations", "routes"):
        assert cached[key] == plain[key], key
