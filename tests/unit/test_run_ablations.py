"""optionpricing/ml/experiments/run_ablations.py's pure parts: variant configs, the done check, job order, the summary.
The runner end to end (processes, training, scoring) is in tests/integration/test_run_ablations.py."""

import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

from optionpricing.ml.experiments import run_ablations as ra
from optionpricing.ml.surrogate.tree.generate_tree_dataset import domain_from
from optionpricing.ml.surrogate.tree.tree_training import TrainConfig
from tests.helpers import write_tree_dataset

EVAL = {"ref_steps": 100, "ref_method": "bbsr", "n_z": 3, "n_sigma": 1, "n_T": 2, "rates": [0.04], "tols": [1e-4, 1e-5]}


def test_every_variant_differs_from_the_baseline_in_one_setting_only():
    # an ablation compares one change; two at once would confound them
    base = ra.run_config("baseline", 42)
    assert base == {f.name: getattr(TrainConfig(seed=42), f.name) for f in fields(TrainConfig)}
    for name in ra.VARIANTS.keys() - ra.COMBINED.keys():
        diff = {k for k, v in ra.run_config(name, 42).items() if v != base[k]}
        assert len(diff) == (0 if name == "baseline" else 1), (name, diff)


@pytest.mark.parametrize("name", list(ra.COMBINED))
def test_a_combined_variant_changes_exactly_its_parts_settings(name):
    # anything beyond its parts' changes would make it no test of whether their gains add up
    base = ra.run_config("baseline", 42)
    config = ra.run_config(name, 42)
    parts = ra.COMBINED[name]
    assert len(parts) >= 2 and all(part not in ra.COMBINED for part in parts)
    for part in parts:
        part_config = ra.run_config(part, 42)
        changed = {k for k, v in part_config.items() if v != base[k]}
        assert {k: config[k] for k in changed} == {k: part_config[k] for k in changed}
    expected = {k for part in parts for k, v in ra.run_config(part, 42).items() if v != base[k]}
    assert {k for k, v in config.items() if v != base[k]} == expected


def test_run_config_sets_the_seed_and_an_epochs_override():
    config = ra.run_config("relu", 7, epochs=3)
    assert (config["seed"], config["epochs"], config["activation"]) == (7, 3, "relu")
    assert ra.run_config("relu", 7)["epochs"] == TrainConfig().epochs


def test_run_config_rejects_an_unknown_variant_and_a_bad_override():
    with pytest.raises(ValueError, match="unknown variant"):
        ra.run_config("nope", 42)
    with pytest.raises(ValueError):  # TrainConfig's own check, before any run starts
        ra.run_config("baseline", 42, epochs=0)


def test_slow_variants_are_ranked_first():
    costs = {name: ra.run_cost(ra.run_config(name, 42)) for name in ra.VARIANTS}
    assert costs["shape_0.1"] > costs["lbfgs_200"] > costs["baseline"] == 0


def _finish(directory, config, evaluation=EVAL):
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({"config": config, "evaluation": evaluation}))


def test_is_done_only_with_a_metrics_file_of_the_same_settings(tmp_path):
    config = ra.run_config("baseline", 42)
    directory = ra.run_dir(tmp_path, "baseline", 42)
    assert not ra.is_done(directory, config, EVAL)  # nothing there yet
    _finish(directory, config)
    assert ra.is_done(directory, config, EVAL)


@pytest.mark.parametrize("change", ["config", "evaluation"])
def test_is_done_refuses_a_finished_run_with_other_settings(tmp_path, change):
    # skipping it would put a run of other settings in the variant's row
    config = ra.run_config("baseline", 42)
    directory = ra.run_dir(tmp_path, "baseline", 42)
    _finish(directory, config)
    other_config = {**config, "epochs": 3} if change == "config" else config
    other_eval = {**EVAL, "ref_steps": 200} if change == "evaluation" else EVAL
    with pytest.raises(ValueError, match="other settings"):
        ra.is_done(directory, other_config, other_eval)


def _metrics(price_mae, train_seconds, tols=("0.0001", "1e-05")):
    """A run's metrics.json content with the fields the summary reads (JSON form: string tol keys)."""
    rows = [
        {
            "name": n,
            "price_mae": price_mae,
            "price_max": 2 * price_mae,
            "iv_mae": 1e-4,
            "delta_mae": 1e-4,
            "gamma_mae": 0.1,
        }
        for n in ("|z| < 1", "exercised", "all")
    ]
    rates = {t: {"continuation": {"any": 0.01}} for t in tols}
    return {
        "grid": {"by_z": rows, "violations": rates},
        "test": {"by_z": rows},
        "best": {"valid_premium_term": 1e-5, "valid_delta_term": 2e-5},
        "train_seconds": train_seconds,
    }


def test_summary_is_the_mean_and_sample_sd_over_seeds_in_report_units():
    table = ra.summarize({"baseline": [_metrics(1e-5, 600), _metrics(3e-5, 1200)], "relu": [_metrics(2e-5, 60)]})
    lines = table.splitlines()
    assert lines[0].startswith("| variant | seeds |") and len(lines) == 4
    base = lines[2].split(" | ")
    # price MAE in $ at K = 100: mean of 0.001 and 0.003, sample sd 0.0014
    assert base[1] == "2" and base[2] == "0.0020 ± 0.0014"
    assert base[-1].rstrip(" |") == "15.0 ± 7.1"  # minutes
    relu = lines[3].split(" | ")
    assert relu[2] == "0.0020"  # one seed: no spread


def test_summary_leaves_tolerances_a_run_was_not_scored_at_empty():
    table = ra.summarize({"baseline": [_metrics(1e-5, 60, tols=("0.0001",))]})
    header, row = table.splitlines()[0].split(" | "), table.splitlines()[2].split(" | ")
    assert row[header.index("viol. > 1e-05 %")] == "-"
    assert row[header.index("viol. > 0.0001 %")] == "1.00"


def test_collect_results_reads_every_finished_run_of_every_variant(tmp_path):
    # Regression: a rerun with --variants shape_0.1 rewrote summary.md with that one row (BUGS.md, B-035)
    for variant, seeds in (("baseline", (42, 43)), ("softplus_output", (42,)), ("lambda_delta_0.3_softplus", (44,))):
        for seed in seeds:
            _finish(ra.run_dir(tmp_path, variant, seed), ra.run_config(variant, seed))
    ra.run_dir(tmp_path, "relu", 42).mkdir(parents=True)  # started, never finished: no metrics.json
    _finish(tmp_path / "not_a_variant" / "seed_42", {})  # a folder the runner did not make
    results = ra.collect_results(tmp_path)
    assert list(results) == ["baseline", "softplus_output", "lambda_delta_0.3_softplus"]  # VARIANTS order
    assert [m["config"]["seed"] for m in results["baseline"]] == [42, 43]


def test_collect_results_of_an_empty_out_dir_is_empty(tmp_path):
    assert ra.collect_results(tmp_path) == {}
    assert len(ra.summarize({}).splitlines()) == 2  # header and separator only


def test_mean_sd_ignores_nan_values():
    assert ra._mean_sd([math.nan, 1.0, 3.0], ".1f") == "2.0 ± 1.4"
    assert ra._mean_sd([math.nan], ".1f") == "-"


# --- keep_awake ---


def _recorded_states(monkeypatch, platform):
    calls = []
    monkeypatch.setattr(ra.sys, "platform", platform)
    monkeypatch.setattr(ra, "_set_execution_state", calls.append)
    return calls


def test_keep_awake_holds_the_system_and_display_on_windows_then_releases(monkeypatch):
    # Regression: on a Modern Standby PC the screen turning off suspended an overnight run (BUGS.md)
    calls = _recorded_states(monkeypatch, "win32")
    with ra.keep_awake():
        assert calls == [ra.ES_CONTINUOUS | ra.ES_SYSTEM_REQUIRED | ra.ES_DISPLAY_REQUIRED]
    assert calls[-1] == ra.ES_CONTINUOUS and len(calls) == 2


def test_keep_awake_releases_even_when_the_run_fails(monkeypatch):
    calls = _recorded_states(monkeypatch, "win32")
    with pytest.raises(KeyboardInterrupt):
        with ra.keep_awake():
            raise KeyboardInterrupt  # Ctrl-C mid-run
    assert calls[-1] == ra.ES_CONTINUOUS


def test_keep_awake_does_nothing_elsewhere(monkeypatch):
    calls = _recorded_states(monkeypatch, "linux")
    with ra.keep_awake():
        pass
    assert calls == []


# --- dataset variants and the grid they are scored on ---


@pytest.mark.parametrize("name", list(ra.DATASET_VARIANTS))
def test_a_dataset_variant_trains_its_base_config_on_its_own_dataset(name):
    # anything but the data differing from its base would confound the data's effect
    base, subdir = ra.DATASET_VARIANTS[name]
    assert base in ra.VARIANTS and name not in ra.VARIANTS
    assert ra.run_config(name, 7, epochs=3) == ra.run_config(base, 7, epochs=3)
    default = Path("somewhere") / "tree_put_dataset.npz"
    assert ra.dataset_path_for(name, default) == Path("somewhere") / subdir / "tree_put_dataset.npz"
    assert ra.dataset_path_for(base, default) == default


def test_collect_results_reads_dataset_variants_after_the_config_ones(tmp_path):
    for variant in ("T_from_0.25", "baseline"):
        _finish(ra.run_dir(tmp_path, variant, 42), ra.run_config(variant, 42))
    assert list(ra.collect_results(tmp_path)) == ["baseline", "T_from_0.25"]


def test_evaluation_settings_hold_the_grid_t_min_only_when_given():
    # without the flag, runs finished before it existed must still compare equal (is_done)
    plain = ra.evaluation_settings(ra.make_parser().parse_args([]))
    assert set(plain) == {"ref_steps", "ref_method", "n_z", "n_sigma", "n_T", "rates", "tols"}
    narrow = ra.evaluation_settings(ra.make_parser().parse_args(["--grid_T_min", "0.25"]))
    assert narrow == {**plain, "grid_T_min": 0.25}


def test_runs_scored_before_grid_t_min_existed_still_count_as_done_without_it(tmp_path):
    # the 27 finished ablation runs store the evaluation settings as they were before the
    # flag; a plain rerun must skip them, and one on the narrow grid must refuse to mix them in
    args = ra.make_parser().parse_args([])
    stored = {k: getattr(args, k) for k in ("ref_steps", "ref_method", "n_z", "n_sigma", "n_T", "rates", "tols")}
    config = ra.run_config("baseline", 42)
    directory = ra.run_dir(tmp_path, "baseline", 42)
    _finish(directory, config, stored)
    assert ra.is_done(directory, config, ra.evaluation_settings(args))
    narrow = ra.evaluation_settings(ra.make_parser().parse_args(["--grid_T_min", "0.25"]))
    with pytest.raises(ValueError, match="other settings"):
        ra.is_done(directory, config, narrow)


def test_dataset_variants_run_only_when_named():
    # they need datasets of their own, which a plain run must not require
    assert ra.make_parser().parse_args([]).variants == list(ra.VARIANTS)
    assert ra.make_parser().parse_args(["--variants", "T_from_0.25"]).variants == ["T_from_0.25"]


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """A tiny default dataset and a T >= 0.25 one next to it, where T_from_0.25 looks."""
    root = tmp_path_factory.mktemp("datasets")
    default = write_tree_dataset(root, n_rows=64)
    write_tree_dataset(root / "T0.25", n_rows=64, extra=("--T_min", "0.25"))
    return default


def test_check_dataset_accepts_a_dataset_covering_the_grid(datasets):
    narrow_grid = domain_from(0.25, ra.dataset_domain(datasets))
    ra.check_dataset("baseline", datasets, narrow_grid, "bbsr", 400)  # the wide dataset covers the narrow grid
    ra.check_dataset("T_from_0.25", ra.dataset_path_for("T_from_0.25", datasets), narrow_grid, "bbsr", 400)


def test_check_dataset_refuses_a_grid_beyond_the_dataset_domain(datasets):
    # its 1 week - 3 months points would be routed, so the T >= 0.25 model needs --grid_T_min
    full_grid = ra.dataset_domain(datasets)
    with pytest.raises(ValueError, match="T_range .*--grid_T_min"):
        ra.check_dataset("T_from_0.25", ra.dataset_path_for("T_from_0.25", datasets), full_grid, "bbsr", 400)


def test_check_dataset_names_the_command_for_a_missing_dataset(datasets, tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_tree_dataset .* --output_dir"):
        ra.check_dataset("exercised_0.25", tmp_path / "none.npz", ra.dataset_domain(datasets), "bbsr", 400)


def test_check_dataset_refuses_a_reference_with_the_teachers_settings(datasets):
    with pytest.raises(ValueError, match="teacher"):  # the helper's datasets are 50-step BBSR
        ra.check_dataset("baseline", datasets, ra.dataset_domain(datasets), "bbsr", 50)


def test_dataset_info_records_the_generator_settings_and_domain(datasets):
    info = ra.dataset_info(ra.dataset_path_for("T_from_0.25", datasets))
    assert info["T_range"] == (0.25, 5.0) and info["n_rows"] == 64 and info["nb_steps"] == 50
    assert info["exercised_keep"] == 1.0 and info["path"].endswith("tree_put_dataset.npz")
    assert "m" not in info and "price" not in info  # settings only, not the columns
    json.dumps(info)  # stored in metrics.json
