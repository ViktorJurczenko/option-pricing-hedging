"""
optionpricing/ml/experiments/run_ablations.py end to end, as a subprocess (its runs are worker
processes): two variants, one seed, 2 epochs, a tiny dataset and grid. Checks
the files each run leaves, the summary, that a rerun skips finished runs and keeps every
variant's row, and that one with other settings is refused. Then a dataset variant trained
on its own T >= 0.25 dataset, scored with its base config on the same --grid_T_min grid.
"""

import json

import pytest

from tests.helpers import run_python, write_tree_dataset

TINY = [
    "--variants",
    "baseline",
    "softplus_output",
    "--seeds",
    "1",
    "--epochs",
    "2",
    "--jobs",
    "2",
    "--threads",
    "1",
    "--ref_steps",
    "100",
    "--n_z",
    "3",
    "--n_sigma",
    "1",
    "--n_T",
    "2",
    "--rates",
    "0.04",
]
TIMEOUT = 600  # two trainings, their scoring and the processes' start-up


@pytest.fixture(scope="module")
def ablation(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ablation")
    dataset, out = write_tree_dataset(tmp), tmp / "runs"
    args = [
        "-m",
        "optionpricing.ml.experiments.run_ablations",
        "--dataset_path",
        str(dataset),
        "--out_dir",
        str(out),
        *TINY,
    ]
    first = run_python(*args, timeout=TIMEOUT)
    return args, out, first


def test_each_run_leaves_its_model_history_report_and_metrics(ablation):
    _, out, first = ablation
    assert first.returncode == 0, first.stderr
    for variant in ("baseline", "softplus_output"):
        directory = out / variant / "seed_1"
        assert all((directory / f).is_file() for f in ("model.pth", "history.json", "report.txt", "log.txt"))
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["config"]["seed"] == 1 and metrics["config"]["epochs"] == 2
        assert set(metrics["grid"]["violations"]) == {"0.0001", "1e-05"}  # both tolerances, one evaluation
        assert "=== Grid" in (directory / "report.txt").read_text()
    assert (
        json.loads((out / "softplus_output" / "seed_1" / "metrics.json").read_text())["config"]["output"] == "softplus"
    )
    assert (out / "reference_grid.npz").is_file()


def test_the_summary_has_a_row_per_variant(ablation):
    _, out, first = ablation
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| baseline | 1 |" in summary and "| softplus_output | 1 |" in summary
    assert "| baseline | 1 |" in first.stdout


def test_a_rerun_skips_finished_runs_and_other_settings_are_refused(ablation):
    args, out, _ = ablation
    before = (out / "baseline" / "seed_1" / "metrics.json").stat().st_mtime_ns
    again = run_python(*args, timeout=TIMEOUT)
    assert again.returncode == 0, again.stderr
    assert "0 runs to do, 2 already done" in again.stdout
    assert (out / "baseline" / "seed_1" / "metrics.json").stat().st_mtime_ns == before

    # Regression: a rerun of a subset rewrote summary.md with only that subset's rows (BUGS.md, B-035)
    i = args.index("--variants")
    subset = run_python(*args[: i + 1], "baseline", *args[i + 3 :], timeout=TIMEOUT)
    assert subset.returncode == 0, subset.stderr
    assert "| softplus_output | 1 |" in (out / "summary.md").read_text(encoding="utf-8")

    i = args.index("--epochs")
    other = run_python(*args[: i + 1], "3", *args[i + 2 :], timeout=TIMEOUT)  # the same runs, 3 epochs
    assert other.returncode != 0 and "other settings" in other.stderr


@pytest.fixture(scope="module")
def t_range_ablation(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t_range")
    dataset, out = write_tree_dataset(tmp), tmp / "runs"
    write_tree_dataset(tmp / "T0.25", extra=("--T_min", "0.25"))  # where T_from_0.25 looks
    i = TINY.index("--seeds")  # the same tiny settings, these two variants
    args = [
        "-m",
        "optionpricing.ml.experiments.run_ablations",
        "--dataset_path",
        str(dataset),
        "--out_dir",
        str(out),
        "--variants",
        "lambda_delta_0.3_softplus",
        "T_from_0.25",
        *TINY[i:],
    ]
    return args, out, run_python(*args, "--grid_T_min", "0.25", timeout=TIMEOUT)


def test_a_dataset_variant_trains_on_its_dataset_and_both_score_on_the_narrow_grid(t_range_ablation):
    _, out, run = t_range_ablation
    assert run.returncode == 0, run.stderr
    for variant, T_low in (("lambda_delta_0.3_softplus", 1 / 52), ("T_from_0.25", 0.25)):
        metrics = json.loads((out / variant / "seed_1" / "metrics.json").read_text())
        assert metrics["dataset"]["T_range"][0] == pytest.approx(T_low)  # each trained on its own data
        assert metrics["config"]["lambda_delta"] == 0.3 and metrics["config"]["output"] == "softplus"
        assert metrics["evaluation"]["grid_T_min"] == 0.25
        by_T = {row["name"]: row["n"] for row in metrics["grid"]["by_T"]}
        assert by_T["T < 1 month"] == by_T["1-3 months"] == 0  # the shared grid starts at 0.25
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| lambda_delta_0.3_softplus | 1 |" in summary and "| T_from_0.25 | 1 |" in summary


def test_a_dataset_variant_on_the_full_grid_is_refused_before_any_run(t_range_ablation):
    # its dataset starts at 0.25, so the default grid's shorter maturities would not be priced by its net
    args, out, _ = t_range_ablation
    full = list(args)
    full[full.index("--out_dir") + 1] = str(out.parent / "full")
    refused = run_python(*full, timeout=TIMEOUT)
    assert refused.returncode != 0 and "--grid_T_min" in refused.stderr
    assert not (out.parent / "full").exists()
