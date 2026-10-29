"""
optionpricing/ml/experiments/run_ablations.py end to end, as a subprocess (its runs are worker
processes): two variants, one seed, 2 epochs, a tiny dataset and grid. Checks
the files each run leaves, the summary, that a rerun skips finished runs and keeps every
variant's row, and that one with other settings is refused.
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
