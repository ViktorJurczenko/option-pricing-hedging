"""optionpricing/ml/experiments/run_ablations.py's pure parts: variant configs, the done check, job order, the summary.
The runner end to end (processes, training, scoring) is in tests/integration/test_run_ablations.py."""
import json
import math
from dataclasses import fields

import pytest

from optionpricing.ml.experiments import run_ablations as ra
from optionpricing.ml.surrogate.tree.tree_training import TrainConfig

EVAL = {"ref_steps": 100, "ref_method": "bbsr", "n_z": 3, "n_sigma": 1, "n_T": 2, "rates": [0.04],
        "tols": [1e-4, 1e-5]}


def test_every_variant_differs_from_the_baseline_in_one_setting_only():
    # an ablation compares one change; two at once would confound them
    base = ra.run_config("baseline", 42)
    assert base == {f.name: getattr(TrainConfig(seed=42), f.name) for f in fields(TrainConfig)}
    for name in ra.VARIANTS:
        diff = {k for k, v in ra.run_config(name, 42).items() if v != base[k]}
        assert len(diff) == (0 if name == "baseline" else 1), (name, diff)

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
    rows = [{"name": n, "price_mae": price_mae, "price_max": 2 * price_mae, "iv_mae": 1e-4,
             "delta_mae": 1e-4, "gamma_mae": 0.1} for n in ("|z| < 1", "exercised", "all")]
    rates = {t: {"continuation": {"any": 0.01}} for t in tols}
    return {"grid": {"by_z": rows, "violations": rates}, "test": {"by_z": rows},
            "best": {"valid_premium_term": 1e-5, "valid_delta_term": 2e-5}, "train_seconds": train_seconds}

def test_summary_is_the_mean_and_sample_sd_over_seeds_in_report_units():
    table = ra.summarize({"baseline": [_metrics(1e-5, 600), _metrics(3e-5, 1200)], "relu": [_metrics(2e-5, 60)]})
    lines = table.splitlines()
    assert lines[0].startswith("| variant | seeds |") and len(lines) == 4
    base = lines[2].split(" | ")
    # price MAE in $ at K = 100: mean of 0.001 and 0.003, sample sd 0.0014
    assert base[1] == "2" and base[2] == "0.0020 +- 0.0014"
    assert base[-1].rstrip(" |") == "15.0 +- 7.1"  # minutes
    relu = lines[3].split(" | ")
    assert relu[2] == "0.0020"  # one seed: no spread

def test_summary_leaves_tolerances_a_run_was_not_scored_at_empty():
    table = ra.summarize({"baseline": [_metrics(1e-5, 60, tols=("0.0001",))]})
    header, row = table.splitlines()[0].split(" | "), table.splitlines()[2].split(" | ")
    assert row[header.index("viol. > 1e-05 %")] == "-"
    assert row[header.index("viol. > 0.0001 %")] == "1.00"

def test_mean_sd_ignores_nan_values():
    assert ra._mean_sd([math.nan, 1.0, 3.0], ".1f") == "2.0 +- 1.4"
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
