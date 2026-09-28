"""
End-to-end smoke tests for the `python main.py <command>` CLI.

These exist mainly to catch wiring/import mistakes across the package layout
(main.py -> scripts/* -> the library packages) that unit tests, which import
modules directly, would not notice.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN = str(ROOT / "main.py")

def run_cli(*args, timeout=120):
    return subprocess.run(
        [sys.executable, MAIN, *args],
        capture_output=True, text=True, timeout=timeout, cwd=ROOT,
    )

def test_price_eu_and_us_command_runs_successfully():
    result = run_cli(
        "price", "--option_class", "both",
        "--nb_predictions", "500", "--nb_steps", "50", "--maturity", "1",
    )
    assert result.returncode == 0, result.stderr
    assert "Black and Scholes" in result.stdout
    assert "Longstaff-Schwartz" in result.stdout

def test_train_hedge_command_runs_successfully():
    result = run_cli(
        "train-hedge",
        "--n_episodes", "2", "--nb_predictions", "200", "--sampling", "20",
        "--warmup_steps", "0",
        "--checkpoint_path", "checkpoints/_smoke_test_policy.pth",
    )
    assert result.returncode == 0, result.stderr

    checkpoint = ROOT / "checkpoints" / "_smoke_test_policy.pth"
    checkpoint.unlink(missing_ok=True)

def test_train_hedge_command_accepts_lr_and_alpha_flags():
    # End-to-end check that --lr/--alpha (newly exposed CLI flags wrapping
    # AmericanHedging.Hedge.train()'s existing lr/alpha kwargs) are accepted
    # and actually reach argparse/train() without erroring, not just that
    # add_arguments() knows about them in isolation.
    result = run_cli(
        "train-hedge",
        "--n_episodes", "2", "--nb_predictions", "200", "--sampling", "20",
        "--warmup_steps", "0", "--lr", "3e-4", "--alpha", "0.05",
        "--checkpoint_path", "checkpoints/_smoke_test_policy_lr_alpha.pth",
    )
    assert result.returncode == 0, result.stderr

    checkpoint = ROOT / "checkpoints" / "_smoke_test_policy_lr_alpha.pth"
    checkpoint.unlink(missing_ok=True)

@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "best_model.pth").exists(),
    reason="no trained MLP checkpoint to test against",
)
def test_test_ml_command_runs_successfully():
    result = run_cli("test-ml")
    assert result.returncode == 0, result.stderr
    assert "MSE" in result.stdout

@pytest.mark.skipif(
    not (ROOT / "checkpoints" / "best_policy_model.pth").exists(),
    reason="no trained policy checkpoint to test against",
)
def test_test_hedge_command_runs_successfully():
    result = run_cli(
        "test-hedge",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode == 0, result.stderr
    assert "Mean P&L" in result.stdout

def test_test_hedge_delta_policy_runs_without_checkpoint():
    # --policy delta needs no trained checkpoint at all, unlike the rl default.
    result = run_cli(
        "test-hedge", "--policy", "delta",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode == 0, result.stderr
    assert "Mean P&L" in result.stdout

def test_test_hedge_no_hedge_policy_runs_without_checkpoint():
    result = run_cli(
        "test-hedge", "--policy", "no_hedge",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode == 0, result.stderr
    assert "Mean P&L" in result.stdout

def test_test_hedge_invalid_policy_choice_is_rejected():
    result = run_cli(
        "test-hedge", "--policy", "bogus",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()

def test_test_hedge_same_seed_and_policy_is_reproducible_across_runs():
    # Basic determinism check: two independent subprocess runs with identical
    # CLI args (same --policy and --seed) must print byte-identical metrics.
    # This alone doesn't prove the --seed reseed added in scripts/test_hedge.py
    # matters (GBM's own --seed-based np.random.seed() already makes a single
    # policy reproducible); the cross-policy fairness property that reseed is
    # actually for is covered by tests/integration/test_hedge_policy_seeding.py,
    # which can observe the np.random state directly instead of just stdout.
    args = (
        "test-hedge", "--policy", "no_hedge", "--seed", "7",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    first = run_cli(*args)
    second = run_cli(*args)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
