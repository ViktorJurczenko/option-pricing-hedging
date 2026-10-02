"""
End-to-end smoke tests for the `python main.py <command>` CLI.

These exist mainly to catch wiring/import mistakes across the package layout
(main.py -> scripts/* -> the library packages) that unit tests, which import
modules directly, would not notice.
"""
import re

import pytest
import torch

from AmericanHedging.RLPolicy import checkpoint_action_dim
from tests.helpers import ROOT, run_cli

def _price(stdout, label):
    """The price printed on the line containing `label` (e.g. 'Binomial tree')."""
    match = re.search(rf"{label}\s*: (-?\d+\.\d+)", stdout)
    assert match, f"no '{label}' line in: {stdout}"
    return float(match.group(1))

def _metrics(stdout):
    """(mean, std, var95, cvar95) parsed from the P&L report test-hedge prints."""
    values = []
    for label in (r"Mean P&L:", r"Std  P&L \(error\):", r"VaR  95%:", r"CVaR 95%:"):
        match = re.search(rf"{label}\s+(-?\d+\.\d+)", stdout)
        assert match, f"no '{label}' line in: {stdout}"
        values.append(float(match.group(1)))
    return tuple(values)

def _assert_sane_pnl_report(stdout):
    # P&L is scaled by the strike, so a hedged or unhedged 1-year call stays well inside +-1.
    mean, std, var95, cvar95 = _metrics(stdout)
    assert 0.0 < std < 1.0
    assert abs(mean) < 1.0
    assert cvar95 <= var95 <= mean  # expected shortfall <= VaR <= mean for the worst-tail definition
    return mean, std, var95, cvar95


def test_price_eu_and_us_call_prices_are_close_to_black_scholes():
    result = run_cli(
        "price", "--option_class", "both",
        "--nb_predictions", "500", "--nb_steps", "50", "--maturity", "1",
    )
    assert result.returncode == 0, result.stderr
    bs = _price(result.stdout, "Black and Scholes")
    assert bs == pytest.approx(10.45, abs=0.005)  # S=K=100, r=5%, sigma=20%, T=1
    # Measured with the default seed: tree 10.41 (50 steps), MC 11.02, LSM 10.93 with 500 paths
    # (standard error ~0.6), so +-1.5 is ~2.6x the worst deviation; the tree has no sampling noise.
    assert _price(result.stdout, "Binomial tree") == pytest.approx(bs, abs=0.1)
    assert _price(result.stdout, "Monte-Carlo") == pytest.approx(bs, abs=1.5)
    assert _price(result.stdout, "Longstaff-Schwartz") == pytest.approx(bs, abs=1.5)

def test_price_put_option_type_reports_put_prices_with_early_exercise_premium():
    result = run_cli(
        "price", "--option_class", "both", "--option_type", "put",
        "--nb_predictions", "5000", "--nb_steps", "500", "--sampling", "52",
    )
    assert result.returncode == 0, result.stderr
    assert "put option price" in result.stdout and "call option price" not in result.stdout
    bs = _price(result.stdout, "Black and Scholes")
    tree = _price(result.stdout, "Binomial tree")
    assert bs == pytest.approx(5.57, abs=0.005)
    assert tree == pytest.approx(6.09, abs=0.02)  # CRR(500), measured 6.0888
    assert tree > bs + 0.3                         # American put premium ~0.5
    # Measured with the default seed: MC 5.55, LSM 6.05 (5000 paths); +-0.4 is several times that error.
    assert _price(result.stdout, "Monte-Carlo") == pytest.approx(bs, abs=0.4)
    assert _price(result.stdout, "Longstaff-Schwartz") == pytest.approx(tree, abs=0.4)

def test_price_invalid_option_type_is_rejected():
    result = run_cli("price", "--option_type", "straddle")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()

def test_train_hedge_command_writes_a_loadable_checkpoint():
    checkpoint = ROOT / "checkpoints" / "_smoke_test_policy.pth"
    checkpoint.unlink(missing_ok=True)
    try:
        result = run_cli(
            "train-hedge",
            "--n_episodes", "2", "--nb_predictions", "200", "--sampling", "20",
            "--warmup_steps", "0",
            "--checkpoint_path", "checkpoints/_smoke_test_policy.pth",
        )
        assert result.returncode == 0, result.stderr
        assert checkpoint.is_file()
        state = torch.load(checkpoint, map_location="cpu")
        assert checkpoint_action_dim(state) == 1  # default residual mode: 1-D action
        assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    finally:
        checkpoint.unlink(missing_ok=True)

def test_train_and_test_hedge_band_action_mode_round_trip():
    # A band policy has a 2-D action: train-hedge must build matching critics,
    # and test-hedge must size the network from the checkpoint to load it.
    checkpoint = ROOT / "checkpoints" / "_smoke_test_policy_band.pth"
    try:
        result = run_cli(
            "train-hedge",
            "--n_episodes", "2", "--nb_predictions", "200", "--sampling", "20",
            "--warmup_steps", "0", "--update_every", "1", "--val_paths", "50",
            "--action_mode", "band", "--band_max", "0.2", "--fixed_tc", "0.01",
            "--checkpoint_path", str(checkpoint.relative_to(ROOT)),
        )
        assert result.returncode == 0, result.stderr
        assert checkpoint.is_file()
        assert checkpoint_action_dim(torch.load(checkpoint, map_location="cpu")) == 2

        result = run_cli(
            "test-hedge",
            "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20", "--fixed_tc", "0.01",
            "--checkpoint_path", str(checkpoint.relative_to(ROOT)),
        )
        assert result.returncode == 0, result.stderr
        _assert_sane_pnl_report(result.stdout)
    finally:
        checkpoint.unlink(missing_ok=True)

@pytest.mark.parametrize("policy", ["delta", "no_hedge"])
def test_test_hedge_baseline_policy_reports_sane_pnl_without_checkpoint(policy):
    # Baselines need no trained checkpoint, unlike the rl default.
    result = run_cli(
        "test-hedge", "--policy", policy,
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode == 0, result.stderr
    _assert_sane_pnl_report(result.stdout)

def test_test_hedge_invalid_policy_choice_is_rejected():
    result = run_cli(
        "test-hedge", "--policy", "bogus",
        "--n_episodes", "5", "--nb_predictions", "200", "--sampling", "20",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr.lower()

def test_train_ml_with_missing_dataset_fails_naming_the_remedy():
    result = run_cli("train-ml", "--dataset_path", "data/_does_not_exist.npz")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr and "_does_not_exist.npz" in result.stderr

def test_test_ml_with_missing_files_fails_naming_the_flag():
    result = run_cli("test-ml", "--dataset_path", "data/_does_not_exist.npz")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr and "--dataset_path" in result.stderr
