"""
Tests for scripts/train_hedge.py::add_arguments()/run() -- specifically the
--lr/--alpha flags added to expose AmericanHedging.Hedge.train()'s existing
`lr`/`alpha` keyword arguments (previously stuck at their hardcoded defaults
with no CLI override, unlike --risk_lambda/--fixed_cost/--grad_steps_per_update
in the same file).

Mirrors tests/integration/test_hedge_policy_seeding.py's approach for the
sibling script (scripts/test_hedge.py): run() is exercised with real GBM
simulation + LongstaffSchwartz fit + env/policy/critic construction (all real,
kept small/fast via tiny nb_predictions/sampling/maturity), while the actual
training loop (AmericanHedging.Hedge.train) is stubbed out so the test stays
fast and can assert on exactly what run() passed it.
"""
import argparse

import pytest

import scripts.train_hedge as train_hedge_script


def _make_parser():
    parser = argparse.ArgumentParser()
    train_hedge_script.add_arguments(parser)
    return parser


def test_add_arguments_registers_lr_with_default_1e_minus_4():
    args = _make_parser().parse_args([])
    assert args.lr == pytest.approx(1e-4)


def test_add_arguments_registers_alpha_with_default_0_2():
    args = _make_parser().parse_args([])
    assert args.alpha == pytest.approx(0.2)


def test_add_arguments_lr_and_alpha_are_overridable_and_parsed_as_float():
    args = _make_parser().parse_args(["--lr", "3e-4", "--alpha", "0.05"])
    assert args.lr == pytest.approx(3e-4)
    assert args.alpha == pytest.approx(0.05)
    assert isinstance(args.lr, float)
    assert isinstance(args.alpha, float)


def _make_run_args(lr, alpha):
    return argparse.Namespace(
        sigma=0.2,
        start_price=100.0,
        risk_free_rate=0.05,
        seed=42,
        nb_predictions=50,
        maturity=0.1,
        sampling=20,
        strike=100.0,
        option_type="call",
        reg_type="simple",
        max_degree=2,
        ridge_lambda=0.06,
        moneyness_bins=30,
        moneyness_bins_window=0.3,
        fixed_cost=0.0,
        risk_lambda=0.0,
        n_episodes=1,
        warmup_steps=0,
        update_every=10,
        grad_steps_per_update=1,
        lr=lr,
        alpha=alpha,
        replay_buffer_size=1000,
        checkpoint_path="unused.pth",
    )


def test_run_forwards_lr_and_alpha_to_train(monkeypatch):
    """Regression test for the exact bug this change fixes: --lr/--alpha
    existing as CLI flags but never reaching the train() call. Stubs out
    train() and asserts run() actually passed args.lr/args.alpha through as
    keyword arguments, rather than silently dropping them and leaving
    Hedge.train()'s hardcoded defaults (1e-4/0.2) in effect."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=9.5e-3, alpha=0.77)
    train_hedge_script.run(args)

    assert captured_kwargs["lr"] == pytest.approx(9.5e-3)
    assert captured_kwargs["alpha"] == pytest.approx(0.77)


def test_run_forwards_non_default_lr_and_alpha_distinctly_from_defaults(monkeypatch):
    """Sanity check on the test above: confirms the assertion isn't
    vacuously true because run() happens to always pass Hedge.train()'s own
    hardcoded defaults (1e-4/0.2) regardless of what args carries."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=1e-4, alpha=0.2)  # matches Hedge.train()'s defaults
    train_hedge_script.run(args)
    assert captured_kwargs["lr"] == pytest.approx(1e-4)
    assert captured_kwargs["alpha"] == pytest.approx(0.2)

    captured_kwargs.clear()
    args = _make_run_args(lr=0.5, alpha=0.5)  # deliberately far from either default
    train_hedge_script.run(args)
    assert captured_kwargs["lr"] == pytest.approx(0.5)
    assert captured_kwargs["alpha"] == pytest.approx(0.5)
