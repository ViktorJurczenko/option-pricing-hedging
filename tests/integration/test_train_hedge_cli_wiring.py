"""
Tests for scripts/train_hedge.py::add_arguments()/run() -- specifically the
--lr/--alpha flags added to expose AmericanHedging.Hedge.train()'s existing
`lr`/`alpha` keyword arguments (previously stuck at their hardcoded defaults
with no CLI override, unlike --risk_lambda/--fixed_cost/--grad_steps_per_update
in the same file), plus the later --reward_mode/--gamma/--reward_scale/
--auto_alpha flags and run()'s accounting-mode value_fn wiring (see
AmericanHedging/option_value.py and HedgeEnv's reward_mode).

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


def test_add_arguments_registers_reward_mode_default_accounting():
    args = _make_parser().parse_args([])
    assert args.reward_mode == "accounting"


def test_add_arguments_registers_gamma_default_1_0():
    args = _make_parser().parse_args([])
    assert args.gamma == pytest.approx(1.0)


def test_add_arguments_registers_reward_scale_default_100_0():
    args = _make_parser().parse_args([])
    assert args.reward_scale == pytest.approx(100.0)


def test_add_arguments_registers_auto_alpha_default_true():
    args = _make_parser().parse_args([])
    assert args.auto_alpha is True


def test_add_arguments_no_auto_alpha_flag_sets_false():
    args = _make_parser().parse_args(["--no-auto_alpha"])
    assert args.auto_alpha is False


def test_add_arguments_registers_alpha_lr_default_none():
    args = _make_parser().parse_args([])
    assert args.alpha_lr is None


def test_add_arguments_alpha_lr_overridable_and_parsed_as_float():
    args = _make_parser().parse_args(["--alpha_lr", "0.05"])
    assert args.alpha_lr == pytest.approx(0.05)
    assert isinstance(args.alpha_lr, float)


def test_add_arguments_risk_lambda_default_changed_to_1_0():
    """Regression guard for the default change: a hedge trained with
    risk_lambda=0 has no incentive to hedge at all (every hedge ratio has
    about the same expected P&L; only the variance penalty rewards actually
    reducing it), which is exactly the plateau-on-no-hedge-baseline bug this
    whole change fixes."""
    args = _make_parser().parse_args([])
    assert args.risk_lambda == pytest.approx(1.0)


def _make_run_args(lr, alpha, reward_mode="accounting", gamma=1.0,
                    reward_scale=100.0, auto_alpha=True, risk_lambda=0.0,
                    option_type="call", alpha_lr=None):
    return argparse.Namespace(
        sigma=0.2,
        start_price=100.0,
        risk_free_rate=0.05,
        seed=42,
        nb_predictions=50,
        maturity=0.1,
        sampling=20,
        strike=100.0,
        option_type=option_type,
        reg_type="simple",
        max_degree=2,
        fixed_cost=0.0,
        risk_lambda=risk_lambda,
        reward_mode=reward_mode,
        gamma=gamma,
        reward_scale=reward_scale,
        auto_alpha=auto_alpha,
        n_episodes=1,
        warmup_steps=0,
        update_every=10,
        grad_steps_per_update=1,
        lr=lr,
        alpha=alpha,
        alpha_lr=alpha_lr,
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


def test_run_forwards_gamma_reward_scale_and_auto_alpha_to_train(monkeypatch):
    """Regression test for the same class of bug as --lr/--alpha above: the
    later --gamma/--reward_scale/--auto_alpha flags must actually reach
    Hedge.train(), not just exist on the parser."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=1e-4, alpha=0.2, gamma=0.5, reward_scale=25.0, auto_alpha=False)
    train_hedge_script.run(args)

    assert captured_kwargs["gamma"] == pytest.approx(0.5)
    assert captured_kwargs["reward_scale"] == pytest.approx(25.0)
    assert captured_kwargs["auto_alpha"] is False


def test_run_forwards_default_gamma_reward_scale_and_auto_alpha_distinctly(monkeypatch):
    """Sanity check on the test above: confirms the assertion isn't
    vacuously true because run() happens to always pass Hedge.train()'s
    hardcoded/CLI defaults regardless of what args carries."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=1e-4, alpha=0.2)  # gamma=1.0, reward_scale=100.0, auto_alpha=True
    train_hedge_script.run(args)
    assert captured_kwargs["gamma"] == pytest.approx(1.0)
    assert captured_kwargs["reward_scale"] == pytest.approx(100.0)
    assert captured_kwargs["auto_alpha"] is True


def test_run_forwards_alpha_lr_to_train(monkeypatch):
    """Regression test for the same class of bug as --lr/--alpha above:
    --alpha_lr must actually reach Hedge.train(), not just exist on the
    parser. args.alpha_lr defaults to None (see _make_run_args), so run()
    would raise AttributeError before this if the field were missing --
    checked here for a non-default explicit value."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=1e-4, alpha=0.2, alpha_lr=0.03)
    train_hedge_script.run(args)

    assert captured_kwargs["alpha_lr"] == pytest.approx(0.03)


def test_run_forwards_default_alpha_lr_of_none(monkeypatch):
    """Sanity check on the test above: confirms run() forwards whatever
    args.alpha_lr carries, including the None default, rather than always
    passing a hardcoded value regardless of args."""
    captured_kwargs = {}

    def fake_train(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)

    args = _make_run_args(lr=1e-4, alpha=0.2)  # alpha_lr=None by default
    train_hedge_script.run(args)

    assert captured_kwargs["alpha_lr"] is None


def test_run_terminal_mode_does_not_build_a_value_fn(monkeypatch):
    """--reward_mode terminal must not call make_option_value_fn at all --
    building the BS/tree value_fn is pointless work for a mode that never
    uses it, and doing so unconditionally would silently change behavior for
    puts (the tree build is the slow path) even though terminal mode's
    reward math never reads it."""
    calls = []

    def fake_make_option_value_fn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("make_option_value_fn must not be called in terminal mode")

    monkeypatch.setattr(train_hedge_script, "make_option_value_fn", fake_make_option_value_fn)
    monkeypatch.setattr(train_hedge_script, "train", lambda *args, **kwargs: None)

    args = _make_run_args(lr=1e-4, alpha=0.2, reward_mode="terminal")
    train_hedge_script.run(args)  # must not raise

    assert calls == []


def test_run_accounting_mode_builds_a_value_fn_via_make_option_value_fn(monkeypatch):
    calls = []
    original = train_hedge_script.make_option_value_fn

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_hedge_script, "make_option_value_fn", spy)
    monkeypatch.setattr(train_hedge_script, "train", lambda *args, **kwargs: None)

    args = _make_run_args(lr=1e-4, alpha=0.2, reward_mode="accounting")
    train_hedge_script.run(args)

    assert len(calls) == 1


def test_run_passes_reward_mode_and_value_fn_presence_to_hedging_env(monkeypatch):
    """End-to-end wiring check: HedgingEnv must actually receive
    reward_mode="accounting" with a non-None value_fn, and
    reward_mode="terminal" with value_fn=None -- not just that run() calls
    make_option_value_fn somewhere (checked separately above)."""
    captured = {}
    real_env_cls = train_hedge_script.HedgingEnv

    class SpyHedgingEnv(real_env_cls):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(train_hedge_script, "HedgingEnv", SpyHedgingEnv)
    monkeypatch.setattr(train_hedge_script, "train", lambda *args, **kwargs: None)

    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, reward_mode="accounting"))
    assert captured["reward_mode"] == "accounting"
    assert captured["value_fn"] is not None

    captured.clear()
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, reward_mode="terminal"))
    assert captured["reward_mode"] == "terminal"
    assert captured["value_fn"] is None
