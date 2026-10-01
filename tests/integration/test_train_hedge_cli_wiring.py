"""
Tests for scripts/train_hedge.py::add_arguments()/run() -- specifically the
--lr/--alpha flags added to expose AmericanHedging.Hedge.train()'s existing
`lr`/`alpha` keyword arguments (previously stuck at their hardcoded defaults
with no CLI override, unlike --risk_lambda/--fixed_tc/--grad_steps_per_update
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

import numpy as np
import pytest

from core.gbm import GBM
from AmericanHedging.HedgeEnv import HedgingEnv

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
                    option_type="call", alpha_lr=None, residual_scale=0.1, action_mode="residual",
                    val_paths=0, val_every=50, val_metric="std", val_seed=None,
                    fixed_tc=0.0, prop_tc=0.001, band_max=0.2):
    # val_paths defaults to 0 (validation off) so the pre-existing tests keep
    # observing exactly one LSM/env (the training ones); the validation tests
    # below opt in explicitly.
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
        prop_tc=prop_tc,
        fixed_tc=fixed_tc,
        band_max=band_max,
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
        residual_scale=residual_scale,
        action_mode=action_mode,
        replay_buffer_size=1000,
        val_paths=val_paths,
        val_every=val_every,
        val_metric=val_metric,
        val_seed=val_seed,
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


@pytest.mark.parametrize("val_paths", [0, 20])
def test_run_passes_reward_mode_and_value_fn_presence_to_hedging_env(monkeypatch, val_paths):
    """End-to-end wiring check: the training HedgingEnv (the last one built;
    with validation on, the terminal-mode validation env is built before it)
    must receive reward_mode="accounting" with a non-None value_fn, and
    reward_mode="terminal" with value_fn=None."""
    captured = []
    real_env_cls = train_hedge_script.HedgingEnv

    class SpyHedgingEnv(real_env_cls):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(train_hedge_script, "HedgingEnv", SpyHedgingEnv)
    monkeypatch.setattr(train_hedge_script, "train", lambda *args, **kwargs: None)

    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, reward_mode="accounting", val_paths=val_paths))
    assert len(captured) == (2 if val_paths else 1)
    assert captured[-1]["reward_mode"] == "accounting"
    assert captured[-1]["value_fn"] is not None

    captured.clear()
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, reward_mode="terminal", val_paths=val_paths))
    assert captured[-1]["reward_mode"] == "terminal"
    assert captured[-1]["value_fn"] is None


@pytest.mark.parametrize("val_paths", [0, 20])
@pytest.mark.parametrize("option_type,expected", [("call", True), ("put", False)])
def test_run_enables_lsm_control_variate_only_for_calls(monkeypatch, option_type, expected, val_paths):
    """The European control variate removes plain LSM's spurious early
    exercise for calls but is worse out of sample for puts, so run() must
    pass control_variate=True for calls only -- for the validation LSM too,
    otherwise validation would score a different exercise rule than training."""
    seen = []
    real_lsm = train_hedge_script.LongstaffSchwartz

    def spy(*args, **kwargs):
        seen.append(kwargs["control_variate"])
        return real_lsm(*args, **kwargs)

    monkeypatch.setattr(train_hedge_script, "LongstaffSchwartz", spy)
    monkeypatch.setattr(train_hedge_script, "train", lambda *a, **k: None)

    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, option_type=option_type, val_paths=val_paths))

    assert seen == [expected] * (2 if val_paths else 1)


def test_add_arguments_residual_scale_default_0_1_and_parsed_as_float():
    args = _make_parser().parse_args([])
    assert args.residual_scale == pytest.approx(0.1)
    args = _make_parser().parse_args(["--residual_scale", "0"])
    assert args.residual_scale == 0.0 and isinstance(args.residual_scale, float)


def test_run_builds_policy_with_residual_scale_from_args(monkeypatch):
    """--residual_scale must reach RLPolicy (and thus the saved checkpoint),
    not just exist on the parser."""
    captured = {}

    def fake_train(policy, *a, **k):
        captured["scale"] = policy.residual_scale.item()

    monkeypatch.setattr(train_hedge_script, "train", fake_train)
    args = _make_run_args(lr=1e-4, alpha=0.2, residual_scale=0.37)
    train_hedge_script.run(args)

    assert captured["scale"] == pytest.approx(0.37)


def test_add_arguments_action_mode_default_and_choices():
    assert _make_parser().parse_args([]).action_mode == "residual"
    assert _make_parser().parse_args(["--action_mode", "gap_fraction"]).action_mode == "gap_fraction"
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--action_mode", "bogus"])


@pytest.mark.parametrize("mode,flag", [("residual", 0.0), ("gap_fraction", 1.0)])
def test_run_builds_policy_with_action_mode_from_args(monkeypatch, mode, flag):
    """--action_mode must reach RLPolicy (and thus the saved checkpoint)."""
    captured = {}

    def fake_train(policy, *a, **k):
        captured["flag"] = policy.gap_fraction.item()

    monkeypatch.setattr(train_hedge_script, "train", fake_train)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, action_mode=mode))

    assert captured["flag"] == flag


# --- validation-set checkpointing wiring -----------------------------------

def test_add_arguments_validation_flag_defaults_and_types():
    args = _make_parser().parse_args([])
    assert (args.val_paths, args.val_every, args.val_metric, args.val_seed) == (1000, 50, "std", None)
    args = _make_parser().parse_args(["--val_paths", "0", "--val_every", "7", "--val_metric", "cvar", "--val_seed", "5"])
    assert (args.val_paths, args.val_every, args.val_metric, args.val_seed) == (0, 7, "cvar", 5)


def test_add_arguments_rejects_unknown_val_metric():
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--val_metric", "mean"])


def _capture_train(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_hedge_script, "train", lambda *a, **k: captured.update(k))
    return captured


def test_run_forwards_validation_settings_and_builds_terminal_validation_env(monkeypatch):
    captured = _capture_train(monkeypatch)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, reward_mode="accounting", val_paths=20,
                                          val_every=7, val_metric="cvar", fixed_tc=0.25))
    assert (captured["val_paths"], captured["val_every"], captured["val_metric"]) == (20, 7, "cvar")
    val_env = captured["val_env"]
    assert isinstance(val_env, HedgingEnv)
    assert val_env.reward_mode == "terminal"  # so the metric is test-hedge's P&L
    assert val_env.fixed_tc == 0.25


def test_run_with_val_paths_zero_disables_validation(monkeypatch):
    captured = _capture_train(monkeypatch)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, val_paths=0))
    assert captured["val_env"] is None


@pytest.mark.parametrize("val_seed,expected_seed", [(None, 42 + 1000), (7, 7)])
def test_run_simulates_validation_paths_with_val_seed(monkeypatch, val_seed, expected_seed):
    captured = _capture_train(monkeypatch)
    args = _make_run_args(lr=1e-4, alpha=0.2, val_paths=20, val_seed=val_seed)
    train_hedge_script.run(args)

    expected = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=expected_seed).simulate_paths(50, 0.1, 20)
    np.testing.assert_array_equal(captured["val_env"].lsm.simulation, expected)


@pytest.mark.parametrize("val_paths", [0, 20])
def test_run_leaves_global_rng_at_the_training_seed_state(monkeypatch, val_paths):
    """GBM.simulate_paths reseeds global np.random. The validation simulation
    must therefore come first, so the stream train() then draws from (env.reset,
    warmup actions) is the --seed one, identical with or without validation."""
    draws = []
    monkeypatch.setattr(train_hedge_script, "train", lambda *a, **k: draws.append(np.random.random(3)))
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, val_paths=val_paths))

    # Reference: the global RNG right after the --seed simulation (GBM seeds
    # np.random and then draws its normals), with nothing reseeding afterwards.
    GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=42).simulate_paths(50, 0.1, 20)
    np.testing.assert_array_equal(draws[0], np.random.random(3))


# --- --prop_tc wiring ------------------------------------------------

def test_add_arguments_prop_tc_default_0_001_and_parsed_as_float():
    args = _make_parser().parse_args([])
    assert args.prop_tc == pytest.approx(0.001)
    args = _make_parser().parse_args(["--prop_tc", "0.02"])
    assert args.prop_tc == pytest.approx(0.02) and isinstance(args.prop_tc, float)


def test_run_passes_prop_tc_to_validation_env(monkeypatch):
    """Validation picks the checkpoint, so it must score the same cost regime
    as training. 0.0123 is distinct from HedgingEnv's 0.001 default so a
    dropped kwarg is caught."""
    captured = _capture_train(monkeypatch)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, val_paths=20, prop_tc=0.0123))
    assert captured["val_env"].tc == pytest.approx(0.0123)


def test_run_passes_prop_tc_to_training_env(monkeypatch):
    envs = []
    real_env_cls = train_hedge_script.HedgingEnv

    class SpyHedgingEnv(real_env_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            envs.append(self)

    monkeypatch.setattr(train_hedge_script, "HedgingEnv", SpyHedgingEnv)
    monkeypatch.setattr(train_hedge_script, "train", lambda *a, **k: None)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, val_paths=20, prop_tc=0.0123))

    assert len(envs) == 2  # validation env first, training env last
    assert [e.tc for e in envs] == [pytest.approx(0.0123)] * 2
    assert envs[-1].reward_mode == "accounting"  # the last one really is the training env


# --- band action mode wiring -------------------------------------------------

def test_add_arguments_accepts_band_action_mode_and_band_max_default_0_2():
    args = _make_parser().parse_args([])
    assert args.band_max == pytest.approx(0.2)
    args = _make_parser().parse_args(["--action_mode", "band", "--band_max", "0.35"])
    assert args.action_mode == "band" and args.band_max == pytest.approx(0.35)


def test_run_band_mode_builds_2d_policy_and_critics_with_band_max(monkeypatch):
    captured = {}

    def fake_train(policy, env, rb, critic1, critic2, **k):
        captured.update(policy=policy, critic1=critic1, critic2=critic2)

    monkeypatch.setattr(train_hedge_script, "train", fake_train)
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, action_mode="band", band_max=0.35))

    policy = captured["policy"]
    assert policy.action_mode == "band" and policy.action_dim == 2
    assert policy.band_max.item() == pytest.approx(0.35)
    for critic in (captured["critic1"], captured["critic2"]):
        assert critic.net[0].in_features == 5 + 2


def test_run_non_band_mode_keeps_1d_critics_and_ignores_band_max(monkeypatch):
    captured = {}
    monkeypatch.setattr(train_hedge_script, "train",
                        lambda policy, env, rb, c1, c2, **k: captured.update(policy=policy, critic=c1))
    train_hedge_script.run(_make_run_args(lr=1e-4, alpha=0.2, action_mode="gap_fraction", band_max=0.35))
    assert captured["policy"].action_dim == 1 and captured["policy"].band_max.item() == 0.0
    assert captured["critic"].net[0].in_features == 5 + 1
