"""
Tests for scripts/test_hedge.py::run() exercised in-process (real GBM
simulation + LongstaffSchwartz fit + HedgingEnv wiring, all real; only
AmericanHedging.Hedge.evaluate is stubbed out so the test stays fast).

tests/integration/test_cli_smoke.py covers the CLI (subprocess) surface for
this same script; this file targets an invariant that can't be observed from
stdout alone: which pseudo-random state actually reaches evaluate().
"""
import argparse

import numpy as np
import pytest
import torch

import scripts.test_hedge as test_hedge_script
from AmericanHedging.baselines import delta_hedge, no_hedge
from AmericanHedging.RLPolicy import RLPolicy


def _make_args(policy, seed, prop_tc=0.001, band=0.05, band_target="delta"):
    return argparse.Namespace(
        sigma=0.2,
        start_price=100.0,
        maturity=0.1,
        sampling=20,
        nb_predictions=50,
        strike=100.0,
        risk_free_rate=0.05,
        option_type="call",
        max_degree=2,
        reg_type="simple",
        seed=seed,
        n_episodes=5,
        prop_tc=prop_tc,
        fixed_cost=0.0,
        checkpoint_path="unused.pth",
        policy=policy,
        band=band,
        band_target=band_target,
    )


def test_same_seed_gives_identical_random_state_entering_evaluate_across_policies(monkeypatch):
    """Regression test for the `np.random.seed(args.seed)` call added right
    before `evaluate(...)` in scripts/test_hedge.py::run(). Its whole purpose
    is that --policy delta and --policy no_hedge (which HedgingEnv.reset()
    samples paths for via the *global* np.random, not a seeded local
    generator) must see the identical draw sequence when --seed matches, so
    that comparing policies isn't just comparing path-sampling noise.

    This stubs out evaluate() itself (so the test doesn't need to actually
    run any episodes) and instead captures np.random.get_state() at the
    instant each run() call reaches evaluate() -- the two states must match
    bit-for-bit even though the two run() calls picked different --policy
    values and therefore executed different code paths beforehand.
    """
    captured_states = []

    def fake_evaluate(action_fn, env, n_episodes=1000):
        captured_states.append(np.random.get_state())
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)

    test_hedge_script.run(_make_args("delta", seed=123))
    test_hedge_script.run(_make_args("no_hedge", seed=123))

    assert len(captured_states) == 2
    algo_a, keys_a, pos_a = captured_states[0][:3]
    algo_b, keys_b, pos_b = captured_states[1][:3]
    assert algo_a == algo_b
    np.testing.assert_array_equal(keys_a, keys_b)
    assert pos_a == pos_b


def test_different_seeds_give_different_random_state_entering_evaluate(monkeypatch):
    """Sanity check on the test above: it isn't vacuously true that any two
    run() calls land on the same np.random state -- a different --seed must
    produce a different one."""
    captured_states = []

    def fake_evaluate(action_fn, env, n_episodes=1000):
        captured_states.append(np.random.get_state())
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)

    test_hedge_script.run(_make_args("no_hedge", seed=1))
    test_hedge_script.run(_make_args("no_hedge", seed=2))

    assert len(captured_states) == 2
    keys_1 = captured_states[0][1]
    keys_2 = captured_states[1][1]
    assert not np.array_equal(keys_1, keys_2)


def test_action_fn_matches_requested_policy(monkeypatch):
    """Confirms --policy delta/no_hedge wire up to the actual baseline
    functions from AmericanHedging.baselines, not e.g. both pointing at the
    same one by accident."""
    seen = {}

    def fake_evaluate(action_fn, env, n_episodes=1000):
        seen["action_fn"] = action_fn
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)

    test_hedge_script.run(_make_args("delta", seed=1))
    assert seen["action_fn"] is delta_hedge

    test_hedge_script.run(_make_args("no_hedge", seed=1))
    assert seen["action_fn"] is no_hedge


def test_action_fn_for_rl_policy_applies_tanh_squashing_to_mean(monkeypatch, tmp_path):
    """Regression test for scripts/test_hedge.py's --policy rl action_fn.

    It used to be `policy_model(state).mean.item()` -- the raw, unbounded
    distribution mean, safe only because HedgingEnv.step() clips it -- which
    silently disagreed with how AmericanHedging.Hedge.train()/_squash derive
    the deterministic action (tanh-squashed mean). It must now match: the
    action_fn built for --policy rl is `torch.tanh(policy_model(state).mean)`.
    """
    # Force the raw (pre-squash) mean far outside [-1, 1] for any input state,
    # by zeroing every weight/bias and then setting only layer3's mean output
    # bias to 5.0 (large enough that tanh(5.0) ~= 0.99991, clearly < 1, but
    # small enough to avoid float32 tanh saturating to exactly 1.0 the way it
    # would at very large magnitudes). This makes the pre-fix ("raw
    # .mean.item()") and post-fix (tanh-squashed) behaviors produce wildly
    # different, easily distinguished values (5.0 vs ~0.9999) instead of
    # relying on a random init that could coincidentally sit close to 0.
    policy = RLPolicy(5, 1)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.layer3.bias[0] = 5.0
    checkpoint_path = tmp_path / "policy.pth"
    torch.save(policy.state_dict(), checkpoint_path)

    seen = {}

    def fake_evaluate(action_fn, env, n_episodes=1000):
        seen["action_fn"] = action_fn
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)

    args = _make_args("rl", seed=1)
    args.checkpoint_path = str(checkpoint_path)
    test_hedge_script.run(args)

    state = np.array([1.0, 0.5, 0.1, 0.02, 0.0], dtype=np.float32)

    reloaded = RLPolicy(5, 1)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    with torch.no_grad():
        raw_mean = reloaded(torch.tensor(state, dtype=torch.float32)).mean
        expected = torch.tanh(raw_mean).item()

    actual = seen["action_fn"](state)
    assert actual == pytest.approx(expected)
    assert -1.0 < actual < 1.0  # would also catch a reversion to the unsquashed mean


@pytest.mark.parametrize("option_type,expected", [("call", True), ("put", False)])
def test_run_enables_lsm_control_variate_only_for_calls(monkeypatch, option_type, expected):
    # Same policy as train-hedge: the control variate helps calls, not puts.
    seen = {}
    real_lsm = test_hedge_script.LongstaffSchwartz

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_lsm(*args, **kwargs)

    monkeypatch.setattr(test_hedge_script, "LongstaffSchwartz", spy)
    monkeypatch.setattr(test_hedge_script, "evaluate", lambda *a, **k: (0.0, 0.0, 0.0))

    args = _make_args("delta", seed=1)
    args.option_type = option_type
    test_hedge_script.run(args)

    assert seen["control_variate"] is expected


def test_action_fn_for_rl_policy_applies_residual_on_delta_from_checkpoint(monkeypatch, tmp_path):
    """A checkpoint saved with residual_scale=c must make the rl action_fn
    return clip(delta + c*tanh(mean), -1, 1), even though run() builds the
    policy without being told c (it comes from the checkpoint)."""
    c = 0.2
    policy = RLPolicy(5, 1, residual_scale=c)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.layer3.bias[0] = 0.5  # tanh(0.5) ~ 0.462 for every state
    checkpoint_path = tmp_path / "policy.pth"
    torch.save(policy.state_dict(), checkpoint_path)

    seen = {}

    def fake_evaluate(action_fn, env, n_episodes=1000):
        seen["fn"] = action_fn
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)
    args = _make_args("rl", seed=1)
    args.checkpoint_path = str(checkpoint_path)
    test_hedge_script.run(args)

    t = float(np.tanh(0.5))
    for delta in (0.3, -0.4, 0.95):  # the last one hits the upper clamp
        state = np.array([1.0, 0.5, delta, 0.02, 0.0], dtype=np.float32)
        expected = float(np.clip(delta + c * t, -1.0, 1.0))
        assert seen["fn"](state) == pytest.approx(expected, abs=1e-6)


def test_run_passes_prop_tc_to_evaluation_env(monkeypatch):
    seen = {}

    def fake_evaluate(action_fn, env, n_episodes=1000):
        seen["tc"] = env.tc
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(test_hedge_script, "evaluate", fake_evaluate)
    test_hedge_script.run(_make_args("delta", seed=1, prop_tc=0.0123))
    assert seen["tc"] == pytest.approx(0.0123)  # distinct from the 0.001 default


def test_add_arguments_prop_tc_band_and_policy_choices():
    parser = argparse.ArgumentParser()
    test_hedge_script.add_arguments(parser)
    args = parser.parse_args([])
    assert args.prop_tc == pytest.approx(0.001)
    assert args.band == pytest.approx(0.05) and args.band_target == "delta"
    args = parser.parse_args(["--policy", "delta_band", "--band", "0.2", "--band_target", "edge",
                              "--prop_tc", "0.01"])
    assert (args.policy, args.band, args.band_target, args.prop_tc) == ("delta_band", 0.2, "edge", 0.01)
    with pytest.raises(SystemExit):
        parser.parse_args(["--band_target", "middle"])


@pytest.mark.parametrize("band_target", ["delta", "edge"])
def test_delta_band_policy_builds_delta_band_from_band_flags(monkeypatch, band_target):
    """The action_fn handed to evaluate() must behave as delta_band(band,
    band_target): a gap of 0.3 is outside band 0.1 (rebalance, to delta or the
    edge depending on target), but inside band 0.5 (hold)."""
    seen = {}
    monkeypatch.setattr(test_hedge_script, "evaluate",
                        lambda action_fn, env, n_episodes=1000: seen.update(fn=action_fn) or (0.0, 0.0, 0.0))

    state = np.array([1.0, 0.5, 0.5, 0.02, 0.8])  # delta 0.5, position 0.8
    test_hedge_script.run(_make_args("delta_band", seed=1, band=0.1, band_target=band_target))
    assert seen["fn"](state) == pytest.approx(0.5 if band_target == "delta" else 0.6)

    test_hedge_script.run(_make_args("delta_band", seed=1, band=0.5, band_target=band_target))
    assert seen["fn"](state) == 0.8
