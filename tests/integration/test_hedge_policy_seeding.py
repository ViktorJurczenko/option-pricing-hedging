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

import scripts.test_hedge as test_hedge_script
from AmericanHedging.baselines import delta_hedge, no_hedge


def _make_args(policy, seed):
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
        fixed_cost=0.0,
        checkpoint_path="unused.pth",
        policy=policy,
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
