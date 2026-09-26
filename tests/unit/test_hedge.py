import numpy as np
import pytest

from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.Hedge import evaluate
from AmericanHedging.baselines import delta_hedge, no_hedge


class FakeLSMModel:
    """Minimal stand-in for a fitted LongstaffSchwartz model, with hand-picked
    values (mirrors tests/unit/test_hedge_env.py's FakeLSMModel)."""

    def __init__(self, delta=None):
        self.option_type = "call"
        self.strike = 100.0
        self.simulation = np.array([
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
        ])
        self.delta = delta if delta is not None else np.zeros((2, 2))
        self.gamma = np.zeros((2, 2))
        self.exercise_time = np.array([2, 2])  # both paths run to maturity


def test_evaluate_returns_mean_std_and_cvar_matching_manual_calculation(monkeypatch):
    """With no_hedge and a fixed path, every episode is identical, so mean_pnl
    must equal that one reward, std_pnl must be 0, and cvar_95 (5th percentile)
    collapses to the same value too."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # force path_idx = 0
    env = HedgingEnv(FakeLSMModel())

    mean_pnl, std_pnl, cvar_95 = evaluate(no_hedge, env, n_episodes=4)

    # path 0, no_hedge (position stays 0 throughout):
    # step1: t=0->1, S 100->105, hedge_pnl=0, cost=0, not done -> reward=0
    # step2: t=1->2 (maturity), S 105->110, hedge_pnl=0, cost=0, done,
    #        call payoff = max(110-100,0)=10 -> reward=(0-0-10)/100=-0.1
    expected_reward = -0.1
    assert mean_pnl == pytest.approx(expected_reward)
    assert std_pnl == pytest.approx(0.0)
    assert cvar_95 == pytest.approx(expected_reward)


def test_evaluate_accepts_plain_callable_action_fn_not_tied_to_nn_module(monkeypatch):
    """Regression test: evaluate() used to call `policy(state).mean.item()`,
    coupling it to RLPolicy's Gaussian-actor interface. It must now drive any
    plain state -> float callable (e.g. a baseline with no .mean attribute)
    without raising, and call it once per environment step."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env = HedgingEnv(FakeLSMModel())

    calls = []

    def action_fn(state):
        calls.append(state.copy())
        return 0.0

    evaluate(action_fn, env, n_episodes=3)

    # exercise_time=2 for the fixed path -> 2 action_fn calls per episode
    assert len(calls) == 3 * 2


def test_evaluate_passes_current_env_state_to_action_fn(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env = HedgingEnv(FakeLSMModel())

    seen_states = []

    def action_fn(state):
        seen_states.append(state.copy())
        return 0.0

    evaluate(action_fn, env, n_episodes=1)

    # First call must see the reset state: S0/K=1.0, t/T=0, delta=gamma=0, position=0.
    np.testing.assert_allclose(seen_states[0], [1.0, 0.0, 0.0, 0.0, 0.0])
    # Second call: after step 1, t=1, S=105/100, t/n_steps=1/3, position now 0 (no_hedge action).
    np.testing.assert_allclose(seen_states[1], [1.05, 1 / 3, 0.0, 0.0, 0.0])


def test_evaluate_with_delta_hedge_baseline_hand_calculated_reward(monkeypatch):
    """End-to-end check that baselines.delta_hedge, evaluate(), and HedgingEnv
    compose correctly under the new action_fn interface, with a non-zero,
    time-varying delta driving the hedge position."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    delta = np.array([
        [0.5, -0.3],
        [0.4, 0.2],
    ])
    env = HedgingEnv(FakeLSMModel(delta=delta))

    mean_pnl, std_pnl, cvar_95 = evaluate(delta_hedge, env, n_episodes=1)

    # path 0:
    # t=0: state delta=0.5 -> action=0.5, position 0->0.5
    #   hedge_pnl = 0 (old position) * (105-100) = 0
    #   cost = 0.001 * |0.5-0| * 100 = 0.05
    #   not done -> reward1 = (0 - 0.05) / 100
    reward1 = (0.0 - 0.001 * 0.5 * 100.0) / 100.0
    # t=1: state delta=0.4 -> action=0.4, position 0.5->0.4
    #   hedge_pnl = 0.5 (old position) * (110-105) = 2.5
    #   cost = 0.001 * |0.4-0.5| * 105 = 0.0105
    #   done -> payoff = max(110-100,0) = 10
    #   reward2 = (2.5 - 0.0105 - 10) / 100
    reward2 = (0.5 * 5.0 - 0.001 * 0.1 * 105.0 - 10.0) / 100.0
    expected_total = reward1 + reward2

    assert mean_pnl == pytest.approx(expected_total)
    assert std_pnl == pytest.approx(0.0)
    assert cvar_95 == pytest.approx(expected_total)
