import numpy as np
import pytest

from AmericanHedging.HedgeEnv import HedgingEnv

class FakeLSMModel:
    """Minimal stand-in for a fitted LongstaffSchwartz model, with hand-picked values."""
    def __init__(self):
        self.option_type = "call"
        self.strike = 100.0
        self.simulation = np.array([
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
        ])
        self.delta = np.zeros((2, 2))
        self.gamma = np.zeros((2, 2))
        self.exercise_time = np.array([2, 2])  # both paths run to maturity

def test_reset_returns_five_dim_state():
    env = HedgingEnv(FakeLSMModel())
    state = env.reset()
    assert state.shape == (5,)

def test_reset_starts_at_t_zero_with_no_position():
    env = HedgingEnv(FakeLSMModel())
    state = env.reset()
    assert state[1] == 0.0  # normalized time
    assert state[4] == 0.0  # current position

def test_step_reward_matches_manual_calculation(monkeypatch):
    env = HedgingEnv(FakeLSMModel())
    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # force path_idx = 0
    env.reset()

    _, reward, done = env.step(action=1.0)

    # path 0: S_t=100 -> S_t1=105, position 0 -> 1 (trade=1)
    # hedge_pnl = 0 * (105-100) = 0 (uses position BEFORE the update)
    # cost = 0.001 * |1| * 100 = 0.1
    # not done yet (t=1 < exercise_time=2), so reward = (hedge_pnl - cost) / strike
    assert reward == pytest.approx((0.0 - 0.1) / 100.0)
    assert not done

def test_episode_ends_at_exercise_time(monkeypatch):
    env = HedgingEnv(FakeLSMModel())
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=0.0)
    _, reward, done = env.step(action=0.0)

    assert done
    # at t=1 -> t=2 (maturity), S_t=105, S_t1=110, option payoff = max(110-100,0)=10
    expected_reward = (0.0 - 0.0 - 10.0) / 100.0
    assert reward == pytest.approx(expected_reward)

def test_action_is_clamped_to_valid_range(monkeypatch):
    env = HedgingEnv(FakeLSMModel())
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=5.0)  # way outside [-1, 1]
    assert env.position == 1.0
