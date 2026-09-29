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


class EarlyExerciseLSMModel:
    """Single-path stand-in with 4 simulation rows where the path exercises
    at t=2, strictly before the final row (t=3) -- used to check the
    accounting/terminal telescoping identity past a mid-episode early
    exercise, not just at the last simulated row."""
    def __init__(self):
        self.option_type = "put"
        self.strike = 100.0
        self.simulation = np.array([
            [100.0],
            [95.0],
            [90.0],
            [85.0],
        ])
        self.delta = np.zeros((3, 1))
        self.gamma = np.zeros((3, 1))
        self.exercise_time = np.array([2])

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

def test_step_charges_fixed_cost_on_nonzero_trade(monkeypatch):
    env = HedgingEnv(FakeLSMModel(), fixed_cost=0.05)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    _, reward, done = env.step(action=1.0)

    # path 0: S_t=100 -> S_t1=105, position 0 -> 1 (trade=1, nonzero -> fixed cost charged)
    # hedge_pnl = 0 * (105-100) = 0
    # cost = 0.001 * |1| * 100 (proportional) + 0.05 (fixed) = 0.15
    expected_cost = 0.001 * 1.0 * 100.0 + 0.05
    assert reward == pytest.approx((0.0 - expected_cost) / 100.0)
    assert not done

def test_step_no_fixed_cost_when_trade_size_is_zero(monkeypatch):
    """A trade of exactly 0 (action == current position) must not incur the
    flat fixed_cost -- it's charged per trade, not per step with a nonzero
    position held."""
    env = HedgingEnv(FakeLSMModel(), fixed_cost=0.05)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=1.0)                       # first trade: nonzero -> fixed cost charged
    _, reward, done = env.step(action=1.0)      # same position again -> trade=0

    # path 0 step2: S_t=105 -> S_t1=110, position stays 1 (trade=0)
    # hedge_pnl = 1 * (110-105) = 5
    # cost = 0.001 * |0| * 105 + 0 (no fixed cost: trade size is 0) = 0
    # done (t=2 == exercise_time=2); call payoff = max(110-100,0) = 10
    expected_reward = (5.0 - 0.0 - 10.0) / 100.0
    assert reward == pytest.approx(expected_reward)
    assert done

def test_fixed_cost_applies_at_final_exercise_step(monkeypatch):
    """fixed_cost isn't special-cased away on the terminal step: it stacks
    with the option-payoff subtraction in the `done` branch."""
    model = FakeLSMModel()
    model.exercise_time = np.array([1, 1])  # exercise after a single step
    env = HedgingEnv(model, fixed_cost=0.05)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    _, reward, done = env.step(action=1.0)

    # path 0: S_t=100 -> S_t1=105, trade=1 (nonzero) -> fixed cost charged
    # hedge_pnl = 0 * (105-100) = 0
    # cost = 0.001 * |1| * 100 + 0.05 = 0.15
    # done at t=1 == exercise_time=1; call payoff = max(105-100,0) = 5
    expected_reward = (0.0 - 0.15 - 5.0) / 100.0
    assert reward == pytest.approx(expected_reward)
    assert done


# --- reward_mode="accounting" -----------------------------------------------

def test_invalid_reward_mode_raises():
    with pytest.raises(ValueError, match="reward_mode"):
        HedgingEnv(FakeLSMModel(), reward_mode="bogus")


def test_accounting_mode_without_value_fn_raises():
    with pytest.raises(ValueError, match="value_fn"):
        HedgingEnv(FakeLSMModel(), reward_mode="accounting")


def test_accounting_mode_precomputes_option_values_grid_from_value_fn():
    """The precomputed self.option_values[t, i] must equal value_fn(t, sim[t])[i]
    for every row/path -- a wrong axis or an off-by-one in the precompute
    loop would silently mis-price every step's reward without raising."""
    def value_fn(t, S):
        return t + 0.01 * S

    model = FakeLSMModel()
    env = HedgingEnv(model, reward_mode="accounting", value_fn=value_fn)

    expected = np.stack([value_fn(t, model.simulation[t]) for t in range(model.simulation.shape[0])])
    np.testing.assert_allclose(env.option_values, expected)


def test_accounting_reward_matches_hand_calculation(monkeypatch):
    """Hand-calculated check of the accounting reward formula:
    (hedge_pnl - cost - (V_{t+1} - V_t)) / K, on a non-terminal step."""
    def value_fn(t, S):
        return 0.1 * S  # simple, easy-to-hand-verify deterministic value

    env = HedgingEnv(FakeLSMModel(), reward_mode="accounting", value_fn=value_fn)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # force path_idx = 0
    env.reset()

    _, reward, done = env.step(action=1.0)

    # path 0: S_t=100 -> S_t1=105, position 0 -> 1 (trade=1)
    # hedge_pnl = 0 * (105-100) = 0; cost = 0.001 * 1 * 100 = 0.1
    # V_t = value_fn(0, 100) = 10.0, V_t1 = value_fn(1, 105) = 10.5 (not done)
    expected_reward = (0.0 - 0.1 - (10.5 - 10.0)) / 100.0
    assert reward == pytest.approx(expected_reward)
    assert not done


def test_accounting_reward_on_exercise_step_replaces_v_t1_with_payoff(monkeypatch):
    """On the done/exercise step, V_{t+1} in the accounting formula is
    replaced by the option's actual payoff, not the precomputed
    option_values lookup for that row."""
    def value_fn(t, S):
        return 0.1 * S

    env = HedgingEnv(FakeLSMModel(), reward_mode="accounting", value_fn=value_fn)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=0.0)
    _, reward, done = env.step(action=0.0)

    # t=1 -> t=2 (maturity): S_t=105 -> S_t1=110, hedge_pnl=0, cost=0
    # V_t = value_fn(1, 105) = 10.5; done -> V_t1 = payoff = max(110-100,0) = 10
    expected_reward = (0.0 - 0.0 - (10.0 - 10.5)) / 100.0
    assert reward == pytest.approx(expected_reward)
    assert done


def test_accounting_episode_sum_telescopes_to_terminal_sum_plus_v0_over_strike(monkeypatch):
    """Key invariant (checked by the implementer to 1e-15 on real LSM envs):
    for any fixed action sequence, sum(accounting rewards) == sum(terminal
    rewards) + option_values[0, path]/K. Uses EarlyExerciseLSMModel so the
    identity is checked across an episode that ends in early exercise
    (t=2), not just at the final simulated row."""
    def value_fn(t, S):
        return 0.1 * S + t  # arbitrary but deterministic per-row function

    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # only one path anyway
    actions = [0.4, -0.2]

    env_terminal = HedgingEnv(EarlyExerciseLSMModel(), reward_mode="terminal")
    env_terminal.reset()
    total_terminal = 0.0
    done = False
    step = 0
    while not done:
        _, reward, done = env_terminal.step(actions[step])
        total_terminal += reward
        step += 1

    env_accounting = HedgingEnv(EarlyExerciseLSMModel(), reward_mode="accounting", value_fn=value_fn)
    env_accounting.reset()
    total_accounting = 0.0
    done = False
    step = 0
    while not done:
        _, reward, done = env_accounting.step(actions[step])
        total_accounting += reward
        step += 1

    v0_over_k = value_fn(0, np.array([100.0]))[0] / 100.0
    assert total_accounting == pytest.approx(total_terminal + v0_over_k, abs=1e-9)
