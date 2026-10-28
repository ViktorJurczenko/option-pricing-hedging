import numpy as np
import pytest

from optionpricing.hedging.env import HedgingEnv
from tests.helpers import make_fake_lsm, two_path_call_lsm


def early_exercise_put_lsm():
    """Single put path with 4 rows, exercised at t=2, strictly before the
    final row (t=3) -- used to check the accounting/terminal telescoping
    identity past a mid-episode early exercise, not just at the last row."""
    return make_fake_lsm([[100.0], [95.0], [90.0], [85.0]], [2], option_type="put")


def test_reset_returns_five_dim_state():
    env = HedgingEnv(two_path_call_lsm())
    state = env.reset()
    assert state.shape == (5,)


def test_reset_starts_at_t_zero_with_no_position():
    env = HedgingEnv(two_path_call_lsm())
    state = env.reset()
    assert state[1] == 0.0  # normalized time
    assert state[4] == 0.0  # current position


def test_step_reward_matches_manual_calculation(monkeypatch):
    env = HedgingEnv(two_path_call_lsm())
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
    env = HedgingEnv(two_path_call_lsm())
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=0.0)
    _, reward, done = env.step(action=0.0)

    assert done
    # at t=1 -> t=2 (maturity), S_t=105, S_t1=110, option payoff = max(110-100,0)=10
    expected_reward = (0.0 - 0.0 - 10.0) / 100.0
    assert reward == pytest.approx(expected_reward)


def test_action_is_clamped_to_valid_range(monkeypatch):
    env = HedgingEnv(two_path_call_lsm())
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=5.0)  # way outside [-1, 1]
    assert env.position == 1.0


def test_step_charges_fixed_tc_on_nonzero_trade(monkeypatch):
    env = HedgingEnv(two_path_call_lsm(), fixed_tc=0.05)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    _, reward, done = env.step(action=1.0)

    # path 0: S_t=100 -> S_t1=105, position 0 -> 1 (trade=1, nonzero -> fixed cost charged)
    # hedge_pnl = 0 * (105-100) = 0
    # cost = 0.001 * |1| * 100 (proportional) + 0.05 (fixed) = 0.15
    expected_cost = 0.001 * 1.0 * 100.0 + 0.05
    assert reward == pytest.approx((0.0 - expected_cost) / 100.0)
    assert not done


def test_step_no_fixed_tc_when_trade_size_is_zero(monkeypatch):
    """A trade of exactly 0 (action == current position) must not incur the
    flat fixed_tc -- it's charged per trade, not per step with a nonzero
    position held."""
    env = HedgingEnv(two_path_call_lsm(), fixed_tc=0.05)
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env.reset()

    env.step(action=1.0)  # first trade: nonzero -> fixed cost charged
    _, reward, done = env.step(action=1.0)  # same position again -> trade=0

    # path 0 step2: S_t=105 -> S_t1=110, position stays 1 (trade=0)
    # hedge_pnl = 1 * (110-105) = 5
    # cost = 0.001 * |0| * 105 + 0 (no fixed cost: trade size is 0) = 0
    # done (t=2 == exercise_time=2); call payoff = max(110-100,0) = 10
    expected_reward = (5.0 - 0.0 - 10.0) / 100.0
    assert reward == pytest.approx(expected_reward)
    assert done


def test_fixed_tc_applies_at_final_exercise_step(monkeypatch):
    """fixed_tc isn't special-cased away on the terminal step: it stacks
    with the option-payoff subtraction in the `done` branch."""
    model = two_path_call_lsm()
    model.exercise_time = np.array([1, 1])  # exercise after a single step
    env = HedgingEnv(model, fixed_tc=0.05)
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
        HedgingEnv(two_path_call_lsm(), reward_mode="bogus")


def test_accounting_mode_without_value_fn_raises():
    with pytest.raises(ValueError, match="value_fn"):
        HedgingEnv(two_path_call_lsm(), reward_mode="accounting")


def test_accounting_mode_precomputes_option_values_grid_from_value_fn():
    """The precomputed self.option_values[t, i] must equal value_fn(t, sim[t])[i]
    for every row/path -- a wrong axis or an off-by-one in the precompute
    loop would silently mis-price every step's reward without raising."""

    def value_fn(t, S):
        return t + 0.01 * S

    model = two_path_call_lsm()
    env = HedgingEnv(model, reward_mode="accounting", value_fn=value_fn)

    expected = np.stack([value_fn(t, model.simulation[t]) for t in range(model.simulation.shape[0])])
    np.testing.assert_allclose(env.option_values, expected)


def test_accounting_reward_matches_hand_calculation(monkeypatch):
    """Hand-calculated check of the accounting reward formula:
    (hedge_pnl - cost - (V_{t+1} - V_t)) / K, on a non-terminal step."""

    def value_fn(t, S):
        return 0.1 * S  # simple, easy-to-hand-verify deterministic value

    env = HedgingEnv(two_path_call_lsm(), reward_mode="accounting", value_fn=value_fn)
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

    env = HedgingEnv(two_path_call_lsm(), reward_mode="accounting", value_fn=value_fn)
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
    rewards) + option_values[0, path]/K. Uses early_exercise_put_lsm so the
    identity is checked across an episode that ends in early exercise
    (t=2), not just at the final simulated row."""

    def value_fn(t, S):
        return 0.1 * S + t  # arbitrary but deterministic per-row function

    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # only one path anyway
    actions = [0.4, -0.2]

    env_terminal = HedgingEnv(early_exercise_put_lsm(), reward_mode="terminal")
    env_terminal.reset()
    total_terminal = 0.0
    done = False
    step = 0
    while not done:
        _, reward, done = env_terminal.step(actions[step])
        total_terminal += reward
        step += 1

    env_accounting = HedgingEnv(early_exercise_put_lsm(), reward_mode="accounting", value_fn=value_fn)
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


# --- reset(path_idx), _build_states, rollout_pnl --------------------------


def random_lsm(option_type="put", n_rows=8, n_paths=12, seed=0, row0_paths=(), strike=100.0):
    """Fake fitted LSM with random (seeded) paths, Greeks and several distinct
    exercise times, so paths drop out of a vectorized rollout at different steps."""
    rng = np.random.default_rng(seed)
    steps = np.exp(0.03 * rng.standard_normal((n_rows - 1, n_paths)))
    exercise_time = np.arange(n_paths) % (n_rows - 1) + 1  # 1..n_rows-1, all present
    exercise_time[list(row0_paths)] = 0  # exercised at row 0: zero-length episodes
    return make_fake_lsm(
        100.0 * np.vstack([np.ones(n_paths), np.cumprod(steps, axis=0)]),
        exercise_time,
        option_type=option_type,
        strike=strike,
        delta=rng.uniform(-1, 1, (n_rows - 1, n_paths)),
        gamma=rng.uniform(0, 0.1, (n_rows - 1, n_paths)),
    )


def _state_dependent_scalar_hedge(state):
    # Uses every state component; sometimes returns exactly the held position
    # (state[4]) so the no-fixed-cost branch is hit, and sometimes leaves [-1, 1].
    if state[1] > 0.3 and state[1] < 0.5:
        return state[4]
    return 2.5 * state[2] + 0.7 * (state[0] - 1.0) * 10 - 0.3 * state[4]


def _batch_hedge(states):
    return np.array([_state_dependent_scalar_hedge(s) for s in states])


def _sum_step_rewards(env, path_idx, hedge_fn):
    state = env.reset(path_idx)
    total, done = 0.0, False
    while not done:
        state, reward, done = env.step(hedge_fn(state))
        total += reward
    return total


def test_reset_with_path_idx_uses_that_path_and_does_not_draw_random(monkeypatch):
    env = HedgingEnv(two_path_call_lsm())

    def boom(n):
        raise AssertionError("np.random.randint must not be called")

    monkeypatch.setattr(np.random, "randint", boom)
    env.reset(1)
    assert env.path_idx == 1
    assert env.t == 0 and env.position == 0.0


def test_reset_with_none_still_draws_random_path(monkeypatch):
    env = HedgingEnv(two_path_call_lsm())
    monkeypatch.setattr(np.random, "randint", lambda n: 1)
    env.reset()
    assert env.path_idx == 1


def test_reset_path_idx_zero_is_not_treated_as_none(monkeypatch):
    # `path_idx or random` style bugs would send 0 to the random branch.
    env = HedgingEnv(two_path_call_lsm())
    monkeypatch.setattr(np.random, "randint", lambda n: 1)
    env.reset(0)
    assert env.path_idx == 0


def test_build_states_scalar_matches_hand_computed_state():
    lsm = two_path_call_lsm()
    lsm.delta = np.array([[0.1, 0.2], [0.3, 0.4]])
    lsm.gamma = np.array([[0.01, 0.02], [0.03, 0.04]])
    env = HedgingEnv(lsm)
    state = env._build_states(1, 1, 0.5)
    assert state.shape == (5,)
    np.testing.assert_allclose(state, [95.0 / 100.0, 1 / 2, 0.4, 0.04, 0.5])


def test_build_states_batch_rows_equal_scalar_calls():
    env = HedgingEnv(random_lsm())
    idx = np.array([3, 0, 7, 7])
    pos = np.array([0.2, -0.5, 1.0, 0.0])
    batch = env._build_states(4, idx, pos)
    assert batch.shape == (4, 5)
    for k in range(4):
        np.testing.assert_array_equal(batch[k], env._build_states(4, idx[k], pos[k]))


def test_build_states_clamps_greeks_to_last_row_at_maturity():
    lsm = random_lsm()
    env = HedgingEnv(lsm)
    last = lsm.simulation.shape[0] - 1  # greeks only have `last` rows
    state = env._build_states(last, np.array([2, 5]), np.zeros(2))
    np.testing.assert_array_equal(state[:, 2], lsm.delta[-1, [2, 5]])
    np.testing.assert_array_equal(state[:, 3], lsm.gamma[-1, [2, 5]])
    np.testing.assert_allclose(state[:, 1], 1.0)


def _rollout_pnl_and_summed_step_rewards(option_type, tc, fixed_tc, reward_mode, row0_paths=()):
    """rollout_pnl on every path of a random_lsm, the per-path sums of
    terminal-mode step() rewards, and the model."""
    # Start at 100: strike 95 (call) / 105 (put) makes the row-0 payoff 5, not 0
    strike = 95.0 if option_type == "call" else 105.0
    lsm = random_lsm(option_type, row0_paths=row0_paths, strike=strike)
    n_paths = lsm.simulation.shape[1]
    kwargs = {}
    if reward_mode == "accounting":
        kwargs = dict(reward_mode="accounting", value_fn=lambda t, S: np.full_like(S, 7.0 + t))
    env = HedgingEnv(lsm, prop_tc=tc, fixed_tc=fixed_tc, **kwargs)
    ref_env = HedgingEnv(lsm, prop_tc=tc, fixed_tc=fixed_tc)  # terminal

    idxs = np.arange(n_paths)
    got = env.rollout_pnl(_batch_hedge, idxs)
    expected = np.array([_sum_step_rewards(ref_env, i, _state_dependent_scalar_hedge) for i in idxs])
    # the scripted hedge must actually leave [-1, 1] and hold, else coverage is vacuous
    assert np.abs(_batch_hedge(env._build_states(0, idxs, np.zeros(n_paths)))).max() > 1
    return got, expected, lsm


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("tc,fixed_tc", [(0.001, 0.0), (0.0, 0.5), (0.002, 0.3)])
@pytest.mark.parametrize("reward_mode", ["terminal", "accounting"])
def test_rollout_pnl_equals_summed_terminal_step_rewards(option_type, tc, fixed_tc, reward_mode):
    """The vectorized P&L duplicates step()'s arithmetic, so it must match the
    per-path episode sums (terminal mode) on every path, even with early
    exercise at different steps, clipped hedges and exact holds."""
    got, expected, lsm = _rollout_pnl_and_summed_step_rewards(option_type, tc, fixed_tc, reward_mode)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert len(set(lsm.exercise_time)) > 3  # the early-exercise coverage is real


@pytest.mark.parametrize("option_type,reward_mode", [("call", "terminal"), ("put", "accounting")])
def test_rollout_pnl_equals_summed_step_rewards_with_paths_exercised_at_row0(option_type, reward_mode):
    # Regression (BUGS.md, "HedgingEnv mishandles options exercised at row 0"):
    # rollout_pnl used to return 0 for these paths and step() used to take a
    # hedging step. The ITM strike makes the payoff real (5 / K).
    got, expected, lsm = _rollout_pnl_and_summed_step_rewards(option_type, 0.002, 0.3, reward_mode, row0_paths=(0, 5))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert (lsm.exercise_time == 0).sum() == 2
    np.testing.assert_allclose(got[[0, 5]], -5.0 / lsm.strike, atol=1e-12)


def test_rollout_pnl_respects_path_order_and_subset():
    env = HedgingEnv(random_lsm())
    full = env.rollout_pnl(_batch_hedge, np.arange(12))
    sub = env.rollout_pnl(_batch_hedge, np.array([9, 2, 5]))
    np.testing.assert_allclose(sub, full[[9, 2, 5]], atol=1e-12)


def test_rollout_pnl_does_not_touch_rng_or_episode_state(monkeypatch):
    env = HedgingEnv(random_lsm())
    env.reset(3)
    env.step(0.4)
    before = (env.t, env.path_idx, env.position)
    np.random.seed(123)
    rng_before = np.random.get_state()[1].copy()
    env.rollout_pnl(_batch_hedge, np.arange(12))
    assert (env.t, env.path_idx, env.position) == before
    np.testing.assert_array_equal(np.random.get_state()[1], rng_before)


def test_rollout_pnl_fixed_tc_not_charged_when_position_held():
    # Holding a constant position 0.5 after the first trade: fixed cost is paid once only.
    lsm = random_lsm("call")
    env = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=1.0)
    free = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=0.0)
    hold = lambda s: np.full(len(s), 0.5)
    idxs = np.arange(12)
    diff = free.rollout_pnl(hold, idxs) - env.rollout_pnl(hold, idxs)
    np.testing.assert_allclose(diff, 1.0 / lsm.strike, atol=1e-12)


# --- _trading_cost / _payoff helpers (shared by step() and rollout_pnl()) ---


def test_trading_cost_scalar_combines_proportional_and_fixed():
    env = HedgingEnv(two_path_call_lsm(), prop_tc=0.01, fixed_tc=0.5)
    # 0.01 * |-2| * 100 + 0.5 = 2.5 ; sign of the trade must not matter
    assert env._trading_cost(-2.0, 100.0) == pytest.approx(2.5)
    assert env._trading_cost(2.0, 100.0) == pytest.approx(2.5)


def test_trading_cost_array_is_elementwise():
    env = HedgingEnv(two_path_call_lsm(), prop_tc=0.01, fixed_tc=0.5)
    trades = np.array([1.0, -2.0, 0.0])
    spots = np.array([100.0, 50.0, 80.0])
    # [1.0 + 0.5, 1.0 + 0.5, 0 + 0 (no trade -> no fixed cost)]
    np.testing.assert_allclose(env._trading_cost(trades, spots), [1.5, 1.5, 0.0])


def test_trading_cost_fixed_tc_threshold_at_trade_eps():
    env = HedgingEnv(two_path_call_lsm(), prop_tc=0.0, fixed_tc=1.0)
    eps = HedgingEnv._TRADE_EPS
    # at or below eps is float noise (strict ">"): no fixed cost; above it: charged
    assert env._trading_cost(eps, 100.0) == 0.0
    assert env._trading_cost(-eps, 100.0) == 0.0
    assert env._trading_cost(eps * 10, 100.0) == pytest.approx(1.0)
    np.testing.assert_allclose(env._trading_cost(np.array([eps, -eps * 10]), 100.0), [0.0, 1.0])


def test_trading_cost_zero_fixed_tc_is_purely_proportional():
    env = HedgingEnv(two_path_call_lsm(), prop_tc=0.002, fixed_tc=0.0)
    assert env._trading_cost(0.5, 120.0) == pytest.approx(0.002 * 0.5 * 120.0)


def test_payoff_call_scalar_and_array():
    env = HedgingEnv(two_path_call_lsm())  # call, strike 100
    assert env._payoff(110.0) == pytest.approx(10.0)
    assert env._payoff(90.0) == 0.0
    assert env._payoff(100.0) == 0.0
    np.testing.assert_allclose(env._payoff(np.array([90.0, 100.0, 125.0])), [0.0, 0.0, 25.0])


def test_payoff_put_scalar_and_array():
    env = HedgingEnv(early_exercise_put_lsm())  # put, strike 100
    assert env._payoff(90.0) == pytest.approx(10.0)
    assert env._payoff(110.0) == 0.0
    assert env._payoff(100.0) == 0.0
    np.testing.assert_allclose(env._payoff(np.array([75.0, 100.0, 125.0])), [25.0, 0.0, 0.0])


# --- options exercised at row 0 (BUGS.md, "HedgingEnv mishandles options
# exercised at row 0") ---------------------------------------------------------


def row0_put_lsm():
    """Two-path put with a deep-ITM start (S0=90, K=100, payoff 10) on path 0,
    exercised at row 0, and an ordinary path 1 exercised at row 2. Nonzero
    delta/position noise makes an ignored action observable."""
    return make_fake_lsm(
        [[90.0, 100.0], [80.0, 95.0], [70.0, 90.0]], [0, 2], option_type="put", delta=np.full((2, 2), -0.5)
    )


def test_step_on_row0_path_terminal_pays_minus_payoff_and_ignores_action_and_costs():
    # Before the fix step() hedged into row 1 and earned/charged on that move.
    env = HedgingEnv(row0_put_lsm(), prop_tc=0.01, fixed_tc=0.5)
    env.reset(0)

    state, reward, done = env.step(1.0)  # a full trade, which would be charged if executed

    assert done
    assert reward == pytest.approx(-10.0 / 100.0)  # -payoff(S0=90)/K; no trade cost despite tc, fixed_tc > 0
    assert env.position == 0.0 and env.t == 0
    assert state[4] == 0.0  # no hedge held


def test_step_on_row0_path_accounting_reward_is_minus_payoff_excess_over_v0():
    env = HedgingEnv(
        row0_put_lsm(), reward_mode="accounting", fixed_tc=0.5, value_fn=lambda t, S: np.full_like(S, 12.0 + t)
    )
    env.reset(0)

    _, reward, done = env.step(1.0)

    assert done
    assert reward == pytest.approx(-(10.0 - 12.0) / 100.0)  # V_0 = 12


def test_row0_accounting_episode_sum_equals_terminal_sum_plus_v0_over_strike():
    value_fn = lambda t, S: np.full_like(S, 12.0 + t)
    terminal = HedgingEnv(row0_put_lsm())
    accounting = HedgingEnv(row0_put_lsm(), reward_mode="accounting", value_fn=value_fn)
    t_sum = _sum_step_rewards(terminal, 0, lambda s: 0.7)
    a_sum = _sum_step_rewards(accounting, 0, lambda s: 0.7)
    assert a_sum == pytest.approx(t_sum + 12.0 / 100.0, abs=1e-12)


def test_row0_path_does_not_disturb_the_ordinary_path_in_the_same_env():
    # Path 1 must still hedge normally (position 0 -> 0.5 costs 0.001*0.5*100).
    env = HedgingEnv(row0_put_lsm())
    env.reset(1)
    _, reward, done = env.step(0.5)
    assert not done
    assert reward == pytest.approx(-0.05 / 100.0)


@pytest.mark.parametrize(
    "make_model, exercise_time, path",
    [
        (two_path_call_lsm, 1, 0),  # exercised before the last row: used to silently step on
        (two_path_call_lsm, 2, 0),  # exercised at the last row: used to run off the grid (IndexError)
        (row0_put_lsm, None, 0),  # exercised at row 0
    ],
)
def test_step_after_done_raises_until_reset(make_model, exercise_time, path):
    model = make_model()
    if exercise_time is not None:
        model.exercise_time = np.full_like(model.exercise_time, exercise_time)
    env = HedgingEnv(model)
    env.reset(path)
    done = False
    while not done:
        _, _, done = env.step(0.0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0.0)
    env.reset(path)  # a new episode clears the guard
    env.step(0.0)


def test_rollout_pnl_with_no_paths_returns_empty_array():
    env = HedgingEnv(random_lsm())
    out = env.rollout_pnl(_batch_hedge, np.array([], dtype=int))
    assert out.shape == (0,)
