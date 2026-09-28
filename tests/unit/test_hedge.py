import numpy as np
import pytest
import tqdm as tqdm_pkg

import AmericanHedging.Hedge as hedge_module
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.Hedge import evaluate, train
from AmericanHedging.ReplayBuffer import ReplayBuffer
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Critic import Critic
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


# --- train() tests --------------------------------------------------------
#
# These target Hedge.py's two Stage 1/2 training-mechanics changes: the
# reward-shaping split (replay buffer gets `reward - risk_lambda*reward**2`
# while episode bookkeeping keeps the raw reward) and grad_steps_per_update
# (N gradient-update cycles per trigger instead of always 1). Real RLPolicy/
# Critic nets are used (they're tiny), but a fake env replaces HedgingEnv so
# the reward sequence is fully controlled and the SAC math itself isn't what
# is being asserted on.

class FakeEnv:
    """Minimal env stand-in for train(): yields a fixed reward sequence and
    ignores the action entirely. The env's own reward formula is HedgingEnv's
    job (covered in test_hedge_env.py) -- this only needs *some* deterministic
    reward per step to check what train() does with it."""

    def __init__(self, raw_rewards, state_dim=5):
        self.raw_rewards = raw_rewards
        self.state_dim = state_dim

    def reset(self):
        self._step_idx = 0
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action):
        reward = self.raw_rewards[self._step_idx]
        self._step_idx += 1
        done = self._step_idx >= len(self.raw_rewards)
        return np.zeros(self.state_dim, dtype=np.float32), reward, done


class SpyReplayBuffer(ReplayBuffer):
    """Real ReplayBuffer behavior, plus a record of exactly what reward value
    was pushed on each call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed_rewards = []

    def push(self, state, action, reward, next_state, done):
        self.pushed_rewards.append(reward)
        super().push(state, action, reward, next_state, done)


class CountingReplayBuffer(ReplayBuffer):
    """Real ReplayBuffer behavior, plus a count of how many times sample()
    was called (once per gradient-update cycle)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_calls = 0

    def sample(self, batch_size):
        self.sample_calls += 1
        return super().sample(batch_size)


def _make_nets():
    return RLPolicy(5, 1), Critic(5, 1), Critic(5, 1)


def test_train_pushes_risk_shaped_reward_to_replay_buffer(tmp_path):
    """The replay buffer (and therefore everything the critic/actor train on)
    must receive `reward - risk_lambda * reward**2`, not the raw env reward."""
    risk_lambda = 0.5
    raw_rewards = [1.5, 2.0]
    env = FakeEnv(raw_rewards)
    buffer = SpyReplayBuffer(max_size=100)
    policy, critic1, critic2 = _make_nets()

    # warmup_steps/batch_size set far above the handful of steps this test
    # takes so the gradient-update block never triggers -- this isolates the
    # reward-shaping/bookkeeping behavior from the (untouched) SAC math.
    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=2, warmup_steps=10 ** 6, batch_size=10 ** 6,
        risk_lambda=risk_lambda, checkpoint_path=str(tmp_path / "policy.pth"),
    )

    expected_shaped = [r - risk_lambda * r ** 2 for r in raw_rewards] * 2  # 2 identical episodes
    assert buffer.pushed_rewards == pytest.approx(expected_shaped)


def test_train_episode_reward_bookkeeping_stays_on_raw_reward(monkeypatch, tmp_path):
    """Regression guard: episode_reward (what feeds the trailing-50-episode
    mean and the checkpoint-save criterion) must track the RAW env reward
    even when risk_lambda != 0, so it stays comparable to test-hedge's
    reported P&L, which never applies the risk penalty. This is easy to get
    backwards (e.g. accidentally accumulating shaped_reward instead).

    train()'s episode_reward is a local variable with no public accessor, so
    it's observed here via a tqdm subclass that records the exact string
    passed to `set_postfix(reward=...)` every episode.
    """
    risk_lambda = 0.5
    raw_rewards = [1.5, 2.0]
    env = FakeEnv(raw_rewards)
    buffer = SpyReplayBuffer(max_size=100)
    policy, critic1, critic2 = _make_nets()

    postfix_log = []

    class RecordingTqdm(tqdm_pkg.tqdm):
        def set_postfix(self, **kwargs):
            postfix_log.append(kwargs)
            return super().set_postfix(**kwargs)

    monkeypatch.setattr(hedge_module, "tqdm", RecordingTqdm)

    hedge_module.train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=10 ** 6, batch_size=10 ** 6,
        risk_lambda=risk_lambda, checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert len(postfix_log) == 1
    logged_reward = float(postfix_log[0]["reward"])
    assert logged_reward == pytest.approx(sum(raw_rewards))  # raw sum, not risk-shaped


def test_train_with_default_risk_lambda_reproduces_prior_behavior(tmp_path):
    """risk_lambda=0.0 (the default) must reproduce pre-Stage-2 behavior
    exactly: shaped_reward == reward, so the buffer receives raw rewards."""
    raw_rewards = [1.5, -2.0, 0.0]
    env = FakeEnv(raw_rewards)
    buffer = SpyReplayBuffer(max_size=100)
    policy, critic1, critic2 = _make_nets()

    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=10 ** 6, batch_size=10 ** 6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert buffer.pushed_rewards == pytest.approx(raw_rewards)


def _run_and_count_sample_calls(tmp_path, grad_steps_per_update):
    env = FakeEnv(raw_rewards=[1.0] * 6)  # 6-step episode
    buffer = CountingReplayBuffer(max_size=1000)
    policy, critic1, critic2 = _make_nets()

    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        grad_steps_per_update=grad_steps_per_update,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )
    return buffer.sample_calls


def test_grad_steps_per_update_runs_n_gradient_cycles_per_trigger(tmp_path):
    """With update_every=1 and batch_size=2, the update condition first
    becomes true once 2 transitions are buffered (step 2 of the 6-step
    episode) and stays true every step after -- 5 triggers total. Each
    trigger must run grad_steps_per_update sample()+critic+actor cycles, not
    always exactly 1."""
    calls_single = _run_and_count_sample_calls(tmp_path, grad_steps_per_update=1)
    calls_triple = _run_and_count_sample_calls(tmp_path, grad_steps_per_update=3)

    assert calls_single == 5
    assert calls_triple == 15


@pytest.mark.parametrize("grad_steps_per_update", [0, -1, -5])
def test_train_raises_on_non_positive_grad_steps_per_update(grad_steps_per_update):
    """grad_steps_per_update <= 0 used to silently disable every critic/actor
    update (range(0) or range(negative) never executes the update block),
    letting a full n_episodes training run complete without ever learning.
    train() now validates this eagerly, before doing any real work -- so all
    other args can be trivial/None, since the ValueError must fire first."""
    with pytest.raises(ValueError, match="grad_steps_per_update"):
        train(
            None, None, None, None, None,
            grad_steps_per_update=grad_steps_per_update,
        )
