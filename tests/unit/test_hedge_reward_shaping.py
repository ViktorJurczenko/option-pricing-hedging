"""train()'s reward shaping: reward_scale then risk_lambda reach the replay
buffer, while episode bookkeeping stays on the raw reward."""
import pytest
import tqdm as tqdm_pkg

import AmericanHedging.Hedge as hedge_module
from AmericanHedging.Hedge import train
from AmericanHedging.ReplayBuffer import ReplayBuffer
from tests.unit.hedge_fakes import FakeEnv, make_nets


class SpyReplayBuffer(ReplayBuffer):
    """Real ReplayBuffer behavior, plus a record of exactly what reward value
    was pushed on each call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed_rewards = []

    def push(self, state, action, reward, next_state, done):
        self.pushed_rewards.append(reward)
        super().push(state, action, reward, next_state, done)


def test_train_pushes_risk_shaped_reward_to_replay_buffer(tmp_path):
    """The replay buffer (and therefore everything the critic/actor train on)
    must receive `reward - risk_lambda * reward**2`, not the raw env reward."""
    risk_lambda = 0.5
    raw_rewards = [1.5, 2.0]
    env = FakeEnv(raw_rewards)
    buffer = SpyReplayBuffer(max_size=100)
    policy, critic1, critic2 = make_nets()

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
    policy, critic1, critic2 = make_nets()

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
    policy, critic1, critic2 = make_nets()

    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=10 ** 6, batch_size=10 ** 6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert buffer.pushed_rewards == pytest.approx(raw_rewards)


def test_train_scales_reward_before_applying_risk_penalty(tmp_path):
    """reward_scale must multiply the reward BEFORE the risk_lambda penalty
    is applied: buffer must receive (scale*r) - risk_lambda*(scale*r)**2,
    not risk_lambda*r**2 scaled afterward (a different, smaller number when
    scale != 1). risk_lambda*(scale*r)**2 grows with scale**2, so this test
    would fail under the wrong operation order even though reward_scale=1.0
    (the default, exercised by the pre-existing risk-shaping tests above)
    can't distinguish the two orderings at all."""
    risk_lambda = 0.5
    reward_scale = 10.0
    raw_rewards = [0.15, -0.2]
    env = FakeEnv(raw_rewards)
    buffer = SpyReplayBuffer(max_size=100)
    policy, critic1, critic2 = make_nets()

    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=10 ** 6, batch_size=10 ** 6,
        risk_lambda=risk_lambda, reward_scale=reward_scale,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    expected_shaped = [
        (reward_scale * r) - risk_lambda * (reward_scale * r) ** 2 for r in raw_rewards
    ]
    assert buffer.pushed_rewards == pytest.approx(expected_shaped)
