import math

import numpy as np
import pytest
import torch
import tqdm as tqdm_pkg

import AmericanHedging.Hedge as hedge_module
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.Hedge import _squash, cvar, evaluate, train
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
    must equal that one reward, std_pnl must be 0, and cvar_95
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
    # Second call: after step 1, t=1, S=105/100, normalized time t/(n_steps-1)=1/2
    # (3 rows -> last row index 2 is expiry), position now 0 (no_hedge action).
    np.testing.assert_allclose(seen_states[1], [1.05, 1 / 2, 0.0, 0.0, 0.0])


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


def test_evaluate_prints_var_above_cvar_and_cvar_is_tail_mean_not_percentile(monkeypatch, capsys):
    """Regression test: the "CVaR 95%" line used to print np.percentile(pnls, 5),
    which is the VaR. With differing P&Ls the two must differ: CVaR is the mean
    of the worst outcome(s), VaR the interpolated 5th percentile."""
    # 10 episodes: the first runs path 0 (P&L -0.1 under no_hedge), the rest path 1 (P&L 0).
    paths = iter([0] + [1] * 9)
    monkeypatch.setattr(np.random, "randint", lambda n: next(paths))
    env = HedgingEnv(FakeLSMModel())

    mean_pnl, std_pnl, cvar_95 = evaluate(no_hedge, env, n_episodes=10)

    pnls = np.array([-0.1] + [0.0] * 9)
    # n=10 -> worst ceil(0.5)=1 outcome: -0.1. 5th percentile interpolates
    # between sorted[0]=-0.1 and sorted[1]=0 at index 0.45 -> -0.055.
    assert cvar_95 == pytest.approx(-0.1)
    assert np.percentile(pnls, 5) == pytest.approx(-0.055)
    assert mean_pnl == pytest.approx(-0.01)
    out = capsys.readouterr().out.splitlines()
    var_lines = [l for l in out if l.startswith("VaR  95%:")]
    cvar_lines = [l for l in out if l.startswith("CVaR 95%:")]
    assert len(var_lines) == 1 and len(cvar_lines) == 1
    assert float(var_lines[0].split(":")[1]) == pytest.approx(-0.055, abs=1e-4)
    assert float(cvar_lines[0].split(":")[1]) == pytest.approx(-0.1, abs=1e-4)
    assert out.index(var_lines[0]) < out.index(cvar_lines[0])  # VaR directly above CVaR
    assert out.index(cvar_lines[0]) - out.index(var_lines[0]) == 1


# --- cvar() tests ---------------------------------------------------------


def test_cvar_n100_is_mean_of_five_smallest():
    # 0..99 shuffled: worst 5 are 0..4 -> mean 2.0. (Guards the float pitfall:
    # (1 - 0.95) * 100 == 5.000000000000004, a bare ceil would average 6 values -> 2.5.)
    pnls = np.random.default_rng(0).permutation(100).astype(float)
    assert cvar(pnls) == pytest.approx(2.0)


def test_cvar_n4_is_the_minimum():
    assert cvar([3.0, -2.0, 5.0, 1.0]) == -2.0


def test_cvar_n21_is_mean_of_two_smallest():
    # ceil(0.05 * 21) = ceil(1.05) = 2
    pnls = list(range(21))
    assert cvar(pnls[::-1]) == pytest.approx(0.5)


def test_cvar_single_sample_returns_it():
    assert cvar([7.0]) == 7.0


def test_cvar_custom_level():
    pnls = np.arange(10, dtype=float)
    assert cvar(pnls, level=0.5) == pytest.approx(2.0)   # worst 5: 0..4
    assert cvar(pnls, level=0.8) == pytest.approx(0.5)   # worst 2: 0, 1


def test_cvar_does_not_mutate_input_and_handles_unsorted_lists():
    pnls = [5.0, -1.0, 3.0, -4.0]
    assert cvar(pnls, level=0.5) == pytest.approx(-2.5)  # worst 2: -4, -1
    assert pnls == [5.0, -1.0, 3.0, -4.0]


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1])
def test_cvar_rejects_level_outside_open_unit_interval(level):
    with pytest.raises(ValueError, match="level"):
        cvar([1.0, 2.0, 3.0], level=level)


def test_cvar_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        cvar([])


@pytest.mark.parametrize("n_episodes", [0, -3])
def test_evaluate_rejects_non_positive_n_episodes(n_episodes):
    # Must fail up front, before touching the env (None would blow up otherwise).
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate(no_hedge, None, n_episodes=n_episodes)


def test_cvar_returns_python_float():
    assert type(cvar(np.array([1.0, 2.0, 3.0]))) is float


def test_cvar_never_exceeds_var_over_random_samples():
    rng = np.random.default_rng(1)
    for n in [1, 2, 5, 20, 21, 100, 333]:
        pnls = rng.normal(size=n)
        assert cvar(pnls) <= np.percentile(pnls, 5) + 1e-12, n


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
    policy, critic1, critic2 = _make_nets()

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


@pytest.mark.parametrize("reward_scale", [0, -1.0, -5])
def test_train_raises_on_non_positive_reward_scale(reward_scale):
    with pytest.raises(ValueError, match="reward_scale"):
        train(None, None, None, None, None, reward_scale=reward_scale)


def test_train_raises_on_non_positive_alpha_with_auto_alpha():
    with pytest.raises(ValueError, match="alpha"):
        train(None, None, None, None, None, alpha=0.0, auto_alpha=True)


def test_auto_alpha_moves_alpha_away_from_its_initial_value(monkeypatch, tmp_path):
    """With auto_alpha=True, each gradient update runs an Adam step on
    log_alpha driven by (log_pi.detach() + target_entropy).mean(), which is
    virtually never exactly zero for a freshly (seeded) initialized Gaussian
    policy -- so alpha must move away from its initial value. Observed the
    same way test_train_episode_reward_bookkeeping_stays_on_raw_reward
    observes episode_reward: via a tqdm subclass recording set_postfix."""
    torch.manual_seed(0)
    initial_alpha = 0.2
    env = FakeEnv(raw_rewards=[1.0] * 6)  # 6-step episode
    buffer = ReplayBuffer(max_size=1000)
    policy, critic1, critic2 = _make_nets()

    postfix_log = []

    class RecordingTqdm(tqdm_pkg.tqdm):
        def set_postfix(self, **kwargs):
            postfix_log.append(kwargs)
            return super().set_postfix(**kwargs)

    monkeypatch.setattr(hedge_module, "tqdm", RecordingTqdm)

    hedge_module.train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        alpha=initial_alpha, auto_alpha=True,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert len(postfix_log) == 1
    assert "alpha" in postfix_log[0]
    final_alpha = float(postfix_log[0]["alpha"])
    assert final_alpha != pytest.approx(initial_alpha)


def test_without_auto_alpha_postfix_omits_alpha_key(monkeypatch, tmp_path):
    """Regression guard: with auto_alpha=False (the default), the tqdm
    postfix must not gain an 'alpha' key at all -- alpha stays whatever was
    passed in, with no log_alpha parameter/optimizer created."""
    env = FakeEnv(raw_rewards=[1.0] * 6)
    buffer = ReplayBuffer(max_size=1000)
    policy, critic1, critic2 = _make_nets()

    postfix_log = []

    class RecordingTqdm(tqdm_pkg.tqdm):
        def set_postfix(self, **kwargs):
            postfix_log.append(kwargs)
            return super().set_postfix(**kwargs)

    monkeypatch.setattr(hedge_module, "tqdm", RecordingTqdm)

    hedge_module.train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        auto_alpha=False,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert len(postfix_log) == 1
    assert "alpha" not in postfix_log[0]


def test_alpha_lr_none_reuses_lr_for_log_alpha_optimizer(monkeypatch, tmp_path):
    """alpha_lr=None (the default) must make the log_alpha Adam optimizer
    reuse `lr`, not some other hardcoded value -- this is the exact fallback
    the alpha_lr parameter introduces. Checked directly on the Adam
    constructor call (via a spy on torch.optim.Adam) rather than on the
    resulting alpha trajectory, since that keeps the test deterministic
    instead of depending on gradient dynamics."""
    captured_lrs = []
    real_adam = torch.optim.Adam

    class SpyAdam(real_adam):
        def __init__(self, params, lr, *args, **kwargs):
            captured_lrs.append(lr)
            super().__init__(params, lr, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", SpyAdam)

    env = FakeEnv(raw_rewards=[1.0] * 6)
    buffer = ReplayBuffer(max_size=1000)
    policy, critic1, critic2 = _make_nets()

    hedge_module.train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        lr=7e-3, auto_alpha=True, alpha_lr=None,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    # Creation order in train(): actor_opt, critic_opt, then (auto_alpha)
    # alpha_opt -- all three must see lr=7e-3 when alpha_lr is None.
    assert captured_lrs == pytest.approx([7e-3, 7e-3, 7e-3])


def test_alpha_lr_explicit_value_used_only_for_log_alpha_optimizer(monkeypatch, tmp_path):
    """An explicit alpha_lr, different from lr, must reach only the log_alpha
    optimizer -- the actor/critic optimizers keep training at `lr`."""
    captured_lrs = []
    real_adam = torch.optim.Adam

    class SpyAdam(real_adam):
        def __init__(self, params, lr, *args, **kwargs):
            captured_lrs.append(lr)
            super().__init__(params, lr, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", SpyAdam)

    env = FakeEnv(raw_rewards=[1.0] * 6)
    buffer = ReplayBuffer(max_size=1000)
    policy, critic1, critic2 = _make_nets()

    hedge_module.train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        lr=7e-3, auto_alpha=True, alpha_lr=0.5,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert captured_lrs == pytest.approx([7e-3, 7e-3, 0.5])


@pytest.mark.parametrize("alpha_lr", [0, -1.0, -5])
def test_train_raises_on_non_positive_alpha_lr(alpha_lr):
    with pytest.raises(ValueError, match="alpha_lr"):
        train(None, None, None, None, None, alpha_lr=alpha_lr)


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


# --- _squash() tests -------------------------------------------------------
#
# Regression coverage for the actor-mean-divergence bug: train() used to feed
# critics/actor loss an unbounded dist.rsample() (or, in an intermediate
# attempt, a clamped action whose log_prob was evaluated at the pre-clamp
# point), letting the actor exploit unreliable out-of-distribution critic
# outputs and drive `mean` to the tens of thousands / `log_std` to its
# clamp ceiling. `_squash` fixes this with a tanh-squashed reparameterized
# sample plus the matching log-prob Jacobian correction.

def test_squash_output_always_within_bounds_for_extreme_distribution():
    """Core invariant _squash must provide: its returned action always lies
    within [-1, 1] (matching HedgingEnv.step()'s own clip range) no matter
    how extreme the underlying Normal's mean/std are. This is exactly the
    property the pre-fix code violated -- a bare `dist.rsample()` has no such
    bound, which is what let the actor loss score critics far outside the
    range they were ever trained on and let `mean` diverge to values like
    -63996.84 / -81410.02 seen in trained checkpoints before this fix.

    Note: at these extreme magnitudes tanh saturates to exactly +/-1.0 in
    floating point (mathematically expected -- tanh(1e5) is indistinguishable
    from 1.0 at float32/64 precision), so the assertion below is a closed
    interval, not a strict one; the log_prob finiteness check is what confirms
    _squash still behaves sanely (no inf/nan) even at that saturation point.
    """
    torch.manual_seed(0)
    extreme_dists = [
        torch.distributions.Normal(torch.tensor([100000.0]), torch.tensor([50.0])),
        torch.distributions.Normal(torch.tensor([-81410.02]), torch.tensor([7.389])),
        torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1e-3])),
    ]
    for dist in extreme_dists:
        for _ in range(10):
            action, log_prob = _squash(dist)
            assert torch.all(action >= -1.0)
            assert torch.all(action <= 1.0)
            assert torch.isfinite(log_prob).all()


def test_squash_log_prob_matches_manual_tanh_jacobian_correction(monkeypatch):
    """Independently recompute _squash's log_prob formula (Normal log-density
    evaluated by hand, not via dist.log_prob, so this doesn't just restate
    the implementation) for a fixed, non-trivial raw sample and check they
    match exactly. This is the formula an earlier, insufficient fix attempt
    got wrong by evaluating log_prob at the *clamped* action under an
    ever-drifting mean instead of at the raw pre-squash sample."""
    mean = torch.tensor([[0.3]])
    std = torch.tensor([[0.7]])
    dist = torch.distributions.Normal(mean, std)

    raw_sample = torch.tensor([[1.234]])
    monkeypatch.setattr(dist, "rsample", lambda *args, **kwargs: raw_sample)

    action, log_prob = _squash(dist)

    expected_action = torch.tanh(raw_sample)
    # Manual Normal log-density: -log(std) - 0.5*log(2*pi) - 0.5*((x-mean)/std)^2
    z = (raw_sample - mean) / std
    manual_normal_log_prob = -torch.log(std) - 0.5 * math.log(2 * math.pi) - 0.5 * z.pow(2)
    manual_correction = torch.log(1 - expected_action.pow(2) + 1e-6)
    expected_log_prob = (manual_normal_log_prob - manual_correction).sum(-1, keepdim=True)

    assert torch.allclose(action, expected_action)
    assert torch.allclose(log_prob, expected_log_prob, atol=1e-5)


def test_train_feeds_only_bounded_actions_to_critics(tmp_path):
    """Integration-level regression test: checks train() actually *uses*
    _squash at both call sites (the Bellman target and the actor loss), not
    just that _squash is correct in isolation. Forward hooks on critic1/
    critic2 record every action tensor any critic sees during a real train()
    run (including the target1/target2 deepcopies -- hooks are plain
    functions, which `copy.deepcopy` treats as atomic and returns unchanged,
    so hooks registered on critic1/critic2 before train() deep-copies them
    into target1/target2 still fire and still append to this same list).

    Before the fix, this would have failed intermittently/eventually: a2 (fed
    to target1/target2) and a_new (fed to critic1/critic2) were raw
    `dist.rsample()` draws with no bound, so as training progressed and
    `mean`/`std` drifted, actions outside (-1, 1) would show up here -- which
    is exactly the out-of-distribution critic-extrapolation mechanism behind
    the bug.
    """
    torch.manual_seed(0)
    seen_actions = []

    def record_action(module, inputs, output):
        seen_actions.append(inputs[1].detach().clone())

    policy, critic1, critic2 = _make_nets()
    critic1.register_forward_hook(record_action)
    critic2.register_forward_hook(record_action)

    env = FakeEnv(raw_rewards=[1.0] * 6)  # 6-step episode
    buffer = ReplayBuffer(max_size=1000)

    train(
        policy, env, buffer, critic1, critic2,
        n_episodes=1, warmup_steps=0, update_every=1, batch_size=2,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    # update triggers at steps 2..6 (5 triggers) x 2 critics each seeing
    # (s,a), (s2,a2) for the critic loss and (s,a_new) for the actor loss.
    assert len(seen_actions) > 0
    for action in seen_actions:
        assert torch.all(action > -1.0), "action fed to a critic was <= -1"
        assert torch.all(action < 1.0), "action fed to a critic was >= 1"


# --- residual-on-delta hedge mapping in train() ---------------------------

class DeltaEnv:
    """Env stand-in whose states carry a known, varying delta (state index 2),
    recording every action passed to step()."""

    DELTAS = [0.30, 0.95, -0.97, 0.5]  # states seen at steps 0..3 (+ terminal)

    def __init__(self):
        self.step_actions = []

    def _state(self, i):
        return np.array([1.0, 0.0, self.DELTAS[i], 0.0, 0.0], dtype=np.float32)

    def reset(self):
        self._i = 0
        return self._state(0)

    def step(self, action):
        self.step_actions.append(action)
        self._i += 1
        done = self._i >= 3
        return self._state(self._i), 0.0, done


class ActionSpyBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []  # (state, action)

    def push(self, state, action, reward, next_state, done):
        self.pushed.append((state.copy(), np.array(action, copy=True)))
        super().push(state, action, reward, next_state, done)


def _run_delta_env(tmp_path, residual_scale, warmup_steps):
    env = DeltaEnv()
    buffer = ActionSpyBuffer(max_size=100)
    policy = RLPolicy(5, 1, residual_scale=residual_scale)
    train(
        policy, env, buffer, Critic(5, 1), Critic(5, 1),
        n_episodes=1, warmup_steps=warmup_steps, batch_size=10 ** 6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )
    return env, buffer


@pytest.mark.parametrize("warmup_steps", [10 ** 6, 0], ids=["warmup_random", "policy_sampled"])
def test_train_sends_residual_hedge_to_env_but_stores_squashed_action(tmp_path, warmup_steps):
    """The env must see clip(delta + c*action, -1, 1) while the replay buffer
    (what the critics train on) keeps the raw squashed action. Deltas 0.95 and
    -0.97 with c=0.5 make the clamp bind for large same-sign actions."""
    torch.manual_seed(0)
    np.random.seed(0)
    c = 0.5
    env, buffer = _run_delta_env(tmp_path, c, warmup_steps)

    assert len(env.step_actions) == len(buffer.pushed) == 3
    for sent, (state, pushed_action) in zip(env.step_actions, buffer.pushed):
        assert pushed_action.shape == (1,)  # one action vector per step
        pushed_action = pushed_action.item()
        assert -1.0 < pushed_action < 1.0
        expected = float(np.clip(state[2] + c * pushed_action, -1.0, 1.0))
        assert sent == pytest.approx(expected, abs=1e-6)
    # The mapping really is applied: the env value differs from the stored action.
    assert env.step_actions[0] != pytest.approx(buffer.pushed[0][1], abs=1e-3)


def test_train_with_zero_residual_scale_sends_squashed_action_unchanged(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    env, buffer = _run_delta_env(tmp_path, 0.0, warmup_steps=10 ** 6)
    for sent, (_, pushed_action) in zip(env.step_actions, buffer.pushed):
        assert sent == pytest.approx(pushed_action, abs=1e-6)  # float32 rounding only


# --- gap_fraction mode in train() and deterministic_hedge --------------------

class PositionEnv:
    """Env stand-in whose states carry a known delta (index 2) and a known,
    varying position (index 4), recording every action passed to step()."""

    DELTAS = [0.30, 0.95, -0.50, 0.5]
    POSITIONS = [0.0, 0.6, -0.4, 0.2]

    def __init__(self):
        self.step_actions = []

    def _state(self, i):
        return np.array([1.0, 0.0, self.DELTAS[i], 0.0, self.POSITIONS[i]], dtype=np.float32)

    def reset(self):
        self._i = 0
        return self._state(0)

    def step(self, action):
        self.step_actions.append(action)
        self._i += 1
        return self._state(self._i), 0.0, self._i >= 3


@pytest.mark.parametrize("warmup_steps", [10 ** 6, 0], ids=["warmup_random", "policy_sampled"])
def test_train_gap_fraction_sends_interpolated_hedge_but_stores_squashed_action(tmp_path, warmup_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    env, buffer = PositionEnv(), ActionSpyBuffer(max_size=100)
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    train(policy, env, buffer, Critic(5, 1), Critic(5, 1),
          n_episodes=1, warmup_steps=warmup_steps, batch_size=10 ** 6,
          checkpoint_path=str(tmp_path / "policy.pth"))

    assert len(env.step_actions) == len(buffer.pushed) == 3
    for sent, (state, pushed_action) in zip(env.step_actions, buffer.pushed):
        assert pushed_action.shape == (1,)  # one action vector per step
        pushed_action = pushed_action.item()
        assert -1.0 < pushed_action < 1.0
        w = (pushed_action + 1) / 2
        expected = float(np.clip(state[4] + w * (state[2] - state[4]), -1.0, 1.0))
        assert sent == pytest.approx(expected, abs=1e-6)
    # The mapping is really applied: env value differs from the stored action.
    assert env.step_actions[0] != pytest.approx(buffer.pushed[0][1], abs=1e-3)


def test_train_gap_fraction_runs_gradient_updates_and_keeps_mode(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    before = policy.layer1.weight.detach().clone()
    path = tmp_path / "policy.pth"
    train(policy, PositionEnv(), ReplayBuffer(max_size=100), Critic(5, 1), Critic(5, 1),
          n_episodes=3, warmup_steps=0, update_every=1, batch_size=2,
          checkpoint_path=str(path))
    assert not torch.equal(before, policy.layer1.weight)
    # The mode is a buffer, so training must never change it.
    assert policy.gap_fraction.item() == 1.0


def test_deterministic_hedge_gap_fraction_fresh_policy_closes_90_percent_of_gap():
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    with torch.no_grad():
        policy.layer3.weight.zero_()
    states = np.array([[1.0, 0.1, 0.8, 0.2, 0.0],
                       [1.0, 0.1, -0.5, 0.2, 0.5]])
    out = hedge_module.deterministic_hedge(policy, states)
    np.testing.assert_allclose(out, [0.9 * 0.8, 0.5 + 0.9 * (-1.0)], atol=1e-6)


# --- deterministic_hedge ----------------------------------------------------

def test_deterministic_hedge_matches_hedge_ratio_of_tanh_mean_for_batch_and_single():
    torch.manual_seed(0)
    policy = RLPolicy(5, 1, residual_scale=0.1)
    states = np.random.default_rng(0).uniform(-1, 1, (6, 5))

    out = hedge_module.deterministic_hedge(policy, states)
    assert isinstance(out, np.ndarray) and out.shape == (6,)

    x = torch.tensor(states, dtype=torch.float32)
    expected = torch.clamp(x[:, 2:3] + 0.1 * torch.tanh(policy(x).mean), -1, 1).detach().numpy().ravel()
    np.testing.assert_allclose(out, expected, atol=1e-6)

    single = hedge_module.deterministic_hedge(policy, states[0])
    assert single.shape == (1,)
    assert single.item() == pytest.approx(out[0], abs=1e-6)


def test_deterministic_hedge_with_zero_residual_scale_is_tanh_of_mean():
    torch.manual_seed(1)
    policy = RLPolicy(5, 1)
    states = np.random.default_rng(1).uniform(-1, 1, (4, 5))
    x = torch.tensor(states, dtype=torch.float32)
    expected = torch.tanh(policy(x).mean).detach().numpy().ravel()
    np.testing.assert_allclose(hedge_module.deterministic_hedge(policy, states), expected, atol=1e-6)


# --- train() with a validation env ------------------------------------------

class ScriptedValEnv:
    """Stands in for a validation HedgingEnv: rollout_pnl returns scripted P&L
    arrays (one per call) and records the paths it was asked for plus a copy of
    the policy's weights at that moment."""

    def __init__(self, scripted_pnls, policy, n_paths=500):
        self.scripted = list(scripted_pnls)
        self.policy = policy
        self.n_paths = n_paths
        self.calls = []
        self.snapshots = []

    def rollout_pnl(self, hedge_fn, path_idxs):
        self.calls.append(np.array(path_idxs))
        self.snapshots.append({k: v.clone() for k, v in self.policy.state_dict().items()})
        return np.asarray(self.scripted[len(self.calls) - 1], dtype=float)


def _train_with_val(tmp_path, scripted, n_episodes, val_every, val_metric="std", val_paths=5,
                    n_paths=500, update=True):
    """Runs train() on 2-step episodes with real updates (so weights change
    between scorings). Returns (val_env, checkpoint_path)."""
    torch.manual_seed(0)
    policy, c1, c2 = _make_nets()
    val_env = ScriptedValEnv(scripted, policy, n_paths=n_paths)
    path = str(tmp_path / "p.pth")
    train(
        policy, FakeEnv([0.1, -0.2]), ReplayBuffer(max_size=100), c1, c2,
        n_episodes=n_episodes, warmup_steps=0 if update else 10 ** 6, update_every=1, batch_size=1,
        val_env=val_env, val_paths=val_paths, val_every=val_every, val_metric=val_metric,
        checkpoint_path=path,
    )
    return val_env, path


def _assert_ckpt_equals(path, snapshot):
    saved = torch.load(path)
    assert saved.keys() == snapshot.keys()
    for k in saved:
        assert torch.equal(saved[k], snapshot[k]), k


@pytest.mark.parametrize("val_metric", ["mean", ""])
def test_train_rejects_unknown_val_metric(val_metric):
    with pytest.raises(ValueError, match="val_metric"):
        train(None, None, None, None, None, val_metric=val_metric)


@pytest.mark.parametrize("kw", [{"val_paths": 0}, {"val_every": 0}, {"val_paths": -1}])
def test_train_rejects_non_positive_val_paths_or_every_when_val_env_given(kw):
    with pytest.raises(ValueError, match="val_paths and val_every"):
        train(None, None, None, None, None, val_env=object(), **kw)


def test_train_scores_and_saves_untrained_policy_before_any_training(tmp_path):
    # Untrained policy must already be the checkpoint: one scoring, zero episodes run.
    val_env, path = _train_with_val(tmp_path, [[1.0, -1.0]], n_episodes=0, val_every=10)
    assert len(val_env.calls) == 1
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_rescoring_schedule_every_val_every_and_after_last_episode(tmp_path):
    # n_episodes=7, val_every=3 -> initial, ep 3, ep 6, ep 7 (not a multiple)
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 4, n_episodes=7, val_every=3)
    assert len(val_env.calls) == 4


def test_train_rescoring_when_n_episodes_is_a_multiple_does_not_double_score(tmp_path):
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 3, n_episodes=4, val_every=2)
    assert len(val_env.calls) == 3  # initial, ep 2, ep 4


def test_train_std_metric_saves_only_on_strict_improvement(tmp_path):
    # std: 2.0 (initial), 3.0 worse, 1.0 better (saved), 1.0 tie (not saved)
    pnls = [[2.0, -2.0], [3.0, -3.0], [1.0, -1.0], [-1.0, 1.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=3, val_every=1, val_metric="std")
    assert len(val_env.calls) == 4
    _assert_ckpt_equals(path, val_env.snapshots[2])
    # the tie really is a different policy, so "not saved" is observable
    assert not all(torch.equal(val_env.snapshots[2][k], val_env.snapshots[3][k]) for k in val_env.snapshots[2])


def test_train_std_metric_keeps_initial_when_nothing_improves(tmp_path):
    pnls = [[1.0, -1.0], [5.0, -5.0], [2.0, -2.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=2, val_every=1, val_metric="std")
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_cvar_metric_higher_is_better(tmp_path):
    # Same shape shifted: initial, worse (-0.5), better (+0.5) -> the last scoring is saved.
    base = np.linspace(-1, 1, 101)
    pnls = [base, base - 0.5, base + 0.5]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=2, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[2])


def test_train_cvar_metric_ranks_by_tail_mean_not_fifth_percentile(tmp_path):
    # Regression test: "cvar" used to score np.percentile(pnls, 5) (the VaR).
    # `a` has a better 5th percentile (0.0) but a catastrophic tail (mean of worst 5 = -6);
    # `b` has a worse percentile (-1.0) but a mild tail (-1.0). CVaR must prefer `b`.
    a = np.concatenate([np.full(3, -10.0), np.zeros(97)])
    b = np.concatenate([np.full(6, -1.0), np.zeros(94)])
    assert np.percentile(a, 5) > np.percentile(b, 5)
    assert cvar(a) < cvar(b)
    val_env, path = _train_with_val(tmp_path, [a, b], n_episodes=1, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[1])


def test_train_cvar_metric_not_fooled_by_lower_std(tmp_path):
    # Second P&L has lower std but a worse left tail (CVaR -0.1 vs 0.0): it must NOT be saved.
    first = np.concatenate([np.full(95, 0.0), np.full(5, 1.0)])       # CVaR 0.0
    second = np.concatenate([np.full(10, -0.1), np.full(90, 0.0)])    # CVaR -0.1, smaller std
    assert np.std(second) < np.std(first)
    val_env, path = _train_with_val(tmp_path, [first, second], n_episodes=1, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_validates_on_first_min_val_paths_and_n_paths_in_order(tmp_path):
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 2, n_episodes=1, val_every=1, val_paths=5, n_paths=500)
    np.testing.assert_array_equal(val_env.calls[0], np.arange(5))
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 2, n_episodes=1, val_every=1, val_paths=50, n_paths=3)
    np.testing.assert_array_equal(val_env.calls[0], np.arange(3))


def test_train_validation_hedge_fn_is_deterministic_hedge_of_the_policy(tmp_path):
    torch.manual_seed(0)
    policy, c1, c2 = _make_nets()
    states = np.random.default_rng(0).uniform(-1, 1, (3, 5))
    seen = []

    class Env(ScriptedValEnv):
        def rollout_pnl(self, hedge_fn, path_idxs):
            seen.append((hedge_fn(states), hedge_module.deterministic_hedge(policy, states)))
            return super().rollout_pnl(hedge_fn, path_idxs)

    train(policy, FakeEnv([0.1]), ReplayBuffer(max_size=10), c1, c2, n_episodes=0,
          val_env=Env([[1.0, -1.0]], policy), val_paths=2, val_every=1,
          checkpoint_path=str(tmp_path / "p.pth"))
    np.testing.assert_array_equal(seen[0][0], seen[0][1])


def test_train_without_val_env_keeps_trailing_50_mean_reward_criterion(tmp_path):
    # No checkpoint may exist before 50 episodes have been recorded; afterwards one is saved.
    policy, c1, c2 = _make_nets()
    path = tmp_path / "p.pth"
    train(policy, FakeEnv([0.1]), ReplayBuffer(max_size=100), c1, c2, n_episodes=49,
          warmup_steps=10 ** 6, batch_size=10 ** 6, checkpoint_path=str(path))
    assert not path.exists()
    train(policy, FakeEnv([0.1]), ReplayBuffer(max_size=100), c1, c2, n_episodes=50,
          warmup_steps=10 ** 6, batch_size=10 ** 6, checkpoint_path=str(path))
    assert path.exists()


def test_train_with_val_env_does_not_use_trailing_mean_criterion(tmp_path):
    # 60 episodes with val_every=100 -> only the initial and the final scoring can save;
    # a worse final score must leave the initial (untrained) weights on disk.
    pnls = [[1.0, -1.0], [9.0, -9.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=60, val_every=100, update=False)
    _assert_ckpt_equals(path, val_env.snapshots[0])


# --- vector actions: ReplayBuffer and target_entropy ------------------------

def test_replay_buffer_sample_keeps_action_dimension():
    scalar, vector = ReplayBuffer(max_size=10), ReplayBuffer(max_size=10)
    for i in range(4):
        scalar.push(np.zeros(5), 0.1 * i, 0.0, np.zeros(5), False)
        vector.push(np.zeros(5), np.array([0.1 * i, -0.1 * i]), 0.0, np.zeros(5), False)
    assert scalar.sample(3)[1].shape == (3, 1)
    assert vector.sample(3)[1].shape == (3, 2)


@pytest.mark.parametrize("action_dim,mode,expected", [(1, "residual", -1.0)])
def test_train_target_entropy_defaults_to_minus_action_dim(monkeypatch, tmp_path, action_dim, mode, expected):
    """The alpha loss is -(log_alpha * (log_pi + target_entropy)).mean(); with
    log_pi forced to 0, its gradient wrt log_alpha is -target_entropy."""
    grads = []
    real_step = torch.optim.Adam.step

    def spy_step(self, *a, **k):
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() == 0 and p.grad is not None:  # log_alpha is the only scalar parameter
                    grads.append(p.grad.item())
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", spy_step)
    monkeypatch.setattr(hedge_module, "_squash",
                        lambda dist: (torch.tanh(dist.mean), torch.zeros(dist.mean.shape[:-1] + (1,))))
    train(RLPolicy(5, action_dim, action_mode=mode), PositionEnv(), ReplayBuffer(max_size=100),
          Critic(5, action_dim), Critic(5, action_dim),
          n_episodes=1, warmup_steps=0, update_every=1, batch_size=2, auto_alpha=True,
          checkpoint_path=str(tmp_path / "policy.pth"))
    assert grads and all(g == pytest.approx(-expected) for g in grads)
