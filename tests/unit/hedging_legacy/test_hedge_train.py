"""optionpricing.ml.hedging.legacy.sac.train(): the training loop wiring, its
argument checks, the entropy coefficient, action squashing and each action
mode."""

import math

import numpy as np
import pytest
import torch
import tqdm as tqdm_pkg

import optionpricing.ml.hedging.legacy.sac as hedge_module
from optionpricing.ml.hedging.legacy.critic import Critic
from optionpricing.ml.hedging.legacy.policy import RLPolicy
from optionpricing.ml.hedging.legacy.replay_buffer import ReplayBuffer
from optionpricing.ml.hedging.legacy.sac import _squash, train
from tests.unit.hedging_legacy.hedge_fakes import (
    ActionSpyBuffer,
    BanditEnv,
    FakeEnv,
    PositionEnv,
    make_nets,
    train_bandit,
)


class CountingReplayBuffer(ReplayBuffer):
    """Real ReplayBuffer behavior, plus a count of how many times sample()
    was called (once per gradient-update cycle)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_calls = 0

    def sample(self, batch_size):
        self.sample_calls += 1
        return super().sample(batch_size)


def _run_and_count_sample_calls(tmp_path, grad_steps_per_update):
    env = FakeEnv(raw_rewards=[1.0] * 6)  # 6-step episode
    buffer = CountingReplayBuffer(max_size=1000)
    policy, critic1, critic2 = make_nets()

    train(
        policy,
        env,
        buffer,
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
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


@pytest.mark.parametrize("reward_scale", [0, -1.0, -5])
def test_train_raises_on_non_positive_reward_scale(reward_scale):
    with pytest.raises(ValueError, match="reward_scale"):
        train(None, None, None, None, None, reward_scale=reward_scale)


def test_train_raises_on_non_positive_alpha_with_auto_alpha():
    with pytest.raises(ValueError, match="alpha"):
        train(None, None, None, None, None, alpha=0.0, auto_alpha=True)


@pytest.mark.parametrize("target_entropy,direction", [(50.0, 1), (-50.0, -1)])
def test_auto_alpha_moves_in_the_direction_the_entropy_gap_implies(monkeypatch, tmp_path, target_entropy, direction):
    """alpha_loss = -(log_alpha * (log_pi + target_entropy)).mean(). A target
    entropy far above the policy's entropy (+50: log_pi + 50 > 0 for any sane
    sample) must push alpha UP to raise exploration; one far below (-50) must
    push it DOWN. The final alpha is read from the tqdm postfix."""
    torch.manual_seed(0)
    initial_alpha = 0.2
    postfix_log = []

    class RecordingTqdm(tqdm_pkg.tqdm):
        def set_postfix(self, **kwargs):
            postfix_log.append(kwargs)
            return super().set_postfix(**kwargs)

    monkeypatch.setattr(hedge_module, "tqdm", RecordingTqdm)
    policy, critic1, critic2 = make_nets()

    hedge_module.train(
        policy,
        FakeEnv(raw_rewards=[1.0] * 6),
        ReplayBuffer(max_size=1000),
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
        alpha=initial_alpha,
        auto_alpha=True,
        alpha_lr=0.05,
        target_entropy=target_entropy,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    final_alpha = float(postfix_log[0]["alpha"])
    assert direction * (final_alpha - initial_alpha) > 0.02  # 5 Adam steps of lr 0.05 on log_alpha: ~±0.05


def test_without_auto_alpha_postfix_omits_alpha_key(monkeypatch, tmp_path):
    """Regression guard: with auto_alpha=False (the default), the tqdm
    postfix must not gain an 'alpha' key at all -- alpha stays whatever was
    passed in, with no log_alpha parameter/optimizer created."""
    env = FakeEnv(raw_rewards=[1.0] * 6)
    buffer = ReplayBuffer(max_size=1000)
    policy, critic1, critic2 = make_nets()

    postfix_log = []

    class RecordingTqdm(tqdm_pkg.tqdm):
        def set_postfix(self, **kwargs):
            postfix_log.append(kwargs)
            return super().set_postfix(**kwargs)

    monkeypatch.setattr(hedge_module, "tqdm", RecordingTqdm)

    hedge_module.train(
        policy,
        env,
        buffer,
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
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
    policy, critic1, critic2 = make_nets()

    hedge_module.train(
        policy,
        env,
        buffer,
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
        lr=7e-3,
        auto_alpha=True,
        alpha_lr=None,
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
    policy, critic1, critic2 = make_nets()

    hedge_module.train(
        policy,
        env,
        buffer,
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
        lr=7e-3,
        auto_alpha=True,
        alpha_lr=0.5,
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
            None,
            None,
            None,
            None,
            None,
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

    policy, critic1, critic2 = make_nets()
    critic1.register_forward_hook(record_action)
    critic2.register_forward_hook(record_action)

    env = FakeEnv(raw_rewards=[1.0] * 6)  # 6-step episode
    buffer = ReplayBuffer(max_size=1000)

    train(
        policy,
        env,
        buffer,
        critic1,
        critic2,
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
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


def _run_delta_env(tmp_path, residual_scale, warmup_steps):
    env = DeltaEnv()
    buffer = ActionSpyBuffer(max_size=100)
    policy = RLPolicy(5, 1, residual_scale=residual_scale)
    train(
        policy,
        env,
        buffer,
        Critic(5, 1),
        Critic(5, 1),
        n_episodes=1,
        warmup_steps=warmup_steps,
        batch_size=10**6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )
    return env, buffer


@pytest.mark.parametrize("warmup_steps", [10**6, 0], ids=["warmup_random", "policy_sampled"])
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
    env, buffer = _run_delta_env(tmp_path, 0.0, warmup_steps=10**6)
    for sent, (_, pushed_action) in zip(env.step_actions, buffer.pushed):
        assert sent == pytest.approx(pushed_action, abs=1e-6)  # float32 rounding only


@pytest.mark.parametrize("warmup_steps", [10**6, 0], ids=["warmup_random", "policy_sampled"])
def test_train_gap_fraction_sends_interpolated_hedge_but_stores_squashed_action(tmp_path, warmup_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    env, buffer = PositionEnv(), ActionSpyBuffer(max_size=100)
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    train(
        policy,
        env,
        buffer,
        Critic(5, 1),
        Critic(5, 1),
        n_episodes=1,
        warmup_steps=warmup_steps,
        batch_size=10**6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

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


def test_train_gap_fraction_learns_the_rewarded_hedge_and_keeps_mode(tmp_path):
    """delta 0.8 from position 0: the hedge 0.8*w is rewarded for hitting 0.4
    (w = 0.5). A fresh policy hedges ~0.72 (w ~ 0.9). Measured over seeds 0-2:
    trained hedge within 0.021 of the target, so 0.05 is ~2.4x that."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = BanditEnv(delta=0.8, position=0.0, target=0.4)
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    before = hedge_module.deterministic_hedge(policy, env.state()).item()
    assert abs(before - 0.4) > 0.25  # the starting point is far from the target

    after = train_bandit(tmp_path, policy, env, 1)

    assert abs(after - 0.4) < 0.05
    assert policy.gap_fraction.item() == 1.0  # the mode buffer is never trained away


def test_deterministic_hedge_gap_fraction_fresh_policy_closes_90_percent_of_gap():
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    with torch.no_grad():
        policy.layer3.weight.zero_()
    states = np.array([[1.0, 0.1, 0.8, 0.2, 0.0], [1.0, 0.1, -0.5, 0.2, 0.5]])
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


# --- vector actions: ReplayBuffer and target_entropy ------------------------


@pytest.mark.parametrize("action_dim,mode,expected", [(1, "residual", -1.0), (2, "band", -2.0)])
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
    monkeypatch.setattr(
        hedge_module, "_squash", lambda dist: (torch.tanh(dist.mean), torch.zeros(dist.mean.shape[:-1] + (1,)))
    )
    train(
        RLPolicy(5, action_dim, action_mode=mode),
        PositionEnv(),
        ReplayBuffer(max_size=100),
        Critic(5, action_dim),
        Critic(5, action_dim),
        n_episodes=1,
        warmup_steps=0,
        update_every=1,
        batch_size=2,
        auto_alpha=True,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )
    assert grads and all(g == pytest.approx(-expected) for g in grads)


# --- band mode: 2-D actions in train() ---------------------------------------


@pytest.mark.parametrize("warmup_steps", [10**6, 0], ids=["warmup_random", "policy_sampled"])
def test_train_band_stores_2d_action_and_sends_band_hedge(tmp_path, warmup_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    env, buffer = PositionEnv(), ActionSpyBuffer(max_size=100)
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    train(
        policy,
        env,
        buffer,
        Critic(5, 2),
        Critic(5, 2),
        n_episodes=1,
        warmup_steps=warmup_steps,
        batch_size=10**6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )

    assert len(env.step_actions) == len(buffer.pushed) == 3
    for sent, (state, pushed_action) in zip(env.step_actions, buffer.pushed):
        assert pushed_action.shape == (2,)
        assert np.all((-1 < pushed_action) & (pushed_action < 1))
        expected = policy.hedge_ratio(torch.tensor(state), torch.tensor(pushed_action, dtype=torch.float32)).item()
        assert sent == pytest.approx(expected, abs=1e-6)


def test_train_band_hold_resends_the_envs_position_exactly(tmp_path):
    """In a closed loop the env's position is the hedge sent last step; a hold
    must send it back unchanged, or HedgingEnv charges the fixed cost."""

    class TrackingEnv:
        DELTAS = [0.5, 0.52, 0.55, 0.9]  # rebalance, hold, hold

        def __init__(self):
            self.step_actions = []

        def _state(self):
            position = self.step_actions[-1] if self.step_actions else 0.0
            return np.array([1.0, 0.0, self.DELTAS[len(self.step_actions)], 0.0, position])

        def reset(self):
            return self._state()

        def step(self, action):
            self.step_actions.append(action)
            return self._state(), 0.0, len(self.step_actions) >= 3

    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    with torch.no_grad():
        policy.layer3.weight.zero_()
        policy.layer3.bias[0] = 0.0  # b = 0.1
        policy.layer3.bias[1] = 20.0  # rebalance to delta
        policy.layer3.bias[2:] = -20.0  # tiny std: sampled ~ mean
    env = TrackingEnv()
    train(
        policy,
        env,
        ReplayBuffer(max_size=100),
        Critic(5, 2),
        Critic(5, 2),
        n_episodes=1,
        warmup_steps=0,
        batch_size=10**6,
        checkpoint_path=str(tmp_path / "policy.pth"),
    )
    first, *holds = env.step_actions
    assert first == pytest.approx(0.5, abs=1e-6)
    assert holds == [first, first]  # exact, not approx


def test_train_band_learns_to_hold_a_rewarded_position_with_auto_alpha(tmp_path):
    """Position 0.6 vs delta 0.3 is rewarded for being held: the policy starts
    with a narrow band (rebalances to ~0.30) and must widen it past the 0.3
    gap. Measured over seeds 0-2: trained hedge 0.6 to 1e-7 (a hold), asserted
    to 0.05."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = BanditEnv(delta=0.3, position=0.6, target=0.6)
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.4)
    before = hedge_module.deterministic_hedge(policy, env.state()).item()
    assert abs(before - 0.6) > 0.25

    after = train_bandit(tmp_path, policy, env, 2, auto_alpha=True)

    assert abs(after - 0.6) < 0.05
    assert policy.action_mode == "band"
