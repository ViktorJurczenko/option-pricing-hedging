"""SAC update math in train(): Bellman target, done mask, Polyak averaging."""
import copy

import numpy as np
import pytest
import torch

import AmericanHedging.Hedge as hedge_module
from AmericanHedging.Hedge import train
from AmericanHedging.ReplayBuffer import ReplayBuffer
from tests.unit.hedge_fakes import make_nets


# --- SAC update math: Bellman target, done mask, Polyak averaging -------------

class SequenceEnv:
    """Deterministic episodes over states k*ones(5) (k = step index): step k
    pays rewards[k] and ends the episode after the last reward."""

    def __init__(self, rewards):
        self.rewards = rewards

    def reset(self):
        self._k = 0
        return np.zeros(5, dtype=np.float32)

    def step(self, action):
        reward = self.rewards[self._k]
        self._k += 1
        return np.full(5, self._k, dtype=np.float32), reward, self._k >= len(self.rewards)


def _single_update_train(monkeypatch, tmp_path, tau, gamma=0.9, alpha=0.3, log_pi_value=2.0):
    """Runs train() on one 2-step episode (rewards 0.5 then -1.0 with done) so
    exactly one gradient update fires, at step 2. _squash is made
    deterministic (tanh of the mean, log_pi == log_pi_value) and
    ReplayBuffer.sample is made order-preserving so the Bellman target can be
    recomputed by hand. Returns what the test needs for that."""
    torch.manual_seed(0)
    np.random.seed(0)
    policy, critic1, critic2 = make_nets()
    init = {"policy": copy.deepcopy(policy), "c1": copy.deepcopy(critic1), "c2": copy.deepcopy(critic2)}

    targets = []
    real_deepcopy = copy.deepcopy
    monkeypatch.setattr(hedge_module.copy, "deepcopy", lambda m: targets.append(real_deepcopy(m)) or targets[-1])
    monkeypatch.setattr(hedge_module, "_squash",
                        lambda dist: (torch.tanh(dist.mean), torch.full(dist.mean.shape[:-1] + (1,), log_pi_value)))
    monkeypatch.setattr("AmericanHedging.ReplayBuffer.random.sample", lambda pop, k: list(pop)[:k])
    captured_targets = []
    real_mse = hedge_module.F.mse_loss
    monkeypatch.setattr(hedge_module.F, "mse_loss",
                        lambda pred, target, *a, **k: captured_targets.append(target.clone()) or real_mse(pred, target, *a, **k))

    buffer = ReplayBuffer(max_size=10)
    train(policy, SequenceEnv([0.5, -1.0]), buffer, critic1, critic2,
          n_episodes=1, warmup_steps=0, update_every=1, batch_size=2, gamma=gamma, tau=tau, alpha=alpha,
          checkpoint_path=str(tmp_path / "policy.pth"))
    return init, targets, captured_targets, buffer, critic1, critic2


def test_critic_bellman_target_uses_reward_gamma_done_mask_min_target_and_entropy(monkeypatch, tmp_path):
    gamma, alpha, c = 0.9, 0.3, 2.0
    init, targets, captured, buffer, _, _ = _single_update_train(monkeypatch, tmp_path, tau=0.01,
                                                                  gamma=gamma, alpha=alpha, log_pi_value=c)
    assert len(captured) == 2  # one update -> critic1 and critic2 losses
    s, a, r, s2, d = buffer.sample(2)
    assert d.flatten().tolist() == [0.0, 1.0]  # a non-terminal and a terminal row

    # Targets at the time of the update are still the initial critics' copies.
    with torch.no_grad():
        a2 = torch.tanh(init["policy"](s2).mean)
        q1, q2 = init["c1"](s2, a2), init["c2"](s2, a2)
        assert not torch.allclose(q1, q2)  # min() is not vacuous
        expected = r + gamma * (1 - d) * (torch.min(q1, q2) - alpha * c)

    torch.testing.assert_close(captured[0], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(captured[1], expected, atol=1e-6, rtol=0)
    # The terminal row's target is exactly its reward: nothing bootstrapped.
    assert captured[0][1].item() == pytest.approx(-1.0)
    # ...while the non-terminal row's is not just its reward.
    assert captured[0][0].item() != pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("tau", [0.0, 0.25, 1.0])
def test_target_networks_follow_critics_by_polyak_average_with_tau(monkeypatch, tmp_path, tau):
    init, targets, _, _, critic1, critic2 = _single_update_train(monkeypatch, tmp_path, tau=tau)
    target1, target2 = targets  # deep copies made in train(), in order
    for trained, target, start in [(critic1, target1, init["c1"]), (critic2, target2, init["c2"])]:
        moved = False
        for p, tp, p0 in zip(trained.parameters(), target.parameters(), start.parameters()):
            torch.testing.assert_close(tp, (1 - tau) * p0 + tau * p, atol=1e-6, rtol=0)
            moved |= not torch.equal(p, p0)
        assert moved  # the critic really was updated, so the three cases differ
    assert not any(p.requires_grad for p in target1.parameters())  # targets stay frozen


def test_critic_q_converges_to_reward_when_every_step_is_terminal(tmp_path):
    """With done=True the (1 - d) mask removes bootstrapping, so Q(s, a) -> r
    for any gamma. Without the mask Q would chase r + gamma * Q' and, at gamma
    0.99 over 600 updates, overshoot well past r. Measured |Q - r| <= 0.0043
    over seeds 0-2 and actions in [-0.5, 0.5]; 0.02 is ~4.6x that (the
    unmasked fixed point is r / (1 - gamma) = 100, so it stays far outside)."""
    torch.manual_seed(0)
    np.random.seed(0)
    reward = 1.0
    policy, critic1, critic2 = make_nets()
    train(policy, SequenceEnv([reward]), ReplayBuffer(max_size=1000), critic1, critic2,
          n_episodes=600, warmup_steps=50, update_every=1, batch_size=32, lr=1e-2, tau=0.1,
          gamma=0.99, alpha=1e-6, checkpoint_path=str(tmp_path / "policy.pth"))
    state = torch.zeros(1, 5)
    for a in (-0.5, 0.0, 0.5):
        for critic in (critic1, critic2):
            assert critic(state, torch.tensor([[a]])).item() == pytest.approx(reward, abs=0.02)
