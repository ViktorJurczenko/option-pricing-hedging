"""Fake envs, replay buffers and nets shared by the test_hedge_*.py files
(not collected by pytest: no test_ prefix)."""
import numpy as np

import AmericanHedging.Hedge as hedge_module
from AmericanHedging.Hedge import train
from AmericanHedging.ReplayBuffer import ReplayBuffer
from AmericanHedging.RLPolicy import RLPolicy
from AmericanHedging.Critic import Critic


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


def make_nets():
    return RLPolicy(5, 1), Critic(5, 1), Critic(5, 1)


class ActionSpyBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []  # (state, action)

    def push(self, state, action, reward, next_state, done):
        self.pushed.append((state.copy(), np.array(action, copy=True)))
        super().push(state, action, reward, next_state, done)


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


class BanditEnv:
    """One-step episodes from a fixed state whose reward is -(hedge - target)^2,
    so the optimal hedge is known and a learning policy must move toward it."""

    def __init__(self, delta, position, target):
        self.delta, self.position, self.target = delta, position, target

    def state(self):
        return np.array([1.0, 0.0, self.delta, 0.0, self.position], dtype=np.float32)

    def reset(self):
        return self.state()

    def step(self, action):
        return self.state(), -(float(action) - self.target) ** 2, True


def train_bandit(tmp_path, policy, env, action_dim, **kwargs):
    train(policy, env, ReplayBuffer(max_size=1000), Critic(5, action_dim), Critic(5, action_dim),
          n_episodes=400, warmup_steps=50, update_every=1, batch_size=32, lr=3e-3,
          reward_scale=10.0, checkpoint_path=str(tmp_path / "policy.pth"), **kwargs)
    return hedge_module.deterministic_hedge(policy, env.state()).item()
