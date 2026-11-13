import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Buffer used to remember past scenarios for the RL training
    """

    def __init__(self, max_size: int = 100_000):
        self.buffer = deque(maxlen=max_size)
        # deque automatically drops oldest entries when full

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Actions are pushed as scalars or (action_dim,) vectors; either way
        # each sampled row is one action vector.
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32).reshape(batch_size, -1),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)
