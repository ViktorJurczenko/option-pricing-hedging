import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # outputs a single Q-value scalar
        )

    def forward(self, state, action):
        # The Q-value for a pair state, action
        return self.net(torch.cat([state, action], dim=-1))
