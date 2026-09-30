import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Index of the LSM delta in HedgingEnv._get_state's state vector.
_DELTA_IDX = 2

class RLPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, residual_scale: float = 0.0) -> None:
        """
        residual_scale: c in `hedge = clip(delta + c * squashed, -1, 1)` (see
            `hedge_ratio`), so the policy learns a correction to the state's
            delta instead of the whole hedge ratio. 0 (default) disables it:
            the squashed action is the hedge ratio itself. Stored as a buffer
            so it is saved in the checkpoint along with the weights.
        """
        super().__init__()
        if not (math.isfinite(residual_scale) and residual_scale >= 0):
            raise ValueError(f"residual_scale must be finite and >= 0, got {residual_scale}")
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim * 2) # Output mean and std for each action
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)  # To ensure std > 0

        return torch.distributions.Normal(mean, std)

    def hedge_ratio(self, state: torch.Tensor, squashed: torch.Tensor) -> torch.Tensor:
        """Map a tanh-squashed action in (-1, 1) to the hedge ratio sent to the env.

        The critic and the entropy term work on `squashed` itself; only the
        env sees this mapping. With residual_scale == 0 it is the identity.
        """
        if self.residual_scale.item() == 0:
            return squashed
        delta = state[..., _DELTA_IDX:_DELTA_IDX + 1]
        return torch.clamp(delta + self.residual_scale * squashed, -1, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before residual_scale existed are absolute-action
        # policies, i.e. residual_scale = 0.
        state_dict.setdefault(prefix + "residual_scale", torch.tensor(0.0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
