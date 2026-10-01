import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Indices of the LSM delta and the current position in HedgingEnv._get_state's state vector.
_DELTA_IDX = 2
_POSITION_IDX = 4

# gap_fraction mode: initial fraction of the gap to delta closed each step.
# Close to 1 so the untrained policy is almost a delta hedge, but not so close
# that tanh starts on its flat end, where gradients vanish.
_INITIAL_GAP_FRACTION = 0.9

_ACTION_MODES = ["residual", "gap_fraction"]

class RLPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, residual_scale: float = 0.0,
                 action_mode: str = "residual") -> None:
        """
        action_mode: how `hedge_ratio` turns the squashed action into a hedge.
            "residual" (default): `hedge = clip(delta + residual_scale * squashed, -1, 1)`,
                a correction to the state's delta; residual_scale 0 makes the
                squashed action the hedge ratio itself.
            "gap_fraction": `hedge = position + w * (delta - position)` with
                `w = (squashed + 1) / 2` in (0, 1), the fraction of the gap to
                delta closed this step. Holding still is w ~ 0, reachable from
                any strongly negative output, which the residual form can only
                do by tracking (position - delta) exactly every step - costly
                under trading costs (BUGS.md). residual_scale is unused. The
                mean output's bias starts so that w ~ 0.9: untrained, the
                policy is almost a delta hedge.
        residual_scale: c above, for action_mode "residual".
        Both are stored as buffers, so they are saved in the checkpoint along
        with the weights.
        """
        super().__init__()
        if not (math.isfinite(residual_scale) and residual_scale >= 0):
            raise ValueError(f"residual_scale must be finite and >= 0, got {residual_scale}")
        if action_mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {_ACTION_MODES}, got {action_mode!r}")
        self.action_dim = action_dim
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim * 2) # Output mean and std for each action
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.register_buffer("gap_fraction", torch.tensor(float(action_mode == "gap_fraction")))
        if action_mode == "gap_fraction":
            with torch.no_grad():
                # squashed = tanh(mean) = 2w - 1
                self.layer3.bias[:action_dim] = math.atanh(2 * _INITIAL_GAP_FRACTION - 1)

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
        env sees this mapping (see __init__'s action_mode). In residual mode
        with residual_scale == 0 it is the identity.
        """
        if self.gap_fraction.item():
            delta    = state[..., _DELTA_IDX:_DELTA_IDX + 1]
            position = state[..., _POSITION_IDX:_POSITION_IDX + 1]
            w        = (squashed + 1) / 2
            return torch.clamp(position + w * (delta - position), -1, 1)
        if self.residual_scale.item() == 0:
            return squashed
        delta = state[..., _DELTA_IDX:_DELTA_IDX + 1]
        return torch.clamp(delta + self.residual_scale * squashed, -1, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before residual_scale existed are absolute-action
        # policies, i.e. residual_scale = 0; before gap_fraction existed, all
        # were residual-mode.
        state_dict.setdefault(prefix + "residual_scale", torch.tensor(0.0))
        state_dict.setdefault(prefix + "gap_fraction", torch.tensor(0.0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
