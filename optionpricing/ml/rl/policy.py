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

# band mode: initial band half-width, as a fraction of band_max, and initial
# rebalancing target (0 = band edge, 1 = delta). A narrow band rebalancing
# almost to delta, so the untrained policy is close to a delta hedge, again
# kept off tanh's flat ends.
_INITIAL_BAND_FRACTION = 0.1
_INITIAL_BAND_TARGET = 0.9

_ACTION_MODES = ["residual", "gap_fraction", "band"]


class RLPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        residual_scale: float = 0.0,
        action_mode: str = "residual",
        band_max: float = 0.2,
    ) -> None:
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
            "band": a learned no-trade band, action_dim must be 2. The
                squashed action (a0, a1) gives the band half-width
                `b = band_max * (a0 + 1) / 2` and the rebalancing target
                `lam = (a1 + 1) / 2` (0 = band edge, 1 = delta). The position is
                held exactly while `|position - delta| <= b`, otherwise the
                hedge is `delta + sign(position - delta) * (1 - lam) * b`. A
                constant action is `baselines.delta_band`. Unlike w ~ 0 in
                gap_fraction mode, a hold is an exact zero trade, so it avoids
                HedgingEnv's fixed cost. The biases start at
                b = 0.1 * band_max and lam = 0.9.
        residual_scale: c above, for action_mode "residual".
        band_max: the widest band, for action_mode "band".
        All three are stored as buffers, so they are saved in the checkpoint
        along with the weights. band_max is stored as 0 outside band mode.
        """
        super().__init__()
        if not (math.isfinite(residual_scale) and residual_scale >= 0):
            raise ValueError(f"residual_scale must be finite and >= 0, got {residual_scale}")
        if action_mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {_ACTION_MODES}, got {action_mode!r}")
        if action_mode == "band":
            if action_dim != 2:
                raise ValueError(f"action_mode 'band' needs action_dim 2 (band width, target), got {action_dim}")
            if not (math.isfinite(band_max) and band_max > 0):
                raise ValueError(f"band_max must be finite and > 0, got {band_max}")
        self.action_dim = action_dim
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim * 2)  # Output mean and std for each action
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.register_buffer("gap_fraction", torch.tensor(float(action_mode == "gap_fraction")))
        self.register_buffer("band_max", torch.tensor(float(band_max if action_mode == "band" else 0.0)))
        # squashed = tanh(mean) = 2 * fraction - 1 for each initial fraction below
        if action_mode == "gap_fraction":
            with torch.no_grad():
                self.layer3.bias[:action_dim] = math.atanh(2 * _INITIAL_GAP_FRACTION - 1)
        elif action_mode == "band":
            with torch.no_grad():
                self.layer3.bias[0] = math.atanh(2 * _INITIAL_BAND_FRACTION - 1)
                self.layer3.bias[1] = math.atanh(2 * _INITIAL_BAND_TARGET - 1)
        self._sync_action_mode()

    def _sync_action_mode(self) -> None:
        """Cache the mode the buffers encode: hedge_ratio runs every env step,
        and reading a buffer's value (.item()) each time isn't free. Called
        again after a checkpoint load, which can change the mode."""
        if self.band_max.item() > 0:
            self._action_mode = "band"
        else:
            self._action_mode = "gap_fraction" if self.gap_fraction.item() else "residual"

    @property
    def action_mode(self) -> str:
        """The mode the buffers encode, so it follows a loaded checkpoint."""
        return self._action_mode

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
        delta = state[..., _DELTA_IDX : _DELTA_IDX + 1]
        position = state[..., _POSITION_IDX : _POSITION_IDX + 1]
        mode = self.action_mode
        if mode == "band":
            b = self.band_max * (squashed[..., 0:1] + 1) / 2
            lam = (squashed[..., 1:2] + 1) / 2
            gap = position - delta
            rebalanced = torch.clamp(delta + torch.sign(gap) * (1 - lam) * b, -1, 1)
            # The position itself, not a clamp or blend of it: HedgingEnv only
            # skips the fixed cost when |trade| <= _TRADE_EPS.
            return torch.where(gap.abs() <= b, position, rebalanced)
        if mode == "gap_fraction":
            w = (squashed + 1) / 2
            return torch.clamp(position + w * (delta - position), -1, 1)
        if self.residual_scale.item() == 0:
            return squashed
        return torch.clamp(delta + self.residual_scale * squashed, -1, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before residual_scale existed are absolute-action
        # policies, i.e. residual_scale = 0; before gap_fraction existed, all
        # were residual-mode; before band_max existed, none were band-mode.
        state_dict.setdefault(prefix + "residual_scale", torch.tensor(0.0))
        state_dict.setdefault(prefix + "gap_fraction", torch.tensor(0.0))
        state_dict.setdefault(prefix + "band_max", torch.tensor(0.0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._sync_action_mode()


def checkpoint_action_dim(state_dict: dict) -> int:
    """Action dimension of a saved RLPolicy, read off its output layer (mean
    and log-std per action), so a loader can build the matching network."""
    return state_dict["layer3.bias"].shape[0] // 2
