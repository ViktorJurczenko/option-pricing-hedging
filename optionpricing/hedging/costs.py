"""Transaction costs of hedge trades: a proportional rate plus a fixed fee per trade.

The P&L charges every trade through `transaction_cost`, so training, evaluation and the baselines pay the
same costs. The fixed fee `f * 1[x != 0]` has no gradient (the indicator is flat everywhere except at 0), so
it is straight-through: the forward pass charges the exact fee and the backward pass uses the slope of the
smooth look-alike `s(x) = 1 - exp(-|x| / kappa)`. The cost is exact and only its gradient is approximated.
"""

import math

import torch


def _check_rate(name: str, value: float) -> None:
    """Raise ValueError unless `value` is finite and >= 0; NaN fails the comparison and is rejected too."""
    if not (value >= 0 and math.isfinite(value)):
        raise ValueError(f"{name} must be finite and >= 0, got {value}")


def transaction_cost(trade, spot, proportional: float = 0.0, fixed: float = 0.0, kappa: float | None = None):
    """Cost of trading `trade` shares at price `spot`, `proportional * |trade| * spot + fixed * 1[trade != 0]`.

    Elementwise, with `trade` and `spot` broadcast together, so one call prices a whole batch of trades. The
    fee is charged in full for any non-zero trade, however small. Its gradient in `trade` is
    `fixed * sign(trade) * exp(-|trade| / kappa) / kappa`, largest for tiny trades and 0 at `trade = 0`.
    `kappa` only shapes that gradient, never the cost.

    Args:
        trade: shares bought (> 0) or sold (< 0), a tensor or array-like.
        spot: price per share, broadcastable with `trade`.
        proportional: cost per dollar traded, >= 0 (0.01 is 1 cent per dollar).
        fixed: fee per non-zero trade, in dollars, >= 0.
        kappa: width in shares of the fee's smooth look-alike, > 0. Needed when `fixed > 0` and `trade`
            requires a gradient. Without a gradient (evaluation), it can be left out.

    Returns:
        A tensor of costs (>= 0), the broadcast shape of `trade` and `spot`, differentiable in both.

    Raises:
        ValueError: if a rate is negative or not finite, `kappa` is not positive and finite, or `kappa` is
            missing when the fee needs a gradient.
    """
    _check_rate("proportional", proportional)
    _check_rate("fixed", fixed)
    if kappa is not None and not (kappa > 0 and math.isfinite(kappa)):
        raise ValueError(f"kappa must be positive and finite, got {kappa}")
    # A list or array becomes float64; a tensor keeps its dtype, device and graph.
    trade = trade if isinstance(trade, torch.Tensor) else torch.as_tensor(trade, dtype=torch.float64)
    if not trade.is_floating_point():
        trade = trade.to(torch.float64)
    spot = torch.as_tensor(spot, dtype=trade.dtype, device=trade.device)

    cost = proportional * trade.abs() * spot
    if fixed == 0:
        return cost
    hard = (trade != 0).to(trade.dtype)  # the exact indicator, no gradient
    if kappa is None:
        if trade.requires_grad and torch.is_grad_enabled():
            raise ValueError("kappa is needed for the fixed fee's gradient when trade requires a gradient")
        return cost + fixed * hard
    soft = 1 - torch.exp(-trade.abs() / kappa)
    # Forward: soft - soft.detach() is exactly 0, so the indicator is exact. Backward: only soft has a gradient.
    return cost + fixed * (hard + (soft - soft.detach()))
