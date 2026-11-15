"""Settlement of the short American put at exercise or expiry, in cash or by delivery of the share.

The hedger is short one put and holds `position` shares when the option ends, at exercise or expiry, at price
`spot`. Settlement returns what ending costs the hedger, `Z` in the P&L `p_0 + gains - costs - Z`: the payoff
owed to the holder plus the cost of the trade that closes the share position. The share position is already
marked to market by the gains term, so selling or buying it back at `spot` changes the P&L only by its cost.

Settlement treats the put as exercised when it is in the money (`spot < strike`). A put exercised before expiry
always is, since the exercise boundary lies below the strike. A put still alive at expiry is exercised if it is
in the money and lapses otherwise, and then settlement only closes the position. When the put is exercised, the
two conventions differ only in the closing trade:
- cash: the hedger pays `K - S` and closes `position` shares;
- physical: the hedger pays `K` and receives one share worth `S`, so it closes `position + 1` shares.

Both give the same `Z` before costs. A put hedge holds about -1 share deep in the money, so physical settlement
closes almost nothing there, while cash settlement buys back about one share.
"""

import torch

from optionpricing.hedging.costs import transaction_cost


def _exercise(spot, position, strike) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `spot`, `position`, `strike` as tensors and the exercise mask, or raise ValueError."""
    # A list or array becomes float64; a tensor position keeps its dtype, device and graph.
    position = position if isinstance(position, torch.Tensor) else torch.as_tensor(position, dtype=torch.float64)
    if not position.is_floating_point():
        position = position.to(torch.float64)
    spot = torch.as_tensor(spot, dtype=position.dtype, device=position.device)
    strike = torch.as_tensor(strike, dtype=position.dtype, device=position.device)
    if not torch.all(spot > 0):
        raise ValueError("spot must be positive")
    if not torch.all(strike > 0):
        raise ValueError("strike must be positive")
    return spot, position, strike, spot < strike


def cash_settlement(
    spot, position, strike, proportional: float = 0.0, fixed: float = 0.0, kappa: float | None = None
) -> torch.Tensor:
    """Cost to the hedger of ending the short put in cash: the payoff `(K - S)^+` plus closing `position` shares.

    The hedger pays the holder `K - S` if the put is in the money, and trades `-position` shares at `spot` to
    close its hedge, paying `transaction_cost` on that trade.

    Args:
        spot: share price at exercise or expiry, > 0, a tensor or array-like.
        position: shares held when the option ends (about -1 to 0 for a put hedge), broadcastable with `spot`.
        strike: strike of each option, > 0, broadcastable with `spot`.
        proportional, fixed, kappa: the cost settings of `transaction_cost`.

    Returns:
        A tensor of costs to the hedger (>= 0), the broadcast shape of the inputs, differentiable in `position`.

    Raises:
        ValueError: if `spot` or `strike` is not positive, or the cost settings are invalid (see
            `transaction_cost`).
    """
    spot, position, strike, exercised = _exercise(spot, position, strike)
    payoff = torch.where(exercised, strike - spot, torch.zeros_like(spot))
    return payoff + transaction_cost(-position, spot, proportional, fixed, kappa)


def physical_settlement(
    spot, position, strike, proportional: float = 0.0, fixed: float = 0.0, kappa: float | None = None
) -> torch.Tensor:
    """Cost to the hedger of ending the short put by delivery: `K - S` if exercised, plus closing what is left.

    If the put is in the money, the holder delivers one share and the hedger pays `K`. That share is worth `S`,
    so it costs the hedger `K - S`, and the hedger then closes `position + 1` shares. Out of the money, the put
    lapses and the hedger closes `position` shares, as in cash settlement.

    Args:
        spot: share price at exercise or expiry, > 0, a tensor or array-like.
        position: shares held when the option ends (about -1 to 0 for a put hedge), broadcastable with `spot`.
        strike: strike of each option, > 0, broadcastable with `spot`.
        proportional, fixed, kappa: the cost settings of `transaction_cost`.

    Returns:
        A tensor of costs to the hedger (>= 0), the broadcast shape of the inputs, differentiable in `position`.

    Raises:
        ValueError: if `spot` or `strike` is not positive, or the cost settings are invalid (see
            `transaction_cost`).
    """
    spot, position, strike, exercised = _exercise(spot, position, strike)
    payoff = torch.where(exercised, strike - spot, torch.zeros_like(spot))
    held = torch.where(exercised, position + 1, position)
    return payoff + transaction_cost(-held, spot, proportional, fixed, kappa)
