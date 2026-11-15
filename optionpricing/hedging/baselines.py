"""Baseline hedging policies for the short American put, with the interface of the Deep Hedging network.

Every baseline is a policy `policy(inputs, position) -> new_position` (see `pnl.py`), so it is booked by the same
`hedge_pnl` as the network and scored by the same functions of `risk.py`. The baselines measure what hedging
achieves without learning:
- `no_hedge`: never trades, the worst case a hedge must beat.
- `delta_hedge`: holds the pricing model's delta. Fed the surrogate's delta it is the surrogate delta hedge, fed
  the tree's delta it is the tree delta hedge. The policy is the same: only the inputs differ.
- `band_hedge`: a no-trade band of fixed half-width around delta. It holds while the position is within the band
  and trades back to its edge or to delta otherwise. Its width is chosen on validation paths by `tune`.
- `whalley_wilmott`: the band of Whalley and Wilmott (1997), whose half-width follows from the cost rate, the
  gamma and a risk aversion. Its risk aversion is chosen by `tune` too.

The baselines read their inputs from one layout, `BASELINE_FEATURES`, built by `baseline_inputs`. The network's
features are its own and do not need to follow it.
"""

import math
from collections.abc import Callable, Sequence

import torch

from optionpricing.hedging.pnl import Policy

# The rows of the `inputs` tensor every baseline reads, in this order. Units:
# - spot: share price, in dollars.
# - delta: dV/dS of the put, in shares per option (in [-1, 0] for a put).
# - gamma: d2V/dS2 of the put, in shares per dollar (per option, not per unit of strike).
# - time_left: time to expiry `T - t`, in years.
# - rate: the risk-free rate, per year.
BASELINE_FEATURES = ("spot", "delta", "gamma", "time_left", "rate")
SPOT, DELTA, GAMMA, TIME_LEFT, RATE = range(len(BASELINE_FEATURES))
BAND_TARGETS = ("edge", "delta")


def baseline_inputs(spot, delta, gamma, time_left, rate) -> torch.Tensor:
    """Stack the quantities the baselines read into the `inputs` tensor `hedge_pnl` slices, in float64.

    `spot` sets the shape `(nb_scenarios, n_days)`, time last as in `hedge_pnl`. The others broadcast to it, so a
    rate per scenario is given as `rate[:, None]` and a value shared by every scenario as a scalar. Values past a
    scenario's `τ` may be NaN: `hedge_pnl` never shows them to the policy.

    Args:
        spot: share prices in dollars, shape `(nb_scenarios, n_days)`.
        delta: delta of the put on each day, in shares.
        gamma: gamma of the put on each day, `d2V/dS2` in shares per dollar. A gamma per unit of strike, as
            `price_put_batch` and `PutSurrogate` return it, is divided by the strike first.
        time_left: time to expiry on each day, in years.
        rate: risk-free rate.

    Returns:
        A float64 tensor of shape `(nb_scenarios, len(BASELINE_FEATURES), n_days)`, rows in the order of
        `BASELINE_FEATURES`.

    Raises:
        ValueError: if `spot` is not 2-D, or another input does not broadcast to its shape.
    """
    spot = torch.as_tensor(spot, dtype=torch.float64)
    if spot.ndim != 2:
        raise ValueError(f"spot must have shape (nb_scenarios, n_days), got {tuple(spot.shape)}")
    rows = [spot]
    for name, value in (("delta", delta), ("gamma", gamma), ("time_left", time_left), ("rate", rate)):
        value = torch.as_tensor(value, dtype=torch.float64)
        try:
            rows.append(value.expand(spot.shape))
        except RuntimeError:
            raise ValueError(
                f"{name} must broadcast to spot's shape {tuple(spot.shape)}, got {tuple(value.shape)}"
            ) from None
    return torch.stack(rows, dim=1)


def no_hedge(inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Hold no shares: the put is left unhedged."""
    return torch.zeros_like(position)


def delta_hedge(inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Hold the delta of the put, rebalancing fully at every hedge date.

    The hedger is short the put, whose value moves by `Δ dS`, and the shares move by `δ dS`. Holding `δ = Δ`
    shares makes the two cancel to first order: the P&L change is `(δ - Δ) dS = 0`. A put has `Δ ≤ 0`, so the
    hedge is short shares.
    """
    return inputs[:, DELTA]


def _check_target(target: str) -> None:
    """Raise ValueError unless `target` is one of `BAND_TARGETS`."""
    if target not in BAND_TARGETS:
        raise ValueError(f"target must be one of {BAND_TARGETS}, got {target!r}")


def _band(delta: torch.Tensor, half_width: torch.Tensor, position: torch.Tensor, target: str) -> torch.Tensor:
    """Hold `position` while it is within `half_width` of `delta`, else trade to the band's edge or to delta.

    Inside the band the returned position is exactly the one held, not a value close to it, so `hedge_pnl` sees a
    zero trade and charges no fixed fee. On the band's edge the position counts as inside.
    """
    if target == "edge":
        # clamp returns its input unchanged when it lies between the bounds
        return torch.clamp(position, delta - half_width, delta + half_width)
    return torch.where((position - delta).abs() <= half_width, position, delta)


def band_hedge(half_width: float, target: str = "edge") -> Policy:
    """No-trade band of fixed half-width around delta.

    Under costs, rebalancing to delta at every date pays for many small trades that hedge little. The band holds
    while `|δ_{k-1} - Δ_k| ≤ half_width`. Outside it, the policy trades:
    - to the nearest edge (`target="edge"`), the least trade that brings the position back inside. Suits
      proportional costs, which grow with the size of the trade.
    - to delta (`target="delta"`), buying as much slack as possible before the next trade. Suits a fixed fee,
      which is the same for any trade size.
    A half-width of 0 is the delta hedge, for either target.

    Args:
        half_width: half-width of the band, in shares, finite and >= 0.
        target: `"edge"` or `"delta"`, where to trade to outside the band.

    Returns:
        The policy.

    Raises:
        ValueError: if `half_width` is negative or not finite, or `target` is unknown.
    """
    if not (math.isfinite(half_width) and half_width >= 0):
        raise ValueError(f"half_width must be finite and >= 0, got {half_width}")
    _check_target(target)

    def policy(inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        width = torch.as_tensor(half_width, dtype=inputs.dtype, device=inputs.device)
        return _band(inputs[:, DELTA], width, position, target)

    return policy


def whalley_wilmott_half_width(inputs: torch.Tensor, risk_aversion: float, proportional: float) -> torch.Tensor:
    """Half-width of the Whalley-Wilmott band, in shares, for each scenario of an `inputs` slice.

        H = (3/2 · e^{-r (T - t)} · ε S Γ² / γ)^{1/3}

    with `ε` the proportional cost rate, `S` the share price, `Γ` the gamma in shares per dollar and `γ` the
    risk aversion of an exponential utility, in 1/dollars. It is the leading term of the optimal band of a hedger
    with exponential utility as the cost rate goes to 0 (Whalley and Wilmott 1997, "An asymptotic analysis of an
    optimal hedging model for option pricing with transaction costs", Mathematical Finance 7(3)). The band widens
    where gamma is large, since delta then drifts away fast and rebalancing to it costs the most. It was derived
    for a European option: here it is used with the American put's gamma, which is 0 in the exercise region, so
    the band closes there.

    Args:
        inputs: one hedge date of `baseline_inputs`, shape `(nb_scenarios, len(BASELINE_FEATURES))`.
        risk_aversion: `γ`, finite and > 0. A larger `γ` narrows the band.
        proportional: `ε`, the proportional cost rate of `transaction_cost`, finite and >= 0. 0 gives a zero
            width (the delta hedge).

    Returns:
        A tensor of shape `(nb_scenarios,)`.

    Raises:
        ValueError: if `risk_aversion` or `proportional` is out of range.
    """
    if not (math.isfinite(risk_aversion) and risk_aversion > 0):
        raise ValueError(f"risk_aversion must be finite and > 0, got {risk_aversion}")
    if not (math.isfinite(proportional) and proportional >= 0):
        raise ValueError(f"proportional must be finite and >= 0, got {proportional}")
    discount = torch.exp(-inputs[:, RATE] * inputs[:, TIME_LEFT])
    cube = 1.5 * discount * proportional * inputs[:, SPOT] * inputs[:, GAMMA] ** 2 / risk_aversion
    # The cube is >= 0 for any valid input, so the real cube root is the plain power. Its gradient is infinite at
    # 0 (zero gamma or zero cost): fine for a fixed policy, but a parameter tuned by gradient through it would get
    # NaN there.
    return cube ** (1 / 3)


def whalley_wilmott(risk_aversion: float, proportional: float, target: str = "edge") -> Policy:
    """No-trade band around delta with the Whalley-Wilmott half-width (`whalley_wilmott_half_width`).

    Outside the band the policy trades to its edge by default, as in Whalley and Wilmott's optimal strategy for
    proportional costs. `target="delta"` trades to delta instead, as `band_hedge` does.

    Args:
        risk_aversion: `γ` of `whalley_wilmott_half_width`, finite and > 0, in 1/dollars.
        proportional: the cost rate the band is built for, finite and >= 0. It is a property of the policy, not
            the cost charged: a policy can be scored at other costs than the ones it was built for.
        target: `"edge"` or `"delta"`, where to trade to outside the band.

    Returns:
        The policy.

    Raises:
        ValueError: if `risk_aversion` or `proportional` is out of range, or `target` is unknown.
    """
    _check_target(target)
    # Validate now rather than at the first hedge date, through the same function the policy calls.
    whalley_wilmott_half_width(torch.zeros(1, len(BASELINE_FEATURES)), risk_aversion, proportional)

    def policy(inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        width = whalley_wilmott_half_width(inputs, risk_aversion, proportional)
        return _band(inputs[:, DELTA], width, position, target)

    return policy


def tune(
    make_policy: Callable[[float], Policy], grid: Sequence[float], score: Callable[[Policy], float]
) -> tuple[float, list[float]]:
    """Pick the parameter of a baseline family that scores best, for example a band's width on validation paths.

    Each value of `grid` builds a policy with `make_policy`, which `score` rates (higher is better, as `cvar` of
    the P&L on validation paths). The best value is returned with every score, so a caller can check that the
    optimum is not on the edge of the grid. Ties go to the first value of the grid.

    Args:
        make_policy: builds a policy from one parameter value, e.g. `band_hedge` or
            `lambda g: whalley_wilmott(g, 0.01)`.
        grid: the parameter values tried, at least one.
        score: rates a policy, higher is better. It closes over the validation paths and the `hedge_pnl`
            settings, so those stay the caller's choice.

    Returns:
        `(best, scores)`: the best value of `grid` and the score of each value, in grid order.

    Raises:
        ValueError: if `grid` is empty or a score is not a finite number.
    """
    if len(grid) == 0:
        raise ValueError("grid must hold at least one parameter value")
    scores = []
    for value in grid:
        result = float(score(make_policy(value)))
        if not math.isfinite(result):
            raise ValueError(f"score of parameter {value} is {result}, not a finite number")
        scores.append(result)
    best = max(range(len(scores)), key=scores.__getitem__)  # max returns the first of equal scores
    return grid[best], scores
