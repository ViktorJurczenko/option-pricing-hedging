from collections.abc import Callable

import numpy as np


def delta_hedge(state: np.ndarray) -> float:
    """Hedge ratio = the LSM's own analytic delta (state index 2 in HedgingEnv._get_state)."""
    return state[2]


def no_hedge(state: np.ndarray) -> float:
    return 0.0


def delta_band(band: float, target: str = "delta") -> Callable[[np.ndarray], float]:
    """No-trade band around the LSM delta: hold the current position (state
    index 4) while |delta - position| <= band, otherwise rebalance.

    target: where to rebalance to once outside the band.
        "delta": all the way to delta. Suits fixed costs: every trade costs
            the same, so each one should buy as much slack as possible.
        "edge": only to the nearest edge of the band (Whalley-Wilmott
            style). Suits proportional costs: cost scales with the trade
            size, so trade the minimum needed to get back inside.
    band = 0 is daily delta hedging for either target.
    """
    if not (np.isfinite(band) and band >= 0):
        raise ValueError(f"band must be finite and >= 0, got {band}")
    if target not in ["delta", "edge"]:
        raise ValueError(f"target must be 'delta' or 'edge', got {target!r}")

    def action_fn(state: np.ndarray) -> float:
        delta, position = state[2], state[4]
        gap = position - delta
        if abs(gap) <= band:
            # Exactly the current position, not a clip or blend toward it:
            # HedgingEnv only skips the fixed cost when |trade| <= _TRADE_EPS,
            # so anything else would pay it on every hold.
            return position
        if target == "delta":
            return delta
        return delta + np.sign(gap) * band

    return action_fn
