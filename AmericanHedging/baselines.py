import numpy as np


def delta_hedge(state: np.ndarray) -> float:
    """Hedge ratio = the LSM's own analytic delta (state index 2 in HedgingEnv._get_state)."""
    return state[2]


def no_hedge(state: np.ndarray) -> float:
    return 0.0
