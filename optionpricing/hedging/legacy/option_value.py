from collections.abc import Callable

import numpy as np

from optionpricing.pricing.european.black_scholes import BlackScholes

# value_fn(t, S) -> option value at simulation row t for spot(s) S.
ValueFn = Callable[[int, np.ndarray], np.ndarray]


def _intrinsic(S: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S - strike, 0.0)
    return np.maximum(strike - S, 0.0)


def black_scholes_value_fn(
    strike: float, risk_free_rate: float, sigma: float, maturity: float, n_steps: int, option_type: str
) -> ValueFn:
    """Closed-form option value on the simulation's time grid.

    `n_steps` is the number of rows of a `GBM.simulate_paths` array: row t
    sits at time t*dt with dt = maturity / (n_steps - 1), and the last row is
    expiry, so the remaining time at row t is (n_steps - 1 - t) * dt. On the
    last row the value is exactly the intrinsic payoff.

    For a call on a non-dividend stock this is also the exact *American*
    value (early exercise is never optimal), which is why the hedging env
    can use it for calls.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if n_steps < 2:
        raise ValueError("n_steps (simulation rows) must be >= 2")
    dt = maturity / (n_steps - 1)

    def value_fn(t: int, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        tau = (n_steps - 1 - t) * dt
        if tau <= 0:
            return _intrinsic(S, strike, option_type)
        bs = BlackScholes(
            sigma=sigma, start_price=S, strike=strike, risk_free_rate=risk_free_rate, option_type=option_type
        )
        return bs.calculate_option_price(T=tau)

    return value_fn


def binomial_value_fn(
    start_price: float,
    strike: float,
    risk_free_rate: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    option_type: str,
    substeps: int = 4,
) -> ValueFn:
    """American option value on the simulation's time grid, from a single
    Cox-Ross-Rubinstein tree rooted at `start_price`.

    The tree has `substeps` steps per simulation step, and the full node
    vector is recorded at every simulation row, so a path's spot at row t can
    be priced by interpolating (in log-price, where CRR nodes are evenly
    spaced) between that row's nodes. More substeps widen each row's node
    range (paths far from S0 early on would otherwise fall outside it) at
    O((n_steps * substeps)^2) build cost. Spots outside a row's node range are
    clamped to the edge node's value, then floored at the intrinsic value
    (an American option is never worth less than immediate exercise).

    substeps=4 default: on the train-hedge-legacy grid (365 steps, sigma=0.2),
    the tree's call value stays within ~0.01 of the exact Black-Scholes value
    at every row checked (0.001 at t=0), and the build (~1.5k tree steps)
    takes well under a second - accurate enough for a reward term whose
    per-step scale is ~1e-2 * K, without the O(n^2) cost of more substeps.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    if n_steps < 2:
        raise ValueError("n_steps (simulation rows) must be >= 2")

    dt_tree = maturity / (n_steps - 1) / substeps
    n_tree = (n_steps - 1) * substeps
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1 / u
    p = (np.exp(risk_free_rate * dt_tree) - d) / (u - d)
    discount = np.exp(-risk_free_rate * dt_tree)
    log_u = np.log(u)

    def node_log_moneyness(i: int) -> np.ndarray:
        # log(S / start_price) at the i+1 nodes of tree step i, ascending.
        return (2 * np.arange(i + 1) - i) * log_u

    values = _intrinsic(start_price * np.exp(node_log_moneyness(n_tree)), strike, option_type)
    grid = {n_steps - 1: values}
    for i in range(n_tree - 1, -1, -1):
        continuation = discount * (p * values[1:] + (1 - p) * values[:-1])
        exercise = _intrinsic(start_price * np.exp(node_log_moneyness(i)), strike, option_type)
        values = np.maximum(exercise, continuation)
        if i % substeps == 0:
            grid[i // substeps] = values

    def value_fn(t: int, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        intrinsic = _intrinsic(S, strike, option_type)
        if t >= n_steps - 1:
            return intrinsic
        x = np.log(S / start_price)
        interpolated = np.interp(x, node_log_moneyness(t * substeps), grid[t])
        return np.maximum(interpolated, intrinsic)

    return value_fn


def make_option_value_fn(
    option_type: str,
    start_price: float,
    strike: float,
    risk_free_rate: float,
    sigma: float,
    maturity: float,
    n_steps: int,
) -> ValueFn:
    """Value function for HedgingEnv's accounting-P&L reward: closed-form
    Black-Scholes for calls (exact for an American call without dividends),
    a binomial-tree grid for puts (where early exercise matters)."""
    if option_type == "call":
        return black_scholes_value_fn(strike, risk_free_rate, sigma, maturity, n_steps, "call")
    if option_type == "put":
        return binomial_value_fn(start_price, strike, risk_free_rate, sigma, maturity, n_steps, "put")
    raise ValueError("option_type must be 'call' or 'put'")
