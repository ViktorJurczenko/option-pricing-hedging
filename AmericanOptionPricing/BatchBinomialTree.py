from typing import NamedTuple

import numpy as np

from EuropeanOptionPricing.BlackScholes import black_scholes_price

METHODS = ("crr", "bbs", "bbsr")

class TreeResult(NamedTuple):
    """Per-option outputs of `binomial_put_batch`, all per unit of strike.

    price: option value / K. delta: dV/dS (the same number for any K, since
    V is homogeneous of degree 1 in (S, K)). gamma: d2V/dS2 times K.
    """
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray

def binomial_put_batch(
        moneyness,
        maturity,
        risk_free_rate,
        sigma,
        nb_steps: int = 1000,
        method: str = "bbsr",
        american: bool = True,
        chunk_size: int | None = None
    ) -> TreeResult:
    """
    Price a batch of puts with Cox-Ross-Rubinstein trees, all options stepped
    together. Every option gets its own tree (its own u, d, p, discount),
    with the same number of steps.

    Vocabulary: an N-step tree has N + 1 *levels*, one per time step. Level i
    is time i * dt (dt = maturity / N): level 0 is today (one node, the
    current spot), level N is expiry, and level i holds the i + 1 spots the
    stock can reach after i up-or-down moves. Backward induction computes the
    option value level by level, from expiry back to today. Here each level
    is one numpy array of shape (n_options, i + 1), so the Python overhead is
    paid once per level instead of once per option.

    Everything is per unit of strike (K = 1): the option is described by its
    moneyness m = S / K. Scale price and gamma back with K * price, gamma / K.

    Methods:
        "crr": plain CRR, the same recursion as `BinomialTree`. Its price
            zig-zags as nb_steps changes (odd and even trees place a node
            on or off the strike), which would be structured noise in labels.
        "bbs": binomial Black-Scholes. The last tree step (level N - 1 to
            expiry) is valued with the European Black-Scholes formula over
            that last dt instead of the tree, which smooths out the payoff's
            kink at the strike, where the zig-zag comes from.
        "bbsr": "bbs" with Richardson extrapolation, 2 * P(N) - P(N / 2).
            The BBS error shrinks roughly like c / N, so this combination
            cancels the leading error term. Needs an even nb_steps. For a
            European put the remainder then shrinks like 1 / N^2; for an
            American put it still shrinks roughly like 1 / N (the early
            exercise boundary falls between nodes differently for each N,
            which no power series in 1 / N captures), but about 10 times
            smaller than BBS's error and without CRR's zig-zag.

    Delta and gamma are finite differences over the nodes of levels 1 and 2
    (they sit dt and 2 dt after today, an O(dt) offset that vanishes as
    nb_steps grows); "bbsr" extrapolates them like the price.

    Args:
        moneyness, maturity, risk_free_rate, sigma: arrays (or scalars) that
            broadcast together; the outputs have their broadcast shape.
        nb_steps: tree steps per option (for "bbsr", of the finer tree).
        method: "crr", "bbs" or "bbsr" (see above).
        american: allow early exercise (False gives the European put, whose
            exact value is known, which tests the tree itself).
        chunk_size: options stepped together per pass (None: the whole batch
            at once). Only affects speed and memory, never the results.

    Returns:
        TreeResult(price, delta, gamma).

    Raises:
        ValueError: for an unknown method, a non-finite (NaN or infinite)
            parameter, moneyness/maturity/sigma <= 0, too
            few steps (gamma needs levels 1 and 2 below the root, after the
            BBS last step is taken off and N halved), an odd nb_steps with
            "bbsr", chunk_size < 1, or a risk-neutral probability outside
            (0, 1) (a step so long that the rate's drift beats the
            volatility's up/down move).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    # one (4, *batch_shape) array: rows are moneyness, maturity, rate, sigma
    params = np.stack(np.broadcast_arrays(moneyness, maturity, risk_free_rate, sigma)).astype(float)
    # NaN passes every `<= 0` check and would silently give NaN prices
    if not np.all(np.isfinite(params)):
        raise ValueError("moneyness, maturity, risk_free_rate and sigma must be finite")
    m, T, r, sigma = params
    if np.any(m <= 0):
        raise ValueError("moneyness must be > 0")
    if np.any(T <= 0):
        raise ValueError("maturity must be > 0")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be > 0")
    if method == "bbsr" and nb_steps % 2:
        raise ValueError(f"nb_steps must be even for 'bbsr', got {nb_steps}")
    coarsest = nb_steps // 2 if method == "bbsr" else nb_steps
    min_steps = 2 if method == "crr" else 3
    if coarsest < min_steps:
        raise ValueError(f"method {method!r} needs nb_steps >= "
                         f"{min_steps * (2 if method == 'bbsr' else 1)}, got {nb_steps}")

    batch_shape = params.shape[1:]
    params = params.reshape(4, -1)  # flat batch: column k is option k
    n_options = params.shape[1]
    step = chunk_size or max(n_options, 1)
    out = np.empty((3, n_options))  # rows: price, delta, gamma
    for start in range(0, n_options, step):
        part = params[:, start:start + step]
        if method == "bbsr":
            out[:, start:start + step] = (2 * _induct(part, nb_steps, bbs=True, american=american)
                                          - _induct(part, nb_steps // 2, bbs=True, american=american))
        else:
            out[:, start:start + step] = _induct(part, nb_steps, bbs=method == "bbs", american=american)
    return TreeResult(*out.reshape(3, *batch_shape))

def _induct(params, n, bbs, american):
    """Backward induction over n-step trees for a (4, n_options) parameter
    array (rows: moneyness, maturity, rate, sigma). Returns a
    (3, n_options) array, rows price, delta, gamma."""
    # Each parameter as a column, shape (n_options, 1), so it broadcasts
    # against a level's (n_options, nodes) arrays: option k's row of nodes
    # is combined with option k's own u, p, discount.
    m, T, r, sigma = params[:, :, None]
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    growth = np.exp(r * dt)
    p = (growth - 1 / u) / (u - 1 / u)  # risk-neutral probability of an up move
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("risk-neutral probability outside (0, 1): use more steps")
    # discounted up/down weights, so each level costs two products, not three
    up, down = p / growth, (1 - p) / growth

    # Spots at the last level the tree reaches, ascending: node j has j up
    # moves, so S = m * u ** (2j - level). BBS stops one level before expiry.
    level = n - 1 if bbs else n
    spot = m * u ** (2 * np.arange(level + 1) - level)
    exercise = np.maximum(1 - spot, 0)
    if not bbs:
        values = exercise  # at expiry the put is worth its payoff
    else:
        # The European value can fall below the payoff (a deep in-the-money
        # put is worth less held than exercised), so an American put takes
        # the larger of the two.
        values = black_scholes_price(spot, 1.0, dt, r, sigma, "put")
        if american:
            values = np.maximum(values, exercise)

    def back_one_level(spot, values):
        values = up * values[:, 1:] + down * values[:, :-1]  # discounted expected value of holding
        spot = spot[:, :-1] * u  # node j at level i is node j at level i+1, one down move earlier
        if american:
            values = np.maximum(values, 1 - spot)  # or exercise now, if that pays more
        return spot, values

    for _ in range(level - 2):
        spot, values = back_one_level(spot, values)
    s2, v2 = spot, values  # level 2: three nodes
    s1, v1 = back_one_level(s2, v2)  # level 1: two nodes
    _, v0 = back_one_level(s1, v1)  # level 0: today

    delta = (np.diff(v1) / np.diff(s1))[:, 0]
    slopes = np.diff(v2) / np.diff(s2)  # (n_options, 2): slopes between nodes 0-1 and 1-2
    gamma = np.diff(slopes)[:, 0] / ((s2[:, 2] - s2[:, 0]) / 2)
    return np.stack([v0[:, 0], delta, gamma])
