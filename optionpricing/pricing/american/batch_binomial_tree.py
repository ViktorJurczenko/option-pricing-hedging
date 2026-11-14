from typing import NamedTuple

import numpy as np

from optionpricing.pricing.european.black_scholes import black_scholes_price

METHODS = ("crr", "bbs", "bbsr")
# price - (1 - m) at or below this means "exercised today": the root's
# exercise value is 1 - m computed from a spot rebuilt by multiplications,
# so it can differ from 1 - m by rounding (~1e-16)
_EXERCISE_TOL = 1e-12


class TreeResult(NamedTuple):
    """Per-option outputs of `price_put_batch`, all per unit of strike.

    price: option value / K. delta: dV/dS (the same number for any K, since
    V is homogeneous of degree 1 in (S, K)). gamma: d2V/dS2 times K.
    exercised: True where the put is exercised today (worth its payoff
    1 - m, with m < 1), which fixes price, delta and gamma to 1 - m, -1 and
    0 exactly; always False for a European put.
    """

    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    exercised: np.ndarray


def price_put_batch(
    moneyness,
    maturity,
    risk_free_rate,
    sigma,
    nb_steps: int = 1000,
    method: str = "bbsr",
    american: bool = True,
    chunk_size: int | None = None,
) -> TreeResult:
    """
    Price a batch of puts, with delta and gamma, on binomial trees.

    Each option gets its own tree, and all are stepped together. Everything
    is per unit of strike (K = 1), so an option is given by its moneyness
    m = S / K: multiply the price by K and divide gamma by K.

        res = price_put_batch(m, T, r, sigma)  # arrays that broadcast together
        res.price, res.delta, res.gamma, res.exercised

    Args:
        moneyness, maturity, risk_free_rate, sigma: arrays (or scalars) that
            broadcast together; the outputs have their broadcast shape.
        nb_steps: tree steps per option (for "bbsr", of the finer tree).
        method: "crr" (plain Cox-Ross-Rubinstein), "bbs" (Black-Scholes over
            the last step, which removes CRR's odd/even zig-zag) or "bbsr"
            (BBS with Richardson extrapolation, 2 P(N) - P(N/2); the most
            accurate, needs an even nb_steps).
        american: allow early exercise (False gives the European put).
        chunk_size: options stepped together per pass (None: all at once).
            Only affects speed and memory, never the results.

    Returns:
        TreeResult(price, delta, gamma, exercised). A put exercised today
        gets its exact values: price 1 - m, delta -1, gamma 0.

    Raises:
        ValueError: for an unknown method, a non-finite parameter,
            moneyness/maturity/sigma <= 0, too few steps, an odd nb_steps
            with "bbsr", chunk_size < 1, or a step so long that the
            risk-neutral probability leaves (0, 1).
    """
    params = _validated_params(moneyness, maturity, risk_free_rate, sigma, nb_steps, method, chunk_size)
    batch_shape = params.shape[1:]
    params = params.reshape(4, -1)  # flat batch: column k is option k
    n_options = params.shape[1]
    step = chunk_size or max(n_options, 1)
    out = np.empty((3, n_options))  # rows: price, delta, gamma
    for start in range(0, n_options, step):
        part = params[:, start : start + step]
        if method == "bbsr":
            fine, _ = _induct(part, nb_steps, bbs=True, american=american)
            coarse, _ = _induct(part, nb_steps // 2, bbs=True, american=american)
            out[:, start : start + step] = 2 * fine - coarse
        else:
            out[:, start : start + step], _ = _induct(part, nb_steps, bbs=method == "bbs", american=american)
    # If the put is exercised today, it is worth exactly 1 - m around today's
    # spot: delta -1, gamma 0. The finite differences can't see that when
    # m is just inside the exercise region, since their outer nodes reach
    # into the continuation region (gamma 0.69 instead of 0 measured), and
    # "bbsr" extrapolates that error past a put's bounds (delta -1.01, gamma
    # -0.19). A "bbsr" price below 1 - m (one tree exercises, the other
    # doesn't) means the same thing. `m < 1` keeps a deep out-of-the-money
    # put, worth ~0 = its payoff, out of this case. (A European put has no
    # exercise region: it can be worth less than its payoff.)
    exercised = np.zeros(n_options, dtype=bool)
    if american:
        intrinsic = 1 - params[0]
        exercised = (intrinsic > 0) & (out[0] <= intrinsic + _EXERCISE_TOL)
        out[0, exercised] = intrinsic[exercised]
        out[1, exercised] = -1.0
        out[2, exercised] = 0.0
    return TreeResult(*out.reshape(3, *batch_shape), exercised.reshape(batch_shape))


def exercise_boundary(
    maturity,
    risk_free_rate,
    sigma,
    nb_steps: int = 1000,
    method: str = "bbsr",
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Early-exercise boundary of American puts over time, per unit of strike.

    The holder of an American put exercises the first time the spot falls to
    or below the boundary S*(t). This returns S*/K at every step of one tree
    per contract, from today to expiry. It does not depend on today's spot.

        boundary = exercise_boundary(T, r, sigma)  # shape (*batch, 1001)
        boundary[..., i]  # S*/K at time i * T / 1000

    "bbsr" gives the same boundary as "bbs" (Richardson extrapolation
    applies to prices, not to the boundary).

    Args:
        maturity, risk_free_rate, sigma: arrays (or scalars) that broadcast
            together, one contract per element.
        nb_steps: tree steps per contract.
        method: "crr", "bbs" or "bbsr", as in `price_put_batch`.
        chunk_size: contracts stepped together per pass (None: all at once).
            Only affects speed and memory, never the results.

    Returns:
        An array of shape (*broadcast_shape, nb_steps + 1). Entry i is S*/K
        at time i * maturity / nb_steps. The last entry, at expiry, is 1
        (exercise exactly when in the money). With a rate <= 0 every entry
        before expiry is NaN: early exercise of a put then never pays.

    Raises:
        ValueError: for an invalid parameter, as `price_put_batch` (with
            moneyness fixed at 1).
    """
    params = _validated_params(1.0, maturity, risk_free_rate, sigma, nb_steps, method, chunk_size)
    batch_shape = params.shape[1:]
    params = params.reshape(4, -1)
    n_options = params.shape[1]
    step = chunk_size or max(n_options, 1)
    out = np.empty((n_options, nb_steps + 1))
    for start in range(0, n_options, step):
        part = params[:, start : start + step]
        _, out[start : start + step] = _induct(
            part,
            nb_steps,
            bbs=method != "crr",
            american=True,
            nb_extra_nodes=_nb_extra_nodes(part, nb_steps),
            record_boundary=True,
        )
    return out.reshape(*batch_shape, nb_steps + 1)


def _nb_extra_nodes(params, nb_steps):
    """Number of nodes to add below every level of trees with nb_steps steps
    started at m = 1, so that even the first levels reach down to the
    boundary.

    A tree started at m = 1 spans only u^(-i) to u^i at level i, too narrow
    near today to reach S*. Neighbouring nodes of one level are a factor u^2
    apart, so with n extra nodes level 0 spans down to u^(-2n). The count is
    the smallest n that puts u^(-2n) at or below the perpetual put's boundary
    2r / (2r + sigma^2), a lower bound on S* at every maturity:
    n = ceil(ln(1 / perpetual) / (2 sigma sqrt(dt))), since ln u = sigma
    sqrt(dt). One count for the whole chunk (the largest), since the
    contracts share one array; 0 when no rate is positive. Why this is
    enough, and its cost: `docs/results/hedging-training.md`, "The exercise
    boundary"."""
    T, r, sigma = params[1:]
    positive = r > 0
    if not positive.any():
        return 0
    r, sigma, dt = r[positive], sigma[positive], T[positive] / nb_steps
    perpetual = 2 * r / (2 * r + sigma**2)
    return int(np.ceil(np.max(np.log(1 / perpetual) / (2 * sigma * np.sqrt(dt)))))


def _validated_params(moneyness, maturity, risk_free_rate, sigma, nb_steps, method, chunk_size):
    """Check the arguments of `price_put_batch` and return the parameters
    as one float array of shape (4, *broadcast_shape), rows moneyness,
    maturity, rate, sigma. Raises ValueError as documented there."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
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
        raise ValueError(
            f"method {method!r} needs nb_steps >= {min_steps * (2 if method == 'bbsr' else 1)}, got {nb_steps}"
        )
    return params


def _induct(params, n, bbs, american, nb_extra_nodes=0, record_boundary=False):
    """Backward induction over n-step trees for a (4, n_options) parameter
    array (rows: moneyness, maturity, rate, sigma).

    `nb_extra_nodes` adds that many nodes below every level. They change none
    of the values of the plain tree's nodes, which keep their arithmetic
    operation for operation. With `record_boundary`, every level also
    records the exercise boundary (see `exercise_boundary`).

    Returns:
        (greeks, boundary): greeks is a (3, n_options) array, rows price,
        delta, gamma; boundary is an (n_options, n + 1) array of S*/K per
        level, or None without `record_boundary`.
    """
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

    boundary = None
    if record_boundary:
        boundary = np.full((params.shape[1], n + 1), np.nan)
        boundary[:, n] = 1.0  # at expiry: exercise exactly when in the money

    # Spots at the last level the tree reaches, ascending: node j has j up
    # moves, so S = m * u ** (2j - level), and the extra nodes have j < 0.
    # BBS stops one level before expiry.
    level = n - 1 if bbs else n
    spot = m * u ** (2 * np.arange(-nb_extra_nodes, level + 1) - level)
    exercise = np.maximum(1 - spot, 0)
    if not bbs:
        values = exercise  # at expiry the put is worth its payoff
    else:
        # The European value can fall below the payoff (a deep in-the-money
        # put is worth less held than exercised), so an American put takes
        # the larger of the two.
        values = black_scholes_price(spot, 1.0, dt, r, sigma, "put")
        if record_boundary:
            boundary[:, level] = _boundary_at_level(spot, values, exercise)
        if american:
            values = np.maximum(values, exercise)

    def back_one_level(spot, values, i):
        values = up * values[:, 1:] + down * values[:, :-1]  # discounted expected value of holding
        spot = spot[:, :-1] * u  # node j at level i is node j at level i+1, one down move earlier
        if record_boundary:
            boundary[:, i] = _boundary_at_level(spot, values, 1 - spot)
        if american:
            values = np.maximum(values, 1 - spot)  # or exercise now, if that pays more
        return spot, values

    for i in range(level - 1, 1, -1):
        spot, values = back_one_level(spot, values, i)
    s2, v2 = spot, values  # level 2
    s1, v1 = back_one_level(s2, v2, 1)  # level 1
    _, v0 = back_one_level(s1, v1, 0)  # level 0: today
    # The Greeks use the plain tree's nodes, which start after the extra
    # ones: today's spot m, its two children at level 1, its three
    # grandchildren at level 2.
    first = nb_extra_nodes
    s1, v1, s2, v2 = s1[:, first:], v1[:, first:], s2[:, first:], v2[:, first:]

    delta = (np.diff(v1) / np.diff(s1))[:, 0]
    slopes = np.diff(v2) / np.diff(s2)  # (n_options, 2): slopes between nodes 0-1 and 1-2
    gamma = np.diff(slopes)[:, 0] / ((s2[:, 2] - s2[:, 0]) / 2)
    return np.stack([v0[:, first], delta, gamma]), boundary


def _boundary_at_level(spot, hold, exercise):
    """S*/K at one level of the trees, from the spots (ascending), the value
    of holding and the value of exercising at each node, all of shape
    (n_options, nodes). The zero of hold - exercise, linearly interpolated
    between the highest node where exercising wins by more than
    _EXERCISE_TOL and the node above it. NaN where no node exercises, or
    where the top node does (the boundary is then above the level's nodes,
    which a tree started at m = 1 never has, since its top node is >= 1)."""
    hold_minus_exercise = hold - exercise
    exercise_wins = hold_minus_exercise < -_EXERCISE_TOL
    nb_nodes = spot.shape[1]
    # Index of the highest node where exercising wins, -1 where none does.
    # For a put the exercise region is the low spots, so this node sits just
    # below the boundary and the node above it just above.
    highest_exercise = np.where(exercise_wins, np.arange(nb_nodes), -1).max(axis=1)
    boundary_found = (highest_exercise >= 0) & (highest_exercise < nb_nodes - 1)
    below = np.clip(highest_exercise, 0, nb_nodes - 2)[:, None]
    above = below + 1
    spot_below = np.take_along_axis(spot, below, 1)[:, 0]
    spot_above = np.take_along_axis(spot, above, 1)[:, 0]
    gap_below = np.take_along_axis(hold_minus_exercise, below, 1)[:, 0]
    gap_above = np.take_along_axis(hold_minus_exercise, above, 1)[:, 0]
    # gap_below < -tol <= gap_above, so the weight is in (0, 1], up to the
    # tolerance. (Rows with no boundary divide garbage, hence the errstate;
    # np.where drops them.)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.minimum(-gap_below / (gap_above - gap_below), 1.0)
        boundary = spot_below + weight * (spot_above - spot_below)
    # Near expiry the node above can sit above the strike, and the
    # interpolation then lands past it. Nobody exercises an out-of-the-money
    # put, so S* <= K.
    return np.where(boundary_found, np.minimum(boundary, 1.0), np.nan)
