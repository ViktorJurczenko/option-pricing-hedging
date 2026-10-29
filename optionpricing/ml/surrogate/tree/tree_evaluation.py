"""
How good a trained PutSurrogate is, as `test-ml` reports it.

- A factorial grid (z x sigma x T x a few rates, inside the training domain)
  against an *independent*, finer tree than the teacher: scoring against the
  teacher's own settings would only measure how well the net copies it.
- One unit per column, never %/$ averaged together: price errors in $ at
  K = 100, relative errors only where the price is not tiny, implied-vol
  errors where vega is not tiny, delta and gamma errors.
- Rows bucketed where the surrogate is used: continuation rows by |z| and
  by T, the rows the reference exercises today on their own (their price is
  the payoff, a different kind of error).
- No-arbitrage violation rates on the w-scaled measure training penalizes
  with (training.scaled_violations), continuation and exercised rows apart:
  around the exercised rows' true zeros any wobble counts, which swamped
  the rate when they were mixed in (.claude/BUGS.md, "Shape-violation rate
  inflated by exercised points").
- How often the payoff floor fires, the theta check (the net's -dV/dT
  against the Black-Scholes PDE from its own price, delta and gamma; 0 on
  exercised rows), the routing outside the domain, and the time per price.
- The held-out test split of the dataset, against the teacher's labels.
"""

import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from optionpricing.ml.surrogate.tree.tree_dataset import TreePutDataset
from optionpricing.ml.surrogate.tree.tree_pricer import DOMAIN_KEYS, ROUTES, PutSurrogate
from optionpricing.ml.surrogate.tree.tree_training import (
    EVAL_CHUNK,
    SHAPE_CONDITIONS,
    ShapePoints,
    price_derivatives,
    scaled_violations,
)
from optionpricing.pricing.american.batch_binomial_tree import TreeResult, binomial_put_batch
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_price

# ---------------------------------------------------------------------------
# Units and buckets
# ---------------------------------------------------------------------------

# The strike the $ figures are quoted at. Everything is computed per unit
# of strike; 100 is the conventional strike of a quoted example, so an
# error of 1e-4 K reads as one cent.
STRIKE = 100.0

# Continuation rows are bucketed by |z|, the distance to the strike in
# standard deviations of the log price (z = ln m / (sigma sqrt T)). The
# edges are whole standard deviations: |z| < 1 is the core, where most
# quoted strikes sit; 1-2 the wings; >= 2 the far wings, out to the
# domain's ends (-3.5 and 3). (name, low, high), low inclusive.
Z_BUCKETS = (("|z| < 1", 0.0, 1.0), ("1 <= |z| < 2", 1.0, 2.0), ("|z| >= 2", 2.0, math.inf))

# And by T: short-dated puts are cheap, so a fixed $ error is a larger
# relative and implied-vol error there. 1 and 3 months are the usual
# listed expiries, and split the 1 week - 5 years domain into the
# short-dated, the near-term and the rest.
T_BUCKETS = (("T < 1 month", 0.0, 1 / 12), ("1-3 months", 1 / 12, 0.25), ("T >= 3 months", 0.25, math.inf))

# ---------------------------------------------------------------------------
# Thresholds: where a metric means something
# ---------------------------------------------------------------------------

# Relative errors only where the reference price is at least this, per
# unit of strike: $0.05 at K = 100, about one quote tick of a listed
# option. Below it, a tiny absolute miss is a huge, meaningless %.
MIN_RELATIVE_PRICE = 5e-4

# Implied-vol errors only where vega dV/dsigma is at least this, per unit
# of strike: one vol point (0.01) then moves the price by >= $0.01 at
# K = 100. Below it, |error| / vega blows up for a price that hardly
# depends on sigma. Never on exercised rows, whose price 1 - m does not
# depend on sigma at all.
MIN_IV_VEGA = 1e-2

# A scaled violation counts above this: a price change of 1e-4 K, about
# the trained net's price error, so smaller violations are fit noise. The
# shape-penalty study used the same threshold.
VIOLATION_TOL = 1e-4

# ---------------------------------------------------------------------------
# Trees: reference, vega, teacher
# ---------------------------------------------------------------------------

# The default reference tree the grid is scored against. BBSR is the most
# accurate method (step-2 convergence study); 4000 steps is 4x the
# teacher's, so its error (~1/N) is ~4x smaller and its settings differ
# from the teacher's, as check_reference requires. Cost: ~48 ms per
# option, ~2 min for the default grid.
REF_METHOD, REF_STEPS = "bbsr", 4000

# Settings of the trees the vega and the timing's "teacher" column use: the
# dataset generator's defaults. Vega is only a scale for the implied-vol
# error, it does not need the reference's accuracy, and at 1000 steps its
# two trees cost ~2.4 ms per option instead of ~48.
TEACHER_METHOD, TEACHER_STEPS = "bbsr", 1000

# Relative sigma bump of the vega's central difference. A trade-off: the
# truncation error grows like the bump squared (negligible at 1%), the
# tree's price jitter (~1.5e-6, its change between N = 1000 and 1002) is
# divided by the difference's width 2 * 1% * sigma. At the lowest sigma
# (0.05) that is a vega error of ~1.5e-3, still ~7x below MIN_IV_VEGA.
VEGA_BUMP = 0.01

# Options per binomial_put_batch call, so the progress bar ticks: 256
# options take ~12 s at the reference's 4000 steps, ~0.6 s at the
# teacher's 1000; the per-call overhead is negligible at either.
TREE_BLOCK = 256

# Options stepped together inside one tree call. 32 was the fastest of
# 16-1000 in the generator's timing: the arrays of one level stay in the
# CPU cache. Results don't depend on it.
TREE_CHUNK = 32

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# 27 z values over the domain's [-3.5, 3] is a step of exactly 0.25
# (6.5 / 0.25 + 1 = 27), so z = 0 and the bucket edges |z| = 1 and 2 fall
# on grid points: the core gets 7 values (-0.75 to 0.75), the wings 8 (4 a
# side), the far wings 12 (7 below the strike, 5 above: the domain is
# asymmetric). Fine enough to follow a price curve's shape, coarse enough
# to keep the reference tree at ~2 min (2342 points with the defaults).
GRID_N_Z = 27

# 6 sigma values log-spaced over 0.05-0.6: 0.05, 0.08, 0.14, 0.22, 0.36,
# 0.6, a factor ~1.6 apart, from a low-vol index to a volatile single
# stock. Log-spaced like the training draws, so every vol regime gets
# the same share of the points.
GRID_N_SIGMA = 6

# 8 maturities log-spaced over 1 week - 5 years, a factor ~2.2 apart:
# 0.02, 0.04, 0.09, 0.21, 0.46, 1.0, 2.3, 5.0 years, which puts 2 in each
# of the short T buckets (< 1 month, 1-3 months) and 4 in the >= 3 months
# one: enough per bucket for a mean, while 1 week and 5 years, the
# domain's ends, are on the grid.
GRID_N_T = 8

# Two rates, a low and a high one well inside the domain's 0-8%, so the
# r table shows whether the error grows with the rate (the early-exercise
# premium does). r = 0 is left out: there is no early exercise then, the
# put is European and the net's premium is ~0, an easy case.
GRID_RATES = (0.02, 0.06)

# z is moved this far inside its ends. The grid builds m = exp(z w) and the
# pricer recomputes z = ln(m) / w, which rounds: z = 3 can come back as
# 3.0000000000000004, which the pricer would send to Black-Scholes instead
# of the net. 1e-9 is far above that rounding (~1e-15) and far below any
# change in price.
Z_INSET = 1e-9

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# The batch timing keeps the best of 3 runs: the fastest is the least
# disturbed by other processes, and 3 is enough at ~15 ms a run.
TIMING_REPEATS = 3
# Single queries: the median of 20, robust to the odd slow call (each
# costs ~1 ms, mostly the fixed overhead of the two autograd passes that
# give delta and gamma).
TIMING_SINGLE = 20
# The teacher tree is timed on this many grid points (~0.6 s): enough to
# average out per-call overhead, too few to slow the report.
TIMING_TREE = 256

# ---------------------------------------------------------------------------
# The reference cache
# ---------------------------------------------------------------------------

# Tag of a reference-grid cache file (grid_reference): the grid, its
# reference tree and vega depend only on the settings stored with them,
# never on the model, and cost ~2 min per evaluation.
REFERENCE_FORMAT = "tree_reference_grid"


class Grid(NamedTuple):
    """Flat (n,) arrays of the evaluation points, per unit of strike."""

    m: np.ndarray
    T: np.ndarray
    r: np.ndarray
    sigma: np.ndarray
    z: np.ndarray


class Reference(NamedTuple):
    """
    What a grid is scored against, the same for every model: the points,
    the reference tree there, its vega, and the reference tree's time per
    option when it was computed (for the timing line).
    """

    grid: Grid
    tree: TreeResult
    vega: np.ndarray
    seconds_per_option: float


class NetDiagnostics(NamedTuple):
    """
    The raw net's quantities (no routing, no payoff floor) at some points,
    (n,) float64 per unit of strike, plus the (n, 4) scaled violations.
    theta = -dV/dT by autograd; theta_pde = r V - r m delta -
    1/2 sigma^2 m^2 gamma from the net's own V, delta, gamma.
    """

    price: np.ndarray
    theta: np.ndarray
    theta_pde: np.ndarray
    violations: np.ndarray


def evaluation_grid(
    domain: dict,
    n_z: int = GRID_N_Z,
    n_sigma: int = GRID_N_SIGMA,
    n_T: int = GRID_N_T,
    rates: Sequence[float] = GRID_RATES,
) -> Grid:
    """
    The factorial grid: z evenly spaced over the domain's z range, sigma and
    T log-spaced over theirs (T from its low end, 1 week in the default
    domain), every rate in `rates`; each combination is one point, so an
    effect of one parameter is not confounded with another's. Points whose
    m = exp(z sigma sqrt(T)) leaves the moneyness cap are dropped (the
    training domain has none: long, volatile options reach the cap before
    the z bounds).

    Raises:
        ValueError: for a count < 1, no rate, or a rate outside the domain.
    """
    # check the request before building anything
    if min(n_z, n_sigma, n_T) < 1 or len(rates) == 0:
        raise ValueError(f"need n_z, n_sigma, n_T >= 1 and a rate, got {n_z}, {n_sigma}, {n_T}, {list(rates)}")
    r_low, r_high = domain["r_range"]
    if any(not r_low <= r <= r_high for r in rates):
        raise ValueError(f"rates {list(rates)} must lie in the domain's r range {domain['r_range']}")

    def spaced(key, n, geometric):
        """n values from one end of domain[key] to the other, evenly or log-spaced."""
        low, high = domain[key]
        if n == 1:  # one value: the middle of the range
            return np.array([math.sqrt(low * high) if geometric else (low + high) / 2])
        return np.geomspace(low, high, n) if geometric else np.linspace(low, high, n)

    # The values of each parameter. z is evenly spaced, then pulled
    # Z_INSET inside its ends so no point leaves the net's route by rounding.
    z_low, z_high = domain["z_range"]
    z = spaced("z_range", n_z, geometric=False).clip(z_low + Z_INSET, z_high - Z_INSET)
    sigma = spaced("sigma_range", n_sigma, geometric=True)
    T = spaced("T_range", n_T, geometric=True)
    r = np.asarray(rates, dtype=float)

    # Every combination once: meshgrid gives one 4-d array per parameter,
    # flattened into one point per (z, sigma, T, r).
    Z, S, T, R = np.meshgrid(z, sigma, T, r, indexing="ij")
    Z, S, T, R = Z.ravel(), S.ravel(), T.ravel(), R.ravel()

    # Back to moneyness, and drop the points beyond the cap.
    m = np.exp(Z * S * np.sqrt(T))
    m_low, m_high = domain["m_range"]
    keep = (m >= m_low) & (m <= m_high)
    return Grid(m[keep], T[keep], R[keep], S[keep], Z[keep])


def tree_prices(m, T, r, sigma, nb_steps: int, method: str, desc: str = "Tree") -> TreeResult:
    """binomial_put_batch over (n,) arrays in blocks of TREE_BLOCK, with a progress bar."""
    n = len(m)
    price, delta, gamma = np.empty((3, n))
    exercised = np.zeros(n, dtype=bool)
    with tqdm(total=n, desc=desc, unit="option") as bar:
        for start in range(0, n, TREE_BLOCK):
            sl = slice(start, start + TREE_BLOCK)
            res = binomial_put_batch(
                m[sl], T[sl], r[sl], sigma[sl], nb_steps=nb_steps, method=method, chunk_size=TREE_CHUNK
            )
            price[sl], delta[sl], gamma[sl], exercised[sl] = res
            bar.update(len(res.price))
    return TreeResult(price, delta, gamma, exercised)


def tree_vega(m, T, r, sigma) -> np.ndarray:
    """
    dV/dsigma per unit of strike: a central difference of teacher-grade
    trees at sigma (1 ± VEGA_BUMP), one pass over both bumps.
    """
    h = VEGA_BUMP * sigma

    # both bumps in one tree pass: the points twice, sigma up then sigma down
    m2, T2, r2 = (np.concatenate([x, x]) for x in (m, T, r))
    sigma2 = np.concatenate([sigma + h, sigma - h])
    both = tree_prices(m2, T2, r2, sigma2, nb_steps=TEACHER_STEPS, method=TEACHER_METHOD, desc="Reference vega").price

    up, down = both[: len(m)], both[len(m) :]
    return (up - down) / (2 * h)


def reference_settings(
    domain: dict, ref_steps: int, ref_method: str, n_z: int, n_sigma: int, n_T: int, rates: Sequence[float]
) -> dict:
    """
    Everything a Reference depends on, as plain JSON values: the grid's
    domain and sizes, the reference tree, and the vega's trees and bump.
    """
    settings = {
        "format": REFERENCE_FORMAT,
        "ref_method": ref_method,
        "ref_steps": int(ref_steps),
        "n_z": int(n_z),
        "n_sigma": int(n_sigma),
        "n_T": int(n_T),
        "rates": [float(r) for r in rates],
        "vega": [TEACHER_METHOD, TEACHER_STEPS, VEGA_BUMP],
        **{key: [float(v) for v in domain[key]] for key in DOMAIN_KEYS},
    }
    return json.loads(json.dumps(settings))  # the form it takes once stored, so the two compare equal


def grid_reference(
    domain: dict,
    ref_steps: int = REF_STEPS,
    ref_method: str = REF_METHOD,
    n_z: int = GRID_N_Z,
    n_sigma: int = GRID_N_SIGMA,
    n_T: int = GRID_N_T,
    rates: Sequence[float] = GRID_RATES,
    cache_path: str | os.PathLike | None = None,
) -> Reference:
    """
    evaluation_grid(domain, ...) with its `ref_method` tree of `ref_steps`
    steps and tree_vega. With `cache_path`, an existing cache file is loaded
    instead of recomputing (~2 min for the default grid), and a missing one
    is computed and written, with the settings it was built with.

    Raises:
        ValueError: if the cache file was built with other settings (another
            domain, grid or reference), or is not a reference cache.
    """
    settings = reference_settings(domain, ref_steps, ref_method, n_z, n_sigma, n_T, rates)
    if cache_path is not None and Path(cache_path).is_file():
        with np.load(cache_path) as data:
            stored = json.loads(str(data["settings"])) if "settings" in data.files else {}
            if stored != settings:
                diff = {k: (stored.get(k), settings.get(k)) for k in settings if stored.get(k) != settings[k]}
                raise ValueError(
                    f"{cache_path} is a reference for other settings ((stored, requested): {diff}); "
                    "pass another --reference_cache path or delete the file"
                )
            grid = Grid(*(data[k] for k in Grid._fields))
            tree = TreeResult(data["price"], data["delta"], data["gamma"], data["exercised"])
            return Reference(grid, tree, data["vega"], float(data["seconds_per_option"]))

    grid = evaluation_grid(domain, n_z, n_sigma, n_T, rates)
    # the reference tree, timed for the report's timing line, then its vega
    start = time.perf_counter()
    tree = tree_prices(grid.m, grid.T, grid.r, grid.sigma, ref_steps, ref_method, desc="Reference tree")
    seconds = (time.perf_counter() - start) / max(len(grid.m), 1)
    ref = Reference(grid, tree, tree_vega(grid.m, grid.T, grid.r, grid.sigma), seconds)

    if cache_path is not None:
        # written to a temporary file, then renamed: an interrupted write leaves no file a later run would trust
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            settings=json.dumps(settings),
            **grid._asdict(),
            price=tree.price,
            delta=tree.delta,
            gamma=tree.gamma,
            exercised=tree.exercised,
            vega=ref.vega,
            seconds_per_option=seconds,
        )
        os.replace(tmp, path)
    return ref


def net_diagnostics(surrogate: PutSurrogate, m, T, r, sigma) -> NetDiagnostics:
    """
    The raw net's price, theta, PDE theta and scaled violations at (n,)
    points, in chunks of EVAL_CHUNK (the double backward's memory). The
    routes and the payoff floor are left out on purpose: these check the net
    itself, which is what the floor and the routes patch.
    """
    n = len(m)
    if n == 0:
        return NetDiagnostics(*(np.empty(0) for _ in range(3)), np.empty((0, len(SHAPE_CONDITIONS))))

    out = {k: [] for k in NetDiagnostics._fields}
    for start in tqdm(range(0, n, EVAL_CHUNK), desc="Net diagnostics", unit="chunk", leave=False):
        sl = slice(start, start + EVAL_CHUNK)
        mc, Tc, rc, sc = m[sl], T[sl], r[sl], sigma[sl]

        # the full price's derivatives: Black-Scholes closed forms + the net's autograd
        pts = ShapePoints.from_params(mc, Tc, rc, sc)
        d = price_derivatives(surrogate.model, surrogate.stats, pts, create_graph=False)
        V = black_scholes_price(mc, 1.0, Tc, rc, sc, "put") + d.premium.detach().double().numpy()
        delta, gamma = d.delta.detach().numpy(), d.gamma.detach().numpy()

        out["price"].append(V)
        # theta = dV/dt = -dV/dT (t is calendar time, T the time left)
        out["theta"].append(-d.dT.detach().numpy())
        # the Black-Scholes PDE per unit of strike, solved for theta
        out["theta_pde"].append(rc * V - rc * mc * delta - 0.5 * sc**2 * mc**2 * gamma)
        out["violations"].append(scaled_violations(d, pts).detach().numpy())

    return NetDiagnostics(*(np.concatenate(out[k]) for k in NetDiagnostics._fields))


def theta_gap(diag: NetDiagnostics, exercised: np.ndarray) -> np.ndarray:
    """
    |theta - theta_pde| where the reference holds the put, |theta| where it
    exercises it today (V = 1 - m does not depend on time, so theta = 0,
    and the PDE does not hold there).
    """
    return np.where(exercised, np.abs(diag.theta), np.abs(diag.theta - diag.theta_pde))


def bucket_masks(values: np.ndarray, buckets, exercised: np.ndarray) -> list:
    """
    [(name, mask)]: one per (name, low, high) bucket of `values` over the
    continuation rows (low <= value < high), then the exercised rows, then all rows.
    """
    groups = [(name, ~exercised & (values >= low) & (values < high)) for name, low, high in buckets]
    return groups + [("exercised", exercised), ("all", np.ones_like(exercised))]


def _mean(x: np.ndarray) -> float:
    """The mean, NaN for no rows (an empty bucket prints as "-")."""
    return float(np.mean(x)) if len(x) else math.nan


def bucket_metrics(
    groups: list,
    price_err: np.ndarray,
    ref_price: np.ndarray,
    delta_err: np.ndarray,
    floored: np.ndarray,
    gamma_err: np.ndarray | None = None,
    vega: np.ndarray | None = None,
    exercised: np.ndarray | None = None,
    gap: np.ndarray | None = None,
) -> list:
    """
    One dict of error metrics per (name, mask) group, every figure per unit
    of strike (the printer converts): n, price_mae, price_max, rel_mae (mean
    |error| / price where the reference price >= MIN_RELATIVE_PRICE, over
    n_rel rows), iv_mae (mean |error| / vega where vega >= MIN_IV_VEGA and
    not exercised, over n_iv rows: the first-order implied-vol error),
    delta_mae, gamma_mae, theta_gap (median), floor_rate (share of rows the
    payoff floor answered). The optional columns are NaN without their input.

    Raises:
        ValueError: if `vega` is given without `exercised`: the implied-vol
            error must leave the exercised rows out, so it needs to know them.
    """
    if vega is not None and exercised is None:
        raise ValueError("vega needs the exercised flags: exercised rows have no implied volatility")

    abs_err = np.abs(price_err)
    # the rows where a relative / implied-vol error means something
    rel_ok = ref_price >= MIN_RELATIVE_PRICE
    iv_ok = None if vega is None else (vega >= MIN_IV_VEGA) & ~exercised

    rows = []
    for name, mask in groups:
        row = {
            "name": name,
            "n": int(mask.sum()),
            "price_mae": _mean(abs_err[mask]),
            "price_max": float(abs_err[mask].max()) if mask.any() else math.nan,
            "n_rel": int((mask & rel_ok).sum()),
            "rel_mae": _mean(abs_err[mask & rel_ok] / ref_price[mask & rel_ok]),
            "delta_mae": _mean(np.abs(delta_err[mask])),
            "gamma_mae": math.nan if gamma_err is None else _mean(np.abs(gamma_err[mask])),
            # the median: one point at the exercise boundary, where the PDE switches off, would dominate a mean
            "theta_gap": float(np.median(gap[mask])) if gap is not None and mask.any() else math.nan,
            "floor_rate": _mean(floored[mask].astype(float)),
        }

        # first-order implied-vol error: how far sigma must move for the true price to reach the surrogate's
        if iv_ok is None:
            row.update(n_iv=0, iv_mae=math.nan)
        else:
            sel = mask & iv_ok
            row.update(n_iv=int(sel.sum()), iv_mae=_mean(abs_err[sel] / vega[sel]))
        rows.append(row)
    return rows


def violation_rates(violations: np.ndarray, exercised: np.ndarray, tol: float = VIOLATION_TOL) -> dict:
    """
    {region: {"n", each SHAPE_CONDITIONS name, "any"}}: the share of the
    region's points whose scaled violation exceeds `tol`, for the
    continuation and exercised regions apart.
    """
    out = {}
    for region, mask in (("continuation", ~exercised), ("exercised", exercised)):
        over = violations[mask] > tol  # (points, conditions) booleans
        rates = {name: _mean(over[:, j].astype(float)) for j, name in enumerate(SHAPE_CONDITIONS)}
        out[region] = {"n": int(mask.sum()), **rates, "any": _mean(over.any(axis=1).astype(float))}
    return out


def violation_rates_by_tol(violations: np.ndarray, exercised: np.ndarray, tols: Sequence[float]) -> dict:
    """{tol: violation_rates(..., tol)} for each tolerance, in the given order."""
    if len(tols) == 0:
        raise ValueError("need at least one violation tolerance")
    return {float(tol): violation_rates(violations, exercised, tol) for tol in tols}


def routing_queries(domain: dict) -> Grid:
    """
    Queries outside the network's domain, a few per way out: z above and
    below its range (T, sigma, r at their geometric/arithmetic middles),
    deep in the money at the lowest rate (below EXERCISE_MIN_RATE in the
    default domain), m beyond each end of its cap (at the longest, most
    volatile corner, where the cap binds before the z bounds), and T,
    sigma, r each beyond their range at m = 1.

    How far out: z by 0.5 and 1.5 (just past the bound, and well past it),
    m by 10%, T and sigma by a factor 2 below / 1.5 above, r by 2 points.
    Enough to be clearly outside, not so far that the query is absurd
    (a half-week put, a 90% vol).
    """
    (T_low, T_high), (s_low, s_high), (r_low, r_high) = (domain[k] for k in ("T_range", "sigma_range", "r_range"))
    (z_low, z_high), (m_low, m_high) = domain["z_range"], domain["m_range"]

    # a typical in-domain option to move off along z
    T0, s0, r0 = math.sqrt(T_low * T_high), math.sqrt(s_low * s_high), (r_low + r_high) / 2
    w0 = s0 * math.sqrt(T0)

    # z beyond its range: far out of the money (Black-Scholes), far in it (exercised today)
    q = [(math.exp(z * w0), T0, r0, s0) for z in (z_high + 0.5, z_high + 1.5, z_low - 0.5, z_low - 1.5)]
    # far in the money at the lowest rate: early exercise may not pay, so the tree answers
    q.append((math.exp((z_low - 0.5) * w0), T0, r_low, s0))
    # m beyond each end of its cap, at the corner where z alone would still be inside
    q += [(m_high * 1.1, T_high, r0, s_high), (m_low / 1.1, T_high, r0, s_high)]
    # T, sigma, r each beyond its range, at the money
    q += [
        (1.0, T_low / 2, r0, s0),
        (1.0, T_high * 1.5, r0, s0),
        (1.0, T0, r0, s_low / 2),
        (1.0, T0, r0, s_high * 1.5),
        (1.0, T0, r_high + 0.02, s0),
    ]

    m, T, r, sigma = (np.array(c) for c in zip(*q))
    return Grid(m, T, r, sigma, np.log(m) / (sigma * np.sqrt(T)))


def time_pricers(surrogate: PutSurrogate, grid: Grid) -> dict:
    """
    Seconds per option: the surrogate on the whole grid at once (best of
    TIMING_REPEATS) and one query at a time (median of TIMING_SINGLE calls;
    the fixed overhead of the delta and gamma autograd passes dominates), against the
    teacher-grade tree on the first TIMING_TREE grid points.
    """
    n = len(grid.m)

    # the surrogate, the whole grid in one call
    batch = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        surrogate.price(grid.m, grid.T, grid.r, grid.sigma)
        batch.append((time.perf_counter() - start) / n)

    # the surrogate, one option per call
    single = []
    for i in range(TIMING_SINGLE):
        j = i % n
        start = time.perf_counter()
        surrogate.price(grid.m[j], grid.T[j], grid.r[j], grid.sigma[j])
        single.append(time.perf_counter() - start)

    # the teacher tree, the alternative the surrogate replaces
    k = min(TIMING_TREE, n)
    start = time.perf_counter()
    binomial_put_batch(
        grid.m[:k],
        grid.T[:k],
        grid.r[:k],
        grid.sigma[:k],
        nb_steps=TEACHER_STEPS,
        method=TEACHER_METHOD,
        chunk_size=TREE_CHUNK,
    )
    teacher = (time.perf_counter() - start) / k

    return {"surrogate_batch": min(batch), "surrogate_single": float(np.median(single)), "teacher_tree": teacher}


def check_reference(teacher: tuple, ref_method: str, ref_steps: int) -> None:
    """
    Raises ValueError if the reference tree has the teacher's (method,
    nb_steps): it would then score how well the net copies its teacher,
    errors included, not how close it is to the true price.
    """
    if (ref_method, ref_steps) == tuple(teacher):
        raise ValueError(
            f"the reference tree ({ref_method}, {ref_steps} steps) has the training teacher's "
            "settings: pick another --ref_method or --ref_steps"
        )


def teacher_settings(dataset_path) -> tuple:
    """The (method, nb_steps) the tree dataset's labels were generated with."""
    with np.load(dataset_path) as data:
        return str(data["method"]), int(data["nb_steps"])


def evaluate_grid(
    surrogate: PutSurrogate,
    ref_steps: int = REF_STEPS,
    ref_method: str = REF_METHOD,
    n_z: int = GRID_N_Z,
    n_sigma: int = GRID_N_SIGMA,
    n_T: int = GRID_N_T,
    rates: Sequence[float] = GRID_RATES,
    tols: Sequence[float] = (VIOLATION_TOL,),
    reference_cache: str | os.PathLike | None = None,
) -> dict:
    """
    Scores the surrogate on evaluation_grid(surrogate.domain, ...) against
    a `ref_method` tree of `ref_steps` steps (see the module docstring),
    loaded from / saved to `reference_cache` if given (grid_reference).

    Returns:
        dict: n (points), by_z, by_T, by_r (bucket_metrics rows),
        violations ({tol: violation_rates} for each of `tols`), routes
        (route name -> count on the grid), routing (one row per route of
        routing_queries: n, mae, max against the reference), timing
        (time_pricers plus "reference_tree", measured when the reference
        was computed).
    """
    reference = grid_reference(surrogate.domain, ref_steps, ref_method, n_z, n_sigma, n_T, rates, reference_cache)
    grid, ref, vega = reference.grid, reference.tree, reference.vega

    # the surrogate as a user gets it (routes and floor included), and the raw net's diagnostics
    res = surrogate.price(grid.m, grid.T, grid.r, grid.sigma)
    diag = net_diagnostics(surrogate, grid.m, grid.T, grid.r, grid.sigma)

    # the errors every table is built from; "exercised" is the reference's flag
    exercised = ref.exercised
    metrics = dict(
        price_err=res.price - ref.price,
        ref_price=ref.price,
        delta_err=res.delta - ref.delta,
        floored=res.route == "payoff_floor",
        gamma_err=res.gamma - ref.gamma,
        vega=vega,
        exercised=exercised,
        gap=theta_gap(diag, exercised),
    )

    # one group per rate (in grid order), then exercised and all, like bucket_masks
    r_groups = [(f"r = {rate:g}", ~exercised & (grid.r == rate)) for rate in dict.fromkeys(grid.r.tolist())]
    r_groups += [("exercised", exercised), ("all", np.ones_like(exercised))]

    routing = _routing_report(surrogate, ref_steps, ref_method)
    timing = {**time_pricers(surrogate, grid), "reference_tree": reference.seconds_per_option}

    return {
        "n": len(grid.m),
        "by_z": bucket_metrics(bucket_masks(np.abs(grid.z), Z_BUCKETS, exercised), **metrics),
        "by_T": bucket_metrics(bucket_masks(grid.T, T_BUCKETS, exercised), **metrics),
        "by_r": bucket_metrics(r_groups, **metrics),
        "violations": violation_rates_by_tol(diag.violations, exercised, tols),
        "routes": {name: int((res.route == name).sum()) for name in ROUTES},
        "routing": routing,
        "timing": timing,
    }


def _routing_report(surrogate: PutSurrogate, ref_steps: int, ref_method: str) -> list:
    """routing_queries priced by the surrogate and the reference, one row per route taken."""
    q = routing_queries(surrogate.domain)
    res = surrogate.price(q.m, q.T, q.r, q.sigma)
    ref = tree_prices(q.m, q.T, q.r, q.sigma, ref_steps, ref_method, desc="Routing reference")
    err = np.abs(res.price - ref.price)

    rows = []
    for name in ROUTES:
        taken = res.route == name
        if taken.any():
            rows.append(
                {"route": name, "n": int(taken.sum()), "mae": _mean(err[taken]), "max": float(err[taken].max())}
            )
    return rows


def evaluate_test_split(
    surrogate: PutSurrogate,
    dataset_path,
    split_ratio: float = 0.8,
    seed: int = 42,
    tols: Sequence[float] = (VIOLATION_TOL,),
) -> dict:
    """
    Scores the surrogate on the dataset's test split (the split the training
    never read: same split_ratio and seed as the training run) against the
    teacher's labels: price and delta errors (no gamma or vega, which the
    dataset class does not load), bucketed by |z| and T with the labels'
    exercised flag, violation rates ({tol: rates} for each of `tols`) and
    the theta gap.

    Raises:
        ValueError: if the dataset's domain is not the surrogate's (a
            checkpoint trained on another dataset), or from TreePutDataset.
    """
    data = TreePutDataset(dataset_path, is_train=False, split_ratio=split_ratio, seed=seed, stats=surrogate.stats)

    # a surrogate trained on another dataset: its "test split" here would not be unseen
    for key in DOMAIN_KEYS:
        if not np.allclose(data.domain.get(key, (math.nan, math.nan)), surrogate.domain[key]):
            raise ValueError(
                f"the dataset's {key} {data.domain.get(key)} is not the surrogate's "
                f"{surrogate.domain[key]}: the checkpoint was trained on another dataset"
            )

    # the labels, rebuilt from the dataset's targets: price = European + premium, delta likewise
    m, T, r, sigma = (data.params[c] for c in ("m", "T", "r", "sigma"))
    args = (m, 1.0, T, r, sigma, "put")
    ref_price = black_scholes_price(*args) + data.premium.double().numpy()
    ref_delta = black_scholes_delta(*args) + data.premium_dm.double().numpy()
    exercised = data.exercised.astype(bool)

    # the surrogate as a user gets it, and the raw net's diagnostics
    res = surrogate.price(m, T, r, sigma)
    diag = net_diagnostics(surrogate, m, T, r, sigma)

    metrics = dict(
        price_err=res.price - ref_price,
        ref_price=ref_price,
        delta_err=res.delta - ref_delta,
        floored=res.route == "payoff_floor",
        exercised=exercised,
        gap=theta_gap(diag, exercised),
    )
    z = np.log(m) / (sigma * np.sqrt(T))
    return {
        "n": len(m),
        "by_z": bucket_metrics(bucket_masks(np.abs(z), Z_BUCKETS, exercised), **metrics),
        "by_T": bucket_metrics(bucket_masks(T, T_BUCKETS, exercised), **metrics),
        "violations": violation_rates_by_tol(diag.violations, exercised, tols),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _fmt(x: float, scale: float = 1.0, spec: str = ".2e") -> str:
    """x * scale formatted with spec, "-" for a missing (None or NaN) value."""
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x * scale, spec)


# (header, key, scale, format) of the bucket tables; the scale turns a
# per-unit-of-strike figure into the header's unit ($ at K = STRIKE, %,
# basis points of vol); None marks a label or a count, printed as it is.
# Gamma per unit of strike is K d2V/dS2, so d2V/dS2 in 1/$ is it / STRIKE.
# The row counts' headers name the threshold that selects their rows.
BUCKET_COLUMNS = (
    ("bucket", "name", None, ""),
    ("n", "n", None, "d"),
    ("price MAE $", "price_mae", STRIKE, ".4f"),
    ("price max $", "price_max", STRIKE, ".4f"),
    ("rel. err %", "rel_mae", 100.0, ".3f"),
    (f"n rel. (price >= ${MIN_RELATIVE_PRICE * STRIKE:g})", "n_rel", None, "d"),
    ("IV err bp", "iv_mae", 1e4, ".2f"),
    (f"n IV (vega >= ${MIN_IV_VEGA * STRIKE:g}/vol)", "n_iv", None, "d"),
    ("delta MAE", "delta_mae", 1.0, ".2e"),
    ("gamma MAE 1/$", "gamma_mae", 1 / STRIKE, ".2e"),
    ("theta gap $/yr (median)", "theta_gap", STRIKE, ".2e"),
    ("floor %", "floor_rate", 100.0, ".1f"),
)
# The test split's: the dataset class loads no gamma, and its vega would cost two trees per row
TEST_COLUMNS = tuple(c for c in BUCKET_COLUMNS if c[1] not in ("iv_mae", "n_iv", "gamma_mae"))


def format_table(rows: list, columns) -> str:
    """Rows of dicts as an aligned text table, one column per (header, key, scale, format)."""
    # every cell as a string: the header row, then one row per dict
    cells = [[h for h, *_ in columns]]
    for row in rows:
        cells.append(
            [format(row[k], spec) if scale is None else _fmt(row[k], scale, spec) for _, k, scale, spec in columns]
        )

    # each column as wide as its widest cell; labels left-aligned, numbers right-aligned
    widths = [max(len(c[j]) for c in cells) for j in range(len(columns))]
    lines = [
        "  ".join(c[j].ljust(widths[j]) if j == 0 else c[j].rjust(widths[j]) for j in range(len(columns)))
        for c in cells
    ]
    lines.insert(1, "-" * len(lines[0]))  # a rule under the header
    return "\n".join(lines)


def format_violations(rates: dict, tol: float) -> str:
    """The violation_rates dict as a table, in %."""
    columns = (
        ("region", "region", None, ""),
        ("n", "n", None, "d"),
        *((name, name, 100.0, ".2f") for name in SHAPE_CONDITIONS),
        ("any", "any", 100.0, ".2f"),
    )
    rows = [{"region": region, **v} for region, v in rates.items()]
    return f"Scaled no-arbitrage violations > {tol:g} (% of points, raw net):\n" + format_table(rows, columns)


def print_report(grid: dict | None = None, test: dict | None = None) -> None:
    """Prints evaluate_grid's and evaluate_test_split's results, one unit per column."""
    note = (
        f"$ at K = {STRIKE:g}; rel. err only where the price >= ${MIN_RELATIVE_PRICE * STRIKE:g}; "
        f"IV err = |error| / vega, where vega >= ${MIN_IV_VEGA * STRIKE:g} per unit of vol, exercised rows excluded; "
        "theta gap = |net theta - PDE theta| (|net theta| on exercised rows); "
        "floor % = rows answered by the payoff floor"
    )

    if test is not None:
        print(f"\n=== Test split ({test['n']} rows, against the teacher's labels) ===")
        for key, title in (("by_z", "|z|"), ("by_T", "T")):
            print(f"\nBy {title} (continuation rows; exercised = the teacher's flag):")
            print(format_table(test[key], TEST_COLUMNS))
        for tol, rates in test["violations"].items():
            print()
            print(format_violations(rates, tol))

    if grid is not None:
        print(f"\n=== Grid ({grid['n']} points, against the reference tree) ===")
        for key, title in (("by_z", "|z|"), ("by_T", "T"), ("by_r", "r")):
            print(f"\nBy {title} (continuation rows; exercised = the reference's flag):")
            print(format_table(grid[key], BUCKET_COLUMNS))
        for tol, rates in grid["violations"].items():
            print()
            print(format_violations(rates, tol))

        print("\nRoutes on the grid: " + ", ".join(f"{k} {v}" for k, v in grid["routes"].items()))
        print("\nOut-of-domain routing against the reference tree:")
        print(
            format_table(
                grid["routing"],
                (
                    ("route", "route", None, ""),
                    ("n", "n", None, "d"),
                    ("MAE $", "mae", STRIKE, ".2e"),
                    ("max $", "max", STRIKE, ".2e"),
                ),
            )
        )

        t = grid["timing"]
        print(
            f"\nTime per price: surrogate {t['surrogate_batch'] * 1e6:.1f} us (batch of {grid['n']}), "
            f"{t['surrogate_single'] * 1e3:.2f} ms (one query); teacher tree "
            f"({TEACHER_METHOD}, {TEACHER_STEPS} steps) {t['teacher_tree'] * 1e3:.2f} ms, "
            f"reference tree {t['reference_tree'] * 1e3:.2f} ms "
            f"(surrogate batch {t['teacher_tree'] / t['surrogate_batch']:.0f}x faster than the teacher)"
        )

    print(f"\n({note})")
