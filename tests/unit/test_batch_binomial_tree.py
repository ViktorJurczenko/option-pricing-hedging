import numpy as np
import pytest

from optionpricing.pricing.american.batch_binomial_tree import (
    _EXERCISE_TOL,
    METHODS,
    _induct,
    exercise_boundary,
    price_put_batch,
)
from optionpricing.pricing.american.binomial_tree import BinomialTree
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_gamma, black_scholes_price

# Seven options spread over the sampling domain (m, T, r, sigma), one per column.
M = np.array([0.6, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5])
T = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
R = np.array([0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08])
SIGMA = np.array([0.6, 0.1, 0.2, 0.3, 0.15, 0.4, 0.05])

# A 25 x 3 x 3 x 3 grid over the old generator's domain, with m up to the
# tree generator's cap: m in (0.4, 2.5), T in (0.25, 5), r in (0, 0.08),
# sigma in (0.05, 0.6). m above 1.6 covers deep out-of-the-money puts,
# whose ~0 price must not be mistaken for an exercised payoff.
DOMAIN_GRID = np.meshgrid(
    np.exp(np.linspace(np.log(0.4), np.log(2.5), 25)),
    [0.25, 1.0, 5.0],
    [0.0, 0.04, 0.08],
    [0.05, 0.2, 0.6],
    indexing="ij",
)


def test_crr_matches_binomial_tree_option_by_option():
    rng = np.random.default_rng(1)
    m, t, r, s = (
        rng.uniform(0.6, 1.4, 20),
        rng.uniform(0.25, 3, 20),
        rng.uniform(0, 0.08, 20),
        rng.uniform(0.1, 0.5, 20),
    )
    batch = price_put_batch(m, t, r, s, nb_steps=200, method="crr").price
    single = []
    for k in range(20):
        tree = BinomialTree(
            sigma=s[k],
            start_price=100 * m[k],
            strike=100.0,
            risk_free_rate=r[k],
            maturity=t[k],
            nb_steps=200,
            option_type="put",
        )
        tree.predict_payoff()
        single.append(tree.calculate_option_price() / 100)
    # Same recursion, so only rounding differs: measured 3.4e-15.
    np.testing.assert_allclose(batch, single, rtol=0, atol=1e-13)


def test_hand_computed_three_step_tree():
    # A 3-step tree small enough to work by hand (dt = 0.25, u = 1.161834,
    # p = 0.504342); early exercise happens only at level 2's lowest node.
    res = price_put_batch(1.0, 0.75, 0.05, 0.3, nb_steps=3, method="crr")
    assert res.price == pytest.approx(0.09540, abs=5e-5)
    assert res.delta == pytest.approx(-0.42326, abs=5e-5)
    assert res.gamma == pytest.approx(1.77997, abs=5e-5)


@pytest.mark.parametrize("method, tol", [("crr", 3e-4), ("bbs", 5e-5), ("bbsr", 5e-5)])
def test_literature_benchmark_put(method, tol):
    # S=36, K=40, r=0.06, sigma=0.2, T=1, the classic Longstaff-Schwartz case:
    # 4.4867 is its usual high-precision value. Measured at 1000 steps:
    # crr 4.48684 (its first-order error), bbs 4.48669, bbsr 4.48670.
    price = 40 * price_put_batch(36 / 40, 1.0, 0.06, 0.2, nb_steps=1000, method=method).price
    assert price == pytest.approx(4.4867, abs=tol)


def test_european_bbsr_matches_black_scholes():
    e = price_put_batch(M, T, R, SIGMA, nb_steps=1000, american=False)
    # Measured max errors over these 7 options: price 1.3e-7, delta 7.6e-8, gamma 4.2e-6.
    np.testing.assert_allclose(e.price, black_scholes_price(M, 1.0, T, R, SIGMA, "put"), rtol=0, atol=1e-6)
    np.testing.assert_allclose(e.delta, black_scholes_delta(M, 1.0, T, R, SIGMA, "put"), rtol=0, atol=1e-6)
    np.testing.assert_allclose(e.gamma, black_scholes_gamma(M, 1.0, T, R, SIGMA), rtol=0, atol=2e-5)


def test_european_bbsr_error_falls_fourfold_per_doubling_at_the_money():
    # BBS's error is c/N and Richardson cancels it, leaving a clean 1/N^2 for
    # a European put: measured ratios 3.92, 3.96, 3.98, 3.99. A wrong last
    # step, dt, level index or Richardson weight breaks this. At the money
    # only: off the money the strike moves relative to the grid as N changes,
    # and the (much smaller, ~1e-8) remainder's ratios scatter from 0.3 to 19.
    bs = black_scholes_price(1.0, 1.0, 1.0, 0.05, 0.2, "put")
    errors = [
        abs(price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=n, american=False).price - bs) for n in (100, 200, 400, 800)
    ]
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios


def test_european_bbs_error_halves_per_doubling():
    # Without Richardson the error is first order: measured ratios 1.99, 2.00, 2.00.
    bs = black_scholes_price(1.0, 1.0, 1.0, 0.05, 0.2, "put")
    errors = [
        abs(price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=n, method="bbs", american=False).price - bs)
        for n in (100, 200, 400, 800)
    ]
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.2)), ratios


@pytest.mark.parametrize("method", METHODS)
def test_american_put_dominates_european_tree_and_intrinsic(method):
    american = price_put_batch(*DOMAIN_GRID, nb_steps=200, method=method).price
    european = price_put_batch(*DOMAIN_GRID, nb_steps=200, method=method, american=False).price
    # Exact by construction (the same tree plus a max at every node), except
    # that a time value below _EXERCISE_TOL is rounded to "exercised today":
    # measured -9.6e-13 for r = 0 deep in-the-money puts.
    assert np.all(american >= european - _EXERCISE_TOL)
    assert np.all(american >= np.maximum(1 - DOMAIN_GRID[0], 0) - 1e-14)


def test_american_bbsr_is_at_least_black_scholes_european_up_to_tree_error():
    # Against the *exact* European price the tree's own error shows where the
    # premium is ~0 (OTM, short T): measured min(American - BS) = -5.1e-6 at
    # 400 steps, -2.7e-6 at 1000.
    american = price_put_batch(*DOMAIN_GRID, nb_steps=400).price
    european = black_scholes_price(DOMAIN_GRID[0], 1.0, *DOMAIN_GRID[1:], "put")
    assert np.all(american >= european - 1e-5)
    # ...while deep in the money the early-exercise premium is far from 0.
    assert np.max(american - european) > 0.05


def test_american_greeks_match_finite_differences_of_the_price():
    m = np.array([0.85, 1.0, 1.15, 1.4])  # continuation region, where V is smooth in m
    price = lambda x: price_put_batch(x, 1.0, 0.05, 0.25, nb_steps=2000).price
    res = price_put_batch(m, 1.0, 0.05, 0.25, nb_steps=2000)
    h = 1e-3
    # The tree's delta is measured dt after today (an O(1/N) offset): measured
    # max |delta - central difference| 1.1e-4, against |delta| 0.1-0.7.
    np.testing.assert_allclose(res.delta, (price(m + h) - price(m - h)) / (2 * h), rtol=0, atol=3e-4)
    # A wider step for gamma: the price wiggles at ~1e-7 as the exercise
    # boundary moves between nodes, which h=1e-3 amplifies to ~0.1 in gamma.
    # Measured max error with h=1e-2: 4.2e-3, against gamma 0.3-2.5.
    h = 1e-2
    np.testing.assert_allclose(res.gamma, (price(m + h) - 2 * price(m) + price(m - h)) / h**2, rtol=0, atol=1e-2)


def test_deep_in_the_money_put_is_exercised_immediately():
    res = price_put_batch(0.5, 1.0, 0.05, 0.2, nb_steps=400)
    assert res.price == pytest.approx(0.5, abs=1e-12)
    assert res.delta == pytest.approx(-1.0, abs=1e-12)
    assert res.gamma == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_greeks_stay_within_put_bounds_over_the_domain(method):
    # Before the exercise-today rule, "bbsr" at 400 steps gave delta -1.0102
    # at (m=0.8, T=5, r=0.08, sigma=0.2) and gamma -0.189 at (m=0.449, T=1,
    # r=0.08, sigma=0.6), both exercised today (BUGS.md).
    res = price_put_batch(*DOMAIN_GRID, nb_steps=400, method=method)
    assert np.all((res.delta >= -1) & (res.delta <= 0))
    assert np.all(res.gamma >= 0)


@pytest.mark.parametrize("nb_steps", [400, 1000])
def test_put_exercised_today_gets_exact_price_and_greeks(nb_steps):
    # m = 0.8 is just inside the exercise region: the gamma stencil's upper
    # node reached the continuation region and gave gamma 0.69 at 1000 steps
    # (true 0), and delta -1.0102 at 400 (BUGS.md).
    res = price_put_batch(0.8, 5.0, 0.08, 0.2, nb_steps=nb_steps)
    assert res.price == 1 - 0.8 and res.delta == -1.0 and res.gamma == 0.0
    assert res.exercised


@pytest.mark.parametrize("method", METHODS)
def test_exercised_flag_marks_exactly_the_rows_at_their_payoff(method):
    res = price_put_batch(*DOMAIN_GRID, nb_steps=200, method=method)
    m = DOMAIN_GRID[0]
    assert res.exercised.shape == m.shape and res.exercised.dtype == bool
    assert 0 < res.exercised.mean() < 1  # the grid has both kinds
    at_payoff = (m < 1) & (res.price == 1 - m) & (res.delta == -1) & (res.gamma == 0)
    np.testing.assert_array_equal(res.exercised, at_payoff)


@pytest.mark.parametrize("method", METHODS)
def test_exercise_region_is_one_interval_below_the_boundary(method):
    # On a fine m grid across the boundary (~0.88 here), the flag must be
    # True up to some m* and False after it: no isolated snapped puts in the
    # continuation region, none missed inside the exercise region. Just
    # past m* the put keeps a time value above the snap tolerance.
    m = np.linspace(0.80, 0.95, 1501)
    res = price_put_batch(m, 1.0, 0.08, 0.2, nb_steps=200, method=method)
    first_out = np.argmin(res.exercised)
    assert 0 < first_out and res.exercised[:first_out].all() and not res.exercised[first_out:].any()
    assert np.all(res.price[first_out:] > 1 - m[first_out:] + _EXERCISE_TOL)


def test_deep_out_of_the_money_put_is_not_treated_as_exercised():
    # Its price (~1e-30) equals its payoff (0) to within the tolerance, but
    # only an in-the-money put can be exercised.
    res = price_put_batch(1.6, 0.25, 0.05, 0.05, nb_steps=200)
    assert res.delta > -0.5
    assert not res.exercised


def test_european_put_below_its_payoff_is_left_alone():
    # A deep in-the-money European put is worth less than its payoff
    # (0.4513 < 0.5 by Black-Scholes); there is no exercise region to snap to.
    res = price_put_batch(0.5, 1.0, 0.05, 0.2, nb_steps=400, american=False)
    assert res.price == pytest.approx(black_scholes_price(0.5, 1.0, 1.0, 0.05, 0.2, "put"), abs=1e-6)
    assert res.delta > -1
    assert not res.exercised


def test_bbsr_has_no_odd_even_zig_zag_in_the_number_of_steps():
    # ATM, T=0.1, sigma=0.1, where CRR's zig-zag is large: measured max
    # |P(N+1) - P(N)| for CRR 2.9e-5 over N=200-220, against 5.4e-7 for
    # BBSR's |P(N+2) - P(N)|.
    args = (1.0, 0.25, 0.05, 0.1)
    crr = [price_put_batch(*args, nb_steps=n, method="crr").price for n in range(200, 221)]
    bbsr = [price_put_batch(*args, nb_steps=n).price for n in range(200, 221, 2)]
    assert np.max(np.abs(np.diff(crr))) > 1e-5
    assert np.max(np.abs(np.diff(bbsr))) < 2e-6


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1000])
def test_chunking_never_changes_the_results(chunk_size):
    whole = price_put_batch(M, T, R, SIGMA, nb_steps=100)
    chunked = price_put_batch(M, T, R, SIGMA, nb_steps=100, chunk_size=chunk_size)
    for a, b in zip(whole, chunked):
        assert np.array_equal(a, b)


def test_output_shape_follows_the_broadcast_inputs():
    scalar = price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=50)
    assert all(np.shape(x) == () for x in scalar)

    m = np.array([0.8, 1.0, 1.2])[:, None]
    sigma = np.array([0.1, 0.2, 0.3, 0.4])[None, :]
    grid = price_put_batch(m, 1.0, 0.05, sigma, nb_steps=50)
    assert all(x.shape == (3, 4) for x in grid)
    # Each grid cell is that option priced on its own: per-option sigma, not paired with the nodes axis.
    alone = price_put_batch(1.2, 1.0, 0.05, 0.3, nb_steps=50)
    for a, b in zip(grid, alone):
        assert a[2, 2] == pytest.approx(float(b), rel=1e-12)


def test_empty_batch_returns_empty_arrays():
    res = price_put_batch(np.array([]), 1.0, 0.05, 0.2, nb_steps=50)
    assert all(x.shape == (0,) for x in res)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        price_put_batch(1.0, 1.0, 0.05, 0.2, method="lsm")


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_parameter_raises(position, bad):
    params = [np.array([1.0, 1.0]), 1.0, 0.05, 0.2]
    params[position] = np.array([1.0, bad]) if position == 0 else bad
    with pytest.raises(ValueError, match="finite"):
        price_put_batch(*params, nb_steps=50)


@pytest.mark.parametrize("position, name", [(0, "moneyness"), (1, "maturity"), (3, "sigma")])
@pytest.mark.parametrize("value", [0.0, -0.1])
def test_non_positive_parameter_raises(position, name, value):
    params = [1.0, 1.0, 0.05, 0.2]
    params[position] = np.array([0.5, value])  # one bad element in a batch is enough
    with pytest.raises(ValueError, match=name):
        price_put_batch(*params, nb_steps=50)


def test_negative_rate_is_accepted():
    assert np.isfinite(price_put_batch(1.0, 1.0, -0.01, 0.2, nb_steps=50).price)


@pytest.mark.parametrize("method, too_few, enough", [("crr", 1, 2), ("bbs", 2, 3), ("bbsr", 4, 6)])
def test_minimum_number_of_steps(method, too_few, enough):
    # Gamma needs levels 1 and 2 below the last computed level (BBS stops one
    # level early, BBSR also runs N/2 steps).
    with pytest.raises(ValueError, match="nb_steps"):
        price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=too_few, method=method)
    assert np.isfinite(price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=enough, method=method).price)


def test_odd_steps_raise_for_bbsr_only():
    with pytest.raises(ValueError, match="even"):
        price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=101)
    for method in ("crr", "bbs"):
        assert np.isfinite(price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=101, method=method).price)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_raises(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        price_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=50, chunk_size=chunk_size)


def test_risk_neutral_probability_outside_unit_interval_raises():
    # The N/2 = 3-step tree has dt = 5/3: e^{r dt} = 1.40 beats u = e^{sigma sqrt(dt)} = 1.013.
    with pytest.raises(ValueError, match="probability"):
        price_put_batch(1.0, 5.0, 0.2, 0.01, nb_steps=6)


# --- exercise_boundary ---

# Contracts over the sampling domain, down to r = 0.5% and T = 1 week: low
# rates push the boundary far down, long maturities and high volatility too.
BOUNDARY_GRID = np.meshgrid([1 / 52, 0.25, 1.0, 5.0], [0.005, 0.02, 0.08], [0.05, 0.2, 0.6], indexing="ij")


def _bisect_exercise_threshold(maturity, rate, sigma, nb_steps):
    """The moneyness where `price_put_batch` ("bbs") switches from
    exercising today to holding, by bisection on its `exercised` flag."""
    lo, hi = 0.3, 1.0
    assert price_put_batch(lo, maturity, rate, sigma, nb_steps=nb_steps, method="bbs").exercised
    for _ in range(40):  # 0.7 / 2^40 ~ 6e-13
        mid = (lo + hi) / 2
        if price_put_batch(mid, maturity, rate, sigma, nb_steps=nb_steps, method="bbs").exercised:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@pytest.mark.parametrize("maturity, rate, sigma", [(1.0, 0.05, 0.2), (0.25, 0.02, 0.3), (1.0, 0.06, 0.15)])
def test_boundary_is_within_one_node_of_the_same_trees_exercise_threshold(maturity, rate, sigma):
    # Level i of an n-step tree is a put with maturity * (1 - i/n) left, the
    # same dt as an (n - i)-step tree for that maturity. Each node is valued
    # from its own subtree, so the pricer's exercise test at moneyness S is
    # the boundary's test at a node with spot S. Bisection on the pricer
    # therefore lands between the two nodes the boundary interpolates
    # between, and so does the boundary: they differ by less than one node
    # spacing S (u^2 - 1). Measured: at most half of it, near expiry.
    n = 100
    boundary = exercise_boundary(maturity, rate, sigma, nb_steps=n, method="bbs")
    spacing = np.exp(2 * sigma * np.sqrt(maturity / n)) - 1
    for level in (0, 1, 50, 90, 96):
        threshold = _bisect_exercise_threshold(maturity * (1 - level / n), rate, sigma, n - level)
        assert abs(boundary[level] - threshold) <= threshold * spacing, level


def test_boundary_is_found_at_every_time_between_the_perpetual_bound_and_the_strike():
    # The perpetual put's boundary 2r / (2r + sigma^2) is below S* at every
    # maturity, and S* <= K. A tree that only fans out from its root has no
    # node at the boundary for the first levels (33 of them for T = 1,
    # r = 5%, sigma = 20% at 1000 steps): those levels would be NaN.
    boundary = exercise_boundary(*BOUNDARY_GRID, nb_steps=400)
    _, r, sigma = BOUNDARY_GRID
    perpetual = 2 * r / (2 * r + sigma**2)
    assert not np.isnan(boundary).any()
    assert np.all(boundary >= perpetual[..., None])
    # Includes T = 5, r = 8%, sigma = 5%, whose interpolation one level
    # before expiry gave 1.0002 before the cap at the strike.
    assert np.all(boundary <= 1.0)


def test_boundary_rises_toward_the_strike_as_expiry_nears():
    # The exact boundary rises with time. The tree's wiggles by up to about a
    # sixth of a node spacing between neighbouring levels (odd and even
    # levels' nodes are offset by one up move), so the check uses every 5% of
    # the maturity, much farther apart than one node spacing moves it.
    n = 400
    boundary = exercise_boundary(*BOUNDARY_GRID, nb_steps=n)
    every_5_percent = boundary[..., np.linspace(0, n, 21).astype(int)]
    assert np.all(np.diff(every_5_percent, axis=-1) > 0)
    assert np.all(boundary[..., -1] == 1.0)  # at expiry: exercise exactly when in the money


@pytest.mark.parametrize("rate", [0.0, -0.01])
def test_no_early_exercise_boundary_without_a_positive_rate(rate):
    # Without interest on the strike, holding a put is always worth at least
    # exercising it (the American put is the European one).
    boundary = exercise_boundary(1.0, rate, 0.2, nb_steps=200)
    assert np.isnan(boundary[:-1]).all()
    assert boundary[-1] == 1.0


def test_bbsr_boundary_is_the_finer_bbs_trees():
    # Richardson extrapolation applies to prices, not to the boundary's location.
    bbsr = exercise_boundary(*BOUNDARY_GRID, nb_steps=100)
    bbs = exercise_boundary(*BOUNDARY_GRID, nb_steps=100, method="bbs")
    assert np.array_equal(bbsr, bbs)


def test_crr_and_bbs_boundaries_agree_to_a_node_spacing():
    # Two discretizations of the same boundary, which differ only in the last
    # step (payoff against Black-Scholes over dt). Node spacing at 1000 steps
    # for T = 1, sigma = 20%: e^(2 * 0.2 * sqrt(0.001)) - 1 = 1.3%.
    crr = exercise_boundary(1.0, 0.05, 0.2, method="crr")
    bbs = exercise_boundary(1.0, 0.05, 0.2, method="bbs")
    assert np.max(np.abs(crr - bbs)) < 0.013


@pytest.mark.parametrize("chunk_size", [1, 5, 1000])
def test_chunking_never_changes_the_boundary(chunk_size):
    # Chunks add different numbers of extra nodes (the most any of their
    # contracts needs), which must not change any contract's boundary.
    whole = exercise_boundary(*BOUNDARY_GRID, nb_steps=50)
    chunked = exercise_boundary(*BOUNDARY_GRID, nb_steps=50, chunk_size=chunk_size)
    assert np.array_equal(whole, chunked)


def test_boundary_shape_follows_the_broadcast_inputs():
    assert exercise_boundary(1.0, 0.05, 0.2, nb_steps=50).shape == (51,)
    rates = np.array([0.02, 0.05, 0.08])[:, None]
    sigmas = np.array([0.1, 0.2, 0.3, 0.4])[None, :]
    grid = exercise_boundary(1.0, rates, sigmas, nb_steps=50)
    assert grid.shape == (3, 4, 51)
    # Each cell is that contract on its own: its own rate and sigma, not paired with the time axis.
    np.testing.assert_array_equal(grid[2, 1], exercise_boundary(1.0, 0.08, 0.2, nb_steps=50))
    assert exercise_boundary(np.array([]), 0.05, 0.2, nb_steps=50).shape == (0, 51)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(method="lsm"), "method"),
        (dict(maturity=0.0), "maturity"),
        (dict(sigma=-0.1), "sigma"),
        (dict(risk_free_rate=np.nan), "finite"),
        (dict(nb_steps=101), "even"),
        (dict(nb_steps=2, method="bbs"), "nb_steps"),
        (dict(chunk_size=0), "chunk_size"),
    ],
)
def test_invalid_boundary_arguments_raise(kwargs, match):
    args = dict(maturity=1.0, risk_free_rate=0.05, sigma=0.2, nb_steps=50) | kwargs
    with pytest.raises(ValueError, match=match):
        exercise_boundary(**args)


def test_extra_nodes_leave_price_delta_and_gamma_unchanged():
    # exercise_boundary widens the tree with nodes below every level; the
    # Greeks must still come from the plain tree's nodes. A wrong slice past
    # the extra nodes would read delta and gamma at the wrong spot. Private
    # on purpose: no public function prices on a widened tree. Same machine,
    # same operations per node, so the results must match bit for bit.
    params = np.stack(np.broadcast_arrays(*[g.ravel() for g in DOMAIN_GRID])).astype(float)
    plain, _ = _induct(params, 200, bbs=True, american=True)
    widened, _ = _induct(params, 200, bbs=True, american=True, nb_extra_nodes=57)
    assert np.array_equal(plain, widened)
