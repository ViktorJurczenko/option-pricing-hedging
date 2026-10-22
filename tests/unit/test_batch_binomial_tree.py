import numpy as np
import pytest

from AmericanOptionPricing.BatchBinomialTree import _EXERCISE_TOL, METHODS, binomial_put_batch
from AmericanOptionPricing.BinomialTree import BinomialTree
from EuropeanOptionPricing.BlackScholes import black_scholes_delta, black_scholes_gamma, black_scholes_price

# Seven options spread over the sampling domain (m, T, r, sigma), one per column.
M = np.array([0.6, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5])
T = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
R = np.array([0.0, 0.01, 0.03, 0.05, 0.06, 0.07, 0.08])
SIGMA = np.array([0.6, 0.1, 0.2, 0.3, 0.15, 0.4, 0.05])

# A 25 x 3 x 3 x 3 grid over the old generator's domain, with m up to the
# tree generator's cap: m in (0.4, 2.5), T in (0.25, 5), r in (0, 0.08),
# sigma in (0.05, 0.6). m above 1.6 covers deep out-of-the-money puts,
# whose ~0 price must not be mistaken for an exercised payoff.
DOMAIN_GRID = np.meshgrid(np.exp(np.linspace(np.log(0.4), np.log(2.5), 25)), [0.25, 1.0, 5.0],
                          [0.0, 0.04, 0.08], [0.05, 0.2, 0.6], indexing="ij")


def test_crr_matches_binomial_tree_option_by_option():
    rng = np.random.default_rng(1)
    m, t, r, s = rng.uniform(0.6, 1.4, 20), rng.uniform(0.25, 3, 20), rng.uniform(0, 0.08, 20), rng.uniform(0.1, 0.5, 20)
    batch = binomial_put_batch(m, t, r, s, nb_steps=200, method="crr").price
    single = []
    for k in range(20):
        tree = BinomialTree(sigma=s[k], start_price=100 * m[k], strike=100.0, risk_free_rate=r[k],
                            maturity=t[k], nb_steps=200, option_type="put")
        tree.predict_payoff()
        single.append(tree.calculate_option_price() / 100)
    # Same recursion, so only rounding differs: measured 3.4e-15.
    np.testing.assert_allclose(batch, single, rtol=0, atol=1e-13)


def test_hand_computed_three_step_tree():
    # REMINDER.md, "The batch binomial tree", sections 4 and 8: the 3-step
    # tree worked by hand (early exercise at level 2's lowest node).
    res = binomial_put_batch(1.0, 0.75, 0.05, 0.3, nb_steps=3, method="crr")
    assert res.price == pytest.approx(0.09540, abs=5e-5)
    assert res.delta == pytest.approx(-0.42326, abs=5e-5)
    assert res.gamma == pytest.approx(1.77997, abs=5e-5)


@pytest.mark.parametrize("method, tol", [("crr", 3e-4), ("bbs", 5e-5), ("bbsr", 5e-5)])
def test_literature_benchmark_put(method, tol):
    # S=36, K=40, r=0.06, sigma=0.2, T=1, the classic Longstaff-Schwartz case:
    # 4.4867 is its usual high-precision value. Measured at 1000 steps:
    # crr 4.48684 (its first-order error), bbs 4.48669, bbsr 4.48670.
    price = 40 * binomial_put_batch(36 / 40, 1.0, 0.06, 0.2, nb_steps=1000, method=method).price
    assert price == pytest.approx(4.4867, abs=tol)


def test_european_bbsr_matches_black_scholes():
    e = binomial_put_batch(M, T, R, SIGMA, nb_steps=1000, american=False)
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
    errors = [abs(binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=n, american=False).price - bs)
              for n in (100, 200, 400, 800)]
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios


def test_european_bbs_error_halves_per_doubling():
    # Without Richardson the error is first order: measured ratios 1.99, 2.00, 2.00.
    bs = black_scholes_price(1.0, 1.0, 1.0, 0.05, 0.2, "put")
    errors = [abs(binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=n, method="bbs", american=False).price - bs)
              for n in (100, 200, 400, 800)]
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all((ratios > 1.8) & (ratios < 2.2)), ratios


@pytest.mark.parametrize("method", METHODS)
def test_american_put_dominates_european_tree_and_intrinsic(method):
    american = binomial_put_batch(*DOMAIN_GRID, nb_steps=200, method=method).price
    european = binomial_put_batch(*DOMAIN_GRID, nb_steps=200, method=method, american=False).price
    # Exact by construction (the same tree plus a max at every node), except
    # that a time value below _EXERCISE_TOL is rounded to "exercised today":
    # measured -9.6e-13 for r = 0 deep in-the-money puts.
    assert np.all(american >= european - _EXERCISE_TOL)
    assert np.all(american >= np.maximum(1 - DOMAIN_GRID[0], 0) - 1e-14)


def test_american_bbsr_is_at_least_black_scholes_european_up_to_tree_error():
    # Against the *exact* European price the tree's own error shows where the
    # premium is ~0 (OTM, short T): measured min(American - BS) = -5.1e-6 at
    # 400 steps, -2.7e-6 at 1000.
    american = binomial_put_batch(*DOMAIN_GRID, nb_steps=400).price
    european = black_scholes_price(DOMAIN_GRID[0], 1.0, *DOMAIN_GRID[1:], "put")
    assert np.all(american >= european - 1e-5)
    # ...while deep in the money the early-exercise premium is far from 0.
    assert np.max(american - european) > 0.05


def test_american_greeks_match_finite_differences_of_the_price():
    m = np.array([0.85, 1.0, 1.15, 1.4])  # continuation region, where V is smooth in m
    price = lambda x: binomial_put_batch(x, 1.0, 0.05, 0.25, nb_steps=2000).price
    res = binomial_put_batch(m, 1.0, 0.05, 0.25, nb_steps=2000)
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
    res = binomial_put_batch(0.5, 1.0, 0.05, 0.2, nb_steps=400)
    assert res.price == pytest.approx(0.5, abs=1e-12)
    assert res.delta == pytest.approx(-1.0, abs=1e-12)
    assert res.gamma == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_greeks_stay_within_put_bounds_over_the_domain(method):
    # Before the exercise-today rule, "bbsr" at 400 steps gave delta -1.0102
    # at (m=0.8, T=5, r=0.08, sigma=0.2) and gamma -0.189 at (m=0.449, T=1,
    # r=0.08, sigma=0.6), both exercised today (BUGS.md).
    res = binomial_put_batch(*DOMAIN_GRID, nb_steps=400, method=method)
    assert np.all((res.delta >= -1) & (res.delta <= 0))
    assert np.all(res.gamma >= 0)


@pytest.mark.parametrize("nb_steps", [400, 1000])
def test_put_exercised_today_gets_exact_price_and_greeks(nb_steps):
    # m = 0.8 is just inside the exercise region: the gamma stencil's upper
    # node reached the continuation region and gave gamma 0.69 at 1000 steps
    # (true 0), and delta -1.0102 at 400 (BUGS.md).
    res = binomial_put_batch(0.8, 5.0, 0.08, 0.2, nb_steps=nb_steps)
    assert res.price == 1 - 0.8 and res.delta == -1.0 and res.gamma == 0.0
    assert res.exercised


@pytest.mark.parametrize("method", METHODS)
def test_exercised_flag_marks_exactly_the_rows_at_their_payoff(method):
    res = binomial_put_batch(*DOMAIN_GRID, nb_steps=200, method=method)
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
    res = binomial_put_batch(m, 1.0, 0.08, 0.2, nb_steps=200, method=method)
    first_out = np.argmin(res.exercised)
    assert 0 < first_out and res.exercised[:first_out].all() and not res.exercised[first_out:].any()
    assert np.all(res.price[first_out:] > 1 - m[first_out:] + _EXERCISE_TOL)


def test_deep_out_of_the_money_put_is_not_treated_as_exercised():
    # Its price (~1e-30) equals its payoff (0) to within the tolerance, but
    # only an in-the-money put can be exercised.
    res = binomial_put_batch(1.6, 0.25, 0.05, 0.05, nb_steps=200)
    assert res.delta > -0.5
    assert not res.exercised


def test_european_put_below_its_payoff_is_left_alone():
    # A deep in-the-money European put is worth less than its payoff
    # (0.4513 < 0.5 by Black-Scholes); there is no exercise region to snap to.
    res = binomial_put_batch(0.5, 1.0, 0.05, 0.2, nb_steps=400, american=False)
    assert res.price == pytest.approx(black_scholes_price(0.5, 1.0, 1.0, 0.05, 0.2, "put"), abs=1e-6)
    assert res.delta > -1
    assert not res.exercised


def test_bbsr_has_no_odd_even_zig_zag_in_the_number_of_steps():
    # ATM, T=0.1, sigma=0.1, where CRR's zig-zag is large: measured max
    # |P(N+1) - P(N)| for CRR 2.9e-5 over N=200-220, against 5.4e-7 for
    # BBSR's |P(N+2) - P(N)|.
    args = (1.0, 0.25, 0.05, 0.1)
    crr = [binomial_put_batch(*args, nb_steps=n, method="crr").price for n in range(200, 221)]
    bbsr = [binomial_put_batch(*args, nb_steps=n).price for n in range(200, 221, 2)]
    assert np.max(np.abs(np.diff(crr))) > 1e-5
    assert np.max(np.abs(np.diff(bbsr))) < 2e-6


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1000])
def test_chunking_never_changes_the_results(chunk_size):
    whole = binomial_put_batch(M, T, R, SIGMA, nb_steps=100)
    chunked = binomial_put_batch(M, T, R, SIGMA, nb_steps=100, chunk_size=chunk_size)
    for a, b in zip(whole, chunked):
        assert np.array_equal(a, b)


def test_output_shape_follows_the_broadcast_inputs():
    scalar = binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=50)
    assert all(np.shape(x) == () for x in scalar)

    m = np.array([0.8, 1.0, 1.2])[:, None]
    sigma = np.array([0.1, 0.2, 0.3, 0.4])[None, :]
    grid = binomial_put_batch(m, 1.0, 0.05, sigma, nb_steps=50)
    assert all(x.shape == (3, 4) for x in grid)
    # Each grid cell is that option priced on its own: per-option sigma, not paired with the nodes axis.
    alone = binomial_put_batch(1.2, 1.0, 0.05, 0.3, nb_steps=50)
    for a, b in zip(grid, alone):
        assert a[2, 2] == pytest.approx(float(b), rel=1e-12)


def test_empty_batch_returns_empty_arrays():
    res = binomial_put_batch(np.array([]), 1.0, 0.05, 0.2, nb_steps=50)
    assert all(x.shape == (0,) for x in res)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        binomial_put_batch(1.0, 1.0, 0.05, 0.2, method="lsm")


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_parameter_raises(position, bad):
    params = [np.array([1.0, 1.0]), 1.0, 0.05, 0.2]
    params[position] = np.array([1.0, bad]) if position == 0 else bad
    with pytest.raises(ValueError, match="finite"):
        binomial_put_batch(*params, nb_steps=50)


@pytest.mark.parametrize("position, name", [(0, "moneyness"), (1, "maturity"), (3, "sigma")])
@pytest.mark.parametrize("value", [0.0, -0.1])
def test_non_positive_parameter_raises(position, name, value):
    params = [1.0, 1.0, 0.05, 0.2]
    params[position] = np.array([0.5, value])  # one bad element in a batch is enough
    with pytest.raises(ValueError, match=name):
        binomial_put_batch(*params, nb_steps=50)


def test_negative_rate_is_accepted():
    assert np.isfinite(binomial_put_batch(1.0, 1.0, -0.01, 0.2, nb_steps=50).price)


@pytest.mark.parametrize("method, too_few, enough", [("crr", 1, 2), ("bbs", 2, 3), ("bbsr", 4, 6)])
def test_minimum_number_of_steps(method, too_few, enough):
    # Gamma needs levels 1 and 2 below the last computed level (BBS stops one
    # level early, BBSR also runs N/2 steps).
    with pytest.raises(ValueError, match="nb_steps"):
        binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=too_few, method=method)
    assert np.isfinite(binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=enough, method=method).price)


def test_odd_steps_raise_for_bbsr_only():
    with pytest.raises(ValueError, match="even"):
        binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=101)
    for method in ("crr", "bbs"):
        assert np.isfinite(binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=101, method=method).price)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_raises(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        binomial_put_batch(1.0, 1.0, 0.05, 0.2, nb_steps=50, chunk_size=chunk_size)


def test_risk_neutral_probability_outside_unit_interval_raises():
    # The N/2 = 3-step tree has dt = 5/3: e^{r dt} = 1.40 beats u = e^{sigma sqrt(dt)} = 1.013.
    with pytest.raises(ValueError, match="probability"):
        binomial_put_batch(1.0, 5.0, 0.2, 0.01, nb_steps=6)
