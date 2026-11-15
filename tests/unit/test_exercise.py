"""Tests for optionpricing.hedging.exercise: the daily tree boundary and the exercise day of each scenario."""

import numpy as np
import pytest

from optionpricing.hedging.exercise import check_method, daily_boundary, exercise_days
from optionpricing.hedging.simulator import ScenarioBank, simulate_scenarios
from optionpricing.pricing.american.batch_binomial_tree import exercise_boundary, price_put_batch
from optionpricing.pricing.european.black_scholes import black_scholes_price

NAN = np.nan


def hand_bank(spot, days, rate=0.05):
    """A bank of one contract per row of `spot`, with the given days to expiry (NaN past them)."""
    spot = np.asarray(spot, dtype=float)
    n = len(spot)
    return ScenarioBank(
        spot=spot,
        contract=np.arange(n),
        days=np.asarray(days),
        sigma=np.full(n, 0.2),
        risk_free_rate=np.full(n, rate),
        moneyness=spot[:, 0] / 100,
        strike=100.0,
    )


# A flat boundary of $90 on days 0-3 and the strike at expiry (day 4), the shape of a real one.
FLAT = np.array([[90.0, 90.0, 90.0, 90.0, 100.0]])


def tau(spot, method="tree", boundary=FLAT, days=4, **kwargs):
    """Exercise day of one hand-built path against a hand-built boundary."""
    bank = hand_bank([spot], [days])
    return int(exercise_days(bank, method, boundary=boundary, **kwargs)[0])


# exercise_days on hand-built paths


def test_tree_exercises_on_the_first_day_at_or_below_the_boundary():
    assert tau([100, 95, 89, 85, 80]) == 2


def test_touching_the_boundary_exactly_exercises():
    assert tau([100, 95, 90, 95, 95]) == 2


def test_a_path_back_above_the_boundary_keeps_its_first_crossing():
    assert tau([100, 89, 95, 99, 99]) == 1


def test_day_zero_counts():
    assert tau([85, 95, 95, 95, 95]) == 0


def test_a_path_that_never_crosses_ends_at_expiry():
    assert tau([100, 95, 91, 95, 120]) == 4


def test_the_expiry_day_is_not_an_early_exercise_even_below_the_boundary():
    # at expiry the put ends anyway: tau is the expiry, and settlement decides whether it pays
    assert tau([100, 95, 95, 95, 50]) == 4


def test_nan_boundary_never_exercises():
    # r <= 0: the tree has no early-exercise boundary, so the put lives to expiry
    no_boundary = np.array([[NAN, NAN, NAN, NAN, 100.0]])
    assert tau([100, 50, 40, 30, 20], boundary=no_boundary) == 4


def test_days_past_a_shorter_expiry_are_ignored():
    # contract 0 expires on day 2: its NaN prices and boundary after that must not count
    bank = hand_bank([[100, 95, 99, NAN, NAN], [100, 95, 95, 89, 99]], [2, 4])
    boundary = np.array([[90.0, 90.0, 100.0, NAN, NAN], FLAT[0]])
    np.testing.assert_array_equal(exercise_days(bank, "tree", boundary=boundary), [2, 3])


def test_late_holder_uses_the_lowered_boundary():
    path = [100, 89, 85, 80, 80]  # below $90 on day 1, below 0.9 * 90 = $81 only on day 3
    assert tau(path) == 1
    assert tau(path, method="late", shift=0.1) == 3


def test_late_holder_who_never_reaches_the_lowered_boundary_holds_to_expiry():
    assert tau([100, 89, 85, 85, 95], method="late", shift=0.1) == 4


def test_never_returns_each_scenarios_expiry():
    bank = hand_bank([[100, 50, 40, NAN], [100, 50, 40, 30]], [2, 3])
    result = exercise_days(bank, "never")
    np.testing.assert_array_equal(result, [2, 3])
    assert result.dtype.kind == "i"


# on simulated banks


@pytest.fixture(scope="module")
def family_bank():
    return simulate_scenarios(
        20, 50, seed=3, days_range=(30, 120), sigma_range=(0.15, 0.3), rate_range=(0.02, 0.06),
        moneyness_range=(0.85, 1.0),
    )  # fmt: skip


@pytest.fixture(scope="module")
def family_boundary(family_bank):
    return daily_boundary(family_bank, nb_steps=200)


def test_every_method_stays_between_day_zero_and_expiry(family_bank, family_boundary):
    expiry = family_bank.days[family_bank.contract]
    for method, kwargs in [("tree", {}), ("late", {"shift": 0.05}), ("lsm", {}), ("never", {})]:
        boundary = family_boundary if method in ("tree", "late") else None
        result = exercise_days(family_bank, method, boundary=boundary, **kwargs)
        assert result.shape == expiry.shape
        assert np.all((result >= 0) & (result <= expiry)), method


def test_early_exercise_happens_only_in_the_money(family_bank, family_boundary):
    # the tree boundary is below the strike and LSM only exercises in-the-money paths
    expiry = family_bank.days[family_bank.contract]
    for method, boundary in [("tree", family_boundary), ("lsm", None)]:
        result = exercise_days(family_bank, method, boundary=boundary)
        early = result < expiry
        assert early.any(), method  # 0.85 <= S_0/K <= 1 over 1-4 months: some paths do cross
        assert np.all(family_bank.spot[early, result[early]] < family_bank.strike), method


def test_late_holders_never_exercise_before_the_optimal_one(family_bank, family_boundary):
    optimal = exercise_days(family_bank, "tree", boundary=family_boundary)
    late = exercise_days(family_bank, "late", boundary=family_boundary, shift=0.03)
    later = exercise_days(family_bank, "late", boundary=family_boundary, shift=0.1)
    assert np.all(late >= optimal) and np.all(later >= late)
    assert np.any(late > optimal)  # the shift does change some dates


def test_late_holder_with_a_vanishing_shift_is_the_optimal_one(family_bank, family_boundary):
    # (1 - shift) S* tends to S*: only a price within 1e-12 of the boundary could tell the two apart
    optimal = exercise_days(family_bank, "tree", boundary=family_boundary)
    late = exercise_days(family_bank, "late", boundary=family_boundary, shift=1e-12)
    np.testing.assert_array_equal(late, optimal)


def test_tree_computes_the_boundary_itself_when_none_is_given(family_bank):
    given = exercise_days(family_bank, "tree", boundary=daily_boundary(family_bank))
    np.testing.assert_array_equal(exercise_days(family_bank, "tree"), given)


def test_lsm_fits_each_contract_on_its_own_paths_only(family_bank):
    # changing contract 1's paths must leave contract 0's dates alone
    before = exercise_days(family_bank, "lsm")
    spot = family_bank.spot.copy()
    spot[family_bank.contract == 1] *= 0.8
    other = ScenarioBank(**{**family_bank.__dict__, "spot": spot})
    after = exercise_days(other, "lsm")
    first = family_bank.contract == 0
    np.testing.assert_array_equal(after[first], before[first])
    assert np.any(after[~first] != before[~first])


def stopped_value(bank, result):
    """Per-path discounted payoff `e^{-r τ/365} (K - S_τ)^+` of a one-contract bank."""
    spot_at_tau = bank.spot[np.arange(len(result)), result]
    return np.exp(-bank.risk_free_rate[0] * result / 365) * np.maximum(bank.strike - spot_at_tau, 0)


@pytest.fixture(scope="module")
def one_contract():
    return simulate_scenarios(
        1, 50_000, seed=11, days_range=(91, 91), sigma_range=(0.3, 0.3), rate_range=(0.06, 0.06),
        moneyness_range=(0.95, 0.95),
    )  # fmt: skip


def test_tree_dates_recover_the_american_price(one_contract):
    # Stopping at the optimal dates is what the American price is worth, so the mean stopped value must match
    # the tree price. Tolerance: 3 Monte-Carlo standard errors ($0.033 at 50,000 paths) plus $0.01 for daily
    # rather than continuous exercise (a 91-step CRR tree, exercisable once a day, prices within $0.003 of a
    # 910-step one). That is $0.11, while the `never` dates land $0.29 below the American price and `late` with
    # shift 0.1 $0.21 below, so a wrong holder fails.
    bank = one_contract
    values = stopped_value(bank, exercise_days(bank, "tree"))
    american = 100 * price_put_batch(0.95, 91 / 365, 0.06, 0.3).price
    standard_error = values.std() / np.sqrt(len(values))
    assert abs(values.mean() - american) < 3 * standard_error + 0.01


def test_never_dates_recover_the_european_price(one_contract):
    # Holding to expiry is the European put, priced exactly by Black-Scholes. Tolerance: 3 standard errors.
    bank = one_contract
    values = stopped_value(bank, exercise_days(bank, "never"))
    european = black_scholes_price(95.0, 100.0, 91 / 365, 0.06, 0.3, "put")
    assert abs(values.mean() - european) < 3 * values.std() / np.sqrt(len(values))


def test_late_and_never_holders_are_worth_less_than_the_optimal_one(one_contract):
    # the same paths for every holder, so the comparison is free of Monte-Carlo noise between holders
    bank = one_contract
    optimal = stopped_value(bank, exercise_days(bank, "tree")).mean()
    late = stopped_value(bank, exercise_days(bank, "late", shift=0.1)).mean()
    never = stopped_value(bank, exercise_days(bank, "never")).mean()
    assert never < late < optimal


# daily_boundary


def test_daily_boundary_equals_the_tree_levels_that_fall_on_days():
    # 10 and 20 days on a 40-step tree: every day is a level, so no interpolation is involved. The two paths
    # start at different prices, which the boundary must ignore: per unit of strike it depends only on the time
    # left, r and sigma.
    bank = hand_bank([[90.0] * 21, [110.0] * 21], [10, 20])
    bank = ScenarioBank(**{**bank.__dict__, "sigma": np.array([0.2, 0.3])})
    bank.spot[0, 11:] = NAN
    result = daily_boundary(bank, nb_steps=40)
    levels = exercise_boundary(bank.maturity, bank.risk_free_rate, bank.sigma, nb_steps=40)
    np.testing.assert_allclose(result[0, :11], 100 * levels[0, ::4], rtol=1e-14)
    np.testing.assert_allclose(result[1], 100 * levels[1, ::2], rtol=1e-14)


def test_daily_boundary_is_the_strike_at_expiry_and_nan_after(family_bank, family_boundary):
    expiry = family_bank.days
    rows = np.arange(len(expiry))
    np.testing.assert_array_equal(family_boundary[rows, expiry], family_bank.strike)
    after = np.arange(family_boundary.shape[1]) > expiry[:, None]
    assert np.all(np.isnan(family_boundary[after]))
    assert np.all(family_boundary[~after] <= family_bank.strike)
    assert not np.any(np.isnan(family_boundary[~after]))


def test_daily_boundary_interpolates_between_levels():
    # 3 days on a 10-step tree: day 1 sits at level 3.33, between levels 3 and 4
    bank = hand_bank([[100.0] * 4], [3])
    levels = exercise_boundary(bank.maturity, bank.risk_free_rate, bank.sigma, nb_steps=10)[0]
    day_one = daily_boundary(bank, nb_steps=10)[0, 1] / 100
    np.testing.assert_allclose(day_one, levels[3] + (levels[4] - levels[3]) / 3, rtol=1e-12)


def test_daily_boundary_is_nan_before_expiry_without_a_positive_rate():
    result = daily_boundary(hand_bank([[100.0] * 6], [5], rate=0.0), nb_steps=20)
    assert np.all(np.isnan(result[0, :5])) and result[0, 5] == 100


def test_daily_boundary_is_in_dollars_at_the_banks_strike():
    bank = hand_bank([[100.0] * 6], [5])
    at_200 = ScenarioBank(**{**bank.__dict__, "spot": 2 * bank.spot, "strike": 200.0})
    np.testing.assert_allclose(daily_boundary(at_200, nb_steps=20), 2 * daily_boundary(bank, nb_steps=20))


# invalid input


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("optimal", {}, "method must be one of"),
        ("late", {}, "needs a shift in"),
        ("late", {"shift": 0.0}, "needs a shift in"),
        ("late", {"shift": 1.0}, "needs a shift in"),
        ("tree", {"shift": 0.1}, "only used by method 'late'"),
        ("never", {"boundary": FLAT}, "only used by methods 'tree' and 'late'"),
        ("lsm", {"boundary": FLAT}, "only used by methods 'tree' and 'late'"),
        ("tree", {"boundary": FLAT[:, :4]}, "boundary must have shape"),
    ],
)
def test_invalid_arguments_raise(method, kwargs, message):
    with pytest.raises(ValueError, match=message):
        exercise_days(hand_bank([[100, 95, 95, 95, 95]], [4]), method, **kwargs)


@pytest.mark.parametrize(
    ("method", "shift", "message"),
    [
        ("optimal", None, "method must be one of"),
        ("late", None, "needs a shift in"),
        ("late", 0.0, "needs a shift in"),
        ("late", 1.0, "needs a shift in"),
        ("never", 0.1, "only used by method 'late'"),
    ],
)
def test_check_method_refuses_what_exercise_days_refuses(method, shift, message):
    with pytest.raises(ValueError, match=message):
        check_method(method, shift)


@pytest.mark.parametrize(("method", "shift"), [("tree", None), ("lsm", None), ("never", None), ("late", 0.2)])
def test_check_method_accepts_every_valid_holder(method, shift):
    check_method(method, shift)  # returns None and raises nothing: there is no value to check


def test_daily_boundary_rejects_an_invalid_tree():
    with pytest.raises(ValueError, match="nb_steps must be even"):
        daily_boundary(hand_bank([[100.0] * 3], [2]), nb_steps=11)
