"""Tests for optionpricing.hedging.simulator: the seeded scenario bank of contracts and their price paths."""

import numpy as np
import pytest

from optionpricing.hedging.simulator import simulate_scenarios

# The narrow family of docs/results/hedging-training.md ("The defaults"): 3 months (92 days) to 1 year.
FAMILY = {
    "days_range": (92, 365),
    "sigma_range": (0.15, 0.30),
    "rate_range": (0.02, 0.06),
    "moneyness_range": (0.9, 1.1),
}


def single_contract(days=365, sigma=0.3, rate=0.05, moneyness=1.0):
    """Ranges that pin every contract to one value."""
    return {
        "days_range": (days, days),
        "sigma_range": (sigma, sigma),
        "rate_range": (rate, rate),
        "moneyness_range": (moneyness, moneyness),
    }


@pytest.fixture(scope="module")
def bank():
    return simulate_scenarios(500, 4, seed=0, **FAMILY)


# contracts


def test_contracts_stay_in_their_ranges(bank):
    assert bank.days.min() >= 92 and bank.days.max() <= 365
    assert np.all((bank.sigma >= 0.15) & (bank.sigma <= 0.30))
    assert np.all((bank.risk_free_rate >= 0.02) & (bank.risk_free_rate <= 0.06))
    assert np.all((bank.moneyness >= 0.9) & (bank.moneyness <= 1.1))


def test_contracts_cover_their_ranges(bank):
    # 500 uniform draws: the extremes land within ~0.2% of each range's ends, so 5% is far from flaky
    for values, (low, high) in [(bank.sigma, (0.15, 0.30)), (bank.moneyness, (0.9, 1.1)), (bank.days, (92, 365))]:
        span = high - low
        assert values.min() < low + 0.05 * span and values.max() > high - 0.05 * span


def test_both_ends_of_the_days_range_are_drawn():
    assert set(simulate_scenarios(200, 1, seed=1, **{**FAMILY, "days_range": (5, 6)}).days.tolist()) == {5, 6}


def test_maturity_is_days_over_365(bank):
    np.testing.assert_array_equal(bank.maturity, bank.days / 365)


# scenarios and their contracts


def test_each_contract_gets_its_paths_grouped_together(bank):
    np.testing.assert_array_equal(bank.contract, np.repeat(np.arange(500), 4))
    assert bank.spot.shape == (2000, bank.days.max() + 1)


def test_spot_starts_at_its_contracts_moneyness_times_strike(bank):
    np.testing.assert_allclose(bank.spot[:, 0], bank.moneyness[bank.contract] * 100.0)


def test_a_non_default_strike_scales_the_prices():
    b = simulate_scenarios(5, 2, seed=0, strike=50.0, **FAMILY)
    assert b.strike == 50.0
    np.testing.assert_allclose(b.spot[:, 0], b.moneyness[b.contract] * 50.0)


def test_paths_of_one_contract_differ():
    spot = simulate_scenarios(1, 3, seed=2, **single_contract()).spot
    assert not np.array_equal(spot[0], spot[1]) and not np.array_equal(spot[1], spot[2])


def test_each_scenario_is_priced_up_to_its_expiry_and_nan_after(bank):
    alive = np.arange(bank.spot.shape[1]) <= bank.days[bank.contract][:, None]
    assert np.all(np.isfinite(bank.spot[alive]) & (bank.spot[alive] > 0))
    assert np.all(np.isnan(bank.spot[~alive]))


def test_times_are_the_daily_grid(bank):
    np.testing.assert_allclose(bank.times, np.arange(bank.spot.shape[1]) / 365)


# path statistics


def test_discounted_price_has_no_drift():
    # risk-neutral paths: E[e^{-rT} S_T] = S_0. 20000 paths of one contract: the relative std of the mean is
    # sigma sqrt(T) / sqrt(n) = 0.3 / 141 = 0.2%, so 1% is 5 standard errors.
    b = simulate_scenarios(1, 20000, seed=3, **single_contract())
    assert (np.exp(-0.05) * b.spot[:, -1] / b.spot[:, 0]).mean() == pytest.approx(1.0, rel=0.01)


def test_daily_variance_follows_each_contracts_sigma_up_to_its_own_expiry():
    # two contracts with different sigma and maturity in one bank, so the short one is NaN-padded to the long
    # one's horizon. Daily log-return variance is sigma^2 / 365. 200 paths give 200 * 100 (short) and
    # 200 * 300 (long) returns. A variance estimate from n returns has relative std sqrt(2 / n): 1.0% and 0.6%,
    # so 6% is 6+ standard errors.
    short = simulate_scenarios(1, 200, seed=4, **single_contract(days=100, sigma=0.15))
    long_ = simulate_scenarios(1, 200, seed=5, **single_contract(days=300, sigma=0.30))
    mixed = simulate_scenarios(2, 200, seed=6, **{**FAMILY, "days_range": (100, 300)})
    assert mixed.days[0] != mixed.days[1]  # otherwise the mixed bank would not mix maturities
    for b in (short, long_, mixed):
        returns = np.diff(np.log(b.spot), axis=1)
        for c in range(b.days.size):
            own = returns[b.contract == c, : b.days[c]]  # steps up to this contract's expiry only
            assert np.isfinite(own).all()
            assert own.var() * 365 == pytest.approx(b.sigma[c] ** 2, rel=0.06)


# seeding


def test_same_seed_gives_the_same_bank():
    a, b = simulate_scenarios(20, 3, seed=7, **FAMILY), simulate_scenarios(20, 3, seed=7, **FAMILY)
    np.testing.assert_array_equal(a.spot, b.spot)
    np.testing.assert_array_equal(a.days, b.days)


def test_different_seeds_give_different_banks():
    assert not np.array_equal(
        simulate_scenarios(20, 1, seed=7, **FAMILY).sigma, simulate_scenarios(20, 1, seed=8, **FAMILY).sigma
    )


def test_leaves_the_global_random_state_alone():
    np.random.seed(11)
    expected_next = np.random.random()
    np.random.seed(11)
    simulate_scenarios(10, 2, seed=0, **FAMILY)
    assert np.random.random() == expected_next


# input validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_contracts": 0}, "n_contracts must be >= 1"),
        ({"paths_per_contract": 0}, "paths_per_contract must be >= 1"),
        ({"strike": 0.0}, "strike must be positive"),
        ({"strike": np.inf}, "strike must be positive"),
        ({"days_range": (0, 10)}, "days_range must be whole numbers"),
        ({"days_range": (30, 10)}, "days_range must be whole numbers"),
        ({"days_range": (91.25, 365)}, "days_range must be whole numbers"),
        ({"sigma_range": (0.0, 0.2)}, "sigma_range must be"),
        ({"sigma_range": (0.1, np.inf)}, "sigma_range must be"),
        ({"rate_range": (-0.01, 0.05)}, "rate_range must be"),
        ({"rate_range": (0.06, 0.02)}, "rate_range must be"),
        ({"moneyness_range": (np.nan, 1.1)}, "moneyness_range must be"),
    ],
)
def test_rejects_invalid_arguments(kwargs, message):
    args = {"n_contracts": 10, "paths_per_contract": 2, "seed": 0, **FAMILY, **kwargs}
    with pytest.raises(ValueError, match=message):
        simulate_scenarios(**args)


def test_a_zero_rate_is_allowed():
    assert simulate_scenarios(3, 1, seed=0, **{**FAMILY, "rate_range": (0.0, 0.0)}).risk_free_rate.tolist() == [0.0] * 3


# select_contracts


def test_select_contracts_keeps_the_block_and_renumbers_it(bank):
    block = bank.select_contracts(10, 13)
    rows = np.flatnonzero((bank.contract >= 10) & (bank.contract < 13))
    np.testing.assert_array_equal(block.spot, bank.spot[rows])  # every column kept, NaN padding included
    np.testing.assert_array_equal(block.contract, bank.contract[rows] - 10)
    for name in ("days", "sigma", "risk_free_rate", "moneyness"):
        np.testing.assert_array_equal(getattr(block, name), getattr(bank, name)[10:13])
    assert block.strike == bank.strike


def test_select_contracts_takes_the_last_block_up_to_the_end(bank):
    assert len(bank.select_contracts(499, 500).days) == 1


@pytest.mark.parametrize(("start", "stop"), [(3, 3), (-1, 2), (0, 501), (5, 4)])
def test_select_contracts_rejects_an_empty_or_out_of_range_block(bank, start, stop):
    with pytest.raises(ValueError, match="need 0 <= start < stop <= 500"):
        bank.select_contracts(start, stop)
