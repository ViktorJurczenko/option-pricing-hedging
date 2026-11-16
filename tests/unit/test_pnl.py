"""Tests for optionpricing.hedging.pnl: the terminal P&L of a hedged short put."""

import math
from dataclasses import asdict

import numpy as np
import pytest
import torch

from optionpricing.hedging.pnl import HedgeOutcome, HedgeSettings, hedge_pnl
from optionpricing.hedging.settlement import cash_settlement, physical_settlement
from optionpricing.hedging.simulator import DAYS_PER_YEAR, simulate_scenarios
from optionpricing.pricing.european.black_scholes import black_scholes_delta, black_scholes_price

STRIKE = 100.0
# 1% proportional, a 5-cent fee, kappa of 0.01 shares: the setting of docs/results/hedging-training.md.
COSTS = {"proportional": 0.01, "fixed": 0.05, "kappa": 0.01}


def no_hedge(inputs, position):
    return torch.zeros_like(position)


def linear_policy(inputs, position):
    """A policy that uses both its inputs and the position it holds, so the order of calls matters."""
    return -0.5 + 0.1 * inputs[:, 0] - 0.3 * position


def random_paths(n=6, n_days=9, seed=0):
    """Positive prices around the strike, with NaN after each path's tau, as the scenario bank stores them."""
    rng = np.random.default_rng(seed)
    spot = STRIKE * np.exp(np.cumsum(rng.normal(0, 0.05, size=(n, n_days)), axis=1))
    tau = rng.integers(1, n_days, size=n)
    tau[0] = n_days - 1  # one path lives to the last day
    spot[np.arange(n_days) > tau[:, None]] = np.nan
    inputs = rng.normal(size=(n, 2, n_days))
    inputs[np.broadcast_to(np.arange(n_days) > tau[:, None, None], inputs.shape)] = np.nan
    premium = rng.uniform(2, 6, size=n)
    rate = rng.uniform(0.0, 0.1, size=n)
    return {
        "inputs": torch.tensor(inputs),
        "spot": torch.tensor(spot),
        "tau": torch.tensor(tau),
        "premium": torch.tensor(premium),
        "strike": STRIKE,
        "rate": torch.tensor(rate),
    }


def step_by_step_pnl(policy, data, rebalance_every, settlement, costs):
    """Independent reference: a cash account that earns r day by day, booked in dollars and discounted at the end.

    Path by path in plain floats: the premium goes into cash, each trade is paid from cash with its cost, cash
    grows by e^{r / 365} every day, and at tau the shares are worth `position * S_tau` and settlement is paid.
    """
    settle = {"cash": cash_settlement, "physical": physical_settlement}[settlement]
    out = []
    for i in range(len(data["tau"])):
        tau, r = int(data["tau"][i]), float(data["rate"][i])
        spot = data["spot"][i].tolist()
        cash, position = float(data["premium"][i]), 0.0
        for day in range(tau):
            if day % rebalance_every == 0:
                new = float(policy(data["inputs"][i : i + 1, :, day], torch.tensor([position], dtype=torch.float64)))
                cost = costs["proportional"] * abs(new - position) * spot[day]
                cost += costs["fixed"] * (new != position)
                cash -= (new - position) * spot[day] + cost
                position = new
            cash *= math.exp(r / DAYS_PER_YEAR)
        wealth = cash + position * spot[tau] - float(settle(spot[tau], position, STRIKE, **costs))
        out.append(math.exp(-r * tau / DAYS_PER_YEAR) * wealth)
    return torch.tensor(out, dtype=torch.float64)


# closed forms and the independent reference


def test_no_hedge_gives_premium_minus_discounted_payoff():
    data = random_paths(n=50, seed=1)
    out = hedge_pnl(no_hedge, **data, **COSTS)
    tau = data["tau"]
    spot_tau = data["spot"][torch.arange(50), tau]
    expected = data["premium"] - torch.exp(-data["rate"] * tau / DAYS_PER_YEAR) * (STRIKE - spot_tau).clamp(min=0)
    torch.testing.assert_close(out.pnl, expected)
    assert torch.all(out.costs == 0)  # nothing traded, nothing to close


@pytest.mark.parametrize("settlement", ["cash", "physical"])
@pytest.mark.parametrize("rebalance_every", [1, 2, 3])
def test_matches_a_cash_account_booked_step_by_step(settlement, rebalance_every):
    # rebalance_every 2 and 3 leave some tau between hedge dates, so the last interval is cut short at tau
    data = random_paths(seed=2)
    out = hedge_pnl(linear_policy, **data, rebalance_every=rebalance_every, settlement=settlement, **COSTS)
    expected = step_by_step_pnl(linear_policy, data, rebalance_every, settlement, COSTS)
    torch.testing.assert_close(out.pnl, expected, rtol=0, atol=1e-10)


def test_position_chosen_at_a_date_earns_from_that_date():
    # B-037: the legacy env let the position chosen at t earn only from t + 1. Here one share bought on day 0 and
    # sold on day 1 must earn exactly S~_1 - S~_0.
    spot = torch.tensor([[100.0, 103.0, 101.0, 104.0]], dtype=torch.float64)
    inputs = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    r = 0.05
    out = hedge_pnl(lambda x, p: x[:, 0], inputs, spot, torch.tensor([3]), 2.0, STRIKE, r)
    discount = torch.exp(-r * torch.arange(4, dtype=torch.float64) / DAYS_PER_YEAR)
    torch.testing.assert_close(out.pnl, 2.0 + discount[1:2] * 103.0 - 100.0)  # out of the money at tau: no payoff
    torch.testing.assert_close(out.positions, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))


def test_daily_black_scholes_delta_hedge_of_a_european_put_shrinks_like_sqrt_dt():
    # Financed Black-Scholes delta, no costs, sigma 20%, r 5%, T one year, at the money. B-038 measured std P&L / K
    # = 0.0035 for the call; the put's error is the same path by path (parity: the difference is a static hedge).
    sigma, r, days = 0.2, 0.05, DAYS_PER_YEAR
    bank = simulate_scenarios(1, 4000, 7, (days, days), (sigma, sigma), (r, r), (1.0, 1.0), STRIKE)
    spot = bank.spot
    time_left = (days - np.arange(days + 1)) / DAYS_PER_YEAR
    delta = np.zeros_like(spot)  # the last day is never a hedge date, its value is unused
    delta[:, :-1] = black_scholes_delta(spot[:, :-1], STRIKE, time_left[:-1], r, sigma, option_type="put")
    premium = black_scholes_price(STRIKE, STRIKE, 1.0, r, sigma, option_type="put")
    args = (torch.tensor(delta[:, None, :]), torch.tensor(spot), torch.full((4000,), days), premium, STRIKE, r)
    daily = hedge_pnl(lambda x, p: x[:, 0], *args).pnl / STRIKE
    weekly4 = hedge_pnl(lambda x, p: x[:, 0], *args, rebalance_every=4).pnl / STRIKE

    # 4,000 paths: the std is estimated to about 1.5% (fat-tailed P&L), the mean to std / sqrt(4000) ~ 6e-5.
    assert daily.std().item() == pytest.approx(0.0035, rel=0.08)
    assert abs(daily.mean().item()) < 4 * daily.std().item() / math.sqrt(4000)
    # The hedging error scales like sqrt(dt): rebalancing every 4 days doubles it. The two estimates share their
    # paths, so their ratio is tighter than either std; 7% leaves room for the first-order sqrt(dt) law.
    assert (weekly4.std() / daily.std()).item() == pytest.approx(2.0, rel=0.07)


# costs, positions and paths that have ended


def test_costs_are_what_the_costless_pnl_loses():
    data = random_paths(seed=3)
    free = hedge_pnl(linear_policy, **data, rebalance_every=2)
    costly = hedge_pnl(linear_policy, **data, rebalance_every=2, **COSTS)
    torch.testing.assert_close(free.pnl - costly.pnl, costly.costs)
    assert torch.all(free.costs == 0) and torch.any(costly.costs > 0)


def test_numpy_integer_rebalance_every_matches_a_python_int():
    # a sweep over np.arange passes numpy integers
    data = random_paths(seed=11)
    torch.testing.assert_close(
        hedge_pnl(linear_policy, **data, rebalance_every=np.int64(2)).pnl,
        hedge_pnl(linear_policy, **data, rebalance_every=2).pnl,
    )


def test_per_scenario_strike_and_rate_match_scenario_by_scenario_runs():
    # one strike and one rate per scenario must be applied to that scenario, not mixed across the batch
    data = random_paths(seed=12)
    data["strike"] = torch.linspace(95.0, 105.0, 6, dtype=torch.float64)
    batch = hedge_pnl(linear_policy, **data, **COSTS).pnl
    for i in range(6):
        single = {key: value[i : i + 1] for key, value in data.items()}
        alone = hedge_pnl(linear_policy, **single, **COSTS).pnl
        torch.testing.assert_close(batch[i : i + 1], alone)


def test_cash_and_physical_agree_without_costs():
    data = random_paths(seed=4)
    cash = hedge_pnl(linear_policy, **data, settlement="cash")
    physical = hedge_pnl(linear_policy, **data, settlement="physical")
    torch.testing.assert_close(cash.pnl, physical.pnl)


def test_ended_paths_keep_their_position_and_trade_nothing():
    data = random_paths(seed=5)
    calls = []

    def growing(inputs, position):
        calls.append(len(calls))
        return position + 1.0  # trades on every call, so a frozen path shows

    out = hedge_pnl(growing, **data, **COSTS)
    n_dates = int(data["tau"].max())
    assert len(calls) == n_dates == out.positions.shape[1]
    for i, tau in enumerate(data["tau"].tolist()):
        expected = torch.minimum(torch.arange(1, n_dates + 1), torch.tensor(tau)).to(torch.float64)
        torch.testing.assert_close(out.positions[i], expected)


def test_exercise_on_day_zero_settles_with_no_hedge():
    data = random_paths(n=3, seed=6)
    data["tau"] = torch.zeros(3, dtype=torch.long)

    def never_called(inputs, position):
        raise AssertionError("no hedge date before tau = 0")

    out = hedge_pnl(never_called, **data, **COSTS)
    expected = data["premium"] - (STRIKE - data["spot"][:, 0]).clamp(min=0)
    torch.testing.assert_close(out.pnl, expected)
    assert out.positions.shape == (3, 0)


# gradients


def test_gradient_matches_a_finite_difference():
    data = random_paths(seed=7)
    costs = {"proportional": 0.01}  # the fixed fee's gradient is a straight-through surrogate, not the true slope
    weight = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)

    def total(w):
        return hedge_pnl(lambda x, p: -0.4 + x @ w - 0.2 * p, **data, rebalance_every=2, **costs).pnl.sum()

    total(weight).backward()
    h = 1e-6
    for i in range(2):
        step = torch.zeros(2, dtype=torch.float64)
        step[i] = h
        with torch.no_grad():
            numeric = (total(weight + step) - total(weight - step)) / (2 * h)
        # central difference, error O(h^2) on a function smooth away from zero trades
        assert weight.grad[i].item() == pytest.approx(numeric.item(), rel=1e-6)


def test_nan_features_after_tau_leave_the_gradient_finite():
    # Features past tau are NaN. Fed to the policy, they would give NaN gradients even though their output is
    # discarded (0 * NaN = NaN in the backward pass), so they must be replaced before the policy sees them.
    data = random_paths(seed=8)
    weight = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    out = hedge_pnl(lambda x, p: x @ weight, **data, **COSTS)
    out.pnl.sum().backward()
    assert torch.isfinite(out.pnl).all() and torch.isfinite(weight.grad).all()


# invalid input


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"settlement": "swap"}, "settlement must be one of"),
        ({"rebalance_every": 0}, "rebalance_every must be an integer >= 1"),
        ({"rebalance_every": 1.0}, "rebalance_every must be an integer >= 1"),
        ({"rebalance_every": True}, "rebalance_every must be an integer >= 1"),
        ({"tau": torch.full((6,), 9)}, r"tau must be in \[0, 8\]"),
        ({"tau": torch.full((6,), -1)}, r"tau must be in \[0, 8\]"),
        ({"tau": torch.full((6,), 2.0)}, "tau must be an int tensor"),
        ({"tau": torch.full((5,), 2)}, "tau must be an int tensor of shape"),
        ({"inputs": torch.zeros(6, 2, 8)}, "inputs must have shape"),
        ({"spot": torch.zeros(9)}, "spot must have shape"),
        ({"proportional": -0.1}, "proportional must be finite and >= 0"),
        ({"premium": torch.ones(2)}, r"premium must be a scalar or of shape \(6,\), got \(2,\)"),
        ({"rate": torch.ones(6, 1)}, r"rate must be a scalar or of shape \(6,\), got \(6, 1\)"),
        ({"strike": torch.full((7,), 100.0)}, r"strike must be a scalar or of shape \(6,\), got \(7,\)"),
    ],
)
def test_invalid_input_raises(change, message):
    data = random_paths(seed=9)
    data.update(change)
    with pytest.raises(ValueError, match=message):
        hedge_pnl(linear_policy, **data)


def test_policy_returning_the_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"policy must return a tensor of shape \(6,\)"):
        hedge_pnl(lambda x, p: x, **random_paths(seed=10))


# HedgeSettings


def test_hedge_settings_book_a_hedge_as_the_same_keywords_do():
    # The field names are hedge_pnl's keywords, so asdict(settings) must reach every one of them: a renamed field
    # would raise, and a field passed to the wrong keyword would change the P&L.
    args = random_paths()
    settings = HedgeSettings(rebalance_every=2, settlement="physical", **COSTS)
    explicit = hedge_pnl(linear_policy, **args, rebalance_every=2, settlement="physical", **COSTS)
    from_settings = hedge_pnl(linear_policy, **args, **asdict(settings))
    torch.testing.assert_close(from_settings.pnl, explicit.pnl, rtol=0, atol=0)
    torch.testing.assert_close(from_settings.costs, explicit.costs, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"settlement": "swap"}, "settlement must be one of"),
        ({"rebalance_every": 0}, "rebalance_every must be an integer >= 1"),
        ({"rebalance_every": 2.5}, "rebalance_every must be an integer >= 1"),
        ({"rebalance_every": True}, "rebalance_every must be an integer >= 1"),
        ({"proportional": -0.01}, "proportional must be finite and >= 0"),
        ({"fixed": 0.05}, "kappa is needed"),
        ({"fixed": 0.05, "kappa": 0.0}, "kappa must be positive"),
    ],
)
def test_hedge_settings_reject_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        HedgeSettings(**overrides)


# HedgeOutcome.trades


def test_trades_count_the_dates_where_the_position_changed():
    # Scenario 0 trades in, holds, then trades: 2. Scenario 1 never leaves 0 shares: 0. Scenario 2 trades on its
    # first date, then repeats its position after its tau: 1.
    positions = torch.tensor([[-0.5, -0.5, -0.3], [0.0, 0.0, 0.0], [-0.2, -0.2, -0.2]], dtype=torch.float64)
    outcome = HedgeOutcome(pnl=torch.zeros(3), costs=torch.zeros(3), positions=positions)
    assert outcome.trades.tolist() == [2, 0, 1]


def test_trades_of_a_hedge_with_no_date_are_zero():
    outcome = HedgeOutcome(pnl=torch.zeros(2), costs=torch.zeros(2), positions=torch.zeros((2, 0)))
    assert outcome.trades.tolist() == [0, 0]
