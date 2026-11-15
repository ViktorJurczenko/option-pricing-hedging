"""Tests for optionpricing.hedging.settlement: cash and physical settlement of the short put."""

import numpy as np
import pytest
import torch

from optionpricing.hedging.settlement import cash_settlement, physical_settlement

# 1% proportional, a 5-cent fee, kappa of 0.01 shares, at K = 100: the setting of docs/results/hedging-training.md.
COSTS = {"proportional": 0.01, "fixed": 0.05, "kappa": 0.01}
STRIKE = 100.0
SETTLEMENTS = {"cash": cash_settlement, "physical": physical_settlement}


def tensor(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


# hand values


def test_cash_hand_value_in_the_money():
    # pays 100 - 90 = 10, buys back 0.8 shares at 90: 1% of 72 = 0.72, plus the 0.05 fee
    assert cash_settlement(90.0, -0.8, STRIKE, **COSTS).item() == pytest.approx(10.77, abs=1e-12)


def test_physical_hand_value_in_the_money():
    # pays 100 for a share worth 90 = 10, then sells the 0.2 shares left: 1% of 18 = 0.18, plus the 0.05 fee
    assert physical_settlement(90.0, -0.8, STRIKE, **COSTS).item() == pytest.approx(10.23, abs=1e-12)


# the two conventions against each other


def test_both_pay_the_payoff_when_costs_are_zero():
    rng = np.random.default_rng(0)
    spot = tensor(rng.uniform(60.0, 140.0, size=200))
    position = tensor(rng.uniform(-1.2, 0.2, size=200))
    payoff = torch.clamp(STRIKE - spot, min=0.0)
    torch.testing.assert_close(cash_settlement(spot, position, STRIKE), payoff)
    torch.testing.assert_close(physical_settlement(spot, position, STRIKE), payoff)


def test_physical_with_a_full_short_share_closes_nothing():
    # delta -1: the delivered share exactly covers the short, so no trade and no cost
    assert physical_settlement(80.0, -1.0, STRIKE, **COSTS).item() == pytest.approx(20.0, abs=1e-12)


def test_physical_pays_cost_on_position_plus_one_only():
    position = tensor([-1.3, -1.0, -0.6, 0.0])
    spot = 85.0
    expected = (STRIKE - spot) + 0.01 * (position + 1).abs() * spot + 0.05 * (position + 1 != 0)
    torch.testing.assert_close(physical_settlement(spot, position, STRIKE, **COSTS), expected)


def test_cash_pays_cost_on_the_whole_position():
    position = tensor([-1.0, -0.6, 0.0])
    spot = 85.0
    expected = (STRIKE - spot) + 0.01 * position.abs() * spot + 0.05 * (position != 0)
    torch.testing.assert_close(cash_settlement(spot, position, STRIKE, **COSTS), expected)


@pytest.mark.parametrize("spot", [100.0, 115.0])
def test_out_of_the_money_or_at_the_money_expiry_just_closes_the_position_under_both(spot):
    # the put lapses (at S = K exercising pays 0 anyway), so no share is delivered
    position = tensor([-0.1, 0.0])
    expected = 0.01 * position.abs() * spot + 0.05 * (position != 0)
    for settle in SETTLEMENTS.values():
        torch.testing.assert_close(settle(spot, position, STRIKE, **COSTS), expected)


def test_per_path_strikes_and_spots_settle_each_path_on_its_own():
    spot, strike = tensor([0.9, 1.2, 95.0]), tensor([1.0, 1.0, 100.0])
    torch.testing.assert_close(cash_settlement(spot, 0.0, strike), tensor([0.1, 0.0, 5.0]))


# gradient


@pytest.mark.parametrize(
    ("settle", "position", "expected"),
    # proportional cost only, at S = 90: d/d(position) of 1% * |trade| * 90 = 0.9 * sign(trade)
    [
        (cash_settlement, -0.8, -0.9),  # closes -0.8: buys 0.8, so a larger short costs more
        (physical_settlement, -0.8, 0.9),  # closes 0.2: sells 0.2, so a larger position costs more
        (physical_settlement, -1.2, -0.9),  # closes -0.2: buys 0.2
    ],
    ids=["cash", "physical-long-remainder", "physical-short-remainder"],
)
def test_gradient_in_position_is_the_closing_trade_slope(settle, position, expected):
    position = tensor(position, requires_grad=True)
    settle(90.0, position, STRIKE, proportional=0.01).backward()
    assert position.grad.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
def test_a_float32_position_stays_float32(settle):
    # training runs in float32; settlement must not silently upcast the graph
    assert settle(90.0, torch.zeros(3, dtype=torch.float32), STRIKE, **COSTS).dtype == torch.float32


# input validation


@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_rejects_a_non_positive_spot(settle, bad):
    with pytest.raises(ValueError, match="spot must be positive"):
        settle(tensor([90.0, bad]), -0.5, STRIKE)


@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_rejects_a_non_positive_strike(settle, bad):
    with pytest.raises(ValueError, match="strike must be positive"):
        settle(90.0, -0.5, bad)


@pytest.mark.parametrize("settle", SETTLEMENTS.values(), ids=SETTLEMENTS.keys())
def test_passes_on_the_cost_validation(settle):
    # the closing trade needs kappa for the fee's gradient, as any other trade does
    with pytest.raises(ValueError, match="kappa is needed"):
        settle(90.0, tensor(-0.5, requires_grad=True), STRIKE, proportional=0.01, fixed=0.05)
