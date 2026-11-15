"""Tests for optionpricing.hedging.costs: proportional cost, straight-through fixed fee, and their gradients."""

import math

import pytest
import torch

from optionpricing.hedging.costs import transaction_cost

# The setting of the gradient table in docs/results/hedging-training.md: 1% proportional, a 5-cent fee, kappa
# of 0.01 shares, at S = 100.
EPS, FEE, KAPPA, SPOT = 0.01, 0.05, 0.01, 100.0


def trades(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def fee_slope(x):
    """The fee's backward slope by hand, f * sign(x) * e^(-|x| / kappa) / kappa."""
    return FEE * math.copysign(1.0, x) * math.exp(-abs(x) / KAPPA) / KAPPA if x != 0 else 0.0


# forward value


@pytest.mark.parametrize("x", [0.5, -0.5])
def test_hand_value_of_a_purchase_and_a_sale(x):
    # 1% of 0.5 * 100 = 0.50, plus the 0.05 fee
    cost = transaction_cost(trades([x]), SPOT, EPS, FEE, KAPPA)
    assert cost.item() == pytest.approx(0.55, abs=1e-12)


def test_zero_trade_costs_nothing_even_with_a_fee():
    assert transaction_cost(trades([0.0]), SPOT, EPS, FEE, KAPPA).item() == 0.0


@pytest.mark.parametrize("x", [1e-12, 5e-4, 0.02, -0.02, 3.0])
def test_fee_is_charged_in_full_however_small_the_trade(x):
    # the smooth look-alike would charge only 1 - e^(-1e-10) ~ 1e-10 of the fee at x = 1e-12
    cost = transaction_cost(trades([x]), SPOT, 0.0, FEE, KAPPA)
    assert cost.item() == pytest.approx(FEE, abs=1e-15)


def test_proportional_cost_scales_with_the_dollar_amount_traded():
    cost = transaction_cost(trades([0.3, -0.3, 0.3]), torch.tensor([100.0, 100.0, 50.0], dtype=torch.float64), EPS)
    torch.testing.assert_close(cost, torch.tensor([0.3, 0.3, 0.15], dtype=torch.float64))


def test_kappa_does_not_change_the_cost():
    x = trades([0.0, 1e-6, -0.004, 0.7])
    torch.testing.assert_close(
        transaction_cost(x, SPOT, EPS, FEE, kappa=1e-4), transaction_cost(x, SPOT, EPS, FEE, kappa=10.0)
    )


def test_no_rates_cost_nothing():
    assert torch.equal(transaction_cost(trades([0.4, -1.0]), SPOT), torch.zeros(2, dtype=torch.float64))


def test_a_scalar_spot_broadcasts_over_a_batch_of_paths_and_steps():
    x = torch.tensor([[0.1, 0.0], [-0.2, 0.3]], dtype=torch.float64)
    expected = EPS * x.abs() * SPOT + FEE * (x != 0)
    torch.testing.assert_close(transaction_cost(x, SPOT, EPS, FEE, KAPPA), expected)


def test_accepts_plain_lists():
    assert transaction_cost([0, 2], [100, 100], EPS, FEE, KAPPA).tolist() == pytest.approx([0.0, 2.05])


def test_an_integer_tensor_is_costed_as_float64():
    cost = transaction_cost(torch.tensor([0, 2]), SPOT, EPS, FEE, KAPPA)
    assert cost.dtype == torch.float64
    assert cost.tolist() == pytest.approx([0.0, 2.05])


def test_a_float32_trade_stays_float32():
    # training runs in float32; the cost must not silently upcast the graph
    x = torch.ones(3, dtype=torch.float32)
    assert transaction_cost(x, torch.full((3,), SPOT, dtype=torch.float64), EPS, FEE, KAPPA).dtype == torch.float32


# gradient


@pytest.mark.parametrize("x", [5e-4, 0.02, 0.2, -5e-4, -0.02, -0.2])
def test_gradient_in_trade_is_the_proportional_slope_plus_the_look_alike_slope(x):
    trade = trades([x], requires_grad=True)
    transaction_cost(trade, SPOT, EPS, FEE, KAPPA).backward()
    expected = EPS * SPOT * math.copysign(1.0, x) + fee_slope(x)
    assert trade.grad.item() == pytest.approx(expected, rel=1e-12)


def test_fee_gradient_matches_the_table_in_the_results_doc():
    # docs/results/hedging-training.md, "What the gradient says": 4.76, 0.68 and ~0 dollars per share
    trade = trades([5e-4, 0.02, 0.2], requires_grad=True)
    transaction_cost(trade, SPOT, 0.0, FEE, KAPPA).sum().backward()
    assert trade.grad.tolist() == pytest.approx([4.76, 0.68, 0.0], abs=5e-3)


def test_gradient_at_a_zero_trade_is_zero():
    trade = trades([0.0], requires_grad=True)
    transaction_cost(trade, SPOT, EPS, FEE, KAPPA).backward()
    assert trade.grad.item() == 0.0


def test_gradient_in_spot_is_the_proportional_rate_times_the_size():
    # the fee does not depend on the price
    spot = torch.tensor([100.0, 80.0], dtype=torch.float64, requires_grad=True)
    transaction_cost(trades([0.4, -0.25]), spot, EPS, FEE, KAPPA).sum().backward()
    torch.testing.assert_close(spot.grad, torch.tensor([0.004, 0.0025], dtype=torch.float64))


def test_without_a_fee_kappa_can_be_left_out_in_training():
    trade = trades([0.5], requires_grad=True)
    transaction_cost(trade, SPOT, EPS).backward()
    assert trade.grad.item() == pytest.approx(EPS * SPOT)


def test_without_kappa_the_fee_is_charged_when_no_gradient_is_needed():
    with torch.no_grad():
        cost = transaction_cost(trades([0.5], requires_grad=True), SPOT, EPS, FEE)
    assert cost.item() == pytest.approx(0.55, abs=1e-12)


def test_missing_kappa_is_rejected_when_the_fee_needs_a_gradient():
    with pytest.raises(ValueError, match="kappa is needed"):
        transaction_cost(trades([0.5], requires_grad=True), SPOT, EPS, FEE)


# input validation


@pytest.mark.parametrize("rate", ["proportional", "fixed"])
@pytest.mark.parametrize("value", [-0.01, math.inf, math.nan])
def test_rejects_a_negative_or_non_finite_rate(rate, value):
    with pytest.raises(ValueError, match=f"{rate} must be finite and >= 0"):
        transaction_cost(trades([0.1]), SPOT, **{rate: value}, kappa=KAPPA)


@pytest.mark.parametrize("kappa", [0.0, -0.01, math.inf, math.nan])
def test_rejects_a_non_positive_or_non_finite_kappa(kappa):
    with pytest.raises(ValueError, match="kappa must be positive"):
        transaction_cost(trades([0.1]), SPOT, EPS, FEE, kappa)


@pytest.mark.parametrize("fixed", [0.0, FEE])
def test_kappa_is_validated_even_when_no_gradient_uses_it(fixed):
    # without a fee, or without a gradient, kappa plays no part, but a bad value is still a caller error
    with torch.no_grad(), pytest.raises(ValueError, match="kappa must be positive"):
        transaction_cost(trades([0.1]), SPOT, EPS, fixed, kappa=-1.0)
