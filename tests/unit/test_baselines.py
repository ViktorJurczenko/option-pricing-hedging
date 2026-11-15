"""Tests for optionpricing.hedging.baselines: the hedging policies the Deep Hedging network is measured against."""

import math

import pytest
import torch

from optionpricing.hedging import baselines as b
from optionpricing.hedging.pnl import hedge_pnl

STRIKE = 100.0


def date_inputs(spot=100.0, delta=-0.4, gamma=0.04, time_left=0.25, rate=0.03, n=1):
    """One hedge date of `baseline_inputs` for `n` scenarios, every row given as a scalar or an `(n,)` tensor."""
    values = [torch.as_tensor(v, dtype=torch.float64).expand(n) for v in (spot, delta, gamma, time_left, rate)]
    return torch.stack(values, dim=1)


# baseline_inputs


def test_baseline_inputs_puts_each_quantity_in_its_named_row_and_broadcasts():
    spot = torch.tensor([[100.0, 101.0, 99.0], [95.0, 96.0, 97.0]], dtype=torch.float64)
    delta = -torch.rand(2, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    gamma = torch.full((2, 3), 0.03, dtype=torch.float64)
    time_left = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64) / 365  # shared by both scenarios
    rate = torch.tensor([[0.02], [0.05]], dtype=torch.float64)  # one rate per scenario
    out = b.baseline_inputs(spot, delta, gamma, time_left, rate)

    assert out.shape == (2, len(b.BASELINE_FEATURES), 3) and out.dtype == torch.float64
    torch.testing.assert_close(out[:, b.SPOT], spot)
    torch.testing.assert_close(out[:, b.DELTA], delta)
    torch.testing.assert_close(out[:, b.GAMMA], gamma)
    torch.testing.assert_close(out[:, b.TIME_LEFT], time_left.expand(2, 3))
    torch.testing.assert_close(out[:, b.RATE], rate.expand(2, 3))


def test_baseline_inputs_rejects_a_shape_that_does_not_broadcast():
    with pytest.raises(ValueError, match="rate must broadcast"):
        b.baseline_inputs(torch.ones(2, 3), 0.0, 0.0, 0.0, torch.ones(3, 1))
    with pytest.raises(ValueError, match="spot must have shape"):
        b.baseline_inputs(torch.ones(3), 0.0, 0.0, 0.0, 0.0)


# no hedge and delta


def test_no_hedge_holds_nothing_and_delta_hedge_holds_the_delta_row():
    inputs = date_inputs(delta=[-0.1, -0.5, -1.0], n=3)
    position = torch.tensor([0.3, -0.2, 0.0], dtype=torch.float64)
    assert torch.equal(b.no_hedge(inputs, position), torch.zeros(3, dtype=torch.float64))
    assert torch.equal(b.delta_hedge(inputs, position), torch.tensor([-0.1, -0.5, -1.0], dtype=torch.float64))


# fixed bands


@pytest.mark.parametrize("target", b.BAND_TARGETS)
def test_band_holds_the_exact_position_inside_and_on_the_edge(target):
    # Held exactly, not to rounding: hedge_pnl charges the fixed fee on any nonzero trade. Binary fractions put the
    # edges exactly 0.25 from delta, so a strict `<` on the edge would trade.
    inputs = date_inputs(delta=-0.5, n=4)
    position = torch.tensor([-0.5, -0.375, -0.75, -0.25], dtype=torch.float64)  # on delta, inside, both edges
    out = b.band_hedge(0.25, target)(inputs, position)
    assert torch.equal(out, position)


def test_band_outside_trades_to_the_nearest_edge_or_to_delta():
    inputs = date_inputs(delta=-0.4, n=2)
    position = torch.tensor([0.0, -0.9], dtype=torch.float64)  # above and below the band
    edge = b.band_hedge(0.1, "edge")(inputs, position)
    to_delta = b.band_hedge(0.1, "delta")(inputs, position)
    torch.testing.assert_close(edge, torch.tensor([-0.3, -0.5], dtype=torch.float64))
    torch.testing.assert_close(to_delta, torch.tensor([-0.4, -0.4], dtype=torch.float64))


@pytest.mark.parametrize("target", b.BAND_TARGETS)
def test_band_of_zero_width_is_the_delta_hedge(target):
    inputs = date_inputs(delta=[-0.2, -0.7, -1.0], n=3)
    position = torch.tensor([0.0, -0.69, -1.0], dtype=torch.float64)
    assert torch.equal(b.band_hedge(0.0, target)(inputs, position), b.delta_hedge(inputs, position))


@pytest.mark.parametrize("half_width", [-0.01, math.inf, math.nan])
def test_band_rejects_an_invalid_width(half_width):
    with pytest.raises(ValueError, match="half_width"):
        b.band_hedge(half_width)


def test_band_rejects_an_unknown_target():
    with pytest.raises(ValueError, match="target must be one of"):
        b.band_hedge(0.1, "middle")
    with pytest.raises(ValueError, match="target must be one of"):
        b.whalley_wilmott(1.0, 0.01, "middle")


# Whalley-Wilmott


def test_whalley_wilmott_width_matches_the_formula_by_hand():
    # S = 100, gamma = 0.04, eps = 1%, gamma_ra = 1, r (T - t) = 0.03 * 0.25:
    # H = (1.5 * e^{-0.0075} * 0.01 * 100 * 0.0016 / 1)^{1/3} = (0.0023820...)^{1/3} = 0.13355...
    width = b.whalley_wilmott_half_width(date_inputs(), risk_aversion=1.0, proportional=0.01)
    expected = (1.5 * math.exp(-0.0075) * 0.01 * 100 * 0.04**2) ** (1 / 3)
    assert width.item() == pytest.approx(expected, rel=1e-14)
    assert expected == pytest.approx(0.13355, abs=1e-5)


def test_whalley_wilmott_width_scales_as_its_cube_root_law():
    # H is proportional to (eps S Gamma^2 e^{-r (T - t)} / gamma_ra)^{1/3}: multiplying one factor by 8 doubles H
    # (Gamma by 8 quadruples it), and these exact ratios catch a wrong exponent or a misplaced factor.
    base = b.whalley_wilmott_half_width(date_inputs(), 1.0, 0.01).item()

    def ratio(inputs=None, risk_aversion=1.0, proportional=0.01):
        width = b.whalley_wilmott_half_width(
            inputs if inputs is not None else date_inputs(), risk_aversion, proportional
        )
        return width.item() / base

    assert ratio(proportional=0.08) == pytest.approx(2.0, rel=1e-12)
    assert ratio(risk_aversion=8.0) == pytest.approx(0.5, rel=1e-12)
    assert ratio(date_inputs(spot=800.0)) == pytest.approx(2.0, rel=1e-12)
    assert ratio(date_inputs(gamma=0.32)) == pytest.approx(4.0, rel=1e-12)
    assert ratio(date_inputs(gamma=-0.04)) == pytest.approx(1.0, rel=1e-12)  # a slightly negative model gamma
    # the discount e^{-r (T - t)} enters under the cube root: r (T - t) = 0.0075 more lowers H by e^{-0.0025}
    assert ratio(date_inputs(time_left=0.5)) == pytest.approx(math.exp(-0.0075 / 3), rel=1e-12)


def test_whalley_wilmott_closes_the_band_where_there_is_no_gamma_or_no_cost():
    # Zero gamma is the put's exercise region, zero cost has nothing to save: both are the delta hedge.
    inputs = date_inputs(delta=-1.0, gamma=0.0)
    position = torch.tensor([-0.9], dtype=torch.float64)
    torch.testing.assert_close(
        b.whalley_wilmott(1.0, 0.01)(inputs, position), torch.tensor([-1.0], dtype=torch.float64)
    )
    inputs = date_inputs(delta=-0.4)
    torch.testing.assert_close(b.whalley_wilmott(1.0, 0.0)(inputs, position), torch.tensor([-0.4], dtype=torch.float64))


@pytest.mark.parametrize("target", b.BAND_TARGETS)
def test_whalley_wilmott_band_holds_inside_and_trades_outside(target):
    inputs = date_inputs(delta=-0.4, n=2)
    width = b.whalley_wilmott_half_width(inputs, 1.0, 0.01)[0].item()  # 0.13355, see the hand value above
    position = torch.tensor([-0.4 + 0.9 * width, -0.4 - 2 * width], dtype=torch.float64)  # inside, below
    out = b.whalley_wilmott(1.0, 0.01, target)(inputs, position)
    assert out[0].item() == position[0].item()  # held exactly
    assert out[1].item() == pytest.approx(-0.4 - width if target == "edge" else -0.4, rel=1e-14)


@pytest.mark.parametrize(
    ("risk_aversion", "proportional", "match"),
    [(0.0, 0.01, "risk_aversion"), (-1.0, 0.01, "risk_aversion"), (math.inf, 0.01, "risk_aversion"),
     (1.0, -0.01, "proportional"), (1.0, math.nan, "proportional")],
)  # fmt: skip
def test_whalley_wilmott_rejects_invalid_settings_when_built(risk_aversion, proportional, match):
    with pytest.raises(ValueError, match=match):
        b.whalley_wilmott(risk_aversion, proportional)


# through hedge_pnl


def test_band_saves_fixed_fees_that_the_delta_hedge_pays():
    # Small daily moves of delta: the delta hedge trades every day and pays the fee each time, a band of 0.05
    # trades only when delta has drifted 0.05 away, so it pays fewer fees.
    n_days = 21
    gen = torch.Generator().manual_seed(1)
    spot = 100.0 * torch.exp(torch.cumsum(0.01 * torch.randn(8, n_days, generator=gen, dtype=torch.float64), dim=1))
    delta = (-0.5 - 0.02 * torch.cumsum(torch.randn(8, n_days, generator=gen, dtype=torch.float64), dim=1)).clamp(-1, 0)
    inputs = b.baseline_inputs(spot, delta, 0.03, 0.1, 0.03)
    args = dict(inputs=inputs, spot=spot, tau=torch.full((8,), n_days - 1), premium=4.0, strike=STRIKE, rate=0.03)
    fee_only = {"fixed": 0.05, "kappa": 0.01}
    delta_fees = hedge_pnl(b.delta_hedge, **args, **fee_only).costs
    band_fees = hedge_pnl(b.band_hedge(0.05, "delta"), **args, **fee_only).costs
    assert torch.all(band_fees < delta_fees)
    # the delta hedge trades on every one of the 20 hedge dates (delta moves each day), plus the closing trade
    torch.testing.assert_close(
        delta_fees,
        torch.full_like(delta_fees, 0.05) * torch.exp(-0.03 * torch.arange(21) / 365).sum(),
        rtol=1e-3,
        atol=0,
    )


# tune


def test_tune_returns_the_best_value_and_every_score():
    # a score peaked at 0.3, built from the policy's output so tune must really build each policy
    inputs, position = date_inputs(delta=0.0), torch.tensor([1.0], dtype=torch.float64)

    def score(policy):
        return -((policy(inputs, position).item() - 0.3) ** 2)  # trades down to the edge w, best at w = 0.3

    best, scores = b.tune(b.band_hedge, [0.0, 0.1, 0.3, 0.5], score)
    assert best == 0.3
    assert scores == pytest.approx([-0.09, -0.04, 0.0, -0.04])


def test_tune_breaks_ties_with_the_first_value():
    best, _ = b.tune(b.band_hedge, [0.2, 0.1, 0.3], lambda policy: 1.0)
    assert best == 0.2


def test_tune_rejects_an_empty_grid_and_a_score_that_is_not_finite():
    with pytest.raises(ValueError, match="at least one"):
        b.tune(b.band_hedge, [], lambda policy: 0.0)
    with pytest.raises(ValueError, match="not a finite number"):
        b.tune(b.band_hedge, [0.1], lambda policy: math.nan)
