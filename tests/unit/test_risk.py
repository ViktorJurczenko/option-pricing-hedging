"""Tests for optionpricing.hedging.risk: CVaR, its Rockafellar-Uryasev training form, entropic risk, variance."""

import math

import numpy as np
import pytest
import torch

from optionpricing.hedging.risk import cvar, cvar_loss, entropic, variance

# 20 hand-made outcomes, worst first. At alpha = 90% the tail is the two worst, so the CVaR is -3.4.
OUTCOMES = [-4.2, -2.6, -1.7, -1.3, -1.0, -0.8, -0.6, -0.4, -0.3, -0.2]
OUTCOMES += [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.9, 1.4, 2.1]


def outcomes(requires_grad=False):
    """The 20 outcomes, shuffled so no function can rely on sorted input."""
    order = np.random.default_rng(0).permutation(len(OUTCOMES))
    return torch.tensor([OUTCOMES[i] for i in order], dtype=torch.float64, requires_grad=requires_grad)


def ru_minimum(pnl, alpha):
    """Exact minimum over w of the Rockafellar-Uryasev objective, the reference for cvar.

    The objective is convex and piecewise linear in w with kinks at the losses, so its minimum is reached at
    one of them: evaluating it at every loss and taking the smallest is exact, with no optimizer involved.
    """
    return min(cvar_loss(pnl, w, alpha).item() for w in (-pnl).tolist())


# cvar


def test_cvar_averages_the_two_worst_of_20_at_90_percent():
    # (-4.2 - 2.6) / 2
    assert cvar(outcomes(), alpha=0.9).item() == pytest.approx(-3.4, abs=1e-12)


def test_cvar_at_95_percent_of_20_outcomes_is_the_worst_one():
    assert cvar(outcomes(), alpha=0.95).item() == pytest.approx(-4.2, abs=1e-12)


def test_cvar_counts_the_edge_outcome_for_its_share_of_a_fractional_tail():
    # 20 * 7.5% = 1.5 outcomes: all of the worst, half of the second worst
    assert cvar(outcomes(), alpha=0.925).item() == pytest.approx((-4.2 - 0.5 * 2.6) / 1.5, abs=1e-12)


def test_cvar_with_a_tail_below_one_outcome_is_the_worst_outcome():
    # 20 * 1% = 0.2 outcomes, less than one path: the worst one alone
    assert cvar(outcomes(), alpha=0.99).item() == pytest.approx(-4.2, abs=1e-12)


def test_cvar_matches_the_mean_of_the_worst_5_of_100_at_95_percent():
    # (1 - 0.95) * 100 is 5.000000000000004 in floating point; the 6th outcome must get no visible weight
    pnl = torch.tensor(np.random.default_rng(1).normal(size=100))
    expected = torch.sort(pnl).values[:5].mean().item()
    assert cvar(pnl, alpha=0.95).item() == pytest.approx(expected, abs=1e-12)


def test_cvar_of_a_constant_sample_is_the_constant():
    assert cvar(torch.full((37,), -1.25, dtype=torch.float64), alpha=0.95).item() == pytest.approx(-1.25)


def test_cvar_is_at_most_the_mean():
    pnl = torch.tensor(np.random.default_rng(2).normal(size=1000))
    assert cvar(pnl, alpha=0.95).item() < pnl.mean().item()


def test_cvar_gradient_is_spread_evenly_over_the_tail():
    pnl = outcomes(requires_grad=True)
    cvar(pnl, alpha=0.9).backward()
    expected = torch.where(pnl.detach() <= -2.6, 0.5, 0.0).to(torch.float64)
    torch.testing.assert_close(pnl.grad, expected)


def test_cvar_accepts_a_plain_list_of_integers():
    assert cvar([3, -1, 2, -5], alpha=0.75).item() == pytest.approx(-5.0)


def test_a_float32_tensor_stays_float32():
    # training runs in float32; the measure must not silently upcast the graph
    assert cvar(torch.zeros(10, dtype=torch.float32)).dtype == torch.float32


# cvar_loss


@pytest.mark.parametrize(
    ("w", "expected"),
    # computed by hand at alpha = 90%: flat at 3.40 for w between the VaR bounds 1.7 and 2.6
    [(0.5, 4.85), (1.0, 3.90), (1.7, 3.40), (2.0, 3.40), (2.6, 3.40), (3.0, 3.60)],
)
def test_cvar_loss_hand_values_around_the_var(w, expected):
    assert cvar_loss(outcomes(), w, alpha=0.9).item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.9, 0.925, 0.95, 0.99, 0.5])
def test_cvar_loss_minimum_is_minus_cvar_on_the_20_outcomes(alpha):
    pnl = outcomes()
    assert ru_minimum(pnl, alpha) == pytest.approx(-cvar(pnl, alpha).item(), abs=1e-12)


@pytest.mark.parametrize("n", [1, 7, 333, 1000])
def test_cvar_loss_minimum_is_minus_cvar_for_any_sample_size(n):
    # sizes where 5% of n is below one, fractional and whole
    pnl = torch.tensor(np.random.default_rng(n).standard_t(df=3, size=n))
    assert ru_minimum(pnl, 0.95) == pytest.approx(-cvar(pnl, 0.95).item(), abs=1e-10)


@pytest.mark.parametrize(
    ("w", "expected_grad"),
    # d/dw = 1 - P(L > w) / (1 - alpha): 1 of 20 losses above 3.0, 4 of 20 above 1.0
    [(3.0, 1 - (1 / 20) / 0.1), (1.0, 1 - (4 / 20) / 0.1)],
)
def test_cvar_loss_gradient_in_w_points_to_the_var(w, expected_grad):
    w = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    cvar_loss(outcomes(), w, alpha=0.9).backward()
    assert w.grad.item() == pytest.approx(expected_grad, abs=1e-12)


def test_cvar_loss_gradient_in_pnl_falls_on_losses_above_w_only():
    pnl = outcomes(requires_grad=True)
    cvar_loss(pnl, 2.0, alpha=0.9).backward()
    # each loss above w = 2.0 adds -1 / (n (1 - alpha)) = -0.5 per unit of P&L
    expected = torch.where(pnl.detach() < -2.0, -0.5, 0.0).to(torch.float64)
    torch.testing.assert_close(pnl.grad, expected)


def test_cvar_loss_rejects_a_vector_w():
    with pytest.raises(ValueError, match="w must be a scalar"):
        cvar_loss(outcomes(), torch.zeros(2), alpha=0.9)


# entropic


def test_entropic_hand_value():
    # (1/1) ln((e^0 + e^1) / 2)
    assert entropic([0.0, -1.0], lam=1.0).item() == pytest.approx(math.log((1 + math.e) / 2), abs=1e-12)


def test_entropic_does_not_overflow_on_a_large_loss():
    # exp(1000) overflows float64; the exact value is 1000 + ln((1 + e^-1000) / 2) = 1000 - ln 2
    value = entropic(torch.tensor([-1000.0, 0.0], dtype=torch.float64), lam=1.0).item()
    assert value == pytest.approx(1000 - math.log(2), abs=1e-9)


def test_entropic_shifts_down_by_a_cash_amount_added_to_every_path():
    pnl = outcomes()
    assert entropic(pnl + 0.7, lam=0.5).item() == pytest.approx(entropic(pnl, lam=0.5).item() - 0.7, abs=1e-12)


def test_entropic_tends_to_minus_the_mean_for_small_lam():
    pnl = outcomes()
    # the expansion is -mean + lam * var / 2 + O(lam^2); var is 1.74, so at lam = 1e-6 the gap is ~1e-6
    assert entropic(pnl, lam=1e-6).item() == pytest.approx(-pnl.mean().item(), abs=2e-6)


def test_entropic_tends_to_the_worst_loss_for_large_lam():
    pnl = outcomes()
    # only the worst path survives the exponential (the next is e^-16000 smaller), leaving
    # (1 / lam) ln(e^(4.2 lam) / 20) = 4.2 - ln(20) / lam, which tends to the worst loss 4.2
    assert entropic(pnl, lam=1e4).item() == pytest.approx(4.2 - math.log(20) / 1e4, abs=1e-12)


def test_entropic_lies_between_minus_the_mean_and_the_worst_loss():
    pnl = outcomes()
    assert -pnl.mean().item() < entropic(pnl, lam=1.0).item() < 4.2


@pytest.mark.parametrize("lam", [0.0, -1.0, math.inf, math.nan])
def test_entropic_rejects_a_non_positive_or_non_finite_lam(lam):
    with pytest.raises(ValueError, match="lam must be positive"):
        entropic(outcomes(), lam=lam)


# variance


def test_variance_hand_value():
    # mean 2.5, squared distances 2.25, 0.25, 0.25, 2.25
    assert variance([1.0, 2.0, 3.0, 4.0]).item() == pytest.approx(1.25)


def test_variance_matches_numpy_population_variance():
    sample = np.random.default_rng(3).normal(size=500)
    assert variance(torch.tensor(sample)).item() == pytest.approx(np.var(sample), rel=1e-12)


def test_variance_of_one_path_is_zero():
    assert variance([3.0]).item() == 0.0


def test_variance_gradient_is_twice_the_deviation_over_n():
    pnl = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    variance(pnl).backward()
    torch.testing.assert_close(pnl.grad, 2 * (pnl.detach() - 2.5) / 4)


# input validation, shared by all four


MEASURES = {
    "cvar": lambda pnl: cvar(pnl, 0.95),
    "cvar_loss": lambda pnl: cvar_loss(pnl, 0.0, 0.95),
    "entropic": lambda pnl: entropic(pnl, 1.0),
    "variance": variance,
}


@pytest.mark.parametrize("measure", MEASURES.values(), ids=MEASURES.keys())
def test_rejects_an_empty_sample(measure):
    with pytest.raises(ValueError, match="empty"):
        measure(torch.zeros(0))


@pytest.mark.parametrize("measure", MEASURES.values(), ids=MEASURES.keys())
def test_rejects_a_two_dimensional_sample(measure):
    with pytest.raises(ValueError, match="1-D"):
        measure(torch.zeros(4, 5))


@pytest.mark.parametrize("risk", [cvar, cvar_loss], ids=["cvar", "cvar_loss"])
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_rejects_alpha_outside_the_open_unit_interval(risk, alpha):
    args = (outcomes(), alpha) if risk is cvar else (outcomes(), 0.0, alpha)
    with pytest.raises(ValueError, match="alpha must be in"):
        risk(*args)
