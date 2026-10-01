import numpy as np
import pytest

from AmericanHedging.baselines import delta_band, delta_hedge, no_hedge

# State layout (AmericanHedging/HedgeEnv.py::HedgingEnv._get_state):
# [S_t/K, t/T, delta, gamma, position]


def test_delta_hedge_returns_state_index_2():
    state = np.array([1.05, 0.5, 0.63, 0.01, 0.2])
    assert delta_hedge(state) == pytest.approx(0.63)


def test_delta_hedge_returns_negative_delta_for_put_state():
    # Put deltas are negative; delta_hedge must pass that sign through unchanged.
    state = np.array([0.9, 0.2, -0.45, 0.02, -0.1])
    assert delta_hedge(state) == pytest.approx(-0.45)


def test_delta_hedge_returns_zero_delta_unchanged():
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert delta_hedge(state) == pytest.approx(0.0)


def test_delta_hedge_ignores_every_component_other_than_index_2():
    # If delta_hedge ever read the wrong index (gamma, position, ...) this would
    # catch it: only state[2] may affect the result.
    base = np.array([1.0, 0.5, 0.33, 0.0, 0.0])
    changed = np.array([2.0, 0.9, 0.33, 5.0, -1.0])
    assert delta_hedge(base) == delta_hedge(changed)


@pytest.mark.parametrize(
    "state",
    [
        np.array([1.0, 0.0, 0.5, 0.1, 0.0]),
        np.array([0.5, 1.0, -0.9, 0.2, 1.0]),
        np.zeros(5),
    ],
)
def test_no_hedge_always_returns_zero_regardless_of_state(state):
    assert no_hedge(state) == 0.0


def test_no_hedge_return_type_is_python_float():
    assert isinstance(no_hedge(np.array([1.0, 2.0, 3.0, 4.0, 5.0])), float)


# --- delta_band ---------------------------------------------------------------

def _state(delta, position):
    return np.array([1.0, 0.5, delta, 0.02, position])


def test_delta_band_holds_inside_band():
    fn = delta_band(0.1)
    assert fn(_state(0.50, 0.55)) == pytest.approx(0.55)
    assert fn(_state(0.50, 0.45)) == pytest.approx(0.45)


def test_delta_band_holds_exactly_at_boundary_for_both_gap_signs():
    # |gap| == band must hold (<=, not <). Values are exact in binary so the
    # gap is exactly the band.
    fn = delta_band(0.25, "delta")
    assert fn(_state(0.5, 0.75)) == 0.75
    assert fn(_state(0.5, 0.25)) == 0.25
    fn_edge = delta_band(0.25, "edge")
    assert fn_edge(_state(0.5, 0.75)) == 0.75
    assert fn_edge(_state(0.5, 0.25)) == 0.25


def test_delta_band_returns_exactly_the_position_when_holding():
    # HedgingEnv only skips the fixed cost when |trade| <= 1e-8, so a hold must
    # return the position bit-for-bit, not an approximation of it.
    position = 0.123456789012345
    assert delta_band(0.5)(_state(0.4, position)) == position


@pytest.mark.parametrize("position", [0.8, 0.2])
def test_delta_band_target_delta_rebalances_all_the_way_to_delta(position):
    assert delta_band(0.1, "delta")(_state(0.5, position)) == pytest.approx(0.5)


def test_delta_band_target_edge_goes_to_nearest_edge_for_both_gap_signs():
    fn = delta_band(0.1, "edge")
    # position above delta: land on the upper edge; below: the lower edge.
    assert fn(_state(0.5, 0.9)) == pytest.approx(0.6)
    assert fn(_state(0.5, 0.1)) == pytest.approx(0.4)


def test_delta_band_default_target_is_delta():
    assert delta_band(0.1)(_state(0.5, 0.9)) == pytest.approx(0.5)


def test_delta_band_works_for_negative_put_deltas():
    assert delta_band(0.1, "edge")(_state(-0.5, -0.9)) == pytest.approx(-0.6)
    assert delta_band(0.1, "edge")(_state(-0.5, -0.1)) == pytest.approx(-0.4)
    assert delta_band(0.1, "delta")(_state(-0.5, -0.9)) == pytest.approx(-0.5)


@pytest.mark.parametrize("target", ["delta", "edge"])
def test_delta_band_zero_band_is_daily_delta_hedging(target):
    fn = delta_band(0.0, target)
    for delta, position in [(0.5, 0.0), (0.5, 0.9), (-0.3, 0.2)]:
        assert fn(_state(delta, position)) == pytest.approx(delta)


@pytest.mark.parametrize("target", ["delta", "edge"])
def test_delta_band_zero_band_holds_when_already_at_delta(target):
    assert delta_band(0.0, target)(_state(0.5, 0.5)) == 0.5


def test_delta_band_ignores_components_other_than_delta_and_position():
    fn = delta_band(0.1, "edge")
    a = np.array([1.0, 0.1, 0.5, 0.0, 0.9])
    b = np.array([7.0, 0.9, 0.5, 9.0, 0.9])
    assert fn(a) == fn(b)


@pytest.mark.parametrize("band", [-0.01, float("nan"), float("inf"), float("-inf")])
def test_delta_band_rejects_invalid_band(band):
    with pytest.raises(ValueError, match="band"):
        delta_band(band)


@pytest.mark.parametrize("target", ["Delta", "", "nearest"])
def test_delta_band_rejects_unknown_target(target):
    with pytest.raises(ValueError, match="target"):
        delta_band(0.1, target)
