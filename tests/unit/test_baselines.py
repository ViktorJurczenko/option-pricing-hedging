import numpy as np
import pytest

from AmericanHedging.baselines import delta_hedge, no_hedge

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
