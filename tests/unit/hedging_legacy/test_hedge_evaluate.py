"""optionpricing.hedging.legacy.evaluation.evaluate() and cvar()."""

import numpy as np
import pytest

from optionpricing.hedging.legacy.baselines import delta_hedge, no_hedge
from optionpricing.hedging.legacy.env import HedgingEnv
from optionpricing.hedging.legacy.evaluation import cvar, evaluate
from tests.helpers import two_path_call_lsm


def test_evaluate_returns_mean_std_and_cvar_matching_manual_calculation(monkeypatch):
    """With no_hedge and a fixed path, every episode is identical, so mean_pnl
    must equal that one reward, std_pnl must be 0, and cvar_95
    collapses to the same value too."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)  # force path_idx = 0
    env = HedgingEnv(two_path_call_lsm())

    mean_pnl, std_pnl, cvar_95 = evaluate(no_hedge, env, n_episodes=4)

    # path 0, no_hedge (position stays 0 throughout):
    # step1: t=0->1, S 100->105, hedge_pnl=0, cost=0, not done -> reward=0
    # step2: t=1->2 (maturity), S 105->110, hedge_pnl=0, cost=0, done,
    #        call payoff = max(110-100,0)=10 -> reward=(0-0-10)/100=-0.1
    expected_reward = -0.1
    assert mean_pnl == pytest.approx(expected_reward)
    assert std_pnl == pytest.approx(0.0)
    assert cvar_95 == pytest.approx(expected_reward)


def test_evaluate_accepts_plain_callable_action_fn_not_tied_to_nn_module(monkeypatch):
    """Regression test: evaluate() used to call `policy(state).mean.item()`,
    coupling it to RLPolicy's Gaussian-actor interface. It must now drive any
    plain state -> float callable (e.g. a baseline with no .mean attribute)
    without raising, and call it once per environment step."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env = HedgingEnv(two_path_call_lsm())

    calls = []

    def action_fn(state):
        calls.append(state.copy())
        return 0.0

    evaluate(action_fn, env, n_episodes=3)

    # exercise_time=2 for the fixed path -> 2 action_fn calls per episode
    assert len(calls) == 3 * 2


def test_evaluate_passes_current_env_state_to_action_fn(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    env = HedgingEnv(two_path_call_lsm())

    seen_states = []

    def action_fn(state):
        seen_states.append(state.copy())
        return 0.0

    evaluate(action_fn, env, n_episodes=1)

    # First call must see the reset state: S0/K=1.0, t/T=0, delta=gamma=0, position=0.
    np.testing.assert_allclose(seen_states[0], [1.0, 0.0, 0.0, 0.0, 0.0])
    # Second call: after step 1, t=1, S=105/100, normalized time t/(n_steps-1)=1/2
    # (3 rows -> last row index 2 is expiry), position now 0 (no_hedge action).
    np.testing.assert_allclose(seen_states[1], [1.05, 1 / 2, 0.0, 0.0, 0.0])


def test_evaluate_with_delta_hedge_baseline_hand_calculated_reward(monkeypatch):
    """End-to-end check that baselines.delta_hedge, evaluate(), and HedgingEnv
    compose correctly under the new action_fn interface, with a non-zero,
    time-varying delta driving the hedge position."""
    monkeypatch.setattr(np.random, "randint", lambda n: 0)
    delta = np.array(
        [
            [0.5, -0.3],
            [0.4, 0.2],
        ]
    )
    env = HedgingEnv(two_path_call_lsm(delta=delta))

    mean_pnl, std_pnl, cvar_95 = evaluate(delta_hedge, env, n_episodes=1)

    # path 0:
    # t=0: state delta=0.5 -> action=0.5, position 0->0.5
    #   hedge_pnl = 0 (old position) * (105-100) = 0
    #   cost = 0.001 * |0.5-0| * 100 = 0.05
    #   not done -> reward1 = (0 - 0.05) / 100
    reward1 = (0.0 - 0.001 * 0.5 * 100.0) / 100.0
    # t=1: state delta=0.4 -> action=0.4, position 0.5->0.4
    #   hedge_pnl = 0.5 (old position) * (110-105) = 2.5
    #   cost = 0.001 * |0.4-0.5| * 105 = 0.0105
    #   done -> payoff = max(110-100,0) = 10
    #   reward2 = (2.5 - 0.0105 - 10) / 100
    reward2 = (0.5 * 5.0 - 0.001 * 0.1 * 105.0 - 10.0) / 100.0
    expected_total = reward1 + reward2

    assert mean_pnl == pytest.approx(expected_total)
    assert std_pnl == pytest.approx(0.0)
    assert cvar_95 == pytest.approx(expected_total)


def test_evaluate_prints_var_above_cvar_and_cvar_is_tail_mean_not_percentile(monkeypatch, capsys):
    """Regression test: the "CVaR 95%" line used to print np.percentile(pnls, 5),
    which is the VaR. With differing P&Ls the two must differ: CVaR is the mean
    of the worst outcome(s), VaR the interpolated 5th percentile."""
    # 10 episodes: the first runs path 0 (P&L -0.1 under no_hedge), the rest path 1 (P&L 0).
    paths = iter([0] + [1] * 9)
    monkeypatch.setattr(np.random, "randint", lambda n: next(paths))
    env = HedgingEnv(two_path_call_lsm())

    mean_pnl, std_pnl, cvar_95 = evaluate(no_hedge, env, n_episodes=10)

    pnls = np.array([-0.1] + [0.0] * 9)
    # n=10 -> worst ceil(0.5)=1 outcome: -0.1. 5th percentile interpolates
    # between sorted[0]=-0.1 and sorted[1]=0 at index 0.45 -> -0.055.
    assert cvar_95 == pytest.approx(-0.1)
    assert np.percentile(pnls, 5) == pytest.approx(-0.055)
    assert mean_pnl == pytest.approx(-0.01)
    out = capsys.readouterr().out.splitlines()
    var_lines = [line for line in out if line.startswith("VaR  95%:")]
    cvar_lines = [line for line in out if line.startswith("CVaR 95%:")]
    assert len(var_lines) == 1 and len(cvar_lines) == 1
    assert float(var_lines[0].split(":")[1]) == pytest.approx(-0.055, abs=1e-4)
    assert float(cvar_lines[0].split(":")[1]) == pytest.approx(-0.1, abs=1e-4)
    assert out.index(var_lines[0]) < out.index(cvar_lines[0])  # VaR directly above CVaR
    assert out.index(cvar_lines[0]) - out.index(var_lines[0]) == 1


# --- cvar() tests ---------------------------------------------------------


def test_cvar_n100_is_mean_of_five_smallest():
    # 0..99 shuffled: worst 5 are 0..4 -> mean 2.0. (Guards the float pitfall:
    # (1 - 0.95) * 100 == 5.000000000000004, a bare ceil would average 6 values -> 2.5.)
    pnls = np.random.default_rng(0).permutation(100).astype(float)
    assert cvar(pnls) == pytest.approx(2.0)


def test_cvar_n4_is_the_minimum():
    assert cvar([3.0, -2.0, 5.0, 1.0]) == -2.0


def test_cvar_n21_is_mean_of_two_smallest():
    # ceil(0.05 * 21) = ceil(1.05) = 2
    pnls = list(range(21))
    assert cvar(pnls[::-1]) == pytest.approx(0.5)


def test_cvar_single_sample_returns_it():
    assert cvar([7.0]) == 7.0


def test_cvar_custom_level():
    pnls = np.arange(10, dtype=float)
    assert cvar(pnls, level=0.5) == pytest.approx(2.0)  # worst 5: 0..4
    assert cvar(pnls, level=0.8) == pytest.approx(0.5)  # worst 2: 0, 1


def test_cvar_does_not_mutate_input_and_handles_unsorted_lists():
    pnls = [5.0, -1.0, 3.0, -4.0]
    assert cvar(pnls, level=0.5) == pytest.approx(-2.5)  # worst 2: -4, -1
    assert pnls == [5.0, -1.0, 3.0, -4.0]


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1])
def test_cvar_rejects_level_outside_open_unit_interval(level):
    with pytest.raises(ValueError, match="level"):
        cvar([1.0, 2.0, 3.0], level=level)


def test_cvar_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        cvar([])


@pytest.mark.parametrize("n_episodes", [0, -3])
def test_evaluate_rejects_non_positive_n_episodes(n_episodes):
    # Must fail up front, before touching the env (None would blow up otherwise).
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate(no_hedge, None, n_episodes=n_episodes)


def test_cvar_returns_python_float():
    assert type(cvar(np.array([1.0, 2.0, 3.0]))) is float


def test_cvar_never_exceeds_var_over_random_samples():
    rng = np.random.default_rng(1)
    for n in [1, 2, 5, 20, 21, 100, 333]:
        pnls = rng.normal(size=n)
        assert cvar(pnls) <= np.percentile(pnls, 5) + 1e-12, n
