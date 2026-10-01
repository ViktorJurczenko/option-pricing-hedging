"""
delta_band run through a real HedgingEnv (real GBM + fitted LSM) with a fixed
cost: steps where it holds must not be charged. This is the reason delta_band
returns exactly the current position -- HedgingEnv only skips the fixed cost
when |trade| <= _TRADE_EPS.
"""
import numpy as np
import pytest

from core.gbm import GBM
from AmericanOptionPricing.LongstaffSchwartz import LongstaffSchwartz
from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.baselines import delta_band

FIXED_COST = 0.5


@pytest.fixture(scope="module")
def lsm():
    sim = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=3).simulate_paths(60, 0.25, 40)
    model = LongstaffSchwartz(sim, 100.0, 0.05, 0.25, "call", "simple", 2, sigma=0.2, control_variate=True)
    model.predict_payoff()
    return model


def _implied_costs(env, action_fn, n_paths):
    """Per step: (traded?, cost implied by the reward). Terminal mode with
    tc=0: reward*K = position*dS - cost - payoff_if_done."""
    out = []
    sim, K = env.lsm.simulation, env.lsm.strike
    for p in range(n_paths):
        state, done = env.reset(p), False
        while not done:
            t, pos = env.t, env.position
            action = action_fn(state)
            traded = abs(np.clip(action, -1, 1) - pos) > env._TRADE_EPS
            state, reward, done = env.step(action)
            payoff = env._payoff(sim[t + 1, p]) if done else 0.0
            cost = pos * (sim[t + 1, p] - sim[t, p]) - payoff - reward * K
            out.append((traded, cost))
    return out


@pytest.mark.parametrize("target", ["delta", "edge"])
def test_delta_band_holds_are_not_charged_fixed_tc_and_trades_are(lsm, target):
    env = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=FIXED_COST)
    steps = _implied_costs(env, delta_band(0.15, target), n_paths=30)

    held = [c for traded, c in steps if not traded]
    traded = [c for traded, c in steps if traded]
    assert held and traded  # both branches exercised, else the check is vacuous
    np.testing.assert_allclose(held, 0.0, atol=1e-9)
    np.testing.assert_allclose(traded, FIXED_COST, atol=1e-9)


def test_delta_band_charges_fewer_fixed_tcs_than_daily_delta(lsm):
    env = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=FIXED_COST)
    banded = sum(c for _, c in _implied_costs(env, delta_band(0.15), 30))
    daily = sum(c for _, c in _implied_costs(env, delta_band(0.0), 30))
    assert banded < daily
