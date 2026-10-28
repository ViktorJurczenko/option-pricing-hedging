"""
delta_band run through a real HedgingEnv (real GBM + fitted LSM) with a fixed
cost: steps where it holds must not be charged. This is the reason delta_band
returns exactly the current position -- HedgingEnv only skips the fixed cost
when |trade| <= _TRADE_EPS.
"""

import numpy as np
import pytest
import torch

from optionpricing.core.gbm import GBM
from optionpricing.hedging.baselines import delta_band
from optionpricing.hedging.env import HedgingEnv
from optionpricing.ml.rl.policy import RLPolicy
from optionpricing.ml.rl.sac import deterministic_hedge
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz

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


# --- the learned band (RLPolicy action_mode="band") in the same env ----------


def _constant_band_policy(width_logit, target_logit):
    """Band-mode policy whose deterministic action is the same for every state."""
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    with torch.no_grad():
        policy.layer3.weight.zero_()
        policy.layer3.bias[0] = width_logit
        policy.layer3.bias[1] = target_logit
    squashed = torch.tanh(policy(torch.zeros(5)).mean).double()
    b = (policy.band_max * (squashed[0] + 1) / 2).item()
    return policy, b


@pytest.mark.parametrize("target,target_logit", [("delta", 20.0), ("edge", -20.0)])
def test_constant_band_policy_pnl_equals_delta_band_baseline(lsm, target, target_logit):
    """Same hedges, so the same P&L, fixed costs included: every band-mode
    hold is an exact zero trade."""
    policy, b = _constant_band_policy(0.0, target_logit)  # b = 0.1
    env = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=FIXED_COST)
    paths = np.arange(env.n_paths)
    learned = env.rollout_pnl(lambda s: deterministic_hedge(policy, s), paths)
    baseline = env.rollout_pnl(lambda s: np.array([delta_band(b, target)(row) for row in s]), paths)
    # Within float32 precision of the network's hedges, far below one fixed
    # cost (FIXED_COST / K = 0.005), so no hold was charged it.
    np.testing.assert_allclose(learned, baseline, rtol=0, atol=1e-6)


def test_constant_band_policy_trades_and_costs_match_baseline_step_by_step(lsm):
    """The step()-based path test-hedge's evaluate() uses: same steps held
    and traded, and same costs, as the baseline."""
    policy, b = _constant_band_policy(0.0, 20.0)
    env = HedgingEnv(lsm, prop_tc=0.0, fixed_tc=FIXED_COST)
    learned = _implied_costs(env, lambda s: deterministic_hedge(policy, s).item(), n_paths=20)
    baseline = _implied_costs(env, delta_band(b, "delta"), n_paths=20)
    assert any(not traded for traded, _ in learned), "the band never held - test is vacuous"
    assert [traded for traded, _ in learned] == [traded for traded, _ in baseline]
    np.testing.assert_allclose([c for _, c in learned], [c for _, c in baseline], atol=1e-9)
