import math

import numpy as np
import pytest
import torch

from AmericanHedging.HedgeEnv import HedgingEnv
from AmericanHedging.RLPolicy import (
    _DELTA_IDX, _INITIAL_BAND_FRACTION, _INITIAL_BAND_TARGET, _INITIAL_GAP_FRACTION, _POSITION_IDX,
    RLPolicy, checkpoint_action_dim,
)
from AmericanHedging.baselines import delta_band
from AmericanOptionPricing import LongstaffSchwartz
from core.gbm import GBM


def _state(delta, batch=None):
    """State in HedgingEnv layout [S/K, t/T, delta, gamma, position]; other
    entries are deliberately non-zero so a wrong index would change results."""
    row = torch.tensor([1.1, 0.4, delta, 0.7, -0.3])
    return row if batch is None else row.repeat(batch, 1)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_residual_scale_raises(bad):
    # NaN fails `>= 0` silently if only negativity is checked; inf would make
    # the residual unbounded, defeating the clamp-around-delta design.
    with pytest.raises(ValueError, match="residual_scale"):
        RLPolicy(5, 1, residual_scale=bad)


@pytest.mark.parametrize("ok", [0.0, 0.1])
def test_valid_residual_scale_accepted(ok):
    assert RLPolicy(5, 1, residual_scale=ok).residual_scale.item() == pytest.approx(ok)


def test_residual_scale_is_buffer_saved_in_state_dict():
    policy = RLPolicy(5, 1, residual_scale=0.25)
    sd = policy.state_dict()
    assert "residual_scale" in sd
    assert sd["residual_scale"].item() == pytest.approx(0.25)
    # A buffer, not a trainable parameter: the optimizer must never touch it.
    assert "residual_scale" not in dict(policy.named_parameters())


def test_hedge_ratio_is_identity_when_residual_scale_zero():
    policy = RLPolicy(5, 1)
    squashed = torch.tensor([0.37])
    out = policy.hedge_ratio(_state(0.9), squashed)
    assert torch.equal(out, squashed)  # delta must be ignored entirely


def test_hedge_ratio_adds_scaled_residual_to_delta_single_state():
    policy = RLPolicy(5, 1, residual_scale=0.1)
    out = policy.hedge_ratio(_state(0.5), torch.tensor([-0.4]))
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.5 + 0.1 * -0.4)


def test_hedge_ratio_batched_uses_each_rows_own_delta():
    policy = RLPolicy(5, 1, residual_scale=0.2)
    states = torch.stack([_state(0.1), _state(-0.6), _state(0.5)])
    squashed = torch.tensor([[1.0], [0.5], [-0.5]])
    out = policy.hedge_ratio(states, squashed)
    assert out.shape == (3, 1)
    torch.testing.assert_close(out, torch.tensor([[0.3], [-0.5], [0.4]]))


@pytest.mark.parametrize("delta,squashed,expected", [
    (0.98, 0.9, 1.0),     # 0.98 + 0.09 = 1.07 -> upper clamp
    (-0.98, -0.9, -1.0),  # lower clamp
])
def test_hedge_ratio_clamped_to_unit_interval(delta, squashed, expected):
    policy = RLPolicy(5, 1, residual_scale=0.1)
    out = policy.hedge_ratio(_state(delta, batch=2), torch.tensor([[squashed]] * 2))
    torch.testing.assert_close(out, torch.full((2, 1), expected))


def test_hedge_ratio_residual_is_bounded_around_delta():
    # Even the extreme action only moves the hedge by residual_scale.
    policy = RLPolicy(5, 1, residual_scale=0.1)
    hi = policy.hedge_ratio(_state(0.0), torch.tensor([1.0])).item()
    lo = policy.hedge_ratio(_state(0.0), torch.tensor([-1.0])).item()
    assert hi == pytest.approx(0.1)
    assert lo == pytest.approx(-0.1)


def test_load_legacy_state_dict_without_residual_scale_sets_zero_strict():
    source = RLPolicy(5, 1)
    legacy = {k: v for k, v in source.state_dict().items() if k != "residual_scale"}
    target = RLPolicy(5, 1, residual_scale=0.3)

    target.load_state_dict(legacy)  # strict=True by default; must not raise

    assert target.residual_scale.item() == 0.0
    # and the mapping is back to absolute-action behaviour
    squashed = torch.tensor([0.42])
    assert torch.equal(target.hedge_ratio(_state(0.9), squashed), squashed)


def test_load_legacy_state_dict_does_not_mutate_callers_dict():
    legacy = {k: v for k, v in RLPolicy(5, 1).state_dict().items() if k != "residual_scale"}
    keys_before = set(legacy)
    RLPolicy(5, 1).load_state_dict(legacy)
    assert set(legacy) == keys_before


def test_load_new_checkpoint_restores_residual_scale_and_weights():
    torch.manual_seed(0)
    source = RLPolicy(5, 1, residual_scale=0.35)
    target = RLPolicy(5, 1)  # default 0

    target.load_state_dict(source.state_dict())

    assert target.residual_scale.item() == pytest.approx(0.35)
    x = _state(0.2)
    torch.testing.assert_close(target(x).mean, source(x).mean)


def test_load_still_rejects_unexpected_keys():
    # The legacy shim must only relax the missing residual_scale key, not strictness in general.
    sd = RLPolicy(5, 1).state_dict()
    sd["bogus"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        RLPolicy(5, 1).load_state_dict(sd)


def test_delta_idx_matches_hedging_env_state_layout():
    # _DELTA_IDX is a hand-copied constant; this fails if HedgingEnv._get_state
    # reorders its entries without the constant being updated.
    np.random.seed(0)
    simulation = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(
        nb_predictions=60, maturity=1, sampling=6)
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0,
                            option_type="put", max_degree=2)
    lsm.predict_payoff()
    env = HedgingEnv(lsm)

    state = env.reset()
    steps_checked = 0
    while True:
        t = min(env.t, lsm.delta.shape[0] - 1)
        expected_delta = lsm.delta[t, env.path_idx]
        assert state[_DELTA_IDX] == expected_delta
        # Guard against coincidental matches: delta must differ from the
        # neighbouring entries (gamma, position, S/K) at the checked point.
        assert expected_delta != lsm.gamma[t, env.path_idx]
        assert expected_delta != state[4]
        steps_checked += 1
        if steps_checked >= 3:
            break
        state, _, done = env.step(0.5)
        if done:
            break
    assert steps_checked >= 2


# --- gap_fraction action mode -------------------------------------------------

def _gap_state(delta, position, batch=None):
    row = torch.tensor([1.1, 0.4, delta, 0.7, position])
    return row if batch is None else row.repeat(batch, 1)


@pytest.mark.parametrize("bad", ["gap", "Residual", "", None])
def test_invalid_action_mode_raises(bad):
    with pytest.raises(ValueError, match="action_mode"):
        RLPolicy(5, 1, action_mode=bad)


def test_default_action_mode_is_residual():
    assert RLPolicy(5, 1).gap_fraction.item() == 0.0
    assert RLPolicy(5, 1, action_mode="residual").gap_fraction.item() == 0.0


def test_gap_fraction_is_buffer_saved_in_state_dict():
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    assert policy.state_dict()["gap_fraction"].item() == 1.0
    assert RLPolicy(5, 1).state_dict()["gap_fraction"].item() == 0.0
    assert "gap_fraction" not in dict(policy.named_parameters())


@pytest.mark.parametrize("squashed,expected", [
    (-1.0, 0.3),    # w = 0: hold exactly the current position
    (1.0, -0.2),    # w = 1: go all the way to delta
    (0.0, 0.05),    # w = 0.5: midpoint of position 0.3 and delta -0.2
    (0.5, -0.075),  # w = 0.75
])
def test_gap_fraction_hedge_ratio_interpolates_position_to_delta(squashed, expected):
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    out = policy.hedge_ratio(_gap_state(delta=-0.2, position=0.3), torch.tensor([squashed]))
    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_gap_fraction_batched_uses_each_rows_own_delta_and_position():
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    states = torch.stack([_gap_state(0.5, 0.1), _gap_state(-0.4, 0.6), _gap_state(0.9, -0.9)])
    squashed = torch.tensor([[1.0], [0.0], [-1.0]])
    out = policy.hedge_ratio(states, squashed)
    assert out.shape == (3, 1)
    torch.testing.assert_close(out, torch.tensor([[0.5], [0.1], [-0.9]]))


def test_gap_fraction_ignores_residual_scale():
    state, squashed = _gap_state(0.5, -0.1), torch.tensor([0.3])
    a = RLPolicy(5, 1, residual_scale=0.0, action_mode="gap_fraction").hedge_ratio(state, squashed)
    b = RLPolicy(5, 1, residual_scale=0.9, action_mode="gap_fraction").hedge_ratio(state, squashed)
    assert torch.equal(a, b)


def test_gap_fraction_output_clamped_to_unit_interval():
    # Out-of-range positions cannot come from the env, but the clamp is part of
    # the contract: the hedge must never leave [-1, 1].
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    assert policy.hedge_ratio(_gap_state(1.0, 1.5), torch.tensor([-1.0])).item() == 1.0
    assert policy.hedge_ratio(_gap_state(-1.0, -1.5), torch.tensor([-1.0])).item() == -1.0


def test_position_idx_matches_hedging_env_state_layout():
    # _POSITION_IDX is a hand-copied constant; fails if HedgingEnv._get_state reorders.
    np.random.seed(0)
    simulation = GBM(sigma=0.2, start_price=100.0, risk_free_rate=0.05, seed=1).simulate_paths(
        nb_predictions=60, maturity=1, sampling=6)
    lsm = LongstaffSchwartz(simulation, strike=100.0, risk_free_rate=0.05, maturity=1.0,
                            option_type="put", max_degree=2)
    lsm.predict_payoff()
    env = HedgingEnv(lsm)
    state = env.reset()
    assert state[_POSITION_IDX] == 0.0
    state, _, _ = env.step(0.37)
    assert state[_POSITION_IDX] == pytest.approx(0.37)


def test_gap_fraction_init_bias_gives_initial_gap_fraction_with_zero_weights():
    policy = RLPolicy(5, 1, action_mode="gap_fraction")
    assert _INITIAL_GAP_FRACTION == 0.9
    with torch.no_grad():
        policy.layer3.weight.zero_()
    mean = policy(_gap_state(0.5, 0.0)).mean
    assert torch.tanh(mean).item() == pytest.approx(0.8, abs=1e-6)
    assert ((torch.tanh(mean) + 1) / 2).item() == pytest.approx(0.9, abs=1e-6)


def test_gap_fraction_init_only_touches_mean_half_of_bias():
    torch.manual_seed(0)
    default = RLPolicy(5, 1)
    torch.manual_seed(0)
    gap = RLPolicy(5, 1, action_mode="gap_fraction")
    assert gap.layer3.bias[1].item() == default.layer3.bias[1].item()  # log_std half
    assert gap.layer3.bias[0].item() == pytest.approx(math.atanh(0.8), abs=1e-6)
    torch.testing.assert_close(gap.layer3.weight, default.layer3.weight)


def test_gap_fraction_init_covers_all_action_dims():
    policy = RLPolicy(5, 3, action_mode="gap_fraction")
    torch.testing.assert_close(policy.layer3.bias[:3], torch.full((3,), math.atanh(0.8)))


def test_residual_mode_keeps_default_bias_init():
    torch.manual_seed(0)
    a = RLPolicy(5, 1)
    torch.manual_seed(0)
    b = RLPolicy(5, 1, residual_scale=0.1, action_mode="residual")
    torch.testing.assert_close(a.layer3.bias, b.layer3.bias)
    assert abs(b.layer3.bias[0].item() - math.atanh(0.8)) > 1e-3


def test_load_pre_gap_fraction_state_dict_loads_as_residual_strict():
    legacy = {k: v for k, v in RLPolicy(5, 1, residual_scale=0.2).state_dict().items()
              if k != "gap_fraction"}
    target = RLPolicy(5, 1, action_mode="gap_fraction")
    target.load_state_dict(legacy)  # strict; must not raise
    assert target.gap_fraction.item() == 0.0
    assert target.residual_scale.item() == pytest.approx(0.2)
    out = target.hedge_ratio(_gap_state(0.5, -0.1), torch.tensor([0.5]))
    assert out.item() == pytest.approx(0.5 + 0.2 * 0.5)


def test_load_gap_fraction_checkpoint_into_default_policy_restores_gap_mode():
    torch.manual_seed(0)
    source = RLPolicy(5, 1, action_mode="gap_fraction")
    target = RLPolicy(5, 1)  # what test-hedge builds
    target.load_state_dict(source.state_dict())
    assert target.gap_fraction.item() == 1.0
    state = _gap_state(0.5, 0.1)
    assert target.hedge_ratio(state, torch.tensor([-1.0])).item() == pytest.approx(0.1)
    torch.testing.assert_close(target(state).mean, source(state).mean)


def test_load_does_not_mutate_callers_dict_when_gap_fraction_key_missing():
    legacy = {k: v for k, v in RLPolicy(5, 1).state_dict().items() if k != "gap_fraction"}
    keys_before = set(legacy)
    RLPolicy(5, 1).load_state_dict(legacy)
    assert set(legacy) == keys_before


# --- band action mode ---------------------------------------------------------

def _band_state(delta, position):
    return torch.tensor([1.1, 0.4, delta, 0.7, position])


def _band_action(b_frac, lam):
    """Squashed action giving band half-width b_frac * band_max and target lam."""
    return torch.tensor([2 * b_frac - 1, 2 * lam - 1])


@pytest.mark.parametrize("action_dim", [1, 3])
def test_band_mode_requires_two_action_dims(action_dim):
    with pytest.raises(ValueError, match="action_dim"):
        RLPolicy(5, action_dim, action_mode="band")


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_band_mode_rejects_invalid_band_max(bad):
    with pytest.raises(ValueError, match="band_max"):
        RLPolicy(5, 2, action_mode="band", band_max=bad)


def test_invalid_band_max_ignored_outside_band_mode():
    assert RLPolicy(5, 1, band_max=0.0).band_max.item() == 0.0


def test_band_max_is_buffer_zero_outside_band_mode():
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.3)
    assert policy.state_dict()["band_max"].item() == pytest.approx(0.3)
    assert "band_max" not in dict(policy.named_parameters())
    assert RLPolicy(5, 1, band_max=0.3).band_max.item() == 0.0
    assert RLPolicy(5, 1, action_mode="gap_fraction", band_max=0.3).band_max.item() == 0.0


@pytest.mark.parametrize("mode,action_dim", [("residual", 1), ("gap_fraction", 1), ("band", 2)])
def test_action_mode_property_reflects_constructor(mode, action_dim):
    assert RLPolicy(5, action_dim, action_mode=mode).action_mode == mode


@pytest.mark.parametrize("position", [0.29, 0.25, 0.15 + 1e-6, 0.123456789])
def test_band_hold_returns_position_exactly_inside_band(position):
    # delta 0.2, b = 0.5 * 0.2 = 0.1: any position within 0.1 of delta is held.
    # Exactly, not approximately: HedgingEnv charges the fixed cost on any
    # trade above _TRADE_EPS.
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    state = _band_state(0.2, position)
    out = policy.hedge_ratio(state, _band_action(0.5, 0.3))
    assert out.shape == (1,)
    assert torch.equal(out, state[_POSITION_IDX:_POSITION_IDX + 1])


@pytest.mark.parametrize("lam,expected", [
    (0.0, 0.2 + 0.1),   # edge: back to delta + b on the side the position drifted to
    (1.0, 0.2),         # all the way to delta
    (0.5, 0.2 + 0.05),  # halfway into the band
])
def test_band_rebalances_between_edge_and_delta_above(lam, expected):
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    out = policy.hedge_ratio(_band_state(0.2, 0.6), _band_action(0.5, lam))
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_band_rebalances_to_lower_edge_when_position_below_delta():
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    out = policy.hedge_ratio(_band_state(0.2, -0.5), _band_action(0.5, 0.0))
    assert out.item() == pytest.approx(0.1, abs=1e-6)


def test_band_rebalanced_hedge_clamped_to_unit_interval():
    # Out-of-range deltas cannot come from the LSM for a plain hedge, but the
    # hedge must never leave [-1, 1].
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    assert policy.hedge_ratio(_band_state(1.05, 0.5), _band_action(0.1, 1.0)).item() == 1.0


def test_band_batched_uses_each_rows_own_state_and_action():
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    states = torch.stack([_band_state(0.2, 0.25), _band_state(0.2, 0.6), _band_state(-0.3, -0.9)])
    actions = torch.stack([_band_action(0.5, 0.0), _band_action(0.5, 1.0), _band_action(1.0, 0.0)])
    out = policy.hedge_ratio(states, actions)
    assert out.shape == (3, 1)
    torch.testing.assert_close(out, torch.tensor([[0.25], [0.2], [-0.5]]))


@pytest.mark.parametrize("target,lam", [("edge", 0.0), ("delta", 1.0)])
def test_band_constant_action_matches_delta_band_baseline(target, lam):
    """A constant action is the fixed-band baseline: same hedges, holds
    included, on random states."""
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    action = _band_action(0.35, lam)
    b = (policy.band_max * (action[0] + 1) / 2).item()  # the exact width hedge_ratio uses
    baseline = delta_band(b, target)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = torch.tensor([1.0, 0.5, rng.uniform(-1, 1), 0.1, rng.uniform(-1, 1)])
        expected = float(np.clip(baseline(state.double().numpy()), -1, 1))
        out = policy.hedge_ratio(state, action).item()
        assert out == pytest.approx(expected, abs=1e-6)


def test_band_init_bias_gives_initial_width_and_target_with_zero_weights():
    policy = RLPolicy(5, 2, action_mode="band", band_max=0.2)
    assert (_INITIAL_BAND_FRACTION, _INITIAL_BAND_TARGET) == (0.1, 0.9)
    with torch.no_grad():
        policy.layer3.weight.zero_()
    squashed = torch.tanh(policy(torch.zeros(5)).mean)
    torch.testing.assert_close(squashed, torch.tensor([-0.8, 0.8]))


def test_band_init_only_touches_mean_half_of_bias():
    torch.manual_seed(0)
    default = RLPolicy(5, 2)
    torch.manual_seed(0)
    band = RLPolicy(5, 2, action_mode="band")
    torch.testing.assert_close(band.layer3.bias[2:], default.layer3.bias[2:])  # log_std half
    torch.testing.assert_close(band.layer3.weight, default.layer3.weight)


def test_band_checkpoint_round_trip_restores_mode_and_band_max():
    """What test-hedge does: size the network from the checkpoint, build it
    with default settings, load."""
    torch.manual_seed(0)
    source = RLPolicy(5, 2, action_mode="band", band_max=0.3)
    state_dict = source.state_dict()
    assert checkpoint_action_dim(state_dict) == 2
    target = RLPolicy(5, checkpoint_action_dim(state_dict))
    target.load_state_dict(state_dict)
    assert target.action_mode == "band"
    assert target.band_max.item() == pytest.approx(0.3)
    state = _band_state(0.2, 0.25)
    torch.testing.assert_close(target(state.float()).mean, source(state.float()).mean)
    action = _band_action(0.5, 0.0)
    assert torch.equal(target.hedge_ratio(state, action), source.hedge_ratio(state, action))


def test_checkpoint_action_dim_of_one_dimensional_policies():
    assert checkpoint_action_dim(RLPolicy(5, 1).state_dict()) == 1
    assert checkpoint_action_dim(RLPolicy(5, 1, action_mode="gap_fraction").state_dict()) == 1


@pytest.mark.parametrize("mode", ["residual", "gap_fraction"])
def test_load_pre_band_checkpoint_keeps_its_mode_strict(mode):
    legacy = {k: v for k, v in RLPolicy(5, 1, residual_scale=0.2, action_mode=mode).state_dict().items()
              if k != "band_max"}
    target = RLPolicy(5, 1)
    target.load_state_dict(legacy)  # strict; must not raise
    assert target.band_max.item() == 0.0
    assert target.action_mode == mode


def test_action_mode_follows_a_loaded_checkpoint():
    """The mode is cached for speed, so a load that changes it must refresh it."""
    policy = RLPolicy(5, 2)
    assert policy.action_mode == "residual"
    policy.load_state_dict(RLPolicy(5, 2, action_mode="band").state_dict())
    assert policy.action_mode == "band"
    policy.load_state_dict(RLPolicy(5, 2, action_mode="gap_fraction").state_dict())
    assert policy.action_mode == "gap_fraction"
