"""train() with a validation env: scoring schedule, metrics and
checkpoint selection."""

import numpy as np
import pytest
import torch

import optionpricing.ml.rl.sac as hedge_module
from optionpricing.hedging.evaluation import cvar
from optionpricing.ml.rl.replay_buffer import ReplayBuffer
from optionpricing.ml.rl.sac import train
from tests.unit.hedge_fakes import FakeEnv, make_nets

# --- train() with a validation env ------------------------------------------


class ScriptedValEnv:
    """Stands in for a validation HedgingEnv: rollout_pnl returns scripted P&L
    arrays (one per call) and records the paths it was asked for plus a copy of
    the policy's weights at that moment."""

    def __init__(self, scripted_pnls, policy, n_paths=500):
        self.scripted = list(scripted_pnls)
        self.policy = policy
        self.n_paths = n_paths
        self.calls = []
        self.snapshots = []

    def rollout_pnl(self, hedge_fn, path_idxs):
        self.calls.append(np.array(path_idxs))
        self.snapshots.append({k: v.clone() for k, v in self.policy.state_dict().items()})
        return np.asarray(self.scripted[len(self.calls) - 1], dtype=float)


def _train_with_val(tmp_path, scripted, n_episodes, val_every, val_metric="std", val_paths=5, n_paths=500, update=True):
    """Runs train() on 2-step episodes with real updates (so weights change
    between scorings). Returns (val_env, checkpoint_path)."""
    torch.manual_seed(0)
    policy, c1, c2 = make_nets()
    val_env = ScriptedValEnv(scripted, policy, n_paths=n_paths)
    path = str(tmp_path / "p.pth")
    train(
        policy,
        FakeEnv([0.1, -0.2]),
        ReplayBuffer(max_size=100),
        c1,
        c2,
        n_episodes=n_episodes,
        warmup_steps=0 if update else 10**6,
        update_every=1,
        batch_size=1,
        val_env=val_env,
        val_paths=val_paths,
        val_every=val_every,
        val_metric=val_metric,
        checkpoint_path=path,
    )
    return val_env, path


def _assert_ckpt_equals(path, snapshot):
    saved = torch.load(path)
    assert saved.keys() == snapshot.keys()
    for k in saved:
        assert torch.equal(saved[k], snapshot[k]), k


@pytest.mark.parametrize("val_metric", ["mean", ""])
def test_train_rejects_unknown_val_metric(val_metric):
    with pytest.raises(ValueError, match="val_metric"):
        train(None, None, None, None, None, val_metric=val_metric)


@pytest.mark.parametrize("kw", [{"val_paths": 0}, {"val_every": 0}, {"val_paths": -1}])
def test_train_rejects_non_positive_val_paths_or_every_when_val_env_given(kw):
    with pytest.raises(ValueError, match="val_paths and val_every"):
        train(None, None, None, None, None, val_env=object(), **kw)


def test_train_scores_and_saves_untrained_policy_before_any_training(tmp_path):
    # Untrained policy must already be the checkpoint: one scoring, zero episodes run.
    val_env, path = _train_with_val(tmp_path, [[1.0, -1.0]], n_episodes=0, val_every=10)
    assert len(val_env.calls) == 1
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_rescoring_schedule_every_val_every_and_after_last_episode(tmp_path):
    # n_episodes=7, val_every=3 -> initial, ep 3, ep 6, ep 7 (not a multiple)
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 4, n_episodes=7, val_every=3)
    assert len(val_env.calls) == 4


def test_train_rescoring_when_n_episodes_is_a_multiple_does_not_double_score(tmp_path):
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 3, n_episodes=4, val_every=2)
    assert len(val_env.calls) == 3  # initial, ep 2, ep 4


def test_train_std_metric_saves_only_on_strict_improvement(tmp_path):
    # std: 2.0 (initial), 3.0 worse, 1.0 better (saved), 1.0 tie (not saved)
    pnls = [[2.0, -2.0], [3.0, -3.0], [1.0, -1.0], [-1.0, 1.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=3, val_every=1, val_metric="std")
    assert len(val_env.calls) == 4
    _assert_ckpt_equals(path, val_env.snapshots[2])
    # the tie really is a different policy, so "not saved" is observable
    assert not all(torch.equal(val_env.snapshots[2][k], val_env.snapshots[3][k]) for k in val_env.snapshots[2])


def test_train_std_metric_keeps_initial_when_nothing_improves(tmp_path):
    pnls = [[1.0, -1.0], [5.0, -5.0], [2.0, -2.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=2, val_every=1, val_metric="std")
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_cvar_metric_higher_is_better(tmp_path):
    # Same shape shifted: initial, worse (-0.5), better (+0.5) -> the last scoring is saved.
    base = np.linspace(-1, 1, 101)
    pnls = [base, base - 0.5, base + 0.5]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=2, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[2])


def test_train_cvar_metric_ranks_by_tail_mean_not_fifth_percentile(tmp_path):
    # Regression test: "cvar" used to score np.percentile(pnls, 5) (the VaR).
    # `a` has a better 5th percentile (0.0) but a catastrophic tail (mean of worst 5 = -6);
    # `b` has a worse percentile (-1.0) but a mild tail (-1.0). CVaR must prefer `b`.
    a = np.concatenate([np.full(3, -10.0), np.zeros(97)])
    b = np.concatenate([np.full(6, -1.0), np.zeros(94)])
    assert np.percentile(a, 5) > np.percentile(b, 5)
    assert cvar(a) < cvar(b)
    val_env, path = _train_with_val(tmp_path, [a, b], n_episodes=1, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[1])


def test_train_cvar_metric_not_fooled_by_lower_std(tmp_path):
    # Second P&L has lower std but a worse left tail (CVaR -0.1 vs 0.0): it must NOT be saved.
    first = np.concatenate([np.full(95, 0.0), np.full(5, 1.0)])  # CVaR 0.0
    second = np.concatenate([np.full(10, -0.1), np.full(90, 0.0)])  # CVaR -0.1, smaller std
    assert np.std(second) < np.std(first)
    val_env, path = _train_with_val(tmp_path, [first, second], n_episodes=1, val_every=1, val_metric="cvar")
    _assert_ckpt_equals(path, val_env.snapshots[0])


def test_train_validates_on_first_min_val_paths_and_n_paths_in_order(tmp_path):
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 2, n_episodes=1, val_every=1, val_paths=5, n_paths=500)
    np.testing.assert_array_equal(val_env.calls[0], np.arange(5))
    val_env, _ = _train_with_val(tmp_path, [[1.0, -1.0]] * 2, n_episodes=1, val_every=1, val_paths=50, n_paths=3)
    np.testing.assert_array_equal(val_env.calls[0], np.arange(3))


def test_train_validation_hedge_fn_is_deterministic_hedge_of_the_policy(tmp_path):
    torch.manual_seed(0)
    policy, c1, c2 = make_nets()
    states = np.random.default_rng(0).uniform(-1, 1, (3, 5))
    seen = []

    class Env(ScriptedValEnv):
        def rollout_pnl(self, hedge_fn, path_idxs):
            seen.append((hedge_fn(states), hedge_module.deterministic_hedge(policy, states)))
            return super().rollout_pnl(hedge_fn, path_idxs)

    train(
        policy,
        FakeEnv([0.1]),
        ReplayBuffer(max_size=10),
        c1,
        c2,
        n_episodes=0,
        val_env=Env([[1.0, -1.0]], policy),
        val_paths=2,
        val_every=1,
        checkpoint_path=str(tmp_path / "p.pth"),
    )
    np.testing.assert_array_equal(seen[0][0], seen[0][1])


def test_train_without_val_env_keeps_trailing_50_mean_reward_criterion(tmp_path):
    # No checkpoint may exist before 50 episodes have been recorded; afterwards one is saved.
    policy, c1, c2 = make_nets()
    path = tmp_path / "p.pth"
    train(
        policy,
        FakeEnv([0.1]),
        ReplayBuffer(max_size=100),
        c1,
        c2,
        n_episodes=49,
        warmup_steps=10**6,
        batch_size=10**6,
        checkpoint_path=str(path),
    )
    assert not path.exists()
    train(
        policy,
        FakeEnv([0.1]),
        ReplayBuffer(max_size=100),
        c1,
        c2,
        n_episodes=50,
        warmup_steps=10**6,
        batch_size=10**6,
        checkpoint_path=str(path),
    )
    assert path.exists()


def test_train_with_val_env_does_not_use_trailing_mean_criterion(tmp_path):
    # 60 episodes with val_every=100 -> only the initial and the final scoring can save;
    # a worse final score must leave the initial (untrained) weights on disk.
    pnls = [[1.0, -1.0], [9.0, -9.0]]
    val_env, path = _train_with_val(tmp_path, pnls, n_episodes=60, val_every=100, update=False)
    _assert_ckpt_equals(path, val_env.snapshots[0])
