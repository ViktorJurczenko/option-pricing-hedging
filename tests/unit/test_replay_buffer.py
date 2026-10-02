import numpy as np
import pytest
import torch

from AmericanHedging.ReplayBuffer import ReplayBuffer


def _push(buffer, i, action=None):
    """Transition i with recognisable values: state i, reward 10 + i, done on odd i."""
    buffer.push(np.full(5, float(i)), float(i) if action is None else action,
                10.0 + i, np.full(5, i + 0.5), bool(i % 2))


def test_len_counts_pushed_transitions():
    buffer = ReplayBuffer(max_size=10)
    assert len(buffer) == 0
    for i in range(3):
        _push(buffer, i)
    assert len(buffer) == 3


def test_oldest_transitions_are_evicted_at_max_size():
    buffer = ReplayBuffer(max_size=3)
    for i in range(5):
        _push(buffer, i)
    assert len(buffer) == 3
    states, *_ = buffer.sample(3)  # all that is left
    assert sorted(states[:, 0].tolist()) == [2.0, 3.0, 4.0]  # 0 and 1 were dropped


def test_sample_returns_aligned_float32_tensors_with_column_rewards_and_dones():
    buffer = ReplayBuffer(max_size=10)
    for i in range(6):
        _push(buffer, i)
    s, a, r, s2, d = buffer.sample(4)

    assert [t.dtype for t in (s, a, r, s2, d)] == [torch.float32] * 5
    assert s.shape == s2.shape == (4, 5)
    assert a.shape == r.shape == d.shape == (4, 1)
    # Each row must come from ONE transition (zip(*batch) alignment).
    i = s[:, 0]
    torch.testing.assert_close(a[:, 0], i)
    torch.testing.assert_close(r[:, 0], 10.0 + i)
    torch.testing.assert_close(s2[:, 0], i + 0.5)
    torch.testing.assert_close(d[:, 0], (i % 2))  # bool done -> 0.0 / 1.0


def test_sample_without_replacement_gives_distinct_transitions():
    buffer = ReplayBuffer(max_size=10)
    for i in range(8):
        _push(buffer, i)
    s, *_ = buffer.sample(8)
    assert sorted(s[:, 0].tolist()) == list(range(8))


@pytest.mark.parametrize("action_dim", [1, 2, 3])
def test_vector_actions_are_sampled_as_batch_by_action_dim(action_dim):
    buffer = ReplayBuffer(max_size=10)
    for i in range(4):
        _push(buffer, i, action=np.arange(action_dim) + 0.1 * i)
    _, a, *_ = buffer.sample(3)
    assert a.shape == (3, action_dim)
    # columns stay in order and aligned within a row
    torch.testing.assert_close(a - a[:, :1], torch.arange(action_dim, dtype=torch.float32).expand(3, -1))


def test_scalar_actions_become_a_single_column():
    buffer = ReplayBuffer(max_size=10)
    for i in range(4):
        _push(buffer, i)
    assert buffer.sample(2)[1].shape == (2, 1)


def test_sampling_more_than_stored_raises_value_error():
    # Current behaviour (random.sample): train() guards with len(buffer) >= batch_size.
    buffer = ReplayBuffer(max_size=10)
    _push(buffer, 0)
    with pytest.raises(ValueError):
        buffer.sample(2)
