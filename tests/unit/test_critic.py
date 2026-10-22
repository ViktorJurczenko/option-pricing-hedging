import pytest
import torch

from ML.rl.critic import Critic


def test_first_layer_takes_state_plus_action_width():
    # A band-mode (2-D action) critic must be wider than a 1-D one, else
    # train-hedge --action_mode band cannot feed the action in.
    assert Critic(5, 2).net[0].in_features == 7


def test_forward_returns_one_q_value_per_row_and_is_batch_independent():
    torch.manual_seed(0)
    critic = Critic(5, 2)
    s, a = torch.randn(7, 5), torch.randn(7, 2)
    q = critic(s, a)
    assert q.shape == (7, 1)
    torch.testing.assert_close(q[3], critic(s[3:4], a[3:4])[0])


def test_forward_depends_on_both_state_and_action():
    torch.manual_seed(0)
    critic = Critic(5, 1)
    s, a = torch.randn(1, 5), torch.randn(1, 1)
    assert not torch.equal(critic(s, a), critic(s + 1, a))
    assert not torch.equal(critic(s, a), critic(s, a + 1))


def test_forward_rejects_mismatched_action_width():
    with pytest.raises(RuntimeError):
        Critic(5, 2)(torch.zeros(1, 5), torch.zeros(1, 1))
