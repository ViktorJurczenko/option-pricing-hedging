import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from AmericanOptionPricing.ML import MLP, intrinsic_value, option_type_feature, smooth_intrinsic_value


def test_option_type_feature_encodes_call_as_one_and_put_as_zero():
    assert option_type_feature("call") == 1.0
    assert option_type_feature("put") == 0.0

@pytest.mark.parametrize("bad", ["both", "Call", "", None])
def test_option_type_feature_rejects_unknown_type(bad):
    with pytest.raises(ValueError):
        option_type_feature(bad)

def test_mlp_takes_five_inputs_and_returns_one_output_per_row():
    mlp = MLP().eval()
    assert mlp.layer1.weight.shape[1] == 5
    assert mlp(torch.zeros(3, 5)).shape == (3, 1)


# --- smooth_intrinsic_value ---

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_converges_to_exact_payoff_as_beta_frac_goes_to_zero(option_type):
    S = np.array([70.0, 99.0, 100.0, 101.0, 130.0])
    exact = intrinsic_value(S, 100.0, option_type)
    gaps = [np.abs(smooth_intrinsic_value(S, 100.0, option_type, beta_frac=b) - exact).max()
            for b in (0.06, 0.01, 1e-4)]
    # The max gap is at S == K and equals beta*log(2) = beta_frac*K*log(2).
    assert gaps[0] == pytest.approx(0.06 * 100.0 * np.log(2))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-2

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_is_above_exact_payoff_and_close_far_from_the_strike(option_type):
    S = np.linspace(20.0, 180.0, 161)
    smooth = smooth_intrinsic_value(S, 100.0, option_type)
    exact = intrinsic_value(S, 100.0, option_type)
    assert np.all(smooth >= exact)  # softplus(z) >= max(z, 0)
    far = np.abs(S - 100.0) > 60.0  # z >= 10 widths of beta=6: gap <= 6*exp(-10) ~ 3e-4
    assert np.abs(smooth - exact)[far].max() < 1e-3

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_smooth_intrinsic_has_no_overflow_or_nan_at_extreme_moneyness(option_type):
    # exp(z) with z = +-1e5/6 would overflow in a naive softplus; the stable form must not.
    S = np.array([1e-6, 1.0, 1e5, 1e8])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = smooth_intrinsic_value(S, 100.0, option_type)
    assert np.all(np.isfinite(out)) and np.all(out >= 0)
    exact = intrinsic_value(S, 100.0, option_type)
    np.testing.assert_allclose(out, exact, atol=1e-6)

def test_smooth_intrinsic_at_the_money_is_beta_log_two_for_both_types():
    assert smooth_intrinsic_value(100.0, 100.0, "put") == pytest.approx(6.0 * np.log(2))
    assert smooth_intrinsic_value(100.0, 100.0, "call") == pytest.approx(6.0 * np.log(2))


# --- MLP.train_model ---

def _loader(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    return DataLoader(TensorDataset(x, torch.randn(n, generator=g)), batch_size=4)

def _frozen_run(tmp_path, epochs, patience, lr):
    torch.manual_seed(0)
    mlp = MLP()
    ckpt = tmp_path / "sub" / "m.pth"
    opt = torch.optim.SGD(mlp.parameters(), lr=lr)
    mlp.train_model(opt, None, patience, _loader(), _loader(seed=1), torch.device("cpu"), epochs, str(ckpt))
    return mlp, ckpt

def test_train_model_creates_checkpoint_directory_and_saves_first_epoch(tmp_path):
    _, ckpt = _frozen_run(tmp_path, epochs=1, patience=3, lr=1e-3)
    assert ckpt.exists()  # parent "sub/" did not exist beforehand

def test_train_model_early_stops_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    # lr=0 => weights never change => validation MSE is identical every epoch (BatchNorm
    # running stats move, so eval MSE is made constant by patching the metric): epoch 1 saves,
    # epochs 2.. count as no-improvement, so with patience=2 training stops after epoch 3.
    import AmericanOptionPricing.ML as ml
    calls = []
    def const_mse(pred, true):
        calls.append(1)
        return 1.0
    monkeypatch.setattr(ml, "mse_fn", const_mse)
    _frozen_run(tmp_path, epochs=50, patience=2, lr=0.0)
    n_valid_batches = 2
    assert len(calls) == 3 * n_valid_batches

def test_train_model_saves_only_on_validation_improvement(tmp_path, monkeypatch):
    # Validation MSE per epoch: 1.0 (save), 2.0 (worse), 0.5 (save), 0.7 (worse). The checkpoint
    # must hold the epoch-3 weights, not the final epoch's.
    import AmericanOptionPricing.ML as ml
    per_epoch = iter([1.0, 2.0, 0.5, 0.7])
    state = {"cur": None, "n": 0}
    def scripted_mse(pred, true):
        state["n"] += 1
        if state["n"] % 2 == 1:  # first of the two validation batches in an epoch
            state["cur"] = next(per_epoch)
        return state["cur"]
    monkeypatch.setattr(ml, "mse_fn", scripted_mse)

    snapshots = []
    real_save = torch.save
    def spy_save(obj, path):
        snapshots.append({k: v.clone() for k, v in obj.items()})
        real_save(obj, path)
    monkeypatch.setattr(ml.torch, "save", spy_save)

    torch.manual_seed(0)
    mlp = MLP()
    ckpt = tmp_path / "m.pth"
    opt = torch.optim.SGD(mlp.parameters(), lr=1e-2)
    mlp.train_model(opt, None, 10, _loader(), _loader(seed=1), torch.device("cpu"), 4, str(ckpt))

    assert len(snapshots) == 2  # epochs 1 and 3 only
    saved = torch.load(ckpt)
    for k, v in saved.items():
        assert torch.equal(v, snapshots[1][k])
    assert not torch.equal(saved["layer1.weight"], mlp.layer1.weight)  # epoch 4 trained further

def test_train_model_steps_scheduler_with_validation_mse(tmp_path, monkeypatch):
    import AmericanOptionPricing.ML as ml
    monkeypatch.setattr(ml, "mse_fn", lambda p, t: 0.25)
    seen = []
    class Sched:
        def step(self, v):
            seen.append(v)
    mlp = MLP()
    mlp.train_model(torch.optim.SGD(mlp.parameters(), lr=0.0), Sched(), 10, _loader(), _loader(seed=1),
                    torch.device("cpu"), 2, str(tmp_path / "m.pth"))
    assert seen == [0.25, 0.25]
