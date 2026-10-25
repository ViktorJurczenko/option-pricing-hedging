"""fit_legacy_mlp, the legacy LSM surrogate's training loop (train_legacy_mlp runs end to end in
tests/integration/test_lsm_pipeline.py)."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from ML.surrogate.legacy.lsm_model import LegacyMLP
from ML.surrogate.legacy.lsm_training import fit_legacy_mlp

def _loader(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    return DataLoader(TensorDataset(x, torch.randn(n, generator=g)), batch_size=4)

def _frozen_run(tmp_path, epochs, patience, lr):
    torch.manual_seed(0)
    mlp = LegacyMLP()
    ckpt = tmp_path / "sub" / "m.pth"
    opt = torch.optim.SGD(mlp.parameters(), lr=lr)
    fit_legacy_mlp(mlp, opt, None, patience, _loader(), _loader(seed=1), torch.device("cpu"), epochs, str(ckpt))
    return mlp, ckpt

def test_fit_legacy_mlp_creates_checkpoint_directory_and_saves_first_epoch(tmp_path):
    _, ckpt = _frozen_run(tmp_path, epochs=1, patience=3, lr=1e-3)
    assert ckpt.exists()  # parent "sub/" did not exist beforehand

def test_fit_legacy_mlp_early_stops_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    # lr=0 => weights never change => validation MSE is identical every epoch (BatchNorm
    # running stats move, so eval MSE is made constant by patching the metric): epoch 1 saves,
    # epochs 2.. count as no-improvement, so with patience=2 training stops after epoch 3.
    import ML.surrogate.legacy.lsm_training as ml
    calls = []
    def const_mse(pred, true):
        calls.append(1)
        return 1.0
    monkeypatch.setattr(ml, "mse_fn", const_mse)
    _frozen_run(tmp_path, epochs=50, patience=2, lr=0.0)
    n_valid_batches = 2
    assert len(calls) == 3 * n_valid_batches

def test_fit_legacy_mlp_saves_only_on_validation_improvement(tmp_path, monkeypatch):
    # Validation MSE per epoch: 1.0 (save), 2.0 (worse), 0.5 (save), 0.7 (worse). The checkpoint
    # must hold the epoch-3 weights, not the final epoch's.
    import ML.surrogate.legacy.lsm_training as ml
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
    mlp = LegacyMLP()
    ckpt = tmp_path / "m.pth"
    opt = torch.optim.SGD(mlp.parameters(), lr=1e-2)
    fit_legacy_mlp(mlp, opt, None, 10, _loader(), _loader(seed=1), torch.device("cpu"), 4, str(ckpt))

    assert len(snapshots) == 2  # epochs 1 and 3 only
    saved = torch.load(ckpt)
    for k, v in saved.items():
        assert torch.equal(v, snapshots[1][k])
    assert not torch.equal(saved["layer1.weight"], mlp.layer1.weight)  # epoch 4 trained further

def test_fit_legacy_mlp_steps_scheduler_with_validation_mse(tmp_path, monkeypatch):
    import ML.surrogate.legacy.lsm_training as ml
    monkeypatch.setattr(ml, "mse_fn", lambda p, t: 0.25)
    seen = []
    class Sched:
        def step(self, v):
            seen.append(v)
    mlp = LegacyMLP()
    fit_legacy_mlp(mlp, torch.optim.SGD(mlp.parameters(), lr=0.0), Sched(), 10, _loader(), _loader(seed=1),
                   torch.device("cpu"), 2, str(tmp_path / "m.pth"))
    assert seen == [0.25, 0.25]
