"""
Training of the legacy LSM surrogate (lsm_model.LegacyMLP), what `train-ml`
ran before the tree surrogate replaced it: Adam with ReduceLROnPlateau and
early stopping on the validation MSE (fit_legacy_mlp).
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from optionpricing.core.metrics import mse_fn
from optionpricing.ml.surrogate.legacy.lsm_dataset import AmericanOptionDataset
from optionpricing.ml.surrogate.legacy.lsm_model import LegacyMLP


def train_legacy_mlp(
    dataset_path: str | os.PathLike,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    checkpoint_path: str | os.PathLike = "artifacts/checkpoints/best_model.pth",
) -> LegacyMLP:
    """
    Trains a LegacyMLP on the training split of an LSM dataset
    (`python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset`), 20% of it held
    out (seeded) for early stopping. The best validation weights are saved
    to `checkpoint_path` as a plain state_dict (which PutSurrogate.load
    refuses: `test-ml` does not score this net).

    Args:
        dataset_path: the american_option_dataset.npz file.
        split_ratio: share of rows in the training split (AmericanOptionDataset).
        batch_size: rows per Adam step.
        epochs: maximum number of epochs.
        patience: epochs without a validation improvement before stopping.
        seed: seeds the train/test split and the train/validation split.
        checkpoint_path: where the best weights are saved.

    Returns:
        The LegacyMLP as it is after the last epoch (the checkpoint holds the
        best one).

    Raises:
        FileNotFoundError: if `dataset_path` is not a file.
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}; generate it with "
            "`python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset`."
        )
    dataset_train = AmericanOptionDataset(dataset_path, is_train=True, split_ratio=split_ratio, seed=seed)

    mlp = LegacyMLP()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    mlp.to(device)

    n_valid = int(0.2 * len(dataset_train))
    n_train = len(dataset_train) - n_valid

    dataset_train, dataset_valid = random_split(
        dataset_train,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed),  # For reproducability
    )

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print("Training model...")
    start_time = time.time()

    fit_legacy_mlp(
        mlp,
        optimizer,
        scheduler,
        patience,
        dataloader_train,
        dataloader_valid,
        device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
    )

    print(f"Model was trained in : {(time.time() - start_time):.2f}s")
    print(f"Model was saved as : {checkpoint_path}")
    return mlp


def fit_legacy_mlp(
    model: LegacyMLP,
    optimizer,
    scheduler,
    patience,
    dataloader_train,
    dataloader_valid,
    device,
    epochs,
    checkpoint_path="artifacts/checkpoints/best_model.pth",
):
    """Trains `model` for up to `epochs` epochs, saving its state_dict to
    `checkpoint_path` on every validation-MSE improvement and stopping
    after `patience` epochs without one.

    Args:
        model: the LegacyMLP to train, already on `device`.
        optimizer: The optimizer to use.
        scheduler: Scheduler for learning rate.
        patience: The number of epochs with no loss improvment before stoping the training
        dataloader_train: The DataLoader for the training set.
        dataloader_valid: The DataLoader for the validation set.
        device: Either CPU or GPU, the device on which the model is trained.
        epochs: The number of epochs to train for.
        checkpoint_path: Where to save the best-performing model's weights.
    """

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_mse = np.inf
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for ep in epoch_bar:
        # Training
        model.train()
        batch_bar = tqdm(dataloader_train, desc=f"Epoch {ep + 1}/{epochs}", unit="batch", leave=False)
        for batch in batch_bar:
            x, y = batch
            x = x.to(device)
            true = y.unsqueeze(1).to(device)

            pred = model(x)
            loss = nn.MSELoss()(pred, true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            mse_run = 0
            for batch in dataloader_valid:
                x, y = batch
                x = x.to(device)
                true = y.unsqueeze(1).to(device)

                pred = model(x)

                mse_run += mse_fn(pred.cpu().numpy(), true.cpu().numpy())

            mse = mse_run / len(dataloader_valid)

        if scheduler:
            scheduler.step(mse)

        if mse < best_mse:
            best_mse = mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            epoch_bar.set_postfix(mse=f"{mse:.4f}", best="saved")
        else:
            epochs_no_improve += 1
            epoch_bar.set_postfix(mse=f"{mse:.4f}", no_improve=f"{epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {ep + 1}")
                break
