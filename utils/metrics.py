import numpy as np

def mse_fn(pred, true) -> float:
    return np.mean((pred - true) ** 2)

def accuracy_fn(pred, true) -> float:
    pred_binary = (pred > 0.5).astype(int)
    return np.mean(pred_binary == true)

def mae_fn(pred, true) -> float:
    return np.mean(np.abs(pred - true))

def r2_fn(pred, true) -> float:
    return 1 - (np.sum((pred - true) ** 2) / np.sum((true - np.mean(true)) ** 2))
