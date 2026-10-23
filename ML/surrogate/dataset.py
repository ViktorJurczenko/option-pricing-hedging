import numpy as np
import torch

from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import Dataset

from EuropeanOptionPricing.BlackScholes import black_scholes_delta

class AmericanOptionDataset(Dataset):
    """
    Custom Dataset for American option pricing data.

    Args:
        data_path (str): Path to the .npz file containing the dataset.
        transform (callable, optional): Optional transform to be applied to features.
        label_transform (callable, optional): Optional transform to be applied to labels.
        is_train (bool): Whether to load the training or test split.
        split_ratio (float): Ratio of data to use for training (default: 0.8).
    """
    def __init__(self,
                 data_path,
                 transform:Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 label_transform:Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 is_train=True,
                 split_ratio=0.8,
                 seed=42
        ):

        data = np.load(data_path, mmap_mode='r') # mmap avoids loading the whole archive eagerly
        features = data["features"]
        label    = data["label"]

        # Shuffle with fixed seed before splitting
        n_samples = len(features)
        rng       = np.random.default_rng(seed=seed)
        rdm_idx   = rng.permutation(n_samples)
        n_train   = int(n_samples * split_ratio)

        idx = rdm_idx[:n_train] if is_train else rdm_idx[n_train:]

        self.features        = features[idx]
        self.label           = label[idx]
        self.transform       = transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label    = torch.tensor(self.label[idx], dtype=torch.float32)

        if self.transform:
            features = self.transform(features)
        if self.label_transform:
            label = self.label_transform(label)

        return features, label

    def __len__(self):
            return len(self.features)


# The MLP's inputs, in column order (see build_features).
FEATURES = ("log_m", "log_T", "log_sigma", "r", "total_vol", "z")
# Columns TreePutDataset reads from a `GenerateTreeDataset` file.
TREE_COLUMNS = ("m", "T", "r", "sigma", "price", "european_price", "delta", "exercised")

def build_features(m, T, r, sigma) -> torch.Tensor:
    """
    The MLP's raw (unstandardized) inputs for puts per unit of strike, one
    column per entry of FEATURES: log m, log T, log sigma, r, the total
    volatility sigma sqrt(T) and z = log m / (sigma sqrt(T)), the distance
    to the strike in standard deviations. Arrays become float64 tensors;
    tensors are used as they are, so autograd flows through (the delta
    loss and the price helper differentiate the features in m).

    Returns:
        (n, len(FEATURES)) tensor.
    """
    m, T, r, sigma = (torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
                      for x in (m, T, r, sigma))
    log_m = torch.log(m)
    total_vol = sigma * torch.sqrt(T)
    return torch.stack([log_m, torch.log(T), torch.log(sigma), r, total_vol, log_m / total_vol], dim=-1)

@dataclass(frozen=True)
class FeatureStats:
    """
    Per-feature mean and std that standardize build_features' output.
    Fitted on the training split only, and stored with the model (to_dict)
    so every consumer applies the same transformation.
    """
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, raw: torch.Tensor) -> "FeatureStats":
        """
        Computes the mean and std of each feature (column) over the rows of
        `raw`, so that `apply` maps these rows to mean 0 and std 1 per
        column. Call it on the training rows only: stats fitted on test rows
        too would leak information about them into the model's inputs.

        Args:
            raw: (n, len(FEATURES)) unstandardized features, as returned by
                build_features.

        Returns:
            FeatureStats with (len(FEATURES),) `mean` and `std`; the std is
            the population one (divides by n), and a constant column's std
            is set to 1.

        Raises:
            ValueError: if `raw` has no rows.
        """
        if len(raw) == 0:
            raise ValueError("cannot fit feature statistics on an empty training split")
        mean, std = raw.mean(dim=0), raw.std(dim=0, unbiased=False)
        # a constant feature (one training row, or a domain with a fixed
        # parameter) is only centered: dividing by 0 would give NaN
        return cls(mean, torch.where(std > 0, std, torch.ones_like(std)))

    def apply(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Standardizes features with the fitted stats: (raw - mean) / std per
        column. Use it on every row fed to the network (training, test,
        later queries), always with the training split's stats, so the
        network sees the same transformation it was trained on. It is
        differentiable, so autograd flows through it.

        Args:
            raw: (n, len(FEATURES)) or (len(FEATURES),) unstandardized
                features; the stats are cast to its dtype and device.

        Returns:
            Tensor of the same shape as `raw`.
        """
        return (raw - self.mean.to(raw)) / self.std.to(raw)

    def to_dict(self) -> dict:
        return {"names": list(FEATURES), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        """
        Raises:
            ValueError: if the stats were saved for other features than
                FEATURES (a model trained before a feature change).
        """
        if list(d["names"]) != list(FEATURES):
            raise ValueError(f"feature statistics are for {list(d['names'])}, expected {list(FEATURES)}")
        return cls(torch.tensor(d["mean"], dtype=torch.float64), torch.tensor(d["std"], dtype=torch.float64))

class TreePutDataset(Dataset):
    """
    The tree-labelled put dataset (`python -m ML.surrogate.GenerateTreeDataset`),
    with the MLP's features and targets built at load time from the raw
    columns, so a feature change never needs a regeneration.

    Each item is `(features, premium, premium_delta, features_dm)`, float32:
    - features: build_features, standardized with `stats`;
    - premium: the early-exercise premium (American - European) / K, the target;
    - premium_delta: its derivative in m, tree delta - Black-Scholes delta,
      the differential-loss target;
    - features_dm: d features / d m, so the network's delta in m is its
      input gradient dotted with this row (chain rule through the features).

    The split is a seeded shuffle like `AmericanOptionDataset`'s. Each row is
    its own parameter draw (puts only), so no group of rows can straddle it.

    Args:
        data_path: the `tree_put_dataset.npz` file.
        is_train: load the training split, else the test split.
        split_ratio: share of rows in the training split.
        seed: seed of the split's shuffle.
        stats: feature statistics to standardize with; by default fitted on
            this file's training split, for either split, so the test split
            never leaks into them. Pass the model's saved stats to reproduce
            its inputs.

    Raises:
        ValueError: if the file is not a tree dataset, or `stats` is None
            and the training split is empty.
    """
    def __init__(self, data_path, is_train: bool = True, split_ratio: float = 0.8, seed: int = 42,
                 stats: Optional[FeatureStats] = None):
        data = np.load(data_path)
        missing = [c for c in ("teacher", *TREE_COLUMNS) if c not in data.files]
        if missing or str(data["teacher"]) != "tree":
            reason = f"missing {missing}" if missing else f"teacher is '{data['teacher']}', expected 'tree'"
            raise ValueError(f"'{data_path}' is not a tree put dataset (GenerateTreeDataset): {reason}")
        # the sampling domain, for routing queries outside it
        self.domain = {k: tuple(float(v) for v in data[k]) for k in data.files if k.endswith("_range")}

        n_samples = len(data["m"])
        rdm_idx = np.random.default_rng(seed=seed).permutation(n_samples)
        n_train = int(n_samples * split_ratio)
        idx = rdm_idx[:n_train] if is_train else rdm_idx[n_train:]

        cols = {c: data[c] for c in TREE_COLUMNS}
        if stats is None:
            train = rdm_idx[:n_train]
            stats = FeatureStats.fit(build_features(*(cols[c][train] for c in ("m", "T", "r", "sigma"))))
        self.stats = stats

        self.params = {c: cols[c][idx] for c in ("m", "T", "r", "sigma")}
        self.exercised = cols["exercised"][idx]
        m = torch.tensor(self.params["m"], dtype=torch.float64, requires_grad=True)
        x = stats.apply(build_features(m, *(self.params[c] for c in ("T", "r", "sigma"))))
        # each row's features depend on its own m only, so the gradient of a
        # column's sum is that column's derivative, row by row
        x_dm = torch.stack([torch.autograd.grad(x[:, j].sum(), m, retain_graph=True)[0]
                            for j in range(len(FEATURES))], dim=-1) if len(m) else torch.zeros_like(x)

        european_delta = black_scholes_delta(self.params["m"], 1.0, self.params["T"], self.params["r"],
                                             self.params["sigma"], "put")
        self.features = x.detach().float()
        self.features_dm = x_dm.float()
        self.premium = torch.tensor(cols["price"][idx] - cols["european_price"][idx], dtype=torch.float32)
        self.premium_delta = torch.tensor(cols["delta"][idx] - european_delta, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.features[idx], self.premium[idx], self.premium_delta[idx], self.features_dm[idx]

    def __len__(self):
        return len(self.premium)
