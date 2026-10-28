import numpy as np
import torch

import os
from dataclasses import dataclass
from typing import Optional, Union
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from optionpricing.pricing.european.black_scholes import black_scholes_delta

# The MLP's inputs, in column order (see build_features).
FEATURES = ("log_m", "log_T", "log_sigma", "r", "total_vol", "z")
# Columns TreePutDataset reads from a `generate_tree_dataset` file.
TREE_COLUMNS = ("m", "T", "r", "sigma", "price", "european_price", "delta", "exercised")

def build_features(m: Union[ArrayLike, torch.Tensor], T: Union[ArrayLike, torch.Tensor],
                   r: Union[ArrayLike, torch.Tensor], sigma: Union[ArrayLike, torch.Tensor]) -> torch.Tensor:
    """
    The MLP's raw (unstandardized) inputs for puts per unit of strike, one
    column per entry of FEATURES: log m, log T, log sigma, r, the total
    volatility sigma sqrt(T) and z = log m / (sigma sqrt(T)), the distance
    to the strike in standard deviations. Arrays and scalars become float64
    tensors; tensors are used as they are, so autograd flows through (the
    delta loss and the price helper differentiate the features in m).
    The inputs are broadcast together, so a curve over m at one (T, r,
    sigma) needs no repeated arrays. Inputs are not validated: m, T or
    sigma <= 0 give NaN or inf.

    Returns:
        (*broadcast shape, len(FEATURES)) tensor, e.g. (n, 6) for (n,) inputs.
    """
    m, T, r, sigma = torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64) if not torch.is_tensor(x)
                                               else x for x in (m, T, r, sigma)))
    log_m = torch.log(m)
    total_vol = sigma * torch.sqrt(T)
    return torch.stack([log_m, torch.log(T), torch.log(sigma), r, total_vol, log_m / total_vol], dim=-1)

# eq=False: the generated __eq__ would compare tensors and raise "Boolean
# value of Tensor is ambiguous"; compare the fields with torch instead
@dataclass(frozen=True, eq=False)
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
        """
        The stats as plain Python lists, with the feature names they were
        fitted for, ready to store in a checkpoint (torch.save or JSON).
        The tensors' dtype is not kept: from_dict rebuilds them as float64.
        """
        return {"names": list(FEATURES), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        """
        Rebuilds stats saved by to_dict, as float64 tensors.

        Raises:
            ValueError: if the stats were saved for other features than
                FEATURES (a model trained before a feature change).
        """
        if list(d["names"]) != list(FEATURES):
            raise ValueError(f"feature statistics are for {list(d['names'])}, expected {list(FEATURES)}")
        return cls(torch.tensor(d["mean"], dtype=torch.float64), torch.tensor(d["std"], dtype=torch.float64))

class TreePutDataset(Dataset):
    """
    The tree-labelled put dataset (`python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`),
    with the MLP's features and targets built at load time from the raw
    columns, so a feature change never needs a regeneration.

    Each item is `(features, features_dm, premium, premium_dm)`, float32,
    each value followed by its derivative in m:
    - features: build_features, standardized with `stats`;
    - premium: the early-exercise premium (American - European) / K, the target;
    - premium_dm: its derivative in m, tree delta - Black-Scholes delta,
      the differential-loss target;
    - features_dm: d features / d m of the *standardized* features (through
      `stats`), so the network's delta in m is its input gradient dotted
      with this row (chain rule through the features).

    The split is a seeded shuffle like the legacy `AmericanOptionDataset`'s. It is not
    grouped by parameter draw: each row is its own draw (puts only), so no
    group of rows can straddle it.

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
        ValueError: if `split_ratio` is outside [0, 1], the file is not a
            tree dataset, or `stats` is None and the training split is empty.
    """
    def __init__(self, data_path: Union[str, os.PathLike], is_train: bool = True, split_ratio: float = 0.8,
                 seed: int = 42, stats: Optional[FeatureStats] = None):
        # slicing would silently accept others: -0.1 acts as 0.9, and above 1 empties the test split
        if not 0 <= split_ratio <= 1:
            raise ValueError(f"split_ratio must be in [0, 1], got {split_ratio}")
        # read everything, then close the file explicitly rather than when
        # garbage collection gets to it
        with np.load(data_path) as data:
            missing = [c for c in ("teacher", *TREE_COLUMNS) if c not in data.files]
            if missing or str(data["teacher"]) != "tree":
                reason = f"missing {missing}" if missing else f"teacher is '{data['teacher']}', expected 'tree'"
                raise ValueError(f"'{data_path}' is not a tree put dataset (generate_tree_dataset): {reason}")
            # the sampling domain, for routing queries outside it
            self.domain = {k: tuple(float(v) for v in data[k]) for k in data.files if k.endswith("_range")}
            cols = {c: data[c] for c in TREE_COLUMNS}

        n_samples = len(cols["m"])
        rdm_idx = np.random.default_rng(seed=seed).permutation(n_samples)
        n_train = int(n_samples * split_ratio)
        idx = rdm_idx[:n_train] if is_train else rdm_idx[n_train:]

        if stats is None:
            train = rdm_idx[:n_train]
            stats = FeatureStats.fit(build_features(*(cols[c][train] for c in ("m", "T", "r", "sigma"))))
        self.stats = stats

        self.params = {c: cols[c][idx] for c in ("m", "T", "r", "sigma")}
        self.exercised = cols["exercised"][idx]
        m = torch.tensor(self.params["m"], dtype=torch.float64, requires_grad=True)
        x = stats.apply(build_features(m, *(self.params[c] for c in ("T", "r", "sigma"))))
        if len(m) == 0:  # nothing to differentiate (autograd.grad would fail)
            x_dm = torch.zeros_like(x)
        else:
            # each row's features depend on its own m only, so the gradient
            # of a column's sum is that column's derivative, row by row
            x_dm = torch.stack([torch.autograd.grad(x[:, j].sum(), m, retain_graph=True)[0]
                                for j in range(len(FEATURES))], dim=-1)

        european_delta = black_scholes_delta(self.params["m"], 1.0, self.params["T"], self.params["r"],
                                             self.params["sigma"], "put")
        self.features = x.detach().float()
        self.features_dm = x_dm.float()
        self.premium = torch.tensor(cols["price"][idx] - cols["european_price"][idx], dtype=torch.float32)
        self.premium_dm = torch.tensor(cols["delta"][idx] - european_delta, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.features[idx], self.features_dm[idx], self.premium[idx], self.premium_dm[idx]

    def __len__(self):
        return len(self.premium)
