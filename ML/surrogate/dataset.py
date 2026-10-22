import numpy as np
import torch

from typing import Callable, Optional
from torch.utils.data import Dataset

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
