import os
import pytest

from optionpricing.ml.surrogate.legacy.lsm_dataset import AmericanOptionDataset

DATASET_PATH = "artifacts/data/american_option_dataset.npz"

@pytest.mark.skipif(not os.path.exists(DATASET_PATH), reason="bundled dataset not present")
def test_bundled_dataset_loads_and_has_expected_feature_width():
    dataset = AmericanOptionDataset(DATASET_PATH, is_train=True, split_ratio=0.8, seed=42)

    assert len(dataset) > 0
    features, label = dataset[0]
    assert features.shape == (5,)  # [S0/K, T_norm, r_norm, sigma_norm, is_call]
    assert label.shape == ()
