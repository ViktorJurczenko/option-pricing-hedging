import numpy as np
import pytest

from utils.metrics import mse_fn, mae_fn, r2_fn, accuracy_fn

def test_mse_fn():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 2.0, 5.0])
    assert mse_fn(pred, true) == pytest.approx((0 + 0 + 4) / 3)

def test_mae_fn():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 0.0, 5.0])
    assert mae_fn(pred, true) == pytest.approx((0 + 2 + 2) / 3)

def test_r2_fn_perfect_prediction():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 2.0, 3.0])
    assert r2_fn(pred, true) == pytest.approx(1.0)

def test_r2_fn_mean_prediction_is_zero():
    true = np.array([1.0, 2.0, 3.0])
    pred = np.full_like(true, true.mean())
    assert r2_fn(pred, true) == pytest.approx(0.0)

def test_accuracy_fn():
    pred = np.array([0.9, 0.4, 0.6, 0.1])
    true = np.array([1, 0, 1, 1])
    # thresholded predictions: [1, 0, 1, 0] vs true [1, 0, 1, 1] -> 3/4 correct
    assert accuracy_fn(pred, true) == pytest.approx(0.75)

@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_r2_fn_constant_targets_raises(offset):
    # The total sum of squares is 0, so R² is undefined; this used to return
    # nan (perfect prediction) or -inf (anything else) with only a warning.
    true = np.full(3, 2.0)
    with pytest.raises(ValueError, match="constant targets"):
        r2_fn(true + offset, true)

def test_r2_fn_can_be_negative_for_a_prediction_worse_than_the_mean():
    true = np.array([1.0, 2.0, 3.0])
    pred = np.array([3.0, 2.0, 1.0])  # SSE 8, SST 2 -> 1 - 4
    assert r2_fn(pred, true) == pytest.approx(-3.0)
