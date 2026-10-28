"""
Evaluation of the legacy LSM surrogate (lsm_model.LegacyMLP): regression
metrics on a test split and the price against a fresh LSM run. No CLI
command uses it: `test-ml` evaluates the tree surrogate
(optionpricing/ml/surrogate/tree/tree_evaluation.py).
"""

import numpy as np
import torch
import torch.nn as nn

from optionpricing.core.metrics import mae_fn, mse_fn, r2_fn
from optionpricing.ml.surrogate.legacy.lsm_model import option_type_feature, smooth_intrinsic_value
from optionpricing.pricing.american.longstaff_schwartz import LongstaffSchwartz


def regression_tests(model: torch.nn.Module, dataloader_test):
    model.eval()  # Set the model on evaluation mode
    all_pred, all_true = [], []

    with torch.no_grad():  # More performence on tests
        for x, y in dataloader_test:
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

        pred = np.concatenate(all_pred).flatten()
        true = np.concatenate(all_true).flatten()

        mse = mse_fn(pred, true)
        mae = mae_fn(pred, true)
        r2 = r2_fn(pred, true)

        print(f"MSE:  {mse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.4f}")


def test_option_price_lsm(
    model: nn.Module,
    simulation,
    moneyness,
    K,
    r,
    r_norm,
    T,
    T_norm,
    sigma_norm,
    option_type="put",
    reg_type="simple",
    max_degree=3,
    sigma=None,
    control_variate=False,
) -> tuple:
    """
    Args:
        sigma, control_variate: passed to the LSM benchmark. Calls should use
            control_variate=True (which needs sigma), matching how
            generate_lsm_dataset.py labels them.


    Returns:
        Tuple of:
        - error: relative error in % (falls back to absolute error in price units
          when the LSM benchmark price is too close to 0 for a percentage to be meaningful).
        - abs_error: absolute error in price units, regardless of price magnitude.
          Useful for buckets (e.g. deep OTM) where the price is small enough that
          a tiny absolute miss inflates the relative error.

    Note: reg_type/max_degree should match whatever was used to generate the
    training dataset's labels - otherwise this benchmarks the MLP against a
    different LSM "teacher" than the one it was actually trained on, conflating
    MLP error with simple-vs-Laguerre basis differences.
    """
    # LSM benchmark
    lsm = LongstaffSchwartz(
        simulation, K, r, T, option_type, reg_type, max_degree, sigma=sigma, control_variate=control_variate
    )
    lsm.predict_payoff()
    option_price_lsm = lsm.calculate_option_price()

    # ML pricing: the model predicts time value (price - smooth_intrinsic) / K
    # (see smooth_intrinsic_value's docstring for why smooth, not exact,
    # intrinsic value is used here), so the same smooth intrinsic value is
    # added back analytically to recover the actual price.
    model.eval()
    with torch.no_grad():
        features = [moneyness, T_norm, r_norm, sigma_norm, option_type_feature(option_type)]
        feat = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        raw_output = model(feat).cpu().numpy()[0, 0]

    S0 = moneyness * K
    option_price_ml = smooth_intrinsic_value(S0, K, option_type) + raw_output * K

    abs_error = np.abs(option_price_ml - option_price_lsm)

    if option_price_lsm < 1e-1:  # Prevent division by 0
        error = abs_error
    else:
        error = abs_error / option_price_lsm * 100

    return error, abs_error
