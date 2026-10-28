"""
The legacy LSM surrogate: `LegacyMLP`, the original net, trained on the
Longstaff-Schwartz dataset (`generate_lsm_dataset.py`) to predict the
time value (price - smooth_intrinsic_value) / K of calls and puts. The
tree-trained `optionpricing.ml.surrogate.tree.tree_model.MLP` replaced it; it is kept as a
library for the ablation against the old architecture and for rerunning
the LSM comparison. No CLI command uses it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def intrinsic_value(S, K, option_type: str = "put"):
    """
    Immediate-exercise (intrinsic) value of a vanilla option: max(K-S, 0) for a
    put, max(S-K, 0) for a call. This is the exact payoff *at expiry* - the
    textbook hockey-stick diagram - with a hard kink at S == K (its slope jumps
    from -1/+1 to 0 there). Kept for reference/tests; the legacy LSM pipeline
    (LegacyMLP) uses smooth_intrinsic_value below instead - see its
    docstring, and the "Softplus reparameterization" section of
    CLAUDE.md, for why.
    """
    if option_type == "put":
        return np.maximum(K - S, 0.0)
    return np.maximum(S - K, 0.0)

def smooth_intrinsic_value(S, K, option_type: str = "put", beta_frac: float = 0.06):
    """
    Smooth (everywhere-differentiable) stand-in for intrinsic_value, via a
    softplus: beta * log(1 + exp(x/beta)), where x = K-S (put) or S-K (call)
    and beta = beta_frac * K. As beta_frac -> 0 this converges to
    intrinsic_value exactly; for beta_frac > 0 it matches intrinsic_value
    closely away from S == K but rounds off the corner over a window about
    `beta_frac * K` wide, instead of having a hard kink there.

    Why this matters: the option's *price* before expiry is smooth in S,
    including at S == K (only the expiry *payoff* has the kink - see
    intrinsic_value's docstring). The MLP is trained on (price - intrinsic)/K
    rather than raw price/K, to shrink the target's dynamic range (deep ITM
    prices are much larger than deep OTM ones). But subtracting the *exact*
    intrinsic_value stamps its hard kink onto that residual at S == K even
    though the real price has no kink there - which showed up empirically as
    the MLP's error spiking specifically around at-the-money (see CLAUDE.md).
    Subtracting this smooth version instead keeps the residual smooth
    everywhere, so training and inference (add this same function back to
    recover price) never introduce that artifact.

    beta_frac=0.06 (i.e. the smoothing window is ~12% of K wide, +/-3*beta_frac)
    was picked by sweeping {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08} against a
    fixed LSM benchmark: smaller values under-smooth (ATM error stays high,
    close to the hard-kink case), larger values over-smooth (the correction
    bleeds into genuinely OTM territory and degrades that bucket instead).
    0.06 was the best 3-way balance found, not a derived/theoretical constant.

    Used by the legacy LSM pipeline (LegacyMLP) only: the tree-trained MLP
    subtracts the Black-Scholes European price instead, which bends exactly
    like the American price at every sigma sqrt(T), so there is no width to
    tune (see .claude/BUGS.md, "ML surrogate pricer's target had a
    self-inflicted kink at ATM", for the history).

    beta_frac is a fixed constant here, not a CLI flag: it must be identical
    between whatever generated the training labels (generate_lsm_dataset.py)
    and whatever reconstructs price from the model's output
    (lsm_evaluation.py::test_option_price_lsm) or the comparison is invalid - see
    CLAUDE.md's note on the same class of bug with reg_type/max_degree drift.
    """
    beta = beta_frac * K
    x = (K - S) if option_type == "put" else (S - K)
    z = x / beta
    softplus_z = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))  # numerically stable softplus(z)
    return beta * softplus_z

def option_type_feature(option_type: str) -> float:
    """The MLP's 5th input, `is_call`: 1.0 for a call, 0.0 for a put."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    return 1.0 if option_type == "call" else 0.0

class LegacyMLP(nn.Module):
    """
    The original surrogate net, trained on the LSM dataset by
    lsm_training.train_legacy_mlp (inputs [S0/K, T_norm, r_norm,
    sigma_norm, is_call], target (price - smooth_intrinsic_value) / K).
    Superseded by optionpricing.ml.surrogate.tree.tree_model.MLP; kept so the ablation can still
    train the old architecture. ReLU makes it piecewise linear (staircase
    delta, zero gamma) and layer3/layer4 compose into a single linear map.
    Trained by lsm_training.fit_legacy_mlp.
    """

    def __init__(self):
        super(LegacyMLP, self).__init__()

        self.layer1 = nn.Linear(5, 64)
        self.bn1    = nn.BatchNorm1d(64)
        self.drop1  = nn.Dropout(0.1)

        self.layer2 = nn.Linear(64, 64)
        self.bn2    = nn.BatchNorm1d(64)
        self.drop2  = nn.Dropout(0.1)

        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.layer1(x))))
        x = self.drop2(F.relu(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
