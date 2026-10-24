import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from tqdm import tqdm

from ML.surrogate.dataset import FEATURES
from utils.metrics import mse_fn

def intrinsic_value(S, K, option_type: str = "put"):
    """
    Immediate-exercise (intrinsic) value of a vanilla option: max(K-S, 0) for a
    put, max(S-K, 0) for a call. This is the exact payoff *at expiry* - the
    textbook hockey-stick diagram - with a hard kink at S == K (its slope jumps
    from -1/+1 to 0 there). Kept for reference/tests; the legacy LSM pipeline
    (LegacyMLP) uses smooth_intrinsic_value below instead - see its
    docstring, and the "MLP residual target: smooth intrinsic, not exact
    intrinsic" note in CLAUDE.md, for why.
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
    tune (see docs/decisions/softplus-reparameterization.md for the history).

    beta_frac is a fixed constant here, not a CLI flag: it must be identical
    between whatever generated the training labels (ML/surrogate/generate_dataset.py)
    and whatever reconstructs price from the model's output
    (evaluation.py::test_option_price_lsm) or the comparison is invalid - see
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

# Hidden-layer activations MLP accepts. The default is smooth: a ReLU net is
# piecewise linear, so its delta is a staircase and its gamma is 0, and the
# price helper reads both off the net by autograd. "relu" is listed for the
# ablation against the old architecture.
ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh, "softplus": nn.Softplus, "relu": nn.ReLU}
# Output layers: "linear" leaves the premium unconstrained, "softplus" forces
# it > 0, as the true premium is (an ablation decides between them).
OUTPUTS = ("linear", "softplus")

class MLP(nn.Module):
    """
    The surrogate net: standardized features (build_features, then
    FeatureStats.apply) -> the early-exercise premium of an American put
    per unit of strike, (American - European) / K, one value per row.

    `depth` hidden layers of `width` units, each followed by `activation`,
    then one linear layer to the output (passed through a softplus if
    `output="softplus"`). No BatchNorm or Dropout: every row's output
    depends on that row only, in training and evaluation alike, which the
    autograd Greeks rely on.

    Args:
        n_inputs: input features per row (len(FEATURES) for the tree dataset).
        depth: number of hidden layers (>= 1).
        width: units per hidden layer (>= 1).
        activation: a key of ACTIVATIONS.
        output: "linear" or "softplus".

    Raises:
        ValueError: for depth/width/n_inputs < 1 or an unknown activation/output.
    """

    def __init__(self, n_inputs: int = len(FEATURES), depth: int = 4, width: int = 128,
                 activation: str = "silu", output: str = "linear"):
        super().__init__()
        if min(n_inputs, depth, width) < 1:
            raise ValueError(f"n_inputs, depth and width must be >= 1, got {n_inputs}, {depth}, {width}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}")
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
        # the constructor's arguments, stored in the checkpoint to rebuild the same net
        self.config = {"n_inputs": n_inputs, "depth": depth, "width": width,
                       "activation": activation, "output": output}

        layers, size = [], n_inputs
        for _ in range(depth):
            layers += [nn.Linear(size, width), ACTIVATIONS[activation]()]
            size = width
        layers.append(nn.Linear(size, 1))
        if output == "softplus":
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n, n_inputs) standardized features -> (n, 1) premium per unit of strike."""
        return self.net(x)

class LegacyMLP(nn.Module):
    """
    The original surrogate net, trained on the LSM dataset by the current
    `train-ml`/`test-ml` (inputs [S0/K, T_norm, r_norm, sigma_norm, is_call],
    target (price - smooth_intrinsic_value) / K). Superseded by `MLP` below;
    kept so that pipeline keeps running until it is switched to the tree
    dataset, and so the planned ablation can still train the old
    architecture. ReLU makes it piecewise linear (staircase delta, zero
    gamma) and layer3/layer4 compose into a single linear map.
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

    def train_model(self, optimizer, scheduler, patience, dataloader_train, dataloader_valid, device, epochs,
                     checkpoint_path="checkpoints/best_model.pth"):
        """ Trains the model for the specified number of epochs on the dataset.

        Args:
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
            self.train()
            batch_bar = tqdm(dataloader_train, desc=f"Epoch {ep + 1}/{epochs}", unit="batch", leave=False)
            for batch in batch_bar:
                x, y = batch
                x = x.to(device)
                true = y.unsqueeze(1).to(device)

                pred = self.forward(x)
                loss = nn.MSELoss()(pred, true)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation
            self.eval()
            with torch.no_grad():
                mse_run = 0
                for it, batch in enumerate(dataloader_valid):
                    x, y = batch
                    x = x.to(device)
                    true = y.unsqueeze(1).to(device)

                    pred = self(x)

                    mse_run += mse_fn(pred.cpu().numpy(), true.cpu().numpy())

                mse = mse_run / len(dataloader_valid)

            if scheduler:
                scheduler.step(mse)

            if mse < best_mse:
                best_mse = mse
                epochs_no_improve = 0
                torch.save(self.state_dict(), checkpoint_path)
                epoch_bar.set_postfix(mse=f"{mse:.4f}", best="saved")
            else:
                epochs_no_improve += 1
                epoch_bar.set_postfix(mse=f"{mse:.4f}", no_improve=f"{epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    tqdm.write(f"Early stopping at epoch {ep + 1}")
                    break
