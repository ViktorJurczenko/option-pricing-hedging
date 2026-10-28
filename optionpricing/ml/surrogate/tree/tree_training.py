import math
import os
from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from optionpricing.pricing.european.black_scholes import (black_scholes_delta, black_scholes_dT, black_scholes_gamma,
                                                black_scholes_vega)
from optionpricing.ml.surrogate.tree.tree_dataset import FEATURES, FeatureStats, TreePutDataset, build_features
from optionpricing.ml.surrogate.tree.generate_tree_dataset import SAMPLED, map_to_domain
from optionpricing.ml.surrogate.tree.tree_model import MLP
from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate

# The no-arbitrage conditions of shape_violations, in its column order.
SHAPE_CONDITIONS = ("decreasing_in_m", "convex_in_m", "increasing_in_T", "increasing_in_sigma")
# Rows per forward pass when a whole split is evaluated (validation, L-BFGS),
# to bound the memory of the double backward of the delta loss. 8 batches'
# worth: one 4 x 128 float32 activation layer is 4 MB at this size, so the
# graph stays in the tens of MB, and the per-chunk overhead is negligible
# against 300k+ rows. A memory bound, not a tuned value: results don't
# depend on it (losses are summed exactly over chunks).
EVAL_CHUNK = 8192
# Early stopping counts an epoch without improvement only once the rate is
# within this fraction of the way from lr_min to lr, i.e. in the last ~20%
# of the cosine, whatever `epochs` is. At a higher rate the validation loss
# jitters from epoch to epoch, and a fixed patience cut 400-epoch runs
# short before their low-rate tail (BUGS.md).
PATIENCE_LR_FRACTION = 0.1

@dataclass(frozen=True)
class TrainConfig:
    """
    Everything a training run depends on: a run is reproduced from its
    config and the dataset.

    Network: depth, width, activation, output (MLP's arguments).
    Data: split_ratio (TreePutDataset's train/test split, the test rows are
        never touched here), valid_ratio (share of the training split held
        out for early stopping and the checkpoint choice).
    Optimization: batch_size; epochs (also the cosine schedule's length);
        lr, decayed to lr_min by a per-step cosine schedule; patience
        (epochs without a validation improvement before stopping, counted
        only once the rate is near lr_min, see PATIENCE_LR_FRACTION: a
        safety net, the schedule is what ends a normal run); lbfgs_steps (L-BFGS
        iterations after Adam, full batch, 0 = none).
    Loss: lambda_delta (weight of the delta term), lambda_shape (of the
        shape penalties), shape_points (fresh random points per step for
        those penalties); 0 disables a term. lambda_shape defaults to 0: on
        a first small run it tripled the price error, about equally in
        every region, for a small drop in violations, and made each step
        ~3x slower than with the delta loss alone (.claude/BUGS.md, "Shape
        penalty's convexity term ~3000x too big"); the ablation on the full
        dataset decides.
    seed: seeds the split, the weights' initialization, the batch order and
        the shape points.

    Raises:
        ValueError: for a non-positive size/count/learning rate, lr_min
            outside [0, lr], valid_ratio outside (0, 1), negative lambdas
            or lbfgs_steps.
    """
    # ~50k weights: enough for a smooth function of 6 inputs, microseconds to evaluate
    depth: int = 4
    width: int = 128
    activation: str = "silu"    # smooth, so the autograd delta and gamma are meaningful (ReLU's gamma is 0)
    output: str = "linear"      # softplus can't fit the labels' slightly negative premiums (-2.7e-6)
    split_ratio: float = 0.8    # the same 80/20 train/test split as every dataset class here
    valid_ratio: float = 0.2    # of the training split: ~84k validation rows at 2^19, plenty for a stable score
    batch_size: int = 1024      # the plan's 512-1024: low-noise gradients, few large ops (fast on CPU)
    epochs: int = 400           # the cosine's length, ~20 min at 2^19 rows; 400 beat 200 and 800 on price (PLAN.md)
    lr: float = 1e-3            # Adam's standard start
    lr_min: float = 1e-5        # 100x smaller: steps fine enough to settle at the end (the plan's value)
    patience: int = 20          # counted in the low-rate tail only: the cosine ends a normal run
    lbfgs_steps: int = 0        # off until the ablation shows the finish is worth its cost
    lambda_delta: float = 1.0   # equal weight to the premium term (both normalized by their variance)
    lambda_shape: float = 0.0   # off: hurt every region on the first trial (see above)
    shape_points: int = 1024    # one batch's worth: as many unlabeled points as labeled ones per step
    seed: int = 42              # the repo's usual seed

    def __post_init__(self):
        for name in ("batch_size", "epochs", "patience", "shape_points"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not self.lr > 0 or not 0 <= self.lr_min <= self.lr:
            raise ValueError(f"need lr > 0 and 0 <= lr_min <= lr, got lr={self.lr}, lr_min={self.lr_min}")
        if not 0 < self.valid_ratio < 1:
            raise ValueError(f"valid_ratio must be in (0, 1), got {self.valid_ratio}")
        for name in ("lambda_delta", "lambda_shape", "lbfgs_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

class LossScales(NamedTuple):
    """
    Variances of the targets over the training rows. Each loss term is
    divided by its target's variance, so it is unitless and a lambda of 1
    weighs it like the premium term, whatever the targets' sizes (the
    premium is ~1e-2 K, its delta ~1e-1).
    """
    premium: float
    delta: float

    @classmethod
    def fit(cls, premium: torch.Tensor, premium_dm: torch.Tensor) -> "LossScales":
        """The variances of the given rows; a zero variance (one row) is set to 1."""
        var = [float(t.double().var(unbiased=False)) if len(t) else 0.0 for t in (premium, premium_dm)]
        # 1.0: leaves that term unscaled rather than dividing by zero
        return cls(*(v if v > 0 else 1.0 for v in var))

class ShapePoints(NamedTuple):
    """
    Unlabeled points for the shape penalties: raw float64 parameters and
    the Black-Scholes put's derivatives there, which the full price's
    derivatives need (the net only adds the premium's).
    """
    m: torch.Tensor
    T: torch.Tensor
    r: torch.Tensor
    sigma: torch.Tensor
    bs_delta: torch.Tensor
    bs_gamma: torch.Tensor
    bs_dT: torch.Tensor
    bs_vega: torch.Tensor

    @classmethod
    def sample(cls, rng: np.random.Generator, n: int, domain: dict, device=None) -> "ShapePoints":
        """
        `n` points drawn like the dataset's rows (generate_tree_dataset's
        map_to_domain, on independent uniforms rather than Sobol points),
        so the penalties act where the network is used.
        """
        p = map_to_domain(rng.random((n, len(SAMPLED))), domain)
        return cls.from_params(p["m"], p["T"], p["r"], p["sigma"], device)

    @classmethod
    def from_params(cls, m, T, r, sigma, device=None) -> "ShapePoints":
        """Points at given (n,) parameter arrays, with their BS derivatives."""
        args = (m, 1.0, T, r, sigma)  # strike 1: everything is per unit of strike
        cols = [m, T, r, sigma, black_scholes_delta(*args, "put"), black_scholes_gamma(*args),
                black_scholes_dT(*args, "put"), black_scholes_vega(*args)]
        return cls(*(torch.as_tensor(c, dtype=torch.float64, device=device) for c in cols))

    def take(self, idx) -> "ShapePoints":
        """The points at `idx` (a slice or an index tensor)."""
        return ShapePoints(*(t[idx] for t in self))

class TrainResult(NamedTuple):
    """The surrogate with the best validation weights, and one dict of metrics per epoch."""
    surrogate: PutSurrogate
    history: list

def premium_and_delta(model: MLP, x: torch.Tensor, x_dm: torch.Tensor, create_graph: bool = True):
    """
    The net's premium and its derivative in m for standardized features `x`:
    the chain rule d net/dm = sum_j d net/dx_j * dx_j/dm, with `x_dm` =
    TreePutDataset's features_dm. Each row's output depends on its own row
    only, so the gradient of the sum is every row's gradient at once.
    create_graph keeps the delta differentiable, for training on it.

    Returns:
        ((n,) premium, (n,) delta), in the model's dtype.
    """
    x = x.detach().requires_grad_(True)
    premium = model(x).squeeze(-1)
    grad, = torch.autograd.grad(premium.sum(), x, create_graph=create_graph)
    return premium, (grad * x_dm).sum(-1)

def labeled_loss(model: MLP, batch, scales: LossScales, lambda_delta: float, create_graph: bool = True):
    """
    Premium MSE / var(premium) + lambda_delta * delta MSE / var(premium delta)
    on labeled rows `batch` = (features, features_dm, premium, premium_dm).
    The delta term is the differential-ML idea (Huge & Savine 2020): the
    slope labels pin the function down between rows, so the net's delta
    and gamma come out right, not only its values. Skipped (no second
    backward pass) when lambda_delta is 0.

    Returns:
        (loss, premium MSE, delta MSE), the MSEs unscaled and detached
        (delta MSE is NaN when skipped).
    """
    x, x_dm, premium, premium_dm = batch
    if lambda_delta == 0:
        pred = model(x).squeeze(-1)
        premium_mse = torch.mean((pred - premium) ** 2)
        return premium_mse / scales.premium, premium_mse.detach(), torch.tensor(float("nan"))
    pred, pred_delta = premium_and_delta(model, x, x_dm, create_graph=create_graph)
    premium_mse = torch.mean((pred - premium) ** 2)
    delta_mse = torch.mean((pred_delta - premium_dm) ** 2)
    loss = premium_mse / scales.premium + lambda_delta * delta_mse / scales.delta
    return loss, premium_mse.detach(), delta_mse.detach()

class PriceDerivatives(NamedTuple):
    """
    The net's premium and the *full* American price's derivatives
    (Black-Scholes put + premium) at some points, per unit of strike:
    delta = dV/dm, gamma = d2V/dm2, dT = dV/dT, vega = dV/dsigma.
    """
    premium: torch.Tensor
    delta: torch.Tensor
    gamma: torch.Tensor
    dT: torch.Tensor
    vega: torch.Tensor

def price_derivatives(model: MLP, stats: FeatureStats, points: ShapePoints,
                      create_graph: bool = True) -> PriceDerivatives:
    """
    The full price's derivatives at `points` (PriceDerivatives): the
    Black-Scholes part from the closed forms stored in ShapePoints, the
    net's from autograd through build_features and the stats, in m, T and
    sigma at once. One function for training (shape_violations) and the
    evaluation (tree_evaluation), so the violations test-ml reports are the
    ones the penalty acts on. create_graph keeps the second derivative
    differentiable (needed to train on it); the first derivatives always
    keep their graph, since gamma differentiates delta again.

    Returns:
        PriceDerivatives of (n,) tensors: premium in the model's dtype, the
        derivatives float64.
    """
    param = next(model.parameters())
    m, T, sigma = (v.detach().requires_grad_(True) for v in (points.m, points.T, points.sigma))
    x = stats.apply(build_features(m, T, points.r, sigma))
    premium = model(x.to(param)).squeeze(-1)

    d_m, d_T, d_sigma = torch.autograd.grad(premium.sum(), (m, T, sigma), create_graph=True)
    d2_m, = torch.autograd.grad(d_m.sum(), m, create_graph=create_graph)

    return PriceDerivatives(premium, points.bs_delta + d_m, points.bs_gamma + d2_m,
                            points.bs_dT + d_T, points.bs_vega + d_sigma)

def scaled_violations(derivs: PriceDerivatives, points: ShapePoints) -> torch.Tensor:
    """
    How much the full price breaks each no-arbitrage condition, one column
    per SHAPE_CONDITIONS: relu(w m dV/dm), relu(-w^2 m^2 d2V/dm2),
    relu(-T dV/dT), relu(-sigma dV/dsigma), w = sigma sqrt(T), 0 where the
    condition holds (see shape_violations for the scaling).

    Returns:
        (n, 4) tensor, float64.
    """
    m, T, sigma = points.m, points.T, points.sigma
    w = sigma * torch.sqrt(T)  # the price's natural length scale in log m (a constant factor)
    return torch.relu(torch.stack([w * m * derivs.delta, -(w * m) ** 2 * derivs.gamma,
                                   -T * derivs.dT, -sigma * derivs.vega], dim=-1))

def shape_violations(model: MLP, stats: FeatureStats, points: ShapePoints, create_graph: bool = True) -> torch.Tensor:
    """
    How much the *full* American price V = BS put + net premium breaks each
    no-arbitrage condition at `points`, one column per SHAPE_CONDITIONS:
    relu(w m dV/dm), relu(-w^2 m^2 d2V/dm2), relu(-T dV/dT),
    relu(-sigma dV/dsigma), w = sigma sqrt(T), 0 where the condition holds.

    Each derivative is multiplied by a positive factor, which never changes
    its sign condition, chosen so that every column is a price change of
    comparable size and their squares can share the premium's
    normalization: the price moves on a scale of w in log m, so w m dV/dm
    and w^2 m^2 d2V/dm2 are the first- and second-order price change over
    a one-standard-deviation move of the stock; T dV/dT and sigma dV/dsigma
    the change for a 100% relative change of T or sigma. The w factors
    matter: with m dV/dm and m^2 d2V/dm2 alone (only "price units"), the
    convexity term is ~price / w^2, and with w down to 0.007 in the domain
    its square outweighed the labeled loss by ~3000x (54.8 vs 0.017),
    which is what made the penalty hurt (BUGS.md).

    The full price, not the premium: the premium alone may fall with T,
    and the European put does (deep in the money, r > 0), while an American
    put never does. The derivatives come from price_derivatives.

    Returns:
        (n, 4) tensor, float64.
    """
    return scaled_violations(price_derivatives(model, stats, points, create_graph), points)

def shape_penalty(model: MLP, stats: FeatureStats, points: ShapePoints, scales: LossScales) -> torch.Tensor:
    """Mean over points of the summed squared violations, / var(premium)."""
    return shape_violations(model, stats, points).pow(2).sum(-1).mean() / scales.premium

def evaluate_labeled(model: MLP, rows, scales: LossScales, lambda_delta: float) -> dict:
    """
    Labeled loss and RMSEs over all of `rows`, in chunks of EVAL_CHUNK.

    Returns:
        dict with loss (premium_term + delta_term, as trained, what the
        checkpoint and early stopping go by), its two halves premium_term
        (premium MSE / its scale) and delta_term (lambda_delta * delta MSE /
        its scale), reported apart because one can improve while the other
        gets worse (an 800-epoch run lowered the loss by trading price for
        delta accuracy, PLAN.md), and premium_rmse and delta_rmse (per unit
        of strike; delta_rmse is computed even when lambda_delta is 0, as a
        diagnostic).
    """
    n = len(rows[0])
    premium_se = delta_se = 0.0
    for start in range(0, n, EVAL_CHUNK):
        chunk = tuple(t[start:start + EVAL_CHUNK] for t in rows)
        # lambda_delta=1.0 (any non-zero value): forces the delta MSE to be computed, the weighting is applied below
        _, premium_mse, delta_mse = labeled_loss(model, chunk, scales, lambda_delta=1.0, create_graph=False)
        premium_se += float(premium_mse) * len(chunk[0])
        delta_se += float(delta_mse) * len(chunk[0])
    premium_mse, delta_mse = premium_se / n, delta_se / n
    premium_term, delta_term = premium_mse / scales.premium, lambda_delta * delta_mse / scales.delta
    return {"loss": premium_term + delta_term, "premium_term": premium_term, "delta_term": delta_term,
            "premium_rmse": math.sqrt(premium_mse), "delta_rmse": math.sqrt(delta_mse)}

def train_surrogate(dataset_path: Union[str, os.PathLike], config: TrainConfig = TrainConfig(),
                    checkpoint_path: Optional[Union[str, os.PathLike]] = None, device=None) -> TrainResult:
    """
    Trains an MLP on the training split of a tree put dataset
    (`python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`) and returns it as a
    PutSurrogate with the dataset's feature stats and domain.

    The training split (TreePutDataset, `config.split_ratio`, `config.seed`)
    is split again, seeded, into rows trained on and `valid_ratio` held-out
    rows; the test split is never read. Each Adam step: a batch of labeled
    rows (labeled_loss) plus, if lambda_shape > 0, `shape_points` fresh
    random points (shape_penalty); the learning rate follows a cosine from
    lr to lr_min over all steps of `epochs` epochs. After each epoch the
    labeled loss on the validation rows decides the checkpoint (best so
    far, saved to `checkpoint_path` if given) and early stopping. An
    optional full-batch L-BFGS finish starts from the best weights and is
    kept only if it improves the validation loss.

    Args:
        dataset_path: the tree_put_dataset.npz file.
        config: the run's settings (TrainConfig).
        checkpoint_path: where to save the best surrogate (PutSurrogate.save);
            None saves nothing.
        device: torch device to train on (None: CPU). The returned surrogate
            is on the CPU.

    Returns:
        TrainResult(surrogate, history), history[i] holding epoch i's
        metrics; L-BFGS adds a final entry with "epoch": "lbfgs".

    Raises:
        ValueError: if the training or validation rows come out empty, or
            from TreePutDataset/MLP for a bad file or architecture.
    """
    device = torch.device(device or "cpu")
    # the training split only, with its features, targets and feature stats built at load time
    data = TreePutDataset(dataset_path, is_train=True, split_ratio=config.split_ratio, seed=config.seed)
    # hold out valid_ratio of it, seeded, to choose the checkpoint and stop early
    n_valid = int(len(data) * config.valid_ratio)
    if n_valid == 0 or n_valid == len(data):
        raise ValueError(f"{len(data)} training rows leave no rows to train on or to validate with "
                         f"(valid_ratio={config.valid_ratio})")
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(config.seed))
    tensors = (data.features, data.features_dm, data.premium, data.premium_dm)
    valid = tuple(t[perm[:n_valid]].to(device) for t in tensors)
    train = tuple(t[perm[n_valid:]].to(device) for t in tensors)
    n_train = len(train[0])
    # Fit on the labels: the targets' variances (trained rows only), which normalize each loss term
    scales = LossScales.fit(train[2], train[3])

    torch.manual_seed(config.seed)  # the weights' initialization
    model = MLP(n_inputs=len(FEATURES), depth=config.depth, width=config.width,
                activation=config.activation, output=config.output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # cosine from lr to lr_min over every step of the run, stepped per batch so the decay is smooth
    steps_per_epoch = math.ceil(n_train / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * steps_per_epoch,
                                                           eta_min=config.lr_min)
    # their own seeded generators, so the batch order and the shape points are reproducible
    batch_order = torch.Generator().manual_seed(config.seed)
    shape_rng = np.random.default_rng(config.seed)

    # the surrogate is saved on the CPU, with the stats and domain it needs to price
    def save(surrogate_model):
        if checkpoint_path is not None:
            PutSurrogate(_cpu_copy(surrogate_model), data.stats, data.domain).save(checkpoint_path)

    best_loss, best_state, epochs_no_improve, history = math.inf, None, 0, []
    # epochs without improvement count towards early stopping only at or below this rate
    patience_lr = config.lr_min + PATIENCE_LR_FRACTION * (config.lr - config.lr_min)
    epoch_bar = tqdm(range(config.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        # a fresh shuffle of the trained rows each epoch
        order = torch.randperm(n_train, generator=batch_order).to(device)
        sums = {"loss": 0.0, "shape": 0.0}
        batch_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch", leave=False)
        for step in batch_bar:
            idx = order[step * config.batch_size:(step + 1) * config.batch_size]
            # premium error + lambda_delta * delta error on this batch of labeled rows
            loss, _, _ = labeled_loss(model, tuple(t[idx] for t in train), scales, config.lambda_delta)
            if config.lambda_shape > 0:
                # no-arbitrage penalty on fresh unlabeled points, drawn like the dataset's rows
                points = ShapePoints.sample(shape_rng, config.shape_points, data.domain, device)
                penalty = shape_penalty(model, data.stats, points, scales)
                loss = loss + config.lambda_shape * penalty
                sums["shape"] += float(penalty.detach())
            # one Adam step, then one step of the cosine schedule
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums["loss"] += float(loss.detach())
            batch_bar.set_postfix(loss=f"{float(loss.detach()):.3e}")

        # score the epoch on the held-out rows (labeled loss only: the random shape term would add noise)
        metrics = evaluate_labeled(model, valid, scales, config.lambda_delta)
        history.append({"epoch": epoch + 1, "train_loss": sums["loss"] / steps_per_epoch,
                        "shape_penalty": sums["shape"] / steps_per_epoch, "lr": optimizer.param_groups[0]["lr"],
                        **{f"valid_{k}": v for k, v in metrics.items()}})
        # the combined loss decides, its two halves are shown apart (one can improve as the other worsens)
        shown = {"valid": f"{metrics['loss']:.3e}", "price": f"{metrics['premium_term']:.2e}",
                 "delta": f"{metrics['delta_term']:.2e}"}
        # keep (and save) the weights only on a strict improvement; else count towards early
        # stopping, once the rate is low enough that a stalled loss is not just high-rate jitter
        if metrics["loss"] < best_loss:
            best_loss, epochs_no_improve = metrics["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            save(model)
            epoch_bar.set_postfix(**shown, best="saved")
        elif optimizer.param_groups[0]["lr"] > patience_lr:
            epoch_bar.set_postfix(**shown, no_improve="not counted (rate high)")
        else:
            epochs_no_improve += 1
            epoch_bar.set_postfix(**shown, no_improve=f"{epochs_no_improve}/{config.patience}")
            if epochs_no_improve >= config.patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is None:  # a NaN loss never compares below best_loss
        raise ValueError("the validation loss was never finite: training diverged (try a lower lr)")
    # back to the best epoch's weights, which the L-BFGS finish (if any) starts from
    model.load_state_dict(best_state)
    if config.lbfgs_steps > 0:
        # a fixed pool, as many points as training rows: the line search
        # compares losses between trial steps, which fresh points would break
        pool = (ShapePoints.sample(shape_rng, n_train, data.domain, device)
                if config.lambda_shape > 0 else None)
        _lbfgs_finish(model, train, scales, config, data.stats, pool)
        # keep the finish only if it beats Adam's best on the held-out rows
        metrics = evaluate_labeled(model, valid, scales, config.lambda_delta)
        kept = metrics["loss"] < best_loss
        history.append({"epoch": "lbfgs", "kept": kept, **{f"valid_{k}": v for k, v in metrics.items()}})
        if kept:
            save(model)
        else:
            model.load_state_dict(best_state)

    return TrainResult(PutSurrogate(_cpu_copy(model), data.stats, data.domain), history)

def _lbfgs_finish(model: MLP, train, scales: LossScales, config: TrainConfig, stats: FeatureStats,
                  pool: Optional[ShapePoints]) -> None:
    """
    `config.lbfgs_steps` full-batch L-BFGS iterations (strong-Wolfe line
    search) on the training loss, the shape penalty taken on the fixed
    `pool`. The gradient is accumulated chunk by chunk, each chunk's loss
    weighted by its share of the rows, so it equals the full-batch one
    without holding the whole batch's graph in memory.
    """
    # max_eval: the cap on loss evaluations (line-search trials included);
    # 1.25 x max_iter is torch's own default, set explicitly so the bar knows it
    max_eval = config.lbfgs_steps * 5 // 4
    # lr=1.0: the first trial step is the full quasi-Newton step, whose natural
    # length is 1; the line search shortens or stretches it from there.
    # history_size=50: (s, y) pairs kept for the curvature estimate, ~400 KB
    # each for 50k weights; between the textbook 3-20 and torch's default 100,
    # not tuned
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=config.lbfgs_steps, max_eval=max_eval,
                                  history_size=50, line_search_fn="strong_wolfe")
    n_train = len(train[0])
    # one tick per loss evaluation; a run that converges early stops short of the total
    bar = tqdm(total=max_eval, desc="L-BFGS", unit="eval", leave=False)

    def closure():
        bar.update()
        optimizer.zero_grad()
        total = 0.0
        for start in range(0, n_train, EVAL_CHUNK):
            chunk = tuple(t[start:start + EVAL_CHUNK] for t in train)
            loss = labeled_loss(model, chunk, scales, config.lambda_delta)[0] * (len(chunk[0]) / n_train)
            loss.backward()
            total += float(loss.detach())
        if pool is not None:
            for start in range(0, len(pool.m), EVAL_CHUNK):
                points = pool.take(slice(start, start + EVAL_CHUNK))
                loss = config.lambda_shape * shape_penalty(model, stats, points, scales) * (len(points.m) / len(pool.m))
                loss.backward()
                total += float(loss.detach())
        bar.set_postfix(loss=f"{total:.3e}")
        return torch.tensor(total)

    optimizer.step(closure)
    bar.close()

def _cpu_copy(model: MLP) -> MLP:
    """A CPU copy of `model` (the surrogate's home), leaving the trained one where it is."""
    copy = MLP(**model.config)
    copy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return copy
