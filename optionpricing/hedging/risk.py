"""Risk measures of a batch of terminal hedging P&Ls, shared by training and evaluation.

Deep Hedging trains on a risk measure of the P&L and is judged on one, so both steps call the functions here:
"trained on CVaR" and "scored on CVaR" are then the same number. All functions are torch functions of a 1-D
tensor of P&Ls (one per path) and keep the autograd graph, so a loss built from them backpropagates into the
policy.

Signs: `cvar` is on the P&L scale, the metric the project reports (higher is better, usually negative). The
training objectives `cvar_loss`, `entropic` and `variance` are losses (lower is better). They are linked by
`min_w cvar_loss(pnl, w, alpha) == -cvar(pnl, alpha)`.
"""

import math

import torch


def _as_pnl(pnl) -> torch.Tensor:
    """Return `pnl` as a 1-D floating tensor, keeping its graph, or raise ValueError."""
    # A list or array becomes float64; a tensor keeps its dtype, device and graph.
    pnl = pnl if isinstance(pnl, torch.Tensor) else torch.as_tensor(pnl, dtype=torch.float64)
    if not pnl.is_floating_point():
        pnl = pnl.to(torch.float64)
    if pnl.ndim != 1:
        raise ValueError(f"pnl must be a 1-D tensor with one P&L per path, got shape {tuple(pnl.shape)}")
    if pnl.numel() == 0:
        raise ValueError("a risk measure of an empty P&L sample is undefined")
    return pnl


def _check_alpha(alpha: float) -> None:
    """Raise ValueError unless `alpha` is in (0, 1); NaN fails the comparison and is rejected too."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def cvar(pnl, alpha: float = 0.95) -> torch.Tensor:
    """Conditional value at risk of a P&L sample: the average of its worst `1 - alpha` share.

    This is the reported metric, on the P&L scale (higher is better). When `(1 - alpha) * n` is not a whole
    number of outcomes, the outcome on the edge of the tail counts for the fraction that falls inside it. This
    makes `cvar` equal to `-min_w cvar_loss` for every sample size, so the training objective and the reported
    metric agree exactly. When the share is whole, it is the plain average of the worst `(1 - alpha) * n`
    outcomes. When it is below one outcome, it is the worst outcome.

    Args:
        pnl: 1-D tensor (or array-like) of terminal P&Ls, one per path.
        alpha: confidence level in (0, 1). 0.95 averages the worst 5%.

    Returns:
        A 0-d tensor, differentiable in `pnl`.

    Raises:
        ValueError: if `pnl` is empty or not 1-D, or `alpha` is outside (0, 1).
    """
    _check_alpha(alpha)
    pnl = _as_pnl(pnl)
    tail = (1 - alpha) * pnl.numel()
    worst_first = torch.sort(pnl).values
    # Outcome i (0 = worst) counts fully while it is inside the tail, partly if the tail ends inside it.
    rank = torch.arange(pnl.numel(), dtype=pnl.dtype, device=pnl.device)
    weights = (tail - rank).clamp(0.0, 1.0)
    return (weights * worst_first).sum() / weights.sum()


def cvar_loss(pnl, w, alpha: float = 0.95) -> torch.Tensor:
    """Rockafellar-Uryasev form of the CVaR of the loss `L = -pnl`, the CVaR training objective.

    `w + mean((L - w)^+) / (1 - alpha)`. Its minimum over `w` is the CVaR of `L`, i.e. `-cvar(pnl, alpha)`,
    reached when `w` is the VaR of `L`. So training lowers it over the policy and `w` together, with no sort.

    Args:
        pnl: 1-D tensor (or array-like) of terminal P&Ls, one per path.
        w: the threshold, a float or a 0-d tensor (trainable in training).
        alpha: confidence level in (0, 1).

    Returns:
        A 0-d tensor, differentiable in `pnl` and `w`.

    Raises:
        ValueError: if `pnl` is empty or not 1-D, `w` is not a scalar, or `alpha` is outside (0, 1).
    """
    _check_alpha(alpha)
    pnl = _as_pnl(pnl)
    w = torch.as_tensor(w, dtype=pnl.dtype, device=pnl.device)
    if w.ndim != 0:
        raise ValueError(f"w must be a scalar, got shape {tuple(w.shape)}")
    return w + torch.relu(-pnl - w).mean() / (1 - alpha)


def entropic(pnl, lam: float) -> torch.Tensor:
    """Entropic risk of a P&L sample, `(1 / lam) * ln mean(exp(-lam * pnl))`, a loss.

    It penalizes every loss, more than linearly in its size, so it uses all paths rather than the tail only.
    It lies between `-mean(pnl)` (as `lam -> 0`) and `-min(pnl)` (as `lam -> inf`). The mean is computed with
    `logsumexp`, so `exp(-lam * pnl)` never overflows, however large `lam * pnl` is.

    Args:
        pnl: 1-D tensor (or array-like) of terminal P&Ls, one per path.
        lam: risk aversion, > 0.

    Returns:
        A 0-d tensor, differentiable in `pnl`.

    Raises:
        ValueError: if `pnl` is empty or not 1-D, or `lam` is not positive and finite.
    """
    if not (lam > 0 and math.isfinite(lam)):
        raise ValueError(f"lam must be positive and finite, got {lam}")
    pnl = _as_pnl(pnl)
    return (torch.logsumexp(-lam * pnl, dim=0) - math.log(pnl.numel())) / lam


def variance(pnl) -> torch.Tensor:
    """Variance of a P&L sample, the mean squared distance to its mean, a loss.

    Uses the `1 / n` (population) form, so one path gives 0 instead of NaN. At the batch sizes training uses
    (thousands of paths) the difference from `1 / (n - 1)` is negligible.

    Args:
        pnl: 1-D tensor (or array-like) of terminal P&Ls, one per path.

    Returns:
        A 0-d tensor, differentiable in `pnl`.

    Raises:
        ValueError: if `pnl` is empty or not 1-D.
    """
    pnl = _as_pnl(pnl)
    return torch.var(pnl, correction=0)
