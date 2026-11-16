"""Load a bank of hedging scenarios written by `generate_hedge_dataset`, and refuse banks that do not belong together.

A training run reads a train bank and a validation bank, and a test reads a test bank. They are only comparable if
they come from the same contract family, holder, trees and surrogate, and from distinct seeds, so their paths are
independent. `check_compatible` enforces both, from the settings each bank stores.

Maturities differ per scenario, so every (scenario, day) array is padded to the longest one and is NaN past each
scenario's expiry (`spot`) or from its `τ` on (`delta`, `gamma`). `hedge_pnl` never reads those days, so the
padding needs no explicit mask.

Training and testing hedge a bank the same way: a `Hedger` (a policy and the features it reads) is booked by
`hedge_pnl` on a batch of scenarios (`hedge_batch`), or on the whole bank in batches (`hedge_bank`).
"""

import math
import os
from collections.abc import Callable
from dataclasses import asdict
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from optionpricing.hedging.pnl import HedgeOutcome, HedgeSettings, Policy, hedge_pnl
from optionpricing.ml.hedging.generate_hedge_dataset import DEFAULT_RANGES, FORMAT

# The settings two banks used together must share. Everything that changes what a path means is here: the
# contract family, the holder, the trees, the surrogate and the units of time and money.
SHARED_SETTINGS = (
    *DEFAULT_RANGES,
    "strike",
    "days_per_year",
    "exercise",
    "shift",
    "tree_steps",
    "surrogate_sha256",
)
# The arrays of a bank file, per scenario and day, per contract, and per scenario.
DAILY_ARRAYS = ("spot", "delta", "gamma")
CONTRACT_ARRAYS = ("boundary", "days", "sigma", "risk_free_rate", "moneyness", "premium")
SCENARIO_ARRAYS = ("tau", "contract")


def _same(a, b) -> bool:
    """Whether two stored settings are equal. NaN (no shift) equals NaN, and ranges compare element by element."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.all((a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))))
    return bool(np.all(a == b))


def check_compatible(reference: dict, other: dict) -> None:
    """Refuse to use two banks together unless they share their settings and differ in seed.

    Args:
        reference: the settings of one bank (`HedgeDataset.settings`), e.g. the training bank.
        other: the settings of the bank to use with it, e.g. the validation or test bank.

    Raises:
        ValueError: listing every `SHARED_SETTINGS` entry that differs, or if both banks have the same seed, so
            their paths would be the same.
    """
    differ = {k: (reference[k], other[k]) for k in SHARED_SETTINGS if not _same(reference[k], other[k])}
    if differ:
        raise ValueError(f"the banks were generated with other settings ((reference, other): {differ})")
    if reference["seed"] == other["seed"]:
        raise ValueError(f"both banks have seed {reference['seed']}, so they share their paths")


class HedgeDataset(Dataset):
    """A bank of hedging scenarios, as tensors, one item per scenario.

    Indexing with an int gives one scenario and with an index tensor or list a batch, so a training loop can draw
    batches by indexing directly. An item is a dict of tensors, one row per scenario:
    - `spot` (dollars), `delta` (shares), `gamma` (shares per dollar), `boundary` (dollars) and `time_left`
      (years to expiry), each of shape `(n_days,)` with `n_days = max(days) + 1`. `time_left` is NaN past
      expiry, like `spot`.
    - `tau` (the exercise day, int64), `premium` (dollars), `sigma` and `rate`, scalars.
    `boundary` is stored per contract and spread to the contract's scenarios here.

    Args:
        path: a file written by `generate_hedge_dataset`.
        dtype: floating dtype of the tensors. float32 by default, as stored. float64 suits an evaluation that
            wants the P&L in double precision.
        reference: the settings of a bank this one is used with (`HedgeDataset.settings` of the training bank),
            checked with `check_compatible`. None skips the check.

    Attributes:
        settings: the generation settings stored in the file (seed, ranges, holder, trees, surrogate).
        strike: the strike of every contract, in dollars.

    Raises:
        ValueError: if the file is not a hedge bank, or does not match `reference`.
    """

    def __init__(self, path: str | os.PathLike, dtype: torch.dtype = torch.float32, reference: dict | None = None):
        with np.load(path) as data:
            if "format" not in data.files or str(data["format"]) != FORMAT:
                raise ValueError(f"'{path}' is not a hedge bank (generate_hedge_dataset, format '{FORMAT}')")
            arrays = {k: data[k] for k in (*DAILY_ARRAYS, *CONTRACT_ARRAYS, *SCENARIO_ARRAYS)}
            names = set(data.files) - set(arrays)
            # 0-d arrays become Python scalars and strings, ranges become tuples
            self.settings = {k: data[k].item() if data[k].ndim == 0 else tuple(data[k].tolist()) for k in sorted(names)}
        if reference is not None:
            check_compatible(reference, self.settings)
        self.strike = float(self.settings["strike"])

        self.contract = torch.as_tensor(arrays["contract"], dtype=torch.int64)
        self.tau = torch.as_tensor(arrays["tau"], dtype=torch.int64)
        self.spot, self.delta, self.gamma = (torch.as_tensor(arrays[k], dtype=dtype) for k in DAILY_ARRAYS)
        self.boundary = torch.as_tensor(arrays["boundary"], dtype=dtype)
        self.days = torch.as_tensor(arrays["days"], dtype=torch.int64)
        self.sigma, self.rate, self.moneyness, self.premium = (
            torch.as_tensor(arrays[k], dtype=dtype) for k in ("sigma", "risk_free_rate", "moneyness", "premium")
        )
        self._day = torch.arange(self.spot.shape[1])

    @property
    def shift(self) -> float | None:
        """The late holder's shift, or None for every other holder (stored as NaN)."""
        shift = self.settings["shift"]
        return None if math.isnan(shift) else shift

    def __len__(self) -> int:
        return len(self.tau)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        contract = self.contract[idx]
        days_left = (self.days[contract].unsqueeze(-1) - self._day).to(self.spot.dtype)
        time_left = torch.where(days_left >= 0, days_left, torch.nan) / self.settings["days_per_year"]
        return {
            "spot": self.spot[idx],
            "delta": self.delta[idx],
            "gamma": self.gamma[idx],
            "boundary": self.boundary[contract],
            "time_left": time_left,
            "tau": self.tau[idx],
            "premium": self.premium[contract],
            "sigma": self.sigma[contract],
            "rate": self.rate[contract],
        }


# Features: a batch of `HedgeDataset` items and the bank's strike in, a policy's `inputs` tensor out.
Features = Callable[[dict[str, torch.Tensor], float], torch.Tensor]


class Hedger(NamedTuple):
    """A policy and the features it reads, so policies with different inputs hedge the same bank.

    Attributes:
        policy: a `pnl.Policy`, `policy(inputs, position) -> position`.
        features: builds the policy's `inputs` from a batch of the bank. The network's are `hedge_inputs`
            (`network_hedger` in `hedge_model.py`), the baselines' `bank_baseline_inputs` in `hedge_evaluation.py`.
    """

    policy: Policy
    features: Features


def hedge_batch(hedger: Hedger, dataset: HedgeDataset, idx, settings: HedgeSettings) -> HedgeOutcome:
    """Hedge the scenarios `idx` of `dataset` with `hedger`, booked by `hedge_pnl` at `settings`.

    The P&L keeps its autograd graph, so a training step can backpropagate through it.

    Args:
        hedger: the policy and its features.
        dataset: the bank.
        idx: the scenarios, an index tensor or list.
        settings: how the hedge is booked.

    Returns:
        The batch's `HedgeOutcome`, scenarios in the order of `idx`.
    """
    batch = dataset[idx]
    return hedge_pnl(
        hedger.policy,
        hedger.features(batch, dataset.strike),
        batch["spot"],
        batch["tau"],
        batch["premium"],
        dataset.strike,
        batch["rate"],
        **asdict(settings),
    )


@torch.no_grad()
def hedge_bank(hedger: Hedger, dataset: HedgeDataset, settings: HedgeSettings, batch_size: int) -> HedgeOutcome:
    """Hedge every scenario of `dataset` with `hedger`, `batch_size` scenarios at a time, without a graph.

    Batches end at different hedge dates, so each batch's `positions` is padded to the bank's number of hedge
    dates with each scenario's last position (0 for a scenario with no hedge date). That is the position it
    holds after its `τ` anyway, so `trades` is unchanged. The positions take `8 x scenarios x dates` bytes:
    about 150 MB for 50,000 daily paths of up to a year.

    Args:
        hedger: the policy and its features.
        dataset: the bank.
        settings: how the hedge is booked.
        batch_size: scenarios hedged at a time, >= 1. It only affects speed and memory.

    Returns:
        The bank's `HedgeOutcome`, scenarios in the bank's order.

    Raises:
        ValueError: if `batch_size` < 1 or the bank has no scenario.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(dataset) == 0:
        raise ValueError("the bank has no scenario to hedge")
    n_dates = -(-int(dataset.tau.max()) // settings.rebalance_every)  # dates 0, m, 2m, ... before the last τ
    pnl, costs, positions = [], [], []
    for start in range(0, len(dataset), batch_size):
        outcome = hedge_batch(hedger, dataset, torch.arange(start, min(start + batch_size, len(dataset))), settings)
        held = outcome.positions
        last = held[:, -1:] if held.shape[1] else torch.zeros((len(held), 1), dtype=held.dtype)
        positions.append(torch.cat([held, last.expand(-1, n_dates - held.shape[1])], dim=1))
        pnl.append(outcome.pnl)
        costs.append(outcome.costs)
    return HedgeOutcome(pnl=torch.cat(pnl), costs=torch.cat(costs), positions=torch.cat(positions))
