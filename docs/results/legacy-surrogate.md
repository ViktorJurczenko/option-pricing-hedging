# Legacy MLP surrogate vs. Longstaff-Schwartz

The first surrogate pricer: `LegacyMLP`, an MLP trained on Longstaff-Schwartz prices. The tree-trained surrogate (README, "MLP surrogate") replaced it. It is kept as a library in `optionpricing/ml/surrogate/legacy/`, and no CLI command runs it any more. This write-up was moved from the README with only its references updated; its numbers are **not re-verified** since the tree surrogate replaced the pipeline.

The legacy pipeline is still runnable:

```
python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset   # the LSM training set (~40 min at defaults)
python -c "from optionpricing.ml.surrogate.legacy.lsm_training import train_legacy_mlp; train_legacy_mlp('artifacts/data/american_option_dataset.npz')"
```

The benchmark below was produced by the old `test-ml`, using the helpers
now in `optionpricing/ml/surrogate/legacy/lsm_evaluation.py`.

An earlier version of this surrogate had a real ~9.5% average error on ITM
puts. Chasing that down changed several things about how the dataset is built
and how the model's target is defined — worth explaining since none of it is
obvious from reading `optionpricing/ml/surrogate/legacy/lsm_model.py` alone:

**1. Training data now covers moneyness continuously.** `optionpricing/ml/surrogate/legacy/generate_lsm_dataset.py`
used to sample moneyness (`S0/K`) from 5 discrete buckets (`{0.5, 0.7, 1.0, 1.2,
1.5}` ± a little noise), leaving large *untrained* gaps between them (e.g.
nothing at all between 0.75 and 0.95). The old evaluation benchmark happened to
probe exactly one of those gaps, so a chunk of the "ITM error" was really the
model extrapolating into territory it had never seen. `gen_rdm_params` now
samples moneyness log-uniformly and continuously over `(0.4, 1.6)` instead.

**2. The label is a *smoothly* decomposed residual, not raw price.** The MLP
predicts `(price - smooth_intrinsic_value) / K` rather than `price / K`
directly (see `optionpricing/ml/surrogate/legacy/lsm_model.py::smooth_intrinsic_value`). The
naive version of this idea — subtracting the *exact* intrinsic value
`max(K-S, 0)` — helps in general (it shrinks the target's dynamic range: deep
ITM prices can be many times larger than deep OTM ones), but that exact
formula has a hard kink at `S == K`, while the option's actual *price* before
expiry has no such kink there (only the expiry *payoff* does). Subtracting a
kinked function from a smooth one reintroduces the kink into the residual —
which showed up empirically as the model's error spiking specifically at the
money once the coverage gap above was fixed. Using a smooth (softplus-based)
approximation of intrinsic value instead removes that artifact while keeping
the dynamic-range benefit. `beta_frac=0.06` (the width of the smoothed corner,
as a fraction of `K`) was picked by sweeping several values against a fixed
LSM benchmark — see the function's docstring for the sweep.

**3. More paths per label (`--path_per_sim`, 5,000 → 10,000)** reduces
label noise. The LSM "teacher" was briefly switched to the Laguerre basis
and back to the simple one (`--reg_type`, default `simple`); at degree 3
the two produce the same labels (README, "LSM delta and gamma, accurate up to expiry", its regression-basis paragraph), so
neither switch had an effect.

Net effect, benchmarked live against a fresh LSM run (put) across a grid of
`(S0, K, σ, r, T)`, with a fixed seed and 20,000 paths per benchmark point:

| | ITM (S < K) | ATM | OTM (S > K) |
|---|---|---|---|
| Before (hard-kink target, bucketed data) | 9.5% | ~3% | ~1.2% |
| After (smooth target, continuous data) | 0.78% | 3.29% | 6.07% ($0.31 avg abs) |
| **Now** (same model, benchmark on the fixed horizon) | **0.57%** | **3.82%** | **6.32%** ($0.34 avg abs) |

**The dataset predates the horizon fix.** The MLP was trained on labels
from the old simulator, which priced each option one time step short of
its maturity (`.claude/BUGS.md`, "Simulated horizon was one step short of the maturity"). The labels are biased low by
up to 0.0012·K (-3.4% ATM, more in relative terms deep OTM, shrinking with
`T`), while the benchmark LSM now runs on the full horizon. The "Now" row
is this checkpoint measured against the corrected benchmark, so part of
its ATM/OTM error is that label bias. The dataset was never regenerated:
the tree-trained surrogate replaced this pipeline first
(`.claude/PLAN_ARCHIVE.md`, "ML surrogate: simple basis, calls and puts,
regenerated LSM dataset"). The moneyness breakdown below was measured
against the old benchmark.

ITM improved by an order of magnitude (mostly from fixing the data gap), ATM
stayed about the same, and OTM got a bit worse — genuinely worse, not just a
percentage-of-a-small-number artifact. (An earlier version of this section
claimed OTM put prices are small to begin with, so the $0.31 miss was mostly
harmless; that's wrong here, because this benchmark sweeps moneyness together
with σ and T rather than independently, so by the time moneyness reaches 1.5
both σ and T are also at their highest swept values — enough for a "deep OTM"
put to still be worth $3-8, not a few cents.) Breaking the OTM bucket down by
moneyness shows a U-shape with two distinct, explainable causes rather than
one uniform problem: error is worst (6-14%) for moneyness 1.05-1.20, just past
the money, inside the same `±3×beta_frac ≈ ±0.18` window that the ATM
smoothing correction (above) deliberately widens to fix the ATM spike — a
known, accepted cost of that fix. It's worst again (7-10%) for moneyness
1.40-1.50, close to the edge of the trained `moneyness_range=(0.4, 1.6)`,
which is a milder version of the same coverage-thinning issue fixed for ITM.
In between (moneyness 1.25-1.38, comfortably clear of both effects) the
surrogate is accurate to <2%, confirming the surrogate itself isn't
struggling with "OTM" as a concept — it's these two specific, bounded effects
stacking at the edges of the bucket. On the held-out split: MSE 0.000017, MAE
0.0028, R² 0.992 (in `(price - smooth_intrinsic)/K` units now, not raw
`price/K`, so not directly comparable to older numbers from before this
change).

## Generating its dataset

The legacy dataset is generated like the tree dataset (resumable, seeded), with chunks in `artifacts/data/chunks/`. All flags are in `docs/architecture.md`, "Command reference".

```
python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset [--path_per_sim N] [--nb_param_comb N] [--samples_per_param N] \
    [--nb_jobs N] [--sampling N] [--option_type call|put|both] [--reg_type simple|laguerre] [--max_degree N] \
    [--output_dir artifacts/data] [--seed N] [--chunk_size N]
```

It parallelizes LSM pricing across randomly sampled `(S0, K, σ, r, T)`
combinations to build `artifacts/data/american_option_dataset.npz` and the
feature-normalization stats in `artifacts/data/features_norm.npz`.
