# Legacy MLP surrogate against Longstaff-Schwartz

The first surrogate pricer, `LegacyMLP`, an MLP trained on Longstaff-Schwartz prices. The tree-trained surrogate replaced it (README, "MLP surrogate"). It stays as a library in `optionpricing/ml/surrogate/legacy/`, and no CLI command runs it.

**Status.** The numbers are **not re-verified** since the tree surrogate replaced the pipeline.

## Setup

**The model.** An MLP predicts `(price - smooth_intrinsic_value) / K` for American calls and puts, from features of `(S0, K, σ, r, T)`. The labels are LSM prices.

**The data.** `generate_lsm_dataset` samples `(S0, K, σ, r, T)` at random, with moneyness `S0/K` log-uniform and continuous over `(0.4, 1.6)`, and labels each sample with LSM on 10,000 paths (`--path_per_sim`).

**The benchmark.** The surrogate's put price against a fresh LSM run with a fixed seed and 20,000 paths per point, over a grid of `(S0, K, σ, r, T)`. The grid sweeps moneyness together with σ and T, so the deep out-of-the-money points also have the highest σ and T. A "deep OTM" put there is still worth $3-8, not a few cents. Errors are mean relative errors per moneyness bucket.

## Results

```
python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset   # the LSM training set, ~40 min at defaults
python -c "from optionpricing.ml.surrogate.legacy.lsm_training import train_legacy_mlp; train_legacy_mlp('artifacts/data/american_option_dataset.npz')"
```

The benchmark used the helpers in `optionpricing/ml/surrogate/legacy/lsm_evaluation.py`.

| Version | ITM (S < K) | ATM | OTM (S > K) |
|---|---|---|---|
| Hard-kink target, bucketed data | 9.5% | ~3% | ~1.2% |
| Smooth target, continuous data | 0.78% | 3.29% | 6.07% ($0.31 avg abs) |
| **Current checkpoint**, benchmark on the fixed horizon | **0.57%** | **3.82%** | **6.32%** ($0.34 avg abs) |

On the held-out split: MSE 0.000017, MAE 0.0028, R² 0.992, in `(price - smooth_intrinsic)/K` units.

**Out of the money by moneyness** (measured against the benchmark before the horizon fix):

| Moneyness | Error | Cause |
|---|---|---|
| 1.05-1.20 | 6-14% | inside the `±3 × beta_frac ≈ ±0.18` window that the smoothed target widens |
| 1.25-1.38 | < 2% | clear of both effects |
| 1.40-1.50 | 7-10% | near the edge of the trained range `(0.4, 1.6)` |

## Discussion

Three changes took the in-the-money error from 9.5% to 0.78%. The current checkpoint's 0.57% is the same model scored on the benchmark with the fixed horizon.

**Continuous moneyness.** Moneyness was first sampled from 5 buckets (`{0.5, 0.7, 1.0, 1.2, 1.5}` plus noise), leaving gaps with no data, such as 0.75 to 0.95. The benchmark probed one of those gaps, so part of the in-the-money error was extrapolation. Log-uniform, continuous sampling over `(0.4, 1.6)` removed it, and gave most of the gain.

**A smooth target.** Predicting `price - intrinsic_value` shrinks the target's range, since deep in-the-money prices are many times larger than out-of-the-money ones. But the intrinsic value `max(K - S, 0)` has a kink at `S = K` that the price before expiry doesn't have. Subtracting it stamps the kink onto the target, and the error spiked at the money. A softplus approximation, `smooth_intrinsic_value`, keeps the range benefit without the kink. Its corner width, `beta_frac = 0.06` (a fraction of `K`), was picked by a sweep against a fixed LSM benchmark, given in the function's docstring. The general pattern is in [architecture.md](../architecture.md), "Design note".

**More paths per label.** Going from 5,000 to 10,000 paths reduced label noise. Switching the LSM basis between simple and Laguerre had no effect, since at degree 3 they give the same labels (README, "LSM delta and gamma, accurate up to expiry").

**Out of the money is genuinely worse, for two bounded reasons.** The U-shape in the moneyness table has two causes. Just past the money, the smoothed target's correction spreads into a window around the strike, a known, accepted cost of the fix. Near 1.5, the training data thins at the edge of the range. In between, the error is under 2%, so the model has no trouble with "out of the money" as such.

## Limits

- **The dataset predates the horizon fix.** Its labels come from the old simulator, which priced each option one time step short of its maturity ([B-007](../../.claude/BUGS.md#b-007-simulated-horizon-was-one-step-short-of-the-maturity)). They are biased low by up to 0.0012·K (-3.4% at the money, more in relative terms deep out of the money, shrinking with `T`). The benchmark now runs on the full horizon, so part of the current checkpoint's at- and out-of-the-money error is this label bias. The dataset was never regenerated, since the tree surrogate replaced the pipeline first (`.claude/PLAN_ARCHIVE.md`, "ML surrogate: simple basis, calls and puts, regenerated LSM dataset").
- The held-out metrics are in residual units, so they don't compare with raw `price/K` numbers.
- The benchmark is LSM, itself a Monte-Carlo estimate. The tree surrogate is scored against an independent finer tree instead.

## Reproducing the runs

The dataset generator is resumable and seeded, with chunks in `artifacts/data/chunks/`. Its flags are in `--help`.

```
python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset --help
```

It writes `artifacts/data/american_option_dataset.npz` and the feature-normalization stats in `artifacts/data/features_norm.npz`.
