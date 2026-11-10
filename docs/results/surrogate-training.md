# Tree surrogate: teacher, training-length and ablation results

Measured results behind the tree MLP surrogate's defaults (`binomial_put_batch`'s `method="bbsr"`, `nb_steps=1000`; `TrainConfig.epochs = 400`). Moved from `.claude/PLAN.md`, where they were recorded as the work went. **Not re-verified** since they were measured (2026-10-22 to 2026-11-10); rerun before quoting them as current.

## Choice of teacher: BBSR at N = 1000

Protocol: 120 seeded random draws over the dataset domain plus a 21-point hard corner (`m` 0.95-1.05, `T = 0.25`, `sigma = 0.05`). Reference: BBSR at 20000 steps, which agrees with CRR averaged over N, N + 1 then Richardson to 3.3e-6 at worst. Errors per unit of strike, max | mean, at N = 1000:

| Method | price | delta | gamma |
|---|---|---|---|
| CRR | 5.5e-5 \| 5.8e-6 | 2.2e-4 \| 1.7e-5 | 0.46 \| 4.1e-3 |
| BBS | 2.1e-5 \| 3.6e-6 | 2.9e-4 \| 1.5e-5 | 0.46 \| 4.1e-3 |
| BBSR | 3.2e-5 \| 9.6e-7 | 8.3e-4 \| 1.4e-5 | 0.33 \| 3.2e-3 |

- **BBSR has the lowest mean price error**, ~4x below BBS and ~6x below CRR. Its max (~3e-5 K = $0.003 at K = 100) is far below what an MLP reaches (~1e-4 K).
- **CRR's odd/even swing** between N = 1000 and 1001 is 8.7e-5. BBSR moves 1.5e-6 between N = 1000 and 1002 (BBS 1.1e-7: Richardson amplifies the exercise boundary's jitter), still noise far below the MLP's error.
- **N = 2000** halves the max error at 3.5x the cost: not worth it.
- **Greeks.** The large gamma errors in the table (max 0.33) were puts exercised today. Those now get their exact Greeks (`.claude/BUGS.md`, "Tree Greeks wrong just inside the exercise region"); the table predates that rule, which only changes exercised puts. With the rule, on 63 options against a 16000-step reference: gamma mean error 1.4e-3, median relative 3e-5 in the continuation region, delta mean 8.5e-6. Only a ~1e-3-wide band in `m` at the exercise boundary stays uncertain, where the true gamma jumps from 0. In the corner every method is accurate in price (BBSR 2.1e-6 max); gamma (~16) was off by up to 3.5 there before the rule, 8.4e-4 after, on a corner sweep.
- **Timing** (2048 random options, chunks of 32, idle machine): BBSR 0.80 / 2.71 / 9.57 ms per option at N = 500 / 1000 / 2000 (CRR 1.84 at 1000), so 200k puts cost ~9 min on one core. The teacher does not limit the dataset size.
- **Labels can be slightly below Black-Scholes** (min premium -2.7e-6 at N = 1000, OTM short T), which matters for the `--output softplus` ablation.

## Sampling: z instead of moneyness

Sampling `m` log-uniform on (0.4, 1.6) puts 66% of the draws in the money (the range is asymmetric in log: ln 0.4 = -0.92, ln 1.6 = +0.47), and 67% of those are already past the exercise boundary: 44% of all rows would be exercised today, with labels known in closed form. The generator samples `z = ln(m) / (sigma sqrt(T))` instead (`map_to_domain` in `optionpricing/ml/surrogate/tree/generate_tree_dataset.py`).

## Learning curve: training steps, not data, were the limit

Protocol: default `TrainConfig`, seed 42, 200 epochs each; `test-ml` grid against BBSR-4000, $ at K = 100.

| rows | \|z\|<1 MAE | all MAE | IV err (bp) | exercised max | train time |
|---|---|---|---|---|---|
| 2^15 | 0.0331 | 0.0236 | 33.5 | 0.894 | ~0.6 min |
| 2^16 | 0.0174 | 0.0125 | 19.4 | 0.634 | 1.3 min |
| 2^17 | 0.0094 | 0.0053 | 7.7 | 0.099 | 2.5 min |
| 2^18 | 0.0037 | 0.0028 | 5.0 | 0.027 | 5.0 min |
| 2^19 | 0.0022 | 0.0014 | 2.5 | 0.022 | 9.9 min |

The error still halves per doubling, but the curve is confounded: epochs are fixed, so a bigger set also gets proportionally more Adam steps, and every run's best epoch is 192-200 (still improving). Data and steps separated (400 epochs with `--patience 400`, since the then-default patience cut 400-epoch runs short, `.claude/BUGS.md`, "Early stopping cut longer cosine schedules short"):

| run | \|z\|<1 MAE | all MAE | IV err (bp) | valid loss | time |
|---|---|---|---|---|---|
| 2^19, 200 epochs | 0.0022 | 0.0014 | 2.46 | 6.90e-5 | 9.9 min |
| 2^18, 400 epochs (same steps) | 0.0018 | 0.0014 | 2.30 | 6.65e-5 | 9.9 min |
| 2^19, 400 epochs | 0.0012 | 0.0011 | 1.75 | 3.04e-5 | 19.6 min |

- **Same steps, half the data: as good.** Same data, twice the steps: ~45% less error near the money, best epoch 394/400 (still improving). 2^19 rows is enough data; the limit is the optimization.
- **Violations > 1e-5** (stricter than the 1e-4 default), grid continuation rows: 2.50% -> 1.53%, mostly `increasing_in_T`. Exercised rows show ~71-76% at 1e-5: wobble around exactly-zero derivatives, answered by the payoff floor.

## Training length: 400 epochs

Protocol: 2^19 rows, seed 42, grid against BBSR-4000, $ at K = 100. The validation terms are the loss's two halves (premium term + delta term), reported apart.

| epochs | \|z\|<1 MAE | all MAE | test MAE | valid premium RMSE | valid delta RMSE |
|---|---|---|---|---|---|
| 200 | 0.0022 | 0.0014 | 0.0013 | 2.62e-5 | 8.10e-4 |
| 400 | 0.0012 | 0.0011 | 0.0009 | 1.71e-5 | 5.38e-4 |
| 800 | 0.0019 | 0.0017 | 0.0015 | 2.35e-5 | 3.86e-4 |

- **800 epochs priced worse** (early-stopped at 665, best 645, ~33 min): it halved the combined loss, but past 400 the delta error kept falling (-28%) while the premium error rose (+38%), and the sum picks the checkpoint.
- 800 also had fewer violations (0.06% > 1e-4 at 800 epochs, 0.61% at 400) and more payoff floors (490 grid points at 800, 179 at 400).
- Single seed each. The seeded ablations below give the seed-to-seed spread.

## How a model is scored, and why these measures

Every model below is scored by `evaluate_grid` in `optionpricing/ml/surrogate/tree/tree_evaluation.py` (what `test-ml` and the ablation runner print): a factorial grid of 2342 puts priced by the surrogate and by an independent BBSR tree at 4000 steps, 4x the teacher's. The reference is not the teacher, so a score measures distance to the true price, not how well the network copies its teacher's own errors. Prices are in $ at K = 100. Each measure answers a question a user of the pricer would ask:

| Measure | Question it answers | Why it matters here |
|---|---|---|
| Mean error, `\|z\|` < 1 (near the money) | How far off is a typical price where options are most traded? | Near the money an option has the most time value and the most vega, and the early-exercise premium is hardest to learn there. The headline accuracy number. |
| Mean error, by `T` bucket and over the whole grid | Is any region of the domain weak? | One average over everything would hide a bad region behind easy ones, such as deep out-of-the-money puts worth almost nothing. |
| Worst error (grid max) | What is the largest miss a user can hit? | A pricer is queried anywhere, so its worst point is a real risk, and a worst point that changes a lot from seed to seed shows an unstable fit. |
| Implied-vol error, bp | How large is the price error in the units traders quote options in? | See below. It makes errors comparable across strikes and maturities. |
| Delta and gamma error | Can the surrogate's Greeks be used to hedge? | The surrogate is meant to give price, delta and gamma together; a hedge with a wrong delta loses money even if the price is right. |
| No-arbitrage violation rate | Does the price surface have the shape any arbitrage-free put price has? | The price must fall and be convex in `m` and rise with `T` and `σ`. A violation is a quote someone could trade against, and makes Greeks and calibration unstable. |
| Payoff-floor share | How often is the network overruled? | Shows where the routing, not the network, answers. |
| µs per option, training minutes | Is the surrogate worth it? | Its only purpose is speed: it must be much faster than the tree it learned from, in batches. |

**The implied-volatility error.** The implied volatility of a price is the `σ` that, put into the pricing model, gives back that price. Traders quote options by it rather than by dollars, because it can be compared across strikes and maturities. The implied-vol error asks: by how much would `σ` have to change for the true price to equal the surrogate's? To first order, that is the price error divided by vega (`∂V/∂σ`, how much the price moves per unit of `σ`):

```
IV error = |surrogate price − reference price| / vega
```

reported in basis points of volatility: 1 bp is 0.0001 in `σ`, for example 20.00% to 20.01%. The error is that of one price at one grid point, not an average: each point's `|error| / vega` uses its own vega, and only then are they averaged over a bucket. In the code (`bucket_metrics`), with `vega` from two teacher-grade trees at `σ(1 ± 1%)` (`tree_vega`):

```python
row.update(n_iv=int(sel.sum()), iv_mae=_mean(abs_err[sel] / vega[sel]))
```

It is computed only where vega is at least $0.01 per vol point (`MIN_IV_VEGA`), and never on exercised rows, whose price does not depend on `σ`. Why it is needed alongside the dollar error: the same dollar error is a different mistake on different options. For an at-the-money put with `σ` = 20% and `r` = 4% (Black-Scholes vega, close to the American put's there), a $0.0008 error is 0.21 bp at 1 year (vega $0.38 per vol point) but 1.45 bp at 1 week (vega $0.055 per vol point). For scale, a one-cent price tick on the 1-year put is 2.6 bp, so the chosen configuration's mean of 0.44 bp is about a sixth of a tick.

**Why the mean absolute error, not the mean squared error.** Training minimizes a squared error, which suits an optimizer (smooth, with a gradient proportional to the error). The report uses the mean absolute error because it is in dollars and is the typical miss. The MSE is in dollars squared, and its square root (RMSE, always ≥ MAE) is pulled up by the few largest errors, which the worst-error column already reports on their own.

**Mean ± sd over seeds, and the ~2 sd rule.** A run's result depends on its seed (initialization, batch order, data split). Each configuration is trained on several seeds, and a difference between two configurations is taken as real only when it is larger than about 2 seed standard deviations (sample sd, `n − 1`), using the larger of the two configurations' sds: a configuration whose results spread widely cannot be called better on the strength of its lucky seeds. Smaller differences may be luck.

## Seeded ablations: lambda_delta 0.3 + softplus has the best mean error, a seed-dependent worst point

```
# the 8 one-change variants below (24 runs)
python -m optionpricing.ml.experiments.run_ablations --variants baseline lambda_delta_0 lambda_delta_0.3 lambda_delta_3 softplus_output relu lbfgs_200 shape_0.1
# the combined variant, lambda_delta 0.3 + softplus (3 runs)
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus
```

Without `--variants` the runner trains every variant, the combined one included. Each command skips runs already finished in `--out_dir`, and `summary.md` covers every finished run.

Protocol: 2^19 rows, 400 epochs, seeds 42, 43, 44; each variant is the default `TrainConfig` (`lambda_delta = 1`, linear output, SiLU, no L-BFGS finish, no shape term) with one change, except the combined variant, which makes two. Grid against BBSR-4000 (27 z x 6 sigma x 8 T, r in {0.02, 0.06}), $ at K = 100; IV error, gamma and the maxima are over the whole grid (or the exercised rows), delta near the money, mean ± sample sd over the 3 seeds. Measured 2026-10-28/29, 4 runs at a time with 2 threads each; `shape_0.1` and the combined variant (2026-10-31, its 3 runs at a time) ran in separate invocations, so their training times are not strictly comparable. Full table (also the validation terms and the 1e-5 violation rate): `artifacts/runs/ablations/summary.md`.

| variant | \|z\|<1 MAE | all MAE | grid max | exercised max | IV err bp | delta MAE \|z\|<1 | gamma MAE 1/$ | viol. > 1e-4 % | train min |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0012 ± 0.0001 | 0.0009 ± 0.0001 | 0.0223 ± 0.0028 | 0.0161 ± 0.0044 | 1.41 ± 0.30 | 1.83e-4 ± 3.6e-5 | 6.2e-4 ± 1.5e-4 | 0.73 ± 0.12 | 28 |
| `lambda_delta` 0.3 | 0.0009 ± 0.0001 | 0.0007 ± 0.0001 | 0.0263 ± 0.0099 | 0.0117 ± 0.0003 | 0.97 ± 0.19 | 2.00e-4 ± 3.8e-5 | 6.7e-4 ± 2.6e-4 | 0.55 ± 0.34 | 29 |
| `lambda_delta` 0 | 0.0015 ± 0.0001 | 0.0011 ± 0.0000 | 0.0417 ± 0.0092 | 0.0229 ± 0.0032 | 1.86 ± 0.05 | 1.21e-3 ± 1.9e-4 | 2.3e-3 ± 2.1e-4 | 0.83 ± 0.07 | 14 |
| `lambda_delta` 3 | 0.0014 ± 0.0004 | 0.0011 ± 0.0004 | 0.0328 ± 0.0133 | 0.0177 ± 0.0116 | 1.84 ± 0.53 | 1.78e-4 ± 1.6e-5 | 4.7e-4 ± 5.5e-5 | 0.83 ± 0.61 | 29 |
| softplus output | 0.0011 ± 0.0001 | 0.0010 ± 0.0002 | 0.0331 ± 0.0041 | 0.0256 ± 0.0116 | 0.80 ± 0.18 | 1.65e-4 ± 1.6e-5 | 4.4e-4 ± 1.1e-4 | 0.06 ± 0.06 | 25 |
| ReLU | 0.0311 ± 0.0088 | 0.0210 ± 0.0052 | 0.5425 ± 0.1607 | 0.4696 ± 0.2443 | 33.0 ± 13.0 | 5.69e-3 ± 5.3e-4 | 8.7e-3 ± 7.8e-4 | 8.54 ± 1.16 | 16 |
| L-BFGS 200 | 0.0011 ± 0.0001 | 0.0008 ± 0.0000 | 0.0235 ± 0.0022 | 0.0174 ± 0.0053 | 1.20 ± 0.12 | 1.73e-4 ± 2.7e-5 | 6.2e-4 ± 1.5e-4 | 0.73 ± 0.12 | 34 |
| shape 0.1 | 0.0015 ± 0.0003 | 0.0014 ± 0.0004 | 0.0217 ± 0.0076 | 0.0217 ± 0.0075 | 2.36 ± 1.01 | 1.96e-4 ± 1.9e-5 | 5.4e-4 ± 6.7e-5 | 0.47 ± 0.13 | 75 |
| `lambda_delta` 0.3 + softplus | 0.0009 ± 0.0001 | 0.0006 ± 0.0002 | 0.0349 ± 0.0240 | 0.0297 ± 0.0280 | 0.59 ± 0.04 | 1.56e-4 ± 1.5e-5 | 4.4e-4 ± 7.5e-5 | 0.02 ± 0.04 | 24 |

- **`lambda_delta` 0.3 prices best, at no measurable cost to delta.** Error near the money falls from 0.0012 to 0.0009 (about 3 baseline sd), the exercised-region maximum from 0.0161 to 0.0117, and the implied-vol error from 1.41 to 0.97 bp. Delta and gamma stay within one sd of the baseline. Its worst grid point spreads more across seeds (0.0263 ± 0.0099).
- **The delta term helps the price too.** Without it (`lambda_delta` 0), delta error is ~7x and gamma error ~4x the baseline's, and the price is ~25% worse. At 3 the price is also worse and delta no better, so the best weight is below 1.
- **Softplus nearly removes the violations, and costs the worst points.** Violations > 1e-4 fall from 0.73% to 0.06% (> 1e-5: 1.50% to 0.08%), with the lowest IV, delta and gamma errors. But the grid maximum rises from 0.0223 to 0.0331, the exercised-region maximum from 0.0161 to 0.0256, and the test MAE from 0.0008 to 0.0010. The labels' slightly negative premiums do not stop it from fitting on average.
- **ReLU fails** (~25x the error, 8.5% violations), which confirms SiLU. **The L-BFGS finish** gains within about 1 sd for 5 more minutes. **The shape term** is worse on price everywhere at ~2.6x the training time; softplus does far better on violations.
- Three seeds give a rough sd. Only the `lambda_delta` 0.3 price gain, the softplus violation drop and the losses of `lambda_delta` 0 and 3, ReLU and the shape term clearly exceed the seed-to-seed spread. For the combined variant, the price and IV gains against the baseline and the violation drop exceed it; its delta and gamma tie softplus alone, and its all-grid MAE ties `lambda_delta` 0.3 alone.
- **Combined, the mean-error gains add up.** `lambda_delta` 0.3 + softplus keeps the price of `lambda_delta` 0.3 (0.0009 near the money) and the violations of softplus (0.02% > 1e-4, 0.04% > 1e-5). It has the lowest IV error (0.59 ± 0.04 bp, against 0.97 for `lambda_delta` 0.3 and 0.80 for softplus, about 1 sd of the latter) and test MAE (0.0005, against 0.0008 for the baseline), and matches or slightly improves the all-grid MAE (0.0006) and delta (1.56e-4).
- **Its worst point depends on the seed.** Grid maximum per seed: 0.0620, 0.0264, 0.0163 (seeds 42, 43, 44). Seed 42's is in the exercised rows, hence the large spread of both maxima. The median seed (0.0264) is close to `lambda_delta` 0.3 alone (median 0.0243) and below softplus alone (0.0332), but above the baseline (0.0211). Softplus alone also has its worst point in the exercised rows on seed 42 (0.0372). The combined variant has the weakest mean worst point of the three (grid max 0.0349, against 0.0263 and 0.0331), and three seeds cannot tell whether seed 42 is typical.
- **Chosen as the final configuration** for its price, IV and violation gains, accepting the worst-point risk: the final model's three seeds should be compared on the grid maximum as well as the median.

## Dataset ablations: keep every exercised row; a `T >= 0.25` model is better there

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset --T_min 0.25 --output_dir artifacts/data/T0.25
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset --exercised_keep 0.25 --output_dir artifacts/data/exercised_0.25
python -m optionpricing.ml.experiments.run_ablations --variants exercised_0.25
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus T_from_0.25 --grid_T_min 0.25 --out_dir artifacts/runs/ablations_T0.25
```

Protocol: both variants train `lambda_delta_0.3_softplus` (seeds 42, 43, 44, 400 epochs) on a regenerated dataset of 2^19 rows (BBSR-1000 labels, Sobol seed 42), so the number of steps matches and only the data differs. `T_from_0.25`: `T` log-uniform on [0.25, 5] instead of [1 week, 5], so ~1.9x the rows per unit of log `T` in that range. `exercised_0.25`: a 2^20-row pool, every continuation row and every 4th exercised row, the first 2^19 kept: 8.9% of rows exercised instead of 28%. Measured 2026-11-01, $ at K = 100, mean ± sample sd over the 3 seeds. Retraining `lambda_delta_0.3_softplus` in the T run's `--out_dir` reproduced the earlier runs exactly (same best epochs, validation terms and worst points).

On the `T >= 0.25` grid (the default grid with its 8 maturities log-spaced over 0.25-5 years), `artifacts/runs/ablations_T0.25/summary.md`:

| trained on | \|z\|<1 MAE | all MAE | grid max | exercised max | IV err bp | delta MAE \|z\|<1 | viol. > 1e-4 % |
|---|---|---|---|---|---|---|---|
| `T` 1 week - 5 y | 0.0012 ± 0.0002 | 0.0008 ± 0.0002 | 0.0349 ± 0.0240 | 0.0297 ± 0.0280 | 0.64 ± 0.05 | 1.97e-4 ± 2.9e-5 | 0.02 ± 0.04 |
| `T` 0.25 - 5 y | 0.0008 ± 0.0001 | 0.0006 ± 0.0001 | 0.0128 ± 0.0022 | 0.0105 ± 0.0023 | 0.50 ± 0.02 | 1.41e-4 ± 2.5e-5 | 0.05 ± 0.04 |

On the full grid, against the `lambda_delta_0.3_softplus` row above, `artifacts/runs/ablations/summary.md`:

| exercised rows | \|z\|<1 MAE | all MAE | grid max | exercised max | IV err bp | delta MAE \|z\|<1 | viol. > 1e-4 % |
|---|---|---|---|---|---|---|---|
| all (28%) | 0.0009 ± 0.0001 | 0.0006 ± 0.0002 | 0.0349 ± 0.0240 | 0.0297 ± 0.0280 | 0.59 ± 0.04 | 1.56e-4 ± 1.5e-5 | 0.02 ± 0.04 |
| 1/4 kept (8.9%) | 0.0010 ± 0.0002 | 0.0008 ± 0.0001 | 0.0445 ± 0.0079 | 0.0445 ± 0.0079 | 0.65 ± 0.04 | 1.43e-4 ± 1.1e-5 | 0.06 ± 0.06 |

- **Covering short maturities costs the longer ones accuracy.** On `T >= 0.25`, the narrow model's near-the-money error is lower on every seed (per seed 0.00071-0.00085 for the `T >= 0.25` model, 0.00100-0.00135 for the wide one, no overlap; likewise 1 <= \|z\| < 2), and its worst point stays at 0.0105-0.0148 where the wide model's is 0.0163-0.0620. Part of the gain may come from its higher row density there, though the learning curve above found the row count barely matters at equal steps.
- **The range stays 1 week - 5 years**, for simplicity: a `T >= 0.25` model would route every shorter put to the tree or need a second network, and the wide model is most accurate in dollars at short maturities: on the full grid (puts not exercised today, 3 seeds, `artifacts/runs/ablations/summary.md`), a mean error of $0.00014 ± 0.00006 under 1 month and $0.00022 ± 0.00004 at 1-3 months, against $0.00098 ± 0.00013 beyond 3 months.
- **Thinning the exercised rows does not help.** The continuation error does not fall (near the money, per seed, 0.00075-0.00113 thinned and 0.00076-0.00092 with every exercised row), and the worst point, in the exercised rows on every seed, rises to 0.038-0.053. The thinned runs stopped early at ~338 of 400 epochs (best epochs 309-319) as their validation loss stopped improving, so they took fewer steps than the reference. Every exercised row stays.

## The candidate on the full grid: `lambda_delta_0.3_softplus`

```
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus lambda_delta_0.3_softplus_width_256 --seeds 42 43 44 45 46 --out_dir artifacts/runs/ablations_b036   # ~2.5 h, 4 runs at a time
# one model on its own:
python main.py train-ml --lambda_delta 0.3 --output softplus --seed 43   # ~25-30 min
python main.py test-ml --seed 43
```

The chosen configuration, at width 128 (the width ablation, "Width: 256 units do not beat 128" below, kept it). From the 5 width-128 runs of the width ablation (`artifacts/runs/ablations_b036/lambda_delta_0.3_softplus/`, measured 2026-11-10), trained with the softplus output's bias term at the mean premium ("Output initialization" below); the final model's numbers replace them in the README once it is chosen.

**The chosen configuration** is 4 hidden layers of 128 SiLU units with a
softplus output (the premium is never negative), trained on the premium error
plus 0.3 times the delta error, with Adam and a cosine decay over 400 epochs.
Each choice was tested against its alternatives: the network and data
choices over 3 or 5 seeds, the training length and the teacher tree in single
studies (the sections above and below):

| Choice | Tested against | Evidence |
|---|---|---|
| Delta error in the loss, weight 0.3 | 0, 1, 3 | best price; without it, delta is ~7x worse |
| Softplus output | linear | shape violations 0.73% -> 0.06% |
| SiLU activations | ReLU | ReLU: ~25x the error, and its gamma is 0 almost everywhere (piecewise linear) |
| Width 128 | 256 (5 seeds) | 256 within 0.6 seed sd on every error (the larger of the two sds), at ~2.6x the training time and ~3x the batch price |
| No shape penalty, no L-BFGS finish | shape weight 0.1, 200 L-BFGS steps | worse or within noise, and slower |
| 400 epochs | 200, 800 (one seed) | 800 lowers the delta error but raises the price error |
| BBSR teacher tree, 1000 steps | CRR, BBS; 500 and 2000 steps | lowest mean price error; 2000 steps halve the maximum at 3.5x the cost |
| Every exercised-today row kept | 1/4 of them, refilled with other puts | no gain, worst exercised error $0.030 -> $0.045 |
| One model from 1 week to 5 years | `T` from 3 months only | simpler; see "Why one model from 1 week to 5 years" below |

Scored against an independent 4000-step BBSR tree on a factorial
grid of 2342 puts (27 `z` x 6 `σ` x 8 maturities from 1 week to 5 years x
`r` in {2%, 6%}, inside the moneyness cap). Mean ± standard deviation over
seeds 42-46, $ at K = 100; the implied-vol error is how far `σ` would
have to move for the true price to reach the surrogate's:

| Puts | Points | Mean error $ | Worst error $ | Implied-vol error (bp) | Delta error |
|---|---|---|---|---|---|
| `T` < 1 month | 494 | 0.00008 ± 0.00002 | 0.0012 ± 0.0002 | 0.28 ± 0.06 | 5.5e-5 |
| `T` 1-3 months | 455 | 0.00015 ± 0.00001 | 0.0014 ± 0.0004 | 0.31 ± 0.06 | 4.0e-5 |
| `T` >= 3 months | 690 | 0.00090 ± 0.00007 | 0.0166 ± 0.0034 | 0.61 ± 0.05 | 1.4e-4 |
| exercised today | 703 | 0.00045 ± 0.00007 | 0.0114 ± 0.0032 | - | 9.4e-5 |
| all | 2342 | 0.00045 ± 0.00004 | 0.0166 ± 0.0034 | 0.44 ± 0.04 | 8.9e-5 |

The first three rows are the puts not exercised today. By distance to the
strike, those within one standard deviation (`|z| < 1`, 646 points) have a
mean error of $0.00079 ± 0.00003 and 0.33 bp, those between 1 and 2 standard
deviations $0.00035, and those beyond $0.00007. The grid's worst errors
understate the true worst case, which lies at rates near zero, outside the
grid (see "The chosen model near zero rates" below).

- **The typical error is a fraction of a cent**: $0.00045 on average, 0.44 bp
  of implied volatility, far below a listed option's tick.
- **Short maturities are the most accurate in dollars**: $0.00008 on average
  under 1 month, $0.00015 at 1-3 months, against $0.00090 beyond 3 months.
- **The worst point is near the money, beyond 3 months, on every seed**:
  $0.0155, $0.0126, $0.0217, $0.0154 and $0.0179 for seeds 42-46, each in the
  `|z| < 1`, `T` >= 3 months bucket. The worst exercised-today error is
  $0.0069-0.0155.
- **No-arbitrage shape**: on the puts not exercised today, the raw
  network's price decreases and is convex in `m` and increases with `σ` at
  every grid point for every seed, and increases with `T` at all but
  0-0.12% of them (counting violations above 1e-4 K).
- **The payoff floor** answers about half of the exercised-today points
  (49% ± 9%), where the true price is the payoff itself, and under 1% of
  the others.
- **Speed**, timed by each run's evaluation while other runs shared the CPU,
  so it depends on the load: 4.5-5.4 µs per option when the grid is priced
  in one call, against 1.5-1.6 ms for the 1000-step tree it learned from
  (~300x), on seeds 45 and 46, which finished with fewer runs in parallel
  (7.5-10.5 µs against 2.6-2.8 ms for seeds 42-44, with 4 runs at a time).
  One option per call takes 1.0-2.9 ms (the autograd passes for delta and
  gamma dominate), no faster than the tree, so the gain needs batches.

**Why one model from 1 week to 5 years: simplicity.** A dataset ablation
trained the same configuration on puts with `T` >= 3 months only (also 2^19
rows) and scored both models on a `T` >= 3-month grid ("Dataset ablations"
above). The narrower model is more accurate there: near the money $0.0008
for the narrow model and $0.0012 for the wide one, a worst point of $0.013
and $0.035, 0.50 and 0.64 bp, and its worst point varies much less across
the 3 seeds ($0.011-0.015 for the narrow model, $0.016-0.062 for the wide
one). Covering short maturities therefore costs the longer ones some
accuracy, though part of the gap may come from the narrow model's ~1.9x
more rows per unit of log `T` in that range. The single model is kept
anyway. A model from 3 months would send every shorter put to the tree, at
milliseconds instead of microseconds, or need a second network with its own
training, checkpoint and routing rule. One network over the whole range is
simpler to train, ship and use, and it stays accurate at short maturities,
its most accurate region in dollars (table above). The accepted cost: beyond
3 months, the wide model's error is 1.3-2.7x what a dedicated model reaches
(1.3x on average over that grid, 1.5x near the money, 2.7x on the worst
point). Both models in that comparison trained before the output-bias fix
("Output initialization" below); the fixed wide model's worst point beyond 3
months is $0.013-0.022 on the full grid (table above), so the worst-point gap
is likely smaller now. Not re-measured on the `T` >= 3-month grid.

## The chosen model near zero rates

Protocol: each of the 3 `lambda_delta_0.3_softplus` checkpoints of `artifacts/runs/ablations/` (seeds 42-44, trained before the output-bias fix, "Output initialization" below; not re-measured on the fixed runs) priced its own test split (104858 rows, `split_ratio` 0.8 and the run's seed) through `PutSurrogate.price`, against the teacher's labels (Black-Scholes plus the dataset's premium, as `evaluate_test_split` rebuilds them), grouped by `r`; measured 2026-11-01 with an ad hoc script that is not in the repo (`PutSurrogate.load` on each run's `model.pth`, `TreePutDataset(..., is_train=False, seed=<run seed>, stats=surrogate.stats)`), $ at K = 100, ranges over the 3 seeds.

| `r` | rows | MAE | max | share > $0.01 |
|---|---|---|---|---|
| < 0.002 | ~2600 | 0.0015-0.0020 | 0.096-0.111 | 3.0-4.7% |
| 0.002-0.005 | ~3900 | 0.0005-0.0007 | 0.014-0.018 | 0.2-0.4% |
| 0.005-0.02 | ~19600 | 0.0004-0.0005 | 0.014-0.017 | 0.02-0.04% |
| 0.02-0.08 | ~78700 | 0.0004-0.0006 | 0.038-0.044 | 0.12-0.16% |

- **The largest errors sit at the domain's low rate edge**, on long-dated (`T` 1.6-5 years) puts deep in the money (`m` 0.42-0.76), at `sigma` 0.06-0.22: all 12 worst test rows of seed 43 have `r` < 0.0005. The grid's rates (0.02 and 0.06) never reach this edge, so its maxima understate the worst case.
- Not investigated, and not planned: rates are not expected back near zero soon, so this edge is accepted as a known limitation. The cause (the net at the domain edge, the raw `r` feature, the labels near `r = 0`) is unknown.

## Training cost by network width

Measured in the width ablation (below), 4 runs at a time with 2 threads each: 29.3 ± 3.2 min per run at width 128 and 76.2 ± 7.6 min at width 256, ~2.6x, as the per-step timing predicts.

Protocol: one training step (premium + 0.3 x delta loss, `labeled_loss`, then an Adam step) on a random batch of 1024 rows, SiLU, softplus output, `torch.set_num_threads(2)`, timed over 200 steps after 20 warm-up steps in one process; an ad hoc script, not in the repo, measured 2026-11-01. The width-128 figure matches the real runs: ~24 min for 400 epochs of 328 steps (`lambda_delta_0.3_softplus` above), ~11 ms per step with 3-4 runs at a time.

| network | weights | ms per step | estimated run (400 epochs) |
|---|---|---|---|
| depth 4, width 128 | 50561 | 12.5 | ~25 min |
| depth 4, width 256 | 199425 | 31.4 | ~60 min |
| depth 4, width 512 | 792065 | 113.5 | ~3.5-4 h |
| depth 6, width 128 | 83585 | 18.7 | ~37 min |

## Output initialization: start at the mean premium

For the softplus output, `initial_model` in `optionpricing/ml/surrogate/tree/tree_training.py` sets the **bias term of the output layer** (the last `nn.Linear`, just before the `nn.Softplus`) to `log(expm1(ȳ))`, where `ȳ` is the mean premium of the trained rows. Every other weight and bias term keeps PyTorch's random initialization, and the linear output keeps PyTorch's initialization entirely. This section explains where that value comes from and why only the softplus output needs it. The failure it fixes is `.claude/BUGS.md`, "Width-256 softplus surrogate stopped learning in its first epoch".

**1. What "the network at initialization" means.** Before training, PyTorch fills each `nn.Linear` layer with random numbers: the weights and the bias terms are drawn uniformly from `±1/√fan_in`, where `fan_in` is the layer's number of inputs. For the output layer `fan_in` is the width, so the range is `±0.088` at width 128 and `±0.0625` at width 256. These initial parameters define a function `f₀(x)`, as any weights do. The initial loss is the loss of `f₀` on the training rows, before the first Adam step. It does not involve a first data point.

**2. At initialization the network is nearly a constant.** The output layer computes

```
f₀(x) = softplus( w·h(x) + b )        softplus output
f₀(x) =           w·h(x) + b          linear output
```

where `h(x)` holds the last hidden layer's activations, `w` the output layer's weights and `b` its bias term. With small random `w`, the part `w·h(x)` that depends on the input varies little from row to row. On 16384 training rows (seeds 0-9, widths 128 and 256), the standard deviation of `f₀(x)` across rows is at most 0.0066 per unit of strike, against 0.029 for the premiums. So `f₀` is close to one constant `c`, the mean of `f₀(x)` over the rows.

**3. What a constant prediction costs.** The premium term of the loss is the premium MSE divided by `var(y)`, the variance of the trained rows' premiums `y` (`LossScales`). For a constant prediction `c`, the MSE splits exactly into the variance of the labels plus the squared offset of `c` from their mean:

```
mean_i (c − y_i)²  =  var(y) + (c − ȳ)²
premium term       =  1 + (c − ȳ)² / var(y)
```

This is smallest at `c = ȳ`, where it equals 1. So among the functions the initial network can be (close to constants), the one at the mean premium has the lowest loss. That is also the value the loss scaling is built around: predicting the mean scores 1.

**4. With the default bias term, the softplus output starts far from the mean.** The softplus net predicts `c ≈ softplus(b)`, between 0.65 and 0.72 per unit of strike for the random `b` here (softplus(0) = ln 2 = 0.693), against `ȳ = 0.0116`. Step 3 then gives a premium term of `1 + (0.69 − 0.0116)² / 0.0294² ≈ 530`, which is what is measured (table below). Adam pushes every output down at once, and the momentum built up by these large first gradients carries the outputs past the target, into the region where the softplus is flat: its slope, `sigmoid` of its input, is ~1e-7 at an input of −16. There the gradients vanish and the outputs never come back. At width 256, all 5 seeds of the width ablation froze in their first epoch; in a step-by-step replay (seeds 42 and 43, second table below) the outputs were in the flat region by step 17.

**5. The bias term that starts it at the mean.** Solving `softplus(b) = ȳ`, with `softplus(b) = log(1 + e^b)`, gives

```
b = log(e^ȳ − 1) = log(expm1(ȳ))
  = log(expm1(0.0116)) = −4.45        (seed 42's training rows)
```

implemented in `initial_model` as

```python
output_layer.bias.fill_(math.log(math.expm1(mean)))
```

with `mean` the mean premium of the trained rows. The initial premium term is then ~1 for any width, depth, activation and seed, because the argument of step 3 does not depend on them. The delta term does not change: a constant has zero slope, so the delta term at initialization comes from the hidden layers. `initial_model` raises `ValueError` if the mean premium is not positive, since no bias term gives `softplus(b) ≤ 0`.

**Why the linear output keeps PyTorch's default.** Its constant is `c ≈ b`, the random bias term itself, between −0.09 and +0.06 here. That gives a premium term between ~1 and ~12, depending on the seed. A high start does not trap the linear output: it has no flat region, so its gradients do not vanish and the first steps bring the loss down. Keeping its initialization also keeps every recorded linear-output ablation run reproducible.

**Why not a lower learning rate or a warm-up.** Both also avoid the collapse at width 256 (second table below). But they treat the symptom: the start stays ~500x too high and is corrected more slowly, and each adds a setting that may need tuning per width. The bias term removes the bad start itself and adds no setting. After 2 epochs its loss is ~10x lower than the default bias term's at width 128, and ~20-25x lower than the warm-up's at width 256.

### Measured

Initial loss terms. Protocol: the training split of `artifacts/data/tree_put_dataset.npz` (`split_ratio` 0.8, seed 42, 419430 rows); each term computed with `labeled_loss` on the first 16384 of those rows, scaled by the full training rows' `LossScales`; depth 4, SiLU, seeds 0-9; an ad hoc script, not in the repo, measured 2026-11-06. On every net the measured premium term is within 0.17 of the formula `1 + (c − ȳ)² / var(y)` of step 3, computed with that net's own `c`. The bias-term-at-the-mean rows read 0.97-0.98 rather than 1 because the variance of these 16384 rows around `ȳ` is 0.975 times the full training rows'.

| output | width | output layer's bias term | `c`, per unit of strike | premium term, min / median / max | delta term |
|---|---|---|---|---|---|
| softplus | 128 | PyTorch default | 0.651 to 0.721 | 475 / 513 / 584 | 1.14-1.44 |
| softplus | 128 | **`log(expm1(ȳ))`** | 0.0113 to 0.0120 | **0.97 / 0.97 / 0.98** | 1.20 |
| softplus | 256 | PyTorch default | 0.661 to 0.722 | 490 / 523 / 585 | 1.17-1.31 |
| softplus | 256 | **`log(expm1(ȳ))`** | 0.0114 to 0.0118 | **0.97 / 0.98 / 0.98** | 1.20 |
| linear | 128 | PyTorch default (kept) | −0.087 to 0.054 | 0.94 / 4.14 / 12.16 | 1.15-1.93 |
| linear | 256 | PyTorch default (kept) | −0.065 to 0.056 | 1.09 / 3.00 / 7.66 | 1.14-1.48 |

The first 2 epochs of training. Protocol: the `lambda_delta_0.3_softplus` settings (`lambda_delta` 0.3, batch 1024, Adam, a constant learning rate of 1e-3 unless stated), 656 steps replayed one by one on the same training split, seeds 42 / 43, 2 torch threads; "largest output before the softplus" is over 4096 held-out rows; an ad hoc script, not in the repo, measured 2026-11-06.

| config | outputs collapse into the flat region? | largest output before the softplus, step 640 | loss at step 640 |
|---|---|---|---|
| width 128, PyTorch default bias term | nearly: below −5 from step ~48 to ~100 / ~140, then recovers | −1.2 / −1.0 | 7.3e-2 / 9.6e-2 |
| width 256, PyTorch default bias term | **yes, by step 17; never recovers** | −12.2 / −12.7 | 1.75 / 1.92 |
| width 256, learning rate 3e-4 | no | −1.2 / −1.0 | 0.11 / 0.15 |
| width 256, linear warm-up over epoch 1 | no | −1.2 / −1.0 | 6.5e-2 / 0.11 |
| width 128, bias term at `log(expm1(ȳ))` | no | −1.2 / −1.1 | **6.9e-3 / 9.1e-3** |
| width 256, bias term at `log(expm1(ȳ))` | no | −1.2 / −1.0 | **2.6e-3 / 4.9e-3** |

**Consequence for the results above.** The softplus-output runs of the seeded and dataset ablations (`softplus_output`, `lambda_delta_0.3_softplus` in `artifacts/runs/ablations/` and the dataset variants) trained with the default bias term and recovered from a near-collapse. Their numbers are valid as measured but do not reproduce from the current code. The width-256 rows of `artifacts/runs/ablations/summary.md` are failed runs, not a measurement of what a wider network can do. "The candidate on the full grid" and the width ablation below use runs trained with the fix.

**What is checked** (`tests/unit/test_tree_training.py`):
- `test_softplus_output_starts_at_the_mean_premium`: the bias term equals `log(expm1(ȳ))` and the initial premium term is 1 ± 0.1, at widths 8 and 256, seeds 0 and 1;
- `test_linear_output_keeps_pytorchs_default_initialization`: the linear net's parameters are exactly PyTorch's seeded ones;
- `test_softplus_start_needs_a_positive_mean_premium`: zero, negative or no premiums raise `ValueError` for the softplus output, not for the linear one;
- `test_training_starts_the_softplus_output_at_the_trained_rows_mean_premium`: `train_surrogate` starts from this bias term, computed on its trained rows.

## Width: 256 units do not beat 128

```
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus lambda_delta_0.3_softplus_width_256 --seeds 42 43 44 45 46 --out_dir artifacts/runs/ablations_b036   # ~2.5 h, 4 runs at a time
```

Protocol: `lambda_delta_0.3_softplus` at depth 4, width 128 (50561 weights) or 256 (199425), everything else equal, both with the softplus bias term at the mean premium; 2^19 rows, 400 epochs, seeds 42-46, 4 runs at a time with 2 threads each. Same grid as above (BBSR-4000, 2342 puts), $ at K = 100, mean ± sample sd over the 5 seeds. Measured 2026-11-10. Full table: `artifacts/runs/ablations_b036/summary.md`.

| width | \|z\|<1 MAE | all MAE | grid max | exercised max | IV err bp | `T` >= 3 months MAE | `T` >= 3 months IV bp | delta MAE \|z\|<1 | viol. > 1e-4 % | train min |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.0008 ± 0.0000 | 0.0004 ± 0.0000 | 0.0166 ± 0.0034 | 0.0114 ± 0.0032 | 0.44 ± 0.04 | 0.0009 ± 0.0001 | 0.61 ± 0.05 | 1.34e-4 ± 1.2e-5 | 0.07 ± 0.05 | 29 |
| 256 | 0.0007 ± 0.0001 | 0.0005 ± 0.0001 | 0.0160 ± 0.0029 | 0.0130 ± 0.0038 | 0.43 ± 0.09 | 0.0008 ± 0.0002 | 0.57 ± 0.10 | 1.27e-4 ± 1.6e-5 | 0.06 ± 0.09 | 76 |

- **Width 256 is within 0.6 seed sd on every error**, measured against the larger of the two configurations' sds. The decision rule (adopt 256 only if it beats 128 by more than ~2 seed sd on the near-the-money error, the implied-vol error or the worst point) is not met. Width 256 against width 128: near the money $0.0007 (256) and $0.0008 (128), implied-vol error 0.43 bp (256) and 0.44 bp (128), worst point $0.0160 (256) and $0.0166 (128). Beyond 3 months, where the wider net was expected to help most: $0.0008 (256) and $0.0009 (128), 0.57 bp (256) and 0.61 bp (128), again within noise. Near the money the gap ($0.00071 ± 0.00014 at width 256, $0.00079 ± 0.00003 at width 128) is 0.6 of width 256's sd but 2.4 of width 128's: it comes from width 256's seeds 42-44 only, and its seeds 45 and 46 give it back (per seed below). Measured against the larger sd, as the rule requires, it stays well under 2.
- **Its spread across seeds is about twice as wide** (per seed below). Width 256's seeds 42-44 have the lowest near-the-money errors measured ($0.00058-0.00064, against $0.00074-0.00084 for every width-128 seed). Its seeds 45 and 46 stopped early on patience, at 354 and 366 of 400 epochs, and are no better than width 128 ($0.00080-0.00089 near the money, 0.42-0.57 bp). Early stopping hit only width 256: every width-128 run trained all 400 epochs.
- **It costs ~2.6x the training time and ~3x the batch price**: 76 min per run at width 256 and 29 min at width 128; 22-27 µs per option at width 256 and 4.5-10.5 µs at width 128, in the runs' own timing (both under load).
- **Width 128 stays.** 400 epochs were chosen at width 128, so a longer schedule could favour width 256; not tested.

Per seed, on the grid ($ at K = 100):

| width | seed | \|z\|<1 MAE | IV err bp | grid max | epochs trained | best epoch |
|---|---|---|---|---|---|---|
| 128 | 42 | 0.00080 | 0.46 | 0.0155 | 400 | 386 |
| 128 | 43 | 0.00079 | 0.43 | 0.0126 | 400 | 392 |
| 128 | 44 | 0.00077 | 0.39 | 0.0217 | 400 | 399 |
| 128 | 45 | 0.00084 | 0.49 | 0.0154 | 400 | 398 |
| 128 | 46 | 0.00074 | 0.42 | 0.0179 | 400 | 396 |
| 256 | 42 | 0.00061 | 0.39 | 0.0139 | 400 | 393 |
| 256 | 43 | 0.00058 | 0.41 | 0.0124 | 400 | 398 |
| 256 | 44 | 0.00064 | 0.35 | 0.0163 | 400 | 399 |
| 256 | 45 | 0.00089 | 0.57 | 0.0199 | 354 | 334 |
| 256 | 46 | 0.00080 | 0.42 | 0.0176 | 366 | 346 |

**The fix improves width 128.** Against the 5 runs trained before it (`artifacts/runs/ablations/summary.md`, seeds 42-46, mean ± sample sd):

| width-128 runs | \|z\|<1 MAE | all MAE | grid max | exercised max | IV err bp | viol. > 1e-4 % |
|---|---|---|---|---|---|---|
| default bias term | 0.0009 ± 0.0001 | 0.0007 ± 0.0002 | 0.0285 ± 0.0193 | 0.0235 ± 0.0216 | 0.60 ± 0.03 | 0.01 ± 0.03 |
| bias term at the mean premium | 0.0008 ± 0.0000 | 0.0004 ± 0.0000 | 0.0166 ± 0.0034 | 0.0114 ± 0.0032 | 0.44 ± 0.04 | 0.07 ± 0.05 |

- The implied-vol error falls by ~4 seed sd (the larger sd, 0.04 bp; before the fix 0.60 bp, with it 0.44 bp) and the all-grid error from $0.0007 (before) to $0.0004 (with). The near-the-money error barely moves.
- The worst point falls on average and, above all, stops depending on the seed: before the fix seed 42's was an exercised-today put at $0.062; with it, every seed's worst point is a continuation put near the money beyond 3 months, $0.013-0.022.
- Violations above 1e-4 rise slightly (before the fix 0.01%, with it 0.07%, all `increasing_in_T` on continuation points), ~1.2 of the larger seed sd (0.05%).
