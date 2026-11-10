# How the tree surrogate was chosen

This page explains how the American-put surrogate's final configuration was found: what was tested, which choice won, and why. Every choice was measured, most of them over several training seeds. The results were measured 2026-10-22 to 2026-11-10 and are **not re-verified** since. The commands that produce them are at the end.

**Units.** Price errors are in **10⁻⁴ $ at K = 100**, a hundredth of a cent: 7.9 means $0.00079. Delta errors are in **10⁻⁵**. Implied-vol errors are in **basis points** (bp) of volatility. "± x" is the sample standard deviation over seeds.

## The decisions at a glance

| Decision | Tested against | Deciding evidence | Section |
|---|---|---|---|
| BBSR teacher tree, 1000 steps | CRR and BBS, 500 and 2000 steps | lowest mean price error, and 2000 steps halve the worst error at 3.5× the cost | Choice of teacher |
| Sample `z`, not moneyness | log-uniform `m` | `m` sampling wastes 44% of rows on puts exercised today | Sampling |
| 400 epochs | 200, 800 | more steps, not more data, lowered the error, and 800 traded price for delta | Training length |
| Delta error in the loss, weight 0.3 | 0, 1, 3 | best price, and without the delta term delta is ~7× worse | Seeded ablations |
| Softplus output | linear | no-arbitrage violations fall from 0.73% (linear) to 0.06% (softplus), at a worse worst point before the fix | Seeded ablations |
| SiLU activations | ReLU | ReLU has ~25× the error and no gamma | Seeded ablations |
| No shape penalty, no L-BFGS finish | penalty weight 0.1, 200 L-BFGS steps | worse or within noise, and slower | Seeded ablations |
| Every exercised-today row kept | a quarter of them | no gain, worse worst point | Dataset ablations |
| One model from 1 week to 5 years | a model from 3 months | simpler, at 1.3-2.7× the error beyond 3 months | Dataset ablations |
| Softplus bias term at the mean premium | PyTorch's default, a lower learning rate, a warm-up | removes a near-collapse in the first epoch | Output initialization |
| Width 128 | 256 | 256 within 0.6 seed sd on every error, at 2.6× the training time | Width |
| Ship seed 43 | retraining | the width runs already are the final configuration, and 43 is the median seed | The chosen model |

## How a model is scored

Every model is scored on a factorial grid of 2342 puts: 27 values of `z` × 6 of `σ` × 8 maturities from 1 week to 5 years × `r` in {2%, 6%}. The reference price is a 4000-step BBSR tree, not the 1000-step teacher. So a score measures distance to the true price, not how well the network copies its teacher's own errors. This is `evaluate_grid` in `optionpricing/ml/surrogate/tree/tree_evaluation.py`, which `test-ml` and the ablation runner both print.

| Measure | Question it answers |
|---|---|
| Mean error near the money (`\|z\|` < 1) | How far off is a typical price where options trade most? This is the headline number. |
| Mean error by maturity and over the whole grid | Is any region weak? One overall average would hide it behind easy, nearly worthless puts. |
| Worst error on the grid | What is the largest miss a user can hit? A worst point that moves a lot between seeds also shows an unstable fit. |
| Implied-vol error | How large is the error in the units traders quote options in? |
| Delta error | Can the surrogate's delta be used to hedge? |
| No-arbitrage violation rate | Does the price fall and stay convex in `m`, and rise with `T` and `σ`, as any put price must? |
| Time per option | Is the surrogate faster than the tree? Speed is its only purpose. |

**The implied-vol error.** The implied volatility of a price is the `σ` that makes the pricing model return that price. The implied-vol error asks how far `σ` would have to move for the true price to equal the surrogate's. To first order, at each grid point:

```
IV error = |surrogate price − reference price| / vega
```

Each point uses its own vega, and the results are then averaged over a bucket. In the code (`bucket_metrics`):

```python
row.update(n_iv=int(sel.sum()), iv_mae=_mean(abs_err[sel] / vega[sel]))
```

It puts errors on options of very different prices on one scale. The same $0.0008 error is 0.21 bp on a 1-year at-the-money put but 1.45 bp on a 1-week one, which has 7× less vega (Black-Scholes, `σ` = 20%, `r` = 4%). For scale, a one-cent price tick on the 1-year put is 2.6 bp. Points with vega below $0.01 per vol point (`MIN_IV_VEGA`) and puts exercised today are left out, since their price barely depends on `σ`.

**Mean absolute error, not squared error.** Training minimizes a squared error because it suits an optimizer. The report uses the mean absolute error because it is in dollars and is the typical miss. The root mean squared error would be pulled up by the few largest errors, which the worst-error column already shows.

**The 2 sd rule.** A run's result depends on its seed (initialization, batch order, data split). So each configuration is trained on several seeds. A difference between two configurations counts as real only if it exceeds about 2 seed standard deviations. The rule uses the larger of the two configurations' standard deviations, so a configuration with widely spread results cannot win on its lucky seeds.

## Choice of teacher: BBSR at 1000 steps

Protocol: 120 seeded random puts over the dataset's domain, plus 21 in a hard corner (`m` 0.95-1.05, `T` = 0.25, `σ` = 0.05). The reference is BBSR at 20000 steps. Errors at 1000 steps:

| Method | Price error, worst / mean (10⁻⁴ $) | Delta error, worst / mean (10⁻⁵) |
|---|---|---|
| CRR | 55 / 5.8 | 22 / 1.7 |
| BBS | 21 / 3.6 | 29 / 1.5 |
| BBSR | 32 / 0.96 | 83 / 1.4 |

- **BBSR has the lowest mean price error**, about 4× below BBS and 6× below CRR. Its worst error, 32, is far below what the network reaches.
- **It does not zig-zag.** CRR's price moves by 87 between 1000 and 1001 steps. BBSR's moves by 1.5 between 1000 and 1002.
- **2000 steps** halve the worst error at 3.5× the cost, which the network's own error does not justify. At 1000 steps a put takes 2.7 ms, so the teacher does not limit the dataset size.
- The tree's labels can dip slightly below Black-Scholes (by up to 2.7 in these units), which matters for the softplus output.

## Sampling: z instead of moneyness

Sampling the moneyness `m` log-uniformly on (0.4, 1.6) puts 66% of the puts in the money, and 44% of all rows would be exercised today, with a price known in closed form. The generator samples `z = ln(m) / (σ√T)`, the distance to the strike in standard deviations, instead (`map_to_domain` in `optionpricing/ml/surrogate/tree/generate_tree_dataset.py`).

## Training length: 400 epochs

Protocol: the configuration of the time (`lambda_delta` 1, linear output), seed 42, one run each.

**More data first looked like the answer.** With 200 epochs, every doubling of the dataset roughly halved the error:

| Rows | Error near the money | Error over the grid | IV error (bp) | Training time |
|---|---|---|---|---|
| 2^15 | 331 | 236 | 33.5 | 0.6 min |
| 2^16 | 174 | 125 | 19.4 | 1.3 min |
| 2^17 | 94 | 53 | 7.7 | 2.5 min |
| 2^18 | 37 | 28 | 5.0 | 5.0 min |
| 2^19 | 22 | 14 | 2.5 | 9.9 min |

**But steps, not data, were the limit.** At a fixed number of epochs, a bigger dataset also gets more optimizer steps, and every run was still improving at its last epoch. Separating the two:

| Run | Error near the money | Error over the grid | IV error (bp) |
|---|---|---|---|
| 2^19 rows, 200 epochs | 22 | 14 | 2.46 |
| 2^18 rows, 400 epochs (same steps) | 18 | 14 | 2.30 |
| 2^19 rows, 400 epochs | 12 | 11 | 1.75 |

Half the data with the same steps is as good. Twice the steps cuts the error near the money by ~45%. So the dataset stays at 2^19 rows and training gets longer.

**800 epochs went too far.** Its error near the money was 19, against 12 at 400 epochs. Past 400 epochs the delta error kept falling (−28%) while the price error rose (+38%), and the loss, which sums both, picked that trade. So training runs 400 epochs.

## Seeded ablations: the loss, the output and the activation

Each variant changes one setting of the configuration of the time (`lambda_delta` 1, linear output, SiLU, no L-BFGS finish, no shape penalty). The combined variant joins the two winners. Protocol: 2^19 rows, 400 epochs, seeds 42-44, 4 runs at a time. Three seeds give only a rough sd, and several gaps below are within 1-2 sd. These runs trained before the output-initialization fix below.

| Variant | Error near the money | Error over the grid | Worst error | IV error (bp) | Delta error near the money | Violations (%) | Training (min) |
|---|---|---|---|---|---|---|---|
| baseline (`lambda_delta` 1) | 12 ± 1 | 9 ± 1 | 223 ± 28 | 1.41 ± 0.30 | 18 ± 4 | 0.73 ± 0.12 | 28 |
| `lambda_delta` 0.3 | 9 ± 1 | 7 ± 1 | 263 ± 99 | 0.97 ± 0.19 | 20 ± 4 | 0.55 ± 0.34 | 29 |
| `lambda_delta` 0 | 15 ± 1 | 11 ± 0 | 417 ± 92 | 1.86 ± 0.05 | 121 ± 19 | 0.83 ± 0.07 | 14 |
| `lambda_delta` 3 | 14 ± 4 | 11 ± 4 | 328 ± 133 | 1.84 ± 0.53 | 18 ± 2 | 0.83 ± 0.61 | 29 |
| softplus output | 11 ± 1 | 10 ± 2 | 331 ± 41 | 0.80 ± 0.18 | 17 ± 2 | 0.06 ± 0.06 | 25 |
| ReLU | 311 ± 88 | 210 ± 52 | 5425 ± 1607 | 33.0 ± 13.0 | 569 ± 53 | 8.54 ± 1.16 | 16 |
| L-BFGS finish, 200 steps | 11 ± 1 | 8 ± 0 | 235 ± 22 | 1.20 ± 0.12 | 17 ± 3 | 0.73 ± 0.12 | 34 |
| shape penalty 0.1 | 15 ± 3 | 14 ± 4 | 217 ± 76 | 2.36 ± 1.01 | 20 ± 2 | 0.47 ± 0.13 | 75 |
| **`lambda_delta` 0.3 + softplus** | **9 ± 1** | **6 ± 2** | 349 ± 240 | **0.59 ± 0.04** | **16 ± 2** | **0.02 ± 0.04** | 24 |

- **A delta weight of 0.3 prices best.** It lowers the error near the money from 12 (weight 1) to 9 (weight 0.3), about 3 baseline sd, at no cost to delta. Without the delta term, delta is ~7× worse and the price ~25% worse, so the term helps the price too. A weight of 3 is worse again.
- **The softplus output nearly removes the violations**, from 0.73% (linear output) to 0.06% (softplus), with the lowest delta and IV errors of the one-change variants. It forces the premium to be positive, and the labels' tiny negative dips do not stop it from fitting. Its cost was the worst point, which rose from 223 (linear) to 331 (softplus). The decision accepted that risk, and the output-initialization fix below later removed it.
- **ReLU fails**, with ~25× the error. It is piecewise linear, so its gamma is 0 almost everywhere.
- **The L-BFGS finish and the shape penalty stay off.** The finish gains less than 1 sd for 5 more minutes. The penalty is worse on price everywhere at 2.6× the training time.
- **Combined, the gains add up**: the price of `lambda_delta` 0.3, the violations of softplus and the lowest IV error. Its worst point depended on the seed (from 163 to 620), which the output-initialization fix later removed.

## Dataset ablations: every exercised row, one model from 1 week to 5 years

Both variants train the combined configuration on a regenerated dataset of 2^19 rows, so only the data differs. Seeds 42-44, before the output-initialization fix.

**A model from 3 months is more accurate beyond 3 months.** Scored on a grid with maturities from 3 months to 5 years:

| Trained on `T` | Error near the money | Error over the grid | Worst error | IV error (bp) |
|---|---|---|---|---|
| 1 week to 5 years | 12 ± 2 | 8 ± 2 | 349 ± 240 | 0.64 ± 0.05 |
| 3 months to 5 years | 8 ± 1 | 6 ± 1 | 128 ± 22 | 0.50 ± 0.02 |

The narrower model wins on every seed near the money. Part of that may come from its 1.9× more rows per unit of log `T` in that range. **The single model is kept anyway, for simplicity.** A model from 3 months would send every shorter put to the tree, at milliseconds instead of microseconds, or need a second network with its own checkpoint and routing. The single model is also most accurate in dollars at short maturities. The accepted cost is 1.3-2.7× the error beyond 3 months: 1.3× on the grid average, 1.5× near the money and 2.7× on the worst point. Both models trained before the output-initialization fix, and this grid was not re-measured since. The worst-point gap is likely smaller now, because the fixed single model's worst point beyond 3 months is 126-217 on the full grid.

**Thinning the exercised rows does not help.** Keeping a quarter of the puts exercised today (9% of rows instead of 28%), scored on the full grid:

| Exercised rows | Error near the money | Error over the grid | Worst error | IV error (bp) |
|---|---|---|---|---|
| all | 9 ± 1 | 6 ± 2 | 349 ± 240 | 0.59 ± 0.04 |
| a quarter | 10 ± 2 | 8 ± 1 | 445 ± 79 | 0.65 ± 0.04 |

The error elsewhere does not fall, and the worst point, among the exercised puts on every seed, gets worse. Every exercised row stays.

## Output initialization: start at the mean premium

**The problem.** At width 256, every seed's softplus output stopped learning in its first epoch (`.claude/BUGS.md`, "Width-256 softplus surrogate stopped learning in its first epoch"). The cause is where the network starts. Before training, its output weights are small, so it predicts nearly one constant `c` for every input. For a constant, the premium term of the loss is

```
premium term = 1 + (c − ȳ)² / var(y)
```

with `ȳ` the mean premium and `var(y)` the premiums' variance. PyTorch's default bias term makes a softplus output start at `c ≈ softplus(0) = 0.69` per unit of strike, against `ȳ = 0.0116`. That is a premium term of ~530. Adam's first large steps then push every output past the target, into the region where the softplus is flat and its gradient vanishes. At width 256 the outputs were there by step 17 and never came back. Width 128 nearly collapsed too, but recovered after ~100-140 steps.

**The fix.** `initial_model` in `optionpricing/ml/surrogate/tree/tree_training.py` sets the output layer's bias term so that the network starts at the mean premium. Solving `softplus(b) = ȳ`:

```python
output_layer.bias.fill_(math.log(math.expm1(mean)))
```

The initial premium term is then ~1 at any width and seed. It raises `ValueError` if the mean premium is not positive. The linear output keeps PyTorch's default: it has no flat region, so a high start cannot trap it.

**Why not a lower learning rate or a warm-up?** Both also avoid the collapse. But the start stays ~500× too high, and each adds a setting to tune per width. After 2 epochs, the fixed start's loss is ~10× lower than the default's at width 128, and 20-25× lower than the warm-up's at width 256.

Measured on 16384 training rows, seeds 0-9:

| Output | Width | Starting bias term | Initial premium term, min / median / max |
|---|---|---|---|
| softplus | 128 | PyTorch default | 475 / 513 / 584 |
| softplus | 128 | at the mean premium | 0.97 / 0.97 / 0.98 |
| softplus | 256 | PyTorch default | 490 / 523 / 585 |
| softplus | 256 | at the mean premium | 0.97 / 0.98 / 0.98 |
| linear | 128 | PyTorch default (kept) | 0.94 / 4.14 / 12.16 |

And over the first 2 epochs (656 steps, seeds 42 / 43, constant learning rate 1e-3 unless stated):

| Configuration | Collapses? | Loss at step 640 |
|---|---|---|
| width 128, default bias term | nearly, recovers after ~100-140 steps | 7.3e-2 / 9.6e-2 |
| width 256, default bias term | **yes, by step 17** | 1.75 / 1.92 |
| width 256, learning rate 3e-4 | no | 0.11 / 0.15 |
| width 256, warm-up over epoch 1 | no | 6.5e-2 / 0.11 |
| width 128, bias term at the mean premium | no | **6.9e-3 / 9.1e-3** |
| width 256, bias term at the mean premium | no | **2.6e-3 / 4.9e-3** |

**In the unit tests.** On their 128-row toy dataset, the softplus output starts so close to the labels (first validation loss 1.3-4.1, against 27-200 for the linear output, seeds 0-2) that a test asserting a large drop has little to measure. At learning rate 1e-2, seed 0 even ended above its first epoch. So the training-loop tests in `tests/unit/test_tree_training.py` use the linear output, and `test_the_default_softplus_output_trains_below_its_start` checks the softplus default against its untrained start instead: at learning rate 1e-2, its best checkpoint reaches 0.11-0.56 of that start on seeds 0-5.

The seeded and dataset ablations above trained before this fix. Their numbers are valid as measured but do not reproduce from the current code. The width ablation and the chosen model below use the fix.

## Width: 256 units do not beat 128

Protocol: the combined configuration at width 128 (50561 weights) or 256 (199425), with the fix, 2^19 rows, 400 epochs, seeds 42-46, 4 runs at a time.

| Width | Error near the money | Error over the grid | Worst error | IV error (bp) | Error beyond 3 months | IV error beyond 3 months (bp) | Training (min) |
|---|---|---|---|---|---|---|---|
| 128 | 7.9 ± 0.3 | 4.5 ± 0.4 | 166 ± 34 | 0.44 ± 0.04 | 9.0 ± 0.7 | 0.61 ± 0.05 | 29 |
| 256 | 7.1 ± 1.4 | 4.8 ± 1.0 | 160 ± 29 | 0.43 ± 0.09 | 8.1 ± 1.6 | 0.57 ± 0.10 | 76 |

Per seed:

| Width | Seed | Error near the money | IV error (bp) | Worst error | Epochs trained |
|---|---|---|---|---|---|
| 128 | 42 / 43 / 44 / 45 / 46 | 8.0 / 7.9 / 7.7 / 8.4 / 7.4 | 0.46 / 0.43 / 0.39 / 0.49 / 0.42 | 155 / 126 / 217 / 154 / 179 | 400 on every seed |
| 256 | 42 / 43 / 44 / 45 / 46 | 6.1 / 5.8 / 6.4 / 8.9 / 8.0 | 0.39 / 0.41 / 0.35 / 0.57 / 0.42 | 139 / 124 / 163 / 199 / 176 | 400 / 400 / 400 / 354 / 366 |

- **Width 256 does not pass the 2 sd rule.** Measured against the larger of the two seed standard deviations, it is at most 0.6 sd better on any error, beyond 3 months included.
- **Its gain near the money comes from three seeds.** Its seeds 42-44 have the lowest errors measured (5.8-6.4). Its seeds 45 and 46 stopped early on patience and are no better than width 128. So its spread across seeds is about twice as wide.
- **It costs 2.6× the training time** (76 min at width 256, 29 min at width 128) and about 3× the batch price.
- **Width 128 stays.** One caveat: 400 epochs were chosen at width 128, and a longer schedule could favour width 256. That was not tested.

**The fix also improves width 128.** Against the 5 width-128 runs trained before it:

| Width-128 runs | Error near the money | Error over the grid | Worst error | IV error (bp) | Violations (%) |
|---|---|---|---|---|---|
| default bias term | 8.6 ± 0.7 | 6.6 ± 1.8 | 285 ± 193 | 0.60 ± 0.03 | 0.01 ± 0.03 |
| bias term at the mean premium | 7.9 ± 0.3 | 4.5 ± 0.4 | 166 ± 34 | 0.44 ± 0.04 | 0.07 ± 0.05 |

The IV error falls by 4 seed sd. The worst point stops depending on the seed: before the fix, seed 42's was an exercised put at 620. With it, every seed's worst point is a put near the money beyond 3 months, at 126-217. The slight rise in violations is about 1 sd.

## The chosen model

The final configuration is `lambda_delta` 0.3 with a softplus output at width 128, now `TrainConfig`'s default. It was not retrained. The 5 width-128 runs above already are this configuration, trained by the same `train_surrogate` call as `train-ml`. The shipped checkpoint, `artifacts/checkpoints/put_surrogate.pth`, is seed 43, the median seed on IV error. The median rather than the best seed keeps its numbers typical of the configuration. `test-ml --seed 43` reproduces its report except for the timing.

**Accuracy on the grid**, over the 5 seeds. The shipped model's own numbers are in the README.

| Puts | Points | Mean error | Worst error | IV error (bp) | Delta error |
|---|---|---|---|---|---|
| near the money (`\|z\|` < 1) | 646 | 7.9 ± 0.3 | 166 ± 34 | 0.33 ± 0.01 | 13 |
| `T` < 1 month | 494 | 0.8 ± 0.2 | 12 ± 2 | 0.28 ± 0.06 | 5.5 |
| `T` 1-3 months | 455 | 1.5 ± 0.1 | 14 ± 4 | 0.31 ± 0.06 | 4.0 |
| `T` ≥ 3 months | 690 | 9.0 ± 0.7 | 166 ± 34 | 0.61 ± 0.05 | 14 |
| exercised today | 703 | 4.5 ± 0.7 | 114 ± 32 | - | 9.4 |
| all | 2342 | 4.5 ± 0.4 | 166 ± 34 | 0.44 ± 0.04 | 8.9 |

The maturity rows are the puts not exercised today.

- **The typical error is $0.00045**, under a twentieth of a cent, and 0.44 bp of implied volatility.
- **Short maturities are the most accurate**, in dollars and in implied vol.
- **The worst point is always near the money beyond 3 months**, at 126-217 depending on the seed.
- **The price surface has the right shape.** On the puts not exercised today, it falls and is convex in `m` and rises with `σ` at every grid point. It rises with `T` at all but 0-0.12% of them.
- **The payoff floor** answers about half of the exercised-today points (49% ± 9%), where the true price is the payoff, and under 1% of the others.

**Speed.** On an idle machine, the shipped model prices the grid in one call at 6-9 µs per option, against 3.1-3.2 ms for the 1000-step teacher tree: 360-510× faster. One option per call takes 2.1-2.3 ms, no faster than the tree, because the autograd passes for delta and gamma dominate. The gain needs batches.

**A known weak spot: rates near zero.** The grid's rates (2% and 6%) never reach the domain's low edge. On the test split, grouped by `r` (seeds 42-44, before the fix, not re-measured):

| `r` | Rows | Mean error | Worst error | Share above 1 cent |
|---|---|---|---|---|
| below 0.2% | ~2600 | 15-20 | 960-1110 | 3.0-4.7% |
| 0.2-0.5% | ~3900 | 5-7 | 140-180 | 0.2-0.4% |
| 0.5-2% | ~19600 | 4-5 | 140-170 | 0.02-0.04% |
| 2-8% | ~78700 | 4-6 | 380-440 | 0.12-0.16% |

The largest errors are long-dated, deep in-the-money puts at `r` below 0.05%. The cause is unknown. It is accepted as a known limitation, since rates are not expected back near zero soon.

## Training cost by network width

One training step on a batch of 1024 rows, 2 threads, in one process:

| Network | Weights | ms per step | Run of 400 epochs |
|---|---|---|---|
| depth 4, width 128 | 50561 | 12.5 | ~25 min |
| depth 4, width 256 | 199425 | 31.4 | ~60 min |
| depth 4, width 512 | 792065 | 113.5 | ~3.5-4 h |
| depth 6, width 128 | 83585 | 18.7 | ~37 min |

The width ablation's real runs, 4 at a time, took 29 ± 3 min at width 128 and 76 ± 8 min at width 256, the 2.6× ratio this table predicts.

## Reproducing the runs

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset                       # the 2^19-row dataset
python -m optionpricing.ml.experiments.run_ablations                                  # every one-change variant and the combined one, seeds 42-44
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset --T_min 0.25 --output_dir artifacts/data/T0.25
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset --exercised_keep 0.25 --output_dir artifacts/data/exercised_0.25
python -m optionpricing.ml.experiments.run_ablations --variants exercised_0.25
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus T_from_0.25 --grid_T_min 0.25 --out_dir artifacts/runs/ablations_T0.25
python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus lambda_delta_0.3_softplus_width_256 --seeds 42 43 44 45 46 --out_dir artifacts/runs/ablations_b036   # ~2.5 h
python main.py test-ml --seed 43                                                      # the shipped model
```

Each ablation run writes its model, report and `metrics.json` under `--out_dir` (default `artifacts/runs/ablations`), with a `summary.md` of every finished run. Run outputs are not part of the repository. The teacher study, the learning curve, the initialization measurements and the near-zero-rate grouping were measured with ad hoc scripts that are not in the repository.
