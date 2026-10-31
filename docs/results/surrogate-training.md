# Tree surrogate: teacher, training-length and ablation results

Measured results behind the tree MLP surrogate's defaults (`binomial_put_batch`'s `method="bbsr"`, `nb_steps=1000`; `TrainConfig.epochs = 400`). Moved from `.claude/PLAN.md`, where they were recorded as the work went. **Not re-verified** since they were measured (2026-10-22 to 2026-10-31); rerun before quoting them as current.

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
- 800 also had fewer violations (0.06% > 1e-4 vs 0.61%) and more payoff floors (490 grid points vs 179).
- Single seed each. The seeded ablations below give the seed-to-seed spread.

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
