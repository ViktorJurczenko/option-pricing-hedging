# Tree surrogate: teacher and training-length results

Measured results behind the tree MLP surrogate's defaults (`binomial_put_batch`'s `method="bbsr"`, `nb_steps=1000`; `TrainConfig.epochs = 400`). Moved from `.claude/PLAN.md`, where they were recorded as the work went. **Not re-verified** since they were measured (2026-10-22 to 2026-10-27); rerun before quoting them as current.

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

Sampling `m` log-uniform on (0.4, 1.6) puts 66% of the draws in the money (the range is asymmetric in log: ln 0.4 = -0.92, ln 1.6 = +0.47), and 67% of those are already past the exercise boundary: 44% of all rows would be exercised today, with labels known in closed form. The generator samples `z = ln(m) / (sigma sqrt(T))` instead (`map_to_domain` in `ML/surrogate/tree/generate_tree_dataset.py`).

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
- Single seed each. The seeded ablations (`python -m experiments.run_ablations`, output `runs/ablations/summary.md`) give the seed-to-seed spread.
