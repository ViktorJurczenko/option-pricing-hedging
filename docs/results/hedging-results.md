# Deep Hedging results

This page records the measured results of the Deep Hedging model described in [hedging-training.md](hedging-training.md). It starts with the checks that pin each built part: the bank of scenarios, the baselines and the training loop. It then gives the evaluations of the hedgers on held-out paths. The network is not trained on the full banks yet, so the first evaluation measures the baselines it will be scored against. Experiments will add their results here as they are run.

**Units.** Positions are in shares per option. Prices and P&L are per option at $`K`$ = 100, in dollars. The 95% conditional value at risk (CVaR) is the mean P&L of the worst 5% of paths, so higher is better.

## Contents

| Section | In short |
|---|---|
| [Dataset checks](#dataset-checks) | The bank is fair at zero cost, and the surrogate's premium is 0.06 cents below the tree's. |
| [Baseline checks](#baseline-checks) | On a small bank with tree deltas, every hedge averages about 0 at zero cost, and the bands beat delta at a 1% cost. |
| [Training checks](#training-checks) | The training loop learns under costs. At the default learning rate the band opens slowly. |
| [First evaluation](#first-evaluation) | At a 1% cost the tuned bands more than halve the delta hedge's tail loss. Whalley-Wilmott is the best baseline. |
| [Limits and open questions](#limits-and-open-questions) | The learning rate, scores per region of the family, a small positive mean, and SAC on the new P&L. |

## Dataset checks

**Setting.** Two banks from `generate_hedge_dataset` with the default family and holder (`tree`), the trained `artifacts/checkpoints/put_surrogate.pth`, 1000-step trees for the boundary and one path per contract: a train bank of 2,000 contracts (seed 1) and a validation bank of 8,000 (seed 2). Each is hedged daily at zero cost by `delta_hedge` on the stored surrogate deltas, booked by `hedge_pnl` in float64.

**The premium is the surrogate's price.** The hedger prices with the surrogate, so it also sells the put at the surrogate's price at $`t_0`$. A pricing error then shows in the P&L, as it would for a desk that quotes from its model. To measure that error, each contract was also priced by a 1000-step BBSR tree:

| Bank | Tree minus surrogate, mean (cents) | Mean absolute gap (cents) | Largest absolute gap (cents) |
|---|---|---|---|
| Train, seed 1 | +0.06 | 0.06 | 0.18 |
| Validation, seed 2 | +0.06 | 0.06 | 0.18 |

So the surrogate sells the put about 0.06 cents below the tree, on premiums from $0.36 to $15.38 (median $6.20, validation bank).

**The bank is fair at zero cost.** With a fair premium and an optimal holder, a delta hedge makes 0 on average. Each bank's mean P&L estimates that average from a finite sample of paths. Its standard error, $`\mathrm{std} / \sqrt{n_\text{paths}}`$, is how far the sample mean typically lands from the true average by chance. So the check is whether the mean is within about 2 standard errors of 0. The table books the same paths with both premiums. The tree premium's P&L is the surrogate premium's P&L plus the premium gap of the path's contract.

| Bank | Paths | Exercised early | Delta, surrogate premium: mean P&L ($) | Delta, tree premium: mean P&L ($) | Standard error ($) | Delta: std P&L ($) | No hedge: std P&L ($) |
|---|---|---|---|---|---|---|---|
| Train, seed 1 | 2,000 | 46% | +0.018 | +0.019 | 0.009 | 0.38 | 7.23 |
| Validation, seed 2 | 8,000 | 46% | +0.004 | +0.005 | 0.004 | 0.40 | 7.20 |

Both banks pass. The train bank is 2.1 standard errors from 0, and the validation bank, four times larger, 0.9, so the train bank's offset looks like chance. The choice of premium does not matter at this scale: the two move the mean by 0.06 cents, about a seventh of the smaller standard error. And the surrogate's deltas hedge as well as the tree's, since the delta hedge's spread of $0.40 matches the baseline checks below, which used tree deltas.

**Cost and size.** The validation bank took 50 s on 6 cores and 47 MB. The default 50,000 contracts take about 5 min and 295 MB. The (scenario, day) arrays `spot`, `delta` and `gamma` and the per-contract boundary (one contract per path) are stored in float32.

## Baseline checks

**Setting.** The baselines of `optionpricing/hedging/baselines.py` on one bank of American puts, the one of `tests/integration/test_baselines_vs_tree.py`: 4 contracts of 30 to 60 days ($`\sigma`$ 15-30%, $`r`$ 2-6%, $`S_0/K`$ 0.9-1.1), 500 paths each, seed 7. The holder exercises at the tree's boundary (`tree`), the premium is the tree price (1000 steps), and the deltas and gammas along the paths come from trees of 200 steps. Hedging is daily unless stated. 59% of the paths are exercised early. The CVaR standard errors are from 1,000 paired bootstrap resamples of the paths.

**At zero cost every strategy's mean P&L is about 0.** On risk-neutral paths, with an optimal holder and the fair premium, the premium is the expected discounted payoff and the discounted share gains have mean 0. So a mean far from 0 would mean a bias in the premium, the exercise days or the accounting.

| Strategy | Mean P&L ($) | Standard error ($) | Std P&L ($) |
|---|---|---|---|
| No hedge | -0.087 | 0.111 | 4.96 |
| Delta, daily | +0.006 | 0.009 | 0.40 |
| Delta, every 4 days | -0.003 | 0.018 | 0.79 |

Each of the 4 contracts' delta-hedge means is also within 2 standard errors of 0. The daily delta hedge leaves 8.1% of the unhedged spread. Rebalancing every 4 days multiplies its error by 1.96, close to the $`\sqrt{4}`$ = 2 of an error that grows like $`\sqrt{\Delta t}`$.

**At a 1% proportional cost the bands beat the delta hedge.** Daily delta hedging pays for many small trades that hedge little. Both bands trade back to their nearest edge when the position leaves them (`target="edge"`). The Whalley-Wilmott band is built for the cost it is charged, $`\varepsilon`$ = 1%. The CVaR is that of the P&L, so higher is better.

| Strategy | Mean costs ($) | CVaR 95% ($) | Bootstrap SE of CVaR ($) |
|---|---|---|---|
| Delta, daily | 3.23 | -5.74 | 0.08 |
| Whalley-Wilmott, $`\gamma`$ = 1 | 1.57 | -3.78 | 0.09 |
| Fixed band, half-width 0.1 | 1.75 | -3.70 | 0.07 |

The bands' CVaR is higher than the delta hedge's by 1.96 for Whalley-Wilmott and 2.04 for the fixed band, with paired bootstrap standard errors of 0.11 and 0.09. Neither band's parameter was tuned here, so these numbers check the machinery, not the best baseline. The tuned baselines are measured on the test bank in the [First evaluation](#first-evaluation).

## Training checks

These are first runs of `train_hedging_model`, to check that the loop learns. They do not set any default.

**It learns under costs, on exact deltas.** The test `tests/integration/test_hedge_training_learns.py` builds banks of European puts of 20 to 40 days, held to expiry, with Black-Scholes premiums, deltas and gammas. A train bank and a validation bank of 1,000 paths each (seeds 1 and 2), a 1% proportional cost, CVaR at 95%, batches of 500, `lr` = 1e-2, 15 epochs. The score is the validation CVaR loss, −CVaR in dollars, so lower is better.

| Network | Mean costs ($) | CVaR loss ($) |
|---|---|---|
| Untrained (epoch 0, the delta hedge) | 1.94 | 4.20 |
| Best epoch (13) | 1.22 | 3.04 |

**At the defaults the band opens slowly.** Setting: a train bank of 4,000 contracts (seed 1) and a validation bank of 2,000 (seed 2), from `generate_hedge_dataset` with the default family, the `tree` holder and the trained `artifacts/checkpoints/put_surrogate.pth`. A 1% proportional cost, CVaR at 95%, daily hedging, cash settlement, a 4 × 32 network, batches of 1,000 (4 steps per epoch), 8 epochs, seed 42. One epoch takes about 4 s on the CPU, validation included.

| `lr` | `init_logit` | CVaR loss, epoch 0 ($) | Best CVaR loss ($) | Best epoch | CVaR loss, epoch 8 ($) |
|---|---|---|---|---|---|
| 1e-3 | −5 | 8.48 | 8.25 | 8 | 8.25 |
| 3e-3 | −5 | 8.48 | 5.00 | 7 | 6.17 |
| 1e-2 | −5 | 8.48 | 4.90 | 4 | 5.64 |
| 1e-3 | −2 | 5.60 | 5.16 | 8 | 5.16 |

At the default rate the band barely moves in 32 steps. At a logit of −5 the slope of softplus is 0.0067, and Adam moves a parameter by about one learning rate per step, so at 1e-3 the band widens by thousandths of a share per epoch. A larger rate opens it within a few epochs, then jitters. At 1e-2 the loss falls from 8.48 to 4.90 by epoch 4 and rises again after, and the validation checkpoint keeps epoch 4. A wider start is better at once. At `init_logit` = −2 (half-width 0.13 shares) the untrained band already scores 5.60, but it no longer starts as the delta hedge.

The learning rate, a possible schedule and `init_logit` are left to be set by a check on the full banks ([Limits and open questions](#limits-and-open-questions)).

## First evaluation

**Setting.** A validation bank of 10,000 contracts (seed 2) and a test bank of 20,000 (seed 3), from `generate_hedge_dataset` with the default family and holder (`tree`), the trained `artifacts/checkpoints/put_surrogate.pth`, 1000-step trees for the boundary and one path per contract. 45% of the validation paths and 46% of the test paths are exercised early. `test-hedge` tunes the two bands on the validation bank and scores every hedger on the test bank, at zero cost and at a 1% proportional cost, with the default grids of `hedge_evaluation.py`. Hedging is daily, settlement is in cash and the bands are tuned on CVaR at 95%. The band network is not trained yet, so this run measures the baselines it will be scored against. The CVaR standard errors are from 1,000 paired bootstrap resamples of the test paths (seed 0).

```
python -m optionpricing.ml.hedging.generate_hedge_dataset --split val --n_contracts 10000 --output_dir artifacts/data/eval1    # 99 s on 6 cores
python -m optionpricing.ml.hedging.generate_hedge_dataset --split test --n_contracts 20000 --output_dir artifacts/data/eval1   # 167 s
python main.py test-hedge --baselines_only --test_path artifacts/data/eval1/hedge_test.npz --valid_path artifacts/data/eval1/hedge_val.npz                       # zero cost, ~15 s
python main.py test-hedge --baselines_only --test_path artifacts/data/eval1/hedge_test.npz --valid_path artifacts/data/eval1/hedge_val.npz --proportional 0.01   # 1% cost
```

**At zero cost the P&L shows no visible bias, and delta hedging leaves 5.6% of the unhedged standard deviation.**

| Hedger | Mean P&L (¢) | Standard error (¢) | Std P&L ($) | CVaR 95% ($) | Bootstrap SE of CVaR ($) |
|---|---|---|---|---|---|
| No hedge | −3.3 | 5.2 | 7.29 | −15.26 | 0.09 |
| Delta, daily | +0.68 | 0.29 | 0.405 | −0.928 | 0.013 |

Both bands equal the delta hedge path for path at zero cost. The fixed band's validation score falls at every step of width, so 0 wins. The Whalley-Wilmott half-width is proportional to $`\varepsilon^{1/3}`$, so it is 0 for every $`\gamma`$. All its scores tie, and `tune` reports the first value of the grid, which means nothing here.

**The delta mean is 2.4 standard errors above 0, and only partly explained.** Two known effects move it, in opposite directions.
- The holder checks the boundary once a day. So the put the hedger owes is a daily Bermudan, worth less than the American its premium prices. On 400 test contracts the gap averages 0.14 ¢, and 0.31 ¢ at most. It was measured with a CRR tree at 20 steps a day, exercising at every step for the American and at every 20th for the Bermudan, in a one-off script that is not in the repository.
- The surrogate sells the put 0.06 ¢ below the tree ([Dataset checks](#dataset-checks)), which costs the hedger.

Together they predict about +0.08 ¢. The remaining +0.6 ¢ is 2.1 standard errors, as far as the train bank of the Dataset checks, so it may be chance. It is under a cent on a median premium of $6.16, and it changes no ranking ([Limits and open questions](#limits-and-open-questions)).

**At a 1% proportional cost the bands more than halve the delta hedge's tail loss.** The CVaR gain is the hedger's CVaR minus the delta hedge's, with its paired bootstrap standard error.

| Hedger | Tuned value | Mean costs ($) | Trades per path | Mean P&L ($) | Std P&L ($) | CVaR 95% ($) | CVaR gain over delta ($) |
|---|---|---|---|---|---|---|---|
| No hedge | | 0 | 0 | −0.03 | 7.29 | −15.26 | −4.83 ± 0.10 |
| Delta, daily | | 5.14 | 191 | −5.13 | 2.32 | −10.42 | 0 |
| Fixed band | half-width 0.075 shares | 1.83 | 51.7 | −1.82 | **1.28** | −4.73 | +5.69 ± 0.05 |
| **Whalley-Wilmott** | $`\gamma`$ = 1.4 per $ | **1.61** | **45.6** | **−1.60** | 1.31 | **−4.61** | **+5.81 ± 0.05** |

- **Daily delta hedging spends most of the premium on costs.** It trades on each of its 191 days per path on average and pays $5.14, 83% of the median premium of $6.16. The hedger is paid the premium once, so at 1% the daily rebalancing eats most of it.
- **The bands trade about 4 times less.** They cut the costs by 64% for the fixed band and 69% for Whalley-Wilmott. Most of their CVaR gain comes from the mean, which rises by $3.3-3.5. The std P&L also falls, from $2.32 to about $1.3.
- **Whalley-Wilmott is the best baseline, by a small margin.** Its CVaR is $0.12 above the fixed band's, with a paired bootstrap standard error of $0.03. Its entropic risk ($`\lambda`$ = 1) is also lower, 2.72 for Whalley-Wilmott and 2.92 for the fixed band. Its half-width $`\big(\tfrac{3}{2} e^{-r(T - t)} \varepsilon S \Gamma^2 / \gamma\big)^{1/3}`$ follows the put's gamma and the share price, so one tuned $`\gamma`$ adapts to every contract of the family. A fixed width does not follow gamma, so it fits the family less well. On the 4 short contracts of [Baseline checks](#baseline-checks), the two untuned bands were level within noise.

**Tuning.** Validation CVaR in dollars, higher is better. The best value is in bold.

| Fixed band half-width (shares) | 0 | 0.025 | 0.05 | 0.0625 | **0.075** | 0.0875 | 0.1 | 0.1125 | 0.125 | 0.15 | 0.175 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation CVaR ($) | −10.48 | −6.36 | −5.03 | −4.74 | **−4.65** | −4.71 | −4.87 | −5.09 | −5.35 | −5.90 | −6.41 | −6.85 | −9.86 |

| Whalley-Wilmott $`\gamma`$ (per $) | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | **1.4** | 2.0 | 3.0 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Validation CVaR ($) | −5.70 | −4.86 | −4.68 | −4.61 | −4.57 | **−4.56** | −4.58 | −4.63 | −4.98 |

Both optima are inside their grids. On the validation bank the fixed band's neighbours score within $0.09 of it, and Whalley-Wilmott's within $0.02. The validation scores also carry over to the test bank within $0.09: the fixed band scores −4.65 on validation and −4.73 on test, Whalley-Wilmott −4.56 and −4.61.

The grids were refined after a first pass. The first grids doubled the half-width at each step (0.05, 0.1, 0.2) and spaced $`\gamma`$ by half-decades (0.3, 1, 3). The fixed band then picked 0.1, which scores $0.19 below 0.075 on the test bank. The finer grids are now the defaults, `BAND_WIDTHS` and `RISK_AVERSIONS`.

## Limits and open questions

- **Learning rate and start of the band.** At `lr` = 1e-3 the band opens too slowly, and at 1e-2 the validation loss jitters after a few epochs ([Training checks](#training-checks)). Whether a higher rate with a decay, or a wider `init_logit`, becomes the default is set by a check on the full banks.
- **Scores per region of the family.** A family score can hide a region where a hedger is weak, so an optional last step scores the test paths per bucket of maturity, volatility and moneyness, with bands tuned per bucket. At 3 × 3 × 2 = 18 buckets, a 10,000-contract validation bank leaves about 28 tail paths per bucket at 95%, so that step needs fewer buckets or a larger validation bank. Buckets replace separate per-contract banks. Those would need the same contracts with fresh paths for validation, but the generator draws contracts and paths from one seed. A bank of 200 contracts × 1,000 paths would also take about 3 GB of RAM with the current evaluation (an estimate from the array sizes).
- **The delta hedge's small positive mean.** At zero cost it is +0.68 ¢ ± 0.29 ¢ on the test bank, and the daily-exercise gap and the premium gap explain about +0.08 ¢ of it ([First evaluation](#first-evaluation)). A bank of 50,000 paths, or a control variate on the mean, would tell chance from a bias.
- **SAC on the new P&L.** Whether SAC is retrained once for one comparison. Decided at the end, once Deep Hedging has results.
