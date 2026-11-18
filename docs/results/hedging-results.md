# Deep Hedging results

This page records the measured results of the Deep Hedging model described in [hedging-training.md](hedging-training.md). It starts with the checks that pin each built part: the bank of scenarios, the baselines and the training loop. It then gives the evaluations of the hedgers on held-out paths. The first evaluation measures the baselines alone. The experiments after it train the network on full banks, and their results are added here once they are conclusive.

**Units.** Positions are in shares per option. Prices and P&L are per option at $`K`$ = 100, in dollars. The 95% conditional value at risk (CVaR) is the mean P&L of the worst 5% of paths, so higher is better.

## Contents

| Section | In short |
|---|---|
| [Dataset checks](#dataset-checks) | The bank is fair at zero cost, and the surrogate's premium is 0.06 cents below the tree's. |
| [Baseline checks](#baseline-checks) | On a small bank with tree deltas, every hedge averages about 0 at zero cost, and the bands beat delta at a 1% cost. |
| [Training checks](#training-checks) | The training loop learns under costs. At the default learning rate the band opens slowly. |
| [First evaluation](#first-evaluation) | At a 1% cost the tuned bands more than halve the delta hedge's tail loss. Whalley-Wilmott is the best baseline. |
| [The boundary input adds at most a cent, and a late holder changes little](#the-boundary-input-adds-at-most-a-cent-and-a-late-holder-changes-little) | The distance to the exercise boundary moves the network's CVaR by half a cent, so it stays off. With a late holder the network keeps its lead of about $0.96 over the best baseline. |
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

## The boundary input adds at most a cent, and a late holder changes little

The holder exercises the first time the spot falls to the boundary $`S^*`$, so the put changes regime on that line. The network can read an optional extra input, the distance to the boundary $`\ln\big(S_k / S^*(T - t_k)\big)`$ (`--boundary_input`). Two questions follow. Does the network hedge better when it sees that distance? And does the answer change for a holder who does not follow the boundary?

**Setting.** Two sets of banks from the default contract family, one per holder. The `tree` holder exercises at $`S^*`$. The `late` holder exercises at $`0.95\,S^*`$ (`--exercise late --shift 0.05`), so it waits until the spot is 5% below the boundary. Each set has a train bank of 50,000 contracts (seed 1), a validation bank of 10,000 (seed 2) and a test bank of 20,000 (seed 3), with one path per contract. The two holders share their contracts and paths, and only their exercise days differ. The tree holder exercises 46% of the test paths early, and the late holder 35%. Every network is the default 4 × 32 band network. It is trained for 50 epochs at a 1% proportional cost on CVaR at 95%, with batches of 4,096, `lr` = 1e-2 and the CVaR threshold's `w_lr` = 1e-2, on seeds 1 to 4. The epoch with the best validation score is kept. `test-hedge` then tunes the bands on the validation bank and scores every hedger on the test bank.

```
D=artifacts/runs/exp_overnight
python -m optionpricing.ml.hedging.generate_hedge_dataset --split train --n_contracts 50000 --output_dir $D/data/tree                          # ~5.5 min, 6 cores
python -m optionpricing.ml.hedging.generate_hedge_dataset --split train --n_contracts 50000 --exercise late --shift 0.05 --output_dir $D/data/late
#   the same for --split val --n_contracts 10000 (~1.5 min) and --split test --n_contracts 20000 (~2.5 min)
python main.py train-hedge --train_path $D/data/tree/hedge_train.npz --valid_path $D/data/tree/hedge_val.npz --checkpoint_path $D/ckpt/bnd_tree_s1.pth --epochs 50 --proportional 0.01 --lr 1e-2 --seed 1 --boundary_input   # ~11 min
python main.py test-hedge --test_path $D/data/tree/hedge_test.npz --valid_path $D/data/tree/hedge_val.npz --checkpoint_path $D/ckpt/bnd_tree_s1.pth   # ~15 s
```

**The boundary input changes the CVaR by half a cent.** Network CVaR 95% on the test bank, in dollars, higher is better:

| Holder | Input off | Input on | On minus off |
|---|---|---|---|
| Tree | −3.668 ± 0.005 | −3.663 ± 0.006 | +0.005 ± 0.006 |
| Late | −3.845 ± 0.009 | −3.849 ± 0.013 | −0.004 ± 0.010 |

Mean ± standard deviation over the 4 training seeds, all scored on the same test bank. "On minus off" pairs the two runs of each seed.

The difference is half a cent on a tail loss of $3.7, and the spread between seeds is as large. A paired bootstrap of seed 1's test paths gives the same picture, +0.009 (standard error 0.003) for the tree holder and +0.005 (standard error 0.005) for the late one. So the input is worth at most a cent, about 0.2% of the tail loss. It stays off by default.

**Why it adds so little.** Per unit of strike, the boundary depends only on the time left, $`r`$ and $`\sigma`$, and the network reads all three. It also reads $`\ln(S_k/K)`$. The distance $`\ln(S_k/S^*) = \ln(S_k/K) - \ln(S^*/K)`$ is therefore a function of inputs it already has. The extra input can only spare the network from learning that function, and 50,000 paths are enough for it to learn it anyway. For the late holder, the distance that matters is to $`0.95\,S^*`$. That distance differs from the input by the constant $`\ln 0.95`$, so the input is just as informative there.

**A late holder costs the unhedged writer $2.25 and every hedger $0.08-0.24, and the network keeps its lead.** CVaR 95% on the test bank, in dollars, for the same paths under each holder. The networks are the four seeds without the boundary input, each trained on its own holder's bank.

| Hedger | Tree holder | Late holder | Late minus tree |
|---|---|---|---|
| No hedge | −15.26 | −17.51 | −2.25 |
| Delta, daily | −10.42 | −10.58 | −0.16 |
| Fixed band | −4.73 (half-width 0.075) | −4.81 (half-width 0.075) | −0.08 |
| Whalley-Wilmott | −4.61 ($`\gamma`$ = 1.4) | −4.86 ($`\gamma`$ = 2.0) | −0.24 |
| **Network** | **−3.67** | **−3.85** | −0.18 |

- **The late holder keeps the put alive longer.** Below the boundary the optimal holder would already have taken $`K - S`$, but the late holder lets the put run on. That widens the unhedged spread from $7.29 to $8.00. It gives the writer no measurable gain on average, since the no-hedge mean P&L is +0.03 ¢ with a standard error of 5.7 ¢.
- **The deltas are slightly wrong for this holder.** The surrogate prices the put for an optimal holder, so between $`0.95\,S^*`$ and $`S^*`$ its delta is that of an exercised put, −1. The late holder's put still has time value there, so its true delta is above −1. The delta hedge also trades on more days, 204 per path for the late holder against 191 for the tree holder.
- **The network's lead over the best baseline holds.** It is $0.96 for the late holder, over the fixed band, and $0.95 for the tree holder, over Whalley-Wilmott. The paired bootstrap standard errors on seed 1 are $0.03 for both. The network is not told that the holder is late. It only sees the late bank's P&L in training. Its CVaR drops by $0.18, between the fixed band's $0.08 and Whalley-Wilmott's $0.24, so there is no sign that it learned the late rule itself.

Both questions are settled. The boundary input stays an option, off by default. The late holder stays a test scenario, and the network's lead holds when the holder misses the boundary by 5%.

**Known limit.** These networks trained for 50 epochs. In most runs the best epoch is among the last five, and the CVaR threshold $`w`$ was still settling until about epoch 40 ([Limits and open questions](#limits-and-open-questions)). The on/off and tree/late comparisons share that setting, so they hold. The absolute CVaR of the network will move when the cost runs are redone with longer training.

## Limits and open questions

- **Learning rate and start of the band.** At `lr` = 1e-3 the band opens too slowly, and at 1e-2 the validation loss jitters after a few epochs ([Training checks](#training-checks)). Whether a higher rate with a decay, or a wider `init_logit`, becomes the default is set by a check on the full banks.
- **The CVaR threshold moves slowly.** `w` starts at the 95% quantile of the untrained network's loss, close to the delta hedge's, and Adam moves it by about `w_lr` per step. A 50,000-path bank in batches of 4,096 gives 13 steps per epoch, so at `w_lr` = 1e-2 the threshold falls by about 0.13 per epoch. At a 1% cost it settles only around epoch 40 ([The boundary input adds at most a cent, and a late holder changes little](#the-boundary-input-adds-at-most-a-cent-and-a-late-holder-changes-little)). A comparison of `w_lr` values sets its rate.
- **Scores per region of the family.** A family score can hide a region where a hedger is weak, so an optional last step scores the test paths per bucket of maturity, volatility and moneyness, with bands tuned per bucket. At 3 × 3 × 2 = 18 buckets, a 10,000-contract validation bank leaves about 28 tail paths per bucket at 95%, so that step needs fewer buckets or a larger validation bank. Buckets replace separate per-contract banks. Those would need the same contracts with fresh paths for validation, but the generator draws contracts and paths from one seed. A bank of 200 contracts × 1,000 paths would also take about 3 GB of RAM with the current evaluation (an estimate from the array sizes).
- **The delta hedge's small positive mean.** At zero cost it is +0.68 ¢ ± 0.29 ¢ on the test bank, and the daily-exercise gap and the premium gap explain about +0.08 ¢ of it ([First evaluation](#first-evaluation)). A bank of 50,000 paths, or a control variate on the mean, would tell chance from a bias.
- **SAC on the new P&L.** Whether SAC is retrained once for one comparison. Decided at the end, once Deep Hedging has results.
