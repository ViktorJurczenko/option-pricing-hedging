# Deep Hedging results

This page records what the Deep Hedging model of [hedging-training.md](hedging-training.md) achieves, and how far each result can be trusted. The model writes an American put, sells it at the surrogate's price, and hedges it daily with shares. A small network sets a no-transaction band around the surrogate's delta, trained to make the worst outcomes as mild as possible.

The page has two parts. The checks come first. They pin each built part on its own: the bank of scenarios, the baselines and the training loop. The evaluations follow. They score the trained network against the hedges a desk would use without a network: no hedge, the daily delta hedge, and two bands tuned on held-out paths.

In short, the network has the mildest loss tail of every hedger under proportional costs. Its edge is about a fifth of the best band's tail loss at a 1% cost, and smaller at 0.1%. It comes from a narrower spread of outcomes, not from cheaper trading. Two design questions are settled along the way. The distance to the exercise boundary adds at most a cent as an input, and the network keeps its lead when the holder exercises late. The fixed-fee experiments stopped partway through, and their results will be added once the rest has run.

## How to read this page

**Units.** Positions are in shares per option. Prices and P&L are per option at $`K`$ = 100, in dollars.

**The risk measure.** The 95% conditional value at risk (CVaR) is the mean P&L of the worst 5% of paths. It is negative when those paths lose money, so higher is better. Training minimizes the CVaR loss, which is −CVaR, so a lower CVaR loss is better.

**Two kinds of uncertainty.**
- "± " after a network's score is the standard deviation over its training seeds. Every seed is scored on the same test paths, so it measures training noise only.
- "SE" is a standard error from 1,000 paired bootstrap resamples of the test paths. It measures how much a score, or a difference between two hedgers on the same paths, would move on another test bank. For a trained network it is computed on seed 1 only.

**Where the numbers come from.** Each table ends with a **Source** line that names the files holding its numbers. The prose quotes few numbers and points to the tables instead. The commands are in [Reproducing the runs](#reproducing-the-runs).

## Contents

| Section | In short |
|---|---|
| **Checks** | |
| [Dataset checks](#dataset-checks) | The bank is fair at zero cost, and the surrogate's premium is a fraction of a cent below the tree's. |
| [Baseline checks](#baseline-checks) | On a small bank, every hedge averages about 0 at zero cost, and the bands beat the delta hedge at a 1% cost. |
| [Training checks](#training-checks) | The training loop learns under costs. At a small learning rate the band opens slowly. |
| **Evaluations** | |
| [First evaluation](#first-evaluation) | At a 1% cost the tuned bands more than halve the delta hedge's tail loss. Whalley-Wilmott is the best baseline. |
| [Setting of the network experiments](#setting-of-the-network-experiments) | The banks, network, training and scoring that every experiment below shares. |
| [The network beats the tuned bands under proportional costs](#the-network-beats-the-tuned-bands-under-proportional-costs) | The network's tail loss is about a fifth below the best band's at 1%, and about 15% below at 0.1%. |
| [Learning rates](#learning-rates) | The network trains best at `lr` = 1e-2. The CVaR threshold's rate does not change the result. |
| [The boundary input adds at most a cent](#the-boundary-input-adds-at-most-a-cent) | The distance to the exercise boundary repeats what the network already knows, so it stays off. |
| [A late holder changes little for a hedger](#a-late-holder-changes-little-for-a-hedger) | A holder who exercises late hurts the unhedged writer, but the network keeps its lead. |
| [Limits and open questions](#limits-and-open-questions) | Fixed fees, scores per region of the family, a small positive mean, and soft actor-critic (SAC). |
| [Reproducing the runs](#reproducing-the-runs) | The commands, with their cost. |

## Dataset checks

**Question.** Is a bank of scenarios fair, so that a perfect hedge would make nothing on average? And how far is the surrogate's premium from the tree's?

**Setting.** Two banks from `generate_hedge_dataset` with the default family and holder (`tree`), the trained `artifacts/checkpoints/put_surrogate.pth`, 1000-step trees for the boundary and one path per contract. The train bank has 2,000 contracts (seed 1), and the validation bank 8,000 (seed 2). Each is hedged daily at zero cost by `delta_hedge` on the stored surrogate deltas, booked by `hedge_pnl` in float64.

**The premium is the surrogate's price.** The hedger prices with the surrogate, so it also sells the put at the surrogate's price at $`t_0`$. A pricing error then shows in the P&L, as it would for a desk that quotes from its model. Each contract was also priced by a 1000-step BBSR tree to measure that error.

| Bank | Tree minus surrogate, mean (cents) | Mean absolute gap (cents) | Largest absolute gap (cents) |
|---|---|---|---|
| Train, seed 1 | +0.06 | 0.06 | 0.18 |
| Validation, seed 2 | +0.06 | 0.06 | 0.18 |

**Source.** A one-off run on these two banks, which were not kept. Not re-verified since.

**The bank is fair at zero cost.** With a fair premium and an optimal holder, a delta hedge makes 0 on average. Each bank's mean P&L estimates that average from a finite sample of paths. Its standard error, $`\mathrm{std} / \sqrt{n_\text{paths}}`$, is how far the sample mean typically lands from the true average by chance. So the check is whether the mean is within about 2 standard errors of 0. The table books the same paths with both premiums.

| Bank | Paths | Exercised early | Delta, surrogate premium: mean P&L ($) | Delta, tree premium: mean P&L ($) | Standard error ($) | Delta: std P&L ($) | No hedge: std P&L ($) |
|---|---|---|---|---|---|---|---|
| Train, seed 1 | 2,000 | 46% | +0.018 | +0.019 | 0.009 | 0.38 | 7.23 |
| Validation, seed 2 | 8,000 | 46% | +0.004 | +0.005 | 0.004 | 0.40 | 7.20 |

**Source.** The same one-off run. Not re-verified since.

**What the tables show.** Both banks pass. The validation bank, four times larger, sits within one standard error of 0, so the train bank's offset of about two standard errors looks like chance. The surrogate sells the put a fraction of a cent below the tree, on premiums of several dollars. That gap is far smaller than the standard error, so the choice of premium does not matter at this scale. The surrogate's deltas also give a delta-hedge spread in the same range as the tree's deltas in the [Baseline checks](#baseline-checks). The two banks differ, so this is a sanity check, not a like-for-like comparison.

**Critique.** "Within two standard errors" is a weak test on 2,000 paths, since it cannot see a bias smaller than a cent or two. The larger test bank of the [First evaluation](#first-evaluation) does see a small positive mean, which is still not fully explained.

**Cost and size.** The validation bank took 50 s on 6 cores and 47 MB. The default 50,000 contracts take about 5 min and 295 MB. The (scenario, day) arrays `spot`, `delta` and `gamma` and the per-contract boundary are stored in float32.

## Baseline checks

**Question.** Do the baselines of `optionpricing/hedging/baselines.py` behave as theory says, before any tuning?

**Setting.** One small bank of American puts, the one of `tests/integration/test_baselines_vs_tree.py`. It has 4 contracts of 30 to 60 days ($`\sigma`$ 15-30%, $`r`$ 2-6%, $`S_0/K`$ 0.9-1.1) with 500 paths each, seed 7. The holder exercises at the tree's boundary, the premium is the tree price (1000 steps), and the deltas and gammas come from trees of 200 steps. 59% of the paths are exercised early. Hedging is daily unless stated.

**At zero cost every strategy's mean P&L is about 0.** On risk-neutral paths, with an optimal holder and the fair premium, the premium is the expected discounted payoff and the discounted share gains have mean 0. A mean far from 0 would reveal a bias in the premium, the exercise days or the accounting.

| Strategy | Mean P&L ($) | Standard error ($) | Std P&L ($) |
|---|---|---|---|
| No hedge | −0.087 | 0.111 | 4.96 |
| Delta, daily | +0.006 | 0.009 | 0.40 |
| Delta, every 4 days | −0.003 | 0.018 | 0.79 |

**Source.** A one-off run on the bank of `tests/integration/test_baselines_vs_tree.py`. The test asserts the same properties with looser tolerances: means within 4 standard errors, the 4-day error about twice the daily one, and the bands at least a dollar of CVaR above the delta hedge. The printed numbers are not re-verified.

Each strategy's mean is within two standard errors of 0, and so is each contract's delta-hedge mean. Rebalancing every 4 days instead of daily roughly doubles the hedging error, as theory predicts for an error that grows like $`\sqrt{\Delta t}`$.

**At a 1% proportional cost the bands beat the delta hedge.** Daily delta hedging pays for many small trades that hedge little. Both bands trade back to their nearest edge when the position leaves them (`target="edge"`). The Whalley-Wilmott band is built for the cost it is charged, $`\varepsilon`$ = 1%.

| Strategy | Mean costs ($) | CVaR 95% ($) | SE of CVaR ($) |
|---|---|---|---|
| Delta, daily | 3.23 | −5.74 | 0.08 |
| Whalley-Wilmott, $`\gamma`$ = 1 | 1.57 | −3.78 | 0.09 |
| Fixed band, half-width 0.1 | 1.75 | −3.70 | 0.07 |

**Source.** The same one-off run. Not re-verified.

**What the table shows.** Both bands roughly halve the costs and gain about two dollars of CVaR over the delta hedge. That gain is many times its standard error. The two bands are level within noise here.

**Critique.** Neither band's parameter was tuned. This table checks the machinery, not the best baseline. Four short contracts are also far from the family the network is trained on. The tuned baselines on the full family are in the [First evaluation](#first-evaluation).

## Training checks

These are first runs of `train_hedging_model`, to check that the loop learns. They set no default. The learning rate is set later, on the full banks ([Learning rates](#learning-rates)).

**It learns under costs, on exact deltas.** The test `tests/integration/test_hedge_training_learns.py` builds banks of European puts of 20 to 40 days, held to expiry, with Black-Scholes premiums, deltas and gammas. It uses a train bank and a validation bank of 1,000 paths each (seeds 1 and 2), a 1% proportional cost, CVaR at 95%, batches of 500, `lr` = 1e-2 and 15 epochs.

| Network | Mean costs ($) | Validation CVaR loss ($) |
|---|---|---|
| Untrained (epoch 0, the delta hedge) | 1.94 | 4.20 |
| Best epoch (13) | 1.22 | 3.04 |

**Source.** A run of the test's setting. The test itself asserts that the costs and the loss fall by set margins, not these values. Not re-verified since.

**At a small learning rate the band opens slowly.** A train bank of 4,000 contracts (seed 1) and a validation bank of 2,000 (seed 2), with the default family and holder. A 1% cost, CVaR at 95%, a 4 × 32 network, batches of 1,000 (4 steps per epoch), 8 epochs, seed 42. One epoch takes about 4 s on the CPU.

| `lr` | `init_logit` | CVaR loss, epoch 0 ($) | Best CVaR loss ($) | Best epoch | CVaR loss, epoch 8 ($) |
|---|---|---|---|---|---|
| 1e-3 | −5 | 8.48 | 8.25 | 8 | 8.25 |
| 3e-3 | −5 | 8.48 | 5.00 | 7 | 6.17 |
| 1e-2 | −5 | 8.48 | 4.90 | 4 | 5.64 |
| 1e-3 | −2 | 5.60 | 5.16 | 8 | 5.16 |

**Source.** The banks `artifacts/data/hedge_train.npz` and `artifacts/data/hedge_val.npz`. The training histories were not kept. Not re-verified since.

**What the table shows.** At 1e-3 the band barely moves in 32 steps. At a logit of −5 the slope of softplus is tiny, and Adam moves a parameter by about one learning rate per step, so the band widens by thousandths of a share per epoch. A larger rate opens it within a few epochs, then jitters. A wider start (`init_logit` = −2) scores well at once, but the untrained band is then no longer the delta hedge.

**Critique.** 32 steps on 4,000 contracts is far from the full training, so this table says which rates can work, not which is best.

## First evaluation

**Question.** How good are the baselines on the full contract family, once each band is tuned? These are the scores the network has to beat.

**Setting.** A validation bank of 10,000 contracts (seed 2) and a test bank of 20,000 (seed 3), with the default family and holder, the trained surrogate, 1000-step trees for the boundary and one path per contract. 46% of the test paths are exercised early. `test-hedge` tunes the two bands on the validation bank and scores every hedger on the test bank, at zero cost and at a 1% proportional cost, with the default grids of `hedge_evaluation.py`. Hedging is daily, settlement is in cash and the bands are tuned on CVaR at 95%.

**At zero cost the delta hedge's mean is small.**

| Hedger | Mean P&L (¢) | Standard error (¢) | Std P&L ($) | CVaR 95% ($) | SE of CVaR ($) |
|---|---|---|---|---|---|
| No hedge | −3.3 | 5.2 | 7.29 | −15.26 | 0.09 |
| Delta, daily | +0.68 | 0.29 | 0.405 | −0.928 | 0.013 |

**Source.** The zero-cost `test-hedge` command in [Reproducing the runs](#reproducing-the-runs), on `artifacts/data/eval1`. Its results are the `0.0` entry of `artifacts/data/eval1/eval1_results.json`.

At zero cost both bands reduce to the delta hedge, path for path. The fixed band's validation score falls at every step of width, so a width of 0 wins. The Whalley-Wilmott half-width is proportional to $`\varepsilon^{1/3}`$, so it is 0 for every $`\gamma`$.

**The delta mean is a little above 0, and only partly explained.** It is under a cent on a premium of several dollars, but about 2.4 standard errors from 0. Two known effects move it, in opposite directions.
- The holder checks the boundary once a day. So the put the hedger owes is a daily Bermudan, worth a little less than the American its premium prices. On 400 test contracts the gap averages a seventh of a cent. It was measured with a CRR tree at 20 steps a day, exercising at every step for the American and at every 20th for the Bermudan, in a one-off script that is not in the repository.
- The surrogate sells the put slightly below the tree ([Dataset checks](#dataset-checks)), which costs the hedger.

Together they explain only about a tenth of the mean. The rest may be chance, since it is about two standard errors. It changes no ranking ([Limits and open questions](#limits-and-open-questions)).

**At a 1% proportional cost the bands more than halve the delta hedge's tail loss.** The CVaR gain is the hedger's CVaR minus the delta hedge's, with its SE.

| Hedger | Tuned value | Mean costs ($) | Trades per path | Mean P&L ($) | Std P&L ($) | CVaR 95% ($) | CVaR gain over delta ($) |
|---|---|---|---|---|---|---|---|
| No hedge | | 0 | 0 | −0.03 | 7.29 | −15.26 | −4.83 ± 0.10 |
| Delta, daily | | 5.14 | 191 | −5.13 | 2.32 | −10.42 | 0 |
| Fixed band | half-width 0.075 shares | 1.83 | 51.7 | −1.82 | **1.28** | −4.73 | +5.69 ± 0.05 |
| **Whalley-Wilmott** | $`\gamma`$ = 1.4 per $ | **1.61** | **45.6** | **−1.60** | 1.31 | **−4.61** | **+5.81 ± 0.05** |

**Source.** The 1% `test-hedge` command in [Reproducing the runs](#reproducing-the-runs). Its metrics and the SEs of the CVaR gains are in `artifacts/data/eval1/eval2_results.json`. The round-1 and round-2 reports of the network experiments score these baselines on a test bank of the same settings and seed, and give the same values (`artifacts/runs/exp_round2/report/cost1_w1e-2_s1.json`).

**What the table shows.**
- **Daily delta hedging spends most of the premium on costs.** It trades nearly every day and pays most of a typical premium in costs. The hedger is paid the premium once, so at 1% the daily rebalancing eats most of it.
- **The bands trade about four times less.** Most of their gain comes from the mean, which rises by several dollars, and the spread also shrinks.
- **Whalley-Wilmott is the best baseline, by a small margin.** Its CVaR beats the fixed band's by about four standard errors of the difference, and its entropic risk is lower too. Its half-width $`\big(\tfrac{3}{2} e^{-r(T - t)} \varepsilon S \Gamma^2 / \gamma\big)^{1/3}`$ follows the put's gamma and the share price, so one tuned $`\gamma`$ adapts to every contract of the family. A fixed width does not follow gamma, so it fits the family less well.

**Tuning.** Validation CVaR in dollars, higher is better. The best value is in bold.

| Fixed band half-width (shares) | 0 | 0.025 | 0.05 | 0.0625 | **0.075** | 0.0875 | 0.1 | 0.1125 | 0.125 | 0.15 | 0.175 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation CVaR ($) | −10.48 | −6.36 | −5.03 | −4.74 | **−4.65** | −4.71 | −4.87 | −5.09 | −5.35 | −5.90 | −6.41 | −6.85 | −9.86 |

| Whalley-Wilmott $`\gamma`$ (per $) | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | **1.4** | 2.0 | 3.0 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Validation CVaR ($) | −5.70 | −4.86 | −4.68 | −4.61 | −4.57 | **−4.56** | −4.58 | −4.63 | −4.98 |

**Source.** The `tuned` field of the same report.

Both optima are inside their grids, and Whalley-Wilmott's score is nearly flat around its optimum. The validation scores carry over to the test bank within ten cents. The grids were refined after a first, coarser pass, which picked a fixed band too wide. The finer grids are now the defaults, `BAND_WIDTHS` and `RISK_AVERSIONS`.

**Critique.** Each band has one parameter for the whole family. A band tuned per region of the family would be a stronger baseline, and the network may owe part of its edge to that gap ([Limits and open questions](#limits-and-open-questions)).

## Setting of the network experiments

Every experiment below shares this setting, unless its section says otherwise.

| Part | Setting |
|---|---|
| Banks | Default family and `tree` holder, one path per contract. Train 50,000 contracts (seed 1), validation 10,000 (seed 2), test 20,000 (seed 3). |
| Network | The default band network, 4 × 32, SiLU, `init_logit` = −5, reading the six default inputs. |
| Training | A 1% proportional cost unless stated, daily hedging, cash settlement, CVaR at 95%. Batches of 4,096 (13 steps per epoch), `lr` = 1e-2, CVaR threshold `w_lr` = 1e-2. The epoch with the best validation score is kept. |
| Seeds | 4 training seeds (1 to 4) per setting. |
| Scoring | `test-hedge` tunes the bands on the validation bank and scores every hedger on the test bank. |

The runs come from two rounds on the same banks, `artifacts/runs/exp_overnight/data/tree`:
- **Round 1**, 50 epochs, `artifacts/runs/exp_overnight`: the network's learning rate, the boundary input, the late holder, and a first fee sweep at three values of the fee's smoothing scale.
- **Round 2**, 100 epochs, `artifacts/runs/exp_round2`: the proportional costs, the threshold's learning rate, and a wider fee sweep, stopped partway through ([Limits and open questions](#limits-and-open-questions)).

Each round's folder holds the training histories, the test reports and the logs. Its analysis folder holds two scripts, one that averages the reports over seeds and one that computes the SEs, each with its output saved next to it as a text file. The **Source** lines below give the full paths.

## The network beats the tuned bands under proportional costs

**Question.** With costs proportional to the traded value, does the trained network hedge the tail better than the best tuned band, and where does its edge come from?

**At a 1% cost.**

| Hedger | Tuned value | CVaR 95% ($) | Mean P&L ($) | Std P&L ($) | Mean costs ($) | Trades per path |
|---|---|---|---|---|---|---|
| No hedge | | −15.256 | −0.03 | 7.29 | 0 | 0 |
| Delta, daily | | −10.421 | −5.13 | 2.32 | 5.14 | 191 |
| Fixed band | half-width 0.075 | −4.729 | −1.82 | 1.28 | 1.83 | 52 |
| Whalley-Wilmott | $`\gamma`$ = 1.4 | −4.613 | **−1.60** | 1.31 | **1.61** | **46** |
| **Network** | | **−3.658 ± 0.006** | −1.68 ± 0.03 | **1.06 ± 0.04** | 1.70 ± 0.03 | 60 |

**At a 0.1% cost.** The Whalley-Wilmott grid is widened above $`\gamma`$ = 10 for this cost, since 10 was the top of the default grid and won in round 1.

| Hedger | Tuned value | CVaR 95% ($) | Mean P&L ($) | Std P&L ($) | Mean costs ($) | Trades per path |
|---|---|---|---|---|---|---|
| Delta, daily | | −1.724 | −0.51 | **0.47** | 0.51 | 191 |
| Fixed band | half-width 0.025 | −1.608 | **−0.28** | 0.56 | **0.29** | **94** |
| Whalley-Wilmott | $`\gamma`$ = 20 | −1.545 | −0.32 | 0.50 | 0.32 | 116 |
| **Network** | | **−1.314 ± 0.001** | −0.31 | 0.51 | 0.32 | 114 |

**The network's lead over the best baseline**, network CVaR minus Whalley-Wilmott CVaR, with its SE:

| Cost | Lead ($) | Share of Whalley-Wilmott's tail loss |
|---|---|---|
| 1% | +0.95 ± 0.03 | about a fifth |
| 0.1% | +0.23 ± 0.01 | about 15% |

**Source.** `artifacts/runs/exp_round2/report/cost1_w1e-2_s{1..4}.json` and `artifacts/runs/exp_round2/report/cost01_s{1..4}.json`, averaged in `artifacts/runs/exp_round2/analysis/aggregate.txt`. The leads and SEs are on seed 1, in `artifacts/runs/exp_round2/analysis/bootstrap.txt`. No-hedge scores at 0.1% equal those at 1%.

**What the tables show.** The network has the best tail at both costs. Its lead is many times its standard error, and the four seeds agree to about a cent, so the result does not hang on one lucky run.

The edge does not come from cheaper trading. At 1% the network trades more often than Whalley-Wilmott, pays slightly more in costs, and has a slightly worse mean P&L. What it gains is a narrower spread of outcomes, about a fifth narrower than either band's. That is the trade a CVaR objective asks for. It gives up a little average P&L to make the worst paths less bad. The entropic risk, which the network was not trained on, ranks it first too (`artifacts/runs/exp_round2/analysis/aggregate.txt`), so the edge is not specific to the measure it was trained on.

At 0.1% every hedger sits close to the delta hedge, because trading is cheap. The network's lead is smaller in dollars but still clear. Here it trades as often as Whalley-Wilmott, with the same costs and mean, and still has the milder tail. So it places its trades better, rather than trading less.

**Critique.**
- **The baselines have one parameter for the whole family.** The network reads the contract's maturity, volatility and rate, and adapts its band per contract and per day. Part of its edge may be just that adaptation, which a band tuned per region of the family would also get. That comparison has not been run ([Limits and open questions](#limits-and-open-questions)).
- **The fixed band may be under-tuned at 0.1%.** Its grid steps from 0 to 0.025 to 0.05 shares, coarse around its optimum. Whalley-Wilmott is the reference at this cost, and its grid is fine around its optimum, with neighbours within a cent.
- **One test bank and one family.** The SEs cover another draw of paths from the same family. They say nothing about other volatilities, maturities or models.
- **The mean matters to some writers.** At 1%, a writer who cares about the average P&L more than the tail would still prefer Whalley-Wilmott, which keeps a few more cents on average.
- **Training is long enough.** Round 1 trained the same networks for 50 epochs instead of 100. The longer runs gained about a cent at 1% and nothing at 0.1%, and none of their best epochs falls in the last five. A larger network has not been tried.

## Learning rates

**Question.** Which learning rate trains the network best on the full banks? And does the CVaR threshold $`w`$ need a rate of its own?

**Background.** The CVaR loss is minimized through the Rockafellar-Uryasev form, $`w + \mathbb{E}\big[(L - w)^+\big] / (1 - \alpha)`$, where $`L`$ is the loss of a path. Its minimum over $`w`$ is the CVaR, reached when $`w`$ is the 95% quantile of the loss. So $`w`$ is learned with the network, by Adam, at its own rate `w_lr`. It starts at the 95% quantile of the untrained network's loss, measured on the first training batch. Adam moves it by about `w_lr` per step, so at 1e-2 it needs many epochs to come down to the trained network's tail.

**The network's rate.** Round 1, 1% cost, 50 epochs, one seed (42):

| `lr` | Best validation CVaR loss ($) | Best epoch |
|---|---|---|
| 1e-3 | 3.784 | 50 |
| 3e-3 | 3.656 | 44 |
| **1e-2** | **3.638** | 50 |

**Source.** `artifacts/runs/exp_overnight/hist/lr{1e-3,3e-3,1e-2}.json`.

**The threshold's rate.** Round 2, 1% cost, 100 epochs, 4 seeds, `lr` = 1e-2:

| `w_lr` | Best validation CVaR loss ($) | Test CVaR ($) | Epochs until $`w`$ is within $0.2 of its final value | Jitter of $`w`$, last 20 epochs ($) |
|---|---|---|---|---|
| **1e-2** | 3.618 ± 0.004 | −3.658 ± 0.006 | 37 | 0.003 |
| 1e-1 | 3.618 ± 0.003 | −3.656 ± 0.003 | 7 | 0.03 |
| 3e-1 | 3.620 ± 0.005 | −3.659 ± 0.004 | 8 | 0.04 |

**Source.** `artifacts/runs/exp_round2/hist/cost1_w*_s*.json` and `artifacts/runs/exp_round2/report/cost1_w*_s*.json`, averaged in `artifacts/runs/exp_round2/analysis/aggregate.txt`. The last two columns are for seed 1, in `artifacts/runs/exp_round2/analysis/bootstrap.txt`.

**What the tables show.** A rate of 1e-3 is clearly too slow, as the [Training checks](#training-checks) suggested. Between 3e-3 and 1e-2 the gap is small. 1e-2 is used in every later experiment and is now the default `lr`.

The threshold's rate does not matter here. A faster rate brings $`w`$ to its level within a few epochs instead of about forty, at the price of more jitter. The final scores are the same within the spread between seeds. So the slow threshold of round 1 did not hold training back at a 1% cost. The default `w_lr` stays 1e-2.

**Why the lag is harmless.** This is a hypothesis, not a measurement. While $`w`$ sits above the 95% quantile, the gradient reaches only the paths whose loss exceeds $`w`$, a smaller and more extreme tail. The network then trains on the right direction, the worst paths, just on fewer of them. Once $`w`$ arrives, it trains on the full 5% tail, with time left to finish.

**Critique.**
- The network's rate was chosen on one seed, and the margin between 3e-3 and 1e-2 is small. Since the training is long and the final scores of the four seeds agree, the choice probably does not matter much.
- The threshold's rate was compared at a 1% cost only. With a fixed fee, $`w`$ starts much higher, since the untrained network pays the fee on every daily trade. A faster threshold could matter there. The fee sweep runs at 1e-2 and has not tested it.

## The boundary input adds at most a cent

**Question.** The holder exercises the first time the spot falls to the boundary $`S^*`$, so the put changes regime on that line. The network can read an optional extra input, the distance to the boundary $`\ln\big(S_k / S^*(T - t_k)\big)`$ (`--boundary_input`). Does it hedge better with it?

**Setting.** Round 1, 50 epochs. The late holder is described in the next section.

| Holder | Input off | Input on | On minus off |
|---|---|---|---|
| Tree | −3.668 ± 0.005 | −3.663 ± 0.006 | +0.005 ± 0.006 |
| Late | −3.845 ± 0.009 | −3.849 ± 0.013 | −0.004 ± 0.010 |

Network CVaR 95% on the test bank, in dollars. "On minus off" pairs the two runs of each seed.

**Source.** `artifacts/runs/exp_overnight/report/{cost1,bnd_tree,nobnd_late,bnd_late}_s{1..4}.json`, averaged in `artifacts/runs/exp_overnight/analysis/aggregate.txt`. The paired SEs below are in `artifacts/runs/exp_overnight/analysis/bootstrap.txt`.

**What the table shows.** The input moves the CVaR by about half a cent, on a tail loss of several dollars. The spread between seeds is as large, and the sign differs between the two holders. A paired bootstrap of seed 1's test paths gives a difference of under a cent for both holders. So the input is worth at most a cent, and it stays off by default.

**Why it adds so little.** Per unit of strike, the boundary depends only on the time left, $`r`$ and $`\sigma`$, and the network reads all three. It also reads $`\ln(S_k/K)`$. The distance $`\ln(S_k/S^*) = \ln(S_k/K) - \ln(S^*/K)`$ is therefore a function of inputs it already has. The extra input can only spare the network from learning that function, and 50,000 paths are enough for it to learn it anyway. For the late holder, the distance that matters is to $`0.95\,S^*`$. It differs from the input by a constant, $`\ln 0.95`$, so the input is just as informative there.

**Critique.** These networks trained for 50 epochs, before round 2 showed that training longer gains about a cent. Both arms share that setting, so the comparison holds, but a cent-sized effect is at the edge of what these runs can resolve. The input could also matter more under a model where the boundary is not a function of the other inputs, such as stochastic volatility.

## A late holder changes little for a hedger

**Question.** Every hedger assumes that the holder exercises optimally. What happens when the holder waits too long?

**Setting.** Round 1, 50 epochs. The `late` holder exercises at $`0.95\,S^*`$ (`--exercise late --shift 0.05`), so it waits until the spot is 5% below the boundary. Its banks share their contracts and paths with the tree holder's, and only the exercise days differ. The late holder exercises 35% of the test paths early, and the tree holder 46%. Each network is trained on its own holder's bank, without the boundary input.

| Hedger | Tree holder, CVaR 95% ($) | Late holder, CVaR 95% ($) | Late minus tree ($) |
|---|---|---|---|
| No hedge | −15.26 | −17.51 | −2.25 |
| Delta, daily | −10.42 | −10.58 | −0.16 |
| Fixed band | −4.73 (half-width 0.075) | −4.81 (half-width 0.075) | −0.08 |
| Whalley-Wilmott | −4.61 ($`\gamma`$ = 1.4) | −4.86 ($`\gamma`$ = 2.0) | −0.24 |
| **Network** | **−3.67** | **−3.85** | −0.18 |

**Source.** `artifacts/runs/exp_overnight/report/cost1_s{1..4}.json` (tree) and `artifacts/runs/exp_overnight/report/nobnd_late_s{1..4}.json` (late), averaged in `artifacts/runs/exp_overnight/analysis/aggregate.txt`. Early-exercise shares from `artifacts/runs/exp_overnight/logs/gen_{tree,late}_test.txt`.

**What the table shows.**
- **The unhedged writer suffers most.** The late holder keeps a deep in-the-money put alive after the optimal holder would have exercised, which widens the unhedged spread. The writer gains nothing measurable on average from the holder's mistake.
- **Every hedger loses a little.** The surrogate prices the put for an optimal holder, so between $`0.95\,S^*`$ and $`S^*`$ its delta is that of an exercised put, −1. The late holder's put still has time value there, so its true delta is above −1. The hedges are slightly wrong in that region, and the delta hedge trades on more days.
- **The network keeps its lead.** Its lead over the best baseline is about the same for both holders. The best baseline is the fixed band for the late holder and Whalley-Wilmott for the tree holder. The network is not told that the holder is late. It only sees the late bank's P&L in training.

**Critique.** The network loses about as much as the bands do, so there is no sign that it learned the late rule itself. It keeps its lead because the late holder costs every hedger about the same. A holder who misses the boundary by more than 5%, or at random, has not been tested. The late shift is also known in advance here. A real desk would not know how late its counterparty is.

## Limits and open questions

- **Fixed fees.** A fixed fee per trade, on top of the 1% cost, is the experiment left unfinished. In round 1 the networks were unstable under a fee, and several got stuck while still trading, with a tail loss between the tuned bands' and no hedge's, closer to no hedge. Round 2 trains longer and over a wider range of the fee's smoothing scale $`\kappa`$, and it stopped after 6 of its 20 runs. Its results will be added when all four seeds of every $`\kappa`$ are done.
- **Scores per region of the family.** A family score can hide a region where a hedger is weak, and a band tuned per region is the stronger baseline the [proportional-cost critique](#the-network-beats-the-tuned-bands-under-proportional-costs) asks for. An optional last step scores the test paths per bucket of maturity, volatility and moneyness, with bands tuned per bucket. At 3 × 3 × 2 = 18 buckets, a 10,000-contract validation bank leaves about 28 tail paths per bucket at 95%, so that step needs fewer buckets or a larger validation bank. A bank of 200 contracts × 1,000 paths would take about 3 GB of RAM with the current evaluation (an estimate from the array sizes).
- **The delta hedge's small positive mean.** At zero cost it is under a cent but about two standard errors above 0, and known effects explain only a small part of it ([First evaluation](#first-evaluation)). A bank of 50,000 paths, or a control variate on the mean, would tell chance from a bias.
- **SAC on the new P&L.** Whether SAC is retrained once for one comparison is decided at the end, once Deep Hedging has all its results.

## Reproducing the runs

All commands run from the project root. Times are on a 6-core CPU.

**Checks and first evaluation.**
```
python -m optionpricing.ml.hedging.generate_hedge_dataset --split val --n_contracts 10000 --output_dir artifacts/data/eval1    # ~100 s
python -m optionpricing.ml.hedging.generate_hedge_dataset --split test --n_contracts 20000 --output_dir artifacts/data/eval1   # ~170 s
python main.py test-hedge --baselines_only --test_path artifacts/data/eval1/hedge_test.npz --valid_path artifacts/data/eval1/hedge_val.npz                       # zero cost, ~15 s
python main.py test-hedge --baselines_only --test_path artifacts/data/eval1/hedge_test.npz --valid_path artifacts/data/eval1/hedge_val.npz --proportional 0.01   # 1% cost
python -m pytest tests/integration/test_baselines_vs_tree.py tests/integration/test_hedge_training_learns.py
```

**Banks of the network experiments.** About 10 min per holder.
```
D=artifacts/runs/exp_overnight/data
python -m optionpricing.ml.hedging.generate_hedge_dataset --split train --n_contracts 50000 --output_dir $D/tree                           # ~5.5 min
python -m optionpricing.ml.hedging.generate_hedge_dataset --split val   --n_contracts 10000 --output_dir $D/tree                           # ~1.5 min
python -m optionpricing.ml.hedging.generate_hedge_dataset --split test  --n_contracts 20000 --output_dir $D/tree                           # ~2.5 min
#   the late holder: the same three with --exercise late --shift 0.05 --output_dir $D/late
```

**One network run.** Train, then test. The flags change per experiment, as listed below.
```
python main.py train-hedge --train_path $D/tree/hedge_train.npz --valid_path $D/tree/hedge_val.npz --checkpoint_path <ckpt> --history_path <hist> --epochs 100 --proportional 0.01 --lr 1e-2 --w_lr 1e-2 --seed 1   # ~25 min at 100 epochs
python main.py test-hedge --test_path $D/tree/hedge_test.npz --valid_path $D/tree/hedge_val.npz --checkpoint_path <ckpt> --report_path <report>                                                                 # ~20 s
```

| Experiment | Training flags | Test flags | Epochs |
|---|---|---|---|
| Proportional costs | `--proportional 0.01` or `0.001` | at 0.001, `--risk_aversions 0.1 0.3 0.5 0.7 1.0 1.4 2.0 3.0 10.0 20.0 30.0 50.0 100.0` | 100 |
| Threshold rate | `--w_lr 1e-2`, `1e-1` or `3e-1` | | 100 |
| Network rate | `--lr 1e-3`, `3e-3` or `1e-2`, `--seed 42` | | 50 |
| Boundary input | add `--boundary_input` | | 50 |
| Late holder | the `$D/late` banks | the `$D/late` banks | 50 |

**Aggregates and bootstrap.** `python artifacts/runs/exp_overnight/analysis/aggregate.py` and `python artifacts/runs/exp_overnight/analysis/bootstrap.py`, and the same two scripts in `artifacts/runs/exp_round2/analysis`. The aggregates take seconds. The bootstraps rescore a few networks on the test bank, a few minutes each.
