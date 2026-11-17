# Option Pricing & Hedging

This project aims at pricing and hedging American options. It is built
from first principles in Python and PyTorch. The project goes from
closed-form Black-Scholes to binomial trees and Longstaff-Schwartz
Monte-Carlo, then to a neural network that prices American puts in
microseconds, and a Deep Hedging network that will learn to hedge an
American put under trading costs. The hedging pipeline and its baselines
are built and measured. The network is not trained on the full banks yet.

It is a personal research project, and my first in quantitative finance.

Everything is implemented from scratch on top of NumPy/SciPy and PyTorch:
GBM path simulation, the pricers, the MLP surrogate, the hedging P&L, its
baselines and the hedging network. No quant or RL library is used. Each result
below gives the command, or the function call, that produces it.

## Highlights

- **Four pricers that agree with each other.** European Monte-Carlo
  matches Black-Scholes, and American Longstaff-Schwartz matches the
  binomial tree within Monte-Carlo noise. Integration tests check the
  convergence. With a European control variate, Longstaff-Schwartz prices
  the American call at exactly its Black-Scholes value.
- **LSM Greeks repaired near expiry.** The textbook delta (the derivative of
  the regression fit) left the valid range up to 91.6% of the time just
  before maturity. A separate regression of pathwise deltas cut the
  out-of-the-money delta error from 0.2-0.9 to under 0.02, and made hedging
  at the LSM delta as good as hedging at the exact delta.
- **No-trade bands more than halve delta hedging's tail loss.** On 20,000
  American puts at a 1% trading cost, a tuned Whalley-Wilmott band raises
  the 95% CVaR of the hedger's P&L from -$10.42 for daily delta hedging to
  -$4.61. These are the baselines the Deep Hedging network must beat. The
  network is not trained on the full banks yet.
- **A neural pricer for American puts.** An MLP trained on binomial-tree
  prices gives price, delta and gamma in 6-9 µs per option in batches,
  360-510× faster than the tree it learned from. Its mean error is $0.0004
  at K = 100 (0.43 bp of implied volatility) against an independent, finer
  tree. Every design choice was picked by seeded ablations.
- **Tested.** 883 tests (unit, integration, CLI smoke tests), plus a
  mutation check on the surrogate's evaluation code.

## Methods and references

| Method | Used for | Reference |
|---|---|---|
| Black-Scholes formula | European prices and deltas, the surrogate's baseline | Black & Scholes (1973), *Journal of Political Economy* |
| Binomial tree (CRR) | American prices, the hedging environment's put values | Cox, Ross & Rubinstein (1979), *Journal of Financial Economics* |
| Binomial Black-Scholes with Richardson extrapolation (BBSR) | The surrogate's training labels | Broadie & Detemple (1996), *Review of Financial Studies* |
| Least-squares Monte-Carlo (LSM) | American exercise decisions and Greeks along simulated paths | Longstaff & Schwartz (2001), *Review of Financial Studies* |
| Pathwise derivative estimates | The LSM delta regression | Broadie & Glasserman (1996), *Management Science* |
| Soft Actor-Critic (SAC) | The legacy RL hedging agent | Haarnoja et al. (2018), *ICML* |
| No-trade band hedging (Whalley-Wilmott, and a fixed band) | The hedging baselines under costs, tuned on validation paths | Whalley & Wilmott (1997), *Mathematical Finance* |
| Deep Hedging | The hedger being built, a network trained through the P&L (not trained on the full banks yet) | Buehler et al. (2019), *Quantitative Finance* |
| No-transaction band network | The Deep Hedging policy's output, a band around delta | Imaki et al. (2021), *arXiv:2103.01775* |

## Project status

| Area | Status |
|---|---|
| European and American pricers | Done |
| LSM Greeks and exercise rule | Done |
| MLP surrogate for American puts | Done |
| RL hedging (SAC) | Legacy: the first hedger, kept for one later comparison with Deep Hedging |
| Deep Hedging | In progress: the pipeline is built and the baselines are measured. The network is not trained on the full banks yet. The CLI and the training runs come next |
| Dividend-paying stocks, fast American-put benchmarks | Planned |

The detailed plan is in the [roadmap](.claude/PLAN.md) and the finished
workstreams are in the [roadmap archive](.claude/PLAN_ARCHIVE.md). Every problem
found along the way, with its root cause and fix, is in the
[design decisions and bug log](.claude/BUGS.md).

## What's inside

| Layer | What it does |
|---|---|
| **European pricing** | Closed-form Black-Scholes and Monte-Carlo simulation, cross-checked against each other. |
| **American pricing** | A Cox-Ross-Rubinstein binomial tree, a vectorized batch tree (CRR, BBS, BBSR), and Longstaff-Schwartz least-squares Monte-Carlo with simple or Laguerre polynomial bases, an optional European control variate, and delta/gamma from a regression of pathwise deltas. |
| **ML surrogate pricer** | A smooth MLP trained on binomial-tree prices of American puts, giving price, delta and gamma in microseconds per option in batches. The Deep Hedging pipeline reads its premiums, deltas and gammas. |
| **Deep Hedging** | One differentiable P&L for the short American put (cash account, proportional and fixed costs, exercise at the tree's boundary), the baselines (no hedge, delta, a fixed band, Whalley-Wilmott) and a no-transaction band network trained on a risk measure of that P&L. The network is not trained on the full banks yet. |
| **Legacy RL hedging** | The earlier Soft Actor-Critic agent and its Gym-style environment, kept in `legacy/` subpackages. |

## Results

Unless stated otherwise, numbers come from the commands shown with their
default parameters (`S0 = K = 100`, `r = 5%`, `T = 1 year`, `σ = 20%`).
Every result is also indexed on one page in [`docs/RESULTS.md`](docs/RESULTS.md),
with links to the full analysis of each topic.

### Pricing models agree with each other

```
python main.py price --option_class both                     # call
python main.py price --option_class both --option_type put   # put
```

| | Monte-Carlo | Black-Scholes | Binomial tree | Longstaff-Schwartz |
|---|---|---|---|---|
| Call | 10.30 | 10.45 | 10.45 | 10.28 |
| Put  | 5.63  | 5.57  | 6.09  | 6.13  |

Rerun 2026-11-05. European Monte-Carlo tracks Black-Scholes, and American
Longstaff-Schwartz tracks the binomial tree. The convergence is tested in
`tests/integration/test_pricing_convergence.py`. The American put is worth
more than the European put because of the early-exercise premium, the
value of the right to exercise before maturity. An American
call on a non-dividend stock is worth exactly its European price, which the
tree reproduces.

`price` runs plain LSM, a Monte-Carlo estimate (about ±0.1 here). With the
European option as a **control variate** (`LongstaffSchwartz(...,
sigma=..., control_variate=True)`), the LSM call prices at 10.4506, the
Black-Scholes value. The regression then targets the cashflow minus the
European payoff, which is exactly 0 on every path not exercised early, so
the regression noise that made plain LSM exercise calls early disappears.
`train-hedge-legacy` and `test-hedge-legacy` enable it for calls only. For puts the plain
exercise rule was worth more out of sample, that is when the rule is fitted
on 10,000 paths and then applied to 100,000 fresh ones (6.078 for the plain
rule, 6.009 with the control variate, true price 6.084). Details: [`.claude/BUGS.md`, "LSM early-exercises American calls
and mis-estimates delta"](.claude/BUGS.md#b-006).

### LSM delta and gamma, accurate up to expiry

**Longstaff-Schwartz, briefly.** Pricing an American option means deciding,
at every date along a simulated path, whether exercising now beats holding
on. That needs the value of holding on (the *continuation value*), which
depends on the unsimulated future. LSM walks the paths backward from expiry
and, at each date, regresses the discounted future cashflow on the current
stock price across the in-the-money paths. The fit estimates the
continuation value, and comparing it with the exercise payoff gives each
path's exercise date.

**The problem.** The obvious Greeks are the derivatives of that fitted
polynomial, and near maturity they are unreliable. A call's delta must lie
in `[0, 1]`, but with the default degree-3 fit it fell outside that range up
to 91.6% of the time at the last date before expiry, with values in the
tens. The hedging environment feeds this delta to the policy, and the delta
baseline trades on it directly.

**Why.** As expiry approaches, the option value develops a sharp corner at
`S = K`. A smooth global polynomial cannot bend that sharply, so it wiggles,
and differentiating amplifies the wiggle. A higher degree makes it worse,
not better.

**The fix.** Delta gets its own regression. Each path's *pathwise delta*
(the derivative of its discounted cashflow with respect to today's spot,
`±S_exercise/S_t` discounted if it pays off) is regressed over all paths on
the polynomial basis plus the European Black-Scholes delta, and gamma is the
derivative of that fit. Against exact references (Black-Scholes for the
call, a binomial tree for the put), on 4 seeds:

| Delta error (mean absolute) | Derivative of the value fit | Pathwise regression |
|---|---|---|
| In the money | 0.03-0.18 | 0.001-0.035 |
| Out of the money | 0.2-0.9 | 0.001-0.017 |
| Largest \|delta\| | 13-48 | 1.8 |

With this delta, hedging at the LSM delta (`python main.py test-hedge-legacy
--policy delta`) is as good as hedging at the exact delta (Black-Scholes for
the call, the binomial tree for the put). P&L std (×K), not re-verified
since 2026-09-30:

| Delta hedge at | Call | Put |
|---|---|---|
| LSM delta, derivative of the value fit (before) | 0.060 | 0.044 |
| LSM delta, pathwise regression (now) | **0.014** | **0.012** |
| Exact delta | 0.015 | 0.013 |

The full incident, including a regularized fix that was tried and reverted,
is in [`.claude/BUGS.md`, "LSM delta/gamma estimates unreliable near
maturity"](.claude/BUGS.md#b-003) and ["LSM early-exercises American calls
and mis-estimates delta"](.claude/BUGS.md#b-006).

**The regression basis does not matter at degree 3.** `--reg_type laguerre`
uses the Laguerre polynomials `L_0..L_d` of `S/K`, which span the same
functions as `1, x, ..., x^d`, so an unregularized least-squares fit gives
the same values in either basis. On 10,000 paths and 2 seeds:

| Degree | Put price, simple / Laguerre | Same exercise date | Max delta difference |
|---|---|---|---|
| 3 | 6.131 / 6.131, 6.068 / 6.068 | 100% | 1e-9 |
| 6 | 6.147 / 6.150, 6.120 / 6.116 | 99.0-99.6% | 0.24-0.29 |
| 10 | 6.180 / 6.169, 6.076 / 6.106 | 59-66% | 0.33-0.39 |

At higher degrees they differ only through floating-point conditioning, and
higher degrees are worse for both. Every LSM consumer defaults to `simple`.

The Laguerre basis is there because Longstaff and Schwartz use it in their
original paper (2001). Their version weights each polynomial by `e^{-x/2}`,
which changes the functions it spans, so this implementation may be a
misreading of the paper. This is being looked into, see the
[roadmap](.claude/PLAN.md#longstaff-schwartz).

### MLP surrogate: a neural pricer for American puts

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset   # 2^19 tree-priced puts, ~15 min on 6 cores
python main.py test-ml --seed 43                                  # score the shipped model (--seed: its training split)
python main.py train-ml --seed 43                                 # or retrain it (the defaults), ~25-30 min
```

The trained model is in the repository,
`artifacts/checkpoints/put_surrogate.pth` (seed 43), and loads with
`PutSurrogate.load`. Scoring it needs the dataset, since its test split is
part of the evaluation. Retraining overwrites that file, and the result can
differ slightly from the shipped model, since training is not bit-for-bit
reproducible across machines and thread counts.

Here is how to price with it next to the tree it learned from (run from the
project root):

```python
import numpy as np

from optionpricing.ml.surrogate.tree.tree_pricer import PutSurrogate
from optionpricing.pricing.american.batch_binomial_tree import price_put_batch

surrogate = PutSurrogate.load("artifacts/checkpoints/put_surrogate.pth")
m = np.array([0.9, 1.0, 1.1])  # S / K: in, at and out of the money
res = surrogate.price(m, T=1.0, r=0.05, sigma=0.2)  # per unit of strike
tree = price_put_batch(m, 1.0, 0.05, 0.2)  # the 1000-step BBSR tree it learned from
print("surrogate price at K = 100:", np.round(100 * res.price, 4), "delta:", np.round(res.delta, 4))
print("tree price at K = 100:     ", np.round(100 * tree.price, 4), "delta:", np.round(tree.delta, 4))
```

```
surrogate price at K = 100: [11.4919  6.0892  2.9856] delta: [-0.6832 -0.4111 -0.2235]
tree price at K = 100:      [11.4928  6.0904  2.9865] delta: [-0.6833 -0.4111 -0.2236]
```

**What it learns.** Per unit of strike, an American put's price depends on
the moneyness `m = S/K`, the maturity `T`, the rate `r` and the volatility
`σ`. The network learns only the early-exercise premium, the American price
minus the European one. It draws its data from 2^19 puts priced by a 1000-step BBSR binomial
tree. The price is Black-Scholes plus that premium, and delta and gamma come
from automatic differentiation through the same sum. The training domain is
`T` from 1 week to 5 years, `σ` 5-60% and `r` 0-8%. Moneyness is sampled
through `z = ln(m)/(σ√T)`, the distance to the strike in standard deviations,
in [-3.5, 3] (and `m` in [0.4, 2.5]). Queries outside the domain are not
extrapolated: they go to Black-Scholes far out of the money, to the payoff
deep in it, or to the tree. A payoff floor keeps every price at or above
`max(1 - m, 0)`.

**The model.** 4 hidden layers of 128 SiLU units with a softplus output (the
premium is never negative), trained on the premium error plus 0.3 times the
delta error, with Adam and a cosine decay over 400 epochs. Every choice was
tested against its alternatives by seeded ablations, scored by the 2 sd rule:
a difference counts only if it exceeds about 2 seed standard deviations.

| Choice | Tested against | Why it won |
|---|---|---|
| Delta error in the loss, weight 0.3 | 0, 1, 3 | best price, and without the delta term delta is ~7× worse |
| Softplus output | linear | no-arbitrage violations fall from 0.73% (linear) to 0.06% (softplus) of grid points |
| SiLU activations | ReLU | ReLU has ~25× the error and no gamma |
| Width 128 | 256 | 256 is within 0.6 seed sd on every error, at 2.6× the training time |
| No shape penalty, no L-BFGS finish | penalty weight 0.1, 200 L-BFGS steps | worse or within noise, and slower |
| 400 epochs | 200, 800 | 800 lowers the delta error but raises the price error |
| BBSR teacher, 1000 steps | CRR and BBS, 500 and 2000 steps | lowest mean price error, and 2000 steps halve the worst error at 3.5× the cost |
| One model from 1 week to 5 years | `T` from 3 months only | simpler to ship, at 1.3-2.7× the error beyond 3 months (measured before the output-initialization fix) |

**Results.** Each model is scored against an independent 4000-step BBSR
tree. It has 4× the teacher's steps, so it is not the labels the network
learned. This "scoring" grid has 2342 puts: 27 values of `z` × 6 of `σ` × 8 maturities
from 1 week to 5 years × `r` in {2%, 6%}. Price errors are in **10⁻⁴ $ at
K = 100**, a hundredth of a cent, so 7.9 means $0.00079. "5 seeds" is the
configuration's mean ± standard deviation over seeds 42-46. "Shipped" is the
checkpoint, seed 43, the median seed on implied-vol error.

The implied-vol error is how far `σ` would have to move for the true price
to equal the surrogate's, in basis points (1 bp = 0.0001 in `σ`). It enables me to put
errors on options of very different prices on one scale to better compare them.

| Puts | Points | Mean error, 5 seeds | Mean error, shipped | Worst error, 5 seeds | Worst error, shipped | IV error (bp), 5 seeds | IV error (bp), shipped |
|---|---|---|---|---|---|---|---|
| near the money (`\|z\|` < 1) | 646 | 7.9 ± 0.3 | 7.9 | 166 ± 34 | 126 | 0.33 ± 0.01 | 0.35 |
| `T` < 1 month | 494 | 0.8 ± 0.2 | 0.9 | 12 ± 2 | 11 | 0.28 ± 0.06 | 0.29 |
| `T` 1-3 months | 455 | 1.5 ± 0.1 | 1.7 | 14 ± 4 | 14 | 0.31 ± 0.06 | 0.31 |
| `T` ≥ 3 months | 690 | 9.0 ± 0.7 | 8.7 | 166 ± 34 | 126 | 0.61 ± 0.05 | 0.59 |
| exercised today | 703 | 4.5 ± 0.7 | 3.8 | 114 ± 32 | 69 | - | - |
| all | 2342 | 4.5 ± 0.4 | 4.2 | 166 ± 34 | 126 | 0.44 ± 0.04 | 0.43 |

The maturity rows are the puts not exercised today. Measured 2026-11-10. The
shipped model's numbers reproduce with `test-ml --seed 43`.

- **The typical error is a fraction of a cent.** It is $0.0004 on average,
  0.43 bp of implied volatility. For scale, a one-cent tick on a 1-year
  at-the-money put is about 2.6 bp.
- **Delta** is within 9.0e-5 of the reference tree's on average, and within
  1.4e-4 near the money (shipped model).
- **The price surface has the right shape.** An American put's price
  falls and is convex in moneyness, and it rises with `T` and with `σ`. On
  the grid puts not exercised today, the shipped model breaks one of these
  conditions at only 0.12% of the points, a reasonable rate for a fitted
  network. The exercised puts are not representative here. Their price is
  the payoff, so most conditions hold with equality, and any tiny wobble of
  the network around them is flagged ([`.claude/BUGS.md`, "Shape-violation
  rate inflated by exercised points"](.claude/BUGS.md#b-027)). `test-ml`
  measures these rates. Training can also penalize the violations, but the
  shipped model is trained without that penalty, which made prices worse in
  the [ablations](docs/results/surrogate-training.md).
- **Speed.** Priced in one call, the grid takes 6-9 µs per option, against
  3.1-3.2 ms for the 1000-step tree it learned from. That is 360-510× faster.

**Known limitations.**
- **The worst errors sit near the money beyond 3 months.** They are
  $0.013-0.022 over the 5 seeds, and $0.0126 for the shipped model. These
  are the largest errors on the 2342 scoring puts, not a bound for the
  model. A put off the grid can have a larger error, as the next point
  shows.
- **Rates near zero are its weakest region**, below the grid's rates. On the
  test split, the largest errors (up to about $0.10) were long-dated, deep
  in-the-money puts at `r` below 0.2%. This was measured on earlier
  checkpoints of the same configuration, not on the shipped one.
- **One option at a time is no faster than the tree** (2.1-2.3 ms), because
  the autograd passes for delta and gamma dominate. The speed-up needs
  batches.
- **Puts only**, on the domain above. Anything outside it goes to the tree
  or a closed form.

How each choice was tested, with the per-seed results, is in
[`docs/results/surrogate-training.md`](docs/results/surrogate-training.md).
This surrogate replaced a first version trained on Longstaff-Schwartz
prices, whose write-up is in
[`docs/results/legacy-surrogate.md`](docs/results/legacy-surrogate.md).

### Deep Hedging: a hedger trained through its P&L

The hedger sells an American put and trades the stock to offset it. Under
trading costs, the textbook delta hedge is no longer the best it can do:
rebalancing every day costs more than the risk it removes. Deep Hedging
(Buehler et al. 2019) learns the hedge by simulation. A network chooses the
position on every day of many simulated paths. The P&L of each path is a
known formula, so training minimizes a risk measure of the final P&L
directly, by backpropagating through whole paths. There is no reward
shaping, so the training objective is the metric the hedge is judged on.

| Piece | What it does |
|---|---|
| P&L | One differentiable function books every hedge: the premium received at the sale, a cash account at the risk-free rate, the gains of the shares held, the cost of every trade and the payoff at exercise. Training, testing and every baseline go through it. |
| Exercise | The holder exercises on the first day the stock is at or below the early-exercise boundary of a binomial tree. |
| Pricing model | The MLP surrogate above gives the premium, delta and gamma of a different contract on every path, in microseconds. |
| Policy | A no-transaction band network (Imaki et al. 2021). From the log-moneyness, the time left, `σ`, `r`, the surrogate's delta and gamma, and the position it holds, it outputs a band around that delta. The position is held while it stays inside the band and moved to the band's edge when it leaves. The untrained network starts as the delta hedge. |
| Objective | The 95% CVaR of the final P&L, or the entropic risk or the variance. |

**Status.** The pipeline is built: the bank of simulated paths, the
network, its training loop and the evaluation. **The network is not
trained on the full banks yet.** The first measured results are the
baselines it will be compared with, below. Its own results will be added
here.

#### Under 1% costs, no-trade bands cut delta hedging's tail loss by more than half

The hedger sells an American put at the surrogate's price and trades the
stock daily until the holder exercises or the put expires. The holder
exercises on the first day the stock is at or below the early-exercise
boundary of a binomial tree. The P&L books the premium, a cash account at
the risk-free rate, the cost of every trade and the payoff. It is
discounted to the sale date, in dollars at K = 100. The 95% CVaR is the
mean P&L of the worst 5% of paths, so higher is better.

```
python -m optionpricing.ml.hedging.generate_hedge_dataset --split val --n_contracts 10000 --output_dir artifacts/data/eval1    # ~2 min on 6 cores
python -m optionpricing.ml.hedging.generate_hedge_dataset --split test --n_contracts 20000 --output_dir artifacts/data/eval1   # ~3 min
```

The scores come from `evaluate_hedgers`, at each cost, on these two banks.
The `test-hedge` command that will wrap it is the next step of the roadmap.

```python
from optionpricing.hedging.pnl import HedgeSettings
from optionpricing.ml.hedging.hedge_evaluation import EvalConfig, evaluate_hedgers

for cost in (0.0, 0.01):  # about 15 s each
    report = evaluate_hedgers(
        "artifacts/data/eval1/hedge_test.npz",
        "artifacts/data/eval1/hedge_val.npz",
        config=EvalConfig(hedging=HedgeSettings(proportional=cost)),
    )
    print(report.metrics, report.tuned)
```

**At zero cost, the daily delta hedge leaves 5.6% of the unhedged standard deviation.**

| Hedger | Mean P&L (¢) | Std P&L ($) | 95% CVaR ($) |
|---|---|---|---|
| No hedge | -3.3 ± 5.2 | 7.29 | -15.26 |
| **Delta, daily** | +0.7 ± 0.3 | **0.40** | **-0.93** |

**At a 1% proportional cost, the no-trade bands more than halve the tail loss.**

| Hedger | Tuned parameter | Mean costs ($) | Trades per path | Std P&L ($) | 95% CVaR ($) |
|---|---|---|---|---|---|
| No hedge | | 0 | 0 | 7.29 | -15.26 |
| Delta, daily | | 5.14 | 191 | 2.32 | -10.42 |
| Fixed band | half-width 0.075 shares | 1.83 | 52 | **1.28** | -4.73 |
| **Whalley-Wilmott band** | risk aversion 1.4 per $ | **1.61** | **46** | 1.31 | **-4.61** |

Protocol: one contract per path, drawn uniformly over `T` from 3 months to
1 year, `σ` 15-30%, `r` 2-6% and `S_0/K` 0.9-1.1. The test bank has 20,000
paths (seed 3). Each band's parameter was tuned on CVaR on a separate
validation bank of 10,000 paths (seed 2), over the default grids of
`hedge_evaluation.py`: 13 half-widths from 0 to 0.4 shares and 9 risk
aversions from 0.1 to 10 per $. Deltas come from the
surrogate, settlement is in cash. Mean P&L is given ± one standard error.
Run on 2026-11-17.

- **The P&L shows no visible bias.** A hedger who sells at a fair premium
  makes 0 on average. The no-hedge mean is within one standard error of 0.
  The delta hedge's mean is 2.4 standard errors above 0, but it is under a
  cent, small next to its $0.40 standard deviation. Its source is still
  being checked.
- **Daily delta hedging spends most of the premium on costs.** It pays
  $5.14 per path on average, 83% of the median premium of $6.16. The two
  bands trade about 4 times less and cut the costs by about two thirds.
- **Whalley-Wilmott is the best baseline.** Its CVaR is $0.12 above the
  fixed band's, with a paired bootstrap standard error of $0.03. Its width
  follows the put's gamma and the share price, so one tuned parameter fits
  every contract of the family. A fixed width does not follow gamma, so it
  fits the family less well.

**The legacy hedger.** Before Deep Hedging, the project hedged with a
Soft Actor-Critic agent. It is kept in the `legacy/` subpackages, and its
results are in [`docs/results/hedging-rl.md`](docs/results/hedging-rl.md).

Full analysis and design: [`docs/results/hedging-training.md`, First
evaluation](docs/results/hedging-training.md#first-evaluation).

## Setup

```
pip install -r requirements.txt
pip install -e .        # optional: makes `optionpricing` importable from anywhere
```

Pinned versions match the development environment (Python 3.13, PyTorch
2.11 CPU build). No GPU is needed: training falls back to CPU when CUDA is
unavailable. Generated datasets, checkpoints, logs and experiment runs go to
`artifacts/`, which is gitignored except for the trained surrogate,
`artifacts/checkpoints/put_surrogate.pth`.

## Usage

Everything runs through `main.py` from the project root:

```
python main.py --help                  # list all commands
python main.py <command> --help        # list a command's flags
```

| Command | Purpose |
|---|---|
| `price` | Compare pricing models (`--option_class {EU,US,both}`, `--option_type {call,put}`, plus strike, volatility, rate, maturity and sampling flags). |
| `train-ml` | Train the put surrogate (`PutSurrogate`) on `artifacts/data/tree_put_dataset.npz`, saved to `artifacts/checkpoints/put_surrogate.pth`. |
| `test-ml` | Evaluate the put surrogate: test-split errors, and a factorial grid against an independent finer binomial tree (price, implied-vol, delta and gamma errors, no-arbitrage violations, timing). |
| `train-hedge-legacy` | Train the legacy SAC hedging policy (`--action_mode {residual,gap_fraction,band}`, `--prop_tc`, `--fixed_tc`, validation checkpointing via `--val_paths`/`--val_metric`). |
| `test-hedge-legacy` | Evaluate the legacy SAC policy or a baseline (`--policy {rl,delta,delta_band,no_hedge}`): mean P&L, std P&L, 95% VaR and 95% CVaR over simulated episodes. |

The surrogate's dataset and the Deep Hedging bank of scenarios have their own
entry points, run as modules. Both are seeded and resumable after an
interruption:

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset            # artifacts/data/tree_put_dataset.npz
python -m optionpricing.ml.hedging.generate_hedge_dataset --split train   # artifacts/data/hedge_train.npz, ~5 min on 6 cores
```

Every flag of every command, and of the dataset generators, is in its
`--help`. [`docs/architecture.md`, "Commands"](docs/architecture.md#commands)
maps each command to its script.

## Testing

```
python -m pytest tests/unit                        # fast (~30 s)
python -m pytest tests/integration                 # convergence checks and CLI smoke tests (~2.5 min)
python -m pytest tests/                            # everything
python -m tests.mutation.tree_evaluation_mutants   # the evaluation's tests must catch 14 planted bugs
```

883 tests (2026-11-10, `pytest --collect-only -q`). The 784 unit tests
cover the closed-form formulas, basis-function derivatives against finite
differences, the batch tree against hand-worked trees, the surrogate's
dataset, model, pricer, training and evaluation, the hedging environment's
reward arithmetic, the policy's action modes, the baselines, the SAC
training loop and the repository's hooks. The 99 integration tests cover
Monte-Carlo and LSM convergence to their closed-form and tree counterparts,
seeded policy comparisons, the ablation runner, and a CLI smoke test for
every command.

## Architecture

```
optionpricing/          the library, one importable package
  core/                 GBM simulator, pricer base class, generic metrics
  pricing/european/     Black-Scholes, Monte-Carlo
  pricing/american/     binomial trees, Longstaff-Schwartz, basis functions
  hedging/              Deep Hedging building blocks (risk measures, costs, settlement, scenario simulator, exercise dates, P&L and baselines so far)
  hedging/legacy/       SAC's hedging env, option values, baselines, P&L evaluation
  ml/surrogate/         MLP surrogate pricer (tree/, legacy/)
  ml/hedging/           Deep Hedging pipeline (the bank of scenarios, the band network, its training and evaluation)
  ml/hedging/legacy/    SAC hedging agent
  ml/experiments/       research drivers (ablation runner)
main.py                 the CLI entry point
scripts/                one module per main.py command
tests/                  unit/, integration/, mutation/
docs/                   architecture.md (code map), RESULTS.md (results index), results/
artifacts/              generated datasets, checkpoints, logs, runs (gitignored, except the trained surrogate)
```

Pricers are grouped by option class, everything learned lives in
`optionpricing/ml/`, and the hedging problem the learners train in lives in
`optionpricing/hedging/`. The SAC agent and its environment are in `legacy/`
subpackages, next to the Deep Hedging stack. Dependencies point one way: `core` <- `pricing` <-
`hedging` <- `ml`. [`docs/architecture.md`](docs/architecture.md) gives
each file's role, and each module's docstrings give its API.

## How it was built

This project is also where I learned agentic development. The code was
written with Claude Code (Anthropic's coding agent), inside a workflow set
up for it: a roadmap entry before each piece of work, a bug log with root
causes, repository hooks that format and lint every edit, run its unit
tests and gate every commit, and a checklist-driven review of each change
by a second agent before it is committed (all in `.claude/`). Almost every
line was reviewed and questioned by me. I use Claude to implement, to
understand what I am doing and why it works, and to judge how good the
result is.

## Further reading

- [`docs/architecture.md`](docs/architecture.md): the module reference, every
  command's flags, the test layout and design notes.
- [`docs/results/`](docs/results/): measured results in full, on
  [surrogate training and ablations](docs/results/surrogate-training.md),
  [the Deep Hedging design](docs/results/hedging-training.md),
  [the legacy SAC hedger](docs/results/hedging-rl.md),
  [hedging baselines under costs](docs/results/hedging-costs.md) and
  [the legacy LSM-trained surrogate](docs/results/legacy-surrogate.md).
- [Design decisions and bug log](.claude/BUGS.md): every problem found, with
  its symptom, root cause and fix.
- [Roadmap](.claude/PLAN.md) and [roadmap archive](.claude/PLAN_ARCHIVE.md):
  what is being worked on, what comes next, and what is done.

## License

Released under the [MIT License](LICENSE).
