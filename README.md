# Option Pricing & Hedging

Pricing and hedging American options, built from first principles in
Python and PyTorch. The project goes from closed-form Black-Scholes to
binomial trees and Longstaff-Schwartz Monte-Carlo, then to a neural network
that prices American puts in microseconds, and a reinforcement-learning agent
that learns to hedge an American option under trading costs.

A personal research project, and my first in quantitative finance.

Everything is implemented from scratch on top of NumPy/SciPy and PyTorch:
GBM path simulation, the pricers, the MLP surrogate, the hedging environment
and the Soft Actor-Critic agent. No quant or RL library is used. Each result
below is produced by a command in this repository.

## Highlights

- **Four pricers that agree with each other.** European Monte-Carlo
  matches Black-Scholes, and American Longstaff-Schwartz matches the
  binomial tree within Monte-Carlo noise; integration tests check the
  convergence. With a European control variate, Longstaff-Schwartz prices
  the American call at exactly its Black-Scholes value.
- **LSM Greeks repaired near expiry.** The textbook delta (the derivative of
  the regression fit) left the valid range up to 91.6% of the time just
  before maturity. A separate regression of pathwise deltas cut the
  out-of-the-money delta error from 0.2-0.9 to under 0.02, and made hedging
  at the LSM delta as good as hedging at the exact delta.
- **An RL hedger that beats delta hedging on the American put**
  (snapshot, 2026-10). The SAC agent's hedging error is 19-25% below daily
  delta hedging on the put, and on par with it on the call. Under trading
  costs, a learned no-trade band matches the best hand-tuned band within
  seed noise or beats it, without any tuning.
- **A neural pricer for American puts (in progress).** An MLP trained on
  binomial-tree prices answers in microseconds per option in batches, with
  sub-cent typical error. The final model is still being selected.
- **Tested.** 870 tests (unit, integration, CLI smoke tests), plus a
  mutation check on the surrogate's evaluation code.

## Methods and references

| Method | Used for | Reference |
|---|---|---|
| Black-Scholes formula | European prices and deltas, the surrogate's baseline | Black & Scholes (1973), *Journal of Political Economy* |
| Binomial tree (CRR) | American prices, the hedging environment's put values | Cox, Ross & Rubinstein (1979), *Journal of Financial Economics* |
| Binomial Black-Scholes with Richardson extrapolation (BBSR) | The surrogate's training labels | Broadie & Detemple (1996), *Review of Financial Studies* |
| Least-squares Monte-Carlo (LSM) | American exercise decisions and Greeks along simulated paths | Longstaff & Schwartz (2001), *Review of Financial Studies* |
| Pathwise derivative estimates | The LSM delta regression | Broadie & Glasserman (1996), *Management Science* |
| Soft Actor-Critic (SAC) | The RL hedging agent | Haarnoja et al. (2018), *ICML* |
| No-trade band hedging (after Whalley & Wilmott; band widths tuned empirically) | The hedging baseline under costs | Whalley & Wilmott (1997), *Mathematical Finance* |
| Deep Hedging | Planned (see Project status) | Buehler et al. (2019), *Quantitative Finance* |

## Project status

| Area | Status |
|---|---|
| European and American pricers | Done |
| LSM Greeks and exercise rule | Done |
| MLP surrogate for American puts | ⚠️ **In progress:** built and trained; the final configuration is being chosen by seeded ablations |
| RL hedging (SAC) | Working, under active development: the results below are a snapshot |
| Deep Hedging, dividend-paying stocks, fast American-put benchmarks | Planned |

The detailed plan is in the [roadmap](.claude/PLAN.md), finished
workstreams in the [roadmap archive](.claude/PLAN_ARCHIVE.md). Every problem
found along the way, with its root cause and fix, is in the
[design decisions and bug log](.claude/BUGS.md).

## What's inside

| Layer | What it does |
|---|---|
| **European pricing** | Closed-form Black-Scholes and Monte-Carlo simulation, cross-checked against each other. |
| **American pricing** | A Cox-Ross-Rubinstein binomial tree, a vectorized batch tree (CRR, BBS, BBSR), and Longstaff-Schwartz least-squares Monte-Carlo with simple or Laguerre polynomial bases, an optional European control variate, and delta/gamma from a regression of pathwise deltas. |
| **ML surrogate pricer** | A smooth MLP trained on binomial-tree prices of American puts, giving price, delta and gamma in microseconds per option in batches. Final configuration pending. Standalone: the hedging environment prices with `LongstaffSchwartz` directly. |
| **RL hedging** | A Soft Actor-Critic agent trained in a custom Gym-style environment to hedge an American option until its exercise date, under proportional and fixed trading costs, compared against delta-hedge, no-trade-band and no-hedge baselines. Under active development. |

## Results

Unless stated otherwise, numbers come from the commands shown with their
default parameters (`S0 = K = 100`, `r = 5%`, `T = 1 year`, `σ = 20%`).

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
Longstaff-Schwartz tracks the binomial tree; the convergence is tested in
`tests/integration/test_pricing_convergence.py`. The American put is worth
more than the European put because early exercise has value. An American
call on a non-dividend stock is worth exactly its European price, which the
tree reproduces.

`price` runs plain LSM, a Monte-Carlo estimate (about ±0.1 here). With the
European option as a **control variate** (`LongstaffSchwartz(...,
sigma=..., control_variate=True)`), the LSM call prices at 10.4506, the
Black-Scholes value. The regression then targets the cashflow minus the
European payoff, which is exactly 0 on every path not exercised early, so
the regression noise that made plain LSM exercise calls early disappears.
`train-hedge` and `test-hedge` enable it for calls only: for puts the plain
exercise rule was worth more out of sample (6.078 for the plain rule, 6.009
with the control variate, true price 6.084). Details: [`.claude/BUGS.md`, "LSM early-exercises American calls
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

With this delta, hedging at the LSM delta (`python main.py test-hedge
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

### MLP surrogate: a neural pricer for American puts

> [!WARNING]
> **Work in progress: the final model is not chosen yet.** Seeded ablations
> are still selecting the configuration (`.claude/PLAN.md`, "Surrogate
> ablations and final model"). This section will get the final model's
> results, the choices behind it and its known limitations once it is
> trained. Until then, the candidates' measured results are in
> [`docs/results/surrogate-training.md`](docs/results/surrogate-training.md).

<!-- TODO(final model): results table, choices table, known limitations, timing. -->

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset   # 2^19 tree-priced puts
python main.py train-ml                                          # the final configuration's flags: to come
python main.py test-ml
```

**What it learns.** Per unit of strike, an American put's price depends on
the moneyness `m = S/K`, the maturity `T`, the rate `r` and the volatility
`σ`. The network learns only the early-exercise premium, the American price
minus the European one, from 2^19 puts priced by a 1000-step BBSR binomial
tree. The price is Black-Scholes plus that premium, and delta and gamma come
from automatic differentiation through the same sum. The training domain is
`T` from 1 week to 5 years, `σ` 5-60% and `r` 0-8%. Moneyness is sampled
through `z = ln(m)/(σ√T)`, the distance to the strike in standard deviations,
in [-3.5, 3] (and `m` in [0.4, 2.5]). Queries outside the domain are not
extrapolated: they go to Black-Scholes far out of the money, to the payoff
deep in it, or to the tree. A payoff floor keeps every price at or above
`max(1 - m, 0)`.

This surrogate replaced a first version trained on Longstaff-Schwartz
prices, whose write-up is in
[`docs/results/legacy-surrogate.md`](docs/results/legacy-surrogate.md).

### RL hedging

The hedger is short an American option and trades the stock to offset it.
`HedgingEnv` simulates GBM paths and wraps a fitted `LongstaffSchwartz`
model. The state is spot/strike, time, the LSM delta and gamma, and the
current position, and each episode runs until the path's LSM exercise date.
Trades pay a proportional cost (`--prop_tc`, a fraction of the traded
value) and optionally a fixed cost per trade (`--fixed_tc`). A Soft
Actor-Critic agent chooses the hedge in one of three action modes
(`--action_mode`):
- `residual` (default): a bounded correction to the LSM delta,
  `clip(delta + 0.1·tanh(u))`, so an untrained policy starts at delta hedging;
- `gap_fraction`: the fraction of the gap to delta closed each day;
- `band`: a no-trade band whose half-width and rebalancing target the
  policy chooses at every step.

It is measured against analytic baselines (`--policy`): `delta` (rebalance
to the LSM delta daily), `delta_band` (a fixed no-trade band, rebalancing
to delta or to the band's edge) and `no_hedge`.

**Protocol.** Each policy is evaluated on the same 2,000 simulated episodes
(`--seed 999 --n_episodes 2000`; `test-hedge` reseeds before evaluating, so
every policy sees identical paths). P&L is the hedge's cash flow minus the
option payoff, ×K. It excludes the option premium, so its mean is not a
measure of hedge quality. Std P&L is the hedging error, and 95% CVaR, the
mean of the worst 5% of episodes, is the tail risk (higher is better). RL
rows give the range over 4 training seeds.

> **Results snapshot.** Measured 2026-09-30 to 2026-10-01, not re-verified.
> The agent is under active development, so these numbers will change. The
> full analysis is in [`docs/results/hedging-rl.md`](docs/results/hedging-rl.md).

**At a 0.1% proportional cost** (`train-hedge` defaults, 1,000 episodes per
training run):

```
python main.py test-hedge --policy rl
python main.py test-hedge --policy delta
python main.py test-hedge --policy no_hedge
```

| Policy | Put: Std P&L | Put: 95% CVaR | Call: Std P&L | Call: 95% CVaR |
|---|---|---|---|---|
| **RL (residual)** | **0.0089-0.0096** | **-0.100 to -0.103** | 0.0142-0.0149 | -0.112 to -0.113 |
| Delta hedge (LSM delta) | 0.0119 | -0.110 | **0.0142** | **-0.112** |
| No hedge | 0.0725 | -0.186 | 0.154 | -0.571 |

- **On the American put, RL beats delta hedging** on every seed: 19-25%
  lower hedging error and a better tail. It even beats hedging at the exact
  tree delta (std 0.012-0.013). A hypothesis, not verified: episodes end at
  LSM's slightly suboptimal exercise date, which a learned hedge can adapt
  to and a model delta cannot.
- **On the call, RL matches delta hedging**: training never improved on the
  untrained policy, which already hedges at delta.

**Under higher trading costs**, daily delta hedging overtrades. The
classical answer is a no-trade band, so the fixed band, not daily delta, is
the bar for RL. Policies are compared on CVaR, because std ignores what a
policy spends on trading:

```
python main.py train-hedge --prop_tc 0.01 --fixed_tc 0.05 --action_mode band --val_metric cvar
python main.py test-hedge  --prop_tc 0.01 --fixed_tc 0.05 --policy rl
python main.py test-hedge  --prop_tc 0.01 --fixed_tc 0.05 --policy delta_band --band 0.175 --band_target delta
```

| 95% CVaR (×K) | 1%: Call | 1%: Put | 1% + $0.05: Call | 1% + $0.05: Put |
|---|---|---|---|---|
| Daily delta | -0.198 | -0.208 | -0.380 | -0.390 |
| RL, gap-fraction action | -0.143 to -0.147 | -0.135 to -0.144 | -0.324 to -0.329 | -0.325 to -0.332 |
| Best fixed band (width, target) | -0.136 (0.1, edge) | -0.130 (0.1, edge) | -0.148 (0.175, delta) | -0.146 (0.175, delta) |
| **RL, learned band** | **-0.136 to -0.137** | **-0.123 to -0.128** | **-0.145 to -0.147** | **-0.132 to -0.138** |

- **The learned band matches the best hand-tuned band within seed noise,
  or beats it.** For calls at 1% it is level (-0.136 to -0.137 for the
  learned band, -0.136 for the fixed band); in the other three cells it is
  better on every seed, by up to 0.014 (puts with a fixed cost). No band
  width or target was tuned by hand.
- **The action parametrization was the limit, not the objective.** With the
  gap-fraction action, the agent learned a smoothed delta hedge that trades
  every day. Under a fixed cost each of those small trades pays the full
  fee, which leaves it barely better than daily delta. Giving the policy the
  band's form, with the same SAC objective, closed the gap. Details:
  [`.claude/BUGS.md`, "RL policy under trading costs keeps trading every
  day; a simple band beats it"](.claude/BUGS.md#b-010). The fixed-band
  baselines over a grid of costs are in
  [`docs/results/hedging-costs.md`](docs/results/hedging-costs.md).

## Setup

```
pip install -r requirements.txt
pip install -e .        # optional: makes `optionpricing` importable from anywhere
```

Pinned versions match the development environment (Python 3.13, PyTorch
2.11 CPU build). No GPU is needed: training falls back to CPU when CUDA is
unavailable. Generated datasets, checkpoints, logs and experiment runs go to
`artifacts/` (gitignored).

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
| `train-hedge` | Train the SAC hedging policy (`--action_mode {residual,gap_fraction,band}`, `--prop_tc`, `--fixed_tc`, validation checkpointing via `--val_paths`/`--val_metric`). |
| `test-hedge` | Evaluate a policy (`--policy {rl,delta,delta_band,no_hedge}`): mean P&L, std P&L, 95% VaR and 95% CVaR over simulated episodes. |

The surrogate's dataset has its own entry point, run as a module. It is
seeded and resumable after an interruption:

```
python -m optionpricing.ml.surrogate.tree.generate_tree_dataset   # artifacts/data/tree_put_dataset.npz
```

Every flag of every command, and of the dataset generators, is in
[`docs/architecture.md`, "Command reference"](docs/architecture.md#command-reference).

## Testing

```
python -m pytest tests/unit                        # fast (~30 s)
python -m pytest tests/integration                 # convergence checks and CLI smoke tests (~2.5 min)
python -m pytest tests/                            # everything
python -m tests.mutation.tree_evaluation_mutants   # the evaluation's tests must catch 14 planted bugs
```

870 tests (2026-11-05, `pytest --collect-only -q`). The 771 unit tests
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
  hedging/              hedging env, option values, baselines, P&L evaluation
  ml/surrogate/         MLP surrogate pricer (tree/, legacy/)
  ml/rl/                SAC hedging agent
  ml/experiments/       research drivers (ablation runner)
main.py                 the CLI entry point
scripts/                one module per main.py command
tests/                  unit/, integration/, mutation/
docs/                   architecture.md (module reference), results/
artifacts/              generated datasets, checkpoints, logs, runs (gitignored)
```

Pricers are grouped by option class, everything learned lives in
`optionpricing/ml/`, and the hedging problem the learners train in lives in
`optionpricing/hedging/`. Dependencies point one way: `core` <- `pricing` <-
`hedging` <- `ml`. Every module's API and invariants are in
[`docs/architecture.md`](docs/architecture.md).

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
  [RL hedging](docs/results/hedging-rl.md),
  [hedging baselines under costs](docs/results/hedging-costs.md) and
  [the legacy LSM-trained surrogate](docs/results/legacy-surrogate.md).
- [Design decisions and bug log](.claude/BUGS.md): every problem found, with
  its symptom, root cause and fix.
- [Roadmap](.claude/PLAN.md) and [roadmap archive](.claude/PLAN_ARCHIVE.md):
  what is being worked on, what comes next, and what is done.
