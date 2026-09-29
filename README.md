# Option Pricing & Hedging

A from-scratch research project in Python/PyTorch that builds up option pricing
and delta-hedging from first principles: closed-form and Monte-Carlo pricing
for European options, tree- and simulation-based pricing for American options,
a neural surrogate that approximates the American pricer at a fraction of the
cost, and a reinforcement-learning agent that learns to delta-hedge an
American option directly from simulated price paths.

Everything — GBM path simulation, Black-Scholes, Longstaff-Schwartz, the MLP,
the SAC hedging agent — is implemented from scratch on top of NumPy/SciPy and
PyTorch, rather than pulled from an existing quant or RL library.

## What's inside

| Layer | What it does |
|---|---|
| **European pricing** | Closed-form Black-Scholes and Monte-Carlo simulation, cross-checked against each other. |
| **American pricing** | Cox-Ross-Rubinstein binomial tree and Longstaff-Schwartz least-squares Monte-Carlo (LSM), with simple-polynomial or Laguerre-polynomial regression bases and analytic delta/gamma. |
| **ML surrogate pricer** | A small MLP trained to reproduce LSM prices ~instantly. Standalone — not currently wired into the hedging environment below, which prices and computes Greeks via `LongstaffSchwartz` directly. |
| **RL delta-hedging** | A Soft Actor-Critic (SAC) agent trained in a custom Gym-style environment to hedge an American option's exposure over its (path-dependent) exercise horizon, under transaction costs, using `LongstaffSchwartz`'s analytic delta/gamma as state features. |

## Results

Numbers below come from running the included commands with their default
parameters (`S0=K=100`, `r=5%`, `T=1y`, `σ=20%`) on this machine; regenerate
them yourself with the commands shown.

### Pricing models agree with each other

```
python main.py price --option_class both                     # call
python main.py price --option_class both --option_type put   # put
```

| | Monte-Carlo | Black-Scholes | Binomial tree | Longstaff-Schwartz |
|---|---|---|---|---|
| Call | 10.28 | 10.45 | 10.45 | 10.31 |
| Put  | 5.63  | 5.57  | 6.09  | 6.12  |

European Monte-Carlo tracks Black-Scholes, and American Longstaff-Schwartz
tracks the binomial tree, as expected (both are validated more rigorously as
convergence tests in `tests/integration/test_pricing_convergence.py`). The put
prices are higher for the American exercise style than the European style, as
expected since early exercise has value for puts.

### LSM delta/gamma accuracy near maturity

**Longstaff-Schwartz, briefly:** pricing an American option means knowing,
at every point along a simulated path, whether exercising now beats holding
on — which needs an estimate of what holding on is worth (the *continuation
value*), not directly observable from a single path since it depends on the
(unsimulated) future. LSM's answer: walk each simulated path backward from
expiry, and at each timestep, regress the already-known future discounted
payoff on the current stock price, across all in-the-money paths at that
timestep, to fit an estimate of continuation value — then compare that
estimate to the immediate exercise payoff to decide the optimal exercise
point for each path. The regression's coefficients also give delta and
gamma for free, as the fitted function's analytic derivatives, without a
separate finite-difference pass.

```
python main.py test-hedge --policy delta
```

That last part turned out to be far less reliable near maturity than the
price convergence above would suggest. A call's delta must be in `[0, 1]`;
with the default degree-3 polynomial fit, delta lands outside that range
**up to 91.6% of the time** at the very last timestep before expiry
(depending on random seed), with values reaching into the tens — or, at a
higher regression degree, the thousands. This matters in practice:
`HedgeEnv` feeds this delta straight into the hedging policy's state, and
`AmericanHedging/baselines.py`'s `delta_hedge` baseline trades on it
directly and unclamped — a negative "delta" near expiry could execute as an
actual short position, the wrong trade direction.

**Why:** a call's value-vs-price curve develops an increasingly sharp
corner at `S = K` as maturity approaches, converging on the kinked expiry
payoff. A single smooth global polynomial can't bend that sharply; it
wiggles slightly to compromise instead, and differentiating a wiggly fit
amplifies the wiggle into large Greek swings — worse at higher polynomial
degree, not better (more degrees of freedom to oscillate), which rules out
"just fit a higher-degree polynomial" as a fix.

**Fix:** two opt-in `LongstaffSchwartz` parameters, both off by default (so
every other consumer — `price`, the ML surrogate's dataset generation — is
unaffected unless it explicitly opts in):
- `ridge_lambda` — L2-regularizes the per-timestep regression, directly
  discouraging the exploding coefficients that cause the overshoot.
- `moneyness_bins` / `moneyness_bins_window` — splits in-the-money paths
  into quantile bins by spot price and fits one regression per bin, but
  *only* for timesteps within `moneyness_bins_window` of maturity. Applying
  this to every timestep was tried first and made the already well-behaved
  middle of the timeline worse — a fit on ~250 points per bin is less
  stable than one on ~5,000, and most timesteps aren't kinked sharply
  enough to need localizing.

`train-hedge`/`test-hedge` now default to `--ridge_lambda 0.06
--moneyness_bins 30 --moneyness_bins_window 0.3` — found by sweeping both
parameters together and picking the combination that maximized
near-maturity Greeks accuracy subject to keeping the option price within
the unregularized baseline's own natural error vs. a Binomial Tree
benchmark. Validated across 3 random seeds:

| | Baseline | Fixed |
|---|---|---|
| Invalid delta, last 20 timesteps (avg) | 9.6-14.5% | 1.8-9.5% |
| Invalid delta, single worst timestep | 15.3-91.6% | 0.9-1.2% |
| Price drift vs. Binomial Tree | -1.33% to +1.31% | within the same noise band |

Full incident writeup (symptom, root cause, ruled-out fixes) is in
`.claude/BUGS.md`.

### MLP surrogate vs. Longstaff-Schwartz

```
python -m datasets.generate_dataset   # regenerate the training set (~40 min at defaults)
python main.py train-ml
python main.py test-ml
```

An earlier version of this surrogate had a real ~9.5% average error on ITM
puts. Chasing that down changed several things about how the dataset is built
and how the model's target is defined — worth explaining since none of it is
obvious from reading `ML.py` alone:

**1. Training data now covers moneyness continuously.** `datasets/generate_dataset.py`
used to sample moneyness (`S0/K`) from 5 discrete buckets (`{0.5, 0.7, 1.0, 1.2,
1.5}` ± a little noise), leaving large *untrained* gaps between them (e.g.
nothing at all between 0.75 and 0.95). The old evaluation benchmark happened to
probe exactly one of those gaps, so a chunk of the "ITM error" was really the
model extrapolating into territory it had never seen. `gen_rdm_params` now
samples moneyness log-uniformly and continuously over `(0.4, 1.6)` instead.

**2. The label is a *smoothly* decomposed residual, not raw price.** The MLP
predicts `(price - smooth_intrinsic_value) / K` rather than `price / K`
directly (see `AmericanOptionPricing/ML.py::smooth_intrinsic_value`). The
naive version of this idea — subtracting the *exact* intrinsic value
`max(K-S, 0)` — helps in general (it shrinks the target's dynamic range: deep
ITM prices can be many times larger than deep OTM ones), but that exact
formula has a hard kink at `S == K`, while the option's actual *price* before
expiry has no such kink there (only the expiry *payoff* does). Subtracting a
kinked function from a smooth one reintroduces the kink into the residual —
which showed up empirically as the model's error spiking specifically at the
money once the coverage gap above was fixed. Using a smooth (softplus-based)
approximation of intrinsic value instead removes that artifact while keeping
the dynamic-range benefit. `beta_frac=0.06` (the width of the smoothed corner,
as a fraction of `K`) was picked by sweeping several values against a fixed
LSM benchmark — see the function's docstring for the sweep.

**3. More paths per label (`--path_per_sim`, 5,000 → 10,000)** and **4. the
Laguerre basis for the LSM "teacher" (`--reg_type`, default now `laguerre`
instead of `simple`)** reduce label noise and use a basis better suited to the
early-exercise boundary — both just make the ground truth the MLP is trained
against a bit more trustworthy.

Net effect, benchmarked live against a fresh LSM run (put) across a grid of
`(S0, K, σ, r, T)`, with a fixed seed and 20,000 paths per benchmark point:

| | ITM (S < K) | ATM | OTM (S > K) |
|---|---|---|---|
| Before (hard-kink target, bucketed data) | 9.5% | ~3% | ~1.2% |
| **Now** | **0.78%** | **3.29%** | **6.07%** ($0.31 avg abs) |

ITM improved by an order of magnitude (mostly from fixing the data gap), ATM
stayed about the same, and OTM got a bit worse — genuinely worse, not just a
percentage-of-a-small-number artifact. (An earlier version of this section
claimed OTM put prices are small to begin with, so the $0.31 miss was mostly
harmless; that's wrong here, because this benchmark sweeps moneyness together
with σ and T rather than independently, so by the time moneyness reaches 1.5
both σ and T are also at their highest swept values — enough for a "deep OTM"
put to still be worth $3-8, not a few cents.) Breaking the OTM bucket down by
moneyness shows a U-shape with two distinct, explainable causes rather than
one uniform problem: error is worst (6-14%) for moneyness 1.05-1.20, just past
the money, inside the same `±3×beta_frac ≈ ±0.18` window that the ATM
smoothing correction (above) deliberately widens to fix the ATM spike — a
known, accepted cost of that fix. It's worst again (7-10%) for moneyness
1.40-1.50, close to the edge of the trained `moneyness_range=(0.4, 1.6)`,
which is a milder version of the same coverage-thinning issue fixed for ITM.
In between (moneyness 1.25-1.38, comfortably clear of both effects) the
surrogate is accurate to <2%, confirming the surrogate itself isn't
struggling with "OTM" as a concept — it's these two specific, bounded effects
stacking at the edges of the bucket. On the held-out split: MSE 0.000017, MAE
0.0028, R² 0.992 (in `(price - smooth_intrinsic)/K` units now, not raw
`price/K`, so not directly comparable to older numbers from before this
change).

### RL hedging policy vs. analytic baselines

```
python main.py test-hedge --policy rl
python main.py test-hedge --policy delta
python main.py test-hedge --policy no_hedge
```

Each policy's deterministic action, evaluated over 1,000 simulated hedging
episodes with a 0.1% proportional transaction cost. All three runs use the
same `--seed`, which `test-hedge` reseeds `np.random` from right before
evaluation (`scripts/test_hedge.py`) so all three see identical simulated
paths — otherwise differences below could just be path-sampling noise rather
than a real difference between policies:

| Policy | Mean P&L | Std P&L | 95% CVaR |
|---|---|---|---|
| RL (SAC) | -0.11 (×K) | 0.15 (×K) | -0.41 (×K) |
| Delta-hedge (analytic) | -0.10 (×K) | 0.05 (×K) | -0.18 (×K) |
| No hedge | -0.11 (×K) | 0.15 (×K) | -0.40 (×K) |

P&L is the hedge's cash flow minus the option payoff at exercise, scaled by
the strike; std P&L is the hedging error, and 95% CVaR is the tail risk a
hedger actually cares about minimizing.

The honest reading of this: at these settings, the trained SAC policy does
**not** beat the trivial baseline of rebalancing to the Longstaff-Schwartz
model's own analytic delta every step (`AmericanHedging/baselines.py`) —
delta-hedging has roughly 3x lower variance and well over 2x better tail
risk. No-hedge is a sanity floor (not hedging at all should be worse than
hedging), and the RL policy currently lands right on top of it rather than
above it — essentially indistinguishable on all three metrics. That's a real
result, not a bug: training is numerically stable (a real actor-divergence
bug was found and fixed here — see `.claude/BUGS.md`) and the state's
delta/gamma features are now reliable near maturity (see the LSM fix
above) — retraining after fixing both confirmed the gap isn't explained by
either of those, so it's a genuine open question in the RL setup itself
(reward shaping, exploration/entropy balance, network capacity, training
budget), tracked in `.claude/PLAN.md`'s "Close the RL hedging policy's gap
to the delta baseline" entry.

### Test suite

```
python -m pytest tests/
```

49 tests pass: 42 unit tests (closed-form formulas, basis-function analytic
derivatives checked against finite differences, dataset splitting, hedging
environment reward arithmetic) and 7 integration tests (Monte-Carlo/LSM
convergence to their closed-form/tree counterparts, plus CLI smoke tests for
every subcommand).

## Setup

```
pip install -r requirements.txt
```

Pinned versions match the environment this was developed against (Python
3.13, PyTorch 2.11 CPU build). No GPU is required — training loops check
`torch.cuda.is_available()` and fall back to CPU automatically.

## Usage

Everything runs through `main.py` from the project root (imports are absolute,
e.g. `from core.gbm import GBM`):

```
python main.py --help                  # list all commands
python main.py <command> --help        # list a command's flags
```

| Command | Purpose |
|---|---|
| `price` | Compare pricing models (`--option_class {EU,US,both}`, `--option_type {call,put}`, plus strike/vol/rate/maturity/sampling flags). |
| `train-ml` | Train the MLP surrogate pricer against a generated dataset. |
| `test-ml` | Evaluate the trained MLP: regression metrics + ITM/ATM/OTM error vs. a fresh LSM benchmark. |
| `train-hedge` | Train the SAC delta-hedging policy. |
| `test-hedge` | Evaluate a trained policy: mean P&L, std P&L, 95% CVaR over simulated episodes. |

Dataset generation for the MLP is a separate entry point and must be run as a
module (absolute imports require the project root on `sys.path`):

```
python -m datasets.generate_dataset [--path_per_sim N] [--nb_param_comb N] [--samples_per_param N] \
    [--nb_jobs N] [--sampling N] [--option_type call|put] [--reg_type simple|laguerre] [--max_degree N] \
    [--output_dir data]
```

This parallelizes LSM pricing across randomly sampled `(S0, K, σ, r, T)`
combinations to build `data/american_option_dataset.npz` and the
feature-normalization stats in `data/features_norm.npz`.

## Testing

```
python -m pytest tests/                 # everything
python -m pytest tests/unit             # fast, no training/simulation-heavy work
python -m pytest tests/integration      # convergence checks + CLI smoke tests; slower
```

## Architecture

Module layout mirrors option class for pricing, and role for everything else:

- **`core/`** — the foundation every pricer depends on: `gbm.py` (GBM path
  simulator) and `base.py` (the `OptionPricing` ABC).
- **`EuropeanOptionPricing/`** — `BlackScholes` (closed form) and `MonteCarlo`
  (path-average payoff).
- **`AmericanOptionPricing/`** — `BinomialTree` (CRR tree with backward
  induction), `LongstaffSchwartz` (LSM with simple/Laguerre polynomial
  regression bases and analytic delta/gamma; optional `ridge_lambda` and
  near-maturity `moneyness_bins`/`moneyness_bins_window` to fix Greeks
  overshoot close to expiry — see Results above), `basis_functions.py` (the
  regression bases, unit-tested independently), `ML.py` (the `MLP` surrogate,
  4 inputs → 1 output: `(price - smooth_intrinsic_value) / K` — a smoothly
  decomposed residual, not raw price; see Results above for why), `evaluation.py`
  (regression/benchmark helpers used by `test-ml`).
- **`AmericanHedging/`** — a SAC-style continuous-control setup: `HedgeEnv.py`
  (wraps a fitted `LongstaffSchwartz` instance; state = spot/strike, time,
  delta, gamma, position; episodes run to each path's optimal exercise time),
  `RLPolicy.py` (Gaussian actor), `Critic.py` (twin Q-networks), `Hedge.py`
  (SAC training loop with target networks, warmup, and CVaR-based evaluation).
- **`datasets/`** — `AmericanOptionDataset` (seeded, disjoint train/test split)
  and `generate_dataset.py` (parallel LSM dataset generation).
- **`scripts/`** — one module per CLI subcommand, wired into `main.py`'s
  `argparse` subparsers.
- **`data/`, `checkpoints/`** — generated artifacts only (gitignored): the
  dataset/normalization `.npz` files and trained model `.pth` weights.

See `.claude/CLAUDE.md` for the full architectural notes (invariants like
"call `predict_payoff()` before `calculate_option_price()`", normalization
consistency requirements, etc.) used when developing this project with Claude
Code.
