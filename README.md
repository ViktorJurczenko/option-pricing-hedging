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
| **American pricing** | Cox-Ross-Rubinstein binomial tree and Longstaff-Schwartz least-squares Monte-Carlo (LSM), with simple-polynomial or Laguerre-polynomial regression bases, an optional European control variate, and delta/gamma from a regression of pathwise deltas. |
| **ML surrogate pricer** | A small MLP trained to reproduce LSM prices ~instantly. Standalone — not currently wired into the hedging environment below, which prices and computes Greeks via `LongstaffSchwartz` directly. |
| **RL delta-hedging** | A Soft Actor-Critic (SAC) agent trained in a custom Gym-style environment to hedge an American option's exposure over its (path-dependent) exercise horizon, under proportional and fixed trading costs, using `LongstaffSchwartz`'s delta/gamma as state features. It is compared against delta-hedge, no-trade-band and no-hedge baselines. |

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
| Call | 10.30 | 10.45 | 10.45 | 10.28 |
| Put  | 5.63  | 5.57  | 6.09  | 6.13  |

European Monte-Carlo tracks Black-Scholes, and American Longstaff-Schwartz
tracks the binomial tree, as expected (both are validated more rigorously as
convergence tests in `tests/integration/test_pricing_convergence.py`). The put
prices are higher for the American exercise style than the European style, as
expected since early exercise has value for puts. An American call on a
non-dividend stock is worth exactly its European price, which the tree
reproduces.

`price` runs plain LSM, which is a Monte-Carlo estimate (about ±0.1 here).
With the European option as a **control variate** (`LongstaffSchwartz(...,
sigma=..., control_variate=True)`), the LSM call prices at 10.4506, the
Black-Scholes value. The regression then targets the cashflow minus the
realized European payoff, which is exactly 0 on every path that isn't
exercised early, so the regression noise that made plain LSM exercise
calls early (19-35% of paths, where the right answer is 0%) disappears.
`train-hedge`/`test-hedge` enable it for calls only: for puts the plain
exercise rule was worth more out of sample (6.078 vs 6.009, true price
6.084). See `.claude/BUGS.md`, "LSM early-exercises American calls".

Prices are for exactly `T=1`: the GBM simulator used to stop one step short
of maturity (364 of 365 days), which every pricer built on it inherited.
That was fixed, so the simulation now ends exactly at expiry.

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
point for each path. Delta and gamma originally came for free as the
fitted function's analytic derivatives, without a separate
finite-difference pass.

That last part turned out to be far less reliable near maturity than the
price convergence above would suggest. A call's delta must be in `[0, 1]`;
with the default degree-3 polynomial fit, delta landed outside that range
**up to 91.6% of the time** at the very last timestep before expiry
(depending on random seed), with values reaching into the tens — or, at a
higher regression degree, the thousands. This matters in practice:
`HedgeEnv` feeds this delta straight into the hedging policy's state, and
`Hedging/baselines.py`'s `delta_hedge` baseline trades on it
directly and unclamped — a negative "delta" near expiry could execute as an
actual short position, the wrong trade direction. Switching to the Laguerre
basis changes nothing at the same degree (see "Which regression basis?"
below).

**Why:** a call's value-vs-price curve develops an increasingly sharp
corner at `S = K` as maturity approaches, converging on the kinked expiry
payoff. A single smooth global polynomial can't bend that sharply; it
wiggles slightly to compromise instead, and differentiating a wiggly fit
amplifies the wiggle into large Greek swings — worse at higher polynomial
degree, not better (more degrees of freedom to oscillate), which rules out
"just fit a higher-degree polynomial" as a fix.

**First fix, later reverted:** ridge regularization plus per-moneyness-bin
regressions near maturity. It brought the "delta inside `[0, 1]`" rate up,
but checking against the exact Black-Scholes delta showed that was the
wrong yardstick. Inside each narrow bin the slope was unidentifiable and
ridge shrank it to ~0, which is a valid-looking but wrong delta. The small
bins also overfit each path's own future, triggering spurious early
exercise (57% of American-call paths, where the correct answer is 0%).

**Current fix:** delta gets its own regression instead of being the
derivative of the continuation fit. Each path's *pathwise delta* (the
derivative of its discounted cashflow with respect to today's spot,
`±S_exercise/S_t` discounted, if it pays off) is regressed over all paths,
in and out of the money, on the polynomial basis plus the European
Black-Scholes delta. Gamma is the derivative of that fit. Against exact
references (Black-Scholes for the call, a binomial tree for the put), on
4 seeds:

| Delta error (mean absolute) | Derivative of ITM value fit | Pathwise regression |
|---|---|---|
| In the money | 0.03-0.18 | 0.001-0.035 |
| Out of the money | 0.2-0.9 | 0.001-0.017 |
| Largest \|delta\| | 13-48 | 1.8 |

With the fixed delta, hedging at the LSM delta (`python main.py test-hedge
--policy delta`) is as good as hedging at the exact delta: P&L std 0.014 vs
0.015 for the call and 0.012 vs 0.013 for the put (×K), down from 0.060 and
0.044 with the old delta.

Full incident writeup (symptom, root cause, ruled-out fixes) is in
`.claude/BUGS.md`.

### Which regression basis?

`--reg_type laguerre` uses the plain (unweighted) Laguerre polynomials
`L_0..L_d` of `S/K`. Each `L_k` is a degree-`k` polynomial, so they span
exactly the same functions as `1, x, ..., x^d`, and an unregularized
least-squares fit gives the same fitted values in either basis. On 10,000
paths and 2 seeds, at degree 3, the two bases give identical prices,
identical exercise times on every path and deltas equal up to 1e-9:

| Degree | Put price, simple / Laguerre | Same exercise time | Max delta difference |
|---|---|---|---|
| 3 | 6.131 / 6.131, 6.068 / 6.068 | 100% | 1e-9 |
| 6 | 6.147 / 6.150, 6.120 / 6.116 | 99.0-99.6% | 0.24-0.29 |
| 10 | 6.180 / 6.169, 6.076 / 6.106 | 59-66% | 0.33-0.39 |

At higher degrees they part ways only through floating-point conditioning
(`x ≈ 1`, so high powers of `x` are nearly collinear), and neither is
right: higher degrees are worse for both (see the Greeks section above).
The basis is therefore a non-choice at the degree-3 default. `price`,
`train-hedge` and `test-hedge` default to `simple`, and dataset generation
and `test-ml` to `laguerre`; switching any of them changes nothing.

### MLP surrogate vs. Longstaff-Schwartz

```
python -m ML.surrogate.generate_dataset   # regenerate the training set (~40 min at defaults)
python main.py train-ml
python main.py test-ml
```

An earlier version of this surrogate had a real ~9.5% average error on ITM
puts. Chasing that down changed several things about how the dataset is built
and how the model's target is defined — worth explaining since none of it is
obvious from reading `ML/surrogate/model.py` alone:

**1. Training data now covers moneyness continuously.** `ML/surrogate/generate_dataset.py`
used to sample moneyness (`S0/K`) from 5 discrete buckets (`{0.5, 0.7, 1.0, 1.2,
1.5}` ± a little noise), leaving large *untrained* gaps between them (e.g.
nothing at all between 0.75 and 0.95). The old evaluation benchmark happened to
probe exactly one of those gaps, so a chunk of the "ITM error" was really the
model extrapolating into territory it had never seen. `gen_rdm_params` now
samples moneyness log-uniformly and continuously over `(0.4, 1.6)` instead.

**2. The label is a *smoothly* decomposed residual, not raw price.** The MLP
predicts `(price - smooth_intrinsic_value) / K` rather than `price / K`
directly (see `ML/surrogate/model.py::smooth_intrinsic_value`). The
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

**3. More paths per label (`--path_per_sim`, 5,000 → 10,000)** reduces
label noise. The LSM "teacher" also switched to the Laguerre basis
(`--reg_type`, default `laguerre`), but at degree 3 that produces the same
labels as the simple basis (see "Which regression basis?" above), so it
had no effect.

Net effect, benchmarked live against a fresh LSM run (put) across a grid of
`(S0, K, σ, r, T)`, with a fixed seed and 20,000 paths per benchmark point:

| | ITM (S < K) | ATM | OTM (S > K) |
|---|---|---|---|
| Before (hard-kink target, bucketed data) | 9.5% | ~3% | ~1.2% |
| After (smooth target, continuous data) | 0.78% | 3.29% | 6.07% ($0.31 avg abs) |
| **Now** (same model, benchmark on the fixed horizon) | **0.57%** | **3.82%** | **6.32%** ($0.34 avg abs) |

**The dataset predates the horizon fix.** The MLP was trained on labels
from the old simulator, which priced each option one time step short of
its maturity (see the pricing section above). The labels are biased low by
up to 0.0012·K (-3.4% ATM, more in relative terms deep OTM, shrinking with
`T`), while the benchmark LSM now runs on the full horizon. The "Now" row
is this checkpoint measured against the corrected benchmark, so part of
its ATM/OTM error is that label bias. Regenerating the dataset is planned
together with a mixed call/put dataset (`.claude/PLAN.md`). The moneyness
breakdown below was measured against the old benchmark.

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

Each policy's deterministic action, evaluated over the same 2,000 simulated
hedging episodes (`--seed 999 --n_episodes 2000`) with a 0.1% proportional
transaction cost. `test-hedge` reseeds `np.random` from `--seed` right
before evaluation (`scripts/test_hedge.py`), so every policy sees identical
simulated paths — otherwise differences below could just be path-sampling
noise rather than a real difference between policies. RL rows give the
range over four policies trained with the `train-hedge` defaults (seeds
42/1/2/3, 1,000 episodes each). All figures are ×K:

| Policy | Put: Std P&L | Put: 95% CVaR | Call: Std P&L | Call: 95% CVaR |
|---|---|---|---|---|
| **RL (default)** | **0.0089-0.0096** | **-0.100 to -0.103** | 0.0142-0.0149 | -0.112 to -0.113 |
| Delta-hedge (LSM delta) | 0.0119 | -0.110 | **0.0142** | **-0.112** |
| No hedge | 0.0725 | -0.186 | 0.154 | -0.571 |

Mean P&L is about -0.08 for every hedged policy (it excludes the option
premium, so it isn't a hedge-quality metric). P&L is the hedge's cash flow
minus the option payoff at exercise, scaled by the strike; std P&L is the
hedging error, and 95% CVaR (the mean of the worst 5% of episodes) is the
tail risk a hedger actually cares about minimizing.

- **On the American put, the RL policy beats delta hedging** on every seed:
  20-25% lower hedging error and a better tail. It even beats hedging with
  the exact binomial-tree delta (std 0.012-0.013).
- **On the call, it matches delta hedging.** Here training never improved
  on the untrained policy, which already hedges at delta.

Two design choices got it there:
- **Residual action:** the policy learns a bounded correction to the LSM
  delta, `clip(delta + 0.1·tanh(u))`, rather than the whole hedge ratio.
  An untrained policy therefore starts at the delta hedge.
- **Validation checkpointing:** the checkpoint is the policy with the lowest
  P&L std on 1,000 held-out paths, with the untrained policy as a candidate.
  Selecting on the recent training reward instead mostly measured which
  paths were drawn. It left the policies behind delta (call std 0.017-0.026,
  put 0.013-0.020), and a plain non-residual policy did worse still (call
  0.029-0.036).

A likely reason the put policy can beat even the exact delta: episodes end
at the *LSM's* exercise time, so the liability being hedged is the put
under LSM's slightly suboptimal exercise rule, which a learned hedge can
adapt to and a model delta cannot. This is a hypothesis, not yet verified.
Full numbers are in `.claude/BUGS.md`, "RL hedging policy trails the LSM
delta hedge". With higher trading costs, where daily delta hedging
overtrades, the picture changes: see the next section.

### Hedging under trading costs

```
python main.py train-hedge --prop_tc 0.01 --action_mode gap_fraction --val_metric cvar
python main.py test-hedge  --prop_tc 0.01 --policy rl
python main.py test-hedge  --prop_tc 0.01 --policy delta_band --band 0.1 --band_target edge
```

With a 0.1% cost, daily delta hedging is already close to optimal. With
larger costs, rebalancing to delta every day spends most of the P&L on
small trades. The classical fix is a **no-trade band** (`--policy
delta_band`): hold the position until it drifts more than `b` from delta,
then rebalance to delta (best under fixed costs) or only back to the
band's edge (best under proportional costs). Under costs the band, not
daily delta, is the bar for RL. The comparison is on CVaR rather than std,
because std ignores how much a policy spends on trading.

At a 1% proportional cost (95% CVaR ×K, same 2,000 test episodes; RL rows
give the range over 4 training seeds):

| Policy | Call | Put |
|---|---|---|
| Daily delta | -0.198 | -0.208 |
| RL, residual action (`--residual_scale 0.25`) | -0.173 to -0.186 | -0.140 to -0.153 |
| RL, gap-fraction action (`--action_mode gap_fraction`) | -0.142 to -0.147 | -0.135 to -0.144 |
| **Band 0.1, rebalance to edge** | **-0.136** | **-0.130** |

- **RL clearly beats daily delta**, and the `gap_fraction` action helped
  most. There the policy chooses what fraction of the gap to delta to close
  each day, so holding still is easy to represent. With the residual
  action, holding means tracking a moving target exactly, and every miss is
  a paid trade.
- **RL still trails the hand-tuned band** by 0.006-0.009, even with 3,000
  training episodes. It learned to close part of the gap every day (a
  smoothed delta hedge) rather than the band's "sit still, then act", and
  it trades about twice as much.
- **Giving the policy the band's form closes that gap** (next paragraph),
  with the same SAC objective. So the action parametrization was the
  limit, not the objective mismatch suspected at first (SAC trains on a
  per-step mean-variance reward but is judged on terminal CVaR). Details
  are in `.claude/BUGS.md`, "RL policy under trading costs keeps trading
  every day".

**Learned no-trade band** (`--action_mode band`). The policy outputs a 2-D
action: a band half-width `b` up to `--band_max` (default 0.2), and how far
into the band to rebalance once outside it, from the band's edge to delta.
Inside the band it holds its position exactly, so holds pay neither cost. A
constant action is the fixed band above, so RL starts from the baseline's
form and can learn a band that varies with time to expiry, moneyness and
gamma:

```
python main.py train-hedge --prop_tc 0.01 --fixed_tc 0.05 --action_mode band --val_metric cvar
python main.py test-hedge  --prop_tc 0.01 --fixed_tc 0.05 --policy rl
```

Same protocol (95% CVaR ×K, 2,000 test episodes, 4 training seeds of 1,000
episodes), at a 1% proportional cost alone and with a $0.05 fixed cost per
trade on top:

| Policy | 1%: Call | 1%: Put | 1% + $0.05: Call | 1% + $0.05: Put |
|---|---|---|---|---|
| Daily delta | -0.198 | -0.208 | -0.380 | -0.390 |
| RL, gap-fraction action | -0.143 to -0.147 | -0.135 to -0.144 | -0.324 to -0.329 | -0.325 to -0.332 |
| Best fixed band | -0.136 (0.1, edge) | -0.130 (0.1, edge) | -0.148 (0.175, delta) | -0.146 (0.175, delta) |
| **RL, learned band** | **-0.136 to -0.137** | **-0.123 to -0.128** | **-0.145 to -0.147** | **-0.132 to -0.138** |

- **The learned band closes the gap to the hand-tuned band** in every
  cell: it matches it for calls at 1% and beats it elsewhere, by up to
  0.014 (puts with a fixed cost), on every seed. No band width or target
  had to be tuned for it.
- **A fixed cost breaks the gap-fraction action.** It can only hold still
  in the limit, so its "holds" are tiny trades that each pay the fixed
  cost; it ends up barely better than daily delta. The band action holds
  exactly and is unaffected.

### Test suite

```
python -m pytest tests/
```

398 tests pass: 320 unit tests (closed-form formulas, basis-function analytic
derivatives checked against finite differences, dataset splitting, hedging
environment reward arithmetic, the policy's action modes, the baselines, the
SAC training loop) and 78 integration tests (Monte-Carlo/LSM convergence to
their closed-form/tree counterparts, CLI argument wiring, seeded policy
comparisons, plus CLI smoke tests for every subcommand).

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
| `test-ml` | Evaluate the tree-trained put surrogate (`PutSurrogate`): test-split errors, plus a factorial grid against an independent finer binomial tree (per-bucket $, relative, implied-vol, delta/gamma errors, no-arbitrage violation rates, theta check, routing, timing). |
| `train-hedge` | Train the SAC hedging policy (`--action_mode {residual,gap_fraction,band}`, `--prop_tc`, `--fixed_tc`, validation checkpointing via `--val_paths`/`--val_metric`). |
| `test-hedge` | Evaluate a policy (`--policy {rl,delta,delta_band,no_hedge}`): mean P&L, std P&L, 95% VaR and 95% CVaR over simulated episodes. |

Dataset generation for the MLP is a separate entry point and must be run as a
module (absolute imports require the project root on `sys.path`):

```
python -m ML.surrogate.generate_dataset [--path_per_sim N] [--nb_param_comb N] [--samples_per_param N] \
    [--nb_jobs N] [--sampling N] [--option_type call|put] [--reg_type simple|laguerre] [--max_degree N] \
    [--output_dir data] [--seed N] [--chunk_size N]
```

This parallelizes LSM pricing across randomly sampled `(S0, K, σ, r, T)`
combinations to build `data/american_option_dataset.npz` and the
feature-normalization stats in `data/features_norm.npz`.

The run is resumable and reproducible. The parameter draw and every sample's
paths are seeded (`--seed`, default 42), and every finished chunk
of `--chunk_size` samples is saved to `data/chunks/`, so after a Ctrl-C or a
kill you re-run the same command and it continues where it stopped (at most
one chunk is lost). Resuming with different settings is refused.

The MLP overhaul's new dataset (in progress, see `.claude/PLAN.md`) is
labelled by a vectorized binomial tree instead of LSM, puts only:

```
python -m ML.surrogate.GenerateTreeDataset [--n_rows N] [--chunk_size N] [--nb_steps N]     [--method crr|bbs|bbsr] [--nb_jobs N] [--seed N] [--output_dir data]
```

Parameters come from a scrambled Sobol sequence (`--n_rows` and
`--chunk_size` are powers of 2), with moneyness sampled through
`z = ln(m)/(σ√T)`, so every maturity and volatility gets the same
coverage around the strike. It writes raw columns (parameters, American and
European prices, delta, gamma, an `exercised` flag) and its settings to
`data/tree_put_dataset.npz`, and resumes like the LSM generator.

## Testing

```
python -m pytest tests/                 # everything
python -m pytest tests/unit             # fast, no training/simulation-heavy work
python -m pytest tests/integration      # convergence checks + CLI smoke tests; slower
```

## Architecture

Pricers are grouped by option class, everything that is learned lives in
`ML/`, and the hedging problem the learners are trained in lives in
`Hedging/`:

- **`core/`** — the foundation every pricer depends on: `gbm.py` (GBM path
  simulator) and `base.py` (the `OptionPricing` ABC).
- **`EuropeanOptionPricing/`** — `BlackScholes` (closed form) and `MonteCarlo`
  (path-average payoff).
- **`AmericanOptionPricing/`** — `BinomialTree` (CRR tree with backward
  induction), `LongstaffSchwartz` (LSM with simple/Laguerre polynomial
  regression bases, an optional European control variate, and delta/gamma
  from a regression of pathwise deltas — see Results above) and
  `basis_functions.py` (the regression bases, unit-tested independently).
- **`ML/`** — everything that is learned:
  - `surrogate/`: the MLP surrogate pricer. `model.py` (`LegacyMLP`, the
    LSM-trained net `train-ml`/`test-ml` still use, 5 inputs → 1 output:
    `(price - smooth_intrinsic_value) / K` — a smoothly decomposed residual,
    not raw price; see Results above for why; and `MLP`, its configurable,
    smooth (SiLU) successor for the tree dataset, which predicts the
    early-exercise premium over Black-Scholes), `pricer.py` (`PutSurrogate`:
    Black-Scholes + the net's premium, delta and gamma by autograd, queries
    outside the training domain routed to Black-Scholes, `1 - m` or the tree,
    saved and loaded as one checkpoint),
    `training.py` (its training loop: premium + delta loss, optional
    no-arbitrage shape penalties, cosine-decayed Adam, early stopping,
    optional L-BFGS finish; not yet behind `train-ml`),
    `dataset.py` (`AmericanOptionDataset`, seeded, disjoint train/test split),
    `generate_dataset.py` (parallel, resumable LSM dataset generation),
    `GenerateTreeDataset.py` (its tree-labelled successor, puts only) and
    `evaluation.py` (regression/benchmark helpers used by `test-ml`).
  - `rl/`: the SAC hedging agent. `policy.py` (the Gaussian actor; its
    tanh-squashed output becomes a hedge in one of three action modes: a
    correction to the LSM delta (`residual`, the default), the fraction of
    the gap to delta closed each step (`gap_fraction`), or a 2-D no-trade
    band (`band`)), `critic.py` (twin Q-networks), `replay_buffer.py`, and
    `sac.py` (the training loop: target networks, warmup, learned entropy
    coefficient, mean-variance reward shaping, checkpointing on held-out
    validation paths).
- **`Hedging/`** — the hedging problem, independent of any learner:
  - `env.py`: `HedgingEnv`, wraps a fitted `LongstaffSchwartz` instance.
    State = spot/strike, time, delta, gamma, position; episodes run to each
    path's exercise time under LSM's exercise rule. Proportional and fixed
    trading costs; a `terminal` reward (hedge P&L minus the payoff) or an
    `accounting` one (minus the option's value change every step, used for
    training); a batched `rollout_pnl` for fast validation.
  - `option_value.py`: the option values the accounting reward needs
    (Black-Scholes for calls, a binomial tree for puts).
  - `baselines.py`: delta hedge, no-trade band and no-hedge policies.
  - `evaluation.py`: `evaluate()` (mean, std, VaR and CVaR of the P&L of
    any policy) and `cvar()`.
- **`scripts/`** — one module per CLI subcommand, wired into `main.py`'s
  `argparse` subparsers.
- **`data/`, `checkpoints/`** — generated artifacts only (gitignored): the
  dataset/normalization `.npz` files and trained model `.pth` weights.

See `.claude/CLAUDE.md` for the full architectural notes (invariants like
"call `predict_payoff()` before `calculate_option_price()`", normalization
consistency requirements, etc.) used when developing this project with Claude
Code.
