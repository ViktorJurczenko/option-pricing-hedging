# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Rewrite REMINDER.md into up-to-date chapters

REMINDER.md (gitignored notes) had grown as a log: commit-dated headers,
"as built (step N)" sections, a summary part duplicating the training
and loss sections, statements made stale by step 9 ("nothing trained
yet", patience 20), and no explanation of the foundations it builds on
(GBM, Black-Scholes, LSM, the hedging env, SAC). Decided with the user:
restructure into numbered chapters that each explain the current code,
add the missing foundations, keep the measured evidence behind each
choice but drop process notes (they are in BUGS.md), move the legacy
"is the MLP useful?" analysis to an appendix, and leave an ablation
chapter to fill. Written with the `optionpricing-docs` skill; a backup
of the old file is kept outside the repo.

- [x] Part I, foundations
- [x] Part II, the surrogate
- [x] Part III, hedging
- [x] Legacy appendix, then check
- [ ] Fill chapter 11 once the ablation runs are done
  Copy the per-variant numbers from `runs/ablations/summary.md` into
  REMINDER.md's chapter 11 table, write the findings, and update the
  final-model paragraph of chapter 10.

## Restructure: ML/ for everything learned, Hedging/ for the hedging problem

ML code was spread over `AmericanOptionPricing/` (`ML.py`, `evaluation.py`),
a top-level `datasets/` (which shadows the HuggingFace `datasets` package
name) and `AmericanHedging/`, where the SAC agent sat next to the
environment it is trained in. Chosen layout (option B of three): `ML/`
holds what is learned, `Hedging/` the finance problem (env, value functions,
baselines, P&L metrics), shared by any learner (SAC now, Deep Hedging later).
Pricing folders unchanged. Done before the MLP overhaul below, so that one
starts in its final place.

- [x] Move files with `git mv` (history kept)
- [x] Split `Hedge.py`
- [x] Rewrite imports and string references
- [x] Docs
- [x] Full test suite (427 unit as before the move, 91 integration)

## MLP surrogate overhaul: binomial-tree teacher, premium target, smooth net

An analysis of the regenerated dataset and of the course (ch. 23) found:
call labels are exactly Black-Scholes (the net re-learns a closed form);
the fixed-width softplus (`beta = 0.06 K`) is wider than the price's own
curvature for the ~33% of rows with `sigma*sqrt(T) < 0.12`, so the kink is
back there and 43% of labels are negative; a ReLU net is piecewise linear
(staircase delta, no gamma) and `layer3`/`layer4` collapse into one linear
map; BatchNorm/Dropout add noise to a noise-free fit; LSM labels are biased
upward and 1.9% of put labels sit below the European price; `test-ml` mixes
% and $ in a bucket over a confounded 50-point grid. For a one-asset GBM put,
a binomial tree is a faster and nearly exact teacher, so LSM is dropped as
the MLP's teacher (it stays in the hedging env). All code is written and
tested before any full generation run. Background and the reasons for each
design choice: `REMINDER.md`, "The MLP surrogate: tree teacher, and why
every part is built the way it is".

- [x] Vectorized smoothed binomial pricer
- [x] Tests and convergence study for the pricer
- [x] New generator `ML/surrogate/tree/generate_tree_dataset.py`
- [x] Dataset class: features, target and stats at load time
- [x] Smooth configurable MLP and the `PutSurrogate` price helper
- [x] Training loop overhaul (`ML/surrogate/tree/tree_training.py`)
- [x] Shape penalty study: `w`-scaled violations, payoff floor in `PutSurrogate`
- [x] Evaluation overhaul in `test-ml`
- [x] Tests, CLI wiring, docs
- [x] Smoke run, then full generation
  Full dataset: 2^19 rows (`data/tree_put_dataset.npz`, a copy of the
  learning curve's 2^19 run). Measured results, kept for the ablations:
  Curve (2026-10-27, default TrainConfig, seed 42, 200 epochs each;
  test-ml grid vs BBSR-4000, $ at K = 100):
  rows     |z|<1 MAE  all MAE  IV err bp  exercised max  train time
  2^15     0.0331     0.0236   33.5       0.894          ~0.6 min
  2^16     0.0174     0.0125   19.4       0.634          1.3 min
  2^17     0.0094     0.0053    7.7       0.099          2.5 min
  2^18     0.0037     0.0028    5.0       0.027          5.0 min
  2^19     0.0022     0.0014    2.5       0.022          9.9 min
  Still ~halving per doubling. Confounded: epochs are fixed, so a bigger
  set also gets proportionally more Adam steps, and every run's best
  epoch is 192-200 (still improving). Then data and steps separated:
  Data vs steps (400 epochs, --patience 400: the default patience cut
  400-epoch runs short, BUGS.md):
  run                 |z|<1 MAE  all MAE  IV err bp  valid loss  time
  2^19, 200 epochs    0.0022     0.0014   2.46       6.90e-5     9.9 min
  2^18, 400 (= steps) 0.0018     0.0014   2.30       6.65e-5     9.9 min
  2^19, 400 epochs    0.0012     0.0011   1.75       3.04e-5    19.6 min
  Same steps, half the data: as good. Same data, twice the steps: ~45%
  less error near the money, best epoch 394/400 (still improving). The
  curve was driven by training steps: 2^19 rows is enough data, the
  limit is the optimization. Violations > 1e-5 (stricter than the 1e-4
  default), grid continuation: 2.50% -> 1.53% (mostly increasing_in_T);
  exercised ~71-76% at 1e-5 is wobble around exactly-zero derivatives,
  answered by the payoff floor.
- [x] Training length: 400 epochs (`TrainConfig.epochs` default)
  2^19 rows, seed 42, grid vs BBSR-4000, $ at K = 100; valid terms are
  the loss's two halves (premium term + delta term), reported apart:
  epochs  |z|<1 MAE  all MAE  test MAE  valid premium RMSE  delta RMSE
  200     0.0022     0.0014   0.0013    2.62e-5             8.10e-4
  400     0.0012     0.0011   0.0009    1.71e-5             5.38e-4
  800     0.0019     0.0017   0.0015    2.35e-5             3.86e-4
  800 (early-stopped at 665, best 645, ~33 min) halved the combined
  loss but priced worse: past 400 the delta error kept falling (-28%)
  while the premium error rose (+38%), and the sum picks the checkpoint.
  800 also had fewer violations (0.06% > 1e-4 vs 0.61%) and more payoff
  floors (490 grid points vs 179). Single seed each. The loss still
  picks the checkpoint, but its two halves are now reported apart
  (history, progress bar, `train-ml` summary).
- [x] Ablation runner (`experiments/run_ablations.py`, reference-grid cache, several tolerances)
- [ ] CLI-only ablations, 3 seeds each (42, 43, 44)
  Every variant is the default config with one change, so it needs only
  `train-ml` flags:
  1. Baseline (defaults): the seed-to-seed spread every other row is
     compared against.
  2. `--lambda_delta` 0, 0.3, 3 (1 = baseline): the price/delta trade-off
     the 800-epoch run showed, scored on price and delta separately.
  3. `--output softplus`: the labels' slightly negative premiums
     (-2.7e-6) may cost it.
  4. `--activation relu`: the old net's activation on the new layout
     (the full `LegacyMLP` cannot train on the tree dataset: 5 inputs,
     time-value target), so the staircase delta/zero gamma cost.
  5. `--lbfgs_steps 200`: whether the full-batch finish pays its cost.
  6. `--lambda_shape 0.1`: one value only, ~3.4x the step time (~95 min
     a run); on the trial it hurt every region, but the trial was 4k rows.
  18 runs + 3 slow ones: ~3 h of training in batches of 4, plus scoring.
- [ ] Dataset ablations (need code first)
  `T` from 1 week vs from 0.25 (rows scaled to the same density), both
  scored on the `T >= 0.25` grid, to confirm the longer range costs the
  old region nothing: needs a `--T_min` (domain) flag on the generator
  and a grid restricted to `T >= 0.25`. All exercised rows vs a thinned
  share (e.g. keep 1/4): needs a training-time row filter. Plan each
  when the CLI-only results are in.
- [ ] Final model and write-up
  Retrain the winning config (3 seeds, keep the median one) as
  `checkpoints/put_surrogate.pth`. Record the numbers here and in
  `REMINDER.md`; README results section for the tree surrogate.

Step 2 convergence study (120 seeded random draws over the domain plus a
21-point hard corner, `m` 0.95-1.05, `T = 0.25`, `sigma = 0.05`; reference
BBSR at 20000 steps, which agrees with CRR averaged over N, N + 1 then
Richardson to 3.3e-6 at worst). Errors per unit of strike, max | mean:

| N = 1000 | price | delta | gamma |
|---|---|---|---|
| CRR | 5.5e-5 \| 5.8e-6 | 2.2e-4 \| 1.7e-5 | 0.46 \| 4.1e-3 |
| BBS | 2.1e-5 \| 3.6e-6 | 2.9e-4 \| 1.5e-5 | 0.46 \| 4.1e-3 |
| BBSR | 3.2e-5 \| 9.6e-7 | 8.3e-4 \| 1.4e-5 | 0.33 \| 3.2e-3 |

Choice: **BBSR, N = 1000** (the defaults). Its mean price error is ~4x
below BBS/6x below CRR; the max (~3e-5 K = $0.003 at K = 100) is far below
what an MLP reaches (~1e-4 K). CRR is out: its odd/even swing between
N = 1000 and 1001 is 8.7e-5. BBSR moves 1.5e-6 between N = 1000 and 1002
(BBS 1.1e-7: Richardson amplifies the exercise boundary's jitter), still
noise far below the MLP's error. N = 2000 halves the max error at 3.5x the
cost, not worth it. Delta and gamma are both usable: the large gamma errors
in the table (max 0.33) were puts exercised today, which now get their exact
Greeks (BUGS.md, "Tree Greeks wrong just inside the exercise region"); with
that rule, on 63 options vs a 16000-step reference: gamma mean error
1.4e-3, median relative 3e-5 in the continuation region, delta mean 8.5e-6.
Only a ~1e-3-wide band in `m` at the exercise boundary stays uncertain,
where the true gamma jumps from 0. In the corner every method is accurate
in price (BBSR 2.1e-6 max); gamma (~16) was off by up to 3.5 there before
the rule, 8.4e-4 after on a corner sweep. Timing,
2048 random options, chunks of 32, idle machine: BBSR 0.80 / 2.71 / 9.57
ms per option at N = 500 / 1000 / 2000 (CRR 1.84 at 1000), so 200k puts
cost ~9 min on one core. The teacher does not limit the dataset size.
Labels can be slightly *below* Black-Scholes (min premium -2.7e-6 at
N = 1000, OTM short T): matters for the softplus-output ablation. The error
table predates the exercise-today rule, which only changes exercised puts.
Sampling `m` log-uniform on (0.4, 1.6) makes 66% of the draws in the money
(the range is asymmetric in log: ln 0.4 = -0.92, ln 1.6 = +0.47), and 67%
of those are already past the exercise boundary: 44% of all rows would be
exercised today, labels known in closed form. Decided: sample
`z = ln(m) / (sigma sqrt(T))` instead (generator step above).

## Faster dataset generation

A profile of one sample showed `LongstaffSchwartz.fit` at ~90% of the time,
mostly work the dataset never uses (Greeks), SVD-based `lstsq` on 10k x 4
systems and scipy's generic `norm.cdf`/`pdf`. Labels must stay the same.

- [x] `compute_greeks` flag on `LongstaffSchwartz`
- [x] Normal-equations least squares (price-only mode; the hedging path keeps `lstsq`, the normal equations cost ~1e-5 in the Laguerre deltas)
- [x] `ndtr` in `BlackScholes`
- [x] Tests, restart the run with `--nb_jobs 10`

## Resumable dataset generation

Two generation runs were lost to a kill (see BUGS.md, "Dataset generation
loses all work when interrupted"), so an interrupted run must resume.

- [x] Seed the parameter draw and each sample's paths (`--seed`)
- [x] Save finished chunks and skip them on a re-run (`--chunk_size`)
- [x] Tests, docs

## Modularize the test helpers

The test-suite cleanup left duplicated helpers across files. Pull them into
shared modules so a layout or CLI change is edited in one place.

- [x] Shared fake LSM factory
- [x] Shared state builder
- [x] Build CLI args from the real parsers
- [x] One subprocess helper
- [x] helpers.py (no conftest.py: no fixture is shared yet)
- [x] Split test_hedge.py by concern

## Test suite cleanup

A review of the whole suite (419 passed, 2 skipped) found it healthy
overall, with strong hand-computed and reference checks in the hedging stack
and the LSM Greeks. But it is thin where silent failures cost most, has
about 15 duplicate tests, and some tests would pass with broken code. Steps
in the reviewer's priority order. No order is set yet relative to the ML
dataset regeneration.

- [x] Fix HedgingEnv for options exercised at row 0
- [x] Test the SAC update math in `train()`
- [x] Fix weak and loose tests
- [x] Add missing coverage
- [x] Remove duplicate tests

## Deep Hedging (Buehler et al. 2019) as an alternative to SAC

Hedge P&L is linear in the positions, so a policy network can be trained by
direct backprop through whole simulated episodes, minimizing a risk measure
(CVaR, entropic, variance) of the *terminal* P&L over batches of paths:
exactly what `test-hedge` reports, with no critics, entropy term, Bellman
target or per-step reward shaping. SAC under trading costs is trained on a
per-step mean-variance reward but judged on terminal CVaR (BUGS.md, "RL
policy under trading costs keeps trading every day"); Deep Hedging
optimizes the judged metric directly. Moved here from a "Later" step of
"Fix the hedging objective so SAC has a reason to hedge". No order is set
yet relative to the other open entries.

- [ ] Explain the theory to the user before writing any code
  Required first; no design decision or implementation until the user has
  it. A deep, maths-first explanation, written up (as a document if the
  user wants one) and walked through, covering:
  - **The hedging problem in discrete time.** Prices `S_0..S_n`, a
    liability `Z`, a strategy `δ_k` decided on the information `I_k`
    available at `t_k`. Terminal P&L
    `PL(Z, p_0, δ) = -Z + p_0 + Σ_k δ_k (S_{k+1} - S_k) - Σ_k c(δ_k - δ_{k-1}, S_k)`,
    with `c` proportional (`ε|Δδ|S`) and/or fixed costs. Why perfect
    replication exists only in complete, frictionless, continuous-time
    markets (Black-Scholes), and why discrete rebalancing, costs,
    stochastic volatility or jumps make the market incomplete, so hedging
    becomes risk minimization rather than replication.
  - **Risk measures.** Monetary, convex and coherent risk measures
    (monotonicity, cash invariance, convexity, positive homogeneity) and
    why they matter for prices. Optimized certainty equivalents
    `ρ(X) = inf_w { w + E[ℓ(-X - w)] }`; CVaR as the special case
    `ℓ(x) = max(x, 0)/(1-α)` (Rockafellar-Uryasev) and why the extra
    variable `w` is learned jointly with the network; the entropic measure
    `(1/λ) log E[exp(-λX)]` and its link to exponential utility; why
    variance/mean-variance is not monotone and what goes wrong with it.
  - **Indifference pricing.** `π(X) = inf_δ ρ(X + (δ·S)_T - C_T(δ))`, the
    price `p(Z) = π(-Z) - π(0)`; how it reduces to the risk-neutral price
    without costs in a complete market, and how costs and risk aversion
    move it away. What this means for the "mean P&L excludes the premium"
    caveat in the README.
  - **The neural network approximation.** Strategies parametrized as
    `δ_k = F_θ(I_k, δ_{k-1})`; why the previous position must be an input
    under costs (the optimal strategy is path-dependent through it);
    universal approximation and the paper's result that the optimal
    risk over network strategies converges to the true optimum as the
    networks grow; one network with time as an input vs one per step.
  - **Training.** Monte-Carlo estimate of the risk over a batch of paths,
    gradient through the whole episode (backprop through time): P&L is
    linear in `δ` and the paths don't depend on `θ`, so the gradient is
    exact pathwise, unlike policy gradients. Subgradients of `|Δδ|`, the
    variance of CVaR gradients (only the tail `1-α` of each batch
    contributes), batch size and learning-rate choices.
  - **Comparison with this project's SAC and with stochastic control.**
    Deep Hedging vs actor-critic RL (known differentiable P&L vs learned
    critics, terminal objective vs per-step reward, no exploration noise);
    vs the HJB / utility-indifference approach (Hodges-Neuberger, Davis-
    Panas-Zariphopoulou) and the asymptotic no-trade band of
    Whalley-Wilmott, which is what `delta_band` approximates.
  - **American options.** The liability depends on the holder's exercise
    time `τ`: hedging against a given exercise rule (here LSM's, with steps
    after `τ` masked, as `HedgingEnv` does) vs robust hedging over all
    stopping times (superhedging, `sup_τ`), and what the choice means for
    the hedge and the price.
  - **Challenges.** Non-convex training and local minima; exploding or
    vanishing gradients over 365 steps; memory of BPTT over a batch of full
    episodes; costs making the objective non-smooth and holding still hard
    to represent with a continuous output (the same issue `gap_fraction`
    hit, and the reason the `band` action exists); dependence on the
    training measure (P vs Q, here GBM under Q), model risk and
    out-of-distribution behaviour when the simulator is wrong; overfitting
    to the simulated paths.
  - **Pros and cons.** Model-agnostic in the dynamics, handles costs,
    constraints and any risk measure, optimizes the reported metric,
    fast at inference; vs needing a good market simulator, no optimality
    guarantee in practice, opaque strategies, sensitivity to the risk
    measure and hyperparameters.
  - **Applications.** Hedging books of vanilla and exotic options under
    costs, liquidity and position limits; stochastic and rough volatility
    models; variance swaps and other path-dependent payoffs; risk-adjusted
    pricing desks (J.P. Morgan's use of it); data-driven market generators
    as the simulator; extensions in the literature (Deep Bellman Hedging,
    robust and model-uncertainty versions).
- [ ] Decide the design with the user
  Risk measure (CVaR via the OCE `w`, entropic, std), network inputs
  (same 5-feature state as `HedgingEnv`?) and action parametrization
  (direct hedge, residual on delta, or the `band` form, which handles
  fixed costs), one network vs per-step, and how to handle exercise.
- [ ] Differentiable batched rollout in PyTorch
  A torch version of `HedgingEnv.rollout_pnl`'s terminal-mode arithmetic
  (trading costs, payoff at exercise, masking after it), so gradients flow
  through the whole episode. It duplicates `step()`'s P&L, so a test must
  check that the two agree exactly, fixed cost and early exercise included.
- [ ] Training loop and CLI
  New module (e.g. `AmericanHedging/DeepHedging.py`) and a
  `train-deep-hedge` subcommand, reusing validation checkpointing on
  held-out paths; `test-hedge --policy deep` (or reuse `rl` with a
  checkpoint type) evaluated through the same `evaluate()`.
- [ ] Dispatch test-agent/review-agent
- [ ] Compare against SAC and the baselines
  Same protocol as the README tables (2000 seed-999 episodes, 4 training
  seeds, call and put): 0.1% cost against delta and SAC, 1% and 1% +
  $0.05 fixed against the best fixed band and the learned band.
- [ ] Update README/CLAUDE.md, record results in BUGS.md

## ML surrogate: simple basis, calls and puts, regenerated dataset

All pending MLP work in one entry, ending with a single dataset
regeneration. It collects the former "Use the simple polynomial basis for
the MLP's LSM" and "Price calls and puts with one ML surrogate" entries, and
the deferred dataset regeneration from "Fix LSM exercise decisions and
Greeks". The current dataset predates the horizon fix: its labels are
biased low by up to 0.0012·K. The code changes come first; the existing
checkpoint stops loading once the MLP takes a 5th input, so `test-ml` stays
broken until the retrain at the end.

Design choices:
- **No legacy 4-input support.** The MLP always takes 5 inputs. The old
  4-feature dataset and MLP checkpoint were deleted (their results stay in
  the README for comparison), so `test-ml` has nothing to run on until the
  retrain.
- **Feature layout:** `[S0/K, T_norm, r_norm, sigma_norm, is_call]`, the
  new column appended so the first four keep their meaning; `is_call` is a
  raw 0/1, not normalized, and `features_norm.npz` is unchanged.
- **Generator:** `--option_type {call,put,both}`, default `both`. Each
  parameter combination gets one GBM simulation and one row per type. Call
  labels use the control-variate LSM (`sigma` passed, exact BS for a call
  without dividends), put labels plain LSM. The 5th column is written for
  single-type datasets too.
- **`test-ml` benchmark:** the same 50-point grid per type, the call one
  against the control-variate LSM, with ITM/ATM/OTM by moneyness flipped
  for calls (ITM = `S > K`).

- [x] Switch the MLP's LSM to the simple basis
- [x] Add the option type as an input feature
- [x] Mixed call/put generation in `generate_dataset.py`
- [x] Report errors per option type in `test-ml`
- [x] Dispatch test-agent/review-agent (APPROVE, 430/430 passing)
- [x] Store the LSM settings in the dataset, read them in `test-ml`
- [ ] Generate the new dataset
  Run `python -m datasets.generate_dataset` with the mixed generation and
  the simple basis. It carries the horizon fix, replacing the stale
  `data/american_option_dataset.npz` and `data/features_norm.npz`
  (~40 min at the defaults for puts alone; roughly double with calls).
- [ ] Retrain and compare
  The mixed model must not regress on puts versus a put-only model trained
  on the same regenerated puts, and must match BS on calls.
- [ ] Update README/CLAUDE.md
  New MLP numbers, the 5-input model, the basis defaults, and drop the
  README's note that the dataset predates the horizon fix.

## Bring the README up to date

- [x] Rerun `price`, `test-ml` and pytest; refresh their numbers
- [x] Document the control variate, horizon fix, simple-vs-Laguerre equivalence and stale ML dataset
- [x] Update the usage table and `AmericanHedging/` architecture notes

## Learned no-trade band action mode (`--action_mode band`)

`gap_fraction` can only hold still in the limit `w -> 0` (tanh never reaches
-1), so a sampled or even deterministic "hold" is a tiny trade that still
pays `fixed_tc`. A learned, state-dependent no-trade band makes an exact hold
the outcome for a whole region of states, and contains the `delta_band`
baseline (the bar to beat under costs) as a constant action.

- [x] Add `band` mode to `RLPolicy`
- [x] Support vector actions in `train()` and `ReplayBuffer`
- [x] Wire the CLI
- [x] Tests and docs
- [x] Compare against `gap_fraction` and the best `delta_band` (matches/beats the band at 1% and at 1% + $0.05 fixed - see BUGS.md)

## Hedge under trading costs (where delta hedging overtrades)

With only a 0.1% proportional cost and an accurate LSM delta, daily delta
hedging is close to optimal: the call policy learned nothing beyond it (see
BUGS.md, "RL hedging policy trails the LSM delta hedge"). With larger costs,
rebalancing to delta every day spends a lot on small trades, so a policy
that trades less should do better. That's the setting where learned hedging
is supposed to pay off.

Two issues shape the plan:
- **Fair baseline.** The classical answer to costs is a no-trade band:
  rebalance to delta only when the position has drifted more than `b` away
  from it. RL has to beat that, not just daily delta.
- **Fixed costs can't currently be avoided by the policy.** HedgingEnv
  charges the fixed cost on any trade above 1e-8. A continuous Gaussian
  policy never outputs exactly its current position, so it would pay the
  fixed cost every step whatever it learns. Proportional costs don't have
  this problem. A fixed-cost regime therefore needs a "no trade" mechanism
  first. That's a design decision for the user, taken after the baseline
  numbers are in.

- [x] Expose the proportional cost as `--prop_tc`
- [x] Add a banded-delta baseline (`--policy delta_band --band b --band_target {delta,edge}`)
- [x] Measure baselines over a cost grid
  95% CVaR (×K) on the 2000 seed-999 episodes. Each band column gives the
  best band and its target:

  | Regime | Call: delta | Call: best band | Put: delta | Put: best band |
  |---|---|---|---|---|
  | tc 0.1% | -0.112 | -0.110 (0.02 edge) | -0.110 | -0.105 (0.05 edge) |
  | tc 0.5% | -0.148 | -0.123 (0.05 edge) | -0.152 | -0.119 (0.05 edge) |
  | tc 1% | -0.198 | -0.136 (0.1 edge) | -0.208 | -0.130 (0.1 edge) |
  | tc 2% | -0.302 | -0.156 (0.1 edge) | -0.322 | -0.151 (0.1 edge) |
  | fixed $0.01 | -0.149 | -0.116 (0.05 delta) | -0.145 | -0.112 (0.1 delta) |
  | fixed $0.05 | -0.295 | -0.126 (0.1 delta) | -0.290 | -0.125 (0.1 delta) |

  - At today's 0.1% cost a band barely helps, which matches RL learning
    nothing for calls.
  - From 0.5% upward, daily delta is badly beaten. At 2% its put CVaR
    (-0.322) is worse than not hedging at all (-0.186).
  - As expected, band-edge wins under proportional costs and
    rebalance-to-delta wins under fixed costs, and the best band widens
    with cost (0.05-0.1).
  - That width matters for the residual policy: with `residual_scale` 0.1
    it can only stay 0.1 from delta by saturating tanh, so cost regimes
    need a wider residual (0.2-0.3).
- [x] Decide with the user (proportional 1%, residual_scale 0.25, fixed-cost mechanism deferred)
- [x] Train and validate RL at proportional 1% (beats daily delta, loses to the band: keeps trading daily - see BUGS.md)
- [x] Decide with the user: action parametrization that can hold still (fraction of gap)
- [x] Add `action_mode="gap_fraction"` to RLPolicy (`--action_mode`)
- [x] Retrain at 1% with `--action_mode gap_fraction` (big gain, still behind the band; still improving at episode 1000 - see BUGS.md)
- [x] Retrain with 3000 episodes (plateaus by episode ~1750, still 0.006-0.009 behind the band - see BUGS.md)
- [x] Decide with the user: next lever (stop here and commit; risk_lambda sweep and Deep Hedging left for later)
- [x] Dispatch test-agent/review-agent (APPROVE, 351/351 passing)
- [x] Update README/CLAUDE.md, record results in BUGS.md

## Checkpoint the hedging policy on a fixed validation set

`train()` saves the policy with the best trailing-50 mean *raw* reward,
which mostly measures which paths were drawn, not hedge quality, and never
considers the untrained policy (≈ delta under the residual form). See
BUGS.md, "RL hedging policy trails the LSM delta hedge".

Design choices:
- **Validation set:** paths simulated with their own seed (default
  `seed + 1000`, which never hits the seed 999 used for evaluation), with
  their own LSM fit, as `test-hedge` does. The first `val_paths` (1000)
  paths are each run once, in fixed order, in terminal mode. The metric is
  then exactly `test-hedge`'s P&L. The validation env is built before the
  training simulation, because `GBM` reseeds the global `np.random` and the
  training stream should stay the one `--seed` gives.
- **Batched rollout:** a sequential rollout takes ~60 ms per episode, so
  1000 paths every 50 episodes would roughly double training time.
  `HedgingEnv.rollout_pnl(hedge_fn, path_idxs)` steps all paths together,
  with one batched policy call per time step. It duplicates `step()`'s
  terminal P&L arithmetic, so a test must check that it matches the
  per-step loop exactly, fixed cost and early exercise included.
- **Criterion:** P&L std by default (the hedging error); `cvar` is an
  option. The untrained policy is scored and saved before training starts,
  then the policy is re-scored every `val_every` (50) episodes and after the
  last one. `--val_paths 0` restores the old trailing-50 criterion.
- **Shared deterministic action:** `deterministic_hedge(policy, states)` in
  `Hedge.py`, used by both validation and `test-hedge`, so the two can't
  drift apart.

- [x] Batched terminal-P&L rollout in HedgingEnv
- [x] Validation checkpointing in train()
- [x] Wire into train-hedge and test-hedge
- [x] Dispatch test-agent/review-agent (APPROVE, 270/270 passing)
- [x] Validate (puts beat LSM delta by 20-25% std, calls match it - see BUGS.md)
- [x] Update README/CLAUDE.md
- [x] Fix "CVaR" being the VaR in evaluate() and val_metric (see BUGS.md)
- [x] Address review-agent suggestions (APPROVE, 293/293 passing)

## Residual-on-delta hedging policy

With the fixed LSM, the plain RL policy's P&L std is ~2x the LSM delta
hedge's (BUGS.md, "LSM early-exercises American calls..."): it tracks the
exact delta (corr 0.92-0.96) but sits 0.07-0.10 away from it. Learning the
whole hedge ratio from scratch wastes the policy's capacity on recovering a
delta the state already contains. Instead the policy outputs a correction:
`hedge = clip(delta + c·tanh(u), -1, 1)`, so an untrained policy (u ~ 0)
already hedges at delta and only has to learn deviations (e.g. trading less
under costs).

Design choices:
- The critic sees the squashed residual `tanh(u)`, not the final hedge
  ratio. Delta is in the state, so no information is lost, and `_squash`'s
  log-prob/entropy machinery stays unchanged. Warmup's uniform random
  actions become uniform residuals, i.e. delta ± c.
- `c` is a `residual_scale` buffer on `RLPolicy`, so it is saved in the
  checkpoint and `test-hedge` cannot use a value that differs from training.
  `0` disables it (the policy outputs the hedge ratio directly, as before).
  Old checkpoints without the key load as `0`.
- Default `c = 0.1` in `train-hedge`, the scale of the RL/exact-delta gap.

- [x] Residual mapping in RLPolicy
- [x] Use it in train() and test-hedge
- [x] Dispatch test-agent/review-agent (APPROVE, 220/220 passing)
- [x] Validate (beats plain RL, still trails LSM delta - see BUGS.md)
- [x] Update README/CLAUDE.md

## Support dividend-paying stocks

Everything assumes a non-dividend stock, which is why an American call
equals the European call (no early exercise) and why the call halves of the
hedging and ML work are closed-form. Dividends make the American call a
genuine American option. No order is set yet relative to the other entries.

- [ ] Explain the theory to the user before writing any code
  Required first; no implementation until the user has it. Cover:
  - Continuous dividend yield `q` vs discrete cash dividends on ex-dates,
    and which one to model.
  - The risk-neutral drift `r - q`.
  - Black-Scholes-Merton with `q`, and put-call parity with `q`.
  - Why dividends make early exercise of an American call optimal (just
    before ex-dividend dates, or deep ITM when `q > r`), and how the put
    changes.
  - The CRR tree probability with `r - q`.
  - What changes in LSM: the control variate stays valid, but it is no
    longer exact for calls.
  - Hedging P&L now includes the dividends earned on the stock position.
- [ ] Add `q` throughout
  `GBM`, `BlackScholes`, `BinomialTree`, `LongstaffSchwartz`,
  `option_value.py`, `HedgingEnv` (dividend income on the hedge
  position) and the CLI flags. Default `q = 0` must reproduce the current
  results exactly.
- [ ] Validate against the tree
  American call and put with `q > 0`: LSM price and exercise agreement vs
  CRR, deltas vs tree deltas. Check whether the call control variate still
  helps now that calls have an early-exercise premium.
- [ ] Dispatch test-agent/review-agent
- [ ] Update README/CLAUDE.md

### The ML surrogate with `q`

With `q > 0` the American call stops being plain Black-Scholes, so the
surrogate needs `q` as an input. Instead of a second network (or an
`is_call` input) for calls, use the put-call symmetry of American options
under GBM (McDonald & Schroder 1998): `C(S, K, r, q, sigma, T) =
P(K, S, q, r, sigma, T)`, the call is the put with spot and strike swapped
and rate and dividend yield swapped. Per unit of strike (homogeneity):
`C / K = m * p(1 / m; rate = q, yield = r)`, where `p` is the put per unit
of strike. One put dataset and one put network then price both.

- [ ] Check the symmetry on the tree first
  `binomial_put_batch` (and a call tree) with `q`: American call vs put
  with `(S, K)` and `(r, q)` swapped, over the domain, to ~1e-12. Only then
  build on it.
- [ ] Put dataset over `(m, T, r, q, sigma)`
  A call query becomes a put query at `1 / m` (so `z -> -z`) with `r` and
  `q` swapped, so the put domain must cover both roles: identical sampling
  ranges for `r` and `q`, and a `z` range and `m` cap symmetric around 0
  (the `m` cap (0.4, 2.5) is; the `z` range [-3.5, 3] must widen to
  [-3.5, 3.5]). Premium target and Black-Scholes
  part become Black-Scholes-Merton with `q`.
- [ ] Price helper and evaluation for calls
  `price` maps a call to the symmetric put and back (`m * p(1 / m)`, delta
  and gamma through the chain rule). `test-ml` scores calls against a call
  tree with `q`, per `|z|` bucket like the puts, plus the `q = 0` case,
  where the call must stay Black-Scholes.

## Fix LSM exercise decisions and Greeks

Diagnosis in BUGS.md ("LSM early-exercises American calls and mis-estimates
delta"). The moneyness bins zero out ITM delta and cause look-ahead bias in
exercise, and the ITM-only value fit extrapolates OTM Greeks. Plain LSM's
spurious call exercise is regression noise against a ~0.1 margin. Both the
ML dataset (price only) and the RL environment (exercise times, delta,
gamma) depend on this. Prototypes in the scratchpad, 4 seeds, call + put,
back each step.

- [x] Pathwise-delta regression for the Greeks, drop bins/ridge (APPROVE, 167/167 passing)
- [x] European control variate in the continuation regression, calls only (APPROVE, 179/179 passing)
- [x] Fix the 364-day horizon (APPROVE, 190/190 passing)
- [x] Reference-based tests
- [x] Dispatch test-agent/review-agent (APPROVE on each step, 196/196 passing)
- [x] Retrain the hedging policies (LSM delta now matches exact delta; RL trails it - see BUGS.md)

## Fix the hedging objective so SAC has a reason to hedge

Analysis of why earlier SAC tuning attempts (alpha, lr, 10x longer
training, a cleaner delta feature) all plateaued on `no_hedge` (full write-up in BUGS.md, "RL hedging objective gives no
incentive to hedge"): the problem is the objective SAC is optimizing, not its
hyperparameters. Under the risk-neutral GBM drift, every hedge has ~the same
*expected* P&L (README table: delta -0.10 vs no-hedge -0.11 mean, only the std
differs), so an expected-reward maximizer is indifferent to hedging and
transaction costs tip it to not trading. `risk_lambda`'s per-step `r - λr²`
penalty actively pushes toward `no_hedge`, since the option liability only
appears in the terminal reward. `gamma=0.99` discounts that terminal payoff to
~2.6% from t=0, and per-step rewards (~1e-2, mean signal ~1e-4) are
drowned by the entropy term (`α·log π` ~ 0.2 per step).

- [x] Accounting-P&L reward in HedgingEnv
- [x] Expose SAC discount gamma, default 1.0 in train-hedge
- [x] Reward scaling + automatic entropy tuning
- [x] Wire new defaults into train-hedge, update docs
- [x] Dispatch test-agent/review-agent (APPROVE, 141/141 passing)
- [x] Validate: rerun training, check action_delta_corr and test-hedge metrics (beats LSM delta on 4 seeds, out-of-sample - see BUGS.md)
- [x] Address review-agent suggestions (APPROVE, 152/152 passing)
- [x] Validate on puts (beats LSM delta on 4 seeds, out-of-sample - see BUGS.md)
- [x] Fix LSM spurious early exercise / delta bias (own entry above)
- [x] Residual-on-delta policy (own entry above)
- [x] Checkpoint on a fixed validation path set (own entry above)
- [ ] Deep Hedging as an alternative to SAC (own entry above)
