# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

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
- [ ] Tests and convergence study for the pricer
  Unit tests: batch result equals `BinomialTree` per option (plain mode);
  the European variant (no early exercise) converges to Black-Scholes; put
  price >= max(BS European, intrinsic); delta/gamma against finite
  differences. A small study (logged in this entry) of error vs `N` against
  a very fine reference over the sampling domain, incl. the hard corner
  (short T, low sigma, ATM), to pick `N` and BBSR vs CRR, and timing per
  option to size the dataset.
  From the review of `fcc60f1`, also test:
  - The European BBSR error ratio is ~4 per doubling of `N` (clean 1/N^2,
    measured 3.9-4.0 for N = 100-1600). This guards the BS last step, the
    `dt`, the level indexing and the Richardson combination. American BBSR
    is only ~1/N (exercise boundary, see the docstring), so don't assert 4
    there.
  - The 4.4867 literature value (S=36, K=40, r=0.06, sigma=0.2, T=1).
  - Chunked == unchunked bit-identical; broadcasting shapes (scalar, 2-D,
    per-option sigma); delta in [-1, 0], gamma > 0.
  - Every `ValueError` branch: method, NaN/inf, m/T/sigma <= 0, too few or
    odd steps, `chunk_size < 1`, `p` outside (0, 1) (few steps, high r,
    low sigma).
  - `black_scholes_price/delta/gamma` against the `BlackScholes` class with
    array sigma.
  For the study, cross-check the reference independently. Use plain CRR
  averaged over `N` and `N + 1` (which cancels the zig-zag), then Richardson.
  At 40000 steps this agreed with BBSR at 40000 to 5e-9.
- [ ] New generator `ML/surrogate/GenerateTreeDataset.py`
  Same resumable/seeded/chunked machinery as `generate_dataset.py` (reuse
  its helpers rather than copying them). Puts only (calls come from BS; the
  `is_call` input comes back with dividends `q`, see "Revisit the ML
  surrogate with `q` as an input"). Samples `m` log-uniform over (0.4, 1.6),
  `sigma`, `T` log-uniform, `r` uniform, as now, but no `S0`/`K` (homogeneity).
  Stores *raw* parameters and raw labels - `m, T, r, sigma`, the American
  price, the BS European price, delta, gamma - plus the generator settings
  (`teacher="tree"`, `N`, method, seed). No normalization in the generator:
  features and their stats move to load time, so a feature change never
  needs a regeneration. Run as `python -m ML.surrogate.GenerateTreeDataset`.
  Calls `binomial_put_batch(..., chunk_size=32)`, with the reason in a
  comment there: each tree level touches a (chunk, level + 1) array several
  times; at 32 options and 1000 steps that is ~256 KB, which stays in the CPU
  cache, while a whole batch of 1000 (8 MB) goes to main memory on every
  pass. 32 was the fastest of 16-1000 for "bbsr" at 1000 steps (2.8 vs 8.9
  ms/option before the discount was folded into the weights, 1.5 ms after);
  smaller chunks pay more Python overhead per level.
- [ ] Dataset class: features, target and stats at load time
  `ML/surrogate/dataset.py` (`AmericanOptionDataset` or a new class) builds
  the inputs from the raw columns: `log m`, standardized `log T`,
  `log sigma`, `r`, plus `sigma*sqrt(T)`
  and `log m / (sigma*sqrt(T))`. Target = early-exercise premium
  `(P_american - P_european) / K`, delta target = premium delta. Standardization
  stats come from the *training* split only and are saved next to the
  checkpoint (replaces `features_norm.npz` for this pipeline). Split grouped
  by parameter draw if several rows share one.
- [ ] Smooth configurable MLP
  Replace `MLP` (`ML/surrogate/model.py`): depth/width/activation as
  constructor args (start 4 x 128, SiLU), no BatchNorm/Dropout, linear output (try softplus output to force
  premium >= 0 in the ablation). Architecture args saved in the checkpoint so
  `test-ml` rebuilds the right net. A `price(m, T, r, sigma)` helper returns
  `BS + K * net`, and `delta`/`gamma` via autograd, so every consumer
  reconstructs the same way. Retire `smooth_intrinsic_value` from the
  pipeline (keep the function and the decision doc as history).
- [ ] Training loop overhaul
  Seed init and batch order (`torch.manual_seed`), batch 512-1024, Adam with
  cosine decay 1e-3 -> 1e-5, early stopping kept as a safety net, optional
  full-batch L-BFGS finish. Loss = premium MSE + `lambda_delta` * delta MSE
  (differential ML, Huge & Savine 2020) + `lambda_shape` * shape penalties
  (on random unlabeled points, through autograd on the *full* price: put
  decreasing and convex in S, increasing in T and sigma). Both lambdas are
  flags, 0 disables.
- [ ] Evaluation overhaul in `test-ml`
  Factorial grid (m x sigma x T, r fixed and a second r) against an
  *independent*, finer tree reference (not the training teacher's settings),
  BS for calls. Per moneyness bucket: mean abs error in $ (K = 100), relative
  error only where the price is not tiny, implied-vol error; delta and gamma
  errors; no-arbitrage violation rates; MLP vs tree time per price. One unit
  per column, no mixed %/$ averages.
- [ ] Tests, CLI wiring, docs
  Unit tests for the dataset features/target, the net's price/delta
  reconstruction, the loss terms; CLI smoke tests for the new generator
  (tiny run) and `train-ml`/`test-ml` on it. Update CLAUDE.md, README.
- [ ] Smoke run, then full generation
  Only now: a tiny generation (a few hundred rows) through train-ml/test-ml
  end to end, then the full run sized from the timing study.
- [ ] Seeded ablations
  Three seeds each: old architecture vs new on the same data; with/without
  the delta loss; with/without shape penalties; linear vs softplus output.
  Record the numbers here and in `REMINDER.md`.

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
- [ ] Revisit the ML surrogate with `q` as an input
  With `q > 0` the call half of the mixed dataset stops being plain
  Black-Scholes.
- [ ] Dispatch test-agent/review-agent
- [ ] Update README/CLAUDE.md

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
