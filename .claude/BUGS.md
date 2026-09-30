# BUGS.md

Log of problems/bugs found while working on this project — indicative of
real issues encountered and their resolutions, not a work-planning tool.
See `.claude/CLAUDE.md`'s "Bug tracking" section for the policy. New
feature implementation that isn't a bug or a fix does not belong here; see
`.claude/PLAN.md` for that.

## RL hedging policy trails the LSM delta hedge (open)

**Symptom:** once the LSM delta was fixed (see "LSM early-exercises American
calls..."), the SAC policy's P&L std was ~2x the LSM delta hedge's. It
tracked the exact delta but sat 0.07-0.10 away from it.

**First attempt: residual-on-delta policy** (PLAN.md, "Residual-on-delta
hedging policy"). The hedge is `clip(delta + 0.1·tanh(u), -1, 1)`. Same
protocol as the table in that entry: `train-hedge` defaults, 1000 episodes,
training seeds 42/1/2/3, evaluated on the same 2000 seed-999 episodes. P&L
in units of K, reported as mean / std / 95% CVaR; the last column is the
policy's mean |hedge - LSM delta| over 200 of those episodes:

| Policy | Call | Put |
|---|---|---|
| LSM delta | -0.084 / **0.014** / **-0.108** | -0.082 / **0.012** / -0.103 |
| RL residual, seed 42 | -0.085 / 0.017 / -0.114 (0.038) | -0.082 / 0.020 / -0.117 (0.046) |
| RL residual, seed 1 | -0.083 / 0.026 / -0.125 (0.063) | -0.080 / 0.015 / -0.106 (0.035) |
| RL residual, seed 2 | -0.084 / 0.022 / -0.120 (0.045) | -0.081 / 0.016 / -0.110 (0.028) |
| RL residual, seed 3 | -0.084 / 0.020 / -0.118 (0.034) | -0.081 / 0.013 / **-0.101** (0.026) |
| RL plain (previous) | 0.029-0.036 std | 0.019-0.023 std |
| No hedge | -0.107 / 0.154 / -0.435 | -0.064 / 0.073 / -0.173 |

- **The residual policy is clearly better than the plain one.** Std fell
  from 0.029-0.036 to 0.017-0.026 for calls and from 0.019-0.023 to
  0.013-0.020 for puts.
- **It still doesn't beat LSM delta.** Only put seed 3 gets close: its CVaR
  is marginally better (-0.101 vs -0.103), but its std is slightly worse
  (0.013 vs 0.012).
- **Std rises with how far the policy strays from delta.** For both option
  types, the seed that deviates least has the lowest std and the one that
  deviates most has the highest. The learned corrections are noise, not
  improvements.

**Root cause (current reading):** in this setup there is nothing for the
residual to learn. Costs are proportional (`tc = 0.001`) with no fixed cost,
and the LSM delta is now close to exact (0.005-0.010 from it). Delta hedging
is then close to optimal, so any learned deviation only adds variance. Two
things make it worse:
- **The checkpoint criterion** (best trailing-50 raw mean reward) selects on
  path luck, not hedge quality.
- **The untrained policy is never a candidate.** It is ≈ delta, but the
  first checkpoint is only saved after 50 episodes of training.

**Status:** open. Next candidates: checkpointing on std/CVaR over a fixed
validation path set (already in PLAN.md), and evaluating with a nonzero
`--fixed_cost` or higher proportional cost, where delta hedging overtrades
and a correction has something real to learn.

## Simulated horizon was one step short of the maturity (fixed)

**Symptom:** found during the LSM diagnosis below. With `maturity=1,
sampling=365`, the LSM call price with an exact control variate came out at
10.433, the Black-Scholes price for 364/365 of a year, not 10.451 for one
year.

**Root cause:** `GBM.simulate_paths` produced `int(maturity*sampling)` rows
spaced `maturity / rows` apart, starting at row 0 = S0, so the last row sat
at `maturity * (rows-1)/rows`. Every consumer treated that last row as
expiry, so they all priced a shorter option:
- `LongstaffSchwartz`, `option_value.py` and `HedgingEnv`.
- The ML dataset labels, where `generate_dataset.py` uses a weekly grid for
  puts. At T=0.25 there are 13 rows, so the option was 7.7% short. Measured
  against a CRR tree, the label error is up to -0.0012·K: -3.4% ATM and
  -20% relative deep OTM. It shrinks to ~-0.0002·K at T=3. For scale, the
  MLP's test MAE is 0.0028·K.
- `MonteCarlo` discounted by the full T while using the short last row.

**Fix:**
- `simulate_paths` returns `int(maturity*sampling) + 1` rows, so the last
  row is exactly at `maturity`.
- LSM (`self.dt`), `option_value.py` and `HedgingEnv`'s normalized time now
  use a spacing of `maturity / (rows - 1)`.
- The LSM call with the control variate now prices 10.4506 = BS(T=1), and
  the CRR put reference is 6.090.

**Status of artifacts built on the old grid:**
- `data/american_option_dataset.npz` and `data/features_norm.npz` carry the
  label bias above.
- The hedging checkpoints were trained with the old state normalization and
  the old LSM.
- Both need regenerating or retraining; see PLAN.md.

## LSM early-exercises American calls and mis-estimates delta (fixed)

**Symptom:** found while sanity-checking the accounting-P&L reward's value
function. For an American call on a non-dividend stock early exercise is
never optimal, yet `LongstaffSchwartz` (seed 42, 3000 paths, degree 3)
flags 57% of paths as early-exercised with the `train-hedge` defaults
(`ridge_lambda=0.06, moneyness_bins=30, moneyness_bins_window=0.3`), and
22% even unregularized. Its delta is also far from the exact Black-Scholes
delta: mean absolute error 0.15-0.36 across the timeline with the fix
(near maturity, mean LSM delta 0.21 vs true 0.56, correlation ~0.6-0.7),
0.16-1.3 without it. The earlier Greeks fix was validated only on "fraction
of deltas inside [0,1]", which these numbers show is a weak criterion. Put
deltas are similarly off, worst on out-of-the-money paths (the regression
is fit on ITM paths only, so OTM Greeks are polynomial extrapolation).

**Impact on hedging:** mostly through delta. The delta state feature is
noisy and the `delta` baseline trades on it, so it's a weaker reference
than a true delta hedge (std 0.060 vs 0.015 for an exact-BS delta hedge on
the same paths). `exercise_time` sets each `HedgingEnv` episode's length,
but measured impact is small: the spurious exercises cluster near
maturity (mean episode length 343 vs 364 steps), and forcing every path
to maturity shifts every policy's metrics by only ~0.001-0.006 std (see
the validation below).

**Diagnosis (seed 42, 10000 paths, degree 3; reference = exact BS for the
call, CRR tree for the put; five regression configs compared):**

| Config | Call price | Call early-ex | Put price | Put early-ex (tree 46%) |
|---|---|---|---|---|
| plain LR | 10.309 | 35% | 6.118 | 43% |
| ridge 0.06 | 10.269 | 36% | 6.020 | 46% |
| bins only | 10.903 | 57% | 6.430 | 50% |
| ridge + bins (hedging default) | 10.400 | 56% | 6.030 | 52% |
| standardized ridge + bins | 10.702 | 54% | 6.281 | 50% |

References: call 10.433 (BS), put 6.084 (tree). On these paths the European call
payoff averages 10.282 ± 0.145, so no feasible stopping rule can beat ~10.28.

1. **Moneyness binning is the main cause of both symptoms, and the earlier
   Greeks fix made delta worse.**
   - 30 quantile bins leave each bin with ~170 paths spanning ~0.3% of S/K.
     Inside a bin the columns `[1, x, x², x³]` are nearly identical, so the
     slope is unidentifiable.
   - Without ridge, delta explodes: ITM MAE 4.7, and 95% of deltas fall
     outside the valid range.
   - With ridge (the default), the weak direction is shrunk away. The fit is
     flat within each bin, so ITM delta is ~0: the bias is -0.47 to -0.56
     for the call and +0.66 to +0.74 for the put over the last 30% of the
     timeline. A zero delta passes the old "inside [0,1]" check, which is
     why the earlier fix looked like a success.
2. **Small bins also cause in-sample foresight bias.** A 4-parameter fit on
   ~170 paths fits the paths' own future noise. Paths whose realized future
   is poor get a low continuation value and are exercised. Evidence:
   - Bins-only prices the call at 10.90, ~4 SE above anything a real
     stopping rule can reach on these paths.
   - Early exercise jumps from 35% to 57% and starts much earlier (median 44
     days before expiry vs 9).
   - The inflated cashflows propagate backward as regression targets:
     targets exceed the reference by +0.8 even in the first half of the
     timeline, where binning is off.
   - With the defaults, ridge's downward shrinkage partly cancels this, so
     the price (10.40) looks right for the wrong reason.
3. **Plain LSM still exercises the call spuriously, but only in the last
   ~10 days.** For an ITM call, continuation minus intrinsic is
   `K(1-e^{-rτ}) + put(τ)`, about 0.11 near expiry. The regression error
   there has std 0.15, so each of those rows is close to a coin flip
   (35% of paths, median 9 days to expiry). Spurious exercise is structural
   whenever the true early-exercise premium is 0. It costs almost nothing
   in price (10.309 vs 10.282 achievable) and, as measured, little in
   hedging.
4. **OTM Greeks are pure extrapolation.** The fit uses ITM paths only, and
   about half of alive paths are OTM.
   - Plain LR: OTM delta bias of +0.4 to +0.9 for the call near expiry. The
     true delta goes to 0, but the cubic's slope stays near 1.
   - The put has similar OTM errors of 0.5 to 0.8.
   - This accounts for most of plain LR's 7-13% invalid deltas. Plain LR's
     ITM delta is reasonable (MAE 0.03-0.12).
5. **Ridge penalizes the intercept.** Ridge runs with `fit_intercept=False`
   and a constant column, so the intercept is shrunk along with the other
   coefficients. The continuation value is biased low (-0.07 for the call,
   up to -0.24 for the put), which moves exercise earlier. The suspicion
   that ridge caused the 22%→57% jump was wrong: ridge alone gives 36%,
   the same as plain LR's 35%. Binning causes the jump.
6. **Minor: the simulated horizon is 364 days, not 365.** 365 rows give
   364 intervals of `T/365`, so every model built on the simulation prices
   a 0.9973y option. GBM, LSM and `option_value.py` agree with each other,
   but comparisons against `BinomialTree`/BS at `T=1` carry a small bias
   (~0.02 in price).

**Prototype results.** A standalone LSM, reproducing the current class
exactly with the plain basis (10.309, 35.2%), was run on 4 seeds × call
and put with 10000 paths.

- **The call's 35% early exercise is estimation noise, not a code bug.**
  - It falls with more paths: 42% at 2500, 35% at 10000, 12% at 40000.
  - It varies from 19% to 35% across seeds.
  - LSM estimates the price, not per-path exercise times. Wrong decisions
    near the boundary cost at most the ~0.1 margin, so the price stays right
    while individual exercise times don't. `HedgingEnv` treats
    `exercise_time` as ground truth, which LSM doesn't guarantee.
- **The European price as an extra basis function does not fix it.** Early
  exercise stays at 28-50% at 10000 paths. The right function being in the
  span doesn't reduce coefficient noise. This retracts the claim in the
  earlier version of this entry.
- **The European price as a control variate on the regression target does
  fix the call.**
  - Regress `y - Z` on the polynomial basis, where `Z` is the realized
    discounted European payoff, and add back the exact European value.
  - For paths never exercised early, `y - Z = 0` exactly, so the noise
    vanishes. The call gets 0% early exercise and 100% agreement on all
    4 seeds.
  - For the put it is mixed: early exercise is 46-51% (tree 44-46%) with
    94-98% agreement, and the in-sample price is 1-2% lower than plain LSM.
    It needs an out-of-sample comparison.
- **Delta: regress pathwise deltas on `[1, x, x², x³, BS delta]` over all
  alive paths.**
  - The pathwise delta target is `±e^{-r(τ-t)} S_τ/S_t` if the path pays
    off, 0 otherwise.
  - Delta MAE is 0.001-0.017 (call) and 0.009-0.035 (put) in every time
    bucket, ITM and OTM alike. At most 1.5% of deltas fall outside the
    valid range, and max |Δ| is 1.8, with no binning or ridge.
  - By comparison, differentiating the value fit gives OTM MAE of 0.3-0.9
    and max |Δ| of 13-48, which is where the old "huge deltas" came from.
  - Without the BS-delta feature, pathwise regression still reaches MAE
    0.15-0.23 near expiry, because a cubic can't follow the steepening
    delta curve.

**Fix:**
1. **Pathwise-delta regression.** Delta is now a regression of pathwise
   deltas over all paths on the basis plus the BS delta, and gamma is its
   derivative. Ridge and the moneyness bins were removed.
2. **European control variate for calls.** It is opt-in; `train-hedge` and
   `test-hedge` enable it for calls only. Out of sample (rule fitted on 10k
   paths, applied to 100k fresh ones):
   - Call: the plain rule is worth 10.355 against 10.380 for the control
     variate, which is the European value on those paths.
   - Put: the plain rule is worth 6.078 against 6.009 for the control
     variate, with a true price of 6.084. That's why it stays off for puts.
3. **Horizon fixed.** See "Simulated horizon was one step short".

**Hedging validation after the fix.** 2000 out-of-sample episodes on seed
999, terminal-mode P&L in units of K, reported as mean / std / 95% CVaR:

| Policy | Call | Put |
|---|---|---|
| LSM delta | -0.084 / **0.014** / -0.108 | -0.082 / **0.012** / -0.103 |
| Exact delta (BS / tree) | -0.084 / 0.015 / -0.108 | -0.082 / 0.013 / -0.104 |
| RL, 4 seeds (retrained) | -0.079 to -0.083 / 0.029-0.036 / -0.126 to -0.137 | -0.082 to -0.085 / 0.019-0.023 / -0.119 to -0.128 |
| No hedge | -0.107 / 0.154 / -0.435 | -0.064 / 0.073 / -0.173 |

- **The LSM delta hedge now matches the exact one.**
  - Its std fell from 0.060 to 0.014 for the call and from 0.044 to 0.012
    for the put.
  - Its mean distance from the exact delta along the episodes is 0.005
    (call) and 0.010 (put).
  - The call has 0% early exercise, so every episode runs the full 365
    steps.
- **The RL policies improved but now trail the delta baseline.**
  - Their std fell from 0.040-0.051 to 0.029-0.036 for the call, and from
    0.023-0.033 to 0.019-0.023 for the put.
  - They track the exact delta (correlation 0.92-0.96) but sit 0.07-0.10
    away from it on average, which is where their extra variance comes
    from.
  - The earlier "RL beats the delta baseline" result held only because the
    old LSM delta was broken.
  - Tracked in "RL hedging policy trails the LSM delta hedge".

## RL hedging objective gives no incentive to hedge (fixed)

**Symptom:** after the actor-divergence fix below, every trained SAC hedging
policy lands on top of the `no_hedge` baseline (README table: mean/std/CVaR
-0.11/0.15/-0.41 for RL vs -0.11/0.15/-0.40 for no-hedge, vs
-0.10/0.05/-0.18 for delta). Five independent diagnostics (`alpha=0.05`,
`lr=3e-4`, 5000 episodes, the Greeks fix, 5000 episodes + Greeks fix) left it
unchanged: `mean_std` settles at ~0.86 within ~250 episodes and never moves,
and `action_delta_corr` stays pure noise.

**Root cause:** the objective, not the optimizer. Two findings:
1. *The reward's expectation doesn't depend on hedging.* SAC maximizes
   expected (discounted) reward. Paths are simulated under the risk-neutral
   drift `r` (`core/gbm.py`), so a hedge's P&L has ~zero expectation
   regardless of the position held; hedging reduces the *variance* of P&L,
   not its mean. That's visible directly in the README table - delta and
   no-hedge differ by 0.01 in mean, 3x in std. An expected-reward maximizer
   is therefore ~indifferent between hedging and not, and transaction costs
   tip it to not trading - exactly the plateau observed.
2. *`risk_lambda` shaping penalizes hedging instead of hedging error.*
   `shaped_reward = r - λ·r²` is applied per step (`Hedge.py`), but
   `HedgingEnv` only subtracts the option payoff on the terminal step. Every
   non-terminal reward is just `position·ΔS/K`, so `λ·r²` penalizes holding
   *any* position - it pushes toward `no_hedge`, and can never reward a
   position for offsetting the option, since the option's value changes
   never appear in those per-step rewards. `Σ r_t²` is not `(Σ r_t)²`.

Contributing (not root) causes: `gamma=0.99` discounts the terminal payoff
by `0.99^365 ≈ 0.026` as seen from t=0, and per-step rewards (~1e-2, mean
signal ~1e-4) are orders of magnitude smaller than the entropy term
(`α·log π` ~ 0.2 per step), so the actor loss is almost pure entropy.

**Fix (see PLAN.md "Fix the hedging objective so SAC has a
reason to hedge"):** accounting-P&L reward (per-step reward includes
`-(V_{t+1} - V_t)`, making each step's reward the local hedging error so
the per-step variance penalty becomes meaningful), `gamma=1.0`, reward
scaling, and automatic entropy tuning.

**First validation (one seed, 1000 training episodes, new `train-hedge`
defaults incl. `risk_lambda=1.0`),** evaluated on 1000 identical
terminal-mode paths (mean / std / 95% CVaR, ×K):

| Policy | Mean | Std | CVaR |
|---|---|---|---|
| RL (new objective) | -0.076 | 0.043 | -0.166 |
| Delta (LSM) | -0.098 | 0.053 | -0.183 |
| No hedge | -0.112 | 0.147 | -0.404 |
| Delta (exact BS, reference) | -0.083 | 0.015 | -0.107 |

The policy now hedges (mean position 0.71; action correlation 0.77 with
the LSM delta, 0.86 with the exact BS delta, vs pure noise before) and
beats the LSM delta baseline on all three metrics. Caveat: this first
evaluation reused the seed-42 paths the policy was trained on (in-sample).

**Out-of-sample confirmation:** four policies trained with the same
defaults (training seeds 42, 1, 2, 3; 1000 episodes each), all evaluated
with the baselines on the same 2000 episodes from paths simulated with
seed 999, which no policy trained on:

| Policy | Mean | Std | CVaR | corr(action, LSM Δ / BS Δ) |
|---|---|---|---|---|
| RL, seed 42 | -0.075 | 0.048 | -0.177 | 0.77 / 0.80 |
| RL, seed 1 | -0.079 | 0.051 | -0.185 | 0.96 / 0.60 |
| RL, seed 2 | -0.076 | 0.044 | -0.170 | 0.76 / 0.83 |
| RL, seed 3 | -0.080 | 0.040 | -0.151 | 0.60 / 0.72 |
| Delta (LSM) | -0.094 | 0.060 | -0.195 | |
| No hedge | -0.106 | 0.144 | -0.439 | |
| Delta (exact BS, reference) | -0.083 | 0.015 | -0.107 | |

All four policies beat the LSM delta baseline on mean, std and CVaR, and
the in-sample seed-42 numbers carry over (std 0.043 -> 0.048), so the fix
generalizes and isn't a single lucky run. Run-to-run spread is real
(std 0.040-0.051): the weakest policy (seed 1) copies the LSM delta almost
exactly (corr 0.96), while the better ones partly correct it toward the
exact delta. All four stay ~3x worse in std than an exact-BS delta hedge.
That gap is the LSM *delta* estimate (bug above), not its exercise
timing: every row, the exact-BS delta hedge included, runs on the same
LSM exercise schedule, and forcing every path to maturity instead (600
seed-999 episodes) moves each policy's metrics by only ~0.001-0.006 std
while keeping the same ranking (RL seed 3 0.041 -> 0.042, LSM delta
0.061 -> 0.067, exact-BS delta 0.015 -> 0.015). The spurious exercises
cluster near maturity (mean episode length 343 vs 364 steps despite 56%
of paths being flagged), where an ITM call's value is already close to
its payoff.

**Put confirmation:** same protocol with `--option_type put` (four
policies, training seeds 42/1/2/3, 1000 episodes each, same 2000 seed-999
evaluation episodes). Reference delta = central difference of the
binomial-tree American put value (`option_value.binomial_value_fn`):

| Policy | Mean | Std | CVaR | corr(action, LSM Δ / tree Δ) |
|---|---|---|---|---|
| RL, seed 42 | -0.077 | 0.023 | -0.113 | 0.84 / 0.82 |
| RL, seed 1 | -0.081 | 0.033 | -0.142 | 0.83 / 0.84 |
| RL, seed 2 | -0.079 | 0.025 | -0.125 | 0.79 / 0.82 |
| RL, seed 3 | -0.084 | 0.029 | -0.135 | 0.76 / 0.83 |
| Delta (LSM) | -0.087 | 0.044 | -0.165 | |
| No hedge | -0.063 | 0.067 | -0.169 | |
| Delta (tree, reference) | -0.079 | 0.012 | -0.100 | |

All four put policies beat the LSM delta baseline on std and CVaR (std
25-48% lower) and hedge in the right direction (mean position -0.28 to
-0.35). The LSM put delta is off from the tree delta by 0.18 on average,
enough that the LSM delta baseline's CVaR (-0.165) barely improves on not
hedging (-0.169). The best policy is ~2x the tree-delta hedge's std (vs
~3x for calls). No-hedge has the best *mean* for puts because the hedge
is a short stock position, which pays the risk-free drift instead of
earning it - mean P&L excludes the premium and isn't a hedge-quality
metric; std/CVaR are. Exercise timing: LSM early-exercises 53% of put
paths vs 46% under the tree's optimal rule, agreeing on exercise-or-not
for 93% of paths - much closer than for calls (57% vs 0%).
Status: objective fixed for calls and puts; the remaining gap (the LSM
delta estimate) is tracked separately.

## SAC actor diverged unboundedly during hedge training (fixed)

**Symptom:** a `risk_lambda` sweep (5 independently-trained policies)
produced bit-identical eval metrics across two clusters instead of a
gradient of results. Inspecting the checkpoints directly showed why: every
policy's raw pre-squash `mean` had diverged to a huge magnitude (e.g.
`-63996.84`, `-81410.02`), and every one's `std` was pinned at exactly
`exp(2) ≈ 7.389` — the ceiling of `RLPolicy`'s `torch.clamp(log_std, -20,
2)`. Since `HedgeEnv.step()` clips the actual action to `[-1,1]`, all that
mattered was the sign of that huge mean — the "sweep" was really just two
constant-action policies (always +1 or always -1), none of which had
learned anything.

**Root cause:** `Hedge.py`'s `train()` fed the actor's *raw, unclamped*
sampled action directly into the critics in both the Bellman target and the
actor loss, while the critics were only ever trained on the env-clipped
`[-1,1]` actions actually taken (via the replay buffer). This scored the
actor loss against critic outputs for out-of-distribution actions the
critics had never seen, and the actor learned to chase whatever spuriously
large Q-value that region produced, running away without bound. An
intermediate fix attempt — clamping only the action fed to the critics,
while still evaluating `log_prob` at that same clamped point — didn't work
either: it just moved the divergence into `std`, since `log_prob` of a
fixed clamped point under an ever-more-distant Gaussian mean shrinks
without bound, giving the entropy term a "free" way to keep inflating `std`
once the clamped action itself stopped changing.

**Fix:** added `_squash(dist)`, which samples via `rsample()`, evaluates
`log_prob` on the *raw* sample (well-behaved regardless of how far `mean`
has drifted, since it's self-consistent with the distribution's own
mean/std), and separately `tanh`-squashes the sample into `(-1,1)` for
whatever gets fed to the critics, with the standard log-prob Jacobian
correction (`- log(1 - tanh(raw)^2)`) so the entropy term can no longer get
a free boost from growing `std`. Used consistently in the Bellman target,
the actor loss, and action selection during training; `test-hedge`'s
deterministic evaluation action was updated to match
(`AmericanHedging/Hedge.py`, `scripts/test_hedge.py`).

## LSM delta/gamma estimates unreliable near maturity (fixed)

**Symptom:** `LongstaffSchwartz.delta` (and by extension `.gamma`) falls
outside a call's theoretically valid `[0,1]` range a meaningful fraction of
the time, worse near maturity — and worse than first measured: while the
original seed used to characterize this showed 15.3% invalid at the last
step before expiry, other random seeds showed it as high as 91.6% invalid
at that same step, with values reaching into the tens (and, at higher
`max_degree`, into the thousands). Feeds directly into
`AmericanHedging/HedgeEnv.py`'s state (`self.lsm.delta[t_clamped, i]`) and
into the `delta_hedge` baseline (`AmericanHedging/baselines.py`), which
returns this value unclamped — since `HedgeEnv.step()` only clips the
executed action to `[-1,1]` (not `[0,1]`), a negative delta near maturity
can execute as an actual short position, the wrong trade direction.

**Root cause:** delta/gamma come from the analytic derivative of a global
polynomial (degree 3 by default) fit via least squares to the continuation
value. Near maturity the true value function's kink at `S=K` sharpens
toward the discontinuous step-function payoff; a low-degree global
polynomial can't bend that sharply, so the fit wiggles slightly to
compromise, and differentiating amplifies that wiggle into large swings in
the reported Greek. Two candidate fixes were empirically ruled out first:
`reg_type=laguerre` produces results identical to `simple` to 10 significant
figures (same-degree polynomial bases span the same function space, so
basis choice can't change the fit), and raising `max_degree` makes it
dramatically *worse* (Runge's phenomenon — more degrees of freedom to
oscillate, amplified further by each derivative order): overall validity
drops from 89% at degree 3 to 65% at degree 5 to 50% at degree 7. A
clamp/soft-clamp post-hoc fix was also considered and rejected — it would
only bound the output without correcting the underlying estimate.

**Fix:** two changes to `LongstaffSchwartz`, both opt-in (default off,
preserving exact prior behavior for every other consumer — `price`,
`train-ml`'s dataset generation, etc. — unless explicitly enabled):
1. `ridge_lambda` (opt-in, 0.0 default): switches the per-timestep
   regression from unregularized `LinearRegression` to `Ridge` when > 0.
   Directly targets the mechanism — the overshoot is an ill-conditioned
   least-squares fit producing exploding coefficients, and an L2 penalty
   discourages that.
2. `moneyness_bins`/`moneyness_bins_window` (opt-in, defaults `1`/`1.0` =
   single global fit everywhere, matching prior behavior): splits
   in-the-money paths into quantile bins by `S_t` and fits one model per
   bin, but *only* for timesteps within `moneyness_bins_window` of maturity
   — applying it uniformly across the whole timeline was tried first and
   made the (already well-behaved) middle of the timeline dramatically
   worse (a fit on ~250 points per bin is less stable than one on ~5000,
   and most timesteps don't have a sharp-enough kink to need localizing).

   Gating to near-maturity-only fixed that: the middle of the timeline stays
   exactly as before, and only the region that actually needs it changes.

Validated with `ridge_lambda=0.06, moneyness_bins=30,
moneyness_bins_window=0.3` across 3 random seeds: invalid fraction over the
last 20 timesteps dropped from a 9.6-14.5% baseline to 1.8-9.5%, and at the
single worst (last) timestep specifically, from 15.3-91.6% down to
0.9-1.2% consistently, while price vs. a Binomial Tree benchmark stayed
within the same noise band as the unregularized baseline's own run-to-run
variation (see the price-drift entry below). Wired as the new default for
`train-hedge`/`test-hedge` (`scripts/train_hedge.py`,
`scripts/test_hedge.py`); every other `LongstaffSchwartz` consumer is
unaffected since its own defaults are unchanged.

## Price-vs-Binomial-Tree drift comparisons are noisy across seeds (open, for later)

**Symptom:** found while validating the Greeks fix above. The unregularized
LSM baseline's own price error vs. a Binomial Tree reference varies by
random seed by more than the size of typical regularization effects:
-1.33% (seed 42), +1.31% (seed 7), +1.05% (seed 123) — all from the *same*
unfixed code, just different Monte Carlo draws. A fix's price effect
(e.g. the Greeks fix above, at roughly ±0.5-1pp) is the same order of
magnitude as this seed-to-seed noise floor, which makes "does this fix
change the price" hard to answer precisely from a single seed, and a
config that looks price-neutral on one seed can look mildly price-positive
or price-negative on another purely from simulation noise, not a real
effect.

**Root cause (partially understood):** LSM pricing is a Monte Carlo
estimate with genuine sampling variance; 10,000 paths may simply not be
enough for the price estimate's own confidence interval to be small
relative to the effect sizes being evaluated here. Not yet confirmed
whether more paths would tighten this to a comfortable margin, or whether
there's an additional source of run-to-run variance beyond plain sampling
noise.

**Status:** not fixed, deliberately deferred. Worth revisiting with either
a larger path count (to check if the seed-to-seed spread shrinks as
expected for Monte Carlo noise) or a paired-seed comparison methodology
(same seed, with vs. without a change) rather than comparing to a fixed
Binomial Tree reference computed independently, before treating any future
LSM price-accuracy claim as precise to better than ~1-2%.

## ML surrogate pricer's target had a self-inflicted kink at ATM (fixed)

**Symptom:** the MLP surrogate pricer (`AmericanOptionPricing/ML.py`)
showed a ~16% average error concentrated right at the money, well above
error elsewhere in the moneyness range.

**Root cause:** the training target is `time_value = (price -
intrinsic_value) / K`, and `intrinsic_value = max(K-S, 0)` has a hard kink
at `S == K` — the *expiry payoff* shape — but the American option's actual
price *before* expiry is smooth there. Subtracting the exact intrinsic
value re-introduced a kink into the training target exactly at the money
that the true price does not have, and the model's error spiked there as
an artifact of the label definition, not because ATM is intrinsically
harder to price.

**Fix:** replaced the exact intrinsic value with `smooth_intrinsic_value`,
a softplus approximation that converges to it away from the kink but
rounds off the corner over a small window (`beta_frac`, fixed at `0.06`).
Used consistently everywhere the component is added back to reconstruct
the true price. Full incident writeup (symptom, ablation, root cause) is
in `docs/decisions/softplus-reparameterization.md`; the general pattern
this generalizes to is documented in `.claude/CLAUDE.md`.
