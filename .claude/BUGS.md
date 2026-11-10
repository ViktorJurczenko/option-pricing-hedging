# BUGS.md

Every problem found while working on this project: what went wrong, why, and how it was fixed. New features and plans belong in `.claude/PLAN.md`, measured results in `docs/results/`.

**How to add an entry**
- Take the next free ID (one above the highest in the index), add a row to the index, and put the entry at the top of its area's section.
- Keep the title a short statement of the symptom. Other docs cite entries by title, so never rename one; an entry's status lives in its header line, not in the title.
- Use this layout:

```markdown
### B-0NN <Symptom in a short sentence>

Status: fixed | open | deferred · Found: yyyy-mm-dd · Files: `path/to/module.py`

**Symptom:** <what went wrong, with numbers and the command or test that showed it>.

**Root cause:** <the mechanism>.

**Impact:** <what was affected, and which earlier results are invalid> (optional).

**Fix:** <the change and its file(s)>.

**Regression test:** `<test_name>`, which fails if the fix is removed (or "none:" and why).
```

For an open problem, replace **Fix:** with **Status:**, giving what is known and the next thing to try. Older entries name their regression test inside **Fix:**. `Files:` lists the main modules the fix changed (not every script it touched), at their current paths; the entry text keeps the paths of its time.

## Index

Open and deferred issues first, then newest first.

| ID | Title | Area | Status | Found |
|---|---|---|---|---|
| [B-002](#b-002) | Price-vs-Binomial-Tree drift comparisons are noisy across seeds | Pricing | deferred | 2026-09-29 |
| [B-036](#b-036) | Width-256 softplus surrogate stopped learning in its first epoch | Surrogate | fixed | 2026-11-06 |
| [B-035](#b-035) | A rerun of some variants dropped the others from summary.md | Tooling and tests | fixed | 2026-10-29 |
| [B-034](#b-034) | Formatting the code broke two mutants' source patterns | Tooling and tests | fixed | 2026-10-29 |
| [B-033](#b-033) | Smoke tests looked for checkpoints outside artifacts/ after the package move | Tooling and tests | fixed | 2026-10-28 |
| [B-032](#b-032) | Overnight ablation run suspended by Modern Standby | Tooling and tests | fixed | 2026-10-28 |
| [B-031](#b-031) | Early stopping cut longer cosine schedules short | Surrogate | fixed | 2026-10-27 |
| [B-030](#b-030) | bucket_metrics crashed on vega without exercised flags | Surrogate | fixed | 2026-10-25 |
| [B-029](#b-029) | test-ml report crashed printing to the Windows console | Tooling and tests | fixed | 2026-10-25 |
| [B-028](#b-028) | Shape penalty's convexity term ~3000x too big | Surrogate | fixed | 2026-10-24 |
| [B-027](#b-027) | Shape-violation rate inflated by exercised points | Surrogate | fixed | 2026-10-24 |
| [B-026](#b-026) | Training test: "always keep the L-BFGS finish" went undetected | Surrogate | fixed | 2026-10-24 |
| [B-025](#b-025) | Training test: a shape point outside the net's domain | Surrogate | fixed | 2026-10-24 |
| [B-024](#b-024) | Diverging training run would crash on missing weights | Surrogate | fixed | 2026-10-24 |
| [B-023](#b-023) | L-BFGS finish seemed to hang: oversized shape pool | Surrogate | fixed | 2026-10-24 |
| [B-022](#b-022) | L-BFGS progress bar did not follow real work | Surrogate | fixed | 2026-10-24 |
| [B-021](#b-021) | PutSurrogate answers m = 0 and m = inf silently | Surrogate | fixed | 2026-10-24 |
| [B-020](#b-020) | An invalid split_ratio is used silently | Surrogate | fixed (in `TreePutDataset`) | 2026-10-24 |
| [B-019](#b-019) | A bad tree flag locks the tree generator's resume directory | Surrogate | fixed | 2026-10-22 |
| [B-018](#b-018) | Tree Greeks wrong just inside the exercise region | Pricing | fixed | 2026-10-22 |
| [B-017](#b-017) | Dataset generation loses all work when interrupted | Surrogate | fixed | 2026-10-20 |
| [B-016](#b-016) | Pipeline tests flake on tqdm's partial-block characters | Tooling and tests | fixed | 2026-10-20 |
| [B-015](#b-015) | LongstaffSchwartz.calculate_option_price returns 0 before predict_payoff | Pricing | fixed | 2026-10-02 |
| [B-014](#b-014) | r2_fn returns nan/-inf for constant targets | Surrogate | fixed | 2026-10-02 |
| [B-013](#b-013) | HedgingEnv.rollout_pnl crashes on an empty path list | Hedging and RL | fixed | 2026-10-02 |
| [B-012](#b-012) | HedgingEnv.step() keeps stepping after done | Hedging and RL | fixed | 2026-10-02 |
| [B-011](#b-011) | HedgingEnv mishandles options exercised at row 0 | Hedging and RL | fixed | 2026-10-02 |
| [B-010](#b-010) | RL policy under trading costs keeps trading every day; a simple band beats it | Hedging and RL | fixed | 2026-10-01 |
| [B-009](#b-009) | "95% CVaR" in hedging evaluation is actually the 95% VaR | Hedging and RL | fixed | 2026-10-01 |
| [B-008](#b-008) | RL hedging policy trails the LSM delta hedge | Hedging and RL | fixed for puts; calls match delta | 2026-09-30 |
| [B-007](#b-007) | Simulated horizon was one step short of the maturity | Pricing | fixed | 2026-09-30 |
| [B-006](#b-006) | LSM early-exercises American calls and mis-estimates delta | Pricing | fixed | 2026-09-29 |
| [B-005](#b-005) | RL hedging objective gives no incentive to hedge | Hedging and RL | fixed | 2026-09-29 |
| [B-004](#b-004) | SAC actor diverged unboundedly during hedge training | Hedging and RL | fixed | 2026-09-29 |
| [B-003](#b-003) | LSM delta/gamma estimates unreliable near maturity | Pricing | fixed | 2026-09-29 |
| [B-001](#b-001) | ML surrogate pricer's target had a self-inflicted kink at ATM | Surrogate | fixed | 2026-09-29 |

## Pricing

<a id="b-018"></a>

### B-018 Tree Greeks wrong just inside the exercise region

Status: fixed · Found: 2026-10-22 · Files: `optionpricing/pricing/american/batch_binomial_tree.py`

**Symptom:** found by step 2 of the MLP overhaul (tests for
`binomial_put_batch`). On a 675-point grid over the sampling domain, `"bbsr"`
at 400 steps returned delta = -1.0102 at (m = 0.8, T = 5, r = 0.08,
sigma = 0.2) and gamma = -0.189 at (m = 0.449, T = 1, r = 0.08, sigma = 0.6),
outside a put's bounds (delta in [-1, 0], gamma >= 0). `"crr"`/`"bbs"` only
broke them by rounding (gamma -1e-10, delta -1 - 1e-13). A first fix clipped
the Greeks to those bounds; a comparison with a 16000-step reference then
showed the same point at 1000 steps had gamma 0.69 against a true 0, inside
the bounds and the worst gamma error of 63 options. The clip hid the bound
violations, not the error.

**Root cause:** every bad point is exercised today: the put is worth exactly
`1 - m` around today's spot, so delta = -1 and gamma = 0. The tree measures
gamma over the level-2 nodes `m d^2, m, m u^2` (delta over level 1), and when
`m` is just inside the exercise region the upper node is in the
continuation region, where the price curves. That error is not smooth in N
(BBS delta -1.0000 at N, -0.9898 at N/2), so Richardson's
`2 D(N) - D(N/2)` pushed it past the bounds. Not the `dt` offset of the
level-1/2 nodes: the extended tree (Pelsser-Vorst), which reads all three
Greeks at t = 0, was no better (delta max error 1.2e-2 vs 2.6e-4).

**Impact:** none yet (no dataset built), but delta and gamma are planned
dataset columns (delta a differential-ML label), and 44% of the domain's
draws are exercised today.

**Fix:** a put exercised today (American, `m < 1`, price at most `1 - m`
plus `_EXERCISE_TOL = 1e-12`) gets its exact values: price `1 - m`, delta
-1, gamma 0. This also covers a `"bbsr"` price below `1 - m` (one of the two
trees exercises, the other doesn't). Measured on the 63 options at 1000
steps: gamma max error 0.69 -> 0.081, mean 1.3e-2 -> 1.4e-3, no bound
violation, so the clip was removed. What remains is inherent: within ~1e-3
of the boundary in `m` the tree's price error (~1e-6) exceeds the true time
value, so the tree can call exercise one step early, and the true gamma
jumps there (gamma 0 instead of 2.1 at m = 0.424, T = 3, r = 0.03,
sigma = 0.4). Unit tests cover the rule, a deep out-of-the-money put
(worth ~0, its payoff, but not exercisable) and the European variant (worth
less than its payoff, no rule).

<a id="b-015"></a>

### B-015 LongstaffSchwartz.calculate_option_price returns 0 before predict_payoff

Status: fixed · Found: 2026-10-02 · Files: `optionpricing/pricing/american/longstaff_schwartz.py`

**Symptom:** found by the test-suite cleanup. Calling
`calculate_option_price()` before `predict_payoff()` returns 0.0. In the
same situation `BinomialTree` raises `AttributeError`, and CLAUDE.md says
there should be no silent default.

**Root cause:** `__init__` preallocates `self.payoff` as zeros, so the
price is the mean of an all-zero row.

**Impact:** none today. Every caller fits first.

**Fix:** `payoff` is `None` until `predict_payoff()` allocates it, and
`calculate_option_price()` raises `AttributeError` while it is `None`, the
same error `BinomialTree` gives.

<a id="b-007"></a>

### B-007 Simulated horizon was one step short of the maturity

Status: fixed · Found: 2026-09-30 · Files: `optionpricing/core/gbm.py`, `optionpricing/pricing/american/longstaff_schwartz.py`, `optionpricing/hedging/env.py`

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
- Both need regenerating or retraining; see PLAN_ARCHIVE.md, "ML surrogate:
  simple basis, calls and puts, regenerated LSM dataset" (superseded).

<a id="b-006"></a>

### B-006 LSM early-exercises American calls and mis-estimates delta

Status: fixed · Found: 2026-09-29 · Files: `optionpricing/pricing/american/longstaff_schwartz.py`

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

<a id="b-003"></a>

### B-003 LSM delta/gamma estimates unreliable near maturity

Status: fixed · Found: 2026-09-29 · Files: `optionpricing/pricing/american/longstaff_schwartz.py`, `optionpricing/pricing/american/basis_functions.py`

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

<a id="b-002"></a>

### B-002 Price-vs-Binomial-Tree drift comparisons are noisy across seeds

Status: deferred · Found: 2026-09-29 · Files: `optionpricing/pricing/american/longstaff_schwartz.py`

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

## Surrogate

<a id="b-036"></a>

### B-036 Width-256 softplus surrogate stopped learning in its first epoch

Status: fixed · Found: 2026-11-06 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** in the width ablation
(`python -m optionpricing.ml.experiments.run_ablations --variants
lambda_delta_0.3_softplus lambda_delta_0.3_softplus_width_256 --seeds
42 43 44 45 46`), all 5 width-256 seeds froze in epoch 1. Their
validation loss was 1.45-1.54 after epoch 1 and moved by at most 8.7e-3
over the remaining epochs until early stopping at epoch 338 (best epoch
11-19). The width-128 runs of the same config fell from 0.10-0.20 to
4.7e-6-9.2e-6. On the grid, the width-256 nets had 0.3452 $ MAE near the
money, against 0.0009 at width 128, 81 bp IV error against 0.60, and
the same numbers on every seed (sd ~0). Their exercised rows showed 0
error only because the payoff floor answered them.

**Root cause:** the softplus output starts far above its target and
overshoots into its flat region. At initialization the output before
the softplus is ~0, so the net predicts softplus(0) = ln 2 ≈ 0.69 per
unit of strike, while the mean training premium is 0.0116 (seed 42's
split). The scaled loss therefore starts at ~515, against ~3.6 for the
linear output. Adam pushes every output down at once, and the large
early gradients build momentum that carries the outputs past the
target into the region where the softplus slope vanishes. Nothing then
brings them back. The trained width-256 seed-42 net's output before
the softplus ranges over [-48.4, -8.1] (median -15.9) on 20000
standard-normal inputs, where the softplus slope is ~1e-7. The
width-128 net's ranges over [-35.5, +7.7] (median -6.8).

Measured by replaying the first 2 epochs step by step (an ad hoc
script in `artifacts/runs/b036_diag/`, seeds 42 and 43, the
`lambda_delta_0.3_softplus` settings, 2 torch threads), with the
largest output before the softplus on 4096 held-out rows:

| config | step the largest output falls below -5 | largest at step 640 | loss at step 640 |
|---|---|---|---|
| width 128, lr 1e-3 (as trained) | below −5 from step ~48 to ~100 / ~140, then recovers | −1.2 / −1.0 | 7.3e-2 / 9.6e-2 |
| width 256, lr 1e-3 (as trained) | 17 / 17, never recovers | −12.2 / −12.7 | 1.75 / 1.92 |
| width 256, linear output | not reached | +0.3 / +0.3 | 1.4e-2 / 2.6e-2 |
| width 256, lr 3e-4 | not reached | −1.2 / −1.0 | 0.11 / 0.15 |
| width 256, lr warm-up over epoch 1 | not reached | −1.2 / −1.0 | 6.5e-2 / 0.11 |
| width 128, output bias at inverse-softplus(mean premium) | not reached | −1.2 / −1.1 | 6.9e-3 / 9.1e-3 |
| width 256, output bias at inverse-softplus(mean premium) | not reached | −1.2 / −1.0 | 2.6e-3 / 4.9e-3 |

(seed 42 / seed 43). Width 256 passes −3 in ~12 steps
instead of ~25 at width 128, because more weights move together under
the same lr. Width 128 at lr 1e-3 nearly dies too, recovering only
after ~100-130 steps below −5. Starting the output bias at
log(expm1(0.0116)) = −4.45 makes the initial loss ~1.3 and removes the
overshoot at both widths. After 2 epochs its loss is ~10x lower than
the width-128 run as trained.

**Impact:** the width-256 rows of `artifacts/runs/ablations/summary.md`
measure a failed optimization, not network capacity. The question
"does a wider net help" is unanswered. The width-128 softplus runs
(`softplus_output`, `lambda_delta_0.3_softplus`, the dataset variants)
trained correctly but lost their first ~150-180 steps to the near-collapse.
Rerun with the fix (2026-11-10, seeds 42-46, `artifacts/runs/ablations_b036/`):
width 256 trains normally and is within 0.6 of the larger seed sd of
width 128 on every error, and the fix lowers width 128's implied-vol error
from 0.60 ± 0.03 to 0.44 ± 0.04 bp (`docs/results/surrogate-training.md`,
"Width: 256 units do not beat 128").

**Fix:** `initial_model` in `tree_training.py`, which `train_surrogate`
now starts from, sets the softplus output layer's bias term to
log(expm1(mean premium of the trained rows)), so the untrained net
predicts the mean premium and the initial premium term is ~1. The
linear output keeps PyTorch's initialization, so its recorded runs
still reproduce. The derivation, why the bias term rather than a lower
lr or a warm-up, and the measurements: `docs/results/surrogate-training.md`,
"Output initialization: start at the mean premium". The width-256
ablation and the 5 width-128 seeds are rerun with the fix in a new
`--out_dir` (`.claude/PLAN.md`).

**Regression test:** `test_softplus_output_starts_at_the_mean_premium`
and `test_training_starts_the_softplus_output_at_the_trained_rows_mean_premium`
in `tests/unit/test_tree_training.py`, which fail without the bias
term (the premium term is then ~500, and the trained bias term is
PyTorch's random one).

<a id="b-031"></a>

### B-031 Early stopping cut longer cosine schedules short

Status: fixed · Found: 2026-10-27 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** in step 9's data-vs-steps check, `train-ml --epochs 400`
(default `--patience 20`) stopped at epoch 278 on 2^18 rows and at
epoch 175 on 2^19 rows, and both nets came out *worse* than the
200-epoch runs on the same data (grid `|z| < 1` price MAE $0.0053 vs
$0.0037 at 2^18, $0.0033 vs $0.0022 at 2^19).

**Root cause:** the cosine schedule spans all `epochs`, so with 400 the
learning rate decays half as fast: at epoch 175 it is still ~6e-4,
where the validation loss jitters from epoch to epoch. 20 epochs
without a new best are then likely, and the run stops before the
low-rate tail where the final gains come from. With 200 epochs the
rate is already small when the jitter would run out, which is why no
default run ever stopped (best epochs 192-200).

**Impact:** any run longer than the default schedule silently loses its
end; the first data-vs-steps comparison was invalid.

**Fix:** the comparison was rerun with `--patience 400` (early stopping
off). In `train_surrogate`, an epoch without improvement now counts only
once the rate is within `PATIENCE_LR_FRACTION` (10%) of the way from
`lr_min` to `lr`, the last ~20% of the cosine whatever `epochs` is
(`ML/surrogate/tree/tree_training.py`); pinned by
`test_early_stopping_does_not_count_epochs_while_the_rate_is_high`,
which fails if the condition is removed.

<a id="b-030"></a>

### B-030 bucket_metrics crashed on vega without exercised flags

Status: fixed · Found: 2026-10-25 · Files: `optionpricing/ml/surrogate/tree/tree_evaluation.py`

**Symptom:** found in the step-7 review: `bucket_metrics(..., vega=v)`
called without `exercised` raised `TypeError: bad operand type for
unary ~: 'NoneType'` in the implied-vol mask, instead of computing or
refusing anything meaningful.

**Root cause:** the implied-vol error is only defined off the exercise
region, so its mask is `(vega >= MIN_IV_VEGA) & ~exercised`; both
arguments are optional, and nothing checked that `exercised` came with
`vega`.

**Impact:** none on results: `evaluate_test_split`/`evaluate_grid` always
pass both. Only a direct caller (a notebook, a new evaluation) would hit
the bare `TypeError`.

**Fix:** `bucket_metrics` raises `ValueError("vega needs the exercised
flags: ...")` up front (`ML/surrogate/tree/tree_evaluation.py`), pinned by
`test_bucket_metrics_refuses_vega_without_the_exercised_flags`.

<a id="b-028"></a>

### B-028 Shape penalty's convexity term ~3000x too big

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** with `lambda_shape = 1` the step-6 training loop tripled the
test price error (4.4e-4 -> 1.5e-3 at 4k rows) in every region alike,
7x at 32k rows, while violations barely moved. An independent agent
study (REMINDER.md, "The shape penalty, investigated") traced it.

**Root cause:** `shape_violations` scaled the derivatives by `m`, `m^2`,
`T`, `sigma` to put them "in price units" so they could share the
premium's `Var(premium)` normalization. That holds for units, not size:
the price bends over a distance `w = sigma sqrt(T)` in log m, so
`m^2 d2V/dm2` is ~price / `w^2`, up to ~20,000x a price at `w = 0.007`.
On a lambda-0 model the penalty was 54.8 against a labeled loss of
0.017, 99.99% of it convexity (re-checked here), 82% from the third of
points with `w < 0.06`; its gradient was ~10,000x the labeled loss's
and opposed to it (cosine -0.77). The optimizer flattened the curvature
at the cost of the fit. The derivatives themselves were right (autograd
matches finite differences).

**Impact:** none on a shipped model: `lambda_shape` already defaulted
to 0 after the first trial; it would have broken any ablation run with
the penalty on, and step 7's violation rates, which reuse the function.

**Fix:** the m columns are `w m dV/dm` and `w^2 m^2 d2V/dm2`, the
first- and second-order price change over a one-standard-deviation
move. With it, lambda 1 is within seed noise of lambda 0 (4.7e-4 vs
4.4e-4), so `lambda_shape` stays 0 (no measurable gain, ~3.4x the step
time). `test_shape_violations_are_the_relu_of_the_full_price_derivatives`
checks the scaled formula; dropping the `w` factors fails it.

<a id="b-027"></a>

### B-027 Shape-violation rate inflated by exercised points

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_evaluation.py`

**Symptom:** found while trying the step-6 training loop on a 4,096-row
dataset. The rate of points breaking a no-arbitrage condition (any
`shape_violations` column > 1e-6) sat at ~30% for every run, with or
without the shape penalty, which made the penalty look useless.

**Root cause:** a measurement artefact, not a network flaw. On puts
exercised today (29% of the points) `V = 1 - m`, so three of the four
conditions hold with equality (`d2V/dm2 = dV/dT = dV/dsigma = 0`). Any
tiny negative wobble of the net around those true zeros counts: 65-85%
of the exercised points were flagged even at a 1e-3 tolerance, against
6-9% of the others.

**Impact:** none yet (no metric is reported); it would have misled
step 7's no-arbitrage column.

**Fix:** `test-ml` (`ML/surrogate/tree/tree_evaluation.py::violation_rates`)
reports the rates for the continuation and exercised regions separately,
counting a point only above a tolerance (`--violation_tol`, default
1e-4) on the `w`-scaled violations (see the convexity-scaling entry
above), computed by the same `price_derivatives` training penalizes
with. The independent study confirmed the diagnosis: ~80% of exercised
points flag for every model, a fit-noise number, not an arbitrage rate.
`test_violation_rates_count_strictly_above_the_tolerance_per_region`
checks the split and the strict tolerance.

<a id="b-026"></a>

### B-026 Training test: "always keep the L-BFGS finish" went undetected

Status: fixed · Found: 2026-10-24 · Files: `tests/unit/test_tree_training.py`

**Symptom:** a manual mutation check of `ML/surrogate/tree/tree_training.py`:
replacing `kept = metrics["loss"] < best_loss` by `kept = True` left
all tests passing.

**Root cause:** in `test_lbfgs_finish_is_kept_only_if_it_improves_validation`
the real L-BFGS finish happened to improve validation on both cases, so
its "reject" branch never ran.

**Impact:** none (caught before the first commit), but a regression
returning L-BFGS's worse weights would have shipped unnoticed.

**Fix:** `test_an_lbfgs_finish_that_makes_validation_worse_is_thrown_away`
replaces the finish with one that wrecks the weights and checks the best
Adam weights come back.

<a id="b-025"></a>

### B-025 Training test: a shape point outside the net's domain

Status: fixed · Found: 2026-10-24 · Files: `tests/unit/test_tree_training.py`

**Symptom:** `test_shape_violations_are_the_relu_of_the_full_price_derivatives`
failed on its first point: `shape_violations` reported 1.7 and 6.2, the
finite differences 0.

**Root cause:** the test's mistake. That point (`m = 0.7`, `T = 0.1`,
`sigma = 0.2`) has `z = -5.6`, so `PutSurrogate.price`, which the
finite differences use, routes it to "exercised" (`1 - m`) and never
sees the net; `shape_violations` differentiates the net directly.

**Fix:** the point moved inside the domain (`T = 1`, `z = -1.8`), and
the test now asserts every point takes the `net` route, so a point
leaving the domain fails loudly instead of comparing two different
functions.

<a id="b-024"></a>

### B-024 Diverging training run would crash on missing weights

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** found reading the step-6 draft: if the validation loss is
NaN at every epoch, `best_state` stays `None` and
`model.load_state_dict(None)` fails with an unrelated-looking error.

**Root cause:** `NaN < best_loss` is always False, so a diverged run
never records a best epoch.

**Impact:** none observed (no run diverged).

**Fix:** `train_surrogate` raises `ValueError("the validation loss was
never finite: training diverged ...")`
(`test_a_validation_loss_that_is_never_finite_raises`).

<a id="b-023"></a>

### B-023 L-BFGS finish seemed to hang: oversized shape pool

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** in the first step-6 trial, the run with `lbfgs_steps=100`
printed nothing for minutes, after three runs of ~5-20 s each. A
20-step run then showed 1.3 s per loss evaluation on 2,621 training
rows.

**Root cause:** the fixed shape pool for L-BFGS was a constant 65,536
points, 25 times the training rows, and the shape term (three
derivatives plus a second-order one per point) dominated each
full-batch evaluation.

**Impact:** a 100-step finish would have taken ~3 minutes on a toy
dataset.

**Fix:** the pool has as many points as training rows, as many
unlabeled points as labeled ones.

<a id="b-022"></a>

### B-022 L-BFGS progress bar did not follow real work

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_training.py`

**Symptom:** found reading the step-6 draft: the L-BFGS finish showed a
bar with `total=1` that filled only when `optimizer.step` returned.

**Root cause:** `torch.optim.LBFGS.step` runs all its iterations inside
one call, and the bar wrapped that call.

**Impact:** no ETA during a finish that can take minutes, against
`.claude/rules/progress-bars.md`.

**Fix:** the closure ticks the bar once per loss evaluation, out of
LBFGS's cap (`max_eval = 1.25 x lbfgs_steps`), and shows the loss.

<a id="b-021"></a>

### B-021 PutSurrogate answers m = 0 and m = inf silently

Status: fixed · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_pricer.py`

**Symptom:** found by the review of step 5 of the MLP overhaul.
`PutSurrogate.price(0.0, 0.5, 0.03, 0.25)` returned price 1, delta -1,
route `exercised`, with no error (`m = 1e-300` the same), and `m = inf`
returned a NaN price through the `black_scholes` route. The docstring,
CLAUDE.md and REMINDER.md all promised invalid input always raises.

**Root cause:** invalid input was meant to fail every domain check and
fall through to `binomial_put_batch`, which validates it. That holds for
NaN (every comparison is False) and for `T`, `sigma`, `r` (bounded by
their ranges), but the two closed-form routes check `z` only, not `m`'s
range: `ln 0 = -inf` satisfies `z < z_low`, and `ln inf = +inf`
satisfies `z > z_high`. The tests tried `m = -1` and `m = NaN` only
(`ln` of both is NaN), so they passed.

**Impact:** none: nothing calls `PutSurrogate` yet (steps 6-7 will).

**Fix:** `price` requires `m_valid = isfinite(m) & (m > 0)` for every
route but the tree, so such queries reach it and it raises.
`test_invalid_queries_reach_the_tree_and_raise` now covers `m = 0`,
`m = inf`, and NaN/inf in `T` and `r`; the `m = 0`/`inf` cases fail
without the fix.

<a id="b-020"></a>

### B-020 An invalid split_ratio is used silently

Status: fixed (in `TreePutDataset`) · Found: 2026-10-24 · Files: `optionpricing/ml/surrogate/tree/tree_dataset.py`

**Symptom:** found by the review of step 4 of the MLP overhaul.
`TreePutDataset(path, split_ratio=-0.1)` built without complaint, and so
does `AmericanOptionDataset`: on 100 rows it gave 90 training and 10 test
rows, i.e. a ratio of 0.9. A ratio above 1 gave an empty test split. (The
review first said the splits overlap; checked, they stay disjoint.)

**Root cause:** `n_train = int(n * split_ratio)` is negative for a
negative ratio, and Python slicing reads `rdm_idx[:n_train]` as "all but
the last `|n_train|` rows", so a ratio `s < 0` acts as `1 + s`. Above 1,
`n_train > n` and the test slice is empty. Neither class validated it.

**Impact:** none observed: every caller passes the default 0.8 or a
`--split_ratio` in (0, 1).

**Fix:** `TreePutDataset` raises a `ValueError` unless
`0 <= split_ratio <= 1` (`test_tree_dataset_rejects_a_split_ratio_outside_0_1`).
`AmericanOptionDataset` is left as is: it goes away when `train-ml`/`test-ml`
switch to the tree pipeline (steps 5-7).

<a id="b-019"></a>

### B-019 A bad tree flag locks the tree generator's resume directory

Status: fixed · Found: 2026-10-22 · Files: `optionpricing/ml/surrogate/tree/generate_tree_dataset.py`

**Symptom:** found by the review of step 3 of the MLP overhaul.
`python -m ML.surrogate.tree.generate_tree_dataset --nb_steps 51` (odd, with the
default `"bbsr"`) failed with "nb_steps must be even", and re-running with
the corrected `--nb_steps 50` then failed with "holds chunks generated with
other settings": the user had to delete `tree_chunks/` by hand. The same
happened with too few steps for the domain (`--nb_steps 20`: "risk-neutral
probability outside (0, 1)").

**Root cause:** `main` only checked the power-of-2 sizes before
`prepare_chunk_dir` saved `config.json`. The tree settings were first used
by `binomial_put_batch` inside the first job, after the config, with the
bad values, was already on disk.

**Impact:** none on data (no chunk was written), only a stuck directory.

**Fix:** `check_teacher` prices the domain's 8 corner puts (every `T`,
`sigma`, `r` bound, at the money) with the requested settings before the
config is written, so the pricer's own checks raise first; the worst
probability case is a corner (longest `T`, lowest `sigma`, highest `r`).
`test_bad_tree_settings_fail_before_the_config_is_written` covers both
flags and the corrected re-run.

<a id="b-017"></a>

### B-017 Dataset generation loses all work when interrupted

Status: fixed · Found: 2026-10-20 · Files: `optionpricing/ml/surrogate/chunks.py`

**Symptom:** the 100,000-sample run was killed at 33% after 31 minutes (the
system ran low on memory with 12 jobs), and again stopped on purpose at 42%.
Each time only `features_norm.npz` was left and everything computed was lost.

**Root cause:** `generate_dataset.py` kept every result in memory and wrote
`american_option_dataset.npz` once, at the very end. The parameter draw was
also unseeded, and so was every sample's `GBM`, so even a re-run could not
have continued the same dataset.

**Fix:** the parameters are drawn from `--seed` (default 42), the samples are
processed in chunks (`--chunk_size`, default 1000) and each finished chunk is
saved atomically to `<output_dir>/chunks/`. Re-running the same command skips
finished chunks, a stored `config.json` makes a resume with different settings
fail instead of mixing datasets, and the chunks are deleted once the dataset
is assembled. Each sample's path simulation is seeded with `--seed` plus its
index, so a redone chunk (and a whole re-run) reproduces its labels too.

<a id="b-014"></a>

### B-014 r2_fn returns nan/-inf for constant targets

Status: fixed · Found: 2026-10-02 · Files: `optionpricing/core/metrics.py`

**Symptom:** found by the test-suite cleanup. With constant targets,
`utils/metrics.py::r2_fn` returns nan for a perfect prediction and -inf
otherwise. Only a numpy RuntimeWarning is shown, so `test-ml` would print
"R²: nan".

**Root cause:** the total sum of squares is 0, and R² divides by it.

**Impact:** none on real test splits, whose labels vary.

**Fix:** `r2_fn` raises `ValueError` when the total sum of squares is 0.
R² is undefined there, and returning a number would be a silent
fallback.

<a id="b-001"></a>

### B-001 ML surrogate pricer's target had a self-inflicted kink at ATM

Status: fixed · Found: 2026-09-29 · Files: `optionpricing/ml/surrogate/legacy/lsm_model.py`

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
harder to price. (The real early-exercise kink sits at the exercise
boundary, a variable level below `K`, not at `K` itself.)

**How this was confirmed (not undertraining):** training on the raw price
(no intrinsic-value subtraction) removed the ATM spike but hurt accuracy
elsewhere, the dynamic-range problem the subtraction was meant to solve;
and more training epochs narrowed the ITM/OTM error but barely touched the
ATM error. More epochs fixing everything *except* the kink region rules
out "just needs more training".

**Fix:** replaced the exact intrinsic value with `smooth_intrinsic_value`,
a softplus approximation that converges to it away from the kink but
rounds off the corner over a small window (`beta_frac`, fixed at `0.06` in
code rather than a CLI flag, so it can't drift out of sync between label
generation and reconstruction). Used consistently everywhere the component
is added back to reconstruct the true price. The width was picked by
sweeping a few values against a fixed benchmark, with training runs fully
seeded (model init, data split, batch shuffling). The general pattern this
generalizes to is documented in `docs/architecture.md`, "Softplus
reparameterization".

## Hedging and RL

<a id="b-013"></a>

### B-013 HedgingEnv.rollout_pnl crashes on an empty path list

Status: fixed · Found: 2026-10-02 · Files: `optionpricing/hedging/env.py`

**Symptom:** found by the test-suite cleanup. `rollout_pnl(hedge_fn, [])`
raises `ValueError` instead of returning an empty array.

**Root cause:** the stepping loop runs to `int(T_exercise.max())`, and
`.max()` of an empty array raises.

**Impact:** none today. Every caller passes at least one path.

**Fix:** an empty `path_idxs` returns `np.zeros(0)`. The former `xfail`
test is now a plain test.

<a id="b-012"></a>

### B-012 HedgingEnv.step() keeps stepping after done

Status: fixed · Found: 2026-10-02 · Files: `optionpricing/hedging/env.py`

**Symptom:** found by the test-suite cleanup. After `step()` returns
`done=True` on a path exercised before the last row, calling `step()` again
silently advances past the exercise row. It charges hedge P&L and, because
`t >= T_exercise` stays true, the payoff again on every extra call. On a
path exercised at the last row it raises `IndexError` instead.

**Root cause:** `step()` has no guard on the episode being over.

**Impact:** none today. Every loop (`train()`, `evaluate()`) stops at
`done`.

**Fix:** `reset()` clears a `done` flag and `step()` sets it. A `step()`
after done raises `RuntimeError` telling the caller to `reset()`. This also
covers zero-length row-0 episodes.

<a id="b-011"></a>

### B-011 HedgingEnv mishandles options exercised at row 0

Status: fixed · Found: 2026-10-02 · Files: `optionpricing/hedging/env.py`

**Symptom:** found by the whole-suite test review. For a deep in-the-money
put (S0=50, K=100), LSM sets `exercise_time = 0` on every path, and
`HedgingEnv`'s two P&L paths disagree on it:
- `rollout_pnl` returns P&L 0.0 for those paths: its loop only touches
  paths with `t < T_exercise`, which is never true for `T_exercise = 0`, so
  the payoff is never charged.
- `step()` still takes one step: `reset(0)` then `step(0.0)` gave reward
  -0.437 with `done=True`, i.e. one hedging step plus the payoff at row 1.

**Root cause:** both assume an episode has at least one step. `step()`
checks `done` only after advancing `t`, and `rollout_pnl` charges the payoff
inside the stepping loop. An option exercised at row 0 should cost its
payoff at S0 with no hedging step, so both are wrong, differently.

**Impact:** none on the published results. At the default S0=K=100 the
intrinsic value at row 0 is 0, so LSM never exercises there. It would
matter for deep-ITM puts, and validation scoring (`rollout_pnl`) and
`test-hedge` (`step()`) would then report different P&Ls on the same paths.
No test covers it: every env test uses `exercise_time >= 1`.

**Fix:** a path exercised at row 0 is a zero-length episode that pays the
payoff at S0, with no hedge ever held.
- `step()` on such a path ignores the action, trades nothing, charges no
  cost and returns `done=True` with reward `-payoff(S0)/K`. In accounting
  mode the reward is `-(payoff(S0) - V_0)/K`, so the episode sum is still
  the terminal sum plus `V_0/K`.
- `rollout_pnl` starts those paths' P&L at `-payoff(S0)`. Its stepping loop
  already skips them.
- Checked by forcing `exercise_time = 0` on two paths of a deep-ITM put
  (S0=50, K=100): both now give -0.5. The regression test sets
  `exercise_time = 0` in the `rollout_pnl` vs summed `step()` test.

<a id="b-010"></a>

### B-010 RL policy under trading costs keeps trading every day; a simple band beats it

Status: fixed · Found: 2026-10-01 · Files: `optionpricing/ml/rl/policy.py`, `optionpricing/hedging/baselines.py`

**Symptom:** found in PLAN_ARCHIVE.md's "Hedge under trading costs". Training with
a 1% proportional cost (`--prop_tc 0.01 --residual_scale 0.25
--val_metric cvar`; 4 seeds, 1000 episodes; evaluated on the 2000
seed-999 episodes at the same cost). 95% CVaR (×K):

| Policy | Call | Put |
|---|---|---|
| Daily delta | -0.198 | -0.208 |
| **Band 0.1, rebalance to edge** | **-0.136** | **-0.130** |
| RL, 4 seeds | -0.173 to -0.186 | -0.140 to -0.153 |

RL beats daily delta clearly, but not the simple no-trade band. Its
validation CVaR improved until about episode 550 for calls, while for puts
it stopped improving after episode 100-150.

**Diagnosis** (300 of the same episodes, call / put):

| Policy | Trading cost (×K) | Turnover | Days holding |
|---|---|---|---|
| Daily delta | 0.066 / 0.062 | 6.6 / 6.2 | 14% / 11% |
| Band 0.1 (edge) | 0.015 / 0.012 | 1.5 / 1.2 | 85% / 84% |
| RL, seeds 3 and 42 | 0.038-0.040 / 0.025-0.027 | 3.8-4.1 / 2.5-2.6 | 26-27% / 14-17% |

RL halves delta's trading but still trades almost every day, while the
band sits still 85% of the time. Cost is the gap: RL's extra trading cost
over the band (≈0.025 call, ≈0.013 put) is the same size as its CVaR gap
to the band.

**Likely root cause (hypothesis):** the residual action is defined
relative to *delta*: `hedge = delta + c·tanh(u)`. To hold still, the
policy must output exactly `tanh(u) = (position - delta)/c` every day,
tracking delta as it moves. Any error in that tracking is a small trade
that costs money. A band holds by construction. So "do nothing", the most
valuable action under costs, is the hardest one for this parametrization
to represent.

**First fix: fraction-of-gap action** (chosen by the user;
`--action_mode gap_fraction`): `hedge = position + w·(delta - position)`
with `w ∈ (0, 1)`, initialised at `w ≈ 0.9`. Same protocol, 1000 episodes:

| Policy | Call CVaR | Put CVaR |
|---|---|---|
| RL residual (above) | -0.173 to -0.186 | -0.140 to -0.153 |
| **RL gap fraction** | **-0.143 to -0.147** | **-0.135 to -0.144** |
| Band 0.1 (edge) | -0.136 | -0.130 |

A large improvement, especially for calls, but still behind the band.
Diagnosis on the same 300 episodes (seeds 3 and 42):

| Policy | Cost (×K) | Turnover | Days with ~no trade | Mean distance from delta |
|---|---|---|---|---|
| Band 0.1 (edge), call / put | 0.015 / 0.012 | 1.5 / 1.2 | 85% / 84% | 0.066 / 0.064 |
| RL gap fraction, call | 0.032-0.033 | 3.2-3.3 | 18-19% | 0.016 |
| RL gap fraction, put | 0.022-0.024 | 2.2-2.4 | 18% | 0.026-0.032 |

The hypothesis is only partly confirmed. The gap-fraction policy improved
by tracking delta more tightly while trading smoothly ("close part of the
gap every day"), not by learning to hold still: it still trades on ~82% of
days. Holding is now easy to represent, but training hasn't found that
behaviour, and validation CVaR was still improving at episodes 900-1000
in every run.

**Longer training doesn't close it.** Same runs with 3000 episodes: call
CVaR -0.142 to -0.145, put -0.136 to -0.137. Every run's best checkpoint
came at episode 900-1750 with no improvement afterwards, so the policy
plateaus 0.006-0.009 behind the band (-0.136 / -0.130).

**Current reading (hypothesis):** an objective mismatch. SAC is trained on
a per-step mean-variance reward, `x - risk_lambda·x²` with `risk_lambda =
1` (accounting mode), but judged on CVaR of terminal P&L. The policy
tracks delta ~4x more tightly than the band (0.016 vs 0.066) and trades
~2x more, i.e. its objective prices tracking error higher, relative to
trading cost, than CVaR does. `risk_lambda` sets that exchange rate, so a
smaller value should move the policy toward the band's trade-off.

**Fix: learned no-trade band** (`--action_mode band`, PLAN_ARCHIVE.md's "Learned
no-trade band action mode"). A 2-D action sets a band half-width `b <=
band_max` (0.2) and a rebalancing target between the band edge and delta;
inside the band the position is held exactly. Same protocol (4 seeds, 1000
episodes, `--val_metric cvar`, 2000 seed-999 test episodes), at the 1%
proportional cost and with a $0.05 fixed cost per trade added (`--fixed_tc
0.05`), where `gap_fraction`'s near-holds each pay the fixed cost. Best
fixed band re-swept for that regime (0.05-0.3, both targets):

| Policy | 1%: Call | 1%: Put | 1% + $0.05: Call | 1% + $0.05: Put |
|---|---|---|---|---|
| Daily delta | -0.198 | -0.208 | -0.380 | -0.390 |
| RL gap fraction | -0.143 to -0.147 | -0.135 to -0.144 | -0.324 to -0.329 | -0.325 to -0.332 |
| Best fixed band | -0.136 (0.1 edge) | -0.130 (0.1 edge) | -0.148 (0.175 delta) | -0.146 (0.175 delta) |
| **RL learned band** | **-0.136 to -0.137** | **-0.123 to -0.128** | **-0.145 to -0.147** | **-0.132 to -0.138** |

(The 1% gap-fraction row was re-evaluated from `checkpoints/tc01gap`; it
matches the numbers above.) The learned band matches the hand-tuned band
for calls at 1% and beats it in the other three cells on every seed,
without a band width or target being tuned. With a fixed cost,
`gap_fraction` is barely better than daily delta, confirming that a
"hold" it can only approach in the limit is paid for every day.

This also settles the hypotheses above: with the same SAC objective
(`risk_lambda = 1`, accounting reward), only the action changed, so the
gap to the band came from the parametrization (holding still was
representable but not reachable by training), not from the
mean-variance-vs-CVaR mismatch. Best checkpoints came at episodes 50-750
for calls and 700-1000 for puts, so puts may still gain from longer
training.

**Status:** fixed by `--action_mode band`. Untried: a `--risk_lambda`
sweep and longer runs for puts.

<a id="b-009"></a>

### B-009 "95% CVaR" in hedging evaluation is actually the 95% VaR

Status: fixed · Found: 2026-10-01 · Files: `optionpricing/hedging/evaluation.py`

**Symptom:** found while reviewing the validation-checkpointing change.
`AmericanHedging/Hedge.py::evaluate()` reports `cvar_95 =
np.percentile(pnls, 5)`, printed as "CVaR 95%". That is the 5th-percentile
P&L, i.e. the 95% VaR. CVaR (expected shortfall) is the *mean* P&L over the
worst 5% of episodes, which is always at or below VaR. `train()`'s
`val_metric="cvar"` reuses the same percentile, so it is consistent with
`test-hedge`, but it carries the same mislabel.

**Impact:** every "CVaR" figure in BUGS.md, README and the `test-hedge`
output is a VaR. Rankings between policies are mostly unaffected, but VaR
ignores how bad the tail is beyond the cutoff, which is what CVaR is meant
to capture.

**Fix:** a new `Hedge.cvar(pnls, level=0.95)` returns the mean of the worst
`max(1, ceil(5% · n))` P&Ls. `evaluate()` returns it as `cvar_95` and
prints the VaR on its own line; `val_metric="cvar"` uses it too.

**Past figures:** every "CVaR" in the entries below this one, and in any
README table before this fix, is the 95% VaR. They are left as logged,
since rankings between policies were mostly unaffected. The
"RL hedging policy trails the LSM delta hedge" entry has recomputed
numbers with both.

<a id="b-008"></a>

### B-008 RL hedging policy trails the LSM delta hedge

Status: fixed for puts; calls match delta · Found: 2026-09-30 · Files: `optionpricing/ml/rl/policy.py`, `optionpricing/ml/rl/sac.py`

**Symptom:** once the LSM delta was fixed (see "LSM early-exercises American
calls..."), the SAC policy's P&L std was ~2x the LSM delta hedge's. It
tracked the exact delta but sat 0.07-0.10 away from it.

All numbers below use the same protocol: `train-hedge` defaults, 1000
episodes, training seeds 42/1/2/3, evaluated with `test-hedge` on the same
2000 seed-999 episodes. P&L is in units of K, reported as mean / std / 95%
VaR / 95% CVaR (CVaR as fixed in the entry above).

**First attempt: residual-on-delta policy** (PLAN_ARCHIVE.md, "Residual-on-delta
hedging policy"). The hedge is `clip(delta + 0.1·tanh(u), -1, 1)`, with the
old checkpoint criterion (best trailing-50 raw mean reward). The last
column is the policy's mean |hedge - LSM delta| over 200 of those episodes:

| Policy | Call | Put |
|---|---|---|
| LSM delta | -0.084 / **0.014** / -0.108 / -0.112 | -0.082 / **0.012** / -0.103 / -0.110 |
| RL residual, seed 42 | -0.085 / 0.017 / -0.114 / -0.122 (0.038) | -0.082 / 0.020 / -0.117 / -0.127 (0.046) |
| RL residual, seed 1 | -0.083 / 0.026 / -0.125 / -0.133 (0.063) | -0.080 / 0.015 / -0.106 / -0.113 (0.035) |
| RL residual, seed 2 | -0.084 / 0.022 / -0.120 / -0.127 (0.045) | -0.081 / 0.016 / -0.110 / -0.118 (0.028) |
| RL residual, seed 3 | -0.084 / 0.020 / -0.118 / -0.126 (0.034) | -0.081 / 0.013 / -0.101 / -0.106 (0.026) |
| RL plain (previous) | 0.029-0.036 std | 0.019-0.023 std |
| No hedge | -0.107 / 0.154 / -0.435 / -0.571 | -0.064 / 0.073 / -0.173 / -0.186 |

- **The residual policy is clearly better than the plain one.** Std fell
  from 0.029-0.036 to 0.017-0.026 for calls and from 0.019-0.023 to
  0.013-0.020 for puts.
- **It still didn't beat LSM delta.** Only put seed 3 got close: its CVaR
  is marginally better (-0.106 vs -0.110), but its std is slightly worse
  (0.013 vs 0.012).
- **Std rose with how far the policy strayed from delta.** For both option
  types, the seed that deviated least had the lowest std and the one that
  deviated most had the highest.

**Root cause:** the checkpoint criterion. The best trailing-50 raw mean
reward selects on which paths were drawn, not on hedge quality, and the
untrained policy (≈ delta under the residual form) was never a candidate,
since the first save only came after 50 training episodes.

**Fix: checkpoint on a fixed validation set** (PLAN_ARCHIVE.md, "Checkpoint the
hedging policy on a fixed validation set"). The policy is scored by P&L
std on 1000 held-out paths (seed `--seed + 1000`); the untrained policy is
saved first, then re-scored every 50 episodes. Same residual policy,
retrained:

| Policy | Call | Put |
|---|---|---|
| LSM delta | -0.084 / 0.0142 / -0.108 / -0.112 | -0.082 / 0.0119 / -0.103 / -0.110 |
| RL, val-checkpointed, seed 42 | -0.084 / 0.0143 / -0.108 / -0.112 | -0.081 / 0.0096 / -0.097 / -0.102 |
| RL, val-checkpointed, seed 1 | -0.084 / 0.0149 / -0.108 / -0.113 | -0.081 / 0.0095 / -0.097 / -0.103 |
| RL, val-checkpointed, seed 2 | -0.084 / 0.0142 / -0.107 / -0.112 | -0.081 / 0.0093 / -0.097 / -0.102 |
| RL, val-checkpointed, seed 3 | -0.084 / 0.0142 / -0.107 / -0.112 | -0.081 / **0.0089** / **-0.096** / **-0.100** |

- **Puts: RL now beats the LSM delta hedge on every seed.** Std is
  0.0089-0.0096 against 0.0119 (20-25% lower), and CVaR is -0.100 to
  -0.103 against -0.110. Validation std fell steadily from ~0.012 to
  ~0.009 over the first 300-550 episodes, then stopped improving; the old
  criterion never kept those policies. The put policies also beat the
  exact tree-delta hedge from the LSM entry below (std 0.012-0.013).
- **Calls: RL matches the delta hedge.** Training never beat the untrained
  policy on validation (except seed 1, marginally, at episode 50), so the
  saved call policies are essentially delta: std 0.0142-0.0149 against
  0.0142.
- **Why puts can beat even the exact delta (hypothesis, not verified):**
  each episode ends at the *LSM's* exercise time. For puts, which LSM
  exercises early on about half the paths, the liability being hedged is
  therefore the put under LSM's slightly suboptimal exercise rule, not the
  tree's optimal one. A learned hedge can adapt to that; a model delta
  cannot. Calls are never exercised early, so there is nothing to adapt
  to, and with only proportional costs delta is already close to optimal.

**Status:** fixed for puts. Calls match delta rather than beat it. Next is
testing with a nonzero `--fixed_tc` or a higher proportional cost, where
delta hedging overtrades and a learned correction has something real to
learn (PLAN_ARCHIVE.md, "Hedge under trading costs").

<a id="b-005"></a>

### B-005 RL hedging objective gives no incentive to hedge

Status: fixed · Found: 2026-09-29 · Files: `optionpricing/hedging/env.py`, `optionpricing/hedging/option_value.py`, `optionpricing/ml/rl/sac.py`

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

**Fix (see PLAN_ARCHIVE.md, "Fix the hedging objective so SAC has a
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

<a id="b-004"></a>

### B-004 SAC actor diverged unboundedly during hedge training

Status: fixed · Found: 2026-09-29 · Files: `optionpricing/ml/rl/sac.py`

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

## Tooling and tests

<a id="b-035"></a>

### B-035 A rerun of some variants dropped the others from summary.md

Status: fixed · Found: 2026-10-29 · Files: `optionpricing/ml/experiments/run_ablations.py`

**Symptom:** after all 24 ablation runs had finished, `artifacts/runs/ablations/summary.md` held a single row, `shape_0.1`. The other 7 variants were run first with `--variants baseline lambda_delta_0 ... lbfgs_200`, then `shape_0.1` on its own with `--variants shape_0.1`, and the second invocation rewrote the summary.

**Root cause:** `main` built the summary from `args.variants` and `args.seeds`, the current invocation's selection, not from what was finished in `--out_dir`. Each run's `metrics.json` was intact, so only the table was wrong.

**Impact:** none on the runs; a rerun of the runner with no `--variants` (0 runs to do) rewrote the full table. Any run of a single new variant, such as the combined one, would have dropped the rest again.

**Fix:** a new `collect_results(out_dir)` reads every `seed_*/metrics.json` of every variant in `VARIANTS` under `--out_dir`, whatever was requested, and `main` summarizes that.

**Regression test:** `test_collect_results_reads_every_finished_run_of_every_variant` (unit), and in `test_a_rerun_skips_finished_runs_and_other_settings_are_refused` (integration), a rerun with `--variants baseline` that must keep the `softplus_output` row; it fails with the old subset-only summary.

<a id="b-034"></a>

### B-034 Formatting the code broke two mutants' source patterns

Status: fixed · Found: 2026-10-29 · Files: `tests/mutation/tree_evaluation_mutants.py`

**Symptom:** after `ruff format .` over the repo, `python -m tests.mutation.tree_evaluation_mutants` stopped at "PDE's 1/2 dropped": "its original text occurs 0 times, expected once". A scan of every mutant against the formatted source found a second stale one, "floor rate counts the wrong route". The unit and integration suites still passed.

**Root cause:** each mutant finds its target by exact source text. The formatter rewrote `sc ** 2` as `sc**2` and split the `floored=..., gamma_err=...` call arguments onto separate lines, so neither pattern occurred any more.

**Impact:** none on results: the runner refuses a mutant whose text is missing instead of skipping it, so nothing passed silently. It does mean any reformatting of `optionpricing/ml/surrogate/tree/tree_evaluation.py` can break the runner.

**Fix:** both patterns rewritten to the formatted source (`- 0.5 * sc**2`; `floored=res.route == "payoff_floor",\n        gamma_err`). 14/14 mutants killed again.

**Regression test:** the runner itself, which fails loudly on a pattern that no longer occurs exactly once.

<a id="b-033"></a>

### B-033 Smoke tests looked for checkpoints outside artifacts/ after the package move

Status: fixed · Found: 2026-10-28 · Files: `tests/integration/test_cli_smoke.py`

**Symptom:** after moving the code into `optionpricing/` and the generated outputs into `artifacts/`, `python -m pytest tests/integration` gave 1 failed, 96 passed: `test_train_hedge_command_writes_a_loadable_checkpoint` asserted that `ROOT / "checkpoints" / "_smoke_test_policy.pth"` existed, while `train-hedge` had written it to `artifacts/checkpoints/`. The band round-trip test passed, but recreated an empty `checkpoints/` folder at the root.

**Root cause:** the scripted path rewrite matched paths written as strings (`"checkpoints/..."`), not paths built from parts (`ROOT / "checkpoints" / ...`). The first test passed the rewritten string to the CLI but checked the part-built path; the second derived its CLI argument from the part-built path, so it stayed self-consistent but wrote outside `artifacts/`.

**Impact:** only the test itself; the failing test left `_smoke_test_policy.pth` behind, since its cleanup removed the wrong path.

**Fix:** both checkpoints are built as `ROOT / "artifacts" / "checkpoints" / ...`. A search for part-built `"data"`, `"checkpoints"`, `"runs"` and `"logs"` paths found no other case.

**Regression test:** the two smoke tests themselves, which fail again if the path and the CLI default disagree.

<a id="b-032"></a>

### B-032 Overnight ablation run suspended by Modern Standby

Status: fixed · Found: 2026-10-28 · Files: `optionpricing/ml/experiments/run_ablations.py`

**Symptom:** the first overnight `python -m experiments.run_ablations`
(started 01:49) had no finished run by 10:57: the 4 running trainings
showed 98-300 s per epoch on their progress bars instead of ~4-14 s.

**Root cause:** this PC's only standby state is Modern Standby
(`powercfg /a`: "Standby (S0 Low Power Idle)"). There, the screen
turning off *is* standby, whatever the sleep timeout ("never" was
set), and desktop programs are suspended. The System event log shows
standby entered at 01:59, when the 10-minute screen timeout ran out,
with only short wakes until 10:57: ~30 min of compute in 9 h. The bars
count wall-clock time, hence the absurd per-epoch figures.

**Impact:** a night lost; no run finished, so none had to be discarded.

**Fix:** `keep_awake()` in `experiments/run_ablations.py` holds
`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
ES_DISPLAY_REQUIRED)` for the whole run and releases it after, also on
an error or Ctrl-C. The display stays on (dim it): on Modern Standby
there is no reliable way to keep a program running with the screen
off. Pinned by the `keep_awake` tests in `tests/unit/test_run_ablations.py`.

<a id="b-029"></a>

### B-029 test-ml report crashed printing to the Windows console

Status: fixed · Found: 2026-10-25 · Files: `optionpricing/ml/surrogate/tree/tree_evaluation.py`

**Symptom:** the first run of the step-7 evaluation report died with
`UnicodeEncodeError: 'charmap' codec can't encode character 'Γ'`
at the first table.

**Root cause:** column headers used `Γ` ("gamma MAE (K·Γ)") and the
timing line `µs`; Python's stdout on this Windows console is cp1252,
which has no `Γ`.

**Impact:** none (caught on the first trial run, before any commit).

**Fix:** ASCII-only labels (gamma now "gamma MAE 1/$", "us") in
`ML/surrogate/tree/tree_evaluation.py`; the `test-ml` CLI smoke test prints the
whole report through a subprocess, so a non-cp1252 character there would
fail it again.

<a id="b-016"></a>

### B-016 Pipeline tests flake on tqdm's partial-block characters

Status: fixed · Found: 2026-10-20 · Files: `tests/helpers.py`

**Symptom:** `test_test_ml_reports_put_and_call_blocks` failed with
`AttributeError: 'NoneType' object has no attribute 'group'` on some runs
only, with a `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f`
from the subprocess reader thread.

**Root cause:** the child writes UTF-8, and `tests/helpers.py::run_python`
decoded its output as cp1252. tqdm prints a partial block (`▏`, byte 0x8f)
only at some progress values, so whether the run failed depended on timing.
A failed decode leaves `stdout` as `None`.

**Fix:** `run_python` decodes as UTF-8 with `errors="replace"`.
