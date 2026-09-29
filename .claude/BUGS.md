# BUGS.md

Log of problems/bugs found while working on this project — indicative of
real issues encountered and their resolutions, not a work-planning tool.
See `.claude/CLAUDE.md`'s "Bug tracking" section for the policy. New
feature implementation that isn't a bug or a fix does not belong here; see
`.claude/PLAN.md` for that.

## LSM early-exercises American calls and mis-estimates delta (open)

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

**Root cause (suspected, not confirmed):** exercise happens whenever
`exercise_payoff >= continue_val`, so any downward bias or noise in the
regressed continuation value triggers spurious exercise. Ridge shrinkage on
the raw, highly collinear `[1, x, x², x³]` features (x = S/K ≈ 1) biases the
fit, which fits the jump from 22% to 57% when ridge + binning are enabled.

**Status:** open. Candidate directions: evaluate Greeks and exercise
against a reference (exact BS for calls, binomial tree for puts) rather than
a range check; standardize features before Ridge; use the
`AmericanHedging/option_value.py` value functions (exact call / tree put)
for the hedging state's delta and the exercise boundary, instead of LSM.

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
Status: objective fixed; the remaining gap is tracked separately.

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
