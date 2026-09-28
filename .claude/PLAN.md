# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Fix unreliable LSM delta/gamma estimates near maturity

Discovered while diagnosing the hedging policy (see the entry below): a
probe of real training states found ~15% of `LongstaffSchwartz.delta`
values near maturity fall outside a call's theoretically valid `[0,1]`
range (89% valid overall; 2% invalid at t=50 climbing steadily to 15.3%
invalid at t=363, the last step before expiry) even though in-the-money
path counts stay ~55% throughout (not a sparse-data problem). Cause: delta
comes from the analytic derivative of a degree-3 polynomial fit to the
continuation value; near maturity the true value function's kink at S=K
sharpens toward the discontinuous step-function payoff, and a low-degree
polynomial's *derivative* overshoots trying to track that (the same
kink-vs-smooth-fit problem already documented for the ML surrogate's
`smooth_intrinsic_value`, but hitting the Greeks directly here). This feeds
straight into `HedgeEnv`'s state (`self.lsm.delta[t_clamped, i]`), so the RL
policy has been training on a meaningfully wrong delta feature roughly 1 in
7 times near maturity - exactly when hedging decisions matter most.

- [ ] Confirm whether reg_type=laguerre behaves better near maturity
  Re-run the same delta-range-by-timestep check with `reg_type="laguerre"`
  (already an available basis in this codebase) instead of `"simple"`, to
  see whether a basis change alone improves the fit near the kink before
  deciding whether clamping is still needed on top of it.
- [ ] Decide fix: clamp vs. basis change (or both)
  Clamping delta/gamma to their theoretically valid range post-regression is
  a cheap safety net but doesn't address the fit quality; a basis change (or
  higher degree) addresses it directly but needs empirical validation it
  doesn't just move the problem elsewhere (e.g. into the gamma estimate, or
  into price accuracy). Likely lands on clamping as a correctness fix
  regardless, informed by whether the basis change is also worth doing.
- [ ] Implement and validate against a benchmark
  Compare corrected Greeks against Black-Scholes delta as a sanity check
  near maturity (European, but directionally informative for an American
  option close to expiry), and confirm the fix doesn't change LSM price
  outputs (only Greeks) or break `tests/unit/test_basis_functions.py`'s
  analytic-vs-finite-difference checks.

## Close the RL hedging policy's gap to the delta baseline

After fixing the SAC actor-divergence bug (tanh squashing), a `risk_lambda`
sweep showed all 5 trained policies landing on top of the `no_hedge`
baseline and clearly behind the analytic `delta` baseline, with no visible
`risk_lambda` trend — training is stable now, just not learning a useful
hedge yet in the default 1000-episode budget.

- [x] Expose lr/alpha as train-hedge CLI flags
- [x] Add lightweight training diagnostics
- [x] Diagnostic: check alpha=0.05 (inconclusive - see below)
- [ ] Diagnostic: check lr=3e-4
  `alpha=0.05` at the same ~500-episode budget didn't shrink `std` or
  stabilize `action_delta_corr` any better than default `alpha=0.2` -
  entropy isn't the sole bottleneck. Next candidate: default `lr=1e-4` with
  `update_every=10` only allows a few hundred gradient updates in 500
  episodes, which may be insufficient for the critic to converge enough to
  give the actor a real sharpening incentive. Check the standard SAC default
  `lr=3e-4` at the same 500-episode budget before committing to a long run.
- [ ] Diagnostic: train longer
  If `lr=3e-4` (or `alpha=0.05`) looks promising, retrain that config for a
  much larger episode budget (e.g. 3000-5000 episodes) to check it keeps
  improving and actually closes the gap to `no_hedge`/`delta`. If all quick
  checks are inconclusive, fall back to this as a plain "does more time
  alone help" check on the default config.
- [ ] If still flat: targeted hyperparameter sweep
  Sweep update frequency (`update_every`/`grad_steps_per_update`) and
  combinations of `lr`/`alpha` against the `delta` baseline. Skip
  `warmup_steps` — it only affects how much random data is collected before
  learning starts, not whether the actor converges once learning does
  start, so it's an unlikely bottleneck here.
- [ ] Re-run the risk_lambda sweep
  Once a training config reliably competes with/beats the `delta` baseline,
  re-run the `risk_lambda` sweep (0.0, 0.01, 0.05, 0.1, 0.5) with that config
  to get a meaningful read on the risk-shaping effect.
