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

- [x] Confirm whether reg_type=laguerre behaves better near maturity (no - dead end, see below)
- [x] Check whether higher max_degree helps (no - much worse, see below)
- [ ] Implement ridge-regularized regression
  A clamp only bounds the output without correcting the estimate, so instead
  address the root cause. `LongstaffSchwartz.fit()` currently fits an
  unregularized `LinearRegression` per timestep; add an opt-in
  `ridge_lambda: float = 0.0` constructor param (default preserves exact
  existing behavior) that switches to `sklearn.linear_model.Ridge` when
  `ridge_lambda > 0`. Ridge directly targets the mechanism: the near-maturity
  overshoot is an ill-conditioned least-squares fit producing exploding
  coefficients, and an L2 penalty discourages exactly that.
- [ ] Implement moneyness-binned ("batch") regression
  Group ITM paths at each timestep into `moneyness_bins` quantile bins (by
  `S_t`) and fit a separate regression per bin (opt-in
  `moneyness_bins: int = 1` constructor param; `1` = current single global
  fit, preserving existing behavior). A single global polynomial has to
  compromise its shape across the whole domain to accommodate the
  near-maturity kink; a local/binned fit only needs to bend sharply in the
  neighborhood where paths actually sit, which is the standard remedy for
  this class of problem (same idea as a regression spline). Watch for a new,
  smaller artifact this could introduce: discontinuities in delta right at
  bin boundaries, from two independently-fit local models disagreeing there.
- [ ] Validate empirically before doing anything else
  Re-run the delta-range-by-timestep diagnostic across candidate
  `(ridge_lambda, moneyness_bins)` combinations and confirm a real
  improvement over the degree-3 baseline (89% valid overall, 15.3% invalid
  at t=363) - including checking bin-boundary discontinuities aren't a new
  problem - before treating this as fixed. Also compare against
  Black-Scholes delta near maturity as an independent sanity check, and
  confirm LSM price outputs (`calculate_option_price`) and
  `tests/unit/test_basis_functions.py`'s analytic-vs-finite-difference
  checks aren't broken - this regression is shared by price, Greeks, and the
  ML surrogate's training-data generation, so a real fix here has a wider
  blast radius than the hedging-only band-aid would have.
- [ ] Dispatch test-agent/review-agent
  Only once the empirical validation above shows a genuine improvement, not
  before - write tests for the new ridge/binning math and re-run the full
  suite to confirm no regressions elsewhere in the shared LSM code path.

## Close the RL hedging policy's gap to the delta baseline

After fixing the SAC actor-divergence bug (tanh squashing), a `risk_lambda`
sweep showed all 5 trained policies landing on top of the `no_hedge`
baseline and clearly behind the analytic `delta` baseline, with no visible
`risk_lambda` trend — training is stable now, just not learning a useful
hedge yet in the default 1000-episode budget.

- [x] Expose lr/alpha as train-hedge CLI flags
- [x] Add lightweight training diagnostics
- [x] Diagnostic: check alpha=0.05 (inconclusive - see below)
- [x] Diagnostic: check lr=3e-4 (inconclusive too - see below)
- [x] Diagnostic: train longer (5000 episodes, still flat - see below)
- [ ] If still flat: targeted hyperparameter sweep
  The 5000-episode run never moved off the `std` plateau (~0.86-0.89) and
  `action_delta_corr` was still pure noise at episode 5000 (+0.63, right
  after +0.02 and -0.61) - three independent diagnostics (`alpha`, `lr`,
  10x duration) all show the same flat plateau, which stopped looking like
  a hyperparameter-sensitivity problem. Before spending more runs on
  `update_every`/`grad_steps_per_update`/combinations, check the "Fix
  unreliable LSM delta/gamma estimates" entry above first - if the delta
  feature itself is unreliable ~15% of the time near maturity, the critic
  may be correctly learning that it's not a trustworthy predictor, which
  would explain a plateau no RL hyperparameter can fix. Only continue this
  sweep if fixing the Greeks doesn't unstick it.
- [ ] Re-run the risk_lambda sweep
  Once a training config reliably competes with/beats the `delta` baseline,
  re-run the `risk_lambda` sweep (0.0, 0.01, 0.05, 0.1, 0.5) with that config
  to get a meaningful read on the risk-shaping effect.
