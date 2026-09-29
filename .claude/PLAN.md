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
- [ ] Decide fix: clamp vs. higher max_degree (or both)
  `reg_type=laguerre` produced results identical to `simple` to 10
  significant figures: `simple` `{1,x,x²,x³}` and Laguerre polynomials of
  degree <=3 span the *same* function space, so a least-squares fit finds
  the same best-fit polynomial either way - switching basis at a fixed
  degree cannot change the fitted function, and delta is that function's
  derivative either way. Ruled out as a fix entirely. Remaining candidates:
  raise `max_degree` (a genuinely higher-capacity fit, unlike a basis swap)
  and/or clamp delta/gamma to their theoretically valid range post-regression
  as a correctness safety net regardless of fit quality.
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
- [ ] Re-run the risk_lambda sweep
  Once a training config reliably competes with/beats the `delta` baseline,
  re-run the `risk_lambda` sweep (0.0, 0.01, 0.05, 0.1, 0.5) with that config
  to get a meaningful read on the risk-shaping effect.
