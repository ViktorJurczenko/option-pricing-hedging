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
- [x] Implement ridge-regularized regression
- [x] Implement moneyness-binned ("batch") regression, gated to near maturity
- [x] Validate empirically (ridge_lambda=0.06, moneyness_bins=30, moneyness_bins_window=0.3 - see BUGS.md)
- [x] Dispatch test-agent/review-agent (APPROVE, 101/101 passing)

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
- [ ] Retrain with Greeks-fixed defaults and refresh README
  The LSM Greeks fix (see the entry above) landed with `train-hedge`/
  `test-hedge` defaulting to the fixed regression. Retrain a policy under
  these new defaults and re-run `test-hedge` for `rl`/`delta`/`no_hedge` to
  see whether a cleaner delta feature unsticks the plateau documented above.
  Also refresh README.md's "RL hedging policy vs. analytic baselines" table,
  which currently shows stale numbers from before the tanh-squashing fix.
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
