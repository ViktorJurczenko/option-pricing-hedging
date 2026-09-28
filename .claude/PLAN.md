# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Close the RL hedging policy's gap to the delta baseline

After fixing the SAC actor-divergence bug (tanh squashing), a `risk_lambda`
sweep showed all 5 trained policies landing on top of the `no_hedge`
baseline and clearly behind the analytic `delta` baseline, with no visible
`risk_lambda` trend — training is stable now, just not learning a useful
hedge yet in the default 1000-episode budget.

- [x] Expose lr/alpha as train-hedge CLI flags
- [x] Add lightweight training diagnostics
- [ ] Diagnostic: train longer
  Retrain a single config (`risk_lambda=0.0`, all other defaults) for a much
  larger episode budget (e.g. 3000-5000 episodes), using the new
  diagnostics, to check whether the policy eventually separates from
  `no_hedge` and starts tracking delta. If this alone closes the gap, no
  hyperparameter tuning is needed.
- [ ] If still flat: targeted hyperparameter sweep
  Sweep `lr` (default `1e-4` vs. the standard SAC default `3e-4`), `alpha`
  (default `0.2` — with rewards scaled by strike, typically O(0.01-0.5), a
  fixed entropy coefficient of `0.2` may be large relative to the Q-value
  scale, which would explain `std` staying stuck around `0.83-0.86` instead
  of shrinking as training progresses), and update frequency
  (`update_every`/`grad_steps_per_update`) against the `delta` baseline.
  Skip `warmup_steps` — it only affects how much random data is collected
  before learning starts, not whether the actor converges once learning
  does start, so it's an unlikely bottleneck here.
- [ ] Re-run the risk_lambda sweep
  Once a training config reliably competes with/beats the `delta` baseline,
  re-run the `risk_lambda` sweep (0.0, 0.01, 0.05, 0.1, 0.5) with that config
  to get a meaningful read on the risk-shaping effect.
