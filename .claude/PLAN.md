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

- [ ] Expose lr/alpha as train-hedge CLI flags
  `train()`'s `lr` (default `1e-4`) and `alpha` (entropy coefficient,
  default `0.2`) are hardcoded keyword defaults, not CLI flags — unlike
  `risk_lambda`/`fixed_cost`/`grad_steps_per_update`, there's currently no
  way to override them without editing source. Needed before either can be
  swept from the CLI, the same way the sweep script already does for
  `risk_lambda`.
- [ ] Add lightweight training diagnostics
  The aggregate P&L test-hedge reports can't distinguish "policy ignores
  the delta signal entirely" from "policy tracks it but poorly calibrated."
  Add a small standalone diagnostic (not permanent library code) that logs,
  over the course of a training run: the actor's `std` trend (is entropy
  actually shrinking as training progresses, or stuck?) and the correlation
  between the policy's chosen action and the LSM's own delta at each state.
  Cheaper to run and more informative than judging purely from the final
  mean reward curve, and directly explains *why* a run is or isn't working
  before committing to a long diagnostic run or a hyperparameter sweep.
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
