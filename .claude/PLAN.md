# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Fix the hedging objective so SAC has a reason to hedge

Analysis of why the five diagnostics in the entry below all plateaued on
`no_hedge` (full write-up in BUGS.md, "RL hedging objective gives no
incentive to hedge"): the problem is the objective SAC is optimizing, not its
hyperparameters. Under the risk-neutral GBM drift, every hedge has ~the same
*expected* P&L (README table: delta -0.10 vs no-hedge -0.11 mean, only the std
differs), so an expected-reward maximizer is indifferent to hedging and
transaction costs tip it to not trading. `risk_lambda`'s per-step `r - λr²`
penalty actively pushes toward `no_hedge`, since the option liability only
appears in the terminal reward. `gamma=0.99` discounts that terminal payoff to
~2.6% from t=0, and per-step rewards (~1e-2, mean signal ~1e-4) are
drowned by the entropy term (`α·log π` ~ 0.2 per step). This supersedes the
"Targeted hyperparameter sweep" step below as the next thing to try.

- [x] Accounting-P&L reward in HedgingEnv
- [x] Expose SAC discount gamma, default 1.0 in train-hedge
- [x] Reward scaling + automatic entropy tuning
- [x] Wire new defaults into train-hedge, update docs
- [x] Dispatch test-agent/review-agent (APPROVE, 141/141 passing)
- [x] Validate: rerun training, check action_delta_corr and test-hedge metrics (beats LSM delta on 4 seeds, out-of-sample - see BUGS.md)
- [x] Address review-agent suggestions (APPROVE, 152/152 passing)
- [ ] Fix LSM spurious early exercise / delta bias (see BUGS.md)
  Found during sanity checks: LSM early-exercises 57% of American-call
  paths (should be 0%) and its delta is far from exact BS delta. The delta
  is what matters for hedging (it explains the ~3x std gap to an exact
  delta hedge); the exercise timing was measured to have little effect,
  since spurious exercises cluster near maturity. Decide
  between fixing LSM (feature standardization before Ridge, reference-based
  validation) and sourcing the hedging state's delta/exercise boundary from
  `option_value.py`'s exact/tree value functions instead.
- [ ] Residual-on-delta policy (only if the above is still flat)
  `action = clip(delta + c·tanh(output))`, so an untrained policy already
  starts at the delta baseline and only has to learn corrections (e.g.
  trading less under costs).
- [ ] Checkpoint on a fixed validation path set
  Replace "best trailing-50 raw mean reward" (dominated by which paths were
  drawn) with std/CVaR of P&L on a fixed held-out set of paths.
- [ ] Later: Deep Hedging (Buehler et al. 2019) as an alternative to SAC
  Hedge P&L is linear in the positions, so a policy network can be trained
  by direct backprop through whole simulated episodes, minimizing
  CVaR/variance of *terminal* P&L over batches of paths - exactly the
  metric `test-hedge` reports, no critics/entropy/credit assignment. New
  module (e.g. `AmericanHedging/DeepHedging.py`) + `train-deep-hedge`
  subcommand, evaluated via the same `evaluate()` so it's directly
  comparable to `rl`/`delta`/`no_hedge`. For the American put, the episode
  still ends at the LSM exercise time per path (masking steps after it).

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
- [x] Retrain with Greeks-fixed defaults and refresh README (still flat - see below)
- [x] Diagnostic: train longer with Greeks-fixed defaults (still flat - see below)
- [ ] Targeted hyperparameter sweep
  Five independent diagnostics now show the same flat plateau (`alpha`,
  `lr`, 10x training duration, a clean/reliable delta feature via the
  Greeks fix, and duration+Greeks combined - the 5000-episode Greeks-fixed
  run's `mean_std` dropped from ~1.04 to ~0.86 within the first 250
  episodes and never moved again through all 5000, `action_delta_corr`
  stayed pure noise the whole time, -0.83 to +0.45 with no trend) - none of
  them explain it, so the RL policy landing on top of `no_hedge` instead of
  above it is a genuine open question in the SAC setup itself, not noise
  from any of those five factors. Remaining candidates:
  `update_every`/`grad_steps_per_update` and combinations of the above,
  plus reward shaping / network capacity if those don't help either. Skip
  `warmup_steps` - it only affects how much random data is collected
  before learning starts, not whether the actor converges once learning
  does start, so it's an unlikely bottleneck here.
- [ ] Re-run the risk_lambda sweep
  Once a training config reliably competes with/beats the `delta` baseline,
  re-run the `risk_lambda` sweep (0.0, 0.01, 0.05, 0.1, 0.5) with that config
  to get a meaningful read on the risk-shaping effect.
