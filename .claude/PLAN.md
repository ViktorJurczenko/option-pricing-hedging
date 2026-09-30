# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Support dividend-paying stocks

Everything assumes a non-dividend stock, which is why an American call
equals the European call (no early exercise) and why the call halves of the
hedging and ML work are closed-form. Dividends make the American call a
genuine American option. No order is set yet relative to the other entries.

- [ ] Explain the theory to the user before writing any code
  Required first; no implementation until the user has it. Cover:
  - Continuous dividend yield `q` vs discrete cash dividends on ex-dates,
    and which one to model.
  - The risk-neutral drift `r - q`.
  - Black-Scholes-Merton with `q`, and put-call parity with `q`.
  - Why dividends make early exercise of an American call optimal (just
    before ex-dividend dates, or deep ITM when `q > r`), and how the put
    changes.
  - The CRR tree probability with `r - q`.
  - What changes in LSM: the control variate stays valid, but it is no
    longer exact for calls.
  - Hedging P&L now includes the dividends earned on the stock position.
- [ ] Add `q` throughout
  `GBM`, `BlackScholes`, `BinomialTree`, `LongstaffSchwartz`,
  `option_value.py`, `HedgingEnv` (dividend income on the hedge
  position) and the CLI flags. Default `q = 0` must reproduce the current
  results exactly.
- [ ] Validate against the tree
  American call and put with `q > 0`: LSM price and exercise agreement vs
  CRR, deltas vs tree deltas. Check whether the call control variate still
  helps now that calls have an early-exercise premium.
- [ ] Revisit the ML surrogate with `q` as an input
  With `q > 0` the call half of the mixed dataset stops being plain
  Black-Scholes.
- [ ] Dispatch test-agent/review-agent
- [ ] Update README/CLAUDE.md

## Price calls and puts with one ML surrogate

The MLP has 4 inputs (`S0/K`, T, r, sigma) and is trained on a
single-option-type dataset (puts by default). A call query to a put model
silently returns a put price. Scheduled after the remaining hedging-fix
steps (retrain on the fixed LSM, residual-on-delta policy, validation-set
checkpointing) and before Deep Hedging.

- [ ] Add the option type as an input feature
  A 5th input `is_call` in {0, 1} (two classes, so one binary input is
  equivalent to a one-hot). Thread it through `MLP`,
  `AmericanOptionDataset`, the normalization stats, `smooth_intrinsic_value`
  (now per-row type), `evaluation.py` and `test-ml`, keeping training
  normalization and reconstruction consistent (see CLAUDE.md).
- [ ] Generate a mixed call/put dataset
  Price both types for each parameter combination on the same simulated
  paths (one GBM run, two labels). Label calls with the control-variate LSM,
  which is exact for an American call without dividends, and puts with
  plain LSM. This regeneration also carries the horizon fix, so the
  put-only dataset doesn't need regenerating separately.
- [ ] Retrain and compare
  The mixed model must not regress on puts versus a put-only model trained
  on the same regenerated puts, and must match BS on calls. Report ITM/ATM/OTM
  errors per option type in `test-ml`.
- [ ] Dispatch test-agent/review-agent
- [ ] Update README/CLAUDE.md

## Fix LSM exercise decisions and Greeks

Diagnosis in BUGS.md ("LSM early-exercises American calls and mis-estimates
delta"). The moneyness bins zero out ITM delta and cause look-ahead bias in
exercise, and the ITM-only value fit extrapolates OTM Greeks. Plain LSM's
spurious call exercise is regression noise against a ~0.1 margin. Both the
ML dataset (price only) and the RL environment (exercise times, delta,
gamma) depend on this. Prototypes in the scratchpad, 4 seeds, call + put,
back each step.

- [x] Pathwise-delta regression for the Greeks, drop bins/ridge (APPROVE, 167/167 passing)
- [x] European control variate in the continuation regression, calls only (APPROVE, 179/179 passing)
- [x] Fix the 364-day horizon (APPROVE, 190/190 passing)
- [x] Reference-based tests
- [x] Dispatch test-agent/review-agent (APPROVE on each step, 196/196 passing)
- [x] Retrain the hedging policies (LSM delta now matches exact delta; RL trails it - see BUGS.md)
- [ ] Regenerate the ML dataset (deferred)
  The old labels carry the horizon bias (up to -0.0012·K). Deferred by the
  user, likely to be folded into the mixed call/put dataset entry above.

## Fix the hedging objective so SAC has a reason to hedge

Analysis of why earlier SAC tuning attempts (alpha, lr, 10x longer
training, a cleaner delta feature) all plateaued on `no_hedge` (full write-up in BUGS.md, "RL hedging objective gives no
incentive to hedge"): the problem is the objective SAC is optimizing, not its
hyperparameters. Under the risk-neutral GBM drift, every hedge has ~the same
*expected* P&L (README table: delta -0.10 vs no-hedge -0.11 mean, only the std
differs), so an expected-reward maximizer is indifferent to hedging and
transaction costs tip it to not trading. `risk_lambda`'s per-step `r - λr²`
penalty actively pushes toward `no_hedge`, since the option liability only
appears in the terminal reward. `gamma=0.99` discounts that terminal payoff to
~2.6% from t=0, and per-step rewards (~1e-2, mean signal ~1e-4) are
drowned by the entropy term (`α·log π` ~ 0.2 per step).

- [x] Accounting-P&L reward in HedgingEnv
- [x] Expose SAC discount gamma, default 1.0 in train-hedge
- [x] Reward scaling + automatic entropy tuning
- [x] Wire new defaults into train-hedge, update docs
- [x] Dispatch test-agent/review-agent (APPROVE, 141/141 passing)
- [x] Validate: rerun training, check action_delta_corr and test-hedge metrics (beats LSM delta on 4 seeds, out-of-sample - see BUGS.md)
- [x] Address review-agent suggestions (APPROVE, 152/152 passing)
- [x] Validate on puts (beats LSM delta on 4 seeds, out-of-sample - see BUGS.md)
- [x] Fix LSM spurious early exercise / delta bias (own entry above)
- [ ] Residual-on-delta policy
  `action = clip(delta + c·tanh(output))`, so an untrained policy already
  starts at the delta baseline and only has to learn corrections (e.g.
  trading less under costs). Now needed: with the fixed LSM, the plain RL
  policy's std is 2x the delta baseline's (see BUGS.md).
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
