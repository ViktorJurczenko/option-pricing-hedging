# PLAN.md

Running log of plans for issues/features tackled in this project, newest
entry on top. See `.claude/CLAUDE.md`'s "Planning" section for how this file
is used: each step starts as `- [ ] Title` with a description, and collapses
to `- [x] Title` (description dropped) once done.

## Bring the README up to date

- [x] Rerun `price`, `test-ml` and pytest; refresh their numbers
- [x] Document the control variate, horizon fix, simple-vs-Laguerre equivalence and stale ML dataset
- [x] Update the usage table and `AmericanHedging/` architecture notes

## Learned no-trade band action mode (`--action_mode band`)

`gap_fraction` can only hold still in the limit `w -> 0` (tanh never reaches
-1), so a sampled or even deterministic "hold" is a tiny trade that still
pays `fixed_tc`. A learned, state-dependent no-trade band makes an exact hold
the outcome for a whole region of states, and contains the `delta_band`
baseline (the bar to beat under costs) as a constant action.

- [x] Add `band` mode to `RLPolicy`
- [x] Support vector actions in `train()` and `ReplayBuffer`
- [x] Wire the CLI
- [x] Tests and docs
- [x] Compare against `gap_fraction` and the best `delta_band` (matches/beats the band at 1% and at 1% + $0.05 fixed - see BUGS.md)

## Hedge under trading costs (where delta hedging overtrades)

With only a 0.1% proportional cost and an accurate LSM delta, daily delta
hedging is close to optimal: the call policy learned nothing beyond it (see
BUGS.md, "RL hedging policy trails the LSM delta hedge"). With larger costs,
rebalancing to delta every day spends a lot on small trades, so a policy
that trades less should do better. That's the setting where learned hedging
is supposed to pay off.

Two issues shape the plan:
- **Fair baseline.** The classical answer to costs is a no-trade band:
  rebalance to delta only when the position has drifted more than `b` away
  from it. RL has to beat that, not just daily delta.
- **Fixed costs can't currently be avoided by the policy.** HedgingEnv
  charges the fixed cost on any trade above 1e-8. A continuous Gaussian
  policy never outputs exactly its current position, so it would pay the
  fixed cost every step whatever it learns. Proportional costs don't have
  this problem. A fixed-cost regime therefore needs a "no trade" mechanism
  first. That's a design decision for the user, taken after the baseline
  numbers are in.

- [x] Expose the proportional cost as `--prop_tc`
- [x] Add a banded-delta baseline (`--policy delta_band --band b --band_target {delta,edge}`)
- [x] Measure baselines over a cost grid
  95% CVaR (×K) on the 2000 seed-999 episodes. Each band column gives the
  best band and its target:

  | Regime | Call: delta | Call: best band | Put: delta | Put: best band |
  |---|---|---|---|---|
  | tc 0.1% | -0.112 | -0.110 (0.02 edge) | -0.110 | -0.105 (0.05 edge) |
  | tc 0.5% | -0.148 | -0.123 (0.05 edge) | -0.152 | -0.119 (0.05 edge) |
  | tc 1% | -0.198 | -0.136 (0.1 edge) | -0.208 | -0.130 (0.1 edge) |
  | tc 2% | -0.302 | -0.156 (0.1 edge) | -0.322 | -0.151 (0.1 edge) |
  | fixed $0.01 | -0.149 | -0.116 (0.05 delta) | -0.145 | -0.112 (0.1 delta) |
  | fixed $0.05 | -0.295 | -0.126 (0.1 delta) | -0.290 | -0.125 (0.1 delta) |

  - At today's 0.1% cost a band barely helps, which matches RL learning
    nothing for calls.
  - From 0.5% upward, daily delta is badly beaten. At 2% its put CVaR
    (-0.322) is worse than not hedging at all (-0.186).
  - As expected, band-edge wins under proportional costs and
    rebalance-to-delta wins under fixed costs, and the best band widens
    with cost (0.05-0.1).
  - That width matters for the residual policy: with `residual_scale` 0.1
    it can only stay 0.1 from delta by saturating tanh, so cost regimes
    need a wider residual (0.2-0.3).
- [x] Decide with the user (proportional 1%, residual_scale 0.25, fixed-cost mechanism deferred)
- [x] Train and validate RL at proportional 1% (beats daily delta, loses to the band: keeps trading daily - see BUGS.md)
- [x] Decide with the user: action parametrization that can hold still (fraction of gap)
- [x] Add `action_mode="gap_fraction"` to RLPolicy (`--action_mode`)
- [x] Retrain at 1% with `--action_mode gap_fraction` (big gain, still behind the band; still improving at episode 1000 - see BUGS.md)
- [x] Retrain with 3000 episodes (plateaus by episode ~1750, still 0.006-0.009 behind the band - see BUGS.md)
- [x] Decide with the user: next lever (stop here and commit; risk_lambda sweep and Deep Hedging left for later)
- [x] Dispatch test-agent/review-agent (APPROVE, 351/351 passing)
- [x] Update README/CLAUDE.md, record results in BUGS.md

## Checkpoint the hedging policy on a fixed validation set

`train()` saves the policy with the best trailing-50 mean *raw* reward,
which mostly measures which paths were drawn, not hedge quality, and never
considers the untrained policy (≈ delta under the residual form). See
BUGS.md, "RL hedging policy trails the LSM delta hedge".

Design choices:
- **Validation set:** paths simulated with their own seed (default
  `seed + 1000`, which never hits the seed 999 used for evaluation), with
  their own LSM fit, as `test-hedge` does. The first `val_paths` (1000)
  paths are each run once, in fixed order, in terminal mode. The metric is
  then exactly `test-hedge`'s P&L. The validation env is built before the
  training simulation, because `GBM` reseeds the global `np.random` and the
  training stream should stay the one `--seed` gives.
- **Batched rollout:** a sequential rollout takes ~60 ms per episode, so
  1000 paths every 50 episodes would roughly double training time.
  `HedgingEnv.rollout_pnl(hedge_fn, path_idxs)` steps all paths together,
  with one batched policy call per time step. It duplicates `step()`'s
  terminal P&L arithmetic, so a test must check that it matches the
  per-step loop exactly, fixed cost and early exercise included.
- **Criterion:** P&L std by default (the hedging error); `cvar` is an
  option. The untrained policy is scored and saved before training starts,
  then the policy is re-scored every `val_every` (50) episodes and after the
  last one. `--val_paths 0` restores the old trailing-50 criterion.
- **Shared deterministic action:** `deterministic_hedge(policy, states)` in
  `Hedge.py`, used by both validation and `test-hedge`, so the two can't
  drift apart.

- [x] Batched terminal-P&L rollout in HedgingEnv
- [x] Validation checkpointing in train()
- [x] Wire into train-hedge and test-hedge
- [x] Dispatch test-agent/review-agent (APPROVE, 270/270 passing)
- [x] Validate (puts beat LSM delta by 20-25% std, calls match it - see BUGS.md)
- [x] Update README/CLAUDE.md
- [x] Fix "CVaR" being the VaR in evaluate() and val_metric (see BUGS.md)
- [x] Address review-agent suggestions (APPROVE, 293/293 passing)

## Residual-on-delta hedging policy

With the fixed LSM, the plain RL policy's P&L std is ~2x the LSM delta
hedge's (BUGS.md, "LSM early-exercises American calls..."): it tracks the
exact delta (corr 0.92-0.96) but sits 0.07-0.10 away from it. Learning the
whole hedge ratio from scratch wastes the policy's capacity on recovering a
delta the state already contains. Instead the policy outputs a correction:
`hedge = clip(delta + c·tanh(u), -1, 1)`, so an untrained policy (u ~ 0)
already hedges at delta and only has to learn deviations (e.g. trading less
under costs).

Design choices:
- The critic sees the squashed residual `tanh(u)`, not the final hedge
  ratio. Delta is in the state, so no information is lost, and `_squash`'s
  log-prob/entropy machinery stays unchanged. Warmup's uniform random
  actions become uniform residuals, i.e. delta ± c.
- `c` is a `residual_scale` buffer on `RLPolicy`, so it is saved in the
  checkpoint and `test-hedge` cannot use a value that differs from training.
  `0` disables it (the policy outputs the hedge ratio directly, as before).
  Old checkpoints without the key load as `0`.
- Default `c = 0.1` in `train-hedge`, the scale of the RL/exact-delta gap.

- [x] Residual mapping in RLPolicy
- [x] Use it in train() and test-hedge
- [x] Dispatch test-agent/review-agent (APPROVE, 220/220 passing)
- [x] Validate (beats plain RL, still trails LSM delta - see BUGS.md)
- [x] Update README/CLAUDE.md

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
- [x] Residual-on-delta policy (own entry above)
- [x] Checkpoint on a fixed validation path set (own entry above)
- [ ] Later: Deep Hedging (Buehler et al. 2019) as an alternative to SAC
  Hedge P&L is linear in the positions, so a policy network can be trained
  by direct backprop through whole simulated episodes, minimizing
  CVaR/variance of *terminal* P&L over batches of paths - exactly the
  metric `test-hedge` reports, no critics/entropy/credit assignment. New
  module (e.g. `AmericanHedging/DeepHedging.py`) + `train-deep-hedge`
  subcommand, evaluated via the same `evaluate()` so it's directly
  comparable to `rl`/`delta`/`no_hedge`. For the American put, the episode
  still ends at the LSM exercise time per path (masking steps after it).
