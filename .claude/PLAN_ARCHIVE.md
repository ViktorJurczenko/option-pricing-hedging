# Plan archive

Finished (or superseded) workstreams from `.claude/PLAN.md`, newest first. Each is a summary of what was done and where the details are, not the step list: the commits, `.claude/BUGS.md` and `docs/results/` hold the detail.

Entry format:
```
## <Workstream> (yyyy-mm-dd -> yyyy-mm-dd, done | superseded)
- <main step or decision>
- ...
**Outcome:** <one line, with the measured result if any>.
**Links:** commits `abc1234`..., BUGS entries, docs/results, README section.
```

## Project reorganisation (2026-10-28 -> 2026-10-29, done)
- CLAUDE.md cut from 207 lines to a short agent guide; its module detail moved verbatim to `docs/architecture.md`, then rewritten for the new layout.
- PLAN.md turned into a roadmap (status table, active, backlog with "done so far"), finished plans summarised here, measured tables moved to `docs/results/`; BUGS.md given IDs, an index and area sections.
- All code moved into one `optionpricing/` package (`git mv`, CamelCase modules renamed), generated outputs to `artifacts/`, personal notes to `notes/`, `AUDIT.md` deleted, `pyproject.toml` added; README revised.
- Review rule: review-agent reuses the dispatcher's reported test runs instead of rerunning them.
- ruff configured and run once over the repo; hooks in `.claude/settings.json`: ruff + matching unit test after edits, a commit gate (message format, ruff, unit suite, mutation check when its inputs change), the roadmap at session start, a living-docs reminder at stop.

**Outcome:** 771 tests (674 unit + 97 integration) pass after the move and the formatting, as before; 14/14 mutants killed; each phase approved by review-agent. Moving `course/` into `notes/` was left to the user (it was the session's working directory).
**Links:** `9a218a1`, `5b7b041`, `3b004a8`, `e5a190a`, `30c8442`, `5a353d9`, `611da60`, `865933c`, plus the hooks commit; BUGS B-033, B-034.

## Rewrite REMINDER.md into up-to-date chapters (2026-10-28, done)
- REMINDER.md (gitignored notes) had grown as a log: commit-dated headers, "as built (step N)" sections, a duplicated summary, stale statements and no foundations.
- Restructured with the user into numbered chapters that each explain the current code: Part I foundations (GBM, Black-Scholes, LSM, the hedging env, SAC), Part II the surrogate, Part III hedging, and a legacy appendix ("is the MLP useful?").
- Measured evidence kept, process notes dropped (they are in BUGS.md); a backup of the old file is outside the repo.

**Outcome:** done, except chapter 11 (ablations), which moved to the active "Surrogate ablations and final model" workstream.
**Links:** `0325254`, `a3cd013` (code comments no longer cite REMINDER.md).

## Restructure: ML/ for everything learned, Hedging/ for the hedging problem (2026-10-22, done)
- ML code was spread over `AmericanOptionPricing/`, a top-level `datasets/` (shadowing the HuggingFace package) and `AmericanHedging/`.
- Chosen layout (option B of three): `ML/` for what is learned, `Hedging/` for the learner-independent finance problem; pricing folders unchanged. Moved with `git mv`, `Hedge.py` split, imports and string references rewritten.

**Outcome:** 427 unit and 91 integration tests passing after the move, as before.
**Links:** `21a2360`.

## Faster dataset generation (2026-10-22, done)
- A profile showed `LongstaffSchwartz.fit` at ~90% of a sample's time, mostly unused Greeks, SVD `lstsq` on 10k x 4 systems and scipy's generic normal cdf/pdf.
- Added a `compute_greeks` flag, normal-equations least squares in price-only mode (the hedging path keeps `lstsq`: the normal equations cost ~1e-5 in the Laguerre deltas) and `ndtr` in Black-Scholes. Labels unchanged.

**Outcome:** the run restarted with `--nb_jobs 10`.
**Links:** `6afd736`.

## Resumable dataset generation (2026-10-20, done)
- Two generation runs were lost to a kill.
- Seeded the parameter draw and each sample's paths (`--seed`); finished chunks are saved and skipped on a re-run (`--chunk_size`). Later shared with the tree generator through `ML/surrogate/chunks.py`.

**Outcome:** an interrupted run resumes, losing at most one chunk.
**Links:** `29e3602`; BUGS "Dataset generation loses all work when interrupted".

## Modularize the test helpers (2026-10-02, done)
- Pulled duplicated helpers into shared modules: a fake LSM factory, a state builder, CLI args built from the real parsers, one subprocess helper (`tests/helpers.py`; no `conftest.py`, as no fixture is shared).
- Split `test_hedge.py` by concern (`tests/unit/test_hedge_*.py`, fakes in `tests/unit/hedge_fakes.py`).

**Outcome:** a layout or CLI change is edited in one place.
**Links:** `f8aef8e`, `d90b14e`.

## Test suite cleanup (2026-10-02, done)
- A whole-suite review (419 passed, 2 skipped) found it thin where silent failures cost most, ~15 duplicate tests, and tests that would pass with broken code.
- Fixed `HedgingEnv` for options exercised at row 0, tested the SAC update math, tightened weak tests, added missing coverage, removed duplicates.

**Outcome:** a stricter suite; the bugs it surfaced are in BUGS.md.
**Links:** `04aa20f`, `91691cb`, `cc607e9`, `d53ee9b`, `1be10fd`, `d90b14e`; BUGS "HedgingEnv mishandles options exercised at row 0", "LongstaffSchwartz.calculate_option_price returns 0 before predict_payoff", "r2_fn returns nan/-inf for constant targets".

## ML surrogate: simple basis, calls and puts, regenerated LSM dataset (2026-10-02 -> 2026-10-20, superseded)
- Switched the LSM surrogate's labels to the simple basis, added `is_call` as a 5th input (one net for calls and puts, calls labelled with the control-variate LSM), mixed call/put generation, per-type `test-ml` errors, and stored the LSM settings in the dataset.
- The dataset regeneration, retrain and comparison were never run: the tree-teacher overhaul replaced the LSM surrogate, which is kept as `ML/surrogate/legacy/` without a CLI command.

**Outcome:** code done (APPROVE, 430/430 passing); the open steps were dropped with the switch to the tree teacher.
**Links:** `737dce8`, `8137fca`, `8411600`.

## Bring the README up to date (2026-10-02, done)
- Reran `price`, `test-ml` and pytest and refreshed their numbers; documented the control variate, the horizon fix, the simple-vs-Laguerre equivalence and the then-stale ML dataset; updated the usage table and the hedging architecture notes.

**Outcome:** README matched the code as of that date.
**Links:** `61d9222`, `e9cd156`.

## Learned no-trade band action mode (2026-10-01, done)
- `gap_fraction` can only hold still in the limit (tanh never reaches -1), so every "hold" still paid the fixed cost.
- Added `--action_mode band`: a 2-D action giving a state-dependent band half-width and a rebalancing target, so an exact hold covers a whole region of states and `delta_band` is a constant action. Vector actions in `train()` and `ReplayBuffer`.

**Outcome:** matches or beats the best fixed band at 1% and at 1% + $0.05 fixed.
**Links:** `59d2a5e`, `36dff7b`, `be73d51`; BUGS "RL policy under trading costs keeps trading every day; a simple band beats it".

## Hedge under trading costs (2026-10-01, done)
- Exposed `--prop_tc`, added the `delta_band` baseline (`--band`, `--band_target {delta,edge}`) and measured it over a cost grid (`docs/results/hedging-costs.md`).
- Decided with the user: proportional 1%, `residual_scale` 0.25, fixed-cost mechanism deferred. RL at 1% beat daily delta but kept trading daily and lost to the band.
- Added `--action_mode gap_fraction` (fraction of the gap to delta closed per step): a big gain, plateauing ~0.006-0.009 behind the band by episode ~1750 of 3000. A `risk_lambda` sweep and Deep Hedging were left for later.

**Outcome:** RL closes most of the gap to the band under costs (APPROVE, 351/351 passing); the band action above closed the rest.
**Links:** `b5fb02f`, `2541bcb`, `6e590d4`; `docs/results/hedging-costs.md`; README "Hedging under trading costs".

## Checkpoint the hedging policy on a fixed validation set (2026-10-01, done)
- `train()` picked the checkpoint by the trailing-50 mean raw reward, which mostly measures which paths were drawn.
- Validation on held-out paths (own seed, default `seed + 1000`, own LSM fit, built before the training simulation since `GBM` reseeds `np.random`), scored in terminal mode with the batched `HedgingEnv.rollout_pnl`, criterion P&L std (or `cvar`); the untrained policy is scored first. `deterministic_hedge` is shared by validation and `test-hedge`.
- Found and fixed "CVaR" being the VaR in `evaluate()`.

**Outcome:** puts beat the LSM delta hedge by 20-25% in P&L std, calls match it (APPROVE, 293/293 passing).
**Links:** `064822b`, `6351d86`; BUGS "RL hedging policy trails the LSM delta hedge", '"95% CVaR" in hedging evaluation is actually the 95% VaR'.

## Residual-on-delta hedging policy (2026-09-30, done)
- The plain RL policy tracked exact delta (corr 0.92-0.96) but sat 0.07-0.10 away from it, with ~2x the LSM delta hedge's P&L std.
- The policy now outputs a correction `hedge = clip(delta + c tanh(u), -1, 1)`; the critic sees the squashed residual; `c` (`residual_scale`, default 0.1) is a buffer saved in the checkpoint, old checkpoints load as 0.

**Outcome:** beats plain RL, still trailed LSM delta until validation checkpointing (APPROVE, 220/220 passing).
**Links:** `c4c34a6`.

## Fix LSM exercise decisions and Greeks (2026-09-30, done)
- Moneyness bins zeroed ITM delta and leaked look-ahead into exercise; the ITM-only fit extrapolated OTM Greeks; plain LSM's spurious call exercise was regression noise against a ~0.1 margin.
- Pathwise-delta regression for the Greeks (bins/ridge dropped), a European control variate in the continuation regression (calls only), the 364-day horizon fixed, reference-based tests; hedging policies retrained.

**Outcome:** LSM delta now matches the exact delta (APPROVE on each step, 196/196 passing).
**Links:** `4b04c0a`, `5b00207`, `f263fb5`, `0a9709f`, `c003285`; BUGS "LSM early-exercises American calls and mis-estimates delta", "Simulated horizon was one step short of the maturity".

## Fix the hedging objective so SAC has a reason to hedge (2026-09-29, done)
- Earlier SAC tuning (alpha, lr, 10x training, a cleaner delta feature) all plateaued on `no_hedge`: under the risk-neutral drift every hedge has ~the same expected P&L, the per-step `risk_lambda` penalty pushed toward not hedging, `gamma = 0.99` discounted the terminal payoff to ~2.6%, and the entropy term drowned the per-step signal.
- Accounting-P&L reward mode in `HedgingEnv`, SAC discount exposed (default 1.0), reward scaling and automatic entropy tuning, new `train-hedge` defaults.
- The follow-ups (LSM fix, residual policy, validation checkpointing) are their own entries above; Deep Hedging is in the backlog.

**Outcome:** beats the LSM delta hedge on 4 seeds out of sample, calls and puts (APPROVE, 152/152 passing).
**Links:** `3b26361`, `7b7cab4`, `040af43`, `9ebd989`, `437e6bd`, `df92cb5`, `1991ac7`; BUGS "RL hedging objective gives no incentive to hedge".
