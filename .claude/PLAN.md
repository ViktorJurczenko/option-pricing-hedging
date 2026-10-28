# Roadmap

What is being worked on and what comes next. Finished workstreams move to `.claude/PLAN_ARCHIVE.md`; measured results go to `docs/results/`, problems to `.claude/BUGS.md`.

**How to use this file**
- One `###` entry per workstream, under **Active** or **Backlog**, and one row in the table below. Before starting new work, add its entry here first.
- An entry has a **Goal** (the problem and the chosen approach, in a paragraph), then its steps. Each open step is `- [ ] Title` with an indented line or two on what it does and why. A finished step becomes `- [x] Title`, and its description is dropped.
- A **Backlog** entry has a **Done so far** summary instead of ticked steps, then the **Next** steps.
- When every step is done, replace the entry with a short summary in `.claude/PLAN_ARCHIVE.md` and remove its table row.
- Every workstream also follows CLAUDE.md's done bar (tests, a `review-agent` pass, README/BUGS.md/REMINDER.md updated), so those steps are not repeated in each entry.

| Workstream | Status | Next step |
|---|---|---|
| [Project reorganisation](#project-reorganisation) | active | BUGS.md index and areas |
| [Surrogate ablations and final model](#surrogate-ablations-and-final-model) | active (runs pending) | CLI-only ablations, 3 seeds |
| [Deep Hedging](#deep-hedging) | backlog | Explain the theory to the user |
| [Dividend-paying stocks](#dividend-paying-stocks) | backlog | Explain the theory to the user |

## Active

### Project reorganisation

**Goal:** make the project easier to navigate. CLAUDE.md was 207 lines of module detail, PLAN.md and BUGS.md were unstructured logs, and the root mixed 9 code folders, generated outputs and personal notes. Decided with the user: a short agent guide, a roadmap plus an archive, an indexed BUGS.md, one `optionpricing/` package, then the README, ruff and hooks. Each phase is reviewed by `review-agent` and committed on its own.

- [x] Slim CLAUDE.md, detail to `docs/architecture.md`
- [x] PLAN.md as a roadmap, `.claude/PLAN_ARCHIVE.md`, results to `docs/results/`
- [ ] BUGS.md with an index and area sections
  IDs `B-001...` in chronological order, a fixed header block per entry (status, date found, files), entries grouped by area; titles unchanged so existing references keep working.
- [ ] One `optionpricing/` package
  `git mv` the code into `optionpricing/{core,pricing/european,pricing/american,ml/{surrogate,rl,experiments},hedging}`, rewrite imports, add `pyproject.toml`, move generated outputs under `artifacts/` and personal notes under `notes/`, delete `AUDIT.md`, rewrite `docs/architecture.md` to the new layout.
- [ ] README
  New paths and setup, a layout tree, a usage table pointing to `--help`, results linking to `docs/results/`, and the stale items fixed (test count, the `--reg_type` sentence, `HedgeEnv`, the tree-surrogate status).
- [ ] ruff
  Add it pinned, configure it in `pyproject.toml`, format and fix the whole repo in one `style:` commit, add the rule to CLAUDE.md.
- [ ] Hooks
  `.claude/settings.json` with Python hooks: ruff and the matching unit test after a `.py` edit; a commit gate (unit suite and message format); the roadmap printed at session start; a reminder at stop when code changed but no living doc did. Each hook unit-tested.

### Surrogate ablations and final model

**Goal:** choose the tree surrogate's final configuration from seeded evidence, then train and document it. The overhaul that built the surrogate is done (see **Done so far**); the defaults were picked from single-seed runs, so each one is now checked against the seed-to-seed spread.

**Done so far:** the old LSM-trained surrogate had structural problems: call labels were exactly Black-Scholes (the net re-learnt a closed form), the fixed-width softplus target let the kink back for the ~33% of rows with `sigma*sqrt(T) < 0.12` (43% of labels negative), 1.9% of LSM put labels sat below the European price, and its ReLU net gave a staircase delta and no gamma. So the MLP surrogate was rebuilt on a binomial-tree teacher (BBSR, 1000 steps, chosen by a convergence study), with z-sampled Sobol data (2^19 rows), an early-exercise-premium target, a smooth SiLU MLP with delta in the loss, the `PutSurrogate` pricer with domain routing and a payoff floor, a training loop with cosine decay and late patience, a new `test-ml` evaluation (independent finer reference tree, buckets, implied-vol error, no-arbitrage violations), and the seeded ablation runner with a cached reference grid. A learning curve showed that training steps, not data, were the limit, which set 400 epochs. Numbers: `docs/results/surrogate-training.md`. Commits `fcc60f1`..`aeee6d5`.

- [ ] CLI-only ablations, 3 seeds each (42, 43, 44)
  `python -m experiments.run_ablations`. Every variant is the default config with one change: the baseline (the seed-to-seed spread), `--lambda_delta` 0 / 0.3 / 3 (the price/delta trade-off the 800-epoch run showed), `--output softplus` (the labels' slightly negative premiums may cost it), `--activation relu` (the old net's staircase delta and zero gamma on the new layout), `--lbfgs_steps 200` (whether the full-batch finish pays), `--lambda_shape 0.1` (~3.4x the step time, ~95 min a run; hurt every region on a 4k-row trial). 18 runs plus 3 slow ones, ~3 h in batches of 4, plus scoring.
- [ ] Dataset ablations (need code first)
  `T` from 1 week vs from 0.25 (same row density), both scored on a `T >= 0.25` grid, to check the longer range costs the old region nothing: needs a `--T_min` flag on the generator and a restricted grid. All exercised rows vs a thinned share (e.g. 1/4): needs a training-time row filter. Plan each once the CLI-only results are in.
- [ ] Final model and write-up
  Retrain the winning config (3 seeds, keep the median) as `checkpoints/put_surrogate.pth`. Record the numbers in `docs/results/`, the README results section for the tree surrogate, and REMINDER.md (fill chapter 11's table from `runs/ablations/summary.md`, write the findings, update chapter 10's final-model paragraph).

## Backlog

### Deep Hedging

**Goal:** train a hedging policy by backpropagating a risk measure of the terminal P&L through whole simulated episodes (Buehler et al. 2019), as an alternative to SAC. P&L is linear in the positions, so the gradient is exact pathwise: no critics, entropy term, Bellman target or per-step reward shaping. SAC under costs is trained on a per-step mean-variance reward but judged on terminal CVaR (BUGS, "RL policy under trading costs keeps trading every day; a simple band beats it"); Deep Hedging optimizes the judged metric directly.

**Done so far:** nothing built. The motivation comes from the hedging-under-costs work (archived), where SAC needed the `band` action to match a fixed band. The theory to cover first is listed below.

**Next:**
- [ ] Explain the theory to the user before writing any code
  Required first: a deep, maths-first explanation (a document if the user wants one), walked through, covering:
  - discrete-time hedging: P&L `-Z + p_0 + Σ δ_k (S_{k+1} - S_k) - Σ c(δ_k - δ_{k-1}, S_k)`, why replication needs a complete frictionless continuous market, and why costs or discreteness turn hedging into risk minimization;
  - risk measures: monetary, convex, coherent; optimized certainty equivalents `ρ(X) = inf_w { w + E[ℓ(-X - w)] }`, CVaR as `ℓ(x) = max(x, 0)/(1-α)` with `w` learned jointly, the entropic measure and exponential utility, why mean-variance is not monotone;
  - indifference pricing `p(Z) = π(-Z) - π(0)`, its reduction to the risk-neutral price without costs, and the README's "mean P&L excludes the premium" caveat;
  - the network `δ_k = F_θ(I_k, δ_{k-1})` (why the previous position is an input under costs), the universal-approximation convergence result, one net with time as input vs one per step;
  - training: Monte-Carlo risk over a batch, backprop through the episode, subgradients of `|Δδ|`, CVaR gradient variance, batch size and learning rate;
  - comparison with SAC (known differentiable P&L vs learned critics) and with stochastic control (HJB, utility indifference, the Whalley-Wilmott band that `delta_band` approximates);
  - American options: hedging against a given exercise rule (LSM's, masked after `τ`) vs superhedging over all stopping times;
  - challenges (non-convexity, gradients over 365 steps, BPTT memory, non-smooth costs, model risk, overfitting to the simulator), pros and cons, applications and extensions.
- [ ] Decide the design with the user
  Risk measure (CVaR via the OCE `w`, entropic, std), inputs (`HedgingEnv`'s 5-feature state?), action parametrization (direct, residual on delta, or `band`), one network vs per step, exercise handling.
- [ ] Differentiable batched rollout in PyTorch
  A torch version of `HedgingEnv.rollout_pnl`'s terminal-mode arithmetic (costs, payoff at exercise, masking after it). It duplicates `step()`'s P&L, so a test must check they agree exactly, fixed cost and early exercise included.
- [ ] Training loop and CLI
  A new module under the RL package and a `train-deep-hedge` command, reusing validation checkpointing on held-out paths; evaluated through `evaluate()` via `test-hedge`.
- [ ] Compare against SAC and the baselines
  The README protocol (2000 seed-999 episodes, 4 training seeds, call and put): 0.1% cost against delta and SAC; 1% and 1% + $0.05 fixed against the best fixed band and the learned band.

### Dividend-paying stocks

**Goal:** support a dividend yield `q`. Everything assumes a non-dividend stock, which is why an American call equals the European one and why the call halves of the hedging and ML work are closed-form. Dividends make the American call a genuine American option.

**Done so far:** nothing built. A design for the surrogate is sketched: instead of a second network for calls, use the put-call symmetry of American options under GBM (McDonald & Schroder 1998), `C(S, K, r, q, sigma, T) = P(K, S, q, r, sigma, T)`, i.e. per unit of strike `C / K = m * p(1 / m; rate = q, yield = r)`, so one put dataset and one put network price both.

**Next:**
- [ ] Explain the theory to the user before writing any code
  Continuous yield vs discrete cash dividends; the drift `r - q`; Black-Scholes-Merton and put-call parity with `q`; why dividends make early call exercise optimal (before ex-dates, or deep ITM when `q > r`) and how the put changes; the CRR probability with `r - q`; LSM's control variate stays valid but is no longer exact for calls; dividend income in the hedging P&L.
- [ ] Add `q` throughout
  `GBM`, Black-Scholes, the binomial trees, LSM, `option_value.py`, `HedgingEnv` (dividends on the hedge position) and the CLI. `q = 0` must reproduce current results exactly.
- [ ] Validate against the tree
  American call and put with `q > 0`: LSM price and exercise agreement vs CRR, deltas vs tree deltas; whether the call control variate still helps.
- [ ] Check the put-call symmetry on the tree
  American call vs put with `(S, K)` and `(r, q)` swapped, over the domain, to ~1e-12, before building on it.
- [ ] Surrogate over `(m, T, r, q, sigma)`
  A call query becomes a put at `1 / m` (`z -> -z`) with `r` and `q` swapped, so the domain needs identical `r` and `q` ranges and a symmetric `z` range (widen [-3.5, 3] to [-3.5, 3.5]; the `m` cap (0.4, 2.5) already is). Premium target on Black-Scholes-Merton.
- [ ] Price helper and evaluation for calls
  Map a call to the symmetric put and back (delta and gamma through the chain rule); `test-ml` scores calls against a call tree with `q`, per `|z|` bucket, plus `q = 0`, where the call must stay Black-Scholes.
