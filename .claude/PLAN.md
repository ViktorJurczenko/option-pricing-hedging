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
| [Surrogate ablations and final model](#surrogate-ablations-and-final-model) | active | Wider network and more seeds: code |
| [README reorganization](#readme-reorganization) | backlog | Move the legacy surrogate out of the README |
| [Deep Hedging](#deep-hedging) | backlog | Explain the theory to the user |
| [Dividend-paying stocks](#dividend-paying-stocks) | backlog | Explain the theory to the user |
| [Benchmark against fast American-put methods](#benchmark-against-fast-american-put-methods) | backlog | Explain the theory to the user |

## Active

### Surrogate ablations and final model

**Goal:** choose the tree surrogate's final configuration from seeded evidence, then train and document it. The overhaul that built the surrogate is done (see **Done so far**); the defaults were picked from single-seed runs, so each one is now checked against the seed-to-seed spread.

**Done so far:** the old LSM-trained surrogate had structural problems: call labels were exactly Black-Scholes (the net re-learnt a closed form), the fixed-width softplus target let the kink back for the ~33% of rows with `sigma*sqrt(T) < 0.12` (43% of labels negative), 1.9% of LSM put labels sat below the European price, and its ReLU net gave a staircase delta and no gamma. So the MLP surrogate was rebuilt on a binomial-tree teacher (BBSR, 1000 steps, chosen by a convergence study), with z-sampled Sobol data (2^19 rows), an early-exercise-premium target, a smooth SiLU MLP with delta in the loss, the `PutSurrogate` pricer with domain routing and a payoff floor, a training loop with cosine decay and late patience, a new `test-ml` evaluation (independent finer reference tree, buckets, implied-vol error, no-arbitrage violations), and the seeded ablation runner with a cached reference grid. A learning curve showed that training steps, not data, were the limit, which set 400 epochs. Numbers: `docs/results/surrogate-training.md`. Commits `fcc60f1`..`aeee6d5`.

- [x] CLI-only ablations, 3 seeds each (42, 43, 44)
- [x] Combined variant, 3 seeds
- [x] Dataset ablations: code
- [x] Dataset ablations: runs
- [ ] Wider network and more seeds: code
  Add a one-change `width_256` variant (`{"width": 256}`) and the `COMBINED` entry `lambda_delta_0.3_softplus_width_256` (parts `lambda_delta_0.3`, `softplus_output`, `width_256`). Rank wider networks first in `run_cost`: add a term of 2 when the config's width is above `TrainConfig`'s default width (compared with the default, not a hardcoded 128), since a width-256 step costs ~2.5x a width-128 one, more than the L-BFGS finish (1) and close to the shape penalty (2); otherwise the width-256 runs tie with the width-128 ones at 0 and can end up alone at the end of a run. Tests in `tests/unit/test_run_ablations.py`: the combined variant changes exactly its three parts' settings (the existing parametrized test covers it once the entry exists), and the cost ranking puts the wider variant first.
- [ ] Wider network and more seeds: runs
  `python -m optionpricing.ml.experiments.run_ablations --variants lambda_delta_0.3_softplus lambda_delta_0.3_softplus_width_256 --seeds 42 43 44 45 46` (full grid, default `--out_dir`: the three finished width-128 seeds are skipped). That is 2 new width-128 runs (~25 min each) and 5 width-256 runs (~60 min each: 31.4 ms per training step against 12.5 ms at width 128, `docs/results/surrogate-training.md`, "Training cost by network width", measured on one process, so 4 concurrent runs may shift the ratio), about 2-2.5 h at 4 runs at a time. Scored on the full grid only, as decided (no `T >= 0.25` grid); the results are the two rows of `artifacts/runs/ablations/summary.md`. Questions: does a 4x larger network (199k weights) lower the error, above all beyond 3 months (the full grid's `T >= 3 months` bucket, where the T-range ablation found the wide range costs accuracy) and on the seed-dependent worst point; and, with 5 seeds, how wide the chosen configuration's worst-point spread really is. Width 256 is adopted only if it beats width 128 by more than ~2 seed sd on the near-the-money error, the implied-vol error or the worst point, weighed against ~2.5x the training time and a slower batch price (the timing column). Caveat: 400 epochs were chosen at width 128; a wider network might want another length. Afterwards, the chosen configuration's README numbers (3 seeds today) are refreshed with its 5 seeds, whichever width wins.
- [ ] Final model and write-up
  The configuration is chosen (`lambda_delta_0.3_softplus`, every exercised row, `T` from 1 week to 5 years; width 128 unless the width ablation says otherwise) and its results are in the README. Make it `TrainConfig`'s default, train it (3 seeds, keep the median) as `artifacts/checkpoints/put_surrogate.pth`, check that `test-ml` reproduces the README numbers, and update REMINDER.md (fill chapter 11's table from `artifacts/runs/ablations/summary.md`, write the findings, update chapter 10's final-model paragraph).

## Backlog

### README reorganization

**Goal:** make the README a document about the project as it is now: what it does, how to use it, the current models' results and the choices behind them. The history (the legacy LSM-trained surrogate, reverted fixes, superseded numbers) moves to `docs/results/` and `.claude/BUGS.md`, where the README links to it. Done after the final model, so the surrogate section is rewritten once.

**Done so far:** nothing. The tree surrogate's results and choices are in the README ("MLP surrogate"); the legacy surrogate's section ("Legacy MLP surrogate vs. Longstaff-Schwartz") still follows it.

**Next:**
- [ ] Move the legacy surrogate out of the README
  Its section, as it is, into `docs/results/legacy-surrogate.md` (with its commands and its "not re-verified" status); the README keeps a one-line pointer.
- [ ] Trim every README section to the current state
  Keep each section's current result and the choice it supports; move incident histories (e.g. "First fix, later reverted" in the LSM Greeks section) to their `.claude/BUGS.md` entries or `docs/results/`, linking to them. Check each remaining number against its command, and mark the ones not rerun "not re-verified".
- [ ] Update the links
  `docs/architecture.md`, the results docs and `CLAUDE.md` pointers to moved sections.

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

### Benchmark against fast American-put methods

**Goal:** find out whether the MLP surrogate is worth it against the classic fast pricers of American puts: analytic approximations (Barone-Adesi-Whaley, Ju-Zhong) and the integral-equation method of Andersen-Lake-Offengeim, compared on accuracy against the reference tree and on time per option, single and in batch.

**Done so far:** nothing. Kept for later: it needs theory the user does not have yet.

**Next:**
- [ ] Explain the theory to the user before writing any code
  The early-exercise boundary and the integral equation it satisfies, the quadratic approximation behind Barone-Adesi-Whaley and Ju-Zhong, and the fixed-point scheme of Andersen-Lake-Offengeim.
- [ ] Decide with the user which method to implement and how to compare it

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
