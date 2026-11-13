# Roadmap

What is being worked on and what comes next. Finished workstreams move to `.claude/PLAN_ARCHIVE.md`; measured results go to `docs/results/`, problems to `.claude/BUGS.md`.

**How to use this file**
- One `###` entry per workstream, under **Active** or **Backlog**, and one row in the table below. Before starting new work, add its entry here first.
- An entry has a **Goal** (the problem and the chosen approach, in a paragraph), then its steps. Each open step is `- [ ] Title` with an indented line or two on what it does and why. A finished step becomes `- [x] Title`, and its description is dropped.
- A **Backlog** entry has a **Done so far** summary instead of ticked steps, then the **Next** steps.
- When every step is done, replace the entry with a short summary in `.claude/PLAN_ARCHIVE.md` and remove its table row.
- Every workstream also follows CLAUDE.md's done bar (tests, a `review-agent` pass, the living docs updated), so those steps are not repeated in each entry.

| Workstream | Status | Next step |
|---|---|---|
| [Hedging rebuild](#hedging-rebuild) | active | Decide the remaining design with the user |
| [Dividend-paying stocks](#dividend-paying-stocks) | backlog | Explain the theory to the user |
| [Benchmark against fast American-put methods](#benchmark-against-fast-american-put-methods) | backlog | Explain the theory to the user |
| [Longstaff-Schwartz](#longstaff-schwartz) | backlog | Check the Laguerre basis against the paper |

## Active

### Hedging rebuild

**Goal:** rebuild the American-put hedger from the ground up as Deep Hedging (Buehler et al. 2019), on a correct P&L, with `PutSurrogate` as the hedger's pricing model. The current stack has two accounting bugs that distort every result so far. Hedges earn one step late (BUGS, "HedgingEnv applies each hedge one step late") and the P&L has no cash account (BUGS, "Hedging P&L has no cash account, so the delta baseline hedges the wrong liability"). It also takes its exercise dates and deltas from LSM. Its exercise rule is fitted on the same paths it is tested on, and its deltas are Monte-Carlo estimates. Their earlier failure near maturity is fixed (BUGS, "LSM delta/gamma estimates unreliable near maturity"). The rebuild has six parts:
- **P&L.** One differentiable torch function books it, discounted, with the premium received at `t_0`, cash at `r`, `δ_k` chosen at `t_k` earning `δ_k (S̃_{k+1} - S̃_k)`, and costs on every trade. It takes any policy with the same interface (inputs in, position out), so training, evaluation and every baseline go through this one function. The B-037 lag survived because the accounting existed twice (`HedgingEnv.step` and `rollout_pnl`). Here it exists once.
- **Building blocks, one file each.** Risk measures (CVaR with `α` as a parameter, entropic, variance), transaction costs (proportional plus fixed), and settlement (one function for cash, one for physical). Training and evaluation use the same risk functions, so "trained on CVaR" and "scored on CVaR" are the same number. The fixed fee's indicator `1[x ≠ 0]` has no gradient, so it is straight-through: the forward pass charges the exact fee and the backward pass uses the slope of `s(x) = 1 - e^{-|x|/κ}`. The P&L stays exact and only the gradient is approximated.
- **Exercise.** The holder exercises the first time `S_t ≤ S*(t)`, with the boundary `S*` read from the tree. Per unit of strike the boundary depends only on the time left, `r` and `σ`, so one tree per contract gives it for every day of the path. The surrogate is not used for it. The price meets the payoff tangentially (smooth pasting), so a 1e-4 price error moves the boundary by about 0.5% of `K`. How `τ` is found is a choice (`tree`, `lsm`, `never`, `late`), so suboptimal holders are a test scenario.
- **Pricing model.** `PutSurrogate` gives the price, delta and gamma. This is where it earns its place. Contracts (`T`, `σ`, `r`, `S_0/K`) are drawn per path, so one policy covers a family of contracts. Pricing a different contract at every (path, step) is 3 ms per option on the tree and microseconds in batches on the surrogate.
- **Policy.** One network with time as an input outputs a no-transaction band around the surrogate delta, `δ_k = clamp(δ_{k-1}, Δ_k + l_k, Δ_k + u_k)` (Imaki et al. 2021). It holds exactly inside the band and starts as the delta hedge. Its inputs are `ln(S_k/K)`, `T - t_k`, `σ`, `r`, `Δ_k`, `Γ_k` and `δ_{k-1}`. An optional 8th input, the distance to the boundary `ln(S_k / S*(T - t_k))`, is switched on by a flag, to measure whether the network learns from it.
- **Objective.** A risk measure of the terminal P&L is minimized by backpropagating through whole paths. The choices are CVaR through the Rockafellar-Uryasev variable `w` (with its own learning rate), the entropic risk, or the variance. There is no critic, no entropy bonus and no per-step reward shaping, so the objective is the judged metric.

The pipeline mirrors the tree surrogate's: generate a dataset, train, test. The dataset stores only what does not depend on training choices: paths, contracts, premiums, surrogate delta and gamma, the boundary and `τ`, and the metadata that produced them. Rebalancing frequency, costs (with `κ`), settlement, risk measure and network settings are chosen at train or test time. So one dataset serves many experiments, and a policy can be tested at other costs than it was trained on.

**Layout.** The non-ML parts go in `optionpricing/hedging/` (`risk.py`, `costs.py`, `settlement.py`, `pnl.py`, `baselines.py`, `simulator.py`, `exercise.py`). The network pipeline goes in `optionpricing/ml/hedging/` (`generate_hedge_dataset.py`, `hedge_dataset.py`, `hedge_model.py`, `hedge_training.py`, `hedge_evaluation.py`). SAC and `HedgingEnv` move to `legacy/` packages, as the LSM-trained surrogate did. `train-hedge` and `test-hedge` become the Deep Hedging commands. The old ones stay as `train-hedge-legacy` and `test-hedge-legacy`. The choices and their reasons are recorded in `docs/results/hedging-training.md`.

**Open, flagged.** With weekly hedging the holder can still exercise between two hedge dates. Two things are not settled yet. Does the holder check the boundary daily or only on hedge dates? And how does the P&L book a `τ` that falls between hedge dates?

**Next:**
- [x] Explain the theory to the user before writing any code
- [ ] Decide the remaining design with the user
  The contract family's ranges (within the surrogate's domain), the default rebalancing grid (daily, or weekly for fast experiments), the flagged exercise question above, and whether SAC is retrained once on the new P&L for one comparison.
- [ ] Move the SAC stack to legacy
  `env.py`, `evaluation.py`, `baselines.py` and `option_value.py` go to `optionpricing/hedging/legacy/`, and `optionpricing/ml/rl/` goes to `optionpricing/ml/hedging/legacy/`, with their tests. The CLI commands become `train-hedge-legacy` and `test-hedge-legacy` and must reproduce the current results unchanged. Done first, so the new modules get the plain names.
- [ ] Exercise boundary from the tree
  A new induction function replaces `_induct` in `batch_binomial_tree.py`. It does the same backward induction and also records, at every level, the highest node where exercising beats holding. That node is interpolated with the first continuation node, to smooth out the node spacing (about 0.6% of `K` at 1000 steps, `T = 1`). A public function returns the boundary `S*/K` over time per contract, and the dataset generation uses it. `binomial_put_batch` keeps its outputs bit for bit (regression test). Under `bbsr` the boundary comes from the finer tree, since Richardson extrapolation applies to prices, not to a location. Tests: against the boundary found by bisection on price = payoff (S* from 81.11 at 1 year to 94.52 at 1 week for the default contract), rising toward `K` as maturity nears, and absent for a European put.
- [ ] Risk measures
  `risk.py` holds four functions. `cvar(pnl, alpha)` is the average of the worst `1 - α` share, the reported metric. `cvar_loss(pnl, w, alpha)` is the Rockafellar-Uryasev training form. `entropic(pnl, lam)` uses `logsumexp`, so `e^{-λ PL}` cannot overflow. `variance(pnl)` completes the set. Tested on the 20-outcome example: the minimum of `cvar_loss` over `w` equals `cvar`.
- [ ] Transaction costs
  `costs.py`: `transaction_cost(trade, spot, proportional, fixed, kappa)` = `ε |x| S + f 1[x ≠ 0]`, with the fee straight-through (`hard + (soft - soft.detach())`). Tests: hand values, a zero trade costs nothing, the forward value equals the exact fee, and the gradient equals `ε S sign(x) + f sign(x) e^{-|x|/κ} / κ` (0 at `x = 0`).
- [ ] Settlement
  `settlement.py`: `cash_settlement` pays `K - S` and buys back `-δ` shares at cost. `physical_settlement` pays `K`, receives one share and closes `δ + 1` shares at cost. Tests: both agree when costs are zero, physical pays cost on `|δ + 1|` only, and an out-of-the-money expiry just closes the position under both.
- [ ] Simulator and exercise dates
  `simulator.py`: a seeded GBM path bank with one contract per path. `exercise.py`: `τ` per path from the chosen method (`tree`, `lsm`, `never`, or `late`, the boundary shifted down).
- [ ] P&L
  `pnl.py`: the one function of the Goal. Tests against closed forms: no hedge gives `p_0 - Z̃`. The step-by-step bookkeeping and the compact formula agree on a hand-worked path. The financed Black-Scholes delta on a European call gives an error that shrinks like `√Δt` (std ≈ 0.0035 K daily, as in the bug entries).
- [ ] Baselines
  `baselines.py`, as policies with the network's interface: no hedge, surrogate delta, tree delta, fixed bands tuned on validation paths, and the Whalley-Wilmott band. This re-measures what delta hedging achieves before any learning, and replaces the invalid results.
- [ ] Dataset
  `generate_hedge_dataset.py` and `hedge_dataset.py` save paths, contracts, premiums, surrogate delta and gamma per day, the boundary and `τ`. The metadata (seeds, ranges, surrogate checkpoint, exercise method) is saved too, so train and test can refuse a mismatched bank. Train, validation and test banks use distinct seeds. The last week before expiry is outside the surrogate's domain (`T` ≥ 1 week) and goes to its tree route, about 18 minutes for 50,000 paths, once. The file is about 220 MB for 50,000 daily paths in float32.
- [ ] Policy and training loop
  `hedge_model.py` and `hedge_training.py`: the band network with the optional boundary input, the objective chosen by flag, `κ` as a setting, and `w` on its own learning rate. Validation checkpointing on held-out paths keeps the untrained (delta) policy as a candidate. Runs are seeded and show a progress bar.
- [ ] Evaluation
  `hedge_evaluation.py`: the network and every baseline on the same test bank (common random numbers), scored with the functions of `risk.py`. Costs and settlement can differ from training.
- [ ] CLI
  `train-hedge` and `test-hedge` rebuilt on the new stack.
- [ ] Experiments
  Each against the right baseline, over 4 seeds:
  - costs (0.1%, 1%, 1% + fixed fee) against delta and the best band, with a few values of `κ` for the fixed fee;
  - one policy over the contract family against per-contract bands;
  - the boundary input on and off, with optimal and late holders;
  - CVaR training with `w` on its own learning rate and on the network's, to check that `w` needs its own;
  - volatility misspecification (realized σ ≠ the hedger's σ) and stochastic volatility, where delta is no longer optimal.


## Backlog

### Benchmark against fast American-put methods

**Goal:** find out whether the MLP surrogate is worth it against the classic fast pricers of American puts: analytic approximations (Barone-Adesi-Whaley, Ju-Zhong) and the integral-equation method of Andersen-Lake-Offengeim, compared on accuracy against the reference tree and on time per option, single and in batch.

**Done so far:** nothing. Kept for later: it needs theory the user does not have yet.

**Next:**
- [ ] Explain the theory to the user before writing any code
  The early-exercise boundary and the integral equation it satisfies, the quadratic approximation behind Barone-Adesi-Whaley and Ju-Zhong, and the fixed-point scheme of Andersen-Lake-Offengeim.
- [ ] Decide with the user which method to implement and how to compare it

### Longstaff-Schwartz

**Goal:** keep the LSM pricer faithful to Longstaff and Schwartz (2001) and collect its open questions in one place, so each one is checked against the paper before the pricer changes. The first one is the Laguerre basis. The paper weights each polynomial by `e^{-x/2}`, while `laguerre_poly_gen` does not, so the code's basis spans the same functions as `simple` (README, "The regression basis does not matter at degree 3").

**Done so far:** the pricer, its pathwise-delta Greeks and the European control variate for calls are built and tested (archived; BUGS, "LSM delta/gamma estimates unreliable near maturity" and "LSM early-exercises American calls and mis-estimates delta").

**Next:**
- [ ] Check the Laguerre basis against the paper
  Reread the paper and decide with the user whether the weight belongs in the code. If it does, compare the weighted basis with `simple` on price against the tree, exercise agreement and delta.

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
