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
| [Hedging rebuild](#hedging-rebuild) | active | Experiments: CVaR threshold learning rate |
| [Dividend-paying stocks](#dividend-paying-stocks) | backlog | Explain the theory to the user |
| [Benchmark against fast American-put methods](#benchmark-against-fast-american-put-methods) | backlog | Explain the theory to the user |
| [Longstaff-Schwartz](#longstaff-schwartz) | backlog | Check the Laguerre basis against the paper |
| [Stochastic volatility](#stochastic-volatility) | backlog | Explain the theory to the user |

## Active

### Hedging rebuild

**Goal:** rebuild the American-put hedger from the ground up as Deep Hedging (Buehler et al. 2019), on a correct P&L, with `PutSurrogate` as the hedger's pricing model. The current stack has two accounting bugs that distort every result so far. Hedges earn one step late ([B-037](BUGS.md#b-037-hedgingenv-applies-each-hedge-one-step-late)) and the P&L has no cash account ([B-038](BUGS.md#b-038-hedging-pl-has-no-cash-account-so-the-delta-baseline-hedges-the-wrong-liability)). It also takes its exercise dates and deltas from LSM. Its exercise rule is fitted on the same paths it is tested on, and its deltas are Monte-Carlo estimates. Their earlier failure near maturity is fixed ([B-003](BUGS.md#b-003-lsm-deltagamma-estimates-unreliable-near-maturity)). The rebuild has six parts:
- **P&L.** One differentiable torch function books it, discounted, with the premium received at `t_0`, cash at `r`, `δ_k` chosen at `t_k` earning `δ_k (S̃_{k+1} - S̃_k)`, and costs on every trade. It takes any policy with the same interface (inputs in, position out), so training, evaluation and every baseline go through this one function. The [B-037](BUGS.md#b-037-hedgingenv-applies-each-hedge-one-step-late) lag survived because the accounting existed twice (`HedgingEnv.step` and `rollout_pnl`). Here it exists once.
- **Building blocks, one file each.** Risk measures (CVaR with `α` as a parameter, entropic, variance), transaction costs (proportional plus fixed), and settlement (one function for cash, one for physical). Training and evaluation use the same risk functions, so "trained on CVaR" and "scored on CVaR" are the same number. The fixed fee's indicator `1[x ≠ 0]` has no gradient, so it is straight-through: the forward pass charges the exact fee and the backward pass uses the slope of `s(x) = 1 - e^{-|x|/κ}`. The P&L stays exact and only the gradient is approximated.
- **Exercise.** The holder exercises the first time `S_t ≤ S*(t)`, with the boundary `S*` read from the tree. Per unit of strike the boundary depends only on the time left, `r` and `σ`, so one tree per contract gives it for every day of the path. The surrogate is not used for it. The price meets the payoff tangentially (smooth pasting), so a 1e-4 price error moves the boundary by about 0.5% of `K`. How `τ` is found is a choice (`tree`, `lsm`, `never`, `late`), so suboptimal holders are a test scenario.
- **Pricing model.** `PutSurrogate` gives the price, delta and gamma. This is where it earns its place. Contracts (`T`, `σ`, `r`, `S_0/K`) are drawn per path, so one policy covers a family of contracts. Pricing a different contract at every (path, step) is 3 ms per option on the tree and microseconds in batches on the surrogate.
- **Policy.** One network with time as an input outputs a no-transaction band around the surrogate delta, `δ_k = clamp(δ_{k-1}, Δ_k + l_k, Δ_k + u_k)` (Imaki et al. 2021). It holds exactly inside the band and starts as the delta hedge. Its inputs are `ln(S_k/K)`, `T - t_k`, `σ`, `r`, `Δ_k`, `Γ_k` and `δ_{k-1}`. An optional 8th input, the distance to the boundary `ln(S_k / S*(T - t_k))`, is switched on by a flag, to measure whether the network learns from it.
- **Objective.** A risk measure of the terminal P&L is minimized by backpropagating through whole paths. The choices are CVaR through the Rockafellar-Uryasev variable `w` (with its own learning rate), the entropic risk, or the variance. There is no critic, no entropy bonus and no per-step reward shaping, so the objective is the judged metric.

The pipeline mirrors the tree surrogate's: generate a dataset, train, test. The dataset stores only what does not depend on training choices: paths, contracts, premiums, surrogate delta and gamma, the boundary and `τ`, and the metadata that produced them. Rebalancing frequency, costs (with `κ`), settlement, risk measure and network settings are chosen at train or test time. So one dataset serves many experiments, and a policy can be tested at other costs than it was trained on.

**Layout.** The non-ML parts go in `optionpricing/hedging/` (`risk.py`, `costs.py`, `settlement.py`, `pnl.py`, `baselines.py`, `simulator.py`, `exercise.py`). The network pipeline goes in `optionpricing/ml/hedging/` (`generate_hedge_dataset.py`, `hedge_dataset.py`, `hedge_model.py`, `hedge_training.py`, `hedge_evaluation.py`). SAC and `HedgingEnv` live in `optionpricing/hedging/legacy/` and `optionpricing/ml/hedging/legacy/`, as the LSM-trained surrogate did, and run as `train-hedge-legacy` and `test-hedge-legacy`. `train-hedge` and `test-hedge` are the Deep Hedging commands. The choices and their reasons are recorded in `docs/results/hedging-training.md`.

**Decided defaults.**
- **Contracts.** `T` from 3 months to 1 year, `σ` from 15% to 30%, `r` from 2% to 6%, `S_0/K` from 0.9 to 1.1. A narrow family first, widened once training works. The ranges are dataset parameters, so widening means generating a new dataset.
- **Rebalancing.** Daily (one date per day to expiry).
- **Exercise between hedge dates.** The holder checks the boundary every day, whatever the hedging grid. If a coarser grid is used, the last position earns until `τ` and settlement happens at `τ`.
- **Risk level.** CVaR at `α` = 95%.
- **Settlement.** Cash, the most common convention. Physical stays available as a flag.
- **Chosen by checks, not fixed in advance.** Dataset sizes, network size and activation, batch size, `κ`, learning rates and how late the late holder is. Each is set by a check, for example more training paths while the validation score still improves with them.
- **SAC.** Whether SAC is retrained once on the new P&L is decided at the end, once Deep Hedging has results.

**Next:**
- [x] Explain the theory to the user before writing any code
- [x] Decide the design with the user
- [x] Move the SAC stack to legacy
- [x] Exercise boundary from the tree
- [x] Risk measures
- [x] Transaction costs
  `costs.py`: `transaction_cost(trade, spot, proportional, fixed, kappa)` = `ε |x| S + f 1[x ≠ 0]`, with the fee straight-through (`hard + (soft - soft.detach())`). Tests: hand values, a zero trade costs nothing, the forward value equals the exact fee, and the gradient equals `ε S sign(x) + f sign(x) e^{-|x|/κ} / κ` (0 at `x = 0`).
- [x] Settlement
  `settlement.py`: `cash_settlement` pays `K - S` and buys back `-δ` shares at cost. `physical_settlement` pays `K`, receives one share and closes `δ + 1` shares at cost. Tests: both agree when costs are zero, physical pays cost on `|δ + 1|` only, and an out-of-the-money expiry just closes the position under both.
- [x] Simulator and exercise dates
- [x] P&L
- [x] Baselines
- [x] Dataset
- [x] Policy and training loop
- [x] Evaluation
- [x] First evaluation run
- [x] CLI
- [x] Keep the PC awake during long runs
- [x] Experiments: boundary input and late holder
- [ ] Experiments: CVaR threshold learning rate
  At a 1% cost, `lr` = 1e-2, 100 epochs, 4 seeds: `w_lr` at 1e-2, 1e-1 and 3e-1. In the first runs `w_lr` equalled `lr`, so the two arms were the same run. `w` also fell only 0.11-0.13 per epoch, so it lagged the network ([Limits](../docs/results/hedging-results.md#limits-and-open-questions)). The winner sets `w_lr` for the two steps below. About 4.5 h.
- [ ] Experiments: proportional costs at 100 epochs
  0.1% and 1%, 4 seeds, with the winning `w_lr`, against delta and the bands tuned on the whole family, which is also the family experiment. Most 50-epoch runs were still improving. At 0.1% the tuned Whalley-Wilmott `γ` was 10, the top of its grid, so the grid goes higher for that cost. The 1% runs are the winning arm of the step above, so only 0.1% is new, about 1.5 h.
- [ ] Experiments: fixed fee over κ
  A $0.05 fee on top of 1%, 100 epochs, 4 seeds, with the winning `w_lr`, at `κ` = 0.001, 0.003, 0.01, 0.03 and 0.1. In the first runs 5 of 12 networks sat at a plateau near the no-hedge score for 37 epochs or more, and 3 never left it, while `w` was still far above the tail. The step checks whether a faster `w` removes that, and which `κ` trains best on the validation bank. About 7.5 h.
- [ ] Experiments: SAC on the new P&L (optional)
  SAC retrained once on the new P&L, for one comparison, if time allows.
- [ ] Per-bucket evaluation (optional)
  Group the test paths by maturity, volatility and moneyness (e.g. 3 × 3 × 2 buckets) and score each bucket, so a family score cannot hide a weak region. Tune the bands per bucket on the validation bank, as a baseline that knows the region ([`docs/results/hedging-results.md`, Limits and open questions](../docs/results/hedging-results.md#limits-and-open-questions)).


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

**Done so far:** the pricer, its pathwise-delta Greeks and the European control variate for calls are built and tested (archived; [B-003](BUGS.md#b-003-lsm-deltagamma-estimates-unreliable-near-maturity) and [B-006](BUGS.md#b-006-lsm-early-exercises-american-calls-and-mis-estimates-delta)).

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

### Stochastic volatility

**Goal:** move the project from constant volatility to the Heston model, where the variance `v_t` is its own random process, correlated with the stock. Under constant volatility the market is complete. The stock alone replicates the put, so at zero cost the delta hedge is optimal, and Deep Hedging can only win on costs and discrete trading. Under Heston a second Brownian motion drives the variance, and the stock cannot hedge it. The market is incomplete, the risk-minimizing hedge is no longer the delta, and the band network has something to learn even at zero cost (Buehler et al. 2019 make this point on Heston). Constant volatility stays as the special case `ξ = 0`, `v_0 = θ`, which must reproduce the current results. This workstream has three layers, and each is validated before the next one is built on it. They are reference pricers (European by Fourier inversion, American on a 2D finite-difference grid), a Heston put surrogate trained on that grid, and the Deep Hedging stack on Heston paths. The mean-reversion speed is written `κ_v`, because `κ` is already the fixed fee's smoothing width.

**Done so far:** nothing. The constant-volatility Hedging rebuild finishes first, and its experiments leave stochastic volatility out.

**Next:**
- [ ] Explain the theory to the user before writing any code
  A full explanation with every derivation, in this order:
  - why constant σ fails: the implied-volatility smile, volatility clustering and the leverage effect;
  - the Heston SDEs, correlated Brownian motions, and the CIR variance with its mean reversion, positivity and Feller condition `2 κ_v θ ≥ ξ²`;
  - the pricing PDE, derived by hedging with the stock and a second option, the market price of volatility risk, and why the risk-neutral measure is no longer unique;
  - the characteristic function, from the affine ansatz and its Riccati ODEs, with the branch-cut fix of Albrecher et al. (2007), and the Fourier inversion that turns it into a price;
  - how `ρ` and `ξ` shape the smile;
  - the American put as a free boundary in `(S, v)`, and how a 2D grid solves it;
  - simulation schemes: full-truncation Euler and Andersen's QE (2008);
  - hedging in an incomplete market: the minimum-variance delta `∂V/∂S + (ρ ξ / S) ∂V/∂v`, and hedging volatility with a second instrument.
- [ ] Decide the design with the user
  The ranges of `κ_v`, `θ`, `ξ`, `ρ` and `v_0`, and whether the Feller condition is enforced. Whether the constant-volatility stack stays beside the Heston one or is replaced by it. The hedging instruments: the stock alone, or the stock plus a variance swap or a European put. What the hedger observes (`v_t`, or only prices). The market price of volatility risk (0, or a parameter).
- [ ] Heston path simulator
  A Heston class beside `GBM` in `optionpricing/core/`, with the QE scheme and full-truncation Euler as a check. Tests: the mean and variance of `v_T` against the CIR closed forms, Monte-Carlo European prices against the Fourier price, and `ξ = 0` matching `GBM`.
- [ ] European Heston pricer
  The semi-closed form by Fourier inversion of the characteristic function. It is the analytic part the surrogate's target subtracts, as Black-Scholes is now. Tests: Monte-Carlo prices, put-call parity, Black-Scholes as `ξ → 0`, and published reference values.
- [ ] American Heston reference pricer
  A 2D finite-difference solver in `(S, v)`, with an ADI scheme (Craig-Sneyd or Hundsdorfer-Verwer) and the early-exercise constraint by operator splitting (Ikonen-Toivanen). It plays the tree's role. One solve gives the price, delta, vega and exercise surface at every `(S, v, t)` for one `(T, r, κ_v, θ, ξ, ρ)`. Tests: its European version against the Fourier price, `ξ → 0` against the CRR tree, the benchmark of Clarke and Parrott (1999), and LSM with `v` as a second regressor.
- [ ] Heston surrogate dataset
  Inputs `(m, T, r, v, κ_v, θ, ξ, ρ)`, eight instead of four. The target is the early-exercise premium over the European Heston price, as for `PutSurrogate`, because it has no kink at the strike. Each grid solve gives many rows, as each tree does now. Sampling, sizes and grid resolution are set by checks, as in `generate_tree_dataset`.
- [ ] Heston surrogate training and evaluation
  The `tree_model.py` network with the new inputs, scored by `test-ml` against the grid per bucket. The hedger uses the vega as much as the delta, so both gradients are scored, not only the price.
- [ ] Exercise surface
  The holder exercises the first time `S_t ≤ S*(t, v_t)`, read from the grid. The boundary is now a surface, so the lookup interpolates in `v` as well as in time.
- [ ] Hedging dataset on Heston paths
  It also stores the variance paths, the Heston parameters of each contract and the surrogate vega.
- [ ] Policy and baselines
  The policy gains `√v_t`, the vega and the Heston parameters as inputs. Baselines: the Black-Scholes delta at `σ = √v_t` (a constant-volatility trader), the Heston delta `∂V/∂S` and the minimum-variance delta. With a second instrument, the policy outputs two positions and the P&L books both.
- [ ] Experiments
  Against the minimum-variance delta, over 4 seeds:
  - zero cost, where the network can beat the delta hedge without costs for the first time;
  - sweeps of `ρ` and `ξ`, to see the gain grow with the volatility risk;
  - `ξ = 0`, which must match the constant-volatility results;
  - costs, as in the Hedging rebuild;
  - model risk: a policy trained on Heston paths and tested on GBM paths, and the reverse.
