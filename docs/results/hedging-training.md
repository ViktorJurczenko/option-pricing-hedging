# How the hedging model is designed

This page records the design choices for the Deep Hedging rebuild of the American-put hedger: what was chosen, what it implies, and why. The work is planned in `.claude/PLAN.md` ("Hedging rebuild") and **not built yet**. So each section gives the reasoning behind a choice, not a measured result. Experiments will add their results here as they are run, in the style of `docs/results/surrogate-training.md`.

**Units.** Positions are in shares per option. Prices and P&L are per option at K = 100, in dollars. Costs use a proportional rate `ε` (1% means 1 cent per dollar traded) and a fixed fee `f` per trade.

## The decisions at a glance

| Decision | Instead of | Main reason | Section |
|---|---|---|---|
| Deep Hedging | SAC, the current reinforcement-learning hedger | the P&L is a known formula, so its gradient is exact and the training objective is the judged metric | Deep Hedging instead of SAC |
| One P&L function, discounted, for every policy | the env's per-step accounting, copied in `rollout_pnl` | the copy is how the one-step lag (B-037) survived | One P&L function |
| A no-transaction band around delta as the output | a position output | holds exactly inside the band and starts as the delta hedge | The band network |
| Distance to the exercise boundary as an optional input | a fixed input list | lets an experiment measure whether the network uses it | The inputs |
| Exercise boundary from the tree | from the surrogate, or from LSM | the surrogate is badly conditioned at the boundary, and LSM's rule is fitted in sample | The exercise boundary |
| Straight-through gradient for the fixed fee | a smooth fee in the P&L, or ignoring the fee in training | the P&L stays exact and the network still gets a signal | The fixed fee: straight-through |
| Risk measures, costs and settlement in their own files | built into the training loop | training and testing score with the same functions | The building blocks |
| Separate dataset, training and test steps | one script | every choice can be changed at its own step | The pipeline |
| Narrow contract family first | the surrogate's whole domain | faster first results, widened once training works | The defaults |
| Daily rebalancing | weekly | realistic, and comparable with the earlier results | The defaults |
| Holder checks the boundary daily | only on hedge dates | the holder does not follow the hedger's schedule | The defaults |
| CVaR at 95%, cash settlement | other levels, physical settlement | the project's level so far, and the most common convention | The defaults |

## Deep Hedging instead of SAC

**Choice.** The hedger is a neural network trained by Deep Hedging (Buehler, Gonon, Teichmann and Wood, 2019). A risk measure of the terminal P&L, computed over a batch of simulated paths, is minimized directly by gradient descent. SAC and `HedgingEnv` move to legacy packages, reachable as `train-hedge-legacy` and `test-hedge-legacy`.

**Why.** SAC treats the P&L as a black box. It estimates the gradient with a second network (the critic) and random exploration, and it optimizes a daily reward `x - λx²` that only matches the variance of the final P&L when each daily reward has zero mean. Without a cash account it doesn't. Here the P&L is a known formula. Backpropagation through the whole path gives its exact gradient, with no critic and no reward shaping, so the objective is the metric the hedger is judged on.

**What it implies.** No labels are needed: training lowers a score of the batch, not an error against a target. The loss is a property of the whole batch (CVaR is the average of the worst 5% of paths), so batches must be large, about 4096 paths.

## One P&L function

**Choice.** One differentiable torch function books the P&L in today's dollars:

```
PL = p_0 + Σ_k δ_k (S̃_{k+1} - S̃_k) - Σ_k c̃_k - Z̃
```

`p_0` is the premium received at `t_0`. `S̃_k = e^{-r t_k} S_k` is the discounted price, so cash earns `r`. `δ_k` is chosen at `t_k` and earns over `[t_k, t_{k+1}]`. `c̃_k` is the discounted cost of the trade at `t_k`, including the opening and closing trades. `Z̃` is the discounted settlement at exercise or expiry. The function takes any policy with one interface (inputs in, position out). The network, every baseline, training and testing all call it.

**Why.** The current env gets two terms wrong. It pays the gain over `[t_k, t_{k+1}]` to the position chosen at `t_{k-1}` (B-037). It also has no premium and no cash account (B-038). The lag survived because the accounting existed twice, in `HedgingEnv.step` and in `rollout_pnl`, and the copy kept the bug. With one function, a baseline and the network can't be booked differently.

## The band network

**Choice.** The network outputs two numbers per day, turned into a band around the surrogate delta `Δ_k`:

```
l_k = -softplus(a_k) ≤ 0,   u_k = softplus(b_k) ≥ 0
δ_k = clamp(δ_{k-1}, Δ_k + l_k, Δ_k + u_k)
```

This is the no-transaction band network of Imaki et al. (2021).

**Why.**
- **It holds exactly.** Inside the band the position is exactly yesterday's, so no cost is paid. The SAC policy's main failure under costs was not being able to hold still (B-010).
- **It starts as the delta hedge.** With `a` and `b` initialized very negative, the band has zero width and the clamp returns `Δ_k`. Training starts from a good strategy.
- **The band can change shape.** Wide where gamma is small, narrow near the strike close to expiry, asymmetric if that helps. A fixed band can't.

## The inputs

**Choice.** At day `k` the network sees `ln(S_k/K)`, `T - t_k`, `σ`, `r`, the surrogate's `Δ_k` and `Γ_k`, and its own previous position `δ_{k-1}`. An optional 8th input, the distance to the exercise boundary `ln(S_k / S*(T - t_k))`, is switched on by a flag.

**Why.** Everything except `δ_{k-1}` depends only on the path, so it is computed once in the dataset step. `δ_{k-1}` is needed because under costs the best trade depends on the current position. The boundary input is optional so that an experiment can train with and without it, against an optimal holder and a late one, and measure whether the network learns from it.

## The exercise boundary

**Choice.** The holder exercises the first time `S_t ≤ S*(t)`. The boundary `S*` comes from the batch binomial tree: a new induction function, replacing `_induct` in `optionpricing/pricing/american/batch_binomial_tree.py`, records at every level the highest node where exercising beats holding. How `τ` is found stays a choice per dataset: `tree` (optimal), `lsm` (to compare with the legacy results), `never`, or `late` (the boundary shifted down).

**Why the tree.** Per unit of strike, the boundary depends only on the time left, `r` and `σ`. So one tree per contract gives `S*` for every day of the path, for the cost of one price (about 3 ms).

**Why not the surrogate.** At the boundary the put price meets its payoff tangentially (smooth pasting). The gap between them grows roughly like the square of the distance to `S*`, so a small price error moves the boundary a lot. On the 1000-step tree for the default contract (`T` = 1, `r` = 5%, `σ` = 20%, `S*` = 81.11), the gap per unit of strike is 7.5e-5 at moneyness `m = S/K` = 0.815 and 2.5e-4 at `m` = 0.82. A 1e-4 price error therefore moves the boundary by about 0.5% of K.

**Why not LSM.** The current env fits LSM's exercise rule on the same paths it then tests on.

## The fixed fee: straight-through

### The problem

A trade of `x` shares at price `S` costs

```
c(x, S) = ε |x| S + f · 1[x ≠ 0]
```

`loss.backward()` needs the derivative of every step between the weights and the loss, including `∂c/∂x`. The proportional part gives `ε S sign(x)`, a real signal toward smaller trades. The fixed part gives nothing. `1[x ≠ 0]` is flat everywhere except at 0, where its derivative is a Dirac delta. On a computer that is 0 for every path, and PyTorch treats the comparison `x != 0` as a constant anyway. The network would see the fee in the P&L but get no gradient from it, so it would learn as if the fee didn't exist.

### The choice

The forward pass charges the exact fee. The backward pass uses the slope of a smooth look-alike `s(x) = 1 - e^{-|x|/κ}`:

```python
hard = (x != 0).float()  # exact step: 0 or 1, no gradient
soft = 1 - torch.exp(-x.abs() / kappa)  # smooth look-alike, has a gradient
indicator = hard + (soft - soft.detach())
cost = eps * x.abs() * S + f * indicator
```

In the forward pass `soft - soft.detach()` is exactly 0, so `indicator` equals `hard` and the fee is exact. In the backward pass `hard` and `soft.detach()` count as constants, so the gradient comes from `soft` alone:

```
d(indicator)/dx = ds/dx = sign(x) / κ × e^(−|x|/κ)
```

### What the gradient says

With a fixed fee `f` = 0.05 dollars and `κ` = 0.01 shares:

| Trade `x` (shares) | Fee charged in the P&L, forward (dollars) | Gradient of the fee, `f · s'(x)`, backward (dollars per share) | What the gradient says |
|--:|--:|--:|---|
| 0.0005 (tiny) | 0.05 | 4.76 | "Shrink this trade, it costs a full fee for almost nothing." |
| 0.02 (small) | 0.05 | 0.68 | "Shrinking it helps a bit." |
| 0.2 (large) | 0.05 | ≈ 0 | "Shrinking it a little won't save the fee, so the fee is no reason to change." |
| 0 (holding) | 0 | 0 | Nothing to save. |

The table lists purchases (`x > 0`). A sale of the same size has the same gradient with a minus sign (−4.76, −0.68, ≈ 0), because the fee grows with `|x|`, so its slope follows `sign(x)`. Gradient descent steps against the gradient, so both a purchase and a sale are pushed toward a smaller trade.

For scale, the proportional cost at `ε` = 1% and `S` = 100 has a gradient of 1 dollar per share. The fee pushes hardest on tiny trades, which are the ones a fixed fee makes wasteful.

### Why straight-through

| Option | P&L used in training | Gradient from the fee | Why not chosen |
|---|---|---|---|
| Ignore the fee in training | exact | none | the network learns as if the fee didn't exist |
| Smooth fee, annealed (`s(x)` in the forward pass too, `κ` shrunk over training) | undercharges small trades until `κ` is small, since `s(x) < 1` | slope of `s` | trains on a P&L that is wrong early on |
| Score-function (REINFORCE) estimator | exact | unbiased but very noisy | brings back the random exploration Deep Hedging avoids |
| Exact boundary term (how the chance of crossing the band edge changes, times `f`) | exact | exact | involved to derive and code through a 365-day loop |
| **Straight-through** | **exact** | **slope of `s`** | chosen |

Straight-through is the only cheap option that keeps the P&L exact. Its approximation is confined to the direction of improvement.

### What it implies

- **The gradient is biased.** It is the slope of `s`, not of the true expected fee. Its direction is right (a smaller trade is closer to no trade), but its size depends on `κ`. A small `κ` gives strong signals only for very small trades, and a large `κ` pushes on bigger trades too. `κ` is a training setting, tried over a few values.
- **It widens the band through trading days only.** On a day the network trades, the clamp puts `δ_k` on a band edge, and "shrink `x_k`" moves that edge toward yesterday's position, which widens the band. Once the trade is exactly 0, the fee and its gradient disappear. Holding days give no fee gradient, which is fine, because no fee was paid.
- **The reported numbers stay honest.** The P&L, the loss value, validation checkpointing and testing all charge the exact fee. A poor `κ` can make training less effective. It can't make the results look better than they are.

## The building blocks

**Choice.** Three small files, each used by training, testing and the baselines alike:
- **Risk measures** (`risk.py`): `cvar(pnl, alpha)`, the average of the worst `1 - α` share, as the reported metric. `cvar_loss(pnl, w, alpha)`, the Rockafellar-Uryasev form `w + mean((L - w)^+) / (1 - α)` with `L = -PL`, for training. `entropic(pnl, lam)`, computed with `logsumexp`. `variance(pnl)`.
- **Costs** (`costs.py`): `transaction_cost(trade, spot, proportional, fixed, kappa)`, with the straight-through fee above.
- **Settlement** (`settlement.py`): `cash_settlement` pays `K - S` and buys back the short shares at cost. `physical_settlement` pays `K`, receives one share and closes `δ + 1` shares at cost.

**Why.** "Trained on CVaR" and "scored on CVaR" must be the same number, so both steps call the same function. In CVaR training, `w` is one extra trainable number with its own, larger learning rate. In earlier experiments outside this repository (not re-run here), `w` on the network's learning rate lagged far behind the true VaR and the network did no better than delta. The training loop takes the two learning rates separately, so an experiment can check this. Both settlements give the same P&L before costs and differ only in the cost of closing the position, so the choice is a flag rather than a design fork.

## The pipeline

**Choice.** Three steps, as for the tree surrogate:
1. **Dataset** (`generate_hedge_dataset.py`) simulates the paths, draws one contract per path, and computes the premium, the surrogate's delta and gamma per day, the boundary and `τ`. It saves them with the metadata that produced them (seeds, ranges, surrogate checkpoint, exercise method).
2. **Training** (`hedge_training.py`) trains the band network on a dataset.
3. **Testing** (`hedge_evaluation.py`) scores the network and every baseline (no hedge, surrogate delta, tree delta, fixed bands, the Whalley-Wilmott band) on a test dataset with its own seed.

| Fixed when the dataset is generated | Chosen at train or test time |
|---|---|
| Paths (daily), contracts, premiums | Rebalancing frequency (weekly dates are a subset of the daily ones) |
| Surrogate delta and gamma per day | Cost rates `ε` and `f`, and `κ` |
| Boundary and `τ`, from the chosen method | Settlement, risk measure, `α` or `λ` |
| Metadata | Network size, learning rates, the boundary input |

**Why.** Every choice can be changed at its own step without redoing the others. One dataset serves many experiments, and a policy can be tested at costs it wasn't trained on. Computing the surrogate features once matters: 50,000 paths × 365 days is about 18 million surrogate queries. The last week before expiry is outside the surrogate's domain (`T` ≥ 1 week) and goes to its tree route, about 18 minutes for 50,000 paths, paid once.

## The defaults

Each default is a flag or a dataset parameter, so an experiment can change it.

| Setting | Default | Why |
|---|---|---|
| Contract family | `T` 3 months to 1 year, `σ` 15-30%, `r` 2-6%, `S_0/K` 0.9-1.1 | Narrow enough to train quickly, wide enough that the surrogate is needed. Widened once training works, by generating a new dataset. |
| Rebalancing | daily | Realistic, and comparable with the earlier results. The dataset is stored daily, so a coarser grid is a subset of its dates. |
| Exercise between hedge dates | the holder checks the boundary every day | The holder does not follow the hedger's schedule. On a coarser grid, the last position earns until `τ` and settlement happens at `τ`. |
| Risk level | CVaR at `α` = 95% | The level used across the project so far. |
| Settlement | cash | The most common convention. Physical settlement stays available. |

**Chosen by checks.** Dataset sizes, network size, batch size, `κ`, the learning rates and how late the late holder is are not fixed in advance. Each is set by a check and recorded here with its evidence. For example, the training set grows while the validation score still improves with more paths.

## Open questions

- **SAC on the new P&L.** Whether SAC is retrained once for one comparison. Decided at the end, once Deep Hedging has results.
