# How the hedging model is designed

This page records the design choices for the Deep Hedging rebuild of the American-put hedger: what was chosen, what it implies, and why. The work is planned in `.claude/PLAN.md` ("Hedging rebuild") and **not built yet**, except the exercise boundary. So each section gives the reasoning behind a choice, not a measured result, apart from the boundary's accuracy. Experiments will add their results here as they are run, in the style of `docs/results/surrogate-training.md`.

**Units.** Positions are in shares per option. Prices and P&L are per option at K = 100, in dollars. Costs use a proportional rate `ε` (1% means 1 cent per dollar traded) and a fixed fee `f` per trade.

## The decisions at a glance

| Decision | Instead of | Main reason | Section |
|---|---|---|---|
| Deep Hedging | SAC, the current reinforcement-learning hedger | the P&L is a known formula, so its gradient is exact and the training objective is the judged metric | Deep Hedging instead of SAC |
| One P&L function, discounted, for every policy | the env's per-step accounting, copied in `rollout_pnl` | the copy is how the one-step lag (B-037) survived | One P&L function |
| A no-transaction band around delta as the output | a position output | holds exactly inside the band and starts as the delta hedge | The band network |
| Distance to the exercise boundary as an optional input | a fixed input list | lets an experiment measure whether the network uses it | The inputs |
| Exercise boundary from the tree | from LSM | LSM's rule is fitted in sample, and one tree per contract gives the boundary for every day | The exercise boundary |
| Straight-through gradient for the fixed fee | a smooth fee in the P&L, or ignoring the fee in training | the P&L stays exact and the network still gets a signal | The fixed fee: straight-through |
| Risk measures, costs and settlement in their own files | built into the training loop | training and testing score with the same functions | The building blocks |
| Separate dataset, training and test steps | one script | every choice can be changed at its own step | The pipeline |
| Narrow contract family first | the surrogate's whole domain | faster first results, widened once training works | The defaults |
| Daily rebalancing | weekly | realistic, and comparable with the earlier SAC results (also daily) | The defaults |
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

**Choice.** The holder exercises the first time `S_t ≤ S*(t)`, with the boundary `S*` from the binomial tree (`exercise_boundary` in `optionpricing/pricing/american/batch_binomial_tree.py`). How `τ` is found stays a choice per dataset: `tree` (optimal), `lsm` (to compare with the legacy results), `never`, or `late` (the boundary shifted down).

**What the boundary is.** A put pays `K - S`, so exercising it is worth considering only when the stock is low. At every date there is a price `S*(t)` below which exercising now beats waiting. It is a price, in dollars, not a time. It rises toward `K` as expiry nears, because waiting is worth less and less. For the default contract (`T` = 1, `r` = 5%, `σ` = 20%), from a 16,000-step tree:

| Time left | 1 year | 3 months | 1 week | 0 (expiry) |
|---|---|---|---|---|
| `S*` ($) | 80.94 | 86.84 | 94.57 | 100 |

The exercise date `τ` of a path is the first day its price is at or below that day's `S*`.

**How the tree finds it.** At each time step the tree lists the prices the stock can reach, from lowest to highest. At each of these nodes it compares two values:
- exercise: the payoff `K - S`;
- hold: the discounted average of the option's values one step later.

For a put, exercising wins at the low prices and holding wins at the high ones. So at each step the nodes split into two groups, exercise below and hold above. The boundary lies between the highest node of the exercise group and the lowest node of the hold group. The nodes are 1.3% apart at 1000 steps, so the code places `S*` where "hold minus exercise" crosses zero, on a straight line between those two nodes. Today's step for the default contract:

| Node price ($) | Hold ($) | Exercise ($) | Hold − exercise ($) |
|---|---|---|---|
| 80.651 | 19.3440 | 19.3487 | −0.0047 |
| 81.678 | 18.3334 | 18.3220 | +0.0114 |

The zero crossing is at $80.95. Deep in the exercise region, hold − exercise is about −$0.005: waiting one step only delays receiving `K`, which loses one step of interest, `100 × 5% × 0.001 year`.

**Why one tree per contract is enough.** The decision at a node depends only on its price, the time left, `r` and `σ`, not on where the tree started. So `S*/K` depends only on the time left, `r` and `σ`. One tree per contract gives the boundary at every one of its steps, for any `S_0`, for about the cost of one price.

**Starting the tree at the strike, with extra nodes below.** To price an option the tree starts at today's price `S_0`. For the boundary the starting price does not matter, so the tree starts at `K`, and the function needs no `S_0`. A tree fans out from where it starts: after `i` steps, its `i + 1` prices go from `K u^{-i}` to `K u^{i}`, with `u` = 1.0063 for the default contract at 1000 steps. Near the start the fan is narrow, about ±6% after 10 steps. The boundary is 19% below `K`, so for the first 33 steps no node is low enough to be exercised, and there is no boundary to read.

The fix widens the fan downward. At every step the tree also computes a few more prices below its lowest one, as if it had also started from `K u^{-2}`, `K u^{-4}`, and so on. This implies three things:
- every step now has nodes on both sides of the boundary;
- the original nodes keep their values exactly, because a node's value comes only from the nodes it can reach later;
- each step costs one more node update per extra node, about `N × nb_extra_nodes` in total on top of the tree's `N²/2`.

**How many extra nodes.** The count, `nb_extra_nodes` in the code (set by `_nb_extra_nodes`), must be large enough for the lowest starting price to sit below the boundary. The steps below use the default contract: `K` = $100, `T` = 1 year, `r` = 5%, `σ` = 20%, 1000 steps.

1. **How the tree moves.** Each step lasts `dt = T / N` = 0.001 year. In one step the price is multiplied either by `u` (up) or by `1/u` (down), with `u = e^{σ √dt}` = `e^{0.006325}` = 1.006345, so ±0.63% per step.
2. **Where the extra prices sit.** At one step, neighbouring prices differ by a factor `u²`: going from one to the next swaps a down move for an up move. The extra prices must stay on this grid, so that each leads to two prices of the next step by one up or one down move. So the starting step holds

   ```
   K u^{-2n}  ...  K u^{-4}   K u^{-2}   K
   ```

   and with `n` extra nodes the lowest starting price is `K u^{-2n}`.
3. **The floor: the lowest the boundary ever goes.** The boundary itself is what is being computed, so it cannot set `n`. A lower bound on it can. An American put with more time left is worth at least as much as one with less: it can follow any exercise plan the shorter one can, and has more besides. So at every price, holding it is worth more, and exercising beats holding only at lower prices. So the longer the maturity, the lower the boundary, and the put that never expires (the *perpetual* put) has the lowest boundary of all. That boundary has a closed form (Merton 1973):

   ```
   S*_∞ = K · 2r / (2r + σ²) = 100 × 0.10 / (0.10 + 0.04) = $71.43
   ```

   Every finite-maturity boundary lies above it: $80.94 today for the default contract.
4. **The count.** `n` is the smallest whole number with the lowest starting price at or below the floor:

   ```
   K u^{-2n} ≤ K · floor   ⟺   n ≥ ln(1 / floor) / (2 ln u) = ln(1 / floor) / (2 σ √dt),   with floor = 2r / (2r + σ²)
   ```

   For the default contract, `ln(1 / 0.7143) / (2 × 0.006325)` = 0.3365 / 0.01265 = 26.6, so 27 extra nodes. The lowest starting price is then `100 × u^{-54}` = $71.07, below the $71.43 floor. Later steps reach lower still, since the fan keeps widening. The code:

   ```python
   perpetual = 2 * r / (2 * r + sigma**2)
   return int(np.ceil(np.max(np.log(1 / perpetual) / (2 * sigma * np.sqrt(dt)))))
   ```

   When contracts are priced together they share one array, so `np.max` takes the largest count among them. With `r ≤ 0` the count is 0, since there is no early exercise to find.

That is enough, because the tree's exercise threshold lies above the exact boundary, which lies above the floor. It was checked on contracts out to `T` = 100 years. For the hedging family the count is at most 125 (at `T` = 3 months, `σ` = 30%, `r` = 2%), so the extra work is up to about 25%. A thousand contracts drawn from the family take 3.6 s at 1000 steps.

**Accuracy.** The reference is the price where an 8000-step tree stops exercising today, found by bisection. Errors in dollars at `K` = 100, for the default contract, read along one tree's curve as the hedger reads it:

| Time left | 1 year | 6 months | 3 months | 5 weeks | 1 week | 4 days |
|---|---|---|---|---|---|---|
| Nearest exercise node, 1000 steps | −0.31 | −0.21 | +0.16 | +0.19 | −0.14 | +0.41 |
| Interpolated, 1000 steps | −0.01 | +0.02 | +0.18 | +0.21 | +0.10 | +0.45 |
| Interpolated, 4000 steps | −0.08 | −0.03 | −0.02 | −0.01 | +0.05 | not measured |

- **The interpolation matters most early in the tree.** With a year left it cuts the error from 31 cents to 1.
- **Late in the tree the error grows.** The tree's threshold sits above the exact boundary, by an amount that shrinks with the step size. Late in a one-year tree the steps are those of the year, so 3 months left get 250 steps instead of the 1000 a 3-month tree would use. Going to 4000 steps cuts the worst error from $0.21 to $0.08 over the first five times.
- **The curve wiggles a little.** Neighbouring steps' nodes are offset by one up move, so the curve dips by up to a sixth of the node spacing (about $0.20) from one step to the next, mostly in the last few percent of the maturity.
- **The reference is itself uncertain** by about $0.03 (the 16,000-step value is 3 cents lower than the 8000-step one with a year left).

**Why these errors are acceptable.** An error in `S*` matters through the exercise date it gives each path, and through what a wrong date costs.
- **Most paths get the same exercise date.** The holder checks once a day. A path's date changes only if, on some day, its price falls between the tree's `S*` and the exact one, a band at most $0.45 wide. Most paths never land in it. A path that crosses the boundary in one move lands in it rarely, and is exercised a day earlier or later. Only a path that drifts along just above the boundary for several days can move by more.
- **A wrong date costs almost nothing.** Inside that band, exercising and holding are worth almost the same. Just above the boundary the put is worth its payoff plus about `½ Γ (S - S*)²` (smooth pasting), so a trigger off by `ε` loses at most about `½ Γ ε²`, however many days the date moves. With `Γ` ≈ 3.5-5 per unit of strike for most of the option's life, an error of $0.21 costs about $0.001. With 4 days left, `Γ` ≈ 17 and the error of $0.45 costs about $0.017.
- **That is far below the noise the experiments measure.** A correctly financed daily delta hedge already has a P&L standard deviation of $0.35 (B-038). The worst boundary loss is 20 times smaller.
- **More accuracy is cheap if a check asks for it.** 4000 steps take 0.3 s per contract. The dataset step chooses the number of steps by such a check, like its other settings.

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
| Rebalancing | daily | Realistic, and comparable with the earlier SAC results (also daily). The dataset is stored daily, so a coarser grid is a subset of its dates. |
| Exercise between hedge dates | the holder checks the boundary every day | The holder does not follow the hedger's schedule. On a coarser grid, the last position earns until `τ` and settlement happens at `τ`. |
| Risk level | CVaR at `α` = 95% | The level used across the project so far. |
| Settlement | cash | The most common convention. Physical settlement stays available. |

**Chosen by checks.** Dataset sizes, network size, batch size, `κ`, the learning rates and how late the late holder is are not fixed in advance. Each is set by a check and recorded here with its evidence. For example, the training set grows while the validation score still improves with more paths.

## Open questions

- **SAC on the new P&L.** Whether SAC is retrained once for one comparison. Decided at the end, once Deep Hedging has results.
