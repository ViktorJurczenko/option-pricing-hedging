# RL hedging: results and analysis

The full results of the SAC hedging agent against the analytic baselines, with the reasoning behind its action modes and checkpointing. The README keeps a summary. Measured 2026-09-30 to 2026-10-01 (commits `c4c34a6`..`be73d51`), **not re-verified** since. These numbers are distorted by two accounting bugs in the environment. Each hedge earns one step late (`.claude/BUGS.md`, "HedgingEnv applies each hedge one step late"), and the P&L has no cash account ("Hedging P&L has no cash account, so the delta baseline hedges the wrong liability"). Most of RL's edge over delta on puts is likely the carry the second bug creates. The hedger is being rebuilt (`.claude/PLAN.md`, "Hedging rebuild"), and these results will be replaced.

## RL hedging policy vs. analytic baselines

```
python main.py test-hedge --policy rl
python main.py test-hedge --policy delta
python main.py test-hedge --policy no_hedge
```

Each policy's deterministic action, evaluated over the same 2,000 simulated
hedging episodes (`--seed 999 --n_episodes 2000`) with a 0.1% proportional
transaction cost. `test-hedge` reseeds `np.random` from `--seed` right
before evaluation (`scripts/test_hedge.py`), so every policy sees identical
simulated paths — otherwise differences below could just be path-sampling
noise rather than a real difference between policies. RL rows give the
range over four policies trained with the `train-hedge` defaults (seeds
42/1/2/3, 1,000 episodes each). All figures are ×K:

| Policy | Put: Std P&L | Put: 95% CVaR | Call: Std P&L | Call: 95% CVaR |
|---|---|---|---|---|
| **RL (default)** | **0.0089-0.0096** | **-0.100 to -0.103** | 0.0142-0.0149 | -0.112 to -0.113 |
| Delta-hedge (LSM delta) | 0.0119 | -0.110 | **0.0142** | **-0.112** |
| No hedge | 0.0725 | -0.186 | 0.154 | -0.571 |

Mean P&L is about -0.08 for every hedged policy (it excludes the option
premium, so it isn't a hedge-quality metric). P&L is the hedge's cash flow
minus the option payoff at exercise, scaled by the strike; std P&L is the
hedging error, and 95% CVaR (the mean of the worst 5% of episodes) is the
tail risk a hedger actually cares about minimizing.

- **On the American put, the RL policy beats delta hedging** on every seed:
  19-25% lower hedging error and a better tail. It even beats hedging with
  the exact binomial-tree delta (std 0.012-0.013).
- **On the call, it matches delta hedging.** Here training never improved
  on the untrained policy, which already hedges at delta.

Two design choices got it there:
- **Residual action:** the policy learns a bounded correction to the LSM
  delta, `clip(delta + 0.1·tanh(u))`, rather than the whole hedge ratio.
  An untrained policy therefore starts at the delta hedge.
- **Validation checkpointing:** the checkpoint is the policy with the lowest
  P&L std on 1,000 held-out paths, with the untrained policy as a candidate.
  Selecting on the recent training reward instead mostly measured which
  paths were drawn. It left the policies behind delta (call std 0.017-0.026,
  put 0.013-0.020), and a plain non-residual policy did worse still (call
  0.029-0.036).

A likely reason the put policy can beat even the exact delta: episodes end
at the *LSM's* exercise time, so the liability being hedged is the put
under LSM's slightly suboptimal exercise rule, which a learned hedge can
adapt to and a model delta cannot. This is a hypothesis, not yet verified.
Full numbers are in `.claude/BUGS.md`, "RL hedging policy trails the LSM
delta hedge". With higher trading costs, where daily delta hedging
overtrades, the picture changes: see the next section.

## Hedging under trading costs

```
python main.py train-hedge --prop_tc 0.01 --action_mode gap_fraction --val_metric cvar
python main.py test-hedge  --prop_tc 0.01 --policy rl
python main.py test-hedge  --prop_tc 0.01 --policy delta_band --band 0.1 --band_target edge
```

With a 0.1% cost, daily delta hedging is already close to optimal. With
larger costs, rebalancing to delta every day spends most of the P&L on
small trades (the baselines over a grid of proportional and fixed costs are
in [`hedging-costs.md`](hedging-costs.md)). The classical fix is a **no-trade
band** (`--policy delta_band`): hold the position until it drifts more
than `b` from delta,
then rebalance to delta (best under fixed costs) or only back to the
band's edge (best under proportional costs). Under costs the band, not
daily delta, is the bar for RL. The comparison is on CVaR rather than std,
because std ignores how much a policy spends on trading.

At a 1% proportional cost (95% CVaR ×K, same 2,000 test episodes; RL rows
give the range over 4 training seeds):

| Policy | Call | Put |
|---|---|---|
| Daily delta | -0.198 | -0.208 |
| RL, residual action (`--residual_scale 0.25`) | -0.173 to -0.186 | -0.140 to -0.153 |
| RL, gap-fraction action (`--action_mode gap_fraction`) | -0.142 to -0.147 | -0.135 to -0.144 |
| **Band 0.1, rebalance to edge** | **-0.136** | **-0.130** |

- **RL clearly beats daily delta**, and the `gap_fraction` action helped
  most. There the policy chooses what fraction of the gap to delta to close
  each day, so holding still is easy to represent. With the residual
  action, holding means tracking a moving target exactly, and every miss is
  a paid trade.
- **RL still trails the hand-tuned band** by 0.006-0.009, even with 3,000
  training episodes. It learned to close part of the gap every day (a
  smoothed delta hedge) rather than the band's "sit still, then act", and
  it trades about twice as much.
- **Giving the policy the band's form closes that gap** (next paragraph),
  with the same SAC objective. So the action parametrization was the
  limit, not the objective mismatch suspected at first (SAC trains on a
  per-step mean-variance reward but is judged on terminal CVaR). Details
  are in `.claude/BUGS.md`, "RL policy under trading costs keeps trading
  every day".

**Learned no-trade band** (`--action_mode band`). The policy outputs a 2-D
action: a band half-width `b` up to `--band_max` (default 0.2), and how far
into the band to rebalance once outside it, from the band's edge to delta.
Inside the band it holds its position exactly, so holds pay neither cost. A
constant action is the fixed band above, so RL starts from the baseline's
form and can learn a band that varies with time to expiry, moneyness and
gamma:

```
python main.py train-hedge --prop_tc 0.01 --fixed_tc 0.05 --action_mode band --val_metric cvar
python main.py test-hedge  --prop_tc 0.01 --fixed_tc 0.05 --policy rl
```

Same protocol (95% CVaR ×K, 2,000 test episodes, 4 training seeds of 1,000
episodes), at a 1% proportional cost alone and with a $0.05 fixed cost per
trade on top:

| Policy | 1%: Call | 1%: Put | 1% + $0.05: Call | 1% + $0.05: Put |
|---|---|---|---|---|
| Daily delta | -0.198 | -0.208 | -0.380 | -0.390 |
| RL, gap-fraction action | -0.143 to -0.147 | -0.135 to -0.144 | -0.324 to -0.329 | -0.325 to -0.332 |
| Best fixed band | -0.136 (0.1, edge) | -0.130 (0.1, edge) | -0.148 (0.175, delta) | -0.146 (0.175, delta) |
| **RL, learned band** | **-0.136 to -0.137** | **-0.123 to -0.128** | **-0.145 to -0.147** | **-0.132 to -0.138** |

- **The learned band closes the gap to the hand-tuned band** in every
  cell: it matches it for calls at 1% and beats it elsewhere, by up to
  0.014 (puts with a fixed cost), on every seed. No band width or target
  had to be tuned for it.
- **A fixed cost breaks the gap-fraction action.** It can only hold still
  in the limit, so its "holds" are tiny trades that each pay the fixed
  cost; it ends up barely better than daily delta. The band action holds
  exactly and is unaffected.
