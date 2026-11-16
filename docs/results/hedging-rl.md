# RL hedging: results and analysis

The SAC hedging agent against the analytic baselines, with the reasons behind its action modes and its checkpointing. The README keeps a summary.

**Status.** Measured 2026-09-30 to 2026-10-01 (commits `c4c34a6`..`be73d51`), **not re-verified** since. Two accounting bugs in the environment distort these numbers. Each hedge earns one step late (`.claude/BUGS.md`, "HedgingEnv applies each hedge one step late"), and the P&L has no cash account ("Hedging P&L has no cash account, so the delta baseline hedges the wrong liability"). Most of RL's edge over delta on puts is likely the carry the second bug creates. The hedger is being rebuilt (`.claude/PLAN.md`, "Hedging rebuild"), and these results will be replaced.

## Setup

**The task.** Hedge a short American option with the underlying, until the option's exercise date under the fitted LSM rule. The legacy environment (`HedgingEnv`) wraps a fitted `LongstaffSchwartz` on simulated GBM paths.

**The policies.**

| Policy | What it does |
|---|---|
| RL, residual action (default) | a bounded correction to the LSM delta, `clip(delta + 0.1·tanh(u))` |
| RL, gap-fraction action | chooses what fraction of the gap to delta to close each day |
| RL, learned band | outputs a band half-width `b` up to `--band_max` (default 0.2) and how far into the band to rebalance once outside it. Holds exactly inside the band |
| Daily delta | rebalances to the LSM delta every day |
| Band | holds until the position drifts more than `b` from delta, then rebalances to delta or to the band's edge |
| No hedge | holds nothing |

**The metrics.** P&L is the hedge's cash flow minus the option payoff at exercise, scaled by the strike (×K). Std P&L is the hedging error. 95% CVaR, the mean of the worst 5% of episodes, is the tail risk a hedger cares about. Under costs, the comparison is on CVaR, since std ignores how much a policy spends on trading. Mean P&L is about -0.08 for every hedged policy. It excludes the option premium, so it isn't a hedge-quality metric.

**The protocol.** Each policy's deterministic action over the same 2,000 simulated episodes (`--seed 999 --n_episodes 2000`). `test-hedge-legacy` reseeds `np.random` from `--seed` right before evaluation (`scripts/test_hedge_legacy.py`), so every policy sees the same paths and differences are not path-sampling noise. RL rows give the range over 4 training seeds (42, 1, 2, 3) of 1,000 episodes each, with the `train-hedge-legacy` defaults unless stated.

## Results

### At the default 0.1% cost

```
python main.py test-hedge-legacy --policy rl
python main.py test-hedge-legacy --policy delta
python main.py test-hedge-legacy --policy no_hedge
```

| Policy | Put: std P&L, ×K | Put: 95% CVaR, ×K | Call: std P&L, ×K | Call: 95% CVaR, ×K |
|---|---|---|---|---|
| **RL (default)** | **0.0089-0.0096** | **-0.100 to -0.103** | 0.0142-0.0149 | -0.112 to -0.113 |
| Delta hedge (LSM delta) | 0.0119 | -0.110 | **0.0142** | **-0.112** |
| No hedge | 0.0725 | -0.186 | 0.154 | -0.571 |

Protocol: 0.1% proportional cost, 2,000 test episodes, 4 training seeds.

### At a 1% proportional cost

```
python main.py train-hedge-legacy --prop_tc 0.01 --action_mode gap_fraction --val_metric cvar
python main.py test-hedge-legacy  --prop_tc 0.01 --policy rl
python main.py test-hedge-legacy  --prop_tc 0.01 --policy delta_band --band 0.1 --band_target edge
```

| Policy | Call: 95% CVaR, ×K | Put: 95% CVaR, ×K |
|---|---|---|
| Daily delta | -0.198 | -0.208 |
| RL, residual action (`--residual_scale 0.25`) | -0.173 to -0.186 | -0.140 to -0.153 |
| RL, gap-fraction action | -0.142 to -0.147 | -0.135 to -0.144 |
| **Band 0.1, rebalance to edge** | **-0.136** | **-0.130** |

Protocol: 2,000 test episodes, 4 training seeds. The baselines over a grid of costs are in [hedging-costs.md](hedging-costs.md).

### The learned band, with and without a fixed cost

```
python main.py train-hedge-legacy --prop_tc 0.01 --fixed_tc 0.05 --action_mode band --val_metric cvar
python main.py test-hedge-legacy  --prop_tc 0.01 --fixed_tc 0.05 --policy rl
```

| Policy | 1%: Call | 1%: Put | 1% + $0.05: Call | 1% + $0.05: Put |
|---|---|---|---|---|
| Daily delta | -0.198 | -0.208 | -0.380 | -0.390 |
| RL, gap-fraction action | -0.143 to -0.147 | -0.135 to -0.144 | -0.324 to -0.329 | -0.325 to -0.332 |
| Best fixed band | -0.136 (0.1, edge) | -0.130 (0.1, edge) | -0.148 (0.175, delta) | -0.146 (0.175, delta) |
| **RL, learned band** | **-0.136 to -0.137** | **-0.123 to -0.128** | **-0.145 to -0.147** | **-0.132 to -0.138** |

Protocol: 95% CVaR ×K, 2,000 test episodes, 4 training seeds of 1,000 episodes. "1% + $0.05" adds a $0.05 fixed cost per trade to the 1% proportional cost.

## Discussion

**At a low cost, RL beats delta on the put and matches it on the call.** On the put, every seed has a 19-25% lower hedging error than delta and a better tail. It even beats hedging with the exact binomial-tree delta (std 0.012-0.013). On the call, training never improved on the untrained policy, which already hedges at delta.

**Two design choices got it there.**
- **Residual action.** The policy learns a correction to the LSM delta rather than the whole hedge ratio, so an untrained policy starts at the delta hedge.
- **Validation checkpointing.** The checkpoint is the policy with the lowest P&L std on 1,000 held-out paths, with the untrained policy as a candidate. Selecting on the recent training reward mostly measured which paths were drawn. It left the policies behind delta (call std 0.017-0.026, put 0.013-0.020), and a plain non-residual policy did worse still (call 0.029-0.036). Full numbers are in `.claude/BUGS.md`, "RL hedging policy trails the LSM delta hedge".

**Under costs, the band is the bar, not daily delta.** With larger costs, rebalancing to delta every day spends most of the P&L on small trades. RL beats daily delta, and the gap-fraction action helps most, since holding still is easy to represent there. With the residual action, holding means tracking a moving target exactly, and every miss is a paid trade.

**The action's form was the limit, not the objective.** With the gap-fraction action, RL still trails the hand-tuned band by 0.006-0.009, even with 3,000 training episodes. It learned to close part of the gap every day, a smoothed delta hedge, rather than the band's "sit still, then act", and trades about twice as much. Giving the policy the band's form closes that gap with the same SAC objective. So the per-step mean-variance reward, judged on terminal CVaR, was not what held it back (`.claude/BUGS.md`, "RL policy under trading costs keeps trading every day").

**The learned band matches or beats the best fixed band in every cell.** It matches it for calls at 1% and beats it elsewhere, by up to 0.014 (puts with a fixed cost), on every seed, without any band width or target to tune. A constant action is the fixed band, so RL starts from the baseline's form and can learn a band that varies with time to expiry, moneyness and gamma.

**A fixed cost breaks the gap-fraction action.** It can only hold still in the limit, so its "holds" are tiny trades that each pay the fixed cost, and it ends up barely better than daily delta. The band action holds exactly and is unaffected.

## Limits

- **The two accounting bugs above distort every number**, and most of RL's edge on puts is likely the carry from the missing cash account.
- **Why the put policy beats even the exact delta is a hypothesis, not yet verified.** Episodes end at the LSM's exercise time, so the liability hedged is the put under LSM's slightly suboptimal exercise rule. A learned hedge can adapt to that rule, and a model delta cannot.
