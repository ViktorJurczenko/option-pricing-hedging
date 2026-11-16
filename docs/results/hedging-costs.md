# Analytic hedging baselines under trading costs

Where daily delta hedging overtrades, and what a no-trade band buys.

**Status.** Measured 2026-10-01, **not re-verified** since. Every number here is distorted by the legacy environment's two accounting bugs, a hedge that earns one step late and a P&L without a cash account (`.claude/BUGS.md`, B-037 and B-038). The hedging rebuild will re-measure them (`.claude/PLAN.md`, "Hedging rebuild").

## Setup

The question is how much a no-trade band improves on daily delta hedging as trading costs grow. Two baselines hedge a short American call and put in the legacy environment:
- **delta:** rebalance to the LSM delta every day.
- **band:** trade only when the position leaves a band of half-width `b` around delta, either back to delta or to the band's edge.

The metric is the 95% CVaR of the terminal P&L, per unit of strike. Higher is better.

## Results

```
python main.py test-hedge-legacy --option_type {call,put} --seed 999 --n_episodes 2000 \
    --prop_tc <tc> --fixed_tc <$> --policy delta
python main.py test-hedge-legacy ... --policy delta_band --band <b> --band_target {delta,edge}
```

| Regime | Call: delta | Call: best band | Put: delta | Put: best band |
|---|---|---|---|---|
| tc 0.1% | -0.112 | -0.110 (0.02 edge) | -0.110 | -0.105 (0.05 edge) |
| tc 0.5% | -0.148 | -0.123 (0.05 edge) | -0.152 | -0.119 (0.05 edge) |
| tc 1% | -0.198 | -0.136 (0.1 edge) | -0.208 | -0.130 (0.1 edge) |
| tc 2% | -0.302 | -0.156 (0.1 edge) | -0.322 | -0.151 (0.1 edge) |
| fixed $0.01 | -0.149 | -0.116 (0.05 delta) | -0.145 | -0.112 (0.1 delta) |
| fixed $0.05 | -0.295 | -0.126 (0.1 delta) | -0.290 | -0.125 (0.1 delta) |

Protocol: 2000 episodes with seed 999, for the call and the put. Each band cell gives the best band width and target found.

## Discussion

- **At the default 0.1% cost a band barely helps.** This matches the RL policy learning nothing beyond delta for calls.
- **From 0.5% upward, daily delta is badly beaten.** At 2%, its put CVaR is -0.322, worse than not hedging at all (-0.186).
- **Band-edge wins under proportional costs, rebalance-to-delta under fixed costs**, as expected. The best band widens with cost, from 0.05 to 0.1.
- **That width matters for the residual RL policy.** With `--residual_scale 0.1` it can only stay 0.1 from delta by saturating tanh, so cost regimes need a wider residual (0.2-0.3).

## Limits

- The two accounting bugs above distort every number, until the rebuild re-measures them.
- "Best band" is the best of the widths and targets tried, not an optimum.

The RL results under costs (`gap_fraction`, `band`) are in [hedging-rl.md](hedging-rl.md), "Hedging under trading costs", and `.claude/BUGS.md`, "RL policy under trading costs keeps trading every day; a simple band beats it".
