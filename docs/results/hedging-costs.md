# Analytic hedging baselines under trading costs

Where daily delta hedging overtrades, and what a no-trade band buys. Moved from `.claude/PLAN.md` ("Hedge under trading costs"). Measured 2026-10-01, **not re-verified** since. Every number here is distorted by the environment's two accounting bugs, a hedge that earns one step late and a P&L without a cash account (`.claude/BUGS.md`, B-037 and B-038). They will be re-measured by the hedging rebuild (`.claude/PLAN.md`, "Hedging rebuild").

```
python main.py test-hedge-legacy --option_type {call,put} --seed 999 --n_episodes 2000 \
    --prop_tc <tc> --fixed_tc <$> --policy delta
python main.py test-hedge-legacy ... --policy delta_band --band <b> --band_target {delta,edge}
```

Protocol: 2000 episodes with seed 999, call and put. Metric: 95% CVaR of the terminal P&L, per unit of strike (higher is better). Each band column gives the best band width and target found.

| Regime | Call: delta | Call: best band | Put: delta | Put: best band |
|---|---|---|---|---|
| tc 0.1% | -0.112 | -0.110 (0.02 edge) | -0.110 | -0.105 (0.05 edge) |
| tc 0.5% | -0.148 | -0.123 (0.05 edge) | -0.152 | -0.119 (0.05 edge) |
| tc 1% | -0.198 | -0.136 (0.1 edge) | -0.208 | -0.130 (0.1 edge) |
| tc 2% | -0.302 | -0.156 (0.1 edge) | -0.322 | -0.151 (0.1 edge) |
| fixed $0.01 | -0.149 | -0.116 (0.05 delta) | -0.145 | -0.112 (0.1 delta) |
| fixed $0.05 | -0.295 | -0.126 (0.1 delta) | -0.290 | -0.125 (0.1 delta) |

- **At the default 0.1% cost a band barely helps**, which matches the RL policy learning nothing beyond delta for calls.
- **From 0.5% upward, daily delta is badly beaten.** At 2% its put CVaR (-0.322) is worse than not hedging at all (-0.186).
- **Band-edge wins under proportional costs, rebalance-to-delta under fixed costs**, as expected, and the best band widens with cost (0.05-0.1).
- **That width matters for the residual policy:** with `--residual_scale 0.1` it can only stay 0.1 from delta by saturating tanh, so cost regimes need a wider residual (0.2-0.3).

The RL results under costs (`gap_fraction`, `band`) are in `.claude/BUGS.md`, "RL policy under trading costs keeps trading every day; a simple band beats it", and `hedging-rl.md`, "Hedging under trading costs".
