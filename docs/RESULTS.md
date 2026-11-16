# Results

Every result of the project on one page: one section per topic, with its takeaway, a key table where one helps, and a link to the full analysis. The topic files analyse each result and its limits, and most end with the commands that reproduce them.

The pricer comparison uses `S0 = K = 100`, `r = 5%`, `T = 1 year`, `σ = 20%`. The other topics give their protocol in their own file or under their table.

## Pricers agree with each other

European Monte-Carlo tracks Black-Scholes, and American Longstaff-Schwartz tracks the binomial tree within Monte-Carlo noise. With a European control variate, LSM prices the American call at exactly its Black-Scholes value.

Full results: [README](../README.md#pricing-models-agree-with-each-other).

## LSM delta and gamma up to expiry

The textbook LSM delta, the derivative of the regression fit, left the valid range up to 91.6% of the time just before maturity. A separate regression of pathwise deltas cut the out-of-the-money delta error from 0.2-0.9 to under 0.02.

Full results: [README](../README.md#lsm-delta-and-gamma-accurate-up-to-expiry).

## A neural pricer for American puts

How the MLP surrogate's configuration was chosen, decision by decision, from the teacher tree to the network's width.

| Puts on the scoring grid | Points | Mean error, 10⁻⁴ $ | Worst error, 10⁻⁴ $ | IV error, bp |
|---|---|---|---|---|
| near the money (`\|z\|` < 1) | 646 | 7.9 ± 0.3 | 166 ± 34 | 0.33 ± 0.01 |
| all | 2342 | 4.5 ± 0.4 | 166 ± 34 | 0.44 ± 0.04 |

Protocol: 5 seeds of the final configuration, scored against an independent 4000-step tree. Errors are at K = 100.

The typical error is $0.00045, under a twentieth of a cent. In batches, the surrogate prices at 6-9 µs per option, 360-510× faster than the 1000-step tree it learned from.

Full analysis: [surrogate-training.md](results/surrogate-training.md).

## The legacy surrogate

The first surrogate, trained on Longstaff-Schwartz prices, before the tree-trained one replaced it.

| Moneyness bucket | ITM (S < K) | ATM | OTM (S > K) |
|---|---|---|---|
| Current checkpoint, mean error against a fresh LSM run on the fixed horizon | 0.57% | 3.82% | 6.32% |

Continuous moneyness sampling, a smooth target and more paths per label took the in-the-money error from 9.5% to 0.78%. The same checkpoint scores 0.57% on the benchmark with the fixed horizon. Part of the remaining error is a label bias from a simulator bug, since fixed.

Full analysis: [legacy-surrogate.md](results/legacy-surrogate.md).

## RL hedging (legacy SAC)

The SAC agent against delta hedging and no-trade bands. These numbers are distorted by two accounting bugs in the legacy environment, and the hedging rebuild will replace them.

| Policy, 1% cost + $0.05 per trade | Call: 95% CVaR, ×K | Put: 95% CVaR, ×K |
|---|---|---|
| Daily delta | -0.380 | -0.390 |
| Best fixed band | -0.148 | -0.146 |
| **RL, learned band** | **-0.145 to -0.147** | **-0.132 to -0.138** |

Protocol: 2,000 test episodes, RL ranges over 4 training seeds.

A learned no-trade band matches or beats the best hand-tuned band in both cost regimes tested, without any tuning. At a low cost, RL has a 19-25% lower hedging error than delta hedging on the put.

Full analysis: [hedging-rl.md](results/hedging-rl.md), and the baselines alone over a grid of costs in [hedging-costs.md](results/hedging-costs.md).

## Deep Hedging rebuild (in progress)

The design of the new hedger: one differentiable P&L, a no-transaction band network around the surrogate delta, and exercise dates from the tree's boundary. Its checks so far cover the dataset, the baselines and the training loop. Results against the baselines come with the evaluation step.

Full design: [hedging-training.md](results/hedging-training.md).
