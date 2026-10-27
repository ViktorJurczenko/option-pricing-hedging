# Notation and terminology

Use these symbols in prose and formulas, and the code names when you refer to code. Docs use Unicode symbols; code uses spelled-out snake_case names. When a doc shows both, map them explicitly: "(`sigma` in the code)".

Every entry was checked against the source when this table was written. Re-check any entry you rely on for a precise claim.

## Market and contract

| Symbol | Meaning | Code name(s) | Notes |
|---|---|---|---|
| `S`, `S_t` | Spot price (at time `t`) | `start_price` (the spot at t = 0), `spot` | README tables use `S0` for the starting spot |
| `K` | Strike | `strike` | |
| `T` | Maturity, in years | `maturity`, `T` | Remaining time `T − t` near expiry |
| `r` | Risk-free rate (continuous) | `risk_free_rate`, `r` | |
| `σ` | Volatility (annual standard deviation of log-returns) | `sigma` | |
| `q` | Dividend yield | not implemented | Only in PLAN.md's dividend entry |
| `m` | Moneyness `S/K` | `moneyness`, `m` | Everything in the surrogate and the batch tree is per unit of strike (K = 1) |
| `z` | Standardized log-moneyness `ln(m)/(σ√T)` | `z` | The tree dataset samples `z`, not `m` |
| `w` | `σ√T` | `sigma sqrt(T)`, a feature | Scales the shape-penalty terms |
| `V` | Option value | `price` | `V/K` when per unit of strike |
| `Δ`, delta | `∂V/∂S` | `delta` | The same number for every K |
| `Γ`, gamma | `∂²V/∂S²` | `gamma` | Stored as `K·∂²V/∂S²` in `TreeResult` and `SurrogateResult` |
| vega | `∂V/∂σ` | `black_scholes_vega`, `tree_vega` | |
| `dT` | `∂V/∂T` (minus theta) | `black_scholes_dT` | |
| payoff / intrinsic value | `max(K − S, 0)` (put), `max(S − K, 0)` (call) | `1 - m` for a put at K = 1 | |
| premium | Early-exercise premium `(American − European)/K` | `premium` | The tree MLP's target |

## Numerical methods

| Symbol | Meaning | Code name |
|---|---|---|
| `N` | Number of tree steps | `nb_steps` |
| `dt` | Time step `T/N` (tree), `maturity/(n_rows − 1)` (simulation) | `dt` |
| `u`, `d = 1/u` | Up and down factors, `u = e^{σ√dt}` | `u` |
| `p` | Risk-neutral up probability `(e^{r·dt} − d)/(u − d)` | `p` (`growth = e^{r·dt}`) |
| level `i` | Tree time `i·dt`, holding `i + 1` nodes `S_{i,j}`, `j = 0…i` | `level` |
| CRR / BBS / BBSR | Cox-Ross-Rubinstein; binomial Black-Scholes; BBS with Richardson `2P(N) − P(N/2)` | `method="crr" \| "bbs" \| "bbsr"` |
| paths | Simulated GBM paths | `nb_predictions` (CLI), `n_paths` |
| sampling | Time steps per year | `sampling` |
| `L_k` | Laguerre polynomial of degree `k` in `S/K` | `reg_type="laguerre"` |

## Hedging

| Symbol / term | Meaning | Code name |
|---|---|---|
| state | `[S_t/K, t/(n_rows−1), delta, gamma, position]` | `HedgingEnv._get_state` |
| `t_ex` | A path's exercise row under the fitted LSM rule (the episode ends there) | `lsm.exercise_time[path]`, `T_exercise` |
| position | Shares held (the hedge ratio), in `[−1, 1]` | `position` |
| `u` (RL) | The actor's raw Gaussian output. Not the tree's `u`: say which one you mean when both appear | |
| `a = tanh(u)` | Squashed action | `squashed` |
| `b` | No-trade band half-width | `band`, `band_max` |
| `λ` (band) | Rebalancing target inside the band (0 = edge, 1 = delta) | `lam` |
| `w` (gap fraction) | Fraction of the gap to delta closed per step. Not the shape-penalty `w = σ√T` | |
| P&L | Hedge cash flow minus the option payoff at exercise, reported ×K | |
| VaR 95% / CVaR 95% | 5th percentile / mean of the worst 5% of P&L | `cvar` |
| prop_tc / fixed_tc | Proportional cost `tc·\|trade\|·S_t` / flat cost per trade | `--prop_tc`, `--fixed_tc` |

## Terms to use consistently

- "Longstaff-Schwartz" or "LSM" (least-squares Monte-Carlo). "Binomial tree", not "lattice".
- "Continuation value", "exercise region", "exercised today" (priced at its payoff at t = 0).
- "Teacher" means the pricer that labels a dataset. "Reference" means the independent pricer used for scoring. Never score against the teacher.
- "Surrogate" means the MLP pricer. Use "tree surrogate" (`PutSurrogate`, `ML/surrogate/tree/`) and "legacy surrogate" (`LegacyMLP`, `ML/surrogate/legacy/`), never just "the MLP" when both could be meant.
- "Per unit of strike" means K = 1. "×K" means a P&L or price divided by K.

## Known overloads (always disambiguate)

- `u`: the tree's up factor, or the RL actor's raw output.
- `w`: `σ√T` in the shape penalty, or the gap fraction in RL.
- "CVaR" in BUGS.md entries older than the fix means the 95% VaR.
