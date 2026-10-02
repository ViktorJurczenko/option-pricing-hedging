# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch option pricing and hedging research project in Python/PyTorch. It implements:
- European option pricing: closed-form Black-Scholes and Monte-Carlo simulation.
- American option pricing: Binomial tree, Longstaff-Schwartz (LSM) least-squares Monte-Carlo, and an MLP trained to approximate LSM prices.
- American option delta-hedging via reinforcement learning (a SAC actor-critic agent trained inside a custom hedging environment).

## Setup

```
pip install -r requirements.txt
```

Pinned versions match the environment this was developed against (Python 3.13, PyTorch 2.11 CPU build). No GPU is required; `torch.cuda.is_available()` is checked at runtime in the training loops and falls back to CPU automatically.

## Running things

Everything is driven through `main.py`, a thin `argparse` subcommand dispatcher. Each subcommand's flags and logic live in `scripts/<name>.py`; `main.py` itself contains no pricing/training logic. Run from the project root so the absolute module imports resolve (e.g. `from core.gbm import GBM`):

```
python main.py <command> [options]
python main.py --help                 # lists all commands
python main.py <command> --help        # lists that command's flags
```

Commands:
- `price` — compare pricing models. `--option_class {EU,US,both}` picks Black-Scholes+Monte-Carlo vs Binomial-tree+Longstaff-Schwartz vs both. Also: `--option_type {call,put}`, `--strike`, `--start_price`, `--sigma`, `--risk_free_rate`, `--maturity`, `--sampling`, `--nb_predictions`, `--nb_steps`, `--reg_type {simple,laguerre}`, `--max_degree`, `--plot`.
- `train-ml` — train the MLP surrogate pricer (`AmericanOptionPricing/ML.py`) against `data/american_option_dataset.npz`. Key flags: `--dataset_path`, `--split_ratio`, `--batch_size`, `--epochs`, `--patience`, `--checkpoint_path` (default `checkpoints/best_model.pth`).
- `test-ml` — evaluate the trained MLP: regression metrics against the held-out split, plus an ITM/ATM/OTM error breakdown against a fresh Longstaff-Schwartz benchmark. Loads from `--checkpoint_path`, normalization stats from `--features_norm_path` (default `data/features_norm.npz`).
- `train-hedge` — train the RL delta-hedging policy (`AmericanHedging/`). Flags: `--n_episodes`, `--warmup_steps`, `--update_every`, `--grad_steps_per_update` (gradient steps per triggered update, default `1`), `--lr` (Adam learning rate for actor + critics, default `1e-4`), `--alpha` (SAC entropy coefficient, default `0.2`), `--replay_buffer_size` (default `100_000`), `--prop_tc` (proportional cost `tc * |trade| * S_t`, default `0.001`), `--fixed_tc` (flat per-trade cost added to the hedging env, default `0.0`), both applied to the training and the validation env, `--risk_lambda` (mean-variance reward-shaping coefficient applied to the *scaled* reward, default `1.0` — must be `> 0` for the policy to have any reason to hedge, see `BUGS.md`), `--reward_mode {terminal,accounting}` (default `accounting`), `--gamma` (SAC discount, default `1.0`), `--reward_scale` (default `100`), `--auto_alpha/--no-auto_alpha` (learned entropy coefficient, default on; `--alpha` is then its initial value, `--alpha_lr` its learning rate, defaulting to `--lr`), `--val_paths` (default `1000`; held-out paths the checkpoint is selected on, `0` falls back to the trailing-50 training reward), `--val_every` (default `50`), `--val_metric {std,cvar}` (default `std`), `--val_seed` (default `--seed + 1000`; the validation env is built before the training simulation because `GBM` reseeds the global `np.random`), `--action_mode {residual,gap_fraction,band}` (default `residual`; see `RLPolicy.py` below - `gap_fraction` or `band` are the ones to use under trading costs; `band` builds a 2-D policy and matching critics), `--band_max` (default `0.2`, the widest no-trade band half-width in `band` mode; saved in the checkpoint), `--residual_scale` (default `0.1`: the policy learns a correction to the LSM delta, `hedge = clip(delta + residual_scale * tanh(u), -1, 1)`; `0` makes it output the hedge ratio directly; saved in the checkpoint), plus the usual pricing params (`--sigma`, `--strike`, `--reg_type`, etc.) used to build the GBM simulation and the Longstaff-Schwartz model the hedging environment wraps (`--sigma` is also passed to it, for the Black-Scholes delta feature of its delta regression; its European control variate is enabled for `--option_type call` only). Saves to `--checkpoint_path` (default `checkpoints/best_policy_model.pth`).
- `test-hedge` — evaluate a hedging policy: mean P&L, std P&L, 95% VaR and 95% CVaR over `--n_episodes` simulated hedging episodes. `--policy {rl,delta,delta_band,no_hedge}` (default `rl`) selects between the trained RL policy (loaded from `--checkpoint_path`), an analytic delta-hedge baseline, a delta hedge with a no-trade band (`--band`, default `0.05`; `--band_target {delta,edge}`, default `delta`), and a no-hedge baseline (`AmericanHedging/baselines.py`) — the baselines need no checkpoint. `--prop_tc`, `--fixed_tc` and the Longstaff-Schwartz params (same defaults as `train-hedge`) should match the values used for the `rl` policy's training run for a fair comparison across `--policy` values (`residual_scale`, `band_max`, the action mode and the action dimension need no flag: they are read from the checkpoint, the dimension via `RLPolicy.checkpoint_action_dim`). Reseeds `np.random` from `--seed` right before evaluation so runs with the same `--seed` across `--policy` values see identical simulated paths, for a fair comparison.

Dataset generation for the MLP surrogate is a separate entry point and **must be run as a module, not a script path** (it uses absolute imports like `from core.gbm import GBM`, which only resolve when the project root is on `sys.path` — `python -m` does this, `python datasets/generate_dataset.py` does not):

```
python -m datasets.generate_dataset [--path_per_sim N] [--nb_param_comb N] [--samples_per_param N] [--nb_jobs N] [--sampling N] [--option_type call|put] [--reg_type simple|laguerre] [--max_degree N] [--output_dir data]
```

It parallelizes LSM pricing (via `joblib`) across randomly sampled `(S0, K, sigma, r, T)` combinations to build `<output_dir>/american_option_dataset.npz`, along with the feature-normalization stats in `<output_dir>/features_norm.npz` that both the dataset class and `test-ml` rely on.

## Planning

Before tackling a new issue or feature, draft a short plan: list the candidate approaches/steps, in the order you'd tackle them, and pick a path. Record that plan in `.claude/PLAN.md` before writing code — a running log, newest entry on top, one entry per issue/feature.

Each step starts as `- [ ] Step title` with a short description underneath explaining what it does and why. When a step is completed, drop the description and change it to `- [x] Step title` — the file stays a compact log of what's actually been done, not an accumulating pile of stale planning notes.

## Bug tracking

Every problem/bug encountered while working on this project must be transcribed into `.claude/BUGS.md`: a symptom/root-cause/fix (or current status, if unresolved) write-up, following the existing entries' format. New feature implementation that isn't a bug or a fix does not belong there — that's what `.claude/PLAN.md` is for.

## Testing

```
python -m pytest tests/                 # everything
python -m pytest tests/unit             # fast, no training/simulation-heavy work
python -m pytest tests/integration      # convergence checks + CLI smoke tests; slower
```

- `tests/unit/` tests individual components in isolation (closed-form formulas, basis-function derivatives against finite differences, dataset splitting, the hedging environment's reward arithmetic, etc.).
- `tests/integration/` checks components together: Monte-Carlo converging to Black-Scholes, Longstaff-Schwartz converging to the Binomial tree for American puts, and `test_cli_smoke.py`, which actually invokes `python main.py <command>` as a subprocess for each command — this is what catches import/wiring mistakes across `main.py` → `scripts/` → the library packages that unit tests (which import modules directly) would not.
- `AmericanOptionPricing/evaluation.py` holds `regression_tests`/`test_option_price_lsm`, which `scripts/test_ml.py` calls as part of the `test-ml` CLI workflow. These are evaluation helpers used by production code, not pytest tests — don't confuse them with `tests/`.

### Workflow: delegate tests and review after every implementation

Whenever you (the coding/general-purpose agent) implement a non-trivial
change in this repo, do not write its tests yourself and do not self-review
it. Instead:

1. Dispatch `test-agent` (`.claude/agents/test-agent.md`) to write the tests.
   This repo **is** a git repository, so point it at `git diff`/`git status`
   to discover what changed rather than hand-listing files — only fall back
   to explicitly listing files for uncommitted work outside git's view
   (gitignored artifacts, or changes made earlier in the session that you
   don't want bundled into the diff it reads). Either way, describe the
   behavior in enough detail that a fresh agent with no conversation context
   can act on it correctly.
2. `test-agent` loads the `write-tests` skill itself, writes and runs the
   tests, then spawns/pings `review-agent` (`.claude/agents/review-agent.md`)
   with the same kind of self-contained file list, which loads the
   `review-changes` skill and reports OK/ISSUE/verdict.
3. Relay the reviewer's verdict to the user yourself; don't consider the
   change done until that review has actually run. If `test-agent` reports it
   couldn't reach `review-agent`, dispatch `review-agent` yourself rather
   than skipping the review.

This applies even when a change feels small enough to test inline — the
point of the pattern is that it doesn't depend on your own judgment call
about "small enough" in the moment.

## Architecture

**Module layout mirrors option class for pricing, and role for everything else:**

- `core/` — the foundation every pricer depends on: `gbm.py` (`GBM`, the Geometric Brownian Motion path simulator whose `simulate_paths` returns a `(int(maturity * sampling) + 1, n_paths)` array: row 0 is the start price and the last row sits exactly at `maturity`, so rows are `maturity / (n_rows - 1)` apart — every consumer (`LongstaffSchwartz`, `option_value.py`, `HedgingEnv`) uses that spacing and treats the last row as expiry) and `base.py` (`OptionPricing`, an ABC with one abstract method `calculate_option_price(T)`). `BlackScholes`, `MonteCarlo`, `BinomialTree`, and `LongstaffSchwartz` all subclass it.
- `EuropeanOptionPricing/` — `BlackScholes` (closed form) and `MonteCarlo` (path-average payoff), both taking `option_type` in `{call, put}` (validated eagerly in `__init__`).
- `AmericanOptionPricing/`:
  - `BinomialTree` — Cox-Ross-Rubinstein tree with backward induction; American exercise is baked into the payoff recursion via `np.maximum(exercise, continuation)`. `calculate_option_price()` requires `predict_payoff()` to have been called first (raises `AttributeError` otherwise — there's no silent default).
  - `LongstaffSchwartz` — the core LSM implementation. `predict_payoff()` walks the simulation backward; at each timestep `fit()` regresses continuation value on in-the-money paths only (`itm_mask`), using either simple polynomial or Laguerre polynomial basis functions (from `basis_functions.py`) of degree `max_degree`. It also stores per-path, per-timestep Greeks (`self.delta`, `self.gamma`) from a *separate* regression, over all paths (not just ITM), of each path's pathwise delta — the derivative of its discounted cashflow under the fitted exercise rule with respect to `S_t`, i.e. `±S_stop/S_t` discounted if the path pays off, else 0. Gamma is the analytic derivative of that fit (the basis derivatives are checked against finite differences in `tests/unit/test_basis_functions.py`). When `sigma` is given, the European Black-Scholes delta is an extra feature of that regression (and its gamma the matching derivative feature): a low-degree polynomial alone can't follow delta's steepening near expiry. Differentiating the ITM-only continuation fit instead — the original approach — extrapolates to OTM paths and gave deltas in the tens; ridge/moneyness-bin regressions tried as a fix for that made delta ~0 and caused spurious early exercise, and were removed (see `BUGS.md`). `control_variate` (default `False`, requires `sigma`) uses the European option as a control variate: the continuation regression targets the realized cashflow minus the realized discounted European payoff, and the exact Black-Scholes price is added back (also in `calculate_option_price`). Non-early-exercised paths then contribute exactly 0, which removes the regression noise that makes plain LSM spuriously early-exercise American calls near expiry (a ~0.1 continuation-vs-intrinsic margin against ~0.15 fit noise). It's off by default because for puts the plain rule was as good or better out of sample; `train-hedge`/`test-hedge` enable it for calls only. `self.exercise_time` records each path's exercise index under the fitted rule, consumed by `AmericanHedging/HedgeEnv.py` to decide episode length. `calculate_option_price` ignores its `T` argument by design (the American price comes from the simulated payoff at t=0, not from an external time-to-maturity).
  - `basis_functions.py` — `simple_poly`/`simple_poly_deriv_k` and `laguerre_poly_gen`/`laguerre_deriv_k`, factored out of `LongstaffSchwartz` so the basis math (and its derivatives) can be unit-tested independent of the pricing class.
  - `ML.py` — `MLP` (4 inputs: `S0/K`, normalized T, normalized r, normalized sigma → 1 output: **time value** `(price - smooth_intrinsic_value) / K`, not raw price) trained as a fast surrogate for LSM. Predicting the residual over intrinsic value shrinks the target's dynamic range (deep ITM prices dwarf deep OTM ones), but `smooth_intrinsic_value` is a *softplus* approximation of `intrinsic_value`, not the exact `max(K-S,0)`/`max(S-K,0)` — see "Softplus reparameterization" below for why. `beta_frac` (default `0.06`) is a fixed constant baked into the function, not a CLI flag, to avoid it drifting out of sync between label generation and reconstruction. `train_model` does early stopping on validation MSE (`--patience`, default `10`) and checkpoints to `checkpoint_path` (default `checkpoints/best_model.pth`). Anywhere that needs an actual price back from the model's output must add `smooth_intrinsic_value(...)` (see `evaluation.py::test_option_price_lsm`).
  - `evaluation.py` — `regression_tests`/`test_option_price_lsm`, used by `scripts/test_ml.py` (see Testing above). Its `reg_type`/`max_degree` must match the dataset's LSM settings — see "Working with this code" below.
- `AmericanHedging/` — a SAC (Soft Actor-Critic)-style continuous-control RL setup for delta hedging:
  - `HedgeEnv.py` — `HedgingEnv` wraps a *fitted* `LongstaffSchwartz` instance (must call `predict_payoff()` first). State = `[S_t/K, t/(n_rows-1), delta, gamma, position]`; episodes run per-path from `t=0` to that path's `exercise_time`; reward is hedge P&L minus transaction cost, minus the option payoff at exercise, all scaled by strike. Transaction cost has two components: `prop_tc` (proportional, `tc * abs(trade) * S_t`) and `fixed_tc` (flat, default `0.0`, charged whenever `abs(trade)` exceeds `_TRADE_EPS` so floating-point noise around holding the same position isn't charged). `reward_mode` (default `"terminal"`, the behavior just described) can be set to `"accounting"`, which also subtracts the option's value change `V_{t+1} - V_t` every step (V from a `value_fn(t, S)`, required in that mode, with the payoff replacing V on the exercise step), so each step's reward is the local hedging error; the episode sum then equals the terminal-mode sum plus the constant `V_0/K`. `train-hedge` uses `accounting`; `test-hedge` and the training validation env keep `terminal` so the reported P&L is unchanged. `reset(path_idx=None)` starts on a given path (random if None). `rollout_pnl(hedge_fn, path_idxs)` computes the terminal-mode P&L of one episode per path with all paths stepped together, so a policy is called once per time step for the whole batch (`hedge_fn` maps `(n, 5)` states to `(n,)` hedges). It shares `_trading_cost`/`_payoff` and `_build_states` with `step()`/`_get_state()` and replicates the rest of `step()`'s terminal-mode arithmetic (the position held before a trade earns that step's move) in vectorized form; a unit test checks the two agree, so any other change to `step()`'s P&L must be mirrored there.
  - `option_value.py` — value functions on the simulation's time grid for the accounting reward: `black_scholes_value_fn` (exact for an American call without dividends), `binomial_value_fn` (American, a CRR tree with `substeps` per simulation step, interpolated in log-price per row, floored at intrinsic), and `make_option_value_fn`, which picks BS for calls and the tree for puts. `n_steps` there is the simulation's row count: row `t` has remaining time `(n_steps - 1 - t) * maturity / (n_steps - 1)`, the last row being expiry.
  - `RLPolicy.py` — Gaussian policy network (actor): outputs `(mean, log_std)` → `torch.distributions.Normal`. The action is this distribution's sample (or, for deterministic evaluation, its mean) squashed into `[-1, 1]` via `tanh` — see `Hedge.py`'s `_squash` for why a hard clamp isn't used instead. `hedge_ratio(state, squashed)` maps that action to the hedge ratio the env receives, depending on `action_mode`. In `"gap_fraction"` mode it is `position + w * (delta - position)` with `w = (squashed + 1) / 2` in (0, 1), the fraction of the gap to delta closed this step; the mean output's bias starts at `w ≈ 0.9` (`_INITIAL_GAP_FRACTION`) so the untrained policy is almost a delta hedge, and holding still (w ≈ 0) is easy to represent, which matters under trading costs (see `BUGS.md`). In `"band"` mode (`action_dim` must be 2) the squashed action `(a0, a1)` gives a no-trade band half-width `b = band_max * (a0 + 1) / 2` and a rebalancing target `lam = (a1 + 1) / 2` (0 = band edge, 1 = delta): the position is held *exactly* while `|position - delta| <= b`, otherwise the hedge is `delta + sign(position - delta) * (1 - lam) * b`, clipped to `[-1, 1]`. A constant action is `baselines.delta_band`, which a unit test checks; the biases start at `b = 0.1 * band_max` (`_INITIAL_BAND_FRACTION`) and `lam = 0.9` (`_INITIAL_BAND_TARGET`). The mode is read back from the buffers via the `action_mode` property. In `"residual"` mode (default) it is `clip(delta + residual_scale * squashed, -1, 1)`, with delta read from the state at `_DELTA_IDX` (must match `HedgingEnv._get_state`'s layout, which a unit test checks), or the action itself when `residual_scale == 0`. The critics, replay buffer and entropy term all work on the squashed action, not the hedge ratio; delta is in the state, so nothing is lost. `residual_scale`, `gap_fraction` (1.0 in gap-fraction mode) and `band_max` (0 outside band mode) are registered buffers, so they are saved in the checkpoint; checkpoints predating them load as `residual_scale = 0`, residual mode and not band mode. `checkpoint_action_dim(state_dict)` reads the action dimension off a checkpoint's output layer, so a loader can build the matching network first. Together with validation checkpointing (`Hedge.py`), the residual form gets the RL policy 20-25% below the LSM delta hedge's P&L std on puts and level with it on calls (see `BUGS.md`, "RL hedging policy trails the LSM delta hedge").
  - `Critic.py` — twin Q-networks (state+action → scalar), the standard SAC double-critic setup.
  - `ReplayBuffer.py` — plain deque-based replay buffer; actions are pushed as scalars or `(action_dim,)` vectors and sampled as `(batch, action_dim)`; `max_size` (default `100_000`) is exposed via `train-hedge --replay_buffer_size`.
  - `Hedge.py` — `_squash(dist)`: samples `dist` (reparameterized) and squashes it into `(-1, 1)` via `tanh`, applying the corresponding log-prob Jacobian correction. Used everywhere a sampled action feeds a critic (the Bellman target and the actor loss) instead of a hard `.clamp(-1, 1)` — a hard clamp has exactly zero gradient once a sample lands outside the range, which both freezes `mean` wherever it drifted to and lets the entropy term keep inflating `std` for free (nothing pulls a saturated policy back). `train()`: twin critics with frozen target copies (Polyak-averaged via `tau`) computing the Bellman target, a warmup phase of random actions (`warmup_steps`) before the policy starts driving exploration, gradient updates every `update_every` environment steps, and checkpointing. With a `val_env` (a terminal-mode `HedgingEnv` on held-out paths; `train-hedge` builds one unless `--val_paths 0`), the checkpoint is the policy with the best `val_metric` (`"std"`, the default, lower is better; or `"cvar"`, higher is better) over the terminal P&L of the first `val_paths` validation paths, i.e. the same numbers `test-hedge` reports. The untrained policy is scored and saved first (under the residual form it already hedges at delta), then the policy is re-scored every `val_every` episodes and after the last one, and replaces the checkpoint only on strict improvement. Without a `val_env`, it falls back to saving whenever the trailing 50-episode mean raw reward improves, which mostly measures which paths were drawn. `grad_steps_per_update` (default `1`, must be `>= 1`) runs that many critic/actor gradient steps — each on a freshly sampled batch — per triggered update. `reward_scale` (default `1.0`) multiplies the env reward first, and `risk_lambda` (default `0.0`) then shapes it: the critics/actor train on `x - risk_lambda * x**2` with `x = reward_scale * reward`. `auto_alpha` (default `False`) learns `log_alpha` toward `target_entropy` (default `None`, i.e. `-policy.action_dim`) instead of keeping `alpha` fixed. The `risk_lambda` penalty is a mean-variance-style penalty that penalizes large-magnitude rewards symmetrically, not just downside — it only penalizes *hedging error* when the env uses `reward_mode="accounting"` (in `terminal` mode it penalizes holding any position, see `BUGS.md`); `episode_reward` and the fallback checkpoint criterion stay on the raw, unshaped reward so training progress remains comparable to `test-hedge`'s P&L. `evaluate(action_fn, env, n_episodes)` is policy-agnostic - `action_fn` is a plain `state -> float` callable, not tied to `RLPolicy`'s Gaussian-actor `.mean` interface (the caller wraps a trained policy around `deterministic_hedge(policy, states)`, the shared deterministic action - `tanh(mean)` mapped through `hedge_ratio` - that validation also scores, e.g. `scripts/test_hedge.py`; a baseline from `baselines.py` already matches the signature directly). Returns `(mean_pnl, std_pnl, cvar_95)` in addition to printing them (plus the 95% VaR). `cvar(pnls, level=0.95)` is the expected shortfall, the mean of the worst 5% of P&Ls; before it existed, `evaluate()` reported the 5th percentile (the VaR) as "CVaR", so "CVaR" figures in older `BUGS.md` entries are VaRs (see `BUGS.md`).
  - `baselines.py` — non-learned reference policies matching `evaluate()`'s `action_fn` signature: `delta_hedge` (hedge ratio = the LSM's own analytic delta, i.e. state index 2 - the "trust the model's Greek, no learning" baseline) `no_hedge` (always 0), and `delta_band(band, target)`, a factory returning a no-trade-band policy: hold the position while `|delta - position| <= band`, otherwise rebalance to delta (`target="delta"`, suits fixed costs) or to the nearest band edge (`"edge"`, Whalley-Wilmott style, suits proportional costs). A hold must return the position *exactly*, since `HedgingEnv` only skips the fixed cost when `|trade| <= _TRADE_EPS`. Under trading costs `delta_band` is the bar RL has to beat, not daily delta. Used via `test-hedge --policy {delta,delta_band,no_hedge}` to give the trained RL policy something to be measured against.
- `datasets/` — `american_option_dataset.py` (`AmericanOptionDataset`, a `torch.utils.data.Dataset` doing a seeded shuffle-then-split so train/test constructed with the same seed are guaranteed disjoint) and `generate_dataset.py` (parallel LSM-based dataset generation script — see Running Things above for its module-invocation requirement). `gen_rdm_params` samples moneyness (`S0/K`) log-uniformly and continuously over `moneyness_range` (default `(0.4, 1.6)`) rather than from a handful of discrete buckets — discrete buckets leave untrained gaps between them that the MLP then has to extrapolate across.
- `data/` — generated artifacts only (`american_option_dataset.npz`, `features_norm.npz`), gitignored, not a Python package.
- `checkpoints/` — generated model weights only (`best_model.pth`, `best_policy_model.pth`), gitignored, not a Python package.
- `scripts/` — one module per CLI subcommand (`price.py`, `train_ml.py`, `test_ml.py`, `train_hedge.py`, `test_hedge.py`), each exposing `add_arguments(parser)` and `run(args)`. `main.py` just wires these into `argparse` subparsers.
- `utils/metrics.py` — generic, model-agnostic metrics (`mse_fn`, `mae_fn`, `r2_fn`, `accuracy_fn`) used by `ML.py`/`evaluation.py`.

## Working with this code

- Always run scripts from the project root — imports are absolute against the repo root (e.g. `from core.gbm import GBM`), not package-relative. `python -m datasets.generate_dataset` is required (not `python datasets/generate_dataset.py`) for the same reason.
- `LongstaffSchwartz.predict_payoff()` (and `BinomialTree.predict_payoff()`) must be called before `calculate_option_price()` — pricers are stateful, not pure functions of the constructor args.
- When touching normalization logic, keep `datasets/generate_dataset.py`'s normalization (log-sigma, log-T, raw-r) consistent with `scripts/test_ml.py` and `datasets/american_option_dataset.py` — they must agree for the MLP's inputs to be meaningful.
- `scripts/test_ml.py`'s `--reg_type`/`--max_degree` (default `simple`/`3`, like the dataset generator) must match whatever `datasets/generate_dataset.py` was run with to produce the dataset the checkpoint was trained on — they're independent CLI flags with no shared source of truth, so if you regenerate the dataset with different LSM settings, update `test_ml.py`'s defaults (or pass matching flags) too.
- `option_type` is validated as `{"call", "put"}` and `reg_type` as `{"simple", "laguerre"}` via `ValueError` in every pricer's constructor — follow that pattern rather than adding silent fallbacks.
- Checkpoint and dataset paths are constructor/CLI arguments with `checkpoints/`/`data/`-relative defaults, not hardcoded absolute paths — pass `--checkpoint_path`/`--dataset_path`/etc. rather than editing defaults in place if you need a different location for one run.
- Path-scoped conventions (e.g. progress-bar requirements for long-running loops) live in `.claude/rules/` and auto-apply on matching paths — check there for anything not covered in this file.
- Before running `git commit`, print the full drafted commit message as plain text in the response (not only inside the `git commit` tool call) so it can be reviewed in the CLI, and wait for explicit confirmation before executing it.

### Softplus reparameterization: general pattern for ML regression targets

When an ML model's *target* is defined as `true_value - analytic_component`
(a common trick to shrink dynamic range or remove a known effect before
regression, e.g. `time_value = price - intrinsic_value` here), check whether
`analytic_component` has a kink (a point where it's continuous but its slope
jumps) that `true_value` does *not* have at that same point. If so, subtracting
it stamps that kink onto the residual as an artifact — the model's error will
spike specifically in a band around that point, easy to mistake for "this
region is intrinsically harder to model" when it's actually self-inflicted by
the label definition.

This is exactly what happened with `smooth_intrinsic_value` in `ML.py` —
see `docs/decisions/softplus-reparameterization.md` for the full incident
(symptom, ablation, root cause).

**Fix pattern**: replace the exact analytic component with a *smooth*
approximation that converges to it away from the kink but rounds off the
corner over a small window (here, a softplus: `smooth_intrinsic_value` in
`AmericanOptionPricing/ML.py`). Use the *same* smooth version consistently
wherever the component is added back to reconstruct the true value
(`evaluation.py::test_option_price_lsm`) - training and reconstruction must
agree, or predictions will be systematically biased. The smoothing width is a
hyperparameter with a real trade-off (too little: kink barely smoothed, error
stays high near it; too much: the smoothing correction bleeds into regions
that didn't need it and degrades them instead) — pick it by sweeping a few
values against a fixed benchmark, with training runs fully seeded (model init,
data split, batch shuffling) so you're comparing the hyperparameter's effect,
not run-to-run training variance.

**When to reach for this pattern again**: any time you're about to subtract
(or divide by) an analytic function from an ML target to make it easier to
learn, ask "does this analytic function have a kink/discontinuity that the
real target doesn't have at that point?" before shipping it.
