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
- `train-hedge` — train the RL delta-hedging policy (`AmericanHedging/`). Flags: `--n_episodes`, `--warmup_steps`, `--update_every`, `--grad_steps_per_update` (gradient steps per triggered update, default `1`), `--replay_buffer_size` (default `100_000`), `--fixed_cost` (flat per-trade cost added to the hedging env, default `0.0`), `--risk_lambda` (mean-variance reward-shaping coefficient, default `0.0`), plus the usual pricing params (`--sigma`, `--strike`, `--reg_type`, etc.) used to build the GBM simulation and the Longstaff-Schwartz model the hedging environment wraps. Saves to `--checkpoint_path` (default `checkpoints/best_policy_model.pth`).
- `test-hedge` — evaluate a hedging policy: mean P&L, std P&L, and 95% CVaR over `--n_episodes` simulated hedging episodes. `--policy {rl,delta,no_hedge}` (default `rl`) selects between the trained RL policy (loaded from `--checkpoint_path`), an analytic delta-hedge baseline, and a no-hedge baseline (`AmericanHedging/baselines.py`) — the latter two need no checkpoint. `--fixed_cost` (default `0.0`) should match the value used for the `rl` policy's training run for a fair comparison across `--policy` values. Reseeds `np.random` from `--seed` right before evaluation so runs with the same `--seed` across `--policy` values see identical simulated paths, for a fair comparison.

Dataset generation for the MLP surrogate is a separate entry point and **must be run as a module, not a script path** (it uses absolute imports like `from core.gbm import GBM`, which only resolve when the project root is on `sys.path` — `python -m` does this, `python datasets/generate_dataset.py` does not):

```
python -m datasets.generate_dataset [--path_per_sim N] [--nb_param_comb N] [--samples_per_param N] [--nb_jobs N] [--sampling N] [--option_type call|put] [--reg_type simple|laguerre] [--max_degree N] [--output_dir data]
```

It parallelizes LSM pricing (via `joblib`) across randomly sampled `(S0, K, sigma, r, T)` combinations to build `<output_dir>/american_option_dataset.npz`, along with the feature-normalization stats in `<output_dir>/features_norm.npz` that both the dataset class and `test-ml` rely on.

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

- `core/` — the foundation every pricer depends on: `gbm.py` (`GBM`, the Geometric Brownian Motion path simulator whose `simulate_paths` returns a `(n_steps, n_paths)` array) and `base.py` (`OptionPricing`, an ABC with one abstract method `calculate_option_price(T)`). `BlackScholes`, `MonteCarlo`, `BinomialTree`, and `LongstaffSchwartz` all subclass it.
- `EuropeanOptionPricing/` — `BlackScholes` (closed form) and `MonteCarlo` (path-average payoff), both taking `option_type` in `{call, put}` (validated eagerly in `__init__`).
- `AmericanOptionPricing/`:
  - `BinomialTree` — Cox-Ross-Rubinstein tree with backward induction; American exercise is baked into the payoff recursion via `np.maximum(exercise, continuation)`. `calculate_option_price()` requires `predict_payoff()` to have been called first (raises `AttributeError` otherwise — there's no silent default).
  - `LongstaffSchwartz` — the core LSM implementation. `predict_payoff()` walks the simulation backward; at each timestep `fit()` regresses continuation value on in-the-money paths only (`itm_mask`), using either simple polynomial or Laguerre polynomial basis functions (from `basis_functions.py`) of degree `max_degree`. It also stores per-path, per-timestep Greeks (`self.delta`, `self.gamma`), obtained by evaluating the regression model's *analytic derivative* basis rather than finite differences — this is exercised by `tests/unit/test_basis_functions.py`, which checks the analytic derivatives against finite differences directly. `self.exercise_time` records each path's optimal exercise index, consumed by `AmericanHedging/HedgeEnv.py` to decide episode length. `calculate_option_price` ignores its `T` argument by design (the American price comes from the simulated payoff at t=0, not from an external time-to-maturity).
  - `basis_functions.py` — `simple_poly`/`simple_poly_deriv_k` and `laguerre_poly_gen`/`laguerre_deriv_k`, factored out of `LongstaffSchwartz` so the basis math (and its derivatives) can be unit-tested independent of the pricing class.
  - `ML.py` — `MLP` (4 inputs: `S0/K`, normalized T, normalized r, normalized sigma → 1 output: **time value** `(price - smooth_intrinsic_value) / K`, not raw price) trained as a fast surrogate for LSM. Predicting the residual over intrinsic value shrinks the target's dynamic range (deep ITM prices dwarf deep OTM ones), but `smooth_intrinsic_value` is a *softplus* approximation of `intrinsic_value`, not the exact `max(K-S,0)`/`max(S-K,0)` — see "Softplus reparameterization" below for why. `beta_frac` (default `0.06`) is a fixed constant baked into the function, not a CLI flag, to avoid it drifting out of sync between label generation and reconstruction. `train_model` does early stopping on validation MSE (`--patience`, default `10`) and checkpoints to `checkpoint_path` (default `checkpoints/best_model.pth`). Anywhere that needs an actual price back from the model's output must add `smooth_intrinsic_value(...)` (see `evaluation.py::test_option_price_lsm`).
  - `evaluation.py` — `regression_tests`/`test_option_price_lsm`, used by `scripts/test_ml.py` (see Testing above). Its `reg_type`/`max_degree` must match the dataset's LSM settings — see "Working with this code" below.
- `AmericanHedging/` — a SAC (Soft Actor-Critic)-style continuous-control RL setup for delta hedging:
  - `HedgeEnv.py` — `HedgingEnv` wraps a *fitted* `LongstaffSchwartz` instance (must call `predict_payoff()` first). State = `[S_t/K, t/T, delta, gamma, position]`; episodes run per-path from `t=0` to that path's `exercise_time`; reward is hedge P&L minus transaction cost, minus the option payoff at exercise, all scaled by strike. Transaction cost has two components: `transaction_cost` (proportional, `tc * abs(trade) * S_t`) and `fixed_cost` (flat, default `0.0`, charged whenever `abs(trade)` exceeds `_TRADE_EPS` so floating-point noise around holding the same position isn't charged).
  - `RLPolicy.py` — Gaussian policy network (actor): outputs `(mean, log_std)` → `torch.distributions.Normal`, action = hedge ratio clamped to `[-1, 1]`.
  - `Critic.py` — twin Q-networks (state+action → scalar), the standard SAC double-critic setup.
  - `ReplayBuffer.py` — plain deque-based replay buffer; `max_size` (default `100_000`) is exposed via `train-hedge --replay_buffer_size`.
  - `Hedge.py` — `train()`: twin critics with frozen target copies (Polyak-averaged via `tau`) computing the Bellman target, a warmup phase of random actions (`warmup_steps`) before the policy starts driving exploration, gradient updates every `update_every` environment steps, and a checkpoint saved whenever the trailing 50-episode mean reward improves. `grad_steps_per_update` (default `1`, must be `>= 1`) runs that many critic/actor gradient steps — each on a freshly sampled batch — per triggered update. `risk_lambda` (default `0.0`) shapes the reward the critics/actor actually train on via `reward - risk_lambda * reward**2` (a mean-variance-style penalty that penalizes large-magnitude rewards symmetrically, not just downside); `episode_reward` and the checkpoint criterion stay on the raw, unshaped reward so training progress remains comparable to `test-hedge`'s P&L. `evaluate(action_fn, env, n_episodes)` is policy-agnostic - `action_fn` is a plain `state -> float` callable, not tied to `RLPolicy`'s Gaussian-actor `.mean` interface (the caller wraps a trained policy in a lambda, e.g. `scripts/test_hedge.py`; a baseline from `baselines.py` already matches the signature directly). Returns `(mean_pnl, std_pnl, cvar_95)` in addition to printing them.
  - `baselines.py` — non-learned reference policies matching `evaluate()`'s `action_fn` signature: `delta_hedge` (hedge ratio = the LSM's own analytic delta, i.e. state index 2 - the "trust the model's Greek, no learning" baseline) and `no_hedge` (always 0). Used via `test-hedge --policy {delta,no_hedge}` to give the trained RL policy something to be measured against.
- `datasets/` — `american_option_dataset.py` (`AmericanOptionDataset`, a `torch.utils.data.Dataset` doing a seeded shuffle-then-split so train/test constructed with the same seed are guaranteed disjoint) and `generate_dataset.py` (parallel LSM-based dataset generation script — see Running Things above for its module-invocation requirement). `gen_rdm_params` samples moneyness (`S0/K`) log-uniformly and continuously over `moneyness_range` (default `(0.4, 1.6)`) rather than from a handful of discrete buckets — discrete buckets leave untrained gaps between them that the MLP then has to extrapolate across.
- `data/` — generated artifacts only (`american_option_dataset.npz`, `features_norm.npz`), gitignored, not a Python package.
- `checkpoints/` — generated model weights only (`best_model.pth`, `best_policy_model.pth`), gitignored, not a Python package.
- `scripts/` — one module per CLI subcommand (`price.py`, `train_ml.py`, `test_ml.py`, `train_hedge.py`, `test_hedge.py`), each exposing `add_arguments(parser)` and `run(args)`. `main.py` just wires these into `argparse` subparsers.
- `utils/metrics.py` — generic, model-agnostic metrics (`mse_fn`, `mae_fn`, `r2_fn`, `accuracy_fn`) used by `ML.py`/`evaluation.py`.

## Working with this code

- Always run scripts from the project root — imports are absolute against the repo root (e.g. `from core.gbm import GBM`), not package-relative. `python -m datasets.generate_dataset` is required (not `python datasets/generate_dataset.py`) for the same reason.
- `LongstaffSchwartz.predict_payoff()` (and `BinomialTree.predict_payoff()`) must be called before `calculate_option_price()` — pricers are stateful, not pure functions of the constructor args.
- When touching normalization logic, keep `datasets/generate_dataset.py`'s normalization (log-sigma, log-T, raw-r) consistent with `scripts/test_ml.py` and `datasets/american_option_dataset.py` — they must agree for the MLP's inputs to be meaningful.
- `scripts/test_ml.py`'s `--reg_type`/`--max_degree` (default `laguerre`/`3`) must match whatever `datasets/generate_dataset.py` was run with to produce the dataset the checkpoint was trained on — they're independent CLI flags with no shared source of truth, so if you regenerate the dataset with different LSM settings, update `test_ml.py`'s defaults (or pass matching flags) too.
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
