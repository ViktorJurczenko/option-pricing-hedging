# Architecture

This page is the map of the code: the packages, what each file is for, and where its reasons are explained. Signatures and shapes are in each module's docstrings, and defaults in the signatures. `python main.py <command> --help` lists every flag, and the parser of each script in `scripts/` holds its defaults. Setup (Python 3.13, PyTorch 2.11 CPU, no GPU needed) is in the README.

```
optionpricing/          the library, one importable package (pip install -e .)
  core/                 GBM simulator, pricer base class, generic metrics, keep-awake
  pricing/european/     Black-Scholes, Monte-Carlo
  pricing/american/     binomial trees, Longstaff-Schwartz, basis functions
  hedging/              Deep Hedging building blocks (rebuild in progress)
  hedging/legacy/       the SAC hedger's environment, baselines and evaluation
  ml/surrogate/         MLP surrogate pricer: tree/ (current), legacy/ (LSM-trained)
  ml/hedging/           Deep Hedging pipeline: scenario bank, band network, training
  ml/hedging/legacy/    SAC agent
  ml/experiments/       research drivers (ablation runner)
scripts/                one module per main.py command
tests/                  unit/, integration/, mutation/, helpers.py
docs/                   this page, RESULTS.md, results/
main.py                 the CLI entry point
artifacts/              generated data, checkpoints, logs, runs (gitignored)
notes/                  personal notes (gitignored)
```

Dependencies point one way: `core` <- `pricing` <- `hedging` <- `ml`. `scripts/` and `main.py` sit on top, and nothing in the package imports them or `tests/`. Pricing is split by option class, everything else by role. The SAC stack sits in `legacy/` subpackages while the hedger is rebuilt as Deep Hedging.

## core/

The pieces every pricer shares.

| File | Role | Explained in |
|---|---|---|
| [gbm.py](../optionpricing/core/gbm.py) | `GBM`, risk-neutral price paths for one contract or many, time last | |
| [base.py](../optionpricing/core/base.py) | `OptionPricing`, the base class of the stateful pricers | |
| [metrics.py](../optionpricing/core/metrics.py) | generic regression metrics, used by the legacy surrogate | |
| [keep_awake.py](../optionpricing/core/keep_awake.py) | `keep_awake`, holds Windows and the display awake during a long run: every `main.py` command, the hedging bank generator and the surrogate's ablations | [B-032](../.claude/BUGS.md#b-032-overnight-ablation-run-suspended-by-modern-standby) |

## pricing/european/

| File | Role | Explained in |
|---|---|---|
| [black_scholes.py](../optionpricing/pricing/european/black_scholes.py) | closed-form price and Greeks, vectorized, and the `BlackScholes` class | |
| [monte_carlo.py](../optionpricing/pricing/european/monte_carlo.py) | `MonteCarlo`, the average discounted payoff over GBM paths | [README](../README.md#pricing-models-agree-with-each-other) |

## pricing/american/

| File | Role | Explained in |
|---|---|---|
| [binomial_tree.py](../optionpricing/pricing/american/binomial_tree.py) | `BinomialTree`, a Cox-Ross-Rubinstein tree for one option | [README](../README.md#pricing-models-agree-with-each-other) |
| [batch_binomial_tree.py](../optionpricing/pricing/american/batch_binomial_tree.py) | `price_put_batch` (CRR, BBS, BBSR trees for many puts at once) and `exercise_boundary` | [surrogate-training.md](results/surrogate-training.md) |
| [longstaff_schwartz.py](../optionpricing/pricing/american/longstaff_schwartz.py) | `LongstaffSchwartz`, least-squares Monte-Carlo with its exercise rule, control variate and delta regression | [README](../README.md#lsm-delta-and-gamma-accurate-up-to-expiry) |
| [basis_functions.py](../optionpricing/pricing/american/basis_functions.py) | the regression bases (simple and Laguerre polynomials) and their derivatives | |

The stateful pricers need `predict_payoff()` before `calculate_option_price()`.

## hedging/

The Deep Hedging building blocks. Each one is a file, and training, evaluation and the baselines all go through the same P&L.

| File | Role | Explained in |
|---|---|---|
| [simulator.py](../optionpricing/hedging/simulator.py) | `ScenarioBank`, seeded put contracts and their daily price paths | [hedging-training.md](results/hedging-training.md) |
| [exercise.py](../optionpricing/hedging/exercise.py) | the day each holder exercises: optimal (tree), late, LSM or never | [hedging-training.md](results/hedging-training.md) |
| [pnl.py](../optionpricing/hedging/pnl.py) | the terminal P&L of a hedged short put, the one place the accounting lives, with `HedgeSettings` (how a hedge is booked) and `HedgeOutcome` | [hedging-training.md](results/hedging-training.md) |
| [costs.py](../optionpricing/hedging/costs.py) | proportional and fixed transaction costs | |
| [settlement.py](../optionpricing/hedging/settlement.py) | settlement at exercise or expiry, in cash or by delivery | |
| [risk.py](../optionpricing/hedging/risk.py) | risk measures of the P&L, and `RiskMeasure`, the one training and testing are judged on | |
| [baselines.py](../optionpricing/hedging/baselines.py) | baseline policies with the network's interface | |

## hedging/legacy/

The environment of the SAC hedger. Its P&L has two known bugs ([B-037](../.claude/BUGS.md#b-037-hedgingenv-applies-each-hedge-one-step-late) and [B-038](../.claude/BUGS.md#b-038-hedging-pl-has-no-cash-account-so-the-delta-baseline-hedges-the-wrong-liability)), which is why the hedger is being rebuilt.

| File | Role | Explained in |
|---|---|---|
| [env.py](../optionpricing/hedging/legacy/env.py) | `HedgingEnv`, a Gym-style environment around a fitted `LongstaffSchwartz` | [hedging-rl.md](results/hedging-rl.md) |
| [option_value.py](../optionpricing/hedging/legacy/option_value.py) | option value functions (Black-Scholes, binomial) for the environment | |
| [baselines.py](../optionpricing/hedging/legacy/baselines.py) | delta hedge, no-trade band, no hedge | [hedging-costs.md](results/hedging-costs.md) |
| [evaluation.py](../optionpricing/hedging/legacy/evaluation.py) | `evaluate` and `cvar`: P&L statistics over episodes | |

## ml/surrogate/

| File | Role | Explained in |
|---|---|---|
| [chunks.py](../optionpricing/ml/surrogate/chunks.py) | resumable, chunked generation shared by every dataset generator | |
| [tree/generate_tree_dataset.py](../optionpricing/ml/surrogate/tree/generate_tree_dataset.py) | the tree-labelled put dataset, Sobol-sampled over a fixed domain | [surrogate-training.md](results/surrogate-training.md) |
| [tree/tree_dataset.py](../optionpricing/ml/surrogate/tree/tree_dataset.py) | `TreePutDataset`, features and targets built at load time | |
| [tree/tree_model.py](../optionpricing/ml/surrogate/tree/tree_model.py) | `MLP`, the network predicting the early-exercise premium | [surrogate-training.md](results/surrogate-training.md) |
| [tree/tree_pricer.py](../optionpricing/ml/surrogate/tree/tree_pricer.py) | `PutSurrogate`, price, delta and gamma from the network, with routing outside the domain | |
| [tree/tree_training.py](../optionpricing/ml/surrogate/tree/tree_training.py) | `train_surrogate`, what `train-ml` runs | [surrogate-training.md](results/surrogate-training.md) |
| [tree/tree_evaluation.py](../optionpricing/ml/surrogate/tree/tree_evaluation.py) | what `test-ml` reports, against an independent finer tree | [surrogate-training.md](results/surrogate-training.md) |
| [legacy/](../optionpricing/ml/surrogate/legacy) | the earlier LSM-trained surrogate: dataset, model, training, evaluation. No CLI command uses it | [legacy-surrogate.md](results/legacy-surrogate.md) |

## ml/hedging/

| File | Role | Explained in |
|---|---|---|
| [generate_hedge_dataset.py](../optionpricing/ml/hedging/generate_hedge_dataset.py) | the bank of hedging scenarios: paths, exercise days, surrogate Greeks | [hedging-training.md](results/hedging-training.md) |
| [hedge_dataset.py](../optionpricing/ml/hedging/hedge_dataset.py) | loads a bank, refuses banks that don't belong together, and hedges it (`Hedger`, `hedge_batch`, `hedge_bank`) | [hedging-training.md](results/hedging-training.md) |
| [hedge_model.py](../optionpricing/ml/hedging/hedge_model.py) | the no-transaction band network around the surrogate delta | |
| [hedge_training.py](../optionpricing/ml/hedging/hedge_training.py) | Deep Hedging training: minimize a risk measure of the P&L | |
| [hedge_evaluation.py](../optionpricing/ml/hedging/hedge_evaluation.py) | the network and every baseline scored on one test bank, the bands tuned on validation | [hedging-training.md](results/hedging-training.md) |
| [legacy/](../optionpricing/ml/hedging/legacy) | the SAC agent: policy, critics, replay buffer, training loop | [hedging-rl.md](results/hedging-rl.md) |

## ml/experiments/

| File | Role | Explained in |
|---|---|---|
| [run_ablations.py](../optionpricing/ml/experiments/run_ablations.py) | the surrogate's seeded ablations, one change per variant | [surrogate-training.md](results/surrogate-training.md) |

## Commands

Everything runs from the project root, since imports are absolute (`from optionpricing.core.gbm import GBM`). Generators run as modules (`python -m ...`), never by file path, for the same reason.

| Command | Script | What it does |
|---|---|---|
| `python main.py price` | [price.py](../scripts/price.py) | compares the pricers on one option |
| `python main.py train-ml` | [train_ml.py](../scripts/train_ml.py) | trains the put surrogate, one flag per `TrainConfig` field |
| `python main.py test-ml` | [test_ml.py](../scripts/test_ml.py) | evaluates it on the test split and on a grid against a reference tree |
| `python main.py train-hedge` | [train_hedge.py](../scripts/train_hedge.py) | trains the Deep Hedging band network, one flag per `HedgeTrainConfig` field |
| `python main.py test-hedge` | [test_hedge.py](../scripts/test_hedge.py) | scores the network and the baselines on a test bank, with the bands tuned on a validation bank |
| `python main.py train-hedge-legacy` | [train_hedge_legacy.py](../scripts/train_hedge_legacy.py) | trains the SAC hedger |
| `python main.py test-hedge-legacy` | [test_hedge_legacy.py](../scripts/test_hedge_legacy.py) | evaluates the SAC hedger or a baseline |
| `python -m optionpricing.ml.surrogate.tree.generate_tree_dataset` | | writes `artifacts/data/tree_put_dataset.npz` |
| `python -m optionpricing.ml.hedging.generate_hedge_dataset` | | writes `artifacts/data/hedge_<split>.npz`, ~5 min on 6 cores at defaults |
| `python -m optionpricing.ml.surrogate.legacy.generate_lsm_dataset` | | writes the legacy surrogate's dataset |

[hedge_flags.py](../scripts/hedge_flags.py) holds the flags `train-hedge` and `test-hedge` share, one per field of `HedgeSettings` and `RiskMeasure`.

`--split_ratio` and `--seed` must match between `train-ml` and `test-ml`. `test-hedge` reads the cost and risk flags from the network's checkpoint unless they are given, so by default it tests at the training settings. The cost flags and the Longstaff-Schwartz settings must match between `train-hedge-legacy` and `test-hedge-legacy` for a fair comparison. The generators resume after an interruption, and refuse to resume with other settings.

## Tests

| Folder | What it covers |
|---|---|
| `tests/unit/` | each component alone: closed forms, derivatives against finite differences, datasets, pricers, training loops, hooks. Fast |
| `tests/integration/` | components together: Monte-Carlo and LSM convergence, the baselines on tree deltas, a CLI smoke test per command. Slower |
| `tests/mutation/` | hand mutation checks: `python -m tests.mutation.tree_evaluation_mutants` plants bugs in `tree_evaluation.py`, and its tests must catch each one |
| `tests/unit/hedging_legacy/`, `tests/integration/hedging_legacy/` | the SAC stack, with shared fakes in `hedge_fakes.py` |

[helpers.py](../tests/helpers.py) holds what several test files share, such as a fake fitted LSM, the scripts' parsed defaults, subprocess runs of the CLI, a small tree dataset and an untrained surrogate checkpoint. Reuse it rather than redefine a fake.

## Design note: a target that subtracts an analytic part

**The pattern.** A common trick defines a model's target as `true_value - analytic_component`, to shrink its range or remove a known effect (the legacy surrogate's `price - intrinsic_value`). If the analytic part has a kink where the true value doesn't, the subtraction stamps that kink onto the target. The model's error then spikes around it, which looks like a hard region but is self-inflicted.

**The fix.** Replace the analytic part by a smooth version that matches it away from the kink (a softplus, `smooth_intrinsic_value` in [lsm_model.py](../optionpricing/ml/surrogate/legacy/lsm_model.py)). Use the same smooth version wherever the part is added back (`lsm_evaluation.py::test_option_price_lsm`), or predictions are biased. Pick the smoothing width by a seeded sweep against a fixed benchmark.

**When it applies again.** Before subtracting an analytic function from a target, check whether it has a kink the real target doesn't. The incident is [B-001](../.claude/BUGS.md#b-001-ml-surrogate-pricers-target-had-a-self-inflicted-kink-at-atm). The tree surrogate avoids the issue by predicting the early-exercise premium, American minus European, which has no kink at the strike.
