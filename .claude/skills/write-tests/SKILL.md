---
name: write-tests
description: Use when writing or expanding automated tests in this repo — new unit tests, integration tests, e2e/smoke tests, or a regression test for a bug that was just fixed. Applies whether invoked directly or dispatched to a subagent whose job is specifically to write tests. Trigger on "write tests", "add test coverage", "test this module", "add a regression test".
---

# Writing tests for this project

A companion `review-tests` skill (for reviewing tests/PRs) will be added later
— this skill only covers *writing* tests. Follow this order:

## 1. Unit tests first, and make them real

Write unit tests before reaching for integration/e2e. Maximize coverage of the
unit under test, but coverage is a byproduct, not the goal — a test that
doesn't fail when the logic it targets is broken is worse than no test,
because it inflates confidence for free. Concretely:

- For every test, ask "what specific behavior breaks if this test is
  deleted?" If you can't name one, rewrite or drop it.
- Avoid tautological assertions (asserting a constant you just set, asserting
  only "no exception was raised" when the function returns a value worth
  checking, asserting on a mock's own return value instead of the code under
  test's behavior).
- Cover edge cases and boundaries, not just the happy path: empty inputs, zero
  and negative values, off-by-one boundaries, the function's stated
  preconditions being violated (does it raise the documented error?).
- For numeric code, check invariants and known closed-form relationships, not
  just "close to a hardcoded number" — e.g. this repo's
  `test_basis_functions.py` checks analytic derivatives against finite
  differences, and `test_pricing_convergence.py` checks Monte-Carlo/LSM
  converge to Black-Scholes/the binomial tree, rather than pinning a single
  magic number. Prefer that style over hardcoded expected values when a
  principled relationship exists to check against instead.
- Seed anything stochastic (RNG, path simulation, weight init, data
  shuffling) so a failure is reproducible and not a coin flip. If you're
  unsure how, mirror how `.claude/CLAUDE.md`'s "Softplus reparameterization"
  section and this repo's training scripts already do it (`torch.manual_seed`,
  `torch.Generator().manual_seed(seed)`, `np.random.default_rng(seed)`).
- If you can mutate the logic slightly (flip a comparison, change a constant)
  and no test fails, that's a real gap — treat it the same as a coverage gap
  even though line-coverage tools won't flag it.

## 2. Never hit the network — use test doubles

If the code under test does any network I/O (HTTP calls, external APIs,
sockets, anything that could flake, be slow, cost money, or require
credentials in CI), the test must not perform it for real:

- Prefer a fake (a lightweight in-memory stand-in with real behavior, e.g. a
  fake in-memory repository) over a mock when the interaction has real logic
  worth exercising; use a mock/stub when you only need to assert "this was
  called with X" or force a specific response/error path.
- Patch at the boundary the code actually calls (the HTTP client, the SDK
  method) rather than deep internals, so the test still exercises real
  logic up to that boundary.
- Add at least one test double that simulates failure (timeout, 4xx/5xx,
  malformed response) — the failure path is usually less tested than the
  happy path and is exactly where production bugs hide.
- If a test unexpectedly needs real network/credentials to pass, that's a
  sign the boundary was mocked at the wrong layer — fix the seam, don't
  suppress the test.

(Nothing in this repo currently does live network I/O — this section is
here for when that changes, e.g. a future data source or API integration.)

## 3. Keep the suite organized

Mirror source layout so anyone can find what tests what without searching:

- One test file per source module: `AmericanOptionPricing/foo.py` →
  `tests/unit/test_foo.py`. This repo already follows this convention
  (`test_gbm.py`, `test_black_scholes.py`, `test_longstaff_schwartz.py`, …) —
  keep new tests consistent with it rather than dumping multiple modules'
  tests into one file or one giant test into multiple files.
- Test function names should state the behavior being checked, not just the
  method name: `test_calculate_option_price_raises_before_predict_payoff`,
  not `test_calculate_option_price_2`.
- Shared setup (seeded simulations, temp checkpoint dirs, common fixtures)
  belongs in a `conftest.py` (this repo doesn't have one yet — add one under
  `tests/` or `tests/unit/` the first time you'd otherwise copy-paste setup
  across 3+ test files, rather than duplicating it further).
- `tests/unit/` = fast, isolated, no simulation/training-heavy work.
  `tests/integration/` = multiple components together, allowed to be slower.
  Don't add a slow test to `tests/unit/` just because it's convenient —
  that's how a fast test suite rots.

## 4. Regression tests for fixed bugs

Whenever a bug is flagged and then fixed:

- Add a test that reproduces the original failure mode and would have caught
  it before the fix. Name it so the bug is traceable later, e.g.
  `test_itm_otm_buckets_not_swapped_for_put_default` — describe the wrong
  behavior being guarded against, not just "regression_1".
- Put it in the test file for the module that actually had the bug (per the
  organization rule above), not a generic `test_regressions.py` dumping
  ground — that defeats "keep track of what code is where."
- A one-line comment on the test explaining what broke and why (not what the
  test does mechanically — that's what the assertions already say) is worth
  the exception to the "don't over-comment" default, since the *why* here is
  genuinely non-obvious from the code alone and is exactly what stops the
  same bug from being reintroduced silently.

## Suggestions for integration tests

Integration tests exercise multiple real components together, catching wiring
bugs unit tests (which import modules directly and mock collaborators) miss.
For this repo, candidates beyond what's already in `tests/integration/`:

- **Pipeline-level checks**: dataset generation → training → evaluation on a
  tiny synthetic dataset (a handful of samples, few epochs) to verify the
  full `generate_dataset.py` → `train_ml.py` → `test_ml.py` path stays wired
  correctly end-to-end when any one of the three changes shape (feature
  count, normalization keys, checkpoint format).
  `tests/integration/test_bundled_dataset.py` is a starting point for this.
- **Cross-module consistency**: a test that fails loudly if
  `datasets/generate_dataset.py`'s normalization (log-sigma, log-T, raw-r)
  and `datasets/american_option_dataset.py`/`scripts/test_ml.py`'s ever
  diverge — this is currently only enforced by a `CLAUDE.md` convention, not
  a test, and is exactly the kind of silent-mismatch bug integration tests
  exist for.
- **Hedging environment against a real fitted pricer**: `HedgingEnv` wrapping
  an actual (small, seeded) `LongstaffSchwartz` instance rather than a fake,
  checking episode length matches `exercise_time` and reward arithmetic holds
  over a full simulated episode, not just the isolated reward formula
  (already unit-tested in `test_hedge_env.py`).

## Suggestions for e2e tests

E2e tests exercise the product the way a user actually invokes it — for a
CLI tool like this, that means the CLI itself, not internal function calls.
`tests/integration/test_cli_smoke.py` already does this per-subcommand; ways
to extend it:

- **Failure-path CLI smoke tests**: invalid `--option_type`, missing
  `--checkpoint_path`, a `test-ml` run before `train-ml` has ever produced a
  checkpoint — assert the CLI exits non-zero with a clear error, not a stack
  trace, mirroring the happy-path smoke tests that already exist.
- **Full workflow in one process**: `train-ml` → `test-ml` → assert the
  printed metrics are within a sanity bound (not pinned exactly, since
  training is stochastic even when seeded across PyTorch versions — bound
  loosely, e.g. R² above some low floor) using a small bundled dataset, so a
  wiring break between training and evaluation is caught without a 40-minute
  full dataset regeneration.
- **`train-hedge` → `test-hedge` smoke run** with a tiny `--n_episodes` and
  few warmup steps, asserting it completes and prints the three metrics
  (mean P&L, std P&L, CVaR) without erroring — this exercises the full SAC
  loop wiring (env, actor, twin critics, replay buffer) together, which no
  current test does.

## Shared fixtures (`conftest.py`)

There's no `conftest.py` yet. Add one — under `tests/` for suite-wide fixtures
or `tests/unit/`/`tests/integration/` for scope-specific ones — the first time
you'd otherwise copy-paste setup (a seeded simulation, a temp checkpoint dir,
a small fitted `LongstaffSchwartz`) across 3+ test files. Don't create it
speculatively before that threshold — an empty or single-use `conftest.py` is
indirection with nothing behind it yet.

## Measure coverage, don't assume it (`pytest-cov`)

"Most coverage possible" isn't checkable by eye once the suite grows. Run
tests with `pytest --cov=<package> --cov-report=term-missing` and look at the
uncovered lines it reports before deciding a module is done. Treat the number
as a gap-finder, not a target to game — 100% line coverage with vacuous
assertions (see the unit-test section above) is worse than 80% coverage where
every test would actually fail on a real bug. Don't add a coverage threshold
that fails the build on it — that belongs with CI wiring, which isn't in
scope yet.

## Mutation testing as a check on test quality (`mutmut`)

Line/branch coverage only tells you code *ran* during a test, not that a test
would *catch* a bug in it — which is exactly the "make sure tests actually
test something" requirement from the top of this skill. Mutation testing
answers that mechanically: it flips small things in the source (a `<` to
`<=`, a `+` to `-`, a constant off by one) and reruns the suite; a "surviving
mutant" (tests still pass) is a concrete, automatically-found instance of a
test gap, not a guess.

Reach for `mutmut` on a module when:
- it's numerically/logically dense and a subtle wrong constant or comparison
  would be easy to miss by inspection (basis functions, LSM regression/fit
  logic, the hedging environment's reward arithmetic are good candidates),
  or
- you've just written unit tests for something and want a second check
  before moving on, beyond your own "what breaks if I delete this test?"
  reasoning.

Don't run it suite-wide by default — it reruns the test suite once per
mutant, so it's slow; point it at one module at a time.

## Property-based testing for numeric code (`hypothesis`)

Hand-picked `(S0, K, σ, r, T)` test points only check the specific values you
thought to try. For pricing/Greeks code, prefer checking a relationship that
must hold over a *range* of inputs instead, and let `hypothesis` generate the
range:

- Put-call parity across generated `(S0, K, r, T, σ)` within sane bounds
  (positive prices, `T > 0`, `σ > 0`).
- Monotonicity: call price non-decreasing in `S0` and `σ`; put price
  non-increasing in `S0`.
  American option price ≥ its European counterpart (early-exercise premium is
  never negative).
- Delta bounded in `[-1, 0]` for puts / `[0, 1]` for calls across generated
  inputs, not just at one `(S0, K)` pair.

Use this alongside, not instead of, the specific closed-form/convergence
checks already in the unit-test section — hypothesis is for invariants that
should hold *everywhere*, not a replacement for a targeted regression test
pinned to one bug.

## Flakiness policy for stochastic code

This codebase is inherently stochastic (GBM path simulation, LSM regression
noise, RNG-seeded model init/data shuffling/training), so "flaky" here
usually means "insufficiently seeded" or "tolerance picked by feel" rather
than genuine nondeterminism. Rule to follow:

- Any test that touches randomness (path simulation, weight init, data
  splits/shuffling) must fix every seed involved — mirror the pattern already
  used in this repo's training/eval code (`torch.manual_seed`,
  `torch.Generator().manual_seed(seed)`, `np.random.default_rng(seed)`,
  `GBM(..., seed=...)` where supported). A test that only sometimes fails is
  not allowed to ship; find and fix the missing seed rather than adding a
  retry or loosening the tolerance to make it pass.
- Any numeric tolerance (`atol`/`rtol`, a percent-error bound) must be
  justified by something — a known convergence rate (Monte Carlo error
  shrinks with `1/sqrt(n_paths)`), a closed-form bound, or an explicit prior
  benchmark run — stated in a comment next to the assertion. "Whatever number
  made it pass locally" is not a tolerance, it's a guess that will silently
  stop meaning anything the next time the underlying implementation changes.
- If a test needs an unusually loose tolerance to pass reliably, that's a
  signal to increase the sample size/path count in the test (slower but
  tight) rather than widen the tolerance (fast but meaningless) — unless the
  test is explicitly a smoke test where "did it run and land in the right
  ballpark" is the actual intent.
