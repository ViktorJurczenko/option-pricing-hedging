# CLAUDE.md

Guidance for agents working in this repository. Module-level detail lives in `docs/architecture.md`; read the part you need there before changing a module.

## The project

A from-scratch option pricing and hedging research project in Python/PyTorch (Python 3.13, PyTorch 2.11 CPU; `pip install -r requirements.txt`, no GPU needed):
- **European options:** closed-form Black-Scholes and Monte-Carlo.
- **American options:** binomial trees (CRR, and a vectorized BBSR batch pricer), Longstaff-Schwartz least-squares Monte-Carlo (LSM), and an MLP surrogate for American puts trained on tree prices (`PutSurrogate`; the older LSM-trained MLP is kept as `legacy`).
- **Hedging:** a SAC reinforcement-learning agent that delta-hedges an American option in a custom environment, measured against analytic baselines (delta, no-trade band, no hedge).

Everything runs through `python main.py <command>` from the project root. Each command's flags and logic live in `scripts/<name>.py`, and `--help` lists the flags.
- `price`: compare pricers.
- `train-ml` / `test-ml`: train and evaluate the put surrogate. Its dataset comes from `python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`.
- `train-hedge` / `test-hedge`: train and evaluate a hedging policy.

Layout: the library is the `optionpricing/` package (`core/`, `pricing/european/`, `pricing/american/`, `hedging/`, `ml/surrogate/`, `ml/rl/`, `ml/experiments/`; `pip install -e .` makes it importable from anywhere). `scripts/` holds the CLI commands, `tests/` the suite, `docs/` the architecture reference and results. Generated outputs (datasets, checkpoints, logs, runs) go to `artifacts/`, personal notes are in `notes/`; both are gitignored.

## How to work here

- **Work directly in the main checkout**, no worktrees.
- **Plan first.** Before a new feature or issue, add an entry to `.claude/PLAN.md`, the roadmap. Its header gives the format.
- **Log every bug** found, with symptom, root cause and fix, in `.claude/BUGS.md`. Its header gives the format.
- **Keep the living docs up to date in the same change.** A change is not done while one of them disagrees with the code:
  - `.claude/PLAN.md`: tick finished steps, archive finished workstreams.
  - `.claude/BUGS.md`: every bug found.
  - `README.md`: usage, results and layout.
  - `notes/REMINDER.md`: the theory and design chapter of anything whose design changed. It is gitignored personal notes, so code and tracked docs never cite it.
- **Git.**
  - Conventional Commits: `type(scope): description`, lowercase, imperative, subject <= 50 characters, body on why.
  - Types: feat, fix, refactor, perf, test, docs, style, chore.
  - Scopes: pricing, surrogate, rl, hedging, cli, tests, docs, tooling.
  - Commit on `main`. Print the full message in the reply and wait for explicit confirmation before `git commit`. Push only when asked.
- **Done bar.** A change is done only when:
  1. new or changed behaviour has tests (see Tests);
  2. `python -m pytest tests/` is green;
  3. the living docs are updated;
  4. `review-agent` has reviewed the change and its verdict has been relayed to the user. Its suggestions are proposed, not applied, until the user says go.

## Skills and agents

- `write-tests` skill: whenever you write or change tests.
- `optionpricing-docs` skill: whenever you write or change a doc, a docstring, a BUGS.md or PLAN.md entry, or a results write-up.
- `review-agent` (loads the `review-changes` skill): dispatch it on every change before committing, with a self-contained description of what changed (point it at `git diff`) and the exact test commands you ran on the final diff with their results (e.g. `python -m pytest tests/unit` -> 674 passed). It reuses those instead of rerunning them, and runs only what they don't cover; say so explicitly when no code changed and no run is needed.
- `test-agent`: only when the user asks for it.
- Both agents pin their model in their frontmatter. Never pass `model` when dispatching them.
- `.claude/rules/` holds path-scoped rules (e.g. progress bars for long loops) that apply automatically on matching files.

## Tests

Production code ships with tests that make sense and are useful:
- Test behaviour through the public interface, not the implementation. A test that only restates the code is not useful.
- Cover the edge cases: empty input, boundaries of each domain (zero, one element, the limits of a range), and invalid input raising `ValueError` with a useful message.
- Compare numerical code against an independent reference (a closed form, a finer tree, a finite difference), with a tolerance explained in a comment.
- Every fixed bug gets a regression test that fails without the fix.
- Seed every random component, so a failure reproduces.
- Reuse `tests/helpers.py` (and `tests/unit/hedge_fakes.py`) rather than redefining fakes.

```
python -m pytest tests/unit          # fast
python -m pytest tests/integration   # convergence checks and CLI smoke tests (main.py as a subprocess)
python -m pytest tests/              # everything
```

## Conventions

- Run from the project root. Run modules with `python -m package.module`, never by file path: imports are absolute against the root.
- Validate arguments eagerly with `ValueError` (e.g. `option_type` in `{call, put}`), never with silent fallbacks.
- Paths (datasets, checkpoints) are CLI and constructor arguments with relative defaults, never hardcoded.
- Pricers are stateful: `predict_payoff()` before `calculate_option_price()`.
- **Docstrings:** every module and public function or class has one. Start with prose on what it does and why, then `Args:`/`Returns:`/`Raises:` when it takes arguments.
- **Comments** explain why, not what. A measured number in a comment cites where it was measured: a `.claude/BUGS.md` entry or a tracked doc.

## Where to read more

- `docs/architecture.md`: every module, the command reference, the test layout, and design notes such as the softplus reparameterization pattern for ML targets.
- `README.md`: usage and measured results.
- `.claude/BUGS.md`: why things are the way they are, incident by incident.
