# Examples to imitate

The best existing passage for each kind of doc. Search for the heading, since line numbers drift. Each one shows the default style of its guide. Where it departs from the guide, the note says so.

## Project md (`guides/project-md.md`)

| What | Where | What to imitate |
|---|---|---|
| Pitch | `README.md`, the opening paragraphs | Full sentences, "my first in quantitative finance", the from-scratch principle |
| Highlights | `README.md`, "Highlights" | Bold finding, then the number. Some bullets carry more numbers than the default of one: trim them when you touch them |
| Methods table | `README.md`, "Methods and references" | Method, use, paper |
| Result section | `README.md`, "Pricing models agree with each other" | Command, table, interpretation. Its "Rerun 2026-11-05" sits in the prose: the default puts it in a protocol line under the table |
| Decisions table | `docs/results/surrogate-training.md`, "The decisions at a glance" | Decision, choice, evidence at the top of a topic file |

## Code docs (`guides/code-docs.md`)

| What | Where | What to imitate |
|---|---|---|
| Result class | `optionpricing/pricing/american/batch_binomial_tree.py::TreeResult` | Every field with its units |
| Constant | `optionpricing/pricing/american/batch_binomial_tree.py::_EXERCISE_TOL` | The why of the value |
| Compact contract | `optionpricing/ml/surrogate/tree/tree_pricer.py::PutSurrogate.price` | Args, Returns, Raises, with the reason in Raises |
| Mode argument | `optionpricing/hedging/legacy/env.py::HedgingEnv.__init__` | One line per named value |

`price_put_batch` and the module docstring of `optionpricing/hedging/exercise.py` are good prose, but longer than the contract-only default. Their design paragraphs belong in REMINDER.

## Theory notes (`guides/theory-notes.md`)

| What | Where in `notes/REMINDER.md` | What to imitate |
|---|---|---|
| Building from first principles | §1 "Pricing by replication" | The situation, the one-step replication, the formula, "Three things to notice" |
| Guard with its reason | §1.3 "`p` must be strictly between 0 and 1" | The assumption, the numbers where it breaks, the guard quoted |
| Worked example | §3.2, "A tiny example you can check by hand" | A 3-step tree in a table, small enough to verify |
| Why not X? | §7.1 "What the network predicts: the early-exercise premium" | The rejected target with its reason |

These chapters write math in Unicode code blocks. New and rewritten chapters use LaTeX (`guides/theory-notes.md`).

## Logs (`guides/logs.md`)

| What | Where | What to imitate |
|---|---|---|
| Bug entry | [B-031](../../../BUGS.md#b-031-early-stopping-cut-longer-cosine-schedules-short) in `.claude/BUGS.md` | Symptom with numbers, mechanism, impact, fix with its test |
| Plan entry | `.claude/PLAN.md`, "Hedging rebuild" | A Goal paragraph with the problem and the approach |

## Explanations in general

`notes/course/CRITIQUE.md` §2 is the author's own checklist for explanations: the order of a concept, nothing before it is taught, derivations line by line, proofs explained, and when to keep a forward reference.
