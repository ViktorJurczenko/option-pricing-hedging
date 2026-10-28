# Anti-patterns

Each item below was seen in this repo's docs during the audit (`AUDIT.md`). The fix says what to do instead.

## Accuracy

| Anti-pattern | Seen in | Instead |
|---|---|---|
| A hard-coded count that goes stale | README "398 tests pass"; the suite now collects 768 | Avoid volatile counts, or state the date and command (`pytest --collect-only -q`) next to them |
| A default stated from memory | README L199–201 "`--reg_type` … default `laguerre`" (it is `simple`); CLAUDE.md giving `main`'s argument as the function default | Read the default from the signature or `--help` at write time. Say whose default it is (function vs CLI) |
| A wrong or outdated class name | README L85 `HedgeEnv` (it is `HedgingEnv`) | Grep every identifier before finalizing |
| A status that went stale | README L159 "no results here yet"; REMINDER "Nothing has been trained … yet" after the training was done | Don't write status in a doc that lasts. If unavoidable, date it and link the PLAN.md entry |
| A path that no longer exists | BUGS.md naming `AmericanOptionPricing/ML.py` | Acceptable in a dated log only. In current-state docs, use the current path |
| A metric name that changed meaning | Old BUGS.md "CVaR" values that are really VaRs | Define the metric where it is used, and keep the definition in `notation.md` |
| A code or test comment pointing to `REMINDER.md` | "(REMINDER.md, step 6)" in `tree_evaluation.py`, since removed | REMINDER.md is gitignored, so other users don't have it. State the fact in the comment, or cite a tracked file (a `.claude/BUGS.md` entry title, `docs/decisions/`) |
| Copying facts from `.claude/CLAUDE.md` unchecked | Wrong `moneyness_range` default | Use CLAUDE.md as a pointer to where to look, never as proof |

## Planning voice inside explanation docs

REMINDER.md exists to explain the logic of every feature and the reasons for the current choices. These patterns make it read like a plan log instead:

| Anti-pattern | Example | Instead |
|---|---|---|
| A dated, commit-anchored header | "Personal notes from 2026-10-22, written after step 1 … (commit `fcc60f1`)" | Start with what the section explains. History lives in git, PLAN.md and BUGS.md |
| Plan step numbers | "as built (step 4)", "step 7 should report…" | Name the component: "the evaluation (`tree_evaluation.py`)" |
| Process narration | "After the review…", "added after step 6", "Where things stood" | State the current behaviour and its reason |
| Self-correction in the text | "(A first reading of the table, 'gamma is not a usable label', was wrong…)"; README "An earlier version of this section claimed…" | Write only the correct statement. If the misreading is a likely trap, phrase it as a caution: "Worst cases mislead here: they all sit in the exercise region." |
| A section about a replaced component, mixed in with current ones | "Is the MLP surrogate pricer actually useful?" (the legacy model) | Move it to the legacy model's section with a clear label, or drop it |

## Explanation quality

These are the author's rules from course/CRITIQUE.md, recast as things to avoid:

- **Using a concept before teaching it**, or promising to explain it "later" instead [C 2.2].
- **Steps hidden behind** "clearly", "it follows that" or "taking logs gives" [C 2.3].
- **A proof or formula with no intuition before it** and no statement after it of what it means and where it is used [C 2.3b].
- **Heavy machinery where a table or a number would do** [C 2.3b].
- **Several new terms in one paragraph**, or a dense paragraph full of cross-references [C 2.1, criticism 5].
- **Forward references that are name-drops** (a class name, a bug label or a number the reader can't use yet) [C 2.4].
- **Notation that appears only in a glossary at the end**, instead of at first use [C 5.4].

## Doc-type mixing

- **Agent-context density in human docs.** CLAUDE.md packs a module's API, invariants, history and rationale into one paragraph. In human docs, split them: the API goes in docstrings, the rationale in the theory note, the history in BUGS.md and PLAN.md.
- **Results without the command and protocol** that produced them. Every README result shows both.
