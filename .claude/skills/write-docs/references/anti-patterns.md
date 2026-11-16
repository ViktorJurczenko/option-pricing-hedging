# Anti-patterns

Mistakes seen in this repo's docs. The guides already cover style and planning voice. This page keeps the accuracy failures, since they are the ones that make a doc wrong.

| Anti-pattern | Seen in | Instead |
|---|---|---|
| A hard-coded count that goes stale | README "398 tests pass" when the suite collected 768 | A description, or the count with its command (`pytest --collect-only -q`) in one place |
| A default stated from memory | README giving `--reg_type`'s default as `laguerre` (it is `simple`) | Read it from the signature or `--help`. Say whose default it is (function or CLI) |
| A wrong class name | README `HedgeEnv` (it is `HedgingEnv`) | Grep every identifier, or run the checker |
| A status that went stale | README "no results here yet" after the results existed | Status sits next to the claim it qualifies and goes away in the change that finishes the work |
| A path that no longer exists | BUGS.md naming `AmericanOptionPricing/ML.py` | Fine in a dated log. Current-state docs use current paths |
| A metric that changed meaning | Old BUGS.md "CVaR" values that are VaRs | Define the metric where it is used, and keep it in `references/notation.md` |
| A tracked file citing `notes/REMINDER.md` | "(REMINDER.md, step 6)" in a code comment | State the fact, or cite a tracked doc (a `.claude/BUGS.md` entry title) |
| Facts copied from `.claude/CLAUDE.md` | A wrong `moneyness_range` default | Use CLAUDE.md as a pointer, never as proof |
| A bare comparison | "P&L std 0.014 against 0.015" | Label both sides, or tabulate |
| A result without its command and protocol | | The command above the table, the protocol line under it |
| Agent-context density in a human doc | CLAUDE.md-style paragraphs packing API, history and rationale together | API in docstrings, rationale in REMINDER, history in BUGS.md and PLAN.md |
