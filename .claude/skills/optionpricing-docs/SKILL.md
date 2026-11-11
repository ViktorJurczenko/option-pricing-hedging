---
name: optionpricing-docs
description: Write or update documentation in this option pricing and hedging repo so that it matches the project's real conventions and the code as it is now. Use it whenever the task is to write, extend or fix README.md, REMINDER.md-style theory and design notes, module or API references, docstrings, decision records under docs/decisions/, BUGS.md incidents, PLAN.md entries or experiment and results write-ups. Also use it when the user asks to "document", "explain how X works", "write up", "add docstrings to", "update the docs/README for" or "describe the results of" anything in this repo (pricers, the MLP surrogate, the hedging env, the SAC agent, CLI commands). It covers reading the code first, checking every claim, choosing the document type, the notation table, templates and a closing accuracy checklist.
---

# Documentation for the option pricing project

Docs here are read by two kinds of people: the author, months from now, who needs the *reasons* behind a design, and a reader who wants to run a command and get the numbers that are claimed. A doc that is wrong about the code is worse than having no doc, because people trust it. The workflow below exists to keep docs true. The style rules exist to keep them followable.

## Workflow

1. **Pin the scope.** Name the code the doc covers (files, functions, CLI command) and the reader it is for. Choose a document type from the table below. If the request mixes types, for example a theory note that also lists every flag, split it into one doc per type and link them.

2. **Read the code before writing a word.** Open the source files, the `--help` output (`python main.py <cmd> --help`) and the tests that pin the behaviour (`tests/unit/test_<module>.py`). Tests often state an invariant more precisely than the code does. Do not take facts from `.claude/CLAUDE.md` without checking them. It is agent context with little review, and it has contained wrong defaults.

3. **Build the claim list as you draft.** Every name, signature, default value, shape, formula, command and number in the draft is a claim. Check each one:
   - names and defaults: grep or read the definition;
   - CLI flags: `--help`;
   - formulas: find the line that implements them and quote it;
   - commands: run the cheap ones (`--help`, an import, a unit test);
   - numbers: give the command and settings that produce them. If you did not rerun a result, write "not re-verified" instead of presenting it as current.
   
   Never invent a fact. Anything you cannot confirm goes to the user as an open question.

4. **Use the template** for the chosen type, from `references/templates/`. Follow the conventions in `references/conventions.md`.

5. **Reuse the notation** in `references/notation.md`. Use its symbols and code names. Do not invent a new symbol for something the table already names. If a new symbol is unavoidable, define it where it first appears, and tell the user it should be added to the table.

6. **Write the current state, not the history.** Explanation and reference docs say what the code does now and why. How the work got there belongs in `.claude/PLAN.md` (the roadmap) and `.claude/PLAN_ARCHIVE.md` (finished plans), `.claude/BUGS.md` (problems and fixes) and `docs/results/` (measured results). See `references/anti-patterns.md`.

7. **Check the identifiers mechanically.** Run `python .claude/skills/optionpricing-docs/scripts/check_doc_claims.py <doc.md>` from the project root. It checks that every backticked path exists and every code name appears in the source, and it skips math symbols (those in `references/notation.md` and anything that looks like a formula). Fix every missing path or name it reports, unless the doc quotes a wrong or old name on purpose (an audit, a caveat, a BUGS.md entry); leave those and say so. Paths it lists as "generated, not checked" are run outputs (`artifacts/data/`, `artifacts/checkpoints/`): make sure the command that creates each one is named. Add `--strict` to also list short filenames that should be written as full paths. The script cannot judge meaning, such as whether a default value is right, so the checklist still applies.

8. **Run the checklist below** before handing the doc over. Report what you verified, what you could not verify, and any open questions.

Rules from the project's `CLAUDE.md` still apply. A bug found while documenting goes into `.claude/BUGS.md`. A new piece of work gets an entry in `.claude/PLAN.md` before any code is written.

## Document types

| Type | Where | Purpose | Template |
|---|---|---|---|
| README | `README.md` | What the project does, measured results, setup, usage, test commands, architecture map | `templates/readme.md` |
| Theory / design note | `REMINDER.md` style, one `#` part per topic | The logic behind a feature, from first principles to code, and why each choice was made | `templates/theory-note.md` |
| Module / API reference | Docstrings; a section of the README architecture | What each public function and class takes, returns and raises, in which units | `templates/module-reference.md` |
| How-to | A README Usage subsection, or its own section | Steps to reach one goal with the CLI | `templates/how-to.md` |
| Decision record | `docs/decisions/<slug>.md` | One design decision, from symptom to evidence to fix | `templates/decision-record.md` |
| Incident / plan / results log | `.claude/BUGS.md`, `.claude/PLAN.md` (+ `PLAN_ARCHIVE.md`), `docs/results/`, README Results | Dated record of a problem, a plan, or a measured result | `templates/log-entries.md` |

There is no CHANGELOG. Git history plus PLAN.md and BUGS.md serve that purpose. Don't create one unless the user asks.

## Reference files (read them when needed)

- `references/conventions.md`: voice, structure, how math is written, code in docs, reproducibility, naming and links. Read it before any non-trivial doc.
- `references/notation.md`: the symbol table, the matching code names and the terminology. Read it before writing any formula.
- `references/examples.md`: paths to the best existing docs, with short excerpts. Read the excerpt for the type you are writing.
- `references/anti-patterns.md`: mistakes found by the audit. Skim it before the checklist.

## Final checklist

- [ ] **Accuracy.** Every name, path, signature, default, flag and shape was checked against the current source or `--help`.
- [ ] **Identifiers checked.** `scripts/check_doc_claims.py` reports nothing missing.
- [ ] **Examples run as a reader would run them**, from the project root (or with `PYTHONPATH=.`). Their output is pasted from the actual run, and any runtime is measured, not guessed.
- [ ] **Commands run.** Each example command works from the project root. Commands that are too expensive to run are labelled with their cost, and their results are marked "not re-verified" unless rerun.
- [ ] **Numbers sourced.** Each number gives its settings (seed, paths, episodes, grid), the data it was measured on and the command that produces it.
- [ ] **Math defined.** Every symbol is defined at first use and matches `notation.md`. Each formula is followed by the code line that implements it. Units ("per unit of strike", "×K", "$ at K = 100") are stated.
- [ ] **Consistent.** Terminology and symbols agree with README.md and REMINDER.md. Nothing contradicts another doc. If another doc is wrong, report it; don't silently diverge.
- [ ] **Current state only.** No "step N", "as built", commit-dated headers or "previously we thought" in explanation or reference docs. Status such as "still in development" appears only next to work that really is unfinished (see `references/anti-patterns.md`, "A status that went stale").
- [ ] **No concept used before it is taught.** In explanation docs, each idea is introduced before the first passage that relies on it.
- [ ] **Open questions listed.** Anything unclear was reported to the user, not guessed.
