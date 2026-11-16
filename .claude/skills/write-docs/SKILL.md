---
name: write-docs
description: Write, update or audit documentation in this option pricing and hedging repo, in the author's own style and true to the code as it is now. Use it whenever the task touches README.md, docs/RESULTS.md or docs/results/, docs/architecture.md, docstrings or code comments, REMINDER.md theory and design notes, BUGS.md or PLAN.md entries, or a commit message body. Also use it when the user asks to "document", "explain how X works", "write up", "add docstrings to", "update the docs/README for" or "describe the results of" anything in this repo (pricers, the surrogate, the hedging stack, CLI commands). It routes to one guide per kind of doc and ends with an accuracy checklist.
---

# Writing docs for the option pricing project

Docs here have three readers:
- a recruiter or professor skimming the README of a showcase project;
- a reader who wants to rerun a command and get the claimed numbers;
- the author, months from now, who needs the reasons behind a design.

A doc that is wrong about the code is worse than no doc, because people trust it. The workflow below keeps docs true. The guides keep them in the author's style.

## Defaults, not absolutes

The guides give defaults, each with its reason. A default exists because it usually reads better, not because it is a law. When a passage reads clearly better another way, write it that way, and keep the reason in mind so the choice is deliberate. The guides name the usual exceptions ("prefer X, because Y. Z is better when ...").

A few rules are hard, because breaking them makes a doc wrong rather than less pleasant:
- every claim matches the current code (names, defaults, shapes, numbers);
- no invented fact or number, and an unverified number says so;
- tracked files never cite the gitignored `notes/REMINDER.md`.

## Which guide

Read `guides/style.md` for any doc. Then read the guide for the kind you are writing.

| Kind of doc | Files | Guide |
|---|---|---|
| Project md | `README.md`, `docs/RESULTS.md`, `docs/results/<topic>.md`, `docs/architecture.md` | `guides/project-md.md` |
| Code docs | module, function and class docstrings, constants, inline comments | `guides/code-docs.md` |
| Theory notes | `notes/REMINDER.md` chapters | `guides/theory-notes.md` |
| Logs | `.claude/BUGS.md`, `.claude/PLAN.md`, `.claude/PLAN_ARCHIVE.md`, commit messages | `guides/logs.md` |

A request that mixes kinds (a theory note that also lists every flag) is usually better split, with one doc per kind and links between them.

References, read when needed:
- `references/notation.md`: symbols, their LaTeX and code names, and the terms to use consistently. Read it before writing any formula.
- `references/examples.md`: the best existing passage for each kind of doc, to imitate.
- `references/anti-patterns.md`: mistakes found in this repo's docs. Skim it before the checklist.

## Workflow

1. **Pin the scope.** Name the code the doc covers (files, functions, CLI command), its reader and its kind.
2. **Read the code before writing.** Open the source, the `--help` output (`python main.py <cmd> --help`) and the tests that pin the behaviour. Tests often state an invariant more precisely than the code. Treat `.claude/CLAUDE.md` as a pointer to where to look, never as proof. It is agent context with little review, and it has held wrong defaults.
3. **Keep a claim list while drafting.** Every name, signature, default, shape, formula, command and number is a claim. Check each one:

   | Claim | How to check it |
   |---|---|
   | name, default | read the definition |
   | CLI flag | `--help` |
   | formula | find the line that implements it |
   | command | run the cheap ones (`--help`, an import, a unit test) |
   | number | give the command and settings that produce it, or write "not re-verified" |

   Anything you cannot confirm goes to the user as an open question.
4. **Write the current state.** Explanation and reference docs say what the code does now and why. How the work got there belongs in git, `.claude/PLAN.md`, `.claude/PLAN_ARCHIVE.md` and `.claude/BUGS.md`.
5. **Check identifiers and links mechanically.** From the project root, run `python .claude/skills/write-docs/scripts/check_doc_claims.py <doc.md>`. It checks that backticked paths exist, that code names appear in the source, and that relative links (`[tree_pricer.py](optionpricing/ml/surrogate/tree/tree_pricer.py#L40-L52)`) point to an existing file and line range. Math is skipped. Fix what it reports, unless the doc quotes an old name on purpose (a BUGS.md entry, an audit), and say so. `--strict` also lists short filenames that could be full paths. The script cannot judge meaning, so the checklist still applies.
6. **Run the checklist** and report what you verified, what you could not, and any open questions.

Rules from `.claude/CLAUDE.md` still apply. A bug found while documenting goes into `.claude/BUGS.md`. New work gets a `.claude/PLAN.md` entry first.

## Final checklist

- [ ] **Accurate.** Every name, path, signature, default, flag and shape matches the current source or `--help`.
- [ ] **Checker clean.** `scripts/check_doc_claims.py` reports nothing missing, or each report is explained.
- [ ] **Commands run** from the project root. Expensive ones carry their cost, and their results say "not re-verified" unless rerun.
- [ ] **Numbers sourced.** Each number has its settings, its data and the command that produces it, and both sides of every comparison are labelled.
- [ ] **Math defined.** Each symbol is defined at first use and matches `references/notation.md`. Units are stated.
- [ ] **Consistent** with README.md and the other docs. If another doc is wrong, report it rather than diverge silently.
- [ ] **In the author's style.** Re-read `guides/style.md`'s checklist against the draft.
- [ ] **Open questions listed** for the user, not guessed.
