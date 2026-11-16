---
name: review-changes
description: Use when reviewing a diff, PR, or set of changes in this repo. It has two modes. Deep mode hunts for bugs and runs the full checklist (correctness, numerics, naming, type hints, documentation, code smells, test quality, diff scope, error handling, commit attribution), consulting an advisor agent on uncertain findings. Quick mode checks docs, style, scope and reported tests for docs-only, test-only or follow-up changes. Both produce a per-point OK/ISSUE report and refuse to approve anything they cannot verify. Trigger on "review this", "review my changes", "review the PR", "code review".
---

# Reviewing changes in this repo

Companion to the `write-tests` skill (how tests get written) and the `write-docs` skill (how docs get written). This one covers how changes get reviewed.

## Two modes

| Mode | Agent | When | Checklist |
|---|---|---|---|
| **Deep** | `review-agent` (Sonnet), consults `review-advisor` (Opus) | any change to code logic under `optionpricing/` or `scripts/`, or when the user asks | "Deep checklist" below |
| **Quick** | `quick-review-agent` (Haiku), no advisor | docs-only or test-only changes, and follow-up fixes after a deep review | "Quick checklist" below |

The dispatcher's prompt names the mode. The rule that picks one is in `.claude/CLAUDE.md`. Without a stated mode, the agent's own mode applies.

**A quick review escalates instead of guessing.** If it finds a change to code logic, or a likely bug it cannot judge, it marks the point **ESCALATE** with a one-line reason and returns the verdict **NEEDS DEEP REVIEW**. Judging logic it was not built to check would give false confidence.

## Ground rules (both modes)

- **Assume no prior context.** You may be run by an agent with no memory of the conversation behind the change. Read the actual diff, enough of the surrounding files to judge consistency, and `.claude/CLAUDE.md` plus any matching file under `.claude/rules/`. Verify claims against the code, not against a commit message or the dispatcher's description.
- **Refuse to approve what you cannot review.** If a point needs something you don't have (you can't run the suite, a file is binary, the diff is too large to read in full), mark it **CANNOT VERIFY** with the reason. Never mark it OK on faith.
- **Don't rerun a test run the dispatcher already reported.** When the prompt gives the exact test commands run on the current diff and their results (e.g. `python -m pytest tests/unit` -> 674 passed), use them and say the result is the dispatcher's. Check the report against `git status` and `git diff`. A file the report doesn't account for, or one changed after the run, is not covered. Only test results may be reused this way, never a description of the diff. Run tests yourself only for what the report does not cover: a suite it doesn't mention (integration when CLI wiring changed), files changed after the run, or a test you have a concrete reason to doubt (skipped, deselected, unable to fail). With no reported results, run the suites the change needs. Reading tests to judge their quality is never skipped.
- **Report format.** Go through the mode's checklist in order, each point **OK**, **ISSUE** or **CANNOT VERIFY** (quick mode also **ESCALATE**). Every ISSUE has a one-line reason and a `file:line` pointer. End with **Suggestions**, then the verdict: **APPROVE**, **REQUEST CHANGES**, **CANNOT REVIEW** (too much came back CANNOT VERIFY), or, in quick mode, **NEEDS DEEP REVIEW**.

## Deep checklist

Read every changed file in full before starting. Points 1 and 2 are the bug hunt, which is why this mode exists. Spend most of the effort there.

1. **Correctness.** Does the code compute what it claims? Check it against an independent reference: a closed form, a finer tree, a finite difference, a hand-computed small case. Trace one concrete input through the changed code. Look for off-by-one errors in time grids (`n_steps` against `n_steps + 1` points), sign errors (short against long, put against call), a wrong axis or broadcast, discounting applied twice or not at all, and train/test leakage (a rule fitted on the paths it is scored on).

2. **Numerics, shapes and units.** Division by zero or `log(0)` at domain edges (`T → 0`, `σ → 0`, `m → 0`), NaN or inf propagation, float precision near a tolerance, dtype or device mismatches in torch, units mixed up (per unit of strike against dollars, years against days, `gamma` against `K · gamma`).

3. **Self-explanatory names.** A reader shouldn't need to open the surrounding function to guess what a name holds. Domain-standard short names already used consistently (`S0`, `K`, `r`, `T`, `sigma`, `lsm`) are fine. Flag opaque ones (`tmp`, `data2`, `res`, single letters outside a tight loop), and a new name that contradicts how the module already names the same concept.

4. **Type hints where they remove real ambiguity.** Not on every parameter, since the repo is inconsistently typed. Required where a new or changed signature's meaning isn't obvious from the name and default (a `list` or an `np.ndarray`? an `int` step count or a `float` year fraction?). Check only signatures the diff touches.

5. **Documentation present and useful.** Every module and public function or class has a docstring that follows `.claude/skills/write-docs/guides/code-docs.md`: a short summary, then typed `Args:`/`Returns:`/`Raises:`. Check above all that it says what a caller could get wrong: call order (`predict_payoff()` before `calculate_option_price()`), units, shapes, special return values, side effects. Comments explain why. A measured number cites a `.claude/BUGS.md` entry or a tracked doc, never the gitignored `notes/REMINDER.md`.

6. **No code smells.** Duplicated logic that should be a shared helper, weighed against the project's anti-overengineering stance (three similar simple lines beat a premature abstraction). Also: needless nesting, unexplained magic numbers (an established, explained constant is fine), bare or silent `except`, mutable default arguments, dead code, a function doing several separable things.

7. **Tests actually test something, and pass.** Tests assert real behaviour, not a count, a non-null check or "no exception" when there is a value to check. Numeric tests compare against an independent reference with a tolerance explained in a comment. Every random component is seeded. Ask of each test "what breaks if it is deleted?", and try one mental mutation of the code (flip a comparison, change a constant) to see whether a test would catch it. Fixtures use sane domain values (`S0`, `K`, `σ`, `T`) and no more data than needed. Expected values trace to a principled source, not to one run's output. For pass/fail, use the reported results or run what they don't cover. The full bar is `.claude/skills/write-tests/SKILL.md`.

8. **Error handling and edge cases.** New code handles its stated failure modes explicitly, raising `ValueError` eagerly rather than falling back silently (`.claude/CLAUDE.md`, "Conventions"). Edge cases that matter for the change (empty input, boundaries, zero or negative values) are exercised by a test, not only handled in the code.

9. **Diff is bounded.** A single-purpose change doesn't touch unrelated files. A diff spanning unrelated areas is flagged with a recommendation to split it.

10. **Living docs agree with the code.** `.claude/PLAN.md` steps are ticked, a bug found has its `.claude/BUGS.md` entry, and README.md or `docs/` claims touched by the change still hold.

11. **Commit attribution.** Attribution lives in the commit message, not in file headers. A commit follows `.claude/CLAUDE.md`'s Git rules and carries a `Co-Authored-By:` trailer when it had more than one contributor. With only a working-tree diff and no commit yet, mark this **CANNOT VERIFY** and say it should be checked at commit time.

### Consulting the advisor (deep mode)

Before writing the report, consult `review-advisor` (Agent tool, `subagent_type: "review-advisor"`) on:
- every ISSUE you are not sure of;
- every finding that would block approval;
- your draft verdict.

It starts with no context, so its prompt must stand alone: the files and line ranges concerned (or "see `git diff`"), each finding with its evidence, and the draft verdict. It answers **CONFIRM**, **REJECT** or **ADJUST** per finding, with a reason. Fold its answers in, and note them in the report ("advisor: confirmed", "advisor: rejected, because ..."). Where you still disagree with it, keep your own judgment and give both views. A review with nothing uncertain and no blocking finding still sends the draft verdict, in one short message.

If the advisor cannot be spawned, say so at the top of the report and continue alone.

## Quick checklist

1. **Scope.** The diff is what the dispatcher says: docs, tests, or a small follow-up. A change to code logic beyond a follow-up fix is **ESCALATE**.
2. **Doc accuracy.** For each changed Markdown file, run `python .claude/skills/write-docs/scripts/check_doc_claims.py <file>` and report what it finds. Spot-check names, defaults and numbers in the changed lines against the code or `--help`.
3. **Style.** Changed docs follow `.claude/skills/write-docs/guides/style.md` and the guide for their kind. The guides give defaults, so flag a departure only when the passage reads worse for it.
4. **Tests.** Reported results cover the current diff (same rule as above). Changed or added tests assert real behaviour, are seeded and explain their tolerances.
5. **Follow-up fixes.** When the prompt lists the findings of an earlier deep review, each one the follow-up claims to fix is actually fixed, and nothing else changed.
6. **Living docs.** PLAN.md, BUGS.md and README.md agree with the change.

## Suggestions

Close with **Suggestions**, separate from the checklist: a clearer name, an optional simplification, a test case worth adding. They don't block approval. The dispatcher proposes them to the user, who decides.
