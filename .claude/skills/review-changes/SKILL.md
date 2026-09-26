---
name: review-changes
description: Use when reviewing a diff, PR, or set of changes in this repo against the project's review checklist (naming, type hints, documentation, code smells, test quality, diff scope, error handling, contributor headers). Produces a per-point OK/ISSUE report and refuses to approve anything it cannot actually verify. Trigger on "review this", "review my changes", "review the PR", "code review".
---

# Reviewing changes in this repo

Companion to the `write-tests` skill (which covers how tests get written);
this one covers how changes get reviewed, including the tests written under
it.

## Ground rules for the reviewer

- **Assume no prior context.** You may be run by an agent with no memory of
  any conversation that led to this change. Don't rely on a description of
  what the diff does from anyone else — read the actual diff, read enough of
  the surrounding file(s) to judge naming/idiom consistency, and read
  `.claude/CLAUDE.md` plus any relevant file under `.claude/rules/` for the
  conventions this repo actually holds itself to. Verify claims against what
  the code does, not against what a commit message or PR description says it
  does.
- **Refuse to approve what you cannot review.** If a checklist point needs
  something you don't have — you can't execute the test suite, a file is
  binary/generated and unreadable, the diff is too large to read in full, you
  have no way to confirm CI status — say so explicitly for that point instead
  of assuming it's fine. "Cannot verify — did not run the tests" is a
  required, valid answer when true; do not mark something OK on faith.
- **Report format:** go through the checklist below in order, one point at a
  time, each marked **OK** or **ISSUE**. Every ISSUE needs a one-line reason
  and a `file:line` pointer. Anything you could not check gets **CANNOT
  VERIFY** with a one-line reason why, not a guessed OK. End with a
  **Suggestions** section (see below), then an overall verdict: **APPROVE**,
  **REQUEST CHANGES**, or **CANNOT REVIEW** (if too much of the checklist
  came back CANNOT VERIFY to form a real judgment).

## Checklist

1. **Self-explanatory variable/function names.**
   A reader shouldn't need to open the surrounding function to guess what a
   name holds. Domain-standard short names already used consistently in this
   project (`S0`, `K`, `r`, `T`, `sigma`, `lsm`, …) are fine — they're
   established pricing notation, not unclear abbreviations. Flag genuinely
   opaque ones instead: `tmp`, `data2`, `res`, single-letter names outside a
   tight loop/comprehension scope. Also flag a new name that contradicts how
   the same concept is already named elsewhere in the module.

2. **Type hints where they remove real ambiguity.**
   Not required on every parameter — this repo is inconsistently typed today,
   so don't demand a blanket retrofit. Required wherever a new/changed
   signature's meaning isn't obvious from the name and default alone (e.g. is
   this argument a `list` or an `np.ndarray`? An `int` step count or a
   `float` year fraction?). Check signatures the diff actually touches, not
   the whole file.

3. **Documentation present where it's actually needed.**
   Same bar as this project's comment policy in `CLAUDE.md`: a docstring is
   owed wherever a caller could reasonably get it wrong without being told —
   required call order (e.g. `predict_payoff()` before
   `calculate_option_price()`), units, array shapes, non-obvious return
   values, side effects. Don't require a docstring that just restates what
   the signature already says.

4. **No code smells.**
   Duplicated logic that should be a shared helper — but weigh this against
   the project's own anti-overengineering stance ("three similar lines is
   better than a premature abstraction"); don't flag duplication that's only
   two or three simple lines. Also check for: deep/needless nesting, magic
   numbers with no stated reason (an already-established constant like
   `beta_frac=0.06` is fine since it's explained where it's defined; a new
   unexplained one isn't), bare/silent `except`, mutable default arguments,
   dead code, and functions doing more than one clearly separable thing.

5. **Data used is relevant.**
   Test fixtures and example inputs should be representative of the actual
   domain being exercised (sane `S0`/`K`/`σ`/`T` ranges, not arbitrary values
   disconnected from what's being tested), and no larger than needed to
   exercise the behavior (don't load/generate a large dataset when a small
   one proves the same point). Hardcoded expected values should trace to a
   principled source — a closed-form formula, a known convergence bound — not
   be copy-pasted from whatever one run happened to output.

6. **Tests actually test something, and pass.**
   Tests must assert real behavior, not just a count, a non-null check, or
   "no exception was raised" when there's a meaningful return value or state
   change to check instead. If you can execute the suite, run it and report
   pass/fail with a summary; if you cannot execute it, mark this **CANNOT
   VERIFY** rather than assuming green. For the full standard this repo holds
   new tests to (unit-first, real coverage, test doubles for any network I/O,
   regression tests for fixed bugs), see `.claude/skills/write-tests/SKILL.md`
   — use it as the bar when judging whether tests in this diff meet it.

7. **Diff is bounded and reviewable.**
   A small, single-purpose change shouldn't touch unrelated files or modules.
   If the diff spans multiple unrelated areas, flag it and recommend it be
   split rather than trying to force a single judgment over all of it.

8. **Error handling and edge cases are covered.**
   New code should handle its stated failure modes explicitly, matching this
   project's existing pattern (raising `ValueError`/`AttributeError` rather
   than silently falling back — see `CLAUDE.md`'s "follow that pattern rather
   than adding silent fallbacks"). Edge cases relevant to the change (empty
   input, boundary values, zero/negative inputs where meaningful) should be
   exercised by a test, not just handled in the implementation and left
   unverified.

9. **Contributors are acknowledged in the commit, not the file.**
   This project tracks attribution through the commit message, not a
   per-file header — check the commit(s) this diff belongs to, not the
   changed files themselves. A commit should follow the standard conventions
   (clear imperative subject, body explaining what/why when the subject
   isn't enough) and carry a `Co-Authored-By:` trailer whenever the change
   had more than one contributor (e.g. a human author plus Claude). If you
   can't see the commit (only a working-tree diff, no commit yet made),
   mark this **CANNOT VERIFY** rather than guessing, and note it should be
   checked once the change is actually committed.

## Suggestions

Always close with a **Suggestions** section, separate from the OK/ISSUE
checklist: naming that could be clearer without being wrong, a simplification
that isn't required but would help, a test case worth adding that isn't
strictly required by the checklist above. These don't block approval on their
own.
