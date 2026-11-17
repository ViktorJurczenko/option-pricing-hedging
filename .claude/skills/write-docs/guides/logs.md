# Logs: BUGS.md, PLAN.md and commit messages

Logs are dated records. They are the right place for history ("found on", "first fix, later reverted"), which explanation docs leave out.

## Formats live in the files

Each log defines its own layout in its header. That header is the single source, so read it before writing an entry, rather than copying a format from here.

| Log | Format in |
|---|---|
| `.claude/BUGS.md` | its header: next free ID, index row, `### B-0NN` entry with Symptom, Root cause, Impact, Fix, Regression test |
| `.claude/PLAN.md` | its header: a table row per workstream, **Goal** then `- [ ]` steps |
| `.claude/PLAN_ARCHIVE.md` | its header: a short summary per finished workstream |
| commit messages | `.claude/CLAUDE.md`, "Git": Conventional Commits, type and scope lists, subject length |

This guide adds only what makes an entry good to read.

## BUGS.md entries

- **The title states the symptom** in a short sentence: "Early stopping cut longer cosine schedules short". Other docs cite entries by id (`guides/style.md`), so an id is never changed or reused.
- **The symptom carries numbers** and the command or test that showed them. "The LSM delta left the valid range up to 91.6% of the time just before maturity" beats "delta was unstable".
- **The root cause explains the mechanism**, so a reader could have predicted the symptom from it.
- **Impact says which earlier results are invalid.** It is the part a later reader needs most.
- **Both sides of a comparison are labelled**, before and after a fix most of all (see `guides/style.md`).
- **The regression test is named**, and the entry says it fails without the fix.
- The old paths and names of the time stay in the entry text. Only `Files:` uses current paths.

## PLAN.md entries

- **The Goal is a paragraph:** the problem, then the chosen approach and why. A reader who skips the steps still knows what the workstream is for.
- **Each open step is a title plus a line or two** on what it does and why. A finished step loses its description.
- **Measured results don't stay in a plan step.** They go to `docs/results/`, and the step links them.
- A step that needs the user first says so: "Explain the theory to the user before writing any code."

## Archive summaries

A handful of bullets with the main decisions, an **Outcome** line with the measured result, and **Links** to commits, BUGS entries and results. The summary says where the detail is rather than repeating it.

## Commit messages

- **The subject says what changed**, in the imperative, lowercase after the scope: `feat(hedging): add exercise dates`.
- **The body says why**, wrapped at 72 characters. What changed is in the diff. Why it changed is not.
- One commit per logical change. A docs-only commit uses `docs(<scope>)`.
- The message is shown to the user for confirmation before `git commit` (CLAUDE.md).

## Checklist

- [ ] The file's header format is followed.
- [ ] Symptoms and outcomes carry numbers, with both sides of each comparison labelled.
- [ ] Results live in `docs/results/` and are linked, not pasted into a plan.
