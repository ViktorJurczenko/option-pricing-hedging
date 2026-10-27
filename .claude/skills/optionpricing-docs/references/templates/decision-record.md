# Template: decision record (`docs/decisions/<kebab-slug>.md`)

Use this for one design decision whose reason isn't obvious from the code, especially one a future change could undo by mistake.

```markdown
# Why `<thing>` <does X>, not <the obvious alternative>

<One paragraph: the general pattern this is an instance of, with a link if one is documented, and what this file records.>

## Symptom

<What was observed, in numbers, and where (`path/file.py`, `Class`).>

## Root cause

<The mechanism, explained so that a reader could have predicted the symptom.>

## How this was confirmed

- <An experiment that ruled out the competing explanation, with its result.>
- <A second one, if any.>

<One sentence on why these results single out the root cause.>

## Fix

<What the code does now, with the files, functions and constants involved. Every place that must stay consistent with it (e.g. training and reconstruction). How any parameter was chosen (seeded sweep, fixed benchmark).>
```

Add a matching short entry to `.claude/BUGS.md` and link the two.
