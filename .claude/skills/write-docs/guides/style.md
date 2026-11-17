# Style: how the author writes

The author's own answers to a style questionnaire set these defaults. README.md and REMINDER.md are the best examples of them in the repo, and `notes/course/CRITIQUE.md` §2 holds the author's rules for explanations. Each default comes with its reason and its usual exceptions. Follow the reason when a case is not covered.

## Voice and tone

| Default | Why | Exception |
|---|---|---|
| **"I"** for the author's purpose and choices: "I chose this target because ..." | The project is a showcase, so the reader should see who decided what | A results table or a reference section, where nobody is deciding anything |
| **Impersonal** for facts: "The surrogate predicts the early-exercise premium." | Facts don't need an actor | |
| **"We"** inside a derivation the reader walks through: "We want its fair price today." | It invites the reader to follow along | |
| **Plain, conversational tone:** "nobody exercises a right that loses money" | The reader is strong in programming but new to finance. A friendly sentence teaches faster than a formal one | Precise definitions stay precise, even in a friendly paragraph |
| **Present tense, full sentences** with a subject and a verb, opening line included | Fragments read as notes | Table cells and list items, where fragments are natural |

Avoid filler that replaces a step: "it is worth noting", "clearly", "it follows that". Write the step instead.

## Sentences

The author found long, chained sentences hard to read. These defaults keep prose light.

- **One idea per sentence.** A sentence is short because it says one thing, not because words were cut. Dropping a repeated verb ("the plan is in X, finished work in Y") leaves the reader to rebuild the sentence.
- **No semicolons in prose.** Two sentences read better.
- **At most one colon per sentence**, to introduce a list or an explanation. Two ideas joined by a colon usually read better as two sentences.
- **Side details out of the sentence.** Dates, run folders, seeds and settings go in a caption or a protocol line, not mid-sentence.
- **A term by its standard name**, defined at first use unless the reader can be expected to know it: "out of sample, that is when the rule is fitted on one set of paths and applied to fresh ones".
- **A label for an object the text comes back to:** "the scoring grid".

## Reasons: a bold lead-in

A design choice states its reason in the same place, with a bold lead-in:

```markdown
**Why log-uniform.** `T` spans two orders of magnitude. A uniform draw would put 90% of the rows above 0.2 years, where prices change slowly.
```

The lead-in lets a skimming reader find every reason. Variants such as `**Current fix:**` or `**Known cost.**` follow the same pattern. A rejected alternative works well as `**Why not X?**`. When a reason is half a sentence, plain prose ("`T` is drawn log-uniform because ...") is lighter than a label.

## Layout: tables first

| Content | Default form |
|---|---|
| Items that share attributes (methods, files, choices and reasons, results) | A table |
| Reasoning, an argument, a derivation | Prose |
| Three or more parallel items with nothing to tabulate, or ordered steps | A list |

**Why tables.** They are scannable, and they make missing cells visible. A list of bullets that each repeat "X does Y because Z" is a table in disguise. A two-row table is usually lighter as a sentence.

## Numbers

- **Scale small numbers in the column header**, not in every cell: a header "price error, 10⁻⁴ $ at K = 100" with values like 7.9 ± 0.3 reads better than 0.00079 or 7.9e-4.
- **Money in prose stays readable:** "$0.0004, four hundredths of a cent". E-notation suits non-money quantities only.
- **Units every time:** "per unit of strike (K = 1)", "×K", "$ at K = 100".
- **Label both sides of every comparison.** "0.014 for the LSM delta, 0.015 for the exact delta", never a bare "0.014 against 0.015". With several pairs, a small table with labelled rows is clearer.
- **Say what a number was measured on:** "the largest error on the 2342 scoring puts" is more accurate than "the model's worst error".
- **Ranges** as `a-b` or `a to b`.

## Explaining a concept

From the author's rules (`notes/course/CRITIQUE.md` §2):
1. **Order:** a plain-language idea or concrete situation, the precise definition, a small numeric example, why it is needed later.
2. **No concept used before it is taught.** "Made precise later" is not an explanation. A forward reference stays only when the current passage needs it, with a one-line explanation in place.
3. **Length is fine, density is not.** One new idea per paragraph. A longer section that explains beats a short one that skips steps.
4. **Bridges from computer science**, said to be bridges: an invariant, a reduction, expected value as a weighted average. Say where the analogy fails.

## Honesty

- Say what is measured and what is a hypothesis: "This is a hypothesis, not yet verified."
- State the known cost of a choice: "a known, accepted cost of that fix".
- Flag unfinished work next to the claim it qualifies, since a reader of a highlight may never reach the status table.

## Headings and names

- Sentence case. `#` for a part, `##` for sections, `###` for subsections.
- A result's heading states the finding: "Pricing models agree with each other", rather than "Pricing results".
- Code names in backticks: `` `price_put_batch` ``, `` `PutSurrogate` ``.
- CLI commands are kebab-case (`train-ml`), flags snake_case (`--option_type`).
- Cross-doc references are links to the section, not quoted titles: [`.claude/BUGS.md`, B-037](../../../../.claude/BUGS.md#b-037-hedgingenv-applies-each-hedge-one-step-late). A BUGS.md entry is linked at its heading, so the reader jumps to the story.
- A link's path is relative to the citing file, so from `docs/results/` it starts with `../../.claude/`. GitHub's anchor is the heading in lowercase, with spaces turned into hyphens and other punctuation dropped, so P&L becomes pl.

## Style checklist

- [ ] Each sentence says one thing. No semicolons, at most one colon.
- [ ] Every design choice carries its reason, most often in a bold lead-in.
- [ ] Items sharing attributes are in a table.
- [ ] Small numbers are scaled in headers, units are stated, comparisons are labelled on both sides.
- [ ] No concept is used before it is taught, and no filler hides a step.
- [ ] Where a default was broken, the passage reads better for it.
