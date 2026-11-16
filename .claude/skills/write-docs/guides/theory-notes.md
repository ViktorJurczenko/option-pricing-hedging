# Theory notes: REMINDER.md

`notes/REMINDER.md` explains the logic of every feature, from first principles to the code, and the reasons for each current choice. It is the author's personal reference and is gitignored, so tracked files never cite it.

**Reader.** A strong programmer new to finance, the yardstick from `notes/course/CRITIQUE.md`: "a sentence is a defect if that reader cannot follow it with only what came before it".

## What a chapter contains

| Part | Default | Why |
|---|---|---|
| The problem | the concrete situation in plain words, and why the easy approach fails | The gap is what the rest of the chapter fills |
| Each idea | plain idea, precise definition, small numeric example, why it is needed later | The author's order for teaching a concept (CRITIQUE §2.1) |
| Formulas | LaTeX, then the code that implements it, then a link to the file | The reader sees that the math and the code agree |
| A worked example | numbers small enough to check by hand (the 3-step tree of §3.2, "A tiny example you can check by hand") | It turns the formula into something the reader can verify |
| Why not X? | each rejected alternative, with the reason it loses | The obvious alternative is the first question a reader asks |
| Guards | when a formula breaks, and the guard that refuses it | Assumptions are part of the method |

A "What is checked" section (which tests pin the behaviour) is optional. It helps for a component with subtle invariants (the batch tree). It is noise for a simple one.

Number sections in build-up order (`## 3.1`, `## 3.2`), so each depends only on earlier ones. A backward reference names the section: "(section 3.6)".

## Math in LaTeX

Inline math in `$...$`, display math in `$$...$$`, one formula per display:

```markdown
The price is a discounted average of the two future values:

$$V = e^{-r\,\Delta t}\,\big(p\,V_u + (1 - p)\,V_d\big), \qquad p = \frac{e^{r\,\Delta t} - d}{u - d}$$
```

- **Define every symbol where it first appears**, including the "obvious" ones. Use the symbols in `references/notation.md`. A new symbol is defined in place, and the user is told it should go into the table.
- **A derivation goes line by line**, each step with a short justification. Say before it what is being shown, and check it on a number after it. An `aligned` block keeps the steps readable:

  ```markdown
  $$
  \begin{aligned}
  \Delta\,S u + B\,e^{r\Delta t} &= V_u \
  \Delta\,S d + B\,e^{r\Delta t} &= V_d && \text{same portfolio, down state} \
  \Delta &= \frac{V_u - V_d}{S u - S d} && \text{subtract the two lines}
  \end{aligned}
  $$
  ```
- **Maths only where it earns its place.** A table of cash flows or a numeric example often makes the argument better than algebra (CRITIQUE §2.3b).
- **Numerical caveats in numbers:** "about ±0.1 here" for Monte-Carlo noise, "against a 16,000-step reference" for an error.
- **Older chapters use Unicode math in code blocks.** When you rewrite a chapter, convert the whole chapter to LaTeX, so one chapter never mixes the two.

## Formula, then code, then link

After a formula, quote the few lines that implement it, map the code names to the symbols, and link the file with its name as the label:

~~~markdown
In the code, with `growth` $= e^{r\Delta t}$ and $d = 1/u$ ([batch_binomial_tree.py](../optionpricing/pricing/american/batch_binomial_tree.py#L236-L238)):

```python
u = np.exp(sigma * np.sqrt(dt))
growth = np.exp(r * dt)
p = (growth - 1 / u) / (u - 1 / u)  # risk-neutral probability of an up move
```
~~~

- **The link label is the file name**, not the full path. The target is relative to `notes/`, so it starts with `../`.
- **A line range** (`#L236-L238`) points to the quoted lines. Line numbers go stale, so the checker verifies the range exists. Recheck it when the code moves.
- **Short excerpts.** Quote the lines that carry the idea, not whole functions.

## Current state, not history

REMINDER explains the code as it is now. It is not a plan log.

| Avoid | Instead |
|---|---|
| "Personal notes from 2026-10-22, after commit `fcc60f1`" | Start with what the section explains |
| "as built (step 4)", "step 7 should report" | Name the component: "the evaluation (`tree_evaluation.py`)" |
| "After the review ...", "Where things stood" | State the current behaviour and its reason |
| "(A first reading of the table was wrong ...)" | Write the correct statement. A likely trap becomes a caution: "Worst cases mislead here: they all sit in the exercise region." |
| "First we tried X, then Y" | `**Why not X?**` with the reason |

A replaced component (the legacy surrogate, the SAC hedger) keeps its chapter only with a clear "legacy" label in its title.

## Checklist

- [ ] Every idea follows idea, definition, example, why needed. Nothing is used before it is taught.
- [ ] Every formula is in LaTeX, defined symbol by symbol, followed by its code and a `[file.py](../path#Lx-Ly)` link.
- [ ] The chapter has a worked numeric example and its "Why not X?" alternatives.
- [ ] No dates, step numbers or self-corrections.
- [ ] `check_doc_claims.py notes/REMINDER.md` reports no broken link.
