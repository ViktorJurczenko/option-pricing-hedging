# Project md: README, results and architecture

These docs present the project. The README is its showcase. This is the author's first finance project, and it will go on a resume, so recruiters, professors and reviewers read it. `docs/RESULTS.md` and `docs/results/` hold the analysis behind each result. `docs/architecture.md` is the map of the code.

| Doc | Reader | Job |
|---|---|---|
| `README.md` | someone deciding in two minutes whether the project is good | pitch, key findings, how to run it |
| `docs/RESULTS.md` | a reader who wants every result at a glance | one section per topic, with its key table and a link to the detail |
| `docs/results/<topic>.md` | a reader checking how a result was reached | paper-style analysis of one topic |
| `docs/architecture.md` | a developer finding their way in the code | the layer map and one table of files per package |

## README.md

### Size and order

The current size, around 500 lines, suits an academic showcase. Detail beyond a result's key table and interpretation belongs in `docs/results/`. Keep the existing section order, and put a new section in its slot rather than at the end:

| Section | What it holds |
|---|---|
| Title and pitch | 2-3 sentences on what the project builds and from what, then "my first in quantitative finance" and the from-scratch principle |
| Highlights | 4-6 bullets, each one finding and one number |
| Methods and references | a table: method, what it is used for, the paper |
| Project status | a table: area, status |
| What's inside | a table: layer, what it does, and whether it is wired into the other layers |
| Results | one `###` per finding (format below) |
| Setup, Usage, Testing | commands, from the project root |
| Architecture | the package list, linking `docs/architecture.md` |
| How it was built, Further reading, License | |

### Highlights: a finding and one number

```markdown
- **LSM Greeks repaired near expiry.** A regression of pathwise deltas cut the out-of-the-money delta error from 0.2-0.9 to under 0.02.
```

The bold title states the finding. One sentence gives the headline number. The rest sits behind the Results section. **Why one number.** A skimming reader remembers one number, and several dilute it. A second number earns its place when the finding is a comparison ("360-510× faster than the tree it learned from").

### A result section

~~~markdown
### <Finding as the title>

```
python main.py <command> <flags>     # what it prints, ~cost
```

| <rows> | <metric, scaled unit> | ... |
|---|---|---|
| ... | **best** | ... |

<Protocol line: seeds, episodes or paths, costs, date of the run, what a range means.>

<One or two short paragraphs, or bullets, on what the table shows and its limits.>

Details: [surrogate-training.md](docs/results/surrogate-training.md).
~~~

- **The command comes right above the table**, so a reader can rerun it.
- **The protocol line goes under the table.** Seeds, dates and settings stay out of the prose.
- **The interpretation says what the table means**, not every number again.
- **The best value is in bold.** Small numbers are scaled in the header (see `guides/style.md`).

### Past bugs in the README

The README describes the current state. A past bug appears only when it was major, tricky to find or fix, and shows good debugging. Even then, frame it as a finding ("LSM Greeks repaired near expiry"), keep it to a sentence or two plus a number, and link the BUGS.md entry by its id for the story, as [B-031](../../../../.claude/BUGS.md#b-031-early-stopping-cut-longer-cosine-schedules-short) (`guides/style.md`). Routine fixes stay in `.claude/BUGS.md`.

### Status

Unfinished work is flagged next to the claim it qualifies ("RL hedging is still in development"), since a reader of a highlight may never reach the status table. The flag goes away in the change that finishes the work.

### Counts that go stale

A test count or a line count is wrong after the next commit. Prefer a description ("unit, integration and CLI smoke tests"). If a count really helps, take it from the command (`python -m pytest --collect-only -q`) and keep it in one place.

## docs/RESULTS.md: the index

One section per topic, in the README's order:

```markdown
## <Topic>

<One or two sentences: the question this topic answers.>

| <the key table, a few rows> |

<The takeaway in one or two sentences.>

Full analysis: [surrogate-training.md](results/surrogate-training.md).
```

**Why an index.** A reader sees every result on one page, and each topic file stays focused. Links use `[file name](relative path)`.

## docs/results/<topic>.md: paper-style analysis

Each topic file reads like a short paper on one question:

| Section | Holds |
|---|---|
| Setup | the question, the data and domain, the reference it is scored against, the metrics and why each one |
| Results | the tables, each with its command above and its protocol line under it |
| Discussion | what the results mean, and the decision taken and why |
| Limits | what the results do not show, open questions, known costs |

- **Keep the decisions, not every run.** An ablation earns a table when it supports a decision. Per-run logs stay in git history and in `artifacts/runs/`. A sentence like "width 256 did not beat 128 on 3 seeds (table)" is enough for a rejected variant.
- **An introduction in prose opens the page.** A few paragraphs on the problem, the approach and what the page holds, so a reader knows why it exists before the first table. Earlier work that the topic replaces gets a sentence or two of context, and the page stands on its own rather than as a diff against it.
- **A contents table follows it**, one row per section: a link to the section and a one-line summary of what it decides or finds. A topic with several decisions puts its decision in the summary, so the table still works as "the decisions at a glance".
- **Reasons read as prose.** A bold lead-in marks a reason a skimming reader looks for (`**Why not LSM.**`). A section that repeats the same three labels (Choice, Why, What it implies) reads like a form, so let the text flow and keep a label only where it helps.
- **A "Reproducing the runs" section** at the end lists the commands with their cost.

## docs/architecture.md: the map

```markdown
# Architecture

<One paragraph: what the package contains and how to read this page.>

<The layer map in a code block, then one sentence on the dependency direction.>

## <package>/

<One or two sentences: the package's role.>

| File | Role | Explained in |
|---|---|---|
| [tree_pricer.py](../optionpricing/ml/surrogate/tree/tree_pricer.py) | `PutSurrogate`, prices a put from the network | [surrogate-training.md](results/surrogate-training.md) |
```

- **One line per file.** A row says what the file is for, not its API. Signatures, shapes and defaults live in docstrings, where they are checked against the code.
- **"Explained in"** points to the tracked doc that gives the reasons: a results doc, a BUGS.md entry or the README section. `notes/REMINDER.md` is gitignored, so other readers don't have it. Leave the cell empty rather than cite it.
- **Commands and tests** get short tables too (command, script, what it does; test folder, what it covers).
- **Before moving an API detail out of this page**, check that the docstring already says it.

## Commands in any project doc

- Fenced block, run from the project root, an end-of-line `#` comment for what it produces or costs: `# artifacts/data/tree_put_dataset.npz, ~40 min`.
- Modules run as modules: `python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`, never a script path.
- Flags and defaults come from `--help` at write time. Say whose default it is (function or CLI) when they differ.

## Math in project docs

Formulas and symbols are LaTeX, in GitHub's code-span forms:

````markdown
The proportional cost is $`\varepsilon \lvert x \rvert S`$, at $`\varepsilon`$ = 1%.

```math
\large \mathrm{PL} = p_0 + \sum_k \delta_k \big(\tilde S_{k+1} - \tilde S_k\big) - \sum_k \tilde c_k - \tilde Z
```
````

- **Why code spans.** GitHub applies Markdown's backslash escapes inside plain `$...$` and `$$...$$` before rendering, so `\{`, `\,`, `\;` and `\%` lose their backslash, and `S^*` can open an italic. Inside $`...`$ and a ```` ```math ```` block the LaTeX reaches the renderer untouched. Checked with GitHub's markdown API.
- **`\large` in display blocks.** GitHub renders display math at text size, small for a formula with sums and fractions. `\large` makes the main formulas read at a glance. Inline math stays at text size, so a sentence keeps its line height.
- **Values stay outside the math:** $`\sigma`$ = 20%, $`T`$ = 1 year. Units and percentages are text, and the formula holds only symbols.
- **Dollar amounts stay plain** ($0.21). A `$` followed by a digit does not open math on GitHub, so it can sit next to math in the same sentence.
- **Code is not math.** A name from the code stays in plain backticks (`init_logit`), so the reader can search for it.
