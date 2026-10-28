# Templates: logs (BUGS.md, PLAN.md, experiment results)

These are dated records. History belongs here, not in explanation docs.

## `.claude/BUGS.md` entry

The template and the rules (next free ID `B-0NN`, a row in the index, the entry at the top of its area's section, titles never renamed) are in the header of `.claude/BUGS.md` itself; follow that, it is the single source.

## `.claude/PLAN.md` entry and `.claude/PLAN_ARCHIVE.md` summary

The roadmap's format (a table row per workstream; **Goal** then `- [ ]` steps for active work; **Goal** / **Done so far** / **Next** for the backlog) is in the header of `.claude/PLAN.md`, and the archive's summary format in the header of `.claude/PLAN_ARCHIVE.md`. Measured results never stay in a plan step: they go to `docs/results/<topic>.md` (format below) and the step links them.

## Experiment / results write-up (README Results subsection or an ablation summary)

```markdown
### <Finding as the title>

```
<exact command(s)>
```

<Protocol: data size, seeds, epochs or episodes, reference or grid, hardware if timing is reported.>

| Variant | <metric (units)> | … | train time |
|---|---|---|---|
| baseline | mean ± sd over <n> seeds | … | … |

- **<Finding.>** <Numbers.>
- <What it does not show, and what is still open.>
```

For experiment drivers (`optionpricing/ml/experiments/run_ablations.py`), name the output directory (`artifacts/runs/ablations/summary.md`). Never paste a whole log into a write-up.
