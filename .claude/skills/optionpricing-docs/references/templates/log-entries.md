# Templates: logs (BUGS.md, PLAN.md, experiment results)

These are dated records. History belongs here, not in explanation docs.

## `.claude/BUGS.md` entry (newest first, under the file's intro)

```markdown
## <Symptom in a short sentence> (fixed | open, for later)

**Symptom:** <what went wrong, with numbers and the command or test that showed it>.

**Root cause:** <the mechanism>.

**Impact:** <what was affected, and which earlier results are invalid>.

**Fix:** <the change and its file(s)>; pinned by `<test_name>`, which fails if the fix is removed.
```

For an open problem, replace **Fix:** with **Status:**, giving what is known and the next thing to try.

## `.claude/PLAN.md` entry (newest on top)

```markdown
## <Feature or issue>

<Context: the problem, the options considered, and which one was chosen and why.>

- [ ] <Step title>
  <What it does and why.>
- [x] <Done step title>
```

When a step is done, drop its description, unless it records measured results that later steps need (for example the step-9 learning curve).

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

For experiment drivers (`experiments/run_ablations.py`), name the output directory (`runs/ablations/summary.md`). Never paste a whole log into a write-up.
