# Template: how-to (goal-oriented CLI recipe)

Use this to get one specific result, for example "Train and evaluate the tree surrogate" or "Compare hedging policies under a 1% cost". It is usually a README Usage subsection.

```markdown
### How to <goal>

<One sentence: what you get at the end, and roughly how long it takes on CPU.>

**Prerequisites:** <files or checkpoints that must exist, and the command that creates each>.

1. <Action.>
   ```
   python main.py <cmd> <flags>     # writes <path>; ~<time>
   ```
   <What to look for in the output.>

2. <Action.>
   ```
   python main.py <cmd> <flags>
   ```

**Keep these consistent:** <flags that must match across steps, e.g. `--split_ratio`/`--seed` between `train-ml` and `test-ml`; `--prop_tc`/`--fixed_tc` between `train-hedge` and `test-hedge`>.

**If it fails:** <the most likely error and its cause, e.g. running a generator by script path instead of `python -m`>.
```

Rules:
- One goal per how-to.
- Theory goes elsewhere: link the theory note instead.
- Commands are run from the project root, with flags checked against `--help`.
