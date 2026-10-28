# Examples to imitate

These are pointers to the parts of existing docs that passed the audit (`AUDIT.md`). Each pointer has a short excerpt showing the pattern. Open the full section when writing that type of doc; line numbers are approximate, so search for the heading.

## Theory / design note: REMINDER.md

**Best section:** `# The batch binomial tree, from the ground up`, sections 0–10. It starts from "what is a price" and builds up to the code, defining each symbol on the way.

The pattern is the formula, then the code line, then the code-to-symbol mapping:
```
V = e^{−r·dt} · ( p·V_u + (1 − p)·V_d ),     p = (e^{r·dt} − d) / (u − d)
```
> In the code, with `growth = e^{r·dt}` and `d = 1/u` (section 2):
> ```python
> p = (growth - 1 / u) / (u - 1 / u)  # risk-neutral probability of an up move
> ```

Assumptions and failure modes are stated, together with the guard that enforces them:
> `p` is a probability only if `d < e^{r·dt} < u`. … The code refuses rather than return garbage:

**Also good:**
- `## The new model, part by part`. A design choice is explained through its target, its inputs and its loss. Rejected alternatives appear as "Why not X?" with measured evidence.
- `# How the RL hedging policy works`, from "What the RLPolicy network outputs" on. It uses a pipeline diagram in a code block, and links to code lines as `[policy.py:67-69](ML/rl/policy.py#L67-L69)`.

**Do not imitate in REMINDER.md:**
- the dated "Personal notes from … after commit …" headers;
- "as built (step N)" headings;
- "Where things stood";
- the self-corrections;
- `# Is the MLP surrogate pricer actually useful?`, which is about the replaced model.

## README: README.md

**Results subsections.** Each is a command block, then the protocol in one sentence, then a table, then bullets that interpret it:
> ```
> python main.py test-hedge --policy rl
> ```
> Each policy's deterministic action, evaluated over the same 2,000 simulated hedging episodes (`--seed 999 --n_episodes 2000`) with a 0.1% proportional transaction cost.

**Problem write-up.** `### LSM delta/gamma accuracy near maturity` gives a plain explanation of the method, then the symptom in numbers, then `**Why:**`, `**First fix, later reverted:**` and `**Current fix:**`, then a before/after table. This dated narrative is fine in a README results section. It does not belong in a theory note.

**Usage.** A command table (`| Command | Purpose |`) followed by fenced invocations with `#` comments.

**Caveats** (see `AUDIT.md`): the test count, the `--reg_type` sentence at L199–201, `HedgeEnv` at L85, and L159–160 on the tree surrogate are wrong or stale. Don't copy them.

## Decision record: the removed docs/decisions/softplus-reparameterization.md

No decision record is in the repo now; this one was removed and its incident moved to `.claude/BUGS.md`, "ML surrogate pricer's target had a self-inflicted kink at ATM". Its shape is still the model for a new one. Its sections were `## Symptom`, `## Root cause`, `## How this was confirmed (not undertraining)` and `## Fix`. The first paragraph links the general pattern it is an instance of:
> Background and the reusable pattern this generalizes to live in `.claude/CLAUDE.md` under "Softplus reparameterization…". This file is the incident this pattern was extracted from.

(The pattern itself has since moved to `docs/architecture.md`, "Softplus reparameterization"; a new record links it there.)

The evidence section rules out the alternative explanation:
> more epochs fixing everything *except* the kink region rules out "just needs more training."

## Docstrings

- `AmericanOptionPricing/BatchBinomialTree.py::binomial_put_batch`. It opens with prose, then a "Vocabulary" paragraph, then a `Methods:` list explaining each option and why it exists, then the units ("Scale price and gamma back with K * price, gamma / K").
- `ML/surrogate/tree/tree_pricer.py::PutSurrogate.price`. A compact `Args:`/`Returns:`/`Raises:` block, where the `Raises:` entry says *why* the error happens.
- `TreeResult` (NamedTuple) documents each field with its units in the class docstring.
- `Hedging/env.py::HedgingEnv.__init__`. A named-value block for a mode argument, explaining what each mode makes possible.

## Incident: .claude/BUGS.md

A typical entry is `### B-031 Early stopping cut longer cosine schedules short`, with its `Status: · Found: · Files:` line, then the parts `**Symptom:**` (with numbers), `**Root cause:**`, `**Impact:**` and `**Fix:**` (naming the file and the test that pins it, and saying that the test fails if the fix is removed). The file's header holds the template and the rules for IDs and the index.

## Writing rules: course/CRITIQUE.md §2

Use it as a checklist for explanation docs: the order of a concept, no concept before it is taught, derivations line by line, proofs explained, and the table for deciding on forward references.
