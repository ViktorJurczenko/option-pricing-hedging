# Conventions

These conventions come only from sources that passed a documentation audit. The source of each rule is given in brackets.

Sources:
- **[R]** `notes/REMINDER.md`, its trusted parts;
- **[RD]** README.md;
- **[C]** notes/course/CRITIQUE.md §2 and §5, the author's own writing rules;
- **[D]** the softplus decision record, formerly `docs/decisions/softplus-reparameterization.md` (removed; its incident is now in `.claude/BUGS.md`, "ML surrogate pricer's target had a self-inflicted kink at ATM");
- **[DS]** docstrings in `optionpricing/pricing/american/batch_binomial_tree.py`, `optionpricing/ml/surrogate/tree/tree_pricer.py`, `optionpricing/hedging/env.py` and `optionpricing/core/gbm.py`;
- **[B]** the BUGS.md entry format.

## Contents
1. Audience and voice
2. Structure of an explanation
3. Math
4. Code in docs
5. Results and reproducibility
6. Docstrings
7. Naming, headings, links, file locations

## 1. Audience and voice

- **Reader.** Write for someone strong in programming who is new to finance, the yardstick the author set: "a sentence is a defect if that reader cannot follow it with only what came before it" [C intro]. So a finance or statistics term is best given by its standard name rather than a paraphrase ("the early-exercise premium"), and defined at first use unless that reader can be expected to know it ("out of sample, that is when the rule is fitted on one set of paths and applied to fresh ones"). An object the text comes back to reads more easily with a label ("the scoring grid"). The README reader wants to run commands and see measured results [RD].
- **Voice.** Prefer plain, declarative, present-tense sentences with a subject and a verb, the opening line included ("This project aims at pricing…" reads better than "Pricing and hedging…, built from…"). Fragments read as notes, which suits a table cell or a list item more than prose. A sentence is short because it says one thing, not because words were cut. Dropping a repeated verb ("the plan is in X, finished work in Y") leaves the reader to rebuild the sentence. A colon is the natural way to introduce a list or an explanation. Joining two separate ideas with a colon or a semicolon is better kept for the rare case where they really belong together, and two sentences usually read better. For person, "we" suits a derivation the reader walks through ("We want its fair price today") [R §0]. Elsewhere the impersonal form ("One can price with it…") or the author's "I" stating their own purpose both fit, whichever reads more naturally [RD].
- **Why before how.** Each design choice states its reason in the same place [R, "Why these values", "Why `d = 1/u`"; RD "Why:"]. A bold lead-in followed by the explanation is the usual pattern: `**Why these values.** …`, `**Current fix:** …` [R §2; RD L92–108]. Metrics and components benefit from a reason too, such as what a metric is for ("to compare errors on options of very different prices") and why a component exists (the paper or the problem it comes from) [RD].
- **Length is fine, density is not.** Put one new idea per paragraph [C 2.1]. A longer section that explains is better than a short one that skips steps [C 2.1].
- **Honest limits.** Say what is a hypothesis and what is measured ("This is a hypothesis, not yet verified") [RD L293–296]. State known costs of a choice ("a known, accepted cost of that fix") [RD L236]. Unfinished work is best flagged next to the claim it qualifies, since a reader of a highlight may never reach the status table. A section on work in progress can say so in its first lines, and a choice meant to change can say it is temporary ("for the moment") [RD].
- **No filler.** Don't write "it is worth noting", "clearly" or "it follows that" in place of steps [C 2.3].

## 2. Structure of an explanation (theory notes, explanation sections)

- **Introduce a concept in this order** [C 2.1]:
  1. a plain-language idea or concrete situation;
  2. the precise definition;
  3. a small numerical example;
  4. why it is needed later.

  REMINDER §0–1 follows this order: the situation, then the one-step replication, then the formula, then "Three things to notice".
- **Never use a concept before it is taught.** "Made precise later" is not an explanation [C 2.2].
- **Number the sections in build-up order** (`## 0. The problem`, `## 1. The one-step tree…`), so each section depends only on earlier ones [R].
- **Forward references.** Keep one only if the current passage needs it, and give it a one-line explanation in place. Remove name-drops [C 2.4]. Backward references name the section: "(section 8)" [R §1].
- **End with what is checked.** Name the tests that pin the behaviour and say what each one checks [R §10, "What is checked (`tests/unit/test_batch_binomial_tree.py`)"].
- **Present rejected alternatives as alternatives.** Write "Why not X?" with the reason, not as a story of what was tried first [R "The new model" §1, "Why not 'price − intrinsic value'?"].

## 3. Math

- **Define every symbol where it first appears**, including the "obvious" ones [C 2.3]. Use the symbols in `notation.md`.
- **Put formulas in fenced code blocks** in plain Unicode (`σ`, `√`, `Δ`, `·`, `−`, `e^{r·dt}`), not LaTeX, one formula per line, aligned when there are several [R §1–2]:
  ```
  V = e^{−r·dt} · ( p·V_u + (1 − p)·V_d ),     p = (e^{r·dt} − d) / (u − d)
  ```
  Inline math goes in backticks: `` `m = S/K` ``.
- **Give the formula, then the code that implements it**, quoted from the source, with the code's variable names [R §1: the `p` formula followed by `p = (growth - 1 / u) / (u - 1 / u)`]. Say how the code names map to the symbols ("with `growth = e^{r·dt}`").
- **Derive line by line.** Each step goes on its own line with a short justification. Before a derivation, say what is being shown. After it, check it on a number [C 2.3, 2.3b]. Prefer a table or a numeric example when that is enough to make the argument [C 2.3b].
- **State the assumptions and where they break.** Example: "`p` must be strictly between 0 and 1 … the code refuses rather than return garbage" [R §1].
- **Give numerical-method caveats in numbers.** Examples: the zig-zag of CRR, the bias of each method, Monte-Carlo noise ("about ±0.1 here") [RD L49; R §6–7, §11].
- **State units every time.** Write "per unit of strike (K = 1)", "×K", "$ at K = 100" [DS `TreeResult`; RD tables].

## 4. Code in docs

- **Show commands as a fenced block run from the project root**, with an end-of-line `#` comment for what each produces or costs (`# artifacts/data/tree_put_dataset.npz`, `# ~40 min at defaults`) [RD Usage].
- **Run modules as modules:** `python -m optionpricing.ml.surrogate.tree.generate_tree_dataset`, never a script path [RD].
- **Name code in backticks:** `` `binomial_put_batch` ``, `` `HedgingEnv` ``. Write `path.py::function` for a specific function [D].
- **Link to code lines** in theory notes, relative to the repo root: `[policy.py:67-69](optionpricing/ml/rl/policy.py#L67-L69)` [R RL part]. Check the line numbers at write time; they go stale.
- **Short excerpts only.** Quote the few lines that carry the idea, not whole functions [R §1–2].

## 5. Results and reproducibility

- **Put the command that produces a result right above it** [RD Results sections].
- **Give the protocol in one sentence:** seed, number of episodes, paths or grid, the cost settings, and what a range means. For example: "RL rows give the range over 4 training seeds" [RD L255–262, L318–320].
- **Results go in tables** with units in the header or caption, and the best value in bold [RD]. Write ranges as `a-b` or `a to b`.
- **Explain any metric a reader could misread**, for example "Mean P&L … isn't a hedge-quality metric" and "CVaR (mean of the worst 5%)" [RD L270–274].
- **Pin the environment** in `requirements.txt` and state it in Setup (Python 3.13, PyTorch 2.11 CPU build) [RD].
- **Say how a number was measured and on what data:** "(measured)", "on 4 seeds" or "against a 16,000-step reference" [R §11; RD]. An error or a rate describes the data it was measured on rather than the model as a whole, so "the largest error on the 2342 scoring puts" is more accurate than "the model's worst error". A derived share, such as a violation rate, is easier to trust when the text says what is checked and how a point is counted [RD].

## 6. Docstrings

- **Format.** Start with a summary paragraph in prose. Then give `Args:`, `Returns:` and `Raises:` sections (Google style) [DS `binomial_put_batch`, `PutSurrogate.price`, `GBM.simulate_paths`]. For a long parameter list, a named-value block is also used (`reward_mode: "terminal" (default): …`) [DS `HedgingEnv.__init__`].
- **Say what is not obvious:** units and scaling ("per unit of strike … Scale price and gamma back with K * price, gamma / K"), array shapes, broadcasting, and why a guard exists [DS].
- **NamedTuple results** document each field in the class docstring [DS `TreeResult`, `SurrogateResult`].
- **Module constants** get a comment saying why they have that value [DS `_EXERCISE_TOL`, `HedgingEnv._TRADE_EPS`].
- **Name other code in backticks**, as in `` `binomial_put_batch` `` [DS].

## 7. Naming, headings, links, file locations

- **Headings.** `#` for a part or topic, `##` for sections, `###` for subsections. Use sentence case. A heading may contain a path in backticks: `# The batch binomial tree, from the ground up (\`optionpricing/pricing/american/batch_binomial_tree.py\`)` [R].
- **Places:**
  - README.md: user-facing.
  - `notes/REMINDER.md`: theory and design notes. `notes/` is gitignored, as personal notes.
  - `docs/decisions/<kebab-slug>.md`: decision records.
  - `docs/architecture.md`: the module reference; `docs/results/<topic>.md`: measured results.
  - `.claude/BUGS.md`, `.claude/PLAN.md` and `.claude/PLAN_ARCHIVE.md`: logs.
- **Cross-document links.** Name the target doc and section in quotes, e.g. `See .claude/BUGS.md, "LSM early-exercises American calls"` [RD]. Decision records start by linking the general pattern they illustrate [D].
- **CLI names.** Commands are kebab-case (`train-ml`). Flags are snake_case with `--` (`--option_type`) [`main.py --help`].
