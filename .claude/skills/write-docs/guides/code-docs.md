# Code docs: docstrings and comments

Docstrings state the contract: what goes in, what comes out, what is refused. The reasons behind a design live in `notes/REMINDER.md`, and the evidence in `docs/results/` and `.claude/BUGS.md`. **Why split them.** A docstring is read at the call site, where the reader wants the contract fast. A long design essay there hides it, and goes stale without anyone noticing.

Lines wrap at 120 characters (`pyproject.toml`). Docstrings use the Google style (`Args:`, `Returns:`, `Raises:`).

## Module docstring: one paragraph

Say what the module is for and where it sits in the pipeline: what feeds it, what uses it.

```python
"""Exercise dates of the American puts in a scenario bank, from the binomial tree's early-exercise boundary.

The hedge dataset generator (`generate_hedge_dataset.py`) calls it once per scenario bank, and settlement
(`settlement.py`) later pays each put on the day it returns.
"""
```

A list of the module's modes or concepts can follow when a reader cannot use the module without it (the holder types of `exercise.py`). Keep their reasons for REMINDER.

## Function and method docstring: the contract

```python
def daily_boundary(bank: ScenarioBank, nb_steps: int = 1000, chunk_size: int | None = None) -> np.ndarray:
    """Early-exercise boundary `S*` of each contract on every day to its expiry, in dollars.

    Args:
        bank (ScenarioBank): the scenario bank whose contracts are used.
        nb_steps (int): tree steps per contract, as in `exercise_boundary`.
        chunk_size (int | None): contracts stepped together per pass. Only affects speed and memory.

    Returns:
        np.ndarray: shape `(n_contracts, max(days) + 1)`. Entry `k` is `S*` on day `k`, the strike at expiry,
        NaN after it.

    Raises:
        ValueError: for an invalid `nb_steps` or `chunk_size < 1`, as `exercise_boundary`.
    """
```

| Part | Default | Why |
|---|---|---|
| Summary | 1-3 sentences: what it computes, in which units | The reader decides from it whether this is the function they need |
| `Args:` | `name (type): meaning, units, valid range.`, lowercase after the colon | The type and range at a glance, without reading the signature and the guards |
| `Returns:` | type, shape, units, and what special values mean (NaN, empty) | Shapes and sentinels are the most common misuse |
| `Raises:` | the exception and its condition | Callers need to know what is refused |

**What a caller could get wrong goes in.** Required call order (`predict_payoff()` before `calculate_option_price()`), units and scaling ("per unit of strike. Scale back with `K * price`, `gamma / K`"), shapes and broadcasting, side effects (writes a file, mutates its input).

**When a short why belongs in the docstring.** When the caller needs it to use the function correctly. "`lsm` fits the rule on the paths it is applied to, so it sees their future" changes how a caller reads the results. A sentence is enough, with a pointer to the tracked doc that has the evidence.

**Trivial helpers** (a private one-liner, an obvious property) get a one-line docstring and no sections.

### Mode arguments

A parameter with a few named values gets a named-value block, one line per value:

```python
        method (str): how the holder exercises.
            "tree" (default): the first day at or below the tree's boundary.
            "late": the boundary shifted down by `shift`, so later than optimal.
```

### NamedTuple and result classes

Document each field, with its units, in the class docstring (`batch_binomial_tree.py`):

```python
class TreeResult(NamedTuple):
    """Per-option outputs of `price_put_batch`, all per unit of strike.

    price: option value / K. delta: dV/dS (the same number for any K, since
    V is homogeneous of degree 1 in (S, K)). gamma: d2V/dS2 times K.
    exercised: True where the put is exercised today (worth its payoff
    1 - m, with m < 1), which fixes price, delta and gamma to 1 - m, -1 and
    0 exactly; always False for a European put.
    """
```

## Constants

A module constant gets a comment on why it has that value (`batch_binomial_tree.py`):

```python
# price - (1 - m) at or below this means "exercised today": the root's
# exercise value is 1 - m computed from a spot rebuilt by multiplications,
# so it can differ from 1 - m by rounding (~1e-16)
_EXERCISE_TOL = 1e-12
```

## Inline comments: moderate

- **A short header per logical block** of a long function: `# Backward induction, one level at a time.` It lets a reader skim the function's structure.
- **The why on anything a reader would question:** a guard, a tolerance, a non-obvious formula, a workaround.
- **Not a narration of what the next line does.** `# increment i` adds nothing.
- **A measured number cites where it was measured:** a `.claude/BUGS.md` entry by its id, as (.claude/BUGS.md, B-031), or a tracked doc, never the gitignored `notes/REMINDER.md`.

```python
    # The bank's daily grid, shared by every contract. These are the days the boundary is wanted on.
    days = np.arange(bank.spot.shape[1])
```

## Existing docstrings

Some docstrings in the repo are longer than this default, with design essays (`batch_binomial_tree.py`, `exercise.py`). Don't rewrite them in a change that only passes by. When you change a function, bring its docstring to this default, and move the reasons it drops to the matching REMINDER chapter so they are not lost.

## Checklist

- [ ] Every new or changed module, public function and class has a docstring.
- [ ] Args carry type, units and range. Returns carry shape and special values. Raises lists what is refused.
- [ ] Call order, units, shapes and side effects are stated where a caller could get them wrong.
- [ ] Comments explain why, head long blocks, and cite tracked sources for numbers.
