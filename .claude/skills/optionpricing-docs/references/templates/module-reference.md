# Template: module / API reference

The API reference lives in **docstrings**. The README's Architecture section gives each module one bullet that points there. Only write a standalone reference page if the user asks for one; it then follows the "Standalone page" layout below.

## Docstring: function

```python
def name(arg1, arg2, option: str = "default") -> ResultType:
    """
    <One or two sentences: what it computes, and the method used.>

    <Optional: vocabulary, units and scaling ("per unit of strike (K = 1)"),
    shapes and broadcasting, and why a non-obvious choice was made.>

    Args:
        arg1: <meaning, units, shape; broadcasting if any>.
        arg2: <…>.
        option: <each allowed value and what it does, and the default's reason>.

    Returns:
        <Type(field, …)>: <what each field is, in which units>.

    Raises:
        ValueError: <the condition, and why it is refused rather than handled>.
    """
```

## Docstring: NamedTuple / result class

```python
class Result(NamedTuple):
    """<What it holds, and its units, e.g. "all per unit of strike">.

    field_a: <meaning>. field_b: <meaning, with scaling such as "times K">.
    """
```

## Docstring: mode argument with several values (`HedgingEnv` style)

```
mode:
    "a" (default): <behaviour, and what it is used for>.
    "b": <behaviour, and what it makes possible that "a" doesn't>.
```

## Constant

```python
# <why this value; what goes wrong if it is changed>
_NAME = 1e-12
```

## Standalone page

```markdown
# `<package/module.py>`

<Purpose in one paragraph, and where it sits: what calls it, what it calls.>

## Public API

### `function(arg, …) -> Type`
<Summary.> Args / Returns / Raises as in the docstring. Keep the two in sync, or link to the docstring.

## Invariants
- <Something callers must do first, e.g. "call `predict_payoff()` before `calculate_option_price()`".>

## Example

Run from the project root (imports are absolute: `from core.gbm import GBM`).
Save it inside the repo and run `python <file>.py`, or run it from elsewhere
with the root on the path (`PYTHONPATH=. python <file>.py`). It takes ~<measured time>.

```python
<a minimal snippet that prints something checkable>
```

Output (<versions, seed, data size>):

```
<the actual output of running it>
```

## Tests
`tests/unit/test_<module>.py`: <what they pin down>.
```

Check every signature, default and exception message against the source.
