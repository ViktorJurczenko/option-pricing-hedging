# Template: README.md (or a new README section)

The section order matches the existing README.md. When adding a section, put it in the right slot rather than at the end.

```markdown
# <Project title>

<2–3 sentences: what the project builds, from what, and for which reader.>

<One sentence on the "from scratch" principle, if it applies to the new part.>

## What's inside

| Layer | What it does |
|---|---|
| **<Layer>** | <One or two sentences. Say whether it is wired into the other layers or standalone.> |

## Results

<One sentence: the default parameters behind all numbers below (`S0=K=100`, `r=5%`, `T=1y`, `σ=20%`), and that you can reproduce them with the commands shown.>

### <Result title, stated as the finding: "Pricing models agree with each other">

```
python main.py <command> <flags>     # <what it prints / how long it takes>
```

<Protocol in one or two sentences: seed(s), episodes or paths, costs, what a range means.>

| <rows> | <metric (units, e.g. ×K)> | … |
|---|---|---|
| … | **best** | … |

- **<Finding in bold.>** <Interpretation with the numbers.>
- <A limit or hypothesis, labelled as such.>

<Optional, for a problem that was fixed: **Why:** / **Current fix:**, then a link to `.claude/BUGS.md`, "<entry title>".>

### Test suite

```
python -m pytest tests/
```

<What the unit and integration tests cover. If you give a count, take it from `pytest --collect-only -q` and date it.>

## Setup

```
pip install -r requirements.txt
```

<Python and PyTorch versions from requirements.txt. GPU optional.>

## Usage

<Run from the project root. Imports are absolute.>

| Command | Purpose |
|---|---|
| `<cmd>` | <Purpose plus its key flags, checked against `--help`.> |

<Separate module entry points: `python -m <module>` with a fenced flag synopsis.>

## Testing

```
python -m pytest tests/unit
python -m pytest tests/integration
```

## Architecture

- **`<package>/`**: <role>. <Key files, one clause each.>
```

Before finishing, check:
- every number has its command and protocol;
- every flag and default comes from `--help`;
- every path exists.
