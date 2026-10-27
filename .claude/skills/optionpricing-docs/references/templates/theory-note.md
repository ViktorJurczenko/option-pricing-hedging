# Template: theory / design note (REMINDER.md style)

Use this for a note that explains a feature from first principles to the code, and why each choice was made. It describes the code as it is now. No dates, step numbers or "as built" wording.

```markdown
# <Topic, from the ground up> (`<main/file.py>`)

<One paragraph: what this part explains, the code it covers, and that all numbers come from running the code (name the command or test).>

## 0. The problem

<The concrete situation in plain words. Define each symbol on first use: "the spot `S` (today's stock price)".>
<Why the easy approach doesn't work. That gap is what the rest of the note fills.>

## 1. <First idea>

<Plain-language idea, or a tiny concrete case.>

<What is being shown, and why it is plausible.>
```
<derivation, one step per line>        <justification>
```

<What the result means and where it is used later.>

<Numeric check: plug in the running example (e.g. `σ = 0.2`, `r = 0.05`, `T = 1`) and give the number.>

In the code (`<file.py>`), with `<code_name> = <symbol>`:

```python
<the 1–5 lines that implement it, quoted exactly>
```

### <Assumption or edge case>

<When the formula breaks, and what the code does about it (the guard, quoted).>

## 2. <Next idea, building only on sections 0–1>

…

### Why not <alternative>?

<The alternative, the measured reason it loses, and the command or test that shows it.>

## <N>. What is checked (`tests/unit/test_<module>.py`)

| Property | How it is checked | Tolerance / result |
|---|---|---|
| <invariant> | <test name or method> | <number> |

## <N+1>. Limits

<Where the method's accuracy degrades, in numbers. Open questions, clearly labelled.>
```

Rules for this type:
- Follow the order idea, then definition, then example, then why it's needed.
- Never use a concept before its section.
- Formulas go in code fences in Unicode, each followed by the code line that implements it.
- Use the symbols from `notation.md`. If one is overloaded (`u`, `w`), say which meaning you intend.
- Results quoted here also give their settings.
