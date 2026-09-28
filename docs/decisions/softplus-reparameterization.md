# Why `smooth_intrinsic_value` uses a softplus, not exact intrinsic value

Background and the reusable pattern this generalizes to live in
`.claude/CLAUDE.md` under "Softplus reparameterization: general pattern for
ML regression targets". This file is the incident this pattern was
extracted from.

## Symptom

The MLP surrogate pricer (`AmericanOptionPricing/ML.py`) is trained on
`time_value = (price - intrinsic_value) / K` rather than raw price, to
shrink the target's dynamic range. Early versions used the exact intrinsic
value, `max(K - S, 0)` for a put. This produced a ~16% average error
concentrated right at the money (ATM), well above the error elsewhere in
the moneyness range.

## Root cause

`intrinsic_value = max(K - S, 0)` has a hard kink at `S == K` — that's the
*expiry payoff* shape. But the American put's actual price *before* expiry
is smooth there: diffusion smooths out the kink, and the real early-exercise
kink lives at the exercise boundary, which is a variable level below `K`,
not at `K` itself. Subtracting the exact intrinsic value therefore
re-introduced a kink into the training target exactly at the money that the
true price does not have — the model's error spike at ATM was an artifact
of the label definition, not evidence that ATM options are intrinsically
harder to price.

## How this was confirmed (not undertraining)

- Training on the raw price directly (no intrinsic-value subtraction)
  removed the ATM spike but hurt accuracy elsewhere (the dynamic-range
  problem the subtraction was meant to solve).
- More training epochs narrowed the ITM/OTM error but barely touched the
  ATM error.

Both are the signature of a structural target-definition problem, not
insufficient training — more epochs fixing everything *except* the kink
region rules out "just needs more training."

## Fix

Replace the exact analytic component with a smooth approximation
(`smooth_intrinsic_value`, a softplus) that converges to the exact value
away from the kink but rounds off the corner over a small window
(`beta_frac`, default `0.06`, fixed in code rather than a CLI flag so it
can't drift out of sync between label generation and reconstruction). The
same smooth version must be used everywhere the component is added back to
reconstruct the true value (`evaluation.py::test_option_price_lsm`) —
training and reconstruction must agree or predictions will be
systematically biased.

The smoothing width was picked by sweeping a few values against a fixed
benchmark, with training runs fully seeded (model init, data split, batch
shuffling) so the comparison isolated the hyperparameter's effect from
run-to-run training variance.
