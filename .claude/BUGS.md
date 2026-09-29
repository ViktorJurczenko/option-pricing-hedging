# BUGS.md

Log of problems/bugs found while working on this project — indicative of
real issues encountered and their resolutions, not a work-planning tool.
See `.claude/CLAUDE.md`'s "Bug tracking" section for the policy. New
feature implementation that isn't a bug or a fix does not belong here; see
`.claude/PLAN.md` for that.

## LSM delta/gamma estimates unreliable near maturity (open)

**Symptom:** `LongstaffSchwartz.delta` (and by extension `.gamma`) falls
outside a call's theoretically valid `[0,1]` range a meaningful fraction of
the time, worse near maturity: 2% invalid at t=50 climbing to 15.3% invalid
at t=363 (the last step before expiry), with values reaching into the tens
(and, at higher `max_degree`, into the thousands). Feeds directly into
`AmericanHedging/HedgeEnv.py`'s state (`self.lsm.delta[t_clamped, i]`) and
into the `delta_hedge` baseline (`AmericanHedging/baselines.py`), which
returns this value unclamped — since `HedgeEnv.step()` only clips the
executed action to `[-1,1]` (not `[0,1]`), a negative delta near maturity
can execute as an actual short position, the wrong trade direction.

**Root cause:** delta/gamma come from the analytic derivative of a global
polynomial (degree 3 by default) fit via least squares to the continuation
value. Near maturity the true value function's kink at `S=K` sharpens
toward the discontinuous step-function payoff; a low-degree global
polynomial can't bend that sharply, so the fit wiggles slightly to
compromise, and differentiating amplifies that wiggle into large swings in
the reported Greek. Confirmed via two ruled-out fixes: `reg_type=laguerre`
produces results identical to `simple` to 10 significant figures (same-
degree polynomial bases span the same function space, so basis choice can't
change the fit), and raising `max_degree` makes it dramatically *worse*
(Runge's phenomenon — more degrees of freedom to oscillate, amplified
further by each derivative order): overall validity drops from 89% at
degree 3 to 65% at degree 5 to 50% at degree 7.

**Status:** investigating a real fix (ridge-regularized regression is the
leading candidate; also considered: local/piecewise regression by
moneyness, finite-difference Greeks, a parametric bounded functional form)
rather than a clamp/soft-clamp band-aid, which would only bound the output
without correcting the underlying estimate. See `.claude/PLAN.md`'s "Fix
unreliable LSM delta/gamma estimates near maturity" entry for the live
task list.

## ML surrogate pricer's target had a self-inflicted kink at ATM (fixed)

**Symptom:** the MLP surrogate pricer (`AmericanOptionPricing/ML.py`)
showed a ~16% average error concentrated right at the money, well above
error elsewhere in the moneyness range.

**Root cause:** the training target is `time_value = (price -
intrinsic_value) / K`, and `intrinsic_value = max(K-S, 0)` has a hard kink
at `S == K` — the *expiry payoff* shape — but the American option's actual
price *before* expiry is smooth there. Subtracting the exact intrinsic
value re-introduced a kink into the training target exactly at the money
that the true price does not have, and the model's error spiked there as
an artifact of the label definition, not because ATM is intrinsically
harder to price.

**Fix:** replaced the exact intrinsic value with `smooth_intrinsic_value`,
a softplus approximation that converges to it away from the kink but
rounds off the corner over a small window (`beta_frac`, fixed at `0.06`).
Used consistently everywhere the component is added back to reconstruct
the true price. Full incident writeup (symptom, ablation, root cause) is
in `docs/decisions/softplus-reparameterization.md`; the general pattern
this generalizes to is documented in `.claude/CLAUDE.md`.
