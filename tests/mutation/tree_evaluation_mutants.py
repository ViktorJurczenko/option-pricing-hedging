"""
Hand mutation check of ML/surrogate/tree_evaluation.py: each mutant is one
small, plausible bug (a flipped comparison, a dropped threshold, a wrong
factor); every one must make tests/unit/test_tree_evaluation.py fail. A
mutant that survives is a behaviour no test pins down.

    python -m tests.mutation.tree_evaluation_mutants

Not a pytest file (no test_ prefix): it reruns the test file once per
mutant (~5 s each). The source is restored after every mutant, also on an
interrupt, and checked to be byte-identical at the end.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "ML" / "surrogate" / "tree_evaluation.py"
TESTS = "tests/unit/test_tree_evaluation.py"

# (what the mutant breaks, original text, mutated text); each original must occur exactly once
MUTANTS = (
    ("violation at the tolerance counts", "over = violations[mask] > tol", "over = violations[mask] >= tol"),
    ("relative error threshold dropped", "rel_ok = ref_price >= MIN_RELATIVE_PRICE", "rel_ok = ref_price > 0"),
    ("exercised rows get an implied vol", "(vega >= MIN_IV_VEGA) & ~exercised", "(vega >= MIN_IV_VEGA)"),
    ("theta sign flipped", 'out["theta"].append(-d.dT', 'out["theta"].append(d.dT'),
    ("PDE's 1/2 dropped", "- 0.5 * sc ** 2", "- sc ** 2"),
    ("theta gap regions swapped", "np.where(exercised, np.abs(diag.theta)", "np.where(~exercised, np.abs(diag.theta)"),
    ("bucket edges inclusive on the wrong side", "(values >= low) & (values < high)", "(values > low) & (values <= high)"),
    ("vega difference not halved", "return (up - down) / (2 * h)", "return (up - down) / h"),
    ("test-split labels offset", "price_err=res.price - ref_price,", "price_err=res.price - ref_price * 1.001,"),
    ("z not inset from its ends", ".clip(z_low + Z_INSET, z_high - Z_INSET)", ""),
    ("moneyness cap not applied", "keep = (m >= m_low) & (m <= m_high)", "keep = m > 0"),
    ("vega accepted without exercised flags", "if vega is not None and exercised is None:", "if False:"),
    ("floor rate counts the wrong route", 'floored=res.route == "payoff_floor", gamma_err', 'floored=res.route == "net", gamma_err'),
    ("r groups include exercised rows", '(f"r = {rate:g}", ~exercised & (grid.r == rate))', '(f"r = {rate:g}", grid.r == rate)'),
)

def main() -> int:
    original = TARGET.read_bytes()
    text = original.decode("utf-8")
    survivors = []
    try:
        for name, old, new in MUTANTS:
            if text.count(old) != 1:
                raise SystemExit(f"mutant '{name}': its original text occurs {text.count(old)} times, expected once")
            TARGET.write_bytes(text.replace(old, new).encode("utf-8"))
            result = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-x", "-p", "no:cacheprovider"],
                                    cwd=ROOT, capture_output=True, text=True)
            killed = result.returncode != 0
            print(f"{'killed  ' if killed else 'SURVIVED'}  {name}", flush=True)
            if not killed:
                survivors.append(name)
    finally:
        TARGET.write_bytes(original)
    assert TARGET.read_bytes() == original, f"{TARGET} was not restored"
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0

if __name__ == "__main__":
    sys.exit(main())
