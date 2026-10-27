"""
Check the backticked paths and code names in a Markdown doc against the repo.

Usage, from the project root:
    python .claude/skills/optionpricing-docs/scripts/check_doc_claims.py [--strict] <doc.md> [...]

- Paths with a "/" must exist relative to the root. Gitignored ones
  (generated data, checkpoints) are listed as "generated, not checked".
- Short names ("env.py", "rl/") must exist somewhere in the repo; README
  architecture lists use them under a parent bullet. --strict also lists
  them, for docs where every path should be written in full.
- Code names are searched in git-tracked .py files only, since untracked
  folders may still use old names.
- Math is skipped: the Symbol column of references/notation.md,
  subscripted symbols (S_t, V_0, t_ex), and formula-like tokens (spaces,
  "=", "^", "·", "−", "..", or a "/" without a file extension).
- Fenced code blocks are skipped: examples are checked by running them.

Exits with 1 if any path or name is missing.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
SKILL_DIR = Path(__file__).resolve().parent.parent
FILE_EXTS = (".py", ".md", ".npz", ".pth", ".ini", ".txt", ".json")
EXCLUDED_DIRS = {".venv", ".claude", ".git", "__pycache__"}
FORMULA_CHARS = set(" =^·−√σΔΓλ")
# A subscripted math symbol such as S_t, gamma_t, V_0, t_ex or S_{i,j}
SUBSCRIPTED_SYMBOL = re.compile(r"[A-Za-z]{1,6}_(t|t\+1|ex|\d+|[ij]|\{[^}]*\})")


def notation_symbols() -> set[str]:
    """First-column backticked tokens of every table in references/notation.md."""
    symbols = set()
    for line in (SKILL_DIR / "references" / "notation.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            first_cell = line.split("|")[1]
            symbols.update(re.findall(r"`([^`]+)`", first_cell))
    return symbols


def git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:  # git not installed
        return None


def repo_files() -> list[Path]:
    """Tracked files, or every file outside EXCLUDED_DIRS without git."""
    listed = git("ls-files")
    if listed is not None and listed.returncode == 0:
        return [Path(p) for p in listed.stdout.split()]
    return [p.relative_to(ROOT) for p in ROOT.rglob("*")
            if p.is_file() and not EXCLUDED_DIRS & set(p.relative_to(ROOT).parts)]


def is_ignored(path: str) -> bool:
    result = git("check-ignore", "-q", path)
    return result is not None and result.returncode == 0


def check(doc: Path, symbols: set[str], files: list[Path], source: str, strict: bool) -> dict[str, list[str]]:
    names = {p.name for p in files} | {part for p in files for part in p.parts[:-1]}
    # Top-level folders holding tracked files; any other folder (data/,
    # checkpoints/, runs/) only holds generated output
    tracked_tops = {p.parts[0] for p in files if len(p.parts) > 1}

    def generated(path: str) -> bool:
        return Path(path).parts[0] not in tracked_tops or is_ignored(path)

    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
    found = {"missing path": [], "not in the source": [], "generated, not checked": [],
             "short name, give the full path": []}
    for token in sorted({t.strip() for t in re.findall(r"`([^`\n]+)`", text)}):
        if token in symbols or FORMULA_CHARS & set(token) or ".." in token or SUBSCRIPTED_SYMBOL.fullmatch(token):
            continue
        if re.fullmatch(r"[0-9a-f]{7,40}", token):  # a commit hash
            continue
        base = token.split("::")[0].split("#")[0]
        if base.startswith((".", "[")):  # an extension (".npz"), a dotfile, quoted link syntax
            continue
        if base.endswith(FILE_EXTS) or base.endswith("/"):
            stripped = base.rstrip("/")
            if "/" in stripped:  # a full path
                if not (ROOT / stripped).exists():
                    found["generated, not checked" if generated(stripped) else "missing path"].append(base)
            elif stripped not in names and not (ROOT / stripped).exists():
                found["generated, not checked" if is_ignored(stripped) else "missing path"].append(base)
            elif strict:
                found["short name, give the full path"].append(base)
            continue
        if "/" in base:  # a formula like V_0/K
            continue
        # A code name: the last dotted part, minus call parentheses, must be in the source
        name = re.sub(r"\(.*", "", base).split(".")[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and not re.search(rf"\b{re.escape(name)}\b", source):
            found["not in the source"].append(token)
    return found


def main(argv: list[str]) -> int:
    # The Windows console defaults to cp1252, which can't print "−" or "σ"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    strict = "--strict" in argv
    docs = [a for a in argv if a != "--strict"]
    if not docs:
        print(__doc__)
        return 2
    files = repo_files()
    source = "\n".join((ROOT / p).read_text(encoding="utf-8", errors="ignore")
                       for p in files if p.suffix == ".py")
    symbols = notation_symbols()
    failed = False
    for doc in docs:
        found = check(Path(doc), symbols, files, source, strict)
        print(f"== {doc}")
        for label, items in found.items():
            for item in items:
                print(f"  {label}: {item}")
        if not any(found.values()):
            print("  OK")
        failed |= bool(found["missing path"] or found["not in the source"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
