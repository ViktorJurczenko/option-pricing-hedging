"""
Check the backticked paths and code names in a Markdown doc against the repo.

Usage, from the project root:
    python .claude/skills/write-docs/scripts/check_doc_claims.py [--strict] <doc.md> [...]

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
- Template placeholders such as docs/results/<topic>.md are skipped:
  they name a pattern, not a file.
- Relative Markdown links, such as [gbm.py](../optionpricing/core/gbm.py#L10-L20),
  must point to an existing file, resolved from the doc's own folder. A
  #Lx or #Lx-Ly anchor must be a line range inside that file. URLs and
  in-page anchors (#section) are not checked.
- Fenced code blocks (``` or ~~~) are skipped: examples are checked by
  running them. So are ```math display blocks.
- Inline math in GitHub's code-span form, $`...`$, is skipped: its
  backticks hold LaTeX, not code names.

Exits with 1 if any path, name or link is missing.
"""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path.cwd()
SKILL_DIR = Path(__file__).resolve().parent.parent
FILE_EXTS = (".py", ".md", ".npz", ".pth", ".ini", ".txt", ".json", ".toml")
EXCLUDED_DIRS = {".venv", ".claude", ".git", "__pycache__"}
FORMULA_CHARS = set(" =^·−√σΔΓλ")
# A subscripted math symbol such as S_t, gamma_t, V_0, t_ex or S_{i,j}
SUBSCRIPTED_SYMBOL = re.compile(r"[A-Za-z]{1,6}_(t|t\+1|ex|\d+|[ij]|\{[^}]*\})")
# A Markdown link or image, [label](target): the target is group 1
LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)\)")
FENCED_BLOCK = re.compile(r"^(```|~~~).*?^\1", flags=re.S | re.M)
# GitHub's inline math with code-span delimiters, $`\sigma`$
INLINE_MATH = re.compile(r"\$`[^`\n]*`\$")


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
    return [
        p.relative_to(ROOT)
        for p in ROOT.rglob("*")
        if p.is_file() and not EXCLUDED_DIRS & set(p.relative_to(ROOT).parts)
    ]


def is_ignored(path: str) -> bool:
    result = git("check-ignore", "-q", path)
    return result is not None and result.returncode == 0


def check_links(doc: Path, text: str) -> list[str]:
    """Relative link targets in `text` that are broken.

    Args:
        doc (Path): the doc the text comes from. Targets resolve from its folder.
        text (str): the doc's text, with code blocks already removed.

    Returns:
        list[str]: each target whose file does not exist, or whose #Lx / #Lx-Ly anchor is not a line range
        inside the file (first line >= 1, first <= last, last <= the file's line count).
    """
    broken = []
    # A link inside inline code is literal text, quoted as an example
    for target in LINK.findall(re.sub(r"`[^`\n]*`", "", text)):
        # A URL (https:, mailto:) or an anchor in the same doc
        if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        path_part, _, anchor = target.partition("#")
        file = doc.parent / unquote(path_part)
        if not file.exists():
            broken.append(target)
            continue
        lines = re.fullmatch(r"L(\d+)(?:-L(\d+))?", anchor)
        if lines and file.is_file():
            first, last = int(lines[1]), int(lines[2] or lines[1])
            n_lines = len(file.read_text(encoding="utf-8", errors="ignore").splitlines())
            if not 1 <= first <= last <= n_lines:
                broken.append(target)
    return broken


def check(doc: Path, symbols: set[str], files: list[Path], source: str, strict: bool) -> dict[str, list[str]]:
    names = {p.name for p in files} | {part for p in files for part in p.parts[:-1]}
    # Top-level folders holding tracked files; any other folder (artifacts/,
    # notes/) only holds generated or untracked files
    tracked_tops = {p.parts[0] for p in files if len(p.parts) > 1}

    def generated(path: str) -> bool:
        return Path(path).parts[0] not in tracked_tops or is_ignored(path)

    text = INLINE_MATH.sub("", FENCED_BLOCK.sub("", doc.read_text(encoding="utf-8")))
    found = {
        "missing path": [],
        "not in the source": [],
        "broken link": check_links(doc, text),
        "generated, not checked": [],
        "short name, give the full path": [],
    }
    for token in sorted({t.strip() for t in re.findall(r"`([^`\n]+)`", text)}):
        if token in symbols or FORMULA_CHARS & set(token) or ".." in token or SUBSCRIPTED_SYMBOL.fullmatch(token):
            continue
        if re.fullmatch(r"[0-9a-f]{7,40}", token):  # a commit hash
            continue
        base = token.split("::")[0].split("#")[0]
        if base.startswith((".", "[")):  # an extension (".npz"), a dotfile, quoted link syntax
            continue
        if re.search(r"<[^<>]+>", base):  # a template placeholder, docs/results/<topic>.md
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
    source = "\n".join((ROOT / p).read_text(encoding="utf-8", errors="ignore") for p in files if p.suffix == ".py")
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
        failed |= bool(found["missing path"] or found["not in the source"] or found["broken link"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
