"""
PreToolUse hook (Bash): the commit gate. For a `git commit` command it checks,
in order, and blocks the commit (exit code 2, reasons on stderr) on failure:

1. the message: Conventional Commits subject with an allowed type and scope,
   a lowercase description, <= 50 characters, and a Co-Authored-By trailer;
2. style: `ruff format --check .` and `ruff check .`;
3. `pytest tests/unit`;
4. the mutation check, when a file it depends on changed (it finds mutants
   by exact source text, so a reformat can break it: .claude/BUGS.md B-034).

Any other Bash command passes through untouched (exit 0).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])

TYPES = ("feat", "fix", "refactor", "perf", "test", "docs", "style", "chore")
SCOPES = ("pricing", "surrogate", "rl", "hedging", "cli", "tests", "docs", "tooling")
SUBJECT_MAX = 50
SUBJECT_RE = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\((?:{'|'.join(SCOPES)})\))?: [a-z0-9`]")
# Files the mutation check reads: the mutated module, its tests, the runner.
MUTATION_INPUTS = (
    "optionpricing/ml/surrogate/tree/tree_evaluation.py",
    "tests/unit/test_tree_evaluation.py",
    "tests/mutation/tree_evaluation_mutants.py",
)


# `git`, optionally with `-C <dir>`/`-c <key=value>` before the subcommand, then `commit`
# (not `commit-tree`); preceded by the start, whitespace, a separator or `(`.
COMMIT_RE = re.compile(r"(?:^|[\s;&|(])git(?:\s+-[Cc]\s+\S+)*\s+commit(?![\w-])")
# A heredoc body: `<<EOF`, `<<'EOF'` or `<<"EOF"` (optionally `<<-`), the body, the delimiter alone on its line.
HEREDOC = r"<<-?\s*([\"']?)(\w+)\1[^\n]*\n(.*?)\n\s*\2[ \t]*(?=\n|$|\))"
FILE_HEREDOC_RE = re.compile(r"-F\s+-\s*" + HEREDOC, re.S)
SUBSHELL_HEREDOC_RE = re.compile(r"-[a-zA-Z]*m\s+\"\$\(cat\s+" + HEREDOC, re.S)
# Each -m (or a flag group ending in m, like -am) with a quoted argument.
DASH_M_RE = re.compile(r"""(?:^|\s)-[a-zA-Z]*m\s+(["'])(.*?)(?<!\\)\1""", re.S)
# Per-check limits, seconds: together below the hook's 600 s timeout in .claude/settings.json,
# so a hung check blocks the commit instead of the harness timing the hook out (which lets it through).
TIMEOUTS = {"ruff format --check": 60, "ruff check": 60, "unit tests": 240, "mutation check": 200}


def is_commit(command: str) -> bool:
    """True when the shell command runs `git commit` (also chained, e.g. `git add x && git commit`, or `git -C dir commit`)."""
    return COMMIT_RE.search(command) is not None


def commit_message(command: str) -> str | None:
    """
    The commit message a `git commit` command passes, in the forms agents use:

    - `-F - <<'EOF'` ... `EOF` (any heredoc quoting);
    - `-m "$(cat <<'EOF'` ... `EOF` `)"`;
    - one or more `-m "..."`/`-m '...'` (or `-am`), joined by blank lines as git does.

    Returns None when it gives none of these (an editor, `-F <file>`, `--no-edit`).
    """
    for pattern in (FILE_HEREDOC_RE, SUBSHELL_HEREDOC_RE):
        match = pattern.search(command)
        if match:
            return match.group(3)
    parts = [m.group(2) for m in DASH_M_RE.finditer(command)]
    return "\n\n".join(parts) if parts else None


def message_problems(message: str) -> list[str]:
    """Every way `message` breaks the commit convention in .claude/CLAUDE.md; empty when it is fine."""
    lines = message.strip("\n").splitlines()
    subject = lines[0] if lines else ""
    problems = []
    if not SUBJECT_RE.match(subject):
        problems.append(
            f"subject {subject!r} is not `type(scope): description` with a lowercase description; "
            f"types {', '.join(TYPES)}; scopes (optional) {', '.join(SCOPES)}"
        )
    if len(subject) > SUBJECT_MAX:
        problems.append(f"subject is {len(subject)} characters, the limit is {SUBJECT_MAX}")
    if len(lines) > 1 and lines[1].strip():
        problems.append("the line after the subject must be blank")
    if not any(line.startswith("Co-Authored-By:") for line in lines):
        problems.append("missing the Co-Authored-By trailer")
    return problems


def needs_mutation_check(changed: list[str]) -> bool:
    """True when any changed path (staged or not) is one of the mutation check's inputs."""
    return any(path.replace("\\", "/") in MUTATION_INPUTS for path in changed)


def changed_files() -> list[str]:
    """Tracked files with staged or unstaged changes: a chained `git add && git commit` stages after this hook."""
    out = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run `cmd` from the project root; a timeout comes back as a failed run, so it blocks the commit."""
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=f"timed out after {timeout:.0f} s")


def tail(result: subprocess.CompletedProcess, n: int = 25) -> str:
    return "\n".join((result.stdout + result.stderr).strip().splitlines()[-n:])


def main() -> int:
    payload = json.load(sys.stdin)
    command = payload.get("tool_input", {}).get("command", "")
    if not is_commit(command):
        return 0
    problems = []
    message = commit_message(command)
    if message is None:
        if "--no-edit" not in command:
            problems.append(
                'commit message not found: pass it with -m "...", -m "$(cat <<\'EOF\' ... EOF)" '
                "or a `-F - <<'EOF'` heredoc"
            )
    else:
        problems += [f"commit message: {p}" for p in message_problems(message)]
    py = sys.executable
    checks = [
        ("ruff format --check", [py, "-m", "ruff", "format", "--check", "."]),
        ("ruff check", [py, "-m", "ruff", "check", "."]),
        ("unit tests", [py, "-m", "pytest", "tests/unit", "-x", "-q", "-p", "no:cacheprovider"]),
    ]
    if needs_mutation_check(changed_files()):
        checks.append(("mutation check", [py, "-m", "tests.mutation.tree_evaluation_mutants"]))
    for name, cmd in checks:
        result = run(cmd, TIMEOUTS[name])
        if result.returncode != 0:
            problems.append(f"{name} failed:\n{tail(result)}")
    if problems:
        print("Commit blocked by .claude/hooks/pre_commit.py:\n\n" + "\n\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
