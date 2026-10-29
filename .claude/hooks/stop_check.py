"""
Stop hook: when library code under optionpricing/ has uncommitted changes but
none of the living docs (.claude/PLAN.md, .claude/BUGS.md, README.md) do,
ask the agent once to update them or to say why they need no update.

Blocks the stop with exit code 2 and the reminder on stderr. It never blocks
twice in a row: when the agent is already continuing because of a stop hook
(`stop_hook_active`), it lets the stop through.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])

LIVING_DOCS = (".claude/PLAN.md", ".claude/BUGS.md", "README.md")


def needs_doc_reminder(changed: list[str]) -> bool:
    """
    True when library code under optionpricing/ changed and no living doc did.
    Code is a .py file, or a new untracked directory (porcelain lists it as
    `optionpricing/newpkg/`, not its files).
    """
    paths = [p.replace("\\", "/") for p in changed]
    code = any(p.startswith("optionpricing/") and p.endswith((".py", "/")) for p in paths)
    docs = any(p in LIVING_DOCS for p in paths)
    return code and not docs


def parse_porcelain(out: str) -> list[str]:
    """
    The paths in `git status --porcelain` (v1) output: two status characters,
    a space, then the path; a rename's new path; quotes git adds around paths
    with special characters removed.
    """
    return [line[3:].split(" -> ")[-1].strip('"') for line in out.splitlines() if len(line) > 3]


def changed_files() -> list[str]:
    """Paths with uncommitted changes, staged or not, untracked included."""
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return parse_porcelain(out)


def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("stop_hook_active"):
        return 0
    if needs_doc_reminder(changed_files()):
        print(
            "Code under optionpricing/ changed but .claude/PLAN.md, .claude/BUGS.md and README.md did not. "
            "Update the ones this change affects (and notes/REMINDER.md for a design change), "
            "or say in your reply why none needs it.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
