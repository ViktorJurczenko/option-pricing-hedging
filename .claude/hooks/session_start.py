"""
SessionStart hook: print the roadmap table and the open steps of
.claude/PLAN.md, then `git status -s`, so a session starts with the context
of what is in progress. Its stdout is added to the agent's context.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])


def roadmap_table(plan: str) -> list[str]:
    """The lines of PLAN.md's first markdown table (header, separator and rows)."""
    lines = plan.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    if start is None:
        return []
    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table.append(line)
    return table


def open_steps(plan: str) -> list[str]:
    """`<workstream>: <step>` for every unticked `- [ ]` step, under its `###` workstream heading."""
    steps, workstream = [], None
    for line in plan.splitlines():
        heading = re.match(r"^###\s+(.*)", line)
        if heading:
            workstream = heading.group(1).strip()
        step = re.match(r"^- \[ \]\s+(.*)", line)
        if step:
            steps.append(f"{workstream}: {step.group(1).strip()}" if workstream else step.group(1).strip())
    return steps


def main() -> int:
    plan_path = ROOT / ".claude" / "PLAN.md"
    plan = plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""
    out = ["Roadmap (.claude/PLAN.md):", *roadmap_table(plan)]
    steps = open_steps(plan)
    out += ["", "Open steps:", *(f"- {s}" for s in steps)] if steps else ["", "No open steps."]
    status = subprocess.run(["git", "status", "-s"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out += ["", "git status -s:", status or "(clean)"]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
