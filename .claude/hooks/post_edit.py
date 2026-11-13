"""
PostToolUse hook (Edit/Write/MultiEdit): format and lint the edited Python
file with ruff, then run its matching unit test files, if any.

A ruff finding it cannot fix, or a failing test, is reported back to the
agent (exit code 2, details on stderr) so the break is seen at the edit, not
at commit time. Anything that is not a .py file inside the project is left
alone (exit 0).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])


def project_python_file(file_path: str | None, root: Path = ROOT) -> Path | None:
    """The edited file relative to the project root, if it is a .py file inside it; else None."""
    if not file_path or not file_path.endswith(".py"):
        return None
    try:
        return Path(file_path).resolve().relative_to(root.resolve())
    except ValueError:  # outside the project (e.g. a scratch script)
        return None


def matching_tests(rel_path: Path, root: Path = ROOT) -> list[Path]:
    """
    The unit test files for an edited file: the file itself when it is a test
    under tests/unit/, else every test_<module>.py under tests/unit/ and its
    subfolders (the legacy stacks keep their tests in tests/unit/*_legacy/).
    A legacy module that shares its name with a current one gets both test
    files, which costs a few seconds but never skips the right one.

    Args:
        rel_path: the edited file, relative to the project root.
        root: the project root.

    Returns:
        The test files relative to the root, sorted, empty when there is none.
    """
    if rel_path.parts[:2] == ("tests", "unit") and rel_path.name.startswith("test_"):
        return [rel_path]
    unit = root / "tests" / "unit"
    return sorted(p.relative_to(root) for p in unit.rglob(f"test_{rel_path.stem}.py"))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    payload = json.load(sys.stdin)
    rel = project_python_file(payload.get("tool_input", {}).get("file_path"))
    if rel is None:
        return 0
    problems = []
    run([sys.executable, "-m", "ruff", "format", str(rel)])
    lint = run([sys.executable, "-m", "ruff", "check", "--fix", str(rel)])
    if lint.returncode != 0:
        problems.append(f"ruff check found issues it could not fix in {rel}:\n{lint.stdout.strip()}")
    tests = matching_tests(rel)
    if tests:
        result = run([sys.executable, "-m", "pytest", *map(str, tests), "-x", "-q", "-p", "no:cacheprovider"])
        if result.returncode != 0:
            tail = "\n".join(result.stdout.strip().splitlines()[-30:])
            names = ", ".join(map(str, tests))
            problems.append(f"{names} fails after editing {rel}:\n{tail}")
    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
