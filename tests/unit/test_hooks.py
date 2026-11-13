"""
The Claude Code hooks in .claude/hooks/: their decision functions, and the
fast paths where a hook must do nothing. The heavy actions (ruff, pytest,
the mutation check) are never run here: each fast-path test replaces the
subprocess runner with one that fails the test if it is called.
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"


def load(name):
    # .claude/hooks is not a package (dot directory), so load each script by path
    spec = importlib.util.spec_from_file_location(f"hook_{name}", HOOKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post_edit = load("post_edit")
pre_commit = load("pre_commit")
session_start = load("session_start")
stop_check = load("stop_check")


def forbid(*_, **__):
    raise AssertionError("the hook ran a subprocess on a path where it must do nothing")


def feed(monkeypatch, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


GOOD_MESSAGE = "fix(pricing): reject NaN inputs\n\nWhy it matters.\n\nCo-Authored-By: Someone <x@y.z>\n"


# ---------------------------------------------------------------- post_edit


def test_post_edit_only_takes_python_files_inside_the_project(tmp_path):
    inside = ROOT / "optionpricing" / "core" / "gbm.py"
    assert post_edit.project_python_file(str(inside)) == Path("optionpricing", "core", "gbm.py")
    assert post_edit.project_python_file(str(ROOT / "README.md")) is None
    assert post_edit.project_python_file(str(tmp_path / "scratch.py")) is None
    assert post_edit.project_python_file(None) is None


def test_post_edit_finds_the_unit_test_named_after_the_module():
    assert post_edit.matching_tests(Path("optionpricing", "core", "gbm.py")) == [Path("tests", "unit", "test_gbm.py")]


def test_post_edit_finds_a_legacy_modules_test_in_a_unit_subfolder():
    # the legacy hedging tests moved to tests/unit/hedging_legacy/; a lookup
    # in tests/unit/ alone stopped running them after an edit
    edited = Path("optionpricing", "ml", "hedging", "legacy", "critic.py")
    assert post_edit.matching_tests(edited) == [Path("tests", "unit", "hedging_legacy", "test_critic.py")]


def test_post_edit_runs_every_test_file_sharing_the_modules_name(tmp_path):
    # a legacy module and a current one with the same name: both files run, sorted
    for folder in ("tests/unit", "tests/unit/hedging_legacy"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / "test_baselines.py").write_text("")
    (tmp_path / "tests" / "unit" / "test_other.py").write_text("")
    found = post_edit.matching_tests(Path("optionpricing", "hedging", "baselines.py"), root=tmp_path)
    assert found == [
        Path("tests", "unit", "hedging_legacy", "test_baselines.py"),
        Path("tests", "unit", "test_baselines.py"),
    ]


def test_post_edit_runs_an_edited_unit_test_file_itself():
    edited = Path("tests", "unit", "hedging_legacy", "test_critic.py")
    assert post_edit.matching_tests(edited) == [edited]


def test_post_edit_finds_no_test_for_a_module_without_one():
    # main.py has no tests/unit/test_main.py; an integration test is not run either
    assert post_edit.matching_tests(Path("main.py")) == []
    assert post_edit.matching_tests(Path("tests", "integration", "test_cli_smoke.py")) == []


def test_post_edit_does_nothing_for_a_non_python_file(monkeypatch):
    monkeypatch.setattr(post_edit, "run", forbid)
    feed(monkeypatch, {"tool_input": {"file_path": str(ROOT / "README.md")}})
    assert post_edit.main() == 0


def test_post_edit_reports_a_failing_test_with_exit_code_2(monkeypatch, capsys):
    def fake_run(cmd):
        failed = "pytest" in cmd
        return type("R", (), {"returncode": int(failed), "stdout": "1 failed" if failed else "", "stderr": ""})()

    monkeypatch.setattr(post_edit, "run", fake_run)
    feed(monkeypatch, {"tool_input": {"file_path": str(ROOT / "optionpricing" / "core" / "gbm.py")}})
    assert post_edit.main() == 2
    err = capsys.readouterr().err
    assert "test_gbm.py fails after editing" in err and "1 failed" in err


# --------------------------------------------------------------- pre_commit


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "fix: x"',
        "git add a.py && git commit -q -F - <<'EOF'\nfix: x\nEOF",
        "cd repo; git commit --amend --no-edit",
    ],
)
def test_pre_commit_recognises_commit_commands(command):
    assert pre_commit.is_commit(command)


@pytest.mark.parametrize("command", ["git status", "git commit-tree abc", "git log --grep commit", "echo committed"])
def test_pre_commit_lets_other_commands_through(command):
    assert not pre_commit.is_commit(command)


def test_pre_commit_reads_the_message_from_a_heredoc():
    command = "git add x && git commit -q -F - <<'EOF'\n" + GOOD_MESSAGE + "EOF\ngit log --oneline -1"
    assert pre_commit.commit_message(command) == GOOD_MESSAGE.rstrip("\n")


@pytest.mark.parametrize("quote", ['"', "'"])
def test_pre_commit_reads_the_message_from_dash_m(quote):
    assert pre_commit.commit_message(f"git commit -m {quote}docs: fix typo{quote}") == "docs: fix typo"


def test_pre_commit_finds_no_message_without_m_or_heredoc():
    assert pre_commit.commit_message("git commit --amend --no-edit") is None


def test_pre_commit_accepts_a_conventional_message():
    assert pre_commit.message_problems(GOOD_MESSAGE) == []
    # the scope is optional
    assert pre_commit.message_problems("refactor: move code\n\nCo-Authored-By: A <a@b.c>") == []


@pytest.mark.parametrize(
    "subject",
    [
        "Fix the bug",  # no type
        "feature: add x",  # unknown type
        "fix(ml): add x",  # scope not in the list
        "fix: Add x",  # capitalised description
        "fix:add x",  # no space after the colon
    ],
)
def test_pre_commit_rejects_a_malformed_subject(subject):
    problems = pre_commit.message_problems(f"{subject}\n\nCo-Authored-By: A <a@b.c>")
    assert len(problems) == 1 and "type(scope): description" in problems[0]


def test_pre_commit_subject_limit_is_inclusive_at_50():
    at_limit = "docs: " + "a" * (pre_commit.SUBJECT_MAX - len("docs: "))
    assert len(at_limit) == 50
    assert pre_commit.message_problems(at_limit + "\n\nCo-Authored-By: A <a@b.c>") == []
    problems = pre_commit.message_problems(at_limit + "a\n\nCo-Authored-By: A <a@b.c>")
    assert problems == ["subject is 51 characters, the limit is 50"]


def test_pre_commit_requires_a_blank_line_and_the_trailer():
    problems = pre_commit.message_problems("fix: x\nbody right away")
    assert any("blank" in p for p in problems)
    assert any("Co-Authored-By" in p for p in problems)


def test_pre_commit_runs_the_mutation_check_only_when_its_inputs_change():
    assert pre_commit.needs_mutation_check(["optionpricing/ml/surrogate/tree/tree_evaluation.py"])
    # git on Windows may report backslashes
    assert pre_commit.needs_mutation_check(["tests\\mutation\\tree_evaluation_mutants.py"])
    assert not pre_commit.needs_mutation_check(["optionpricing/ml/surrogate/tree/tree_training.py", "README.md"])
    assert not pre_commit.needs_mutation_check([])


def test_pre_commit_does_nothing_for_a_non_commit_command(monkeypatch):
    monkeypatch.setattr(pre_commit, "run", forbid)
    monkeypatch.setattr(pre_commit, "changed_files", forbid)
    feed(monkeypatch, {"tool_input": {"command": "git status"}})
    assert pre_commit.main() == 0


def test_pre_commit_blocks_a_bad_message_even_when_every_check_passes(monkeypatch, capsys):
    passed = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(pre_commit, "run", lambda cmd, timeout: passed)
    monkeypatch.setattr(pre_commit, "changed_files", lambda: [])
    feed(monkeypatch, {"tool_input": {"command": 'git commit -m "Fixed stuff"'}})
    assert pre_commit.main() == 2
    assert "commit message" in capsys.readouterr().err


def test_pre_commit_blocks_on_a_failing_check_and_lets_a_clean_commit_through(monkeypatch, capsys):
    command = "git commit -F - <<'EOF'\n" + GOOD_MESSAGE + "EOF"
    feed(monkeypatch, {"tool_input": {"command": command}})
    monkeypatch.setattr(pre_commit, "changed_files", lambda: [])

    def fake_run(cmd, timeout):
        failed = "pytest" in cmd
        return type("R", (), {"returncode": int(failed), "stdout": "1 failed", "stderr": ""})()

    monkeypatch.setattr(pre_commit, "run", fake_run)
    assert pre_commit.main() == 2
    assert "unit tests failed" in capsys.readouterr().err

    feed(monkeypatch, {"tool_input": {"command": command}})
    monkeypatch.setattr(
        pre_commit, "run", lambda cmd, timeout: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    )
    assert pre_commit.main() == 0


def test_pre_commit_adds_the_mutation_check_when_tree_evaluation_changed(monkeypatch):
    ran = []
    monkeypatch.setattr(
        pre_commit,
        "run",
        lambda cmd, timeout: ran.append(cmd) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(pre_commit, "changed_files", lambda: ["optionpricing/ml/surrogate/tree/tree_evaluation.py"])
    feed(monkeypatch, {"tool_input": {"command": "git commit -F - <<'EOF'\n" + GOOD_MESSAGE + "EOF"}})
    assert pre_commit.main() == 0
    assert any("tests.mutation.tree_evaluation_mutants" in cmd for cmd in ran)


# ------------------------------------------------------------ session_start

PLAN = """# Roadmap

| Workstream | Status | Next step |
|---|---|---|
| [A](#a) | active | step two |

## Active

### A

- [x] Step one
- [ ] Step two
  Why it matters.

## Backlog

### B

- [ ] Explain the theory
"""


def test_session_start_extracts_the_roadmap_table():
    assert session_start.roadmap_table(PLAN) == [
        "| Workstream | Status | Next step |",
        "|---|---|---|",
        "| [A](#a) | active | step two |",
    ]
    assert session_start.roadmap_table("# No table here\n") == []


def test_session_start_lists_only_open_steps_under_their_workstream():
    assert session_start.open_steps(PLAN) == ["A: Step two", "B: Explain the theory"]


def test_session_start_reads_the_real_plan():
    plan = (ROOT / ".claude" / "PLAN.md").read_text(encoding="utf-8")
    assert session_start.roadmap_table(plan)[0].startswith("| Workstream")


# --------------------------------------------------------------- stop_check


def test_stop_reminds_when_only_code_changed():
    assert stop_check.needs_doc_reminder(["optionpricing/core/gbm.py"])
    assert stop_check.needs_doc_reminder(["optionpricing\\core\\gbm.py"])


@pytest.mark.parametrize(
    "changed",
    [
        ["optionpricing/core/gbm.py", ".claude/PLAN.md"],
        ["optionpricing/core/gbm.py", "README.md"],
        [".claude/BUGS.md"],
        ["tests/unit/test_gbm.py"],  # tests alone are not library code
        ["optionpricing/ml/surrogate/legacy/__init__.txt"],  # not a .py file
        [],
    ],
)
def test_stop_stays_quiet_otherwise(changed):
    assert not stop_check.needs_doc_reminder(changed)


def test_stop_parses_porcelain_status():
    out = ' M optionpricing/core/gbm.py\nR  old.py -> optionpricing/new.py\n?? "notes/a b.md"\n'
    assert stop_check.parse_porcelain(out) == ["optionpricing/core/gbm.py", "optionpricing/new.py", "notes/a b.md"]


def test_stop_never_blocks_twice_in_a_row(monkeypatch):
    monkeypatch.setattr(stop_check, "changed_files", forbid)
    feed(monkeypatch, {"stop_hook_active": True})
    assert stop_check.main() == 0


def test_stop_blocks_with_the_reminder_when_docs_lag(monkeypatch, capsys):
    monkeypatch.setattr(stop_check, "changed_files", lambda: ["optionpricing/core/gbm.py"])
    feed(monkeypatch, {"stop_hook_active": False})
    assert stop_check.main() == 2
    assert "PLAN.md" in capsys.readouterr().err


# ------------------------------------------- review follow-ups (Phase 7)

TRAILER = "Co-Authored-By: A <a@b.c>"


@pytest.mark.parametrize(
    "command",
    [
        # the subshell form agents often use
        "git commit -m \"$(cat <<'EOF'\nfix: x\n\n" + TRAILER + '\nEOF\n)"',
        # a double-quoted heredoc delimiter
        'git commit -F - <<"EOF"\nfix: x\n\n' + TRAILER + "\nEOF",
        # an unquoted delimiter, followed by more commands
        "git commit -F - <<EOF\nfix: x\n\n" + TRAILER + "\nEOF\ngit log -1",
    ],
)
def test_pre_commit_reads_heredoc_message_forms(command):
    assert pre_commit.commit_message(command) == "fix: x\n\n" + TRAILER


def test_pre_commit_joins_several_dash_m_like_git():
    # git puts each -m in its own paragraph, so the trailer in the last -m counts
    message = pre_commit.commit_message(f'git commit -am "fix: x" -m "Why." -m "{TRAILER}"')
    assert message == f"fix: x\n\nWhy.\n\n{TRAILER}"
    assert pre_commit.message_problems(message) == []


def test_pre_commit_does_not_take_amend_for_a_message_flag():
    assert pre_commit.commit_message("git commit --amend --no-edit") is None


@pytest.mark.parametrize("command", ["git -C repo commit -m x", "git -c user.name=a commit -m x", "(git commit -m x)"])
def test_pre_commit_recognises_commit_with_git_options(command):
    assert pre_commit.is_commit(command)


def test_pre_commit_lets_no_edit_through_without_a_message(monkeypatch):
    passed = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(pre_commit, "run", lambda cmd, timeout: passed)
    monkeypatch.setattr(pre_commit, "changed_files", lambda: [])
    feed(monkeypatch, {"tool_input": {"command": "git commit --amend --no-edit"}})
    assert pre_commit.main() == 0


def test_pre_commit_blocks_when_no_message_can_be_read(monkeypatch, capsys):
    passed = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(pre_commit, "run", lambda cmd, timeout: passed)
    monkeypatch.setattr(pre_commit, "changed_files", lambda: [])
    feed(monkeypatch, {"tool_input": {"command": "git commit -F msg.txt"}})
    assert pre_commit.main() == 2
    assert "commit message not found" in capsys.readouterr().err


def test_pre_commit_counts_a_timed_out_check_as_failed(monkeypatch):
    # a hung check must block the commit, not let the harness time the hook out (which lets it through)
    def hang(*args, **kwargs):
        raise pre_commit.subprocess.TimeoutExpired(cmd="pytest", timeout=kwargs["timeout"])

    monkeypatch.setattr(pre_commit.subprocess, "run", hang)
    result = pre_commit.run(["pytest"], timeout=5)
    assert result.returncode != 0 and "timed out after 5 s" in result.stderr


def test_pre_commit_check_timeouts_fit_in_the_hook_timeout():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    hook = settings["hooks"]["PreToolUse"][0]["hooks"][0]
    assert "pre_commit.py" in hook["command"]
    assert sum(pre_commit.TIMEOUTS.values()) < hook["timeout"]


def test_post_edit_reports_what_ruff_cannot_fix(monkeypatch, capsys):
    def fake_run(cmd):
        unfixable = "check" in cmd
        return type("R", (), {"returncode": int(unfixable), "stdout": "F821 undefined name", "stderr": ""})()

    monkeypatch.setattr(post_edit, "run", fake_run)
    feed(monkeypatch, {"tool_input": {"file_path": str(ROOT / "main.py")}})  # main.py has no unit test file
    assert post_edit.main() == 2
    err = capsys.readouterr().err
    assert "could not fix" in err and "F821" in err


def test_stop_counts_a_new_untracked_package_as_code():
    assert stop_check.needs_doc_reminder(["optionpricing/newpkg/"])
    assert not stop_check.needs_doc_reminder(["docs/newdir/"])
