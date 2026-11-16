"""Tests for the write-docs skill's doc checker (`.claude/skills/write-docs/scripts/check_doc_claims.py`).

The script lives outside the package, so it is loaded from its path, as `test_hooks.py` loads the hooks.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".claude" / "skills" / "write-docs" / "scripts" / "check_doc_claims.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_doc_claims", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    """A doc in `notes/` and a 5-line source file in `pkg/`, so links resolve from the doc's folder."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    return tmp_path


def broken(checker, tree, text):
    return checker.check_links(tree / "notes" / "doc.md", text)


def test_a_link_to_an_existing_file_relative_to_the_doc_passes(checker, tree):
    assert broken(checker, tree, "See [mod.py](../pkg/mod.py).") == []


def test_a_link_to_a_missing_file_is_reported(checker, tree):
    assert broken(checker, tree, "See [gone.py](../pkg/gone.py).") == ["../pkg/gone.py"]


def test_a_link_resolves_from_the_docs_folder_not_the_root(checker, tree):
    # From notes/, "pkg/mod.py" means notes/pkg/mod.py, which does not exist
    assert broken(checker, tree, "[mod.py](pkg/mod.py)") == ["pkg/mod.py"]


@pytest.mark.parametrize("anchor", ["#L1", "#L5", "#L1-L5", "#L2-L4", "#L3-L3"])
def test_a_line_range_inside_the_file_passes(checker, tree, anchor):
    assert broken(checker, tree, f"[mod.py](../pkg/mod.py{anchor})") == []


@pytest.mark.parametrize("anchor", ["#L0", "#L6", "#L4-L6", "#L4-L2"])
def test_a_line_range_outside_the_file_or_reversed_is_reported(checker, tree, anchor):
    # The file has 5 lines: line 0, line 6 and a range ending at 6 don't exist, and L4-L2 is reversed
    target = f"../pkg/mod.py{anchor}"
    assert broken(checker, tree, f"[mod.py]({target})") == [target]


def test_a_heading_anchor_on_an_existing_file_is_not_checked_as_lines(checker, tree):
    assert broken(checker, tree, "[mod.py](../pkg/mod.py#some-heading)") == []


@pytest.mark.parametrize(
    "text",
    ["[site](https://example.com/x.py)", "[mail](mailto:a@b.c)", "[above](#a-section)", "[it](../pkg/mod.py)"],
)
def test_urls_and_in_page_anchors_are_skipped(checker, tree, text):
    assert broken(checker, tree, text) == []


def test_a_link_inside_inline_code_is_literal_text(checker, tree):
    assert broken(checker, tree, "Write links as `[file.py](../path#Lx-Ly)`.") == []


def test_a_percent_encoded_path_is_decoded(checker, tree):
    (tree / "pkg" / "my file.md").write_text("x\n", encoding="utf-8")
    assert broken(checker, tree, "[f](../pkg/my%20file.md)") == []


def test_every_broken_link_in_a_text_is_reported(checker, tree):
    text = "[a](../pkg/a.py) then [ok](../pkg/mod.py) then [b](../pkg/mod.py#L9)"
    assert broken(checker, tree, text) == ["../pkg/a.py", "../pkg/mod.py#L9"]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_main_skips_links_in_fenced_blocks_and_fails_on_a_broken_one(checker, tree, monkeypatch, capsys, fence):
    monkeypatch.chdir(tree)
    monkeypatch.setattr(checker, "ROOT", tree)
    doc = tree / "notes" / "doc.md"
    doc.write_text(f"{fence}markdown\n[x](../pkg/gone.py)\n{fence}\n\nSee [mod.py](../pkg/mod.py#L2).\n", "utf-8")
    assert checker.main([str(doc)]) == 0
    doc.write_text("See [mod.py](../pkg/mod.py#L7).\n", encoding="utf-8")
    assert checker.main([str(doc)]) == 1
    assert "broken link: ../pkg/mod.py#L7" in capsys.readouterr().out
