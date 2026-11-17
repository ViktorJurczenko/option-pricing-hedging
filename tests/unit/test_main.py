"""Tests for main.py: the command line dispatches to the chosen command, with the PC held awake while it runs."""

import contextlib

import pytest

import main


@contextlib.contextmanager
def _recording_keep_awake(events):
    events.append("hold")
    try:
        yield
    finally:
        events.append("release")


class _FakeCommand:
    """A command module that records the order of events and the flags it was run with."""

    def __init__(self, events):
        self.events = events

    def add_arguments(self, parser):
        parser.add_argument("--flag", default="default")

    def run(self, args):
        self.events.append(f"run {args.flag}")


def _patch(monkeypatch, events, command=None):
    monkeypatch.setattr(main, "keep_awake", lambda: _recording_keep_awake(events))
    if command is not None:
        monkeypatch.setitem(main.COMMANDS, "train-hedge", command)


def test_main_runs_the_command_inside_keep_awake(monkeypatch):
    # an overnight train-hedge must not be suspended when the screen turns off (.claude/BUGS.md, B-032)
    events = []
    _patch(monkeypatch, events, _FakeCommand(events))
    main.main(["train-hedge", "--flag", "x"])
    assert events == ["hold", "run x", "release"]


def test_main_releases_keep_awake_when_the_command_fails(monkeypatch):
    events = []

    class Failing(_FakeCommand):
        def run(self, args):
            raise ValueError("bad flag value")

    _patch(monkeypatch, events, Failing(events))
    with pytest.raises(ValueError, match="bad flag value"):
        main.main(["train-hedge"])
    assert events == ["hold", "release"]


def test_main_does_not_hold_the_pc_for_an_invalid_command_line(monkeypatch):
    # argparse exits before any command runs, so nothing is held
    events = []
    _patch(monkeypatch, events)
    with pytest.raises(SystemExit) as exit_:
        main.main(["no-such-command"])
    assert exit_.value.code == 2
    assert events == []
