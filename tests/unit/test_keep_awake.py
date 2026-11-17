"""Tests for optionpricing.core.keep_awake: holding Windows awake during a long run, and releasing it after."""

import pytest

from optionpricing.core import keep_awake as ka


def _recorded_states(monkeypatch, platform):
    calls = []
    monkeypatch.setattr(ka.sys, "platform", platform)
    monkeypatch.setattr(ka, "_set_execution_state", calls.append)
    return calls


def test_keep_awake_holds_the_system_and_display_on_windows_then_releases(monkeypatch):
    # Regression: on a Modern Standby PC the screen turning off suspended an overnight run (.claude/BUGS.md, B-032)
    calls = _recorded_states(monkeypatch, "win32")
    with ka.keep_awake():
        assert calls == [ka.ES_CONTINUOUS | ka.ES_SYSTEM_REQUIRED | ka.ES_DISPLAY_REQUIRED]
    assert calls[-1] == ka.ES_CONTINUOUS and len(calls) == 2


def test_keep_awake_releases_even_when_the_run_fails(monkeypatch):
    calls = _recorded_states(monkeypatch, "win32")
    with pytest.raises(KeyboardInterrupt):
        with ka.keep_awake():
            raise KeyboardInterrupt  # Ctrl-C mid-run
    assert calls[-1] == ka.ES_CONTINUOUS


def test_keep_awake_does_nothing_elsewhere(monkeypatch):
    calls = _recorded_states(monkeypatch, "linux")
    with ka.keep_awake():
        pass
    assert calls == []
