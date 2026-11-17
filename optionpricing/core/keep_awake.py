"""Keep Windows from suspending a long run (training, dataset generation, an overnight batch of runs).

This PC's only standby state is Modern Standby, where the screen turning off is standby and desktop programs are
suspended, whatever the sleep timeout. A first overnight run got ~30 min of compute in 9 h (.claude/BUGS.md,
B-032). So `keep_awake` holds both the system and the display on, and the screen is dimmed rather than
turned off. It does nothing on other systems.
"""

import contextlib
import sys

# SetThreadExecutionState flags (winbase.h): keep the setting until changed,
# keep the system running, keep the display on
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def _set_execution_state(flags: int) -> None:
    """The Win32 call, apart so tests can replace it."""
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(flags)


@contextlib.contextmanager
def keep_awake():
    """Hold the system and the display on while inside, and release them after, also on an error or Ctrl-C.

    On Windows it sets `ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED` on entry and `ES_CONTINUOUS`
    alone on exit, which returns to the normal power settings. On other systems it does nothing.
    """
    if sys.platform != "win32":
        yield
        return
    _set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        yield
    finally:
        _set_execution_state(ES_CONTINUOUS)  # back to the normal power settings
