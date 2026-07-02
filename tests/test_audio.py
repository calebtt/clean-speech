"""Regression: SIGKILL (or any hard kill) of the daemon process used to orphan
the parec child, since PulseMonitorReader.__exit__ never gets a chance to run.
The orphan then keeps capturing system audio indefinitely -- a real privacy
problem, confirmed once in this project's history. Fixed via PR_SET_PDEATHSIG
so the kernel kills the child the moment its parent dies, no cleanup required."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import unittest

_PARENT_SCRIPT = (
    "import subprocess\n"
    "import time\n"
    "from clean_speech_daemon.audio import _die_with_parent\n"
    "child = subprocess.Popen(['sleep', '30'], preexec_fn=_die_with_parent)\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(30)\n"
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PdeathsigTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "PR_SET_PDEATHSIG is Linux-only")
    def test_child_dies_when_parent_is_sigkilled(self) -> None:
        parent = subprocess.Popen(
            [sys.executable, "-c", _PARENT_SCRIPT],
            stdout=subprocess.PIPE,
            text=True,
        )
        child_pid: int | None = None
        try:
            line = parent.stdout.readline()
            child_pid = int(line.strip())
            self.assertTrue(_pid_alive(child_pid), "child never started")

            # Simulate the exact failure mode: the parent is killed with no
            # chance to run its own cleanup (SIGKILL, not SIGTERM).
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=2.0)

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and _pid_alive(child_pid):
                time.sleep(0.05)

            self.assertFalse(_pid_alive(child_pid), "child survived a SIGKILL of its parent")
        finally:
            if parent.poll() is None:
                parent.kill()
            if child_pid is not None and _pid_alive(child_pid):
                os.kill(child_pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
