"""Bugs fixed: a silent client used to wedge the single-threaded control
server forever (no timeout on the accepted socket), and bad enum values from
a control client used to be queued and applied with no rejection."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from queue import Empty, Queue

from clean_speech_daemon.config import validate_control_command
from clean_speech_daemon.control import ControlServer, send_control_command


class ControlServerTests(unittest.TestCase):
    def _start(self, directory: str) -> tuple[ControlServer, Queue, str]:
        path = os.path.join(directory, "control.sock")
        queue: Queue = Queue()
        server = ControlServer(path, queue)
        server.start()
        return server, queue, path

    def test_valid_command_is_queued_and_acked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server, queue, path = self._start(directory)
            try:
                response = send_control_command(path, {"reference_delay_ms": 12.0})
                self.assertTrue(response["ok"])
                self.assertEqual(queue.get_nowait(), {"reference_delay_ms": 12.0})
            finally:
                server.stop()

    def test_invalid_echo_canceller_is_rejected_not_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server, queue, path = self._start(directory)
            try:
                response = send_control_command(path, {"echo_canceller": "not-a-real-backend"})
                self.assertFalse(response["ok"])
                self.assertIn("echo_canceller", response["error"])
                with self.assertRaises(Empty):
                    queue.get_nowait()
            finally:
                server.stop()

    def test_silent_client_does_not_wedge_the_server(self) -> None:
        # Regression: recv() on the accepted socket had no timeout, so a client
        # that connects and sends nothing blocked _serve() forever, starving
        # every subsequent connection.
        with tempfile.TemporaryDirectory() as directory:
            server, queue, path = self._start(directory)
            try:
                silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                silent.connect(path)
                # Deliberately never send anything on `silent`.

                # A second, well-behaved client must still get served promptly
                # instead of queuing up behind the stuck one.
                start = time.monotonic()
                response = send_control_command(path, {"reference_delay_ms": 5.0}, timeout=3.0)
                elapsed = time.monotonic() - start

                self.assertTrue(response["ok"])
                self.assertLess(elapsed, 3.0, "server appears wedged behind the silent client")
            finally:
                silent.close()
                server.stop()


class ValidateControlCommandTests(unittest.TestCase):
    def test_accepts_known_values(self) -> None:
        self.assertIsNone(validate_control_command({"echo_canceller": "hybrid_localvqe"}))
        self.assertIsNone(validate_control_command({"reference_delay_mode": "auto"}))
        self.assertIsNone(validate_control_command({}))

    def test_rejects_unknown_echo_canceller(self) -> None:
        error = validate_control_command({"echo_canceller": "bogus"})
        self.assertIsNotNone(error)
        self.assertIn("bogus", error)

    def test_rejects_unknown_reference_delay_mode(self) -> None:
        error = validate_control_command({"reference_delay_mode": "sometimes"})
        self.assertIsNotNone(error)
        self.assertIn("sometimes", error)


if __name__ == "__main__":
    unittest.main()
