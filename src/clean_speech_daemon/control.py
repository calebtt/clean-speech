"""Unix-socket runtime tuning interface for the live daemon."""

from __future__ import annotations

import json
import os
import socket
import threading
from queue import Queue
from typing import Callable

from .config import validate_control_command

# A connected client that sends nothing (or nothing terminated by '\n') would
# otherwise block recv() forever on this single-threaded server, wedging the
# control socket for every other client until that connection is killed.
_CLIENT_RECV_TIMEOUT = 5.0


class ControlServer:
    def __init__(self, socket_path: str, command_queue: Queue[dict[str, object]]) -> None:
        self.socket_path = socket_path
        self.command_queue = command_queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None

    def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        server.listen(8)
        server.settimeout(0.5)
        self._server = server
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="clean-speech-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _addr = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            client.settimeout(_CLIENT_RECV_TIMEOUT)
            # Handle each connection on its own thread: this loop used to process
            # clients one at a time inline, so even with a recv timeout a single
            # slow/silent client would stall every other client behind it for up
            # to that timeout. A thread per connection means a stuck client only
            # blocks itself.
            threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            try:
                payload = self._read_json_line(client)
                if isinstance(payload, dict):
                    error = validate_control_command(payload)
                    if error is not None:
                        response = {"ok": False, "error": error}
                    else:
                        self.command_queue.put(payload)
                        response = {"ok": True, "queued": True}
                else:
                    response = {"ok": False, "error": "expected JSON object"}
            except Exception as exc:  # noqa: BLE001
                response = {"ok": False, "error": str(exc)}
            try:
                client.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except OSError:
                pass  # client disconnected/timed out before we could reply

    @staticmethod
    def _read_json_line(client: socket.socket) -> object:
        data = b""
        while b"\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        line = data.split(b"\n", 1)[0].strip()
        if not line:
            raise ValueError("empty control message")
        return json.loads(line.decode("utf-8"))


def send_control_command(socket_path: str, payload: dict[str, object], timeout: float = 1.0) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(socket_path)
        client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk
        line = data.split(b"\n", 1)[0].strip()
        if not line:
            return {"ok": False, "error": "no response"}
        result = json.loads(line.decode("utf-8"))
        return result if isinstance(result, dict) else {"ok": False, "error": "invalid response"}