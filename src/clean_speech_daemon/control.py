"""Unix-socket runtime tuning interface for the live daemon."""

from __future__ import annotations

import json
import os
import socket
import threading
from queue import Queue
from typing import Callable


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
            with client:
                try:
                    payload = self._read_json_line(client)
                    if isinstance(payload, dict):
                        self.command_queue.put(payload)
                        response = {"ok": True, "queued": True}
                    else:
                        response = {"ok": False, "error": "expected JSON object"}
                except Exception as exc:  # noqa: BLE001
                    response = {"ok": False, "error": str(exc)}
                client.sendall((json.dumps(response) + "\n").encode("utf-8"))

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