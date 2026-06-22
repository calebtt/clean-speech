"""The realtime loop must never block on a slow monitor consumer.

A blocking ``sendall`` to an unresponsive client stalls the audio loop, overflows
the mic/parec input queues, and decimates every stream into sped-up/chipmunk audio.
These tests pin the non-blocking, drop-old-packets behaviour of the socket outputs.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest

import numpy as np

from clean_speech_daemon.outputs import MultiStreamSocketOutput


def _read_metadata(sock: socket.socket) -> dict:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise OSError("closed before metadata")
        buf += chunk
    return json.loads(buf.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise OSError("closed")
        data += chunk
    return data


class NonBlockingOutputTests(unittest.TestCase):
    def _wait_for_client(self, out: MultiStreamSocketOutput) -> None:
        for _ in range(300):
            with out.lock:
                if out.clients:
                    return
            time.sleep(0.01)
        self.fail("server never registered the client")

    def test_unresponsive_client_never_blocks_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "streams.sock")
            out = MultiStreamSocketOutput(path, 48_000, 960)
            out.start()
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
                client.connect(path)
                _read_metadata(client)  # then never read again -> socket buffer fills
                self._wait_for_client(out)

                frame = np.zeros(960, dtype=np.float32)
                done = threading.Event()

                def hammer() -> None:
                    for _ in range(20_000):
                        out.write(frame, frame, frame, frame, frame, frame)
                    done.set()

                thread = threading.Thread(target=hammer, daemon=True)
                thread.start()
                thread.join(timeout=15)
                self.assertTrue(done.is_set(), "write() blocked on an unresponsive client")

                with out.lock:
                    self.assertTrue(out.clients, "slow client was dropped instead of buffered")
                    self.assertGreater(out.clients[0].dropped, 0, "expected old packets to be dropped under backpressure")
            finally:
                out.stop()
                client.close()

    def test_responsive_client_receives_contiguous_packets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "streams.sock")
            out = MultiStreamSocketOutput(path, 48_000, 960)
            out.start()
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            received: list[int] = []
            try:
                client.connect(path)
                _read_metadata(client)
                self._wait_for_client(out)

                stop = threading.Event()

                def reader() -> None:
                    client.settimeout(2.0)
                    while not stop.is_set():
                        try:
                            length = struct.unpack("<I", _recv_exact(client, 4))[0]
                            packet = json.loads(_recv_exact(client, length).decode("utf-8"))
                        except (OSError, ValueError):
                            return
                        received.append(int(packet["frame_index"]))

                thread = threading.Thread(target=reader, daemon=True)
                thread.start()

                frame = np.zeros(960, dtype=np.float32)
                for _ in range(300):
                    out.write(frame, frame, frame, frame, frame, frame)
                    time.sleep(0.001)
                time.sleep(0.2)
                stop.set()
                thread.join(timeout=2)
            finally:
                out.stop()
                client.close()

            self.assertGreater(len(received), 100, "responsive client received almost nothing")
            # A keeping-up reader should see every packet in order with no gaps/corruption.
            self.assertEqual(received, list(range(received[0], received[0] + len(received))))


if __name__ == "__main__":
    unittest.main()
