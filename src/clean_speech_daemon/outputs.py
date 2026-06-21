from __future__ import annotations

import base64
import errno
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import soundfile as sf

from .audio import pcm16_bytes


MULTI_STREAM_NAMES = ["mic_raw", "system_reference", "reference_aligned", "reference_matched", "after_echo", "cleaned_output"]


class SocketPcmOutput:
    def __init__(self, path: str, sample_rate: int, channels: int) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.server: socket.socket | None = None

    def start(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(8)
        os.chmod(self.path, 0o666)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self) -> None:
        assert self.server is not None
        self.server.settimeout(0.25)
        while not self.stop_event.is_set():
            try:
                client, _ = self.server.accept()
                metadata = {
                    "format": "s16le",
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "note": "One JSON metadata line followed by raw PCM frames.",
                }
                client.sendall((json.dumps(metadata) + "\n").encode("utf-8"))
                with self.lock:
                    self.clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def write(self, frame) -> None:  # noqa: ANN001
        data = pcm16_bytes(frame)
        stale: list[socket.socket] = []
        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(data)
                except OSError:
                    stale.append(client)
            for client in stale:
                self.clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.close()
        with self.lock:
            for client in self.clients:
                client.close()
            self.clients.clear()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class MultiStreamSocketOutput:
    def __init__(self, path: str, sample_rate: int, frame_samples: int) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.server: socket.socket | None = None
        self.frame_index = 0

    def start(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(8)
        os.chmod(self.path, 0o666)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()

    def _accept_loop(self) -> None:
        assert self.server is not None
        self.server.settimeout(0.25)
        while not self.stop_event.is_set():
            try:
                client, _ = self.server.accept()
                metadata = {
                    "protocol": "clean-speech-multistream-v1",
                    "encoding": "json-base64-pcm16le",
                    "sample_rate": self.sample_rate,
                    "channels": 1,
                    "frame_samples": self.frame_samples,
                    "streams": MULTI_STREAM_NAMES,
                    "note": "One JSON metadata line, then uint32le packet length followed by JSON packets.",
                }
                client.sendall((json.dumps(metadata) + "\n").encode("utf-8"))
                with self.lock:
                    self.clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def write(self, mic, reference, reference_aligned, reference_matched, after_echo, cleaned) -> None:  # noqa: ANN001
        self.frame_index += 1
        zero = np.zeros(self.frame_samples, dtype=np.float32)
        packet = {
            "frame_index": self.frame_index,
            "time": time.time(),
            "reference_present": reference is not None,
            "streams": {
                "mic_raw": _encode_pcm16(mic),
                "system_reference": _encode_pcm16(reference if reference is not None else zero),
                "reference_aligned": _encode_pcm16(reference_aligned if reference_aligned is not None else zero),
                "reference_matched": _encode_pcm16(reference_matched if reference_matched is not None else zero),
                "after_echo": _encode_pcm16(after_echo if after_echo is not None else zero),
                "cleaned_output": _encode_pcm16(cleaned),
            },
        }
        body = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        data = struct.pack("<I", len(body)) + body
        stale: list[socket.socket] = []
        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(data)
                except OSError:
                    stale.append(client)
            for client in stale:
                self.clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.close()
        with self.lock:
            for client in self.clients:
                client.close()
            self.clients.clear()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _encode_pcm16(frame) -> str:  # noqa: ANN001
    return base64.b64encode(pcm16_bytes(frame)).decode("ascii")


class FifoPcmOutput:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.fd: int | None = None
        self.last_retry = 0.0

    def start(self) -> None:
        if self.path.exists() and not self.path.is_fifo():
            self.path.unlink()
        if not self.path.exists():
            os.mkfifo(self.path, 0o666)
        os.chmod(self.path, 0o666)

    def _connect(self) -> bool:
        if self.fd is not None:
            return True
        now = time.monotonic()
        if now - self.last_retry < 0.5:
            return False
        self.last_retry = now
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
            return True
        except OSError as exc:
            if exc.errno in (errno.ENXIO, errno.ENOENT):
                return False
            raise

    def write(self, frame) -> None:  # noqa: ANN001
        if not self._connect() or self.fd is None:
            return
        try:
            os.write(self.fd, pcm16_bytes(frame))
        except BlockingIOError:
            return
        except BrokenPipeError:
            os.close(self.fd)
            self.fd = None
        except OSError:
            if self.fd is not None:
                os.close(self.fd)
            self.fd = None

    def stop(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class DebugWavOutput:
    def __init__(self, path: str, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.file: sf.SoundFile | None = None

    def start(self) -> None:
        if self.path:
            self.file = sf.SoundFile(self.path, mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16")

    def write(self, frame) -> None:  # noqa: ANN001
        if self.file is not None:
            self.file.write(frame)

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()


class PulsePipeSource:
    def __init__(self, source_name: str, description: str, fifo_path: str, sample_rate: int, channels: int) -> None:
        self.source_name = source_name
        self.description = description
        self.fifo_path = fifo_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.module_id: str | None = None

    def start(self) -> None:
        command = [
            "pactl",
            "load-module",
            "module-pipe-source",
            f"source_name={self.source_name}",
            f"file={self.fifo_path}",
            "format=s16le",
            f"rate={self.sample_rate}",
            f"channels={self.channels}",
            f"source_properties=device.description={self.description}",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.module_id = result.stdout.strip()
            print(f"loaded virtual microphone {self.source_name} as module {self.module_id}", flush=True)
        except FileNotFoundError:
            print("pactl not found; socket/FIFO outputs are still available", flush=True)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            print(f"could not load Pulse/PipeWire virtual source: {message}", flush=True)

    def stop(self) -> None:
        if self.module_id:
            subprocess.run(["pactl", "unload-module", self.module_id], check=False)
            self.module_id = None
