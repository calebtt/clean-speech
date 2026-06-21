from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Iterable

import numpy as np
import sounddevice as sd


@dataclass(slots=True)
class AudioFrame:
    data: np.ndarray
    timestamp: float | None


def list_input_devices() -> list[str]:
    devices = sd.query_devices()
    rows: list[str] = []
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            rows.append(
                f"{index}: {device['name']} "
                f"({device['max_input_channels']} in, default {device['default_samplerate']:.0f} Hz)"
            )
    return rows


def list_pulse_sources() -> list[str]:
    try:
        result = subprocess.run(["pactl", "list", "short", "sources"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return [f"unable to list Pulse/PipeWire sources: {exc}"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def resolve_input_device(selector: str) -> int | None:
    if selector in ("", "auto", "default", None):
        return None
    try:
        return int(selector)
    except ValueError:
        pass

    selector_lower = selector.lower()
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0 and selector_lower in str(device["name"]).lower():
            return index
    raise ValueError(f"No input device matched {selector!r}")


class InputStreamReader:
    def __init__(self, device: str, sample_rate: int, channels: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.queue: Queue[AudioFrame] = Queue(maxsize=80)
        self.device = resolve_input_device(device)
        self.stream: sd.InputStream | None = None

    def __enter__(self) -> "InputStreamReader":
        def callback(indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
            if status:
                print(f"audio input status: {status}", flush=True)
            frame = np.asarray(indata, dtype=np.float32)
            if frame.ndim == 2:
                frame = frame.mean(axis=1)
            if frames != self.frame_samples:
                frame = _fit_frame(frame, self.frame_samples)
            try:
                self.queue.put_nowait(AudioFrame(frame.copy(), getattr(time, "inputBufferAdcTime", None)))
            except Exception:
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(AudioFrame(frame.copy(), getattr(time, "inputBufferAdcTime", None)))
                except Exception:
                    pass

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def read(self, timeout: float = 1.0) -> AudioFrame:
        return self.queue.get(timeout=timeout)

    def read_latest_or_none(self) -> AudioFrame | None:
        latest: AudioFrame | None = None
        while True:
            try:
                latest = self.queue.get_nowait()
            except Empty:
                return latest

    def read_next_or_none(self, timeout: float = 0.0) -> AudioFrame | None:
        try:
            if timeout > 0:
                return self.queue.get(timeout=timeout)
            return self.queue.get_nowait()
        except Empty:
            return None


class PulseMonitorReader:
    def __init__(self, source: str, sample_rate: int, frame_ms: int) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2
        self.queue: Queue[AudioFrame] = Queue(maxsize=80)
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.resolved_source = ""
        self.frames_read = 0
        self.bytes_read = 0
        self.frames_dropped = 0
        self.last_read_time = 0.0
        self.last_error = ""

    def __enter__(self) -> "PulseMonitorReader":
        source = resolve_pulse_monitor_source(self.source)
        self.resolved_source = source
        command = [
            "parec",
            f"--device={source}",
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            "--channels=1",
            # Without an explicit low latency, parec batches with a large internal
            # buffer and delivers the monitor in bursts at ~80% of real-time, which
            # starves the reference buffer (reference present only ~20% of the time)
            # so the echo canceller never converges. A small latency makes parec
            # deliver steadily at real-time.
            "--latency-msec=20",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"system audio reference: {source}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop_event.set()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while not self.stop_event.is_set():
            data = self.process.stdout.read(self.frame_bytes)
            if not data:
                self.last_error = self._read_stderr() or "parec stdout ended"
                break
            if len(data) != self.frame_bytes:
                self.last_error = f"short read: {len(data)} of {self.frame_bytes} bytes"
                continue
            frame = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            self.frames_read += 1
            self.bytes_read += len(data)
            self.last_read_time = time.monotonic()
            try:
                self.queue.put_nowait(AudioFrame(frame.copy(), None))
            except Exception:
                try:
                    self.queue.get_nowait()
                    self.frames_dropped += 1
                    self.queue.put_nowait(AudioFrame(frame.copy(), None))
                except Exception:
                    pass

    def _read_stderr(self) -> str:
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def read_latest_or_none(self) -> AudioFrame | None:
        latest: AudioFrame | None = None
        while True:
            try:
                latest = self.queue.get_nowait()
            except Empty:
                return latest

    def read_next_or_none(self, timeout: float = 0.0) -> AudioFrame | None:
        try:
            if timeout > 0:
                return self.queue.get(timeout=timeout)
            return self.queue.get_nowait()
        except Empty:
            return None


def resolve_pulse_monitor_source(selector: str) -> str:
    if selector in ("", "auto", None):
        running = _running_monitor_source()
        if running:
            return running
        return _default_monitor_source()
    if selector == "default":
        return _default_monitor_source()
    if selector == "running":
        running = _running_monitor_source()
        if running:
            return running
        return _default_monitor_source()
    if selector.startswith("@"):
        return selector
    if ".monitor" not in selector:
        match = _monitor_matching(selector)
        if match:
            return match
    return selector


def _default_monitor_source() -> str:
    result = subprocess.run(["pactl", "get-default-sink"], check=True, capture_output=True, text=True)
    return result.stdout.strip() + ".monitor"


def _running_monitor_source() -> str | None:
    monitors = _pulse_monitor_rows()
    for fields in monitors:
        if len(fields) >= 5 and fields[4] == "RUNNING":
            return fields[1]
    return None


def _monitor_matching(selector: str) -> str | None:
    selector_lower = selector.lower()
    for fields in _pulse_monitor_rows():
        if len(fields) >= 2 and selector_lower in fields[1].lower():
            return fields[1]
    return None


def _pulse_monitor_rows() -> list[list[str]]:
    try:
        result = subprocess.run(["pactl", "list", "short", "sources"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].endswith(".monitor") and "clean_speech_microphone" not in fields[1]:
            rows.append(fields)
    return rows


def _fit_frame(frame: np.ndarray, frame_samples: int) -> np.ndarray:
    if len(frame) == frame_samples:
        return frame
    if len(frame) > frame_samples:
        return frame[:frame_samples]
    out = np.zeros(frame_samples, dtype=np.float32)
    out[: len(frame)] = frame
    return out


def pcm16_bytes(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def frames_from_socket_bytes(chunks: Iterable[bytes], frame_samples: int) -> Iterable[np.ndarray]:
    pending = b""
    frame_bytes = frame_samples * 2
    for chunk in chunks:
        pending += chunk
        while len(pending) >= frame_bytes:
            raw = pending[:frame_bytes]
            pending = pending[frame_bytes:]
            yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
