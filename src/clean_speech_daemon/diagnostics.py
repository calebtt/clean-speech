from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import soundfile as sf


def audio_metrics(frame: np.ndarray | None) -> dict[str, float | int | None]:
    if frame is None or len(frame) == 0:
        return {
            "rms": None,
            "rms_dbfs": None,
            "peak": None,
            "peak_dbfs": None,
            "clipped_pct": None,
            "zero_pct": None,
            "zcr": None,
            "dc": None,
        }
    x = np.asarray(frame, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(np.abs(x)))
    clipped = float(np.mean(np.abs(x) >= 0.98) * 100.0)
    zero = float(np.mean(np.abs(x) <= 1e-5) * 100.0)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))))) if len(x) > 1 else 0.0
    dc = float(np.mean(x))
    return {
        "rms": rms,
        "rms_dbfs": 20.0 * math.log10(rms),
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
        "clipped_pct": clipped,
        "zero_pct": zero,
        "zcr": zcr,
        "dc": dc,
    }


def correlation(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return None
    ax = np.asarray(a, dtype=np.float32)
    bx = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(ax) * np.linalg.norm(bx))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(ax, bx) / denom)


class DiagnosticsLogger:
    """Throttled diagnostics writer.

    The actual disk I/O (status overwrite + JSONL append) runs on a background
    thread so it never stalls the real-time audio loop. A hot-loop stall lets the
    reference reader burst frames, which the clock-sync buffer then has to absorb;
    keeping the writer off the audio thread removes that periodic hiccup.
    """

    def __init__(self, log_path: str, status_path: str, interval_ms: int, print_interval_ms: int) -> None:
        self.log_path = log_path
        self.status_path = status_path
        self.interval_ms = interval_ms
        self.print_interval_ms = print_interval_ms
        self.last_write = 0.0
        self._pending: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, name="diagnostics-writer", daemon=True)
        self._thread.start()

    def maybe_write(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self.last_write < self.interval_ms / 1000.0:
            return
        self.last_write = now
        payload = dict(payload)
        payload["time"] = time.time()
        with self._lock:
            self._pending = payload  # latest-wins; never blocks the audio loop
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _writer_loop(self) -> None:
        last_print = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self._lock:
                payload = self._pending
                self._pending = None
            if payload is None:
                continue
            try:
                log_path = Path(self.log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(payload, sort_keys=True) + "\n")
                status_path = Path(self.status_path)
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except OSError:
                pass

            now = time.monotonic()
            if now - last_print >= self.print_interval_ms / 1000.0:
                last_print = now
                out = payload.get("output", {})
                mic = payload.get("mic", {})
                pipeline = payload.get("pipeline", {})
                print(
                    "diag "
                    f"mic_rms={mic.get('rms_dbfs')} output_rms={out.get('rms_dbfs')} "
                    f"vad={pipeline.get('vad_score')} speech={pipeline.get('speech_ratio')} "
                    f"echo_gain={pipeline.get('echo_gain')} ref_corr={payload.get('reference_correlation')}",
                    flush=True,
                )


@dataclass(slots=True)
class StageWavWriter:
    directory: str
    sample_rate: int
    files: dict[str, sf.SoundFile] = field(default_factory=dict)

    def start(self) -> None:
        Path(self.directory).mkdir(parents=True, exist_ok=True)

    def write(self, stages: dict[str, np.ndarray]) -> None:
        for name, frame in stages.items():
            if name not in self.files:
                path = Path(self.directory) / f"{name}.wav"
                self.files[name] = sf.SoundFile(path, mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16")
            self.files[name].write(np.asarray(frame, dtype=np.float32))

    def stop(self) -> None:
        for file in self.files.values():
            file.close()
        self.files.clear()
