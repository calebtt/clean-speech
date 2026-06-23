#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import socket
import struct
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf


DEFAULT_STREAMS_SOCKET = "/tmp/clean-speech-daemon-streams.sock"
DEFAULT_CONTROL_SOCKET = "/tmp/clean-speech-daemon-control.sock"
DEFAULT_STATUS = Path("/tmp/clean-speech-daemon-status.json")
DEFAULT_RECORD_DIR = Path.home() / "clean-speech-recordings"

ALIGNMENT_INSTRUCTIONS = """\
Goal: cancel the system audio echo. The adaptive (NLMS) filter can only subtract \
echo it has already received, so the reference must LEAD the echo in the mic. On \
many machines the parec monitor reference arrives 15-30 ms LATE, which makes \
cancellation impossible until you delay the mic to compensate.

Workflow:
  1. Play steady system audio (music) and connect. Watch "Live Alignment" below.
  2. Read "mic vs reference": this is the echo. If correlation is near zero there
     is no echo to cancel (try louder audio / different output).
  3. Read the lag direction:
       - "reference LATE" (negative)  -> echo is non-causal. Raise Mic Delay by
         about the reported amount + 20 ms, then press "Reset Echo Filter".
       - "reference leads" (positive) -> good; the filter can work.
  4. Keep Reference Delay mode = manual, delay = 0, and let the filter's taps
     absorb the residual lead. (Auto mode currently wanders and resets the
     filter, preventing convergence.)
  5. After each change press "Reset Echo Filter" and wait ~2 s to re-converge.
  6. Success = "after_echo vs reference" drops well below "mic vs reference",
     AND "after_echo / mic level" stays <= 0 dB (no boost = no divergence).
  7. Listen: set Play stream = "after_echo" to hear the canceller in isolation.
  8. Press "Save Streams" to write a 12 s capture + alignment report for offline
     replay (testbed/replay_session.py).
"""
STREAM_NAMES = ("mic_raw", "system_reference", "reference_aligned", "reference_matched", "after_echo", "cleaned_output")
STREAM_LABELS = {
    "mic_raw": "Mic Raw",
    "system_reference": "System Reference",
    "reference_aligned": "Reference Aligned",
    "reference_matched": "Reference Matched",
    "after_echo": "After Echo Cancellation",
    "cleaned_output": "Cleaned Output",
}
BUFFER_SECONDS = 12


@dataclass(slots=True)
class MultiStreamPacket:
    streams: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]
    reference_present: bool


class MultiStreamSocketClient:
    def __init__(self, socket_path: str, queue: Queue[MultiStreamPacket], status: Queue[str]) -> None:
        self.socket_path = socket_path
        self.queue = queue
        self.status = status
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.metadata: dict[str, object] = {"sample_rate": 48_000, "frame_samples": 960}

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._connect_and_read()
            except FileNotFoundError:
                self.status.put(f"Socket not found: {self.socket_path}")
            except ConnectionRefusedError:
                self.status.put("Daemon stream socket refused connection")
            except OSError as exc:
                self.status.put(f"Stream socket error: {exc}")
            except Exception as exc:
                self.status.put(f"Stream decode error: {exc}")
            if not self.stop_event.is_set():
                time.sleep(1.0)

    def _connect_and_read(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(self.socket_path)
            metadata_line = b""
            while not metadata_line.endswith(b"\n"):
                chunk = client.recv(1)
                if not chunk:
                    raise OSError("socket closed before metadata")
                metadata_line += chunk
            self.metadata = json.loads(metadata_line.decode("utf-8"))
            sample_rate = int(self.metadata.get("sample_rate", 48_000))
            streams = ", ".join(self.metadata.get("streams", STREAM_NAMES))
            self.status.put(f"Connected: {sample_rate} Hz streams: {streams}")

            while not self.stop_event.is_set():
                length_raw = recv_exact(client, 4)
                length = struct.unpack("<I", length_raw)[0]
                payload = json.loads(recv_exact(client, length).decode("utf-8"))
                frames = {
                    name: decode_pcm16(payload["streams"].get(name, ""))
                    for name in STREAM_NAMES
                }
                metrics = {name: frame_metrics(frame) for name, frame in frames.items()}
                packet = MultiStreamPacket(
                    streams=frames,
                    metrics=metrics,
                    reference_present=bool(payload.get("reference_present", False)),
                )
                try:
                    self.queue.put_nowait(packet)
                except Exception:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(packet)
                    except Exception:
                        pass


class PlaybackOutput:
    """Faithful monitor playback.

    The callback's block size is chosen by PortAudio and is generally not equal to
    the 960-sample stream frame, so frame-at-a-time output dropped or zero-padded
    part of every frame (garbled / pitch-shifted audio). Instead we keep a sample
    buffer and let the callback drain exactly the block size it asks for.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.lock = threading.Lock()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.max_samples = sample_rate  # cap latency at ~1 s
        self.stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self.stream is not None:
            return

        def callback(outdata, frames, _time, status) -> None:  # noqa: ANN001
            with self.lock:
                n = min(frames, len(self.buffer))
                outdata[:n, 0] = self.buffer[:n]
                self.buffer = self.buffer[n:]
            if n < frames:
                outdata[n:].fill(0)

        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32", callback=callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)

    def write(self, frame: np.ndarray) -> None:
        chunk = np.asarray(frame, dtype=np.float32).ravel()
        with self.lock:
            self.buffer = np.concatenate([self.buffer, chunk])
            if len(self.buffer) > self.max_samples:
                self.buffer = self.buffer[-self.max_samples :]


class CleanSpeechTestbed(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Clean Speech Testbed")
        self.geometry("960x760")
        self.minsize(780, 640)

        self.audio_queue: Queue[MultiStreamPacket] = Queue(maxsize=300)
        self.status_queue: Queue[str] = Queue()
        self.client: MultiStreamSocketClient | None = None
        self.playback = PlaybackOutput(48_000)
        self.record_dir = DEFAULT_RECORD_DIR
        self.frame_rate = 50
        self.buffer_frames = BUFFER_SECONDS * self.frame_rate
        self.buffers = {name: deque(maxlen=self.buffer_frames) for name in STREAM_NAMES}
        self.last_packet: MultiStreamPacket | None = None
        self.last_packet_time = 0.0
        self.frames_seen = 0
        # Draining the socket queue must stay cheap so the reader thread is never
        # GIL-starved (which would make the daemon drop packets to us and corrupt
        # saved audio). Redraw the heavy waveforms at a throttled rate instead of
        # on every drain tick.
        self._last_draw = 0.0

        self.socket_var = tk.StringVar(value=DEFAULT_STREAMS_SOCKET)
        self.control_socket_var = tk.StringVar(value=DEFAULT_CONTROL_SOCKET)
        self.status_var = tk.StringVar(value="Disconnected")
        self.level_var = tk.DoubleVar(value=0.0)
        self.peak_var = tk.DoubleVar(value=0.0)
        self.play_var = tk.BooleanVar(value=False)
        self.play_stream_var = tk.StringVar(value="cleaned_output")
        self.frames_var = tk.StringVar(value="Frames: 0")
        self.buffer_var = tk.StringVar(value=f"In-memory buffer: last {BUFFER_SECONDS}s, not saving to disk")
        self.diagnostics_var = tk.StringVar(value="Diagnostics: waiting for daemon status file")
        self.canvas_by_stream: dict[str, tk.Canvas] = {}
        self.level_label_by_stream = {name: tk.StringVar(value="RMS n/a, peak n/a") for name in STREAM_NAMES}

        # Live-tuning controls (pushed to the daemon control socket).
        self.mic_delay_var = tk.IntVar(value=60)
        self.ref_delay_var = tk.IntVar(value=0)
        self.delay_mode_var = tk.StringVar(value="manual")
        self.control_status_var = tk.StringVar(value="Control: idle")
        # Live alignment readout, computed from the in-memory buffers.
        self.align_echo_var = tk.StringVar(value="mic vs reference: waiting for audio")
        self.align_lag_var = tk.StringVar(value="lag: --")
        self.align_residual_var = tk.StringVar(value="after_echo vs reference: --")
        self.align_diverge_var = tk.StringVar(value="after_echo / mic level: --")
        self.align_verdict_var = tk.StringVar(value="Verdict: waiting...")
        self.align_verdict_label: ttk.Label | None = None

        # Live AEC model switcher + effectiveness readout.
        self.model_var = tk.StringVar(value="hybrid")
        self.mask_smooth_var = tk.DoubleVar(value=0.6)
        self.aec_active_var = tk.StringVar(value="active model: (waiting for daemon)")
        self.aec_reduction_var = tk.StringVar(value="echo removed (mic → after_echo): -- dB")
        self.aec_reduction_label: ttk.Label | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(40, self._pump_audio)
        self.after(250, self._pump_status)
        self.after(1000, self._pump_diagnostics)
        self.after(750, self._update_alignment)
        self.after(200, self.connect)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        connection = ttk.LabelFrame(outer, text="Daemon Multi-Stream Connection", padding=10)
        connection.pack(fill=tk.X)

        row1 = ttk.Frame(connection)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="Stream socket", width=14).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.socket_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row1, text="Connect", command=self.connect).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Disconnect", command=self.disconnect).pack(side=tk.LEFT)
        row2 = ttk.Frame(connection)
        row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row2, text="Control socket", width=14).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.control_socket_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Button(row2, text="Workflow / Help", command=self.show_instructions).pack(side=tk.LEFT)

        self._build_aec_model_panel(outer)
        self._build_alignment_panel(outer)

        meters = ttk.LabelFrame(outer, text="Stream Overview", padding=10)
        meters.pack(fill=tk.X, pady=12)
        ttk.Label(meters, textvariable=self.status_var).pack(anchor=tk.W)
        ttk.Label(meters, textvariable=self.frames_var).pack(anchor=tk.W)
        ttk.Label(meters, textvariable=self.buffer_var).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(meters, text="Cleaned Output RMS Level").pack(anchor=tk.W)
        ttk.Progressbar(meters, variable=self.level_var, maximum=1.0).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(meters, text="Cleaned Output Peak Level").pack(anchor=tk.W)
        ttk.Progressbar(meters, variable=self.peak_var, maximum=1.0).pack(fill=tk.X)

        waveforms = ttk.LabelFrame(outer, text="In-Memory Waveforms", padding=10)
        waveforms.pack(fill=tk.BOTH, expand=True)
        for name in STREAM_NAMES:
            ttk.Label(waveforms, text=f"{STREAM_LABELS[name]}: {name}").pack(anchor=tk.W)
            ttk.Label(waveforms, textvariable=self.level_label_by_stream[name]).pack(anchor=tk.W)
            canvas = tk.Canvas(waveforms, height=95, background="#101418", highlightthickness=1, highlightbackground="#30363d")
            canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
            self.canvas_by_stream[name] = canvas

        controls = ttk.LabelFrame(outer, text="Test Controls", padding=10)
        controls.pack(fill=tk.X, pady=(12, 0))
        ttk.Checkbutton(controls, text="Play stream", variable=self.play_var, command=self._toggle_playback).pack(side=tk.LEFT)
        play_menu = ttk.Combobox(
            controls,
            textvariable=self.play_stream_var,
            values=("cleaned_output", "after_echo"),
            state="readonly",
            width=16,
        )
        play_menu.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controls, text="Save Streams", command=self.save_streams).pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="Choose Save Folder", command=self.choose_record_dir).pack(side=tk.LEFT)
        ttk.Button(controls, text="Clear Buffer", command=self.clear_buffer).pack(side=tk.RIGHT)

        diagnostics = ttk.LabelFrame(outer, text="Daemon Diagnostics", padding=10)
        diagnostics.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(diagnostics, textvariable=self.diagnostics_var, justify=tk.LEFT).pack(anchor=tk.W)

    def _build_aec_model_panel(self, outer: ttk.Frame) -> None:
        panel = ttk.LabelFrame(outer, text="AEC Model (live switch via control socket)", padding=10)
        panel.pack(fill=tk.X, pady=(12, 0))

        row = ttk.Frame(panel)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Canceller").pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.model_var, state="readonly", width=9,
                     values=("hybrid", "dtln", "nkf", "nlms", "scalar")).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row, text="DTLN mask smoothing").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0.0, to=0.95, increment=0.05, width=5,
                    textvariable=self.mask_smooth_var, format="%.2f").pack(side=tk.LEFT, padx=(4, 14))
        ttk.Button(row, text="Apply Model", command=self._apply_model).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.aec_active_var, foreground="#9aa4b2").pack(side=tk.RIGHT)

        self.aec_reduction_label = ttk.Label(panel, textvariable=self.aec_reduction_var,
                                             font=("TkDefaultFont", 11, "bold"))
        self.aec_reduction_label.pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(panel, text="hybrid = NKF→DTLN (deep + voice kept) · dtln = deep/wavy · "
                              "nkf = clean/shallow · nlms = linear baseline",
                  foreground="#9aa4b2").pack(anchor=tk.W)

    def _apply_model(self) -> None:
        self._send_control({
            "echo_canceller": self.model_var.get(),
            "dtln_mask_smoothing": float(self.mask_smooth_var.get()),
        })

    def _build_alignment_panel(self, outer: ttk.Frame) -> None:
        align = ttk.LabelFrame(outer, text="Echo Alignment (live tuning via control socket)", padding=10)
        align.pack(fill=tk.X, pady=12)

        hint = (
            "Reference must LEAD the echo for cancellation. If the readout says "
            "\"reference LATE\", raise Mic Delay by that amount + ~20 ms and Reset. "
            "Click Workflow / Help for the full procedure."
        )
        ttk.Label(align, text=hint, wraplength=900, justify=tk.LEFT, foreground="#9aa4b2").pack(anchor=tk.W, pady=(0, 8))

        controls = ttk.Frame(align)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Mic delay (ms)").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=0, to=200, increment=5, width=5, textvariable=self.mic_delay_var).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Button(controls, text="-5", width=3, command=lambda: self._nudge_mic_delay(-5)).pack(side=tk.LEFT)
        ttk.Button(controls, text="+5", width=3, command=lambda: self._nudge_mic_delay(5)).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(controls, text="Apply", command=self._apply_mic_delay).pack(side=tk.LEFT, padx=(0, 14))

        ttk.Label(controls, text="Ref delay (ms)").pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=0, to=200, increment=1, width=5, textvariable=self.ref_delay_var).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(controls, text="mode").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Combobox(controls, textvariable=self.delay_mode_var, values=("manual", "auto"), state="readonly", width=7).pack(side=tk.LEFT)
        ttk.Button(controls, text="Apply", command=self._apply_ref_delay).pack(side=tk.LEFT, padx=(2, 14))

        ttk.Button(controls, text="Reset Echo Filter", command=self._reset_echo_filter).pack(side=tk.LEFT)

        readout = ttk.Frame(align)
        readout.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(readout, textvariable=self.align_echo_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(readout, textvariable=self.align_lag_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(readout, textvariable=self.align_residual_var).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(readout, textvariable=self.align_diverge_var).grid(row=1, column=1, sticky=tk.W)
        self.align_verdict_label = ttk.Label(readout, textvariable=self.align_verdict_var, font=("TkDefaultFont", 10, "bold"))
        self.align_verdict_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Label(align, textvariable=self.control_status_var, foreground="#9aa4b2").pack(anchor=tk.W, pady=(8, 0))

    def show_instructions(self) -> None:
        top = tk.Toplevel(self)
        top.title("Echo Alignment Workflow")
        top.geometry("680x560")
        text = tk.Text(top, wrap=tk.WORD, padx=12, pady=12, font=("TkDefaultFont", 10))
        text.insert("1.0", ALIGNMENT_INSTRUCTIONS)
        text.configure(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=8)

    def _send_control(self, payload: dict[str, object]) -> None:
        try:
            response = send_control_command(self.control_socket_var.get(), payload)
        except FileNotFoundError:
            self.control_status_var.set(f"Control: socket not found ({self.control_socket_var.get()}) — is the daemon running?")
            return
        except Exception as exc:  # noqa: BLE001
            self.control_status_var.set(f"Control: error — {exc}")
            return
        sent = ", ".join(f"{k}={v}" for k, v in payload.items())
        ok = response.get("ok")
        self.control_status_var.set(f"Control: {'sent' if ok else 'rejected'} ({sent}) -> {response}")

    def _apply_mic_delay(self) -> None:
        self._send_control({"mic_delay_ms": float(self.mic_delay_var.get())})

    def _nudge_mic_delay(self, delta: int) -> None:
        self.mic_delay_var.set(max(0, self.mic_delay_var.get() + delta))
        self._apply_mic_delay()

    def _apply_ref_delay(self) -> None:
        self._send_control(
            {"reference_delay_mode": self.delay_mode_var.get(), "reference_delay_ms": float(self.ref_delay_var.get())}
        )

    def _reset_echo_filter(self) -> None:
        self._send_control({"reset_echo_filter": True})

    def _update_alignment(self) -> None:
        try:
            self._refresh_alignment_metrics()
        finally:
            self.after(1000, self._update_alignment)

    def _refresh_alignment_metrics(self) -> None:
        sample_rate = 48_000
        window = sample_rate * 5  # judge on the last ~5 s
        mic = self._recent_samples("mic_raw", window)
        ref = self._recent_samples("system_reference", window)
        echo = self._recent_samples("after_echo", window)
        if mic is None or ref is None or len(mic) < sample_rate or len(ref) < sample_rate:
            self.align_verdict_var.set("Verdict: waiting for buffered audio...")
            self._set_verdict_color("#9aa4b2")
            return

        lag, mic_corr = best_lag_samples(mic, ref, sample_rate)
        self.align_echo_var.set(f"mic vs reference: {abs(mic_corr):.3f}  (echo strength)")
        lag_ms = lag * 1000.0 / sample_rate
        if lag >= 0:
            self.align_lag_var.set(f"lag: reference leads by {lag_ms:.0f} ms (causal)")
        else:
            self.align_lag_var.set(f"lag: reference LATE by {-lag_ms:.0f} ms (non-causal)")

        residual_corr = None
        rms_ratio = None
        if echo is not None and len(echo) >= sample_rate:
            _, residual_corr = best_lag_samples(echo, ref, sample_rate)
            mic_rms = float(np.sqrt(np.mean(mic * mic)) + 1e-12)
            echo_rms = float(np.sqrt(np.mean(echo * echo)) + 1e-12)
            rms_ratio = echo_rms / mic_rms
            self.align_residual_var.set(f"after_echo vs reference: {abs(residual_corr):.3f}  (lower = more removed)")
            self.align_diverge_var.set(f"after_echo / mic level: {db(rms_ratio):+.1f} dB  (want <= 0)")
            # Headline AEC effectiveness: how much echo energy was removed.
            removed_db = -db(rms_ratio)
            self.aec_reduction_var.set(
                f"echo removed (mic → after_echo): {removed_db:+.1f} dB   ·   residual vs ref {abs(residual_corr):.3f}"
            )
            if self.aec_reduction_label is not None:
                color = "#86efac" if removed_db >= 6.0 else ("#fbbf24" if removed_db >= 1.0 else "#fb7185")
                self.aec_reduction_label.configure(foreground=color)
        else:
            self.align_residual_var.set("after_echo vs reference: --")
            self.align_diverge_var.set("after_echo / mic level: --")

        self._set_verdict(abs(mic_corr), lag, None if residual_corr is None else abs(residual_corr), rms_ratio)

    def _set_verdict(self, mic_corr: float, lag: int, residual_corr: float | None, rms_ratio: float | None) -> None:
        if mic_corr < 0.05:
            self.align_verdict_var.set("Verdict: no measurable echo — play louder system audio to align.")
            self._set_verdict_color("#9aa4b2")
            return
        if lag < 0:
            need = int(-lag * 1000.0 / 48_000.0) + 20
            self.align_verdict_var.set(
                f"Verdict: reference is LATE — non-causal. Raise Mic Delay to ~{self.mic_delay_var.get() + need} ms and Reset."
            )
            self._set_verdict_color("#fb7185")
            return
        if rms_ratio is not None and rms_ratio > 1.05:
            self.align_verdict_var.set("Verdict: filter is DIVERGING (boosting level). Reset Echo Filter; check delay.")
            self._set_verdict_color("#fb7185")
            return
        if residual_corr is not None and residual_corr < 0.6 * mic_corr:
            self.align_verdict_var.set("Verdict: GOOD — echo is being cancelled (residual well below mic). Listen to after_echo.")
            self._set_verdict_color("#86efac")
            return
        self.align_verdict_var.set("Verdict: causal but little cancellation yet — press Reset Echo Filter and wait ~2 s.")
        self._set_verdict_color("#fbbf24")

    def _set_verdict_color(self, color: str) -> None:
        if self.align_verdict_label is not None:
            self.align_verdict_label.configure(foreground=color)

    def _recent_samples(self, name: str, count: int) -> np.ndarray | None:
        frames = list(self.buffers[name])
        if not frames:
            return None
        audio = np.concatenate(frames)
        return audio[-count:] if len(audio) > count else audio

    def connect(self) -> None:
        self.disconnect()
        self.client = MultiStreamSocketClient(self.socket_var.get(), self.audio_queue, self.status_queue)
        self.client.start()
        self.status_var.set("Connecting...")

    def disconnect(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None
        self.playback.stop()
        self.status_var.set("Disconnected")

    def choose_record_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self.record_dir.parent))
        if path:
            self.record_dir = Path(path)
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self.status_var.set(f"Save folder: {self.record_dir}")

    def clear_buffer(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()
        self.frames_seen = 0
        self.frames_var.set("Frames: 0")

    def save_streams(self) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        saved: list[Path] = []
        saved_audio: dict[str, np.ndarray] = {}
        for name in STREAM_NAMES:
            frames = list(self.buffers[name])
            if not frames:
                continue
            path = self.record_dir / f"{stamp}-{name}.wav"
            audio = np.concatenate(frames).astype(np.float32)
            sf.write(path, audio, 48_000, subtype="PCM_16")
            saved.append(path)
            saved_audio[name] = audio
        if not saved:
            messagebox.showwarning("No Streams Saved", "No stream frames are currently buffered.")
            return
        report_path = self.record_dir / f"{stamp}-alignment_report.json"
        report_path.write_text(json.dumps(alignment_report(saved_audio), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        saved.append(report_path)
        messagebox.showinfo("Streams Saved", "\n".join(str(path) for path in saved))

    def _toggle_playback(self) -> None:
        if self.play_var.get():
            self.playback.start()
        else:
            self.playback.stop()

    def _pump_audio(self) -> None:
        latest: MultiStreamPacket | None = None
        drained = 0
        while True:
            try:
                packet = self.audio_queue.get_nowait()
            except Empty:
                break
            latest = packet
            self.last_packet = packet
            drained += 1
            self.frames_seen += 1
            self.last_packet_time = time.monotonic()
            for name in STREAM_NAMES:
                self.buffers[name].append(packet.streams[name].copy())
            if self.play_var.get():
                stream_name = self.play_stream_var.get()
                if stream_name not in packet.streams:
                    stream_name = "cleaned_output"
                self.playback.write(packet.streams[stream_name])

        if latest is not None:
            cleaned = latest.metrics["cleaned_output"]
            self.level_var.set(min(1.0, cleaned["rms"] * 18.0))
            self.peak_var.set(min(1.0, cleaned["peak"]))
            self.frames_var.set(
                f"Frames: {self.frames_seen} (+{drained}), "
                f"reference present: {latest.reference_present}, "
                f"buffered: {len(self.buffers['cleaned_output']) / self.frame_rate:.1f}s"
            )
            for name in STREAM_NAMES:
                metrics = latest.metrics[name]
                self.level_label_by_stream[name].set(
                    f"RMS {db(metrics['rms']):.1f} dBFS, peak {db(max(metrics['peak'], 1e-12)):.1f} dBFS"
                )
            now = time.monotonic()
            if now - self._last_draw >= 0.15:
                self._last_draw = now
                for name in STREAM_NAMES:
                    self._draw_waveform(name)
        elif self.last_packet_time and time.monotonic() - self.last_packet_time > 1.5:
            self.level_var.set(0.0)
            self.peak_var.set(0.0)

        self.after(40, self._pump_audio)

    def _pump_status(self) -> None:
        while True:
            try:
                status = self.status_queue.get_nowait()
            except Empty:
                break
            self.status_var.set(status)
        self.after(250, self._pump_status)

    def _pump_diagnostics(self) -> None:
        try:
            payload = json.loads(DEFAULT_STATUS.read_text(encoding="utf-8"))
            mic = payload.get("mic", {})
            reference = payload.get("reference", {})
            output = payload.get("output", {})
            pipeline = payload.get("pipeline", {})
            config = payload.get("config", {})
            reader = config.get("reference_reader", {})
            stages = payload.get("stages", {})
            active = config.get("echo_canceller", "?")
            swap = config.get("echo_swap_status", "")
            smooth = config.get("dtln_mask_smoothing")
            smooth_txt = f", smooth={smooth}" if smooth is not None and active in ("dtln", "hybrid") else ""
            self.aec_active_var.set(f"active model: {active}{smooth_txt}" + (f"  ·  {swap}" if swap and "active" not in swap else ""))
            self.diagnostics_var.set(
                "Diagnostics: "
                f"mic {fmt_db(mic.get('rms_dbfs'))} peak {fmt_db(mic.get('peak_dbfs'))}, "
                f"ref {fmt_db(reference.get('rms_dbfs'))}, "
                f"out {fmt_db(output.get('rms_dbfs'))} peak {fmt_db(output.get('peak_dbfs'))}, "
                f"clip {fmt_pct(output.get('clipped_pct'))}\n"
                f"VAD {fmt_float(pipeline.get('vad_score'))}, "
                f"speech ratio {fmt_pct_ratio(pipeline.get('speech_ratio'))}, "
                f"ref delay {fmt_float(pipeline.get('reference_delay_ms'))} ms, "
                f"delay corr {fmt_float(pipeline.get('reference_delay_correlation'))}, "
                f"ref gain {fmt_float(pipeline.get('reference_gain'))}, "
                f"echo gain {fmt_float(pipeline.get('echo_gain'))}, "
                f"ref corr {fmt_float(payload.get('reference_correlation'))}, "
                f"residual ref corr {fmt_float(pipeline.get('residual_ref_correlation', payload.get('residual_ref_correlation')))}\n"
                f"AEC: backend={config.get('echo_canceller')} "
                f"taps={config.get('echo_filter_taps')} "
                f"step={config.get('echo_step_size')} "
                f"sync_latency={config.get('reference_sync_latency_frames')} "
                f"drift_comp={config.get('reference_drift_compensation')} "
                f"mic_delay_ms={config.get('mic_delay_ms')}\n"
                f"Stages: highpass={config.get('enable_highpass')} "
                f"ref_delay={config.get('enable_reference_delay_align')} "
                f"delay_mode={config.get('reference_delay_mode')} "
                f"ref_level={config.get('enable_reference_level_match')} "
                f"echo={config.get('enable_echo_cancellation')} "
                f"noise={config.get('enable_noise_suppression')} "
                f"vad={config.get('enable_vad')} "
                f"enhance={config.get('enable_speech_enhancement')}\n"
                f"Reference source: {config.get('resolved_reference_source')}\n"
                f"Reference reader: frames={reader.get('frames_read')} "
                f"age_ms={reader.get('last_read_age_ms')} "
                f"rc={reader.get('process_returncode')} "
                f"err={reader.get('last_error')}\n"
                f"Stage RMS: raw={fmt_db(stage_db(stages, 'mic_raw'))} "
                f"ref={fmt_db(stage_db(stages, 'reference'))} "
                f"aligned={fmt_db(stage_db(stages, 'reference_aligned'))} "
                f"matched={fmt_db(stage_db(stages, 'reference_matched'))} "
                f"hp={fmt_db(stage_db(stages, 'after_highpass'))} "
                f"echo={fmt_db(stage_db(stages, 'after_echo'))} "
                f"noise={fmt_db(stage_db(stages, 'after_noise'))} "
                f"final={fmt_db(stage_db(stages, 'output'))}"
            )
        except FileNotFoundError:
            self.diagnostics_var.set(f"Diagnostics: missing {DEFAULT_STATUS}")
        except Exception as exc:
            self.diagnostics_var.set(f"Diagnostics: {exc}")
        self.after(1000, self._pump_diagnostics)

    def _draw_waveform(self, name: str) -> None:
        canvas = self.canvas_by_stream[name]
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        mid = height / 2
        frames = list(self.buffers[name])[-160:]
        if not frames:
            return
        values = np.concatenate(frames)
        if len(values) > width:
            indexes = np.linspace(0, len(values) - 1, num=max(2, width // 2)).astype(int)
            values = values[indexes]
        if len(values) < 2:
            return
        step = width / (len(values) - 1)
        points: list[float] = []
        for index, value in enumerate(values):
            points.extend([index * step, mid - float(np.clip(value, -1.0, 1.0)) * mid * 0.9])
        colors = {
            "mic_raw": "#93c5fd",
            "system_reference": "#fbbf24",
            "reference_aligned": "#fb7185",
            "reference_matched": "#c084fc",
            "after_echo": "#86efac",
            "cleaned_output": "#5eead4",
        }
        color = colors.get(name, "#5eead4")
        canvas.create_line(*points, fill=color, width=2, smooth=True)
        canvas.create_line(0, mid, width, mid, fill="#30363d")

    def _on_close(self) -> None:
        self.disconnect()
        self.destroy()


def send_control_command(socket_path: str, payload: dict[str, object], timeout: float = 1.0) -> dict[str, object]:
    """Send one JSON-line tuning command to the daemon control socket and read the reply."""
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


def best_lag_samples(a: np.ndarray, b: np.ndarray, sample_rate: int, max_lag_ms: int = 120) -> tuple[int, float]:
    """Best integer lag (samples) and normalized correlation between a and b via FFT.

    Positive lag means ``a`` lags ``b`` (b is earlier). With a=mic_raw, b=reference,
    a positive lag means the reference leads the echo (causal, good); negative means
    the reference arrives after the echo (non-causal, needs more mic delay).
    """
    n = min(len(a), len(b))
    if n < 256:
        return 0, 0.0
    a = np.asarray(a[-n:], dtype=np.float64)
    b = np.asarray(b[-n:], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0, 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    xcorr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    max_lag = min(n - 1, int(sample_rate * max_lag_ms / 1000))
    corr = np.concatenate([xcorr[-max_lag:], xcorr[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    index = int(np.argmax(np.abs(corr)))
    return int(lags[index]), float(corr[index] / (na * nb))


def recv_exact(client: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = client.recv(length - len(data))
        if not chunk:
            raise OSError("socket closed")
        data += chunk
    return data


def decode_pcm16(encoded: str) -> np.ndarray:
    if not encoded:
        return np.zeros(960, dtype=np.float32)
    return np.frombuffer(base64.b64decode(encoded), dtype="<i2").astype(np.float32) / 32768.0


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(frame * frame)) + 1e-12),
        "peak": float(np.max(np.abs(frame))) if len(frame) else 0.0,
    }


def alignment_report(streams: dict[str, np.ndarray]) -> dict[str, object]:
    report: dict[str, object] = {
        "sample_rate": 48_000,
        "hop_ms": 20,
        "interpretation": "positive offset_ms means delay the named stream by that amount to better match mic_raw; negative means the named stream appears later than mic_raw",
        "offsets_vs_mic_raw": {},
    }
    mic = streams.get("mic_raw")
    if mic is None or len(mic) == 0:
        report["error"] = "mic_raw missing"
        return report
    mic_env = rms_envelope(mic)
    offsets: dict[str, object] = {}
    for name, audio in streams.items():
        if name == "mic_raw" or len(audio) == 0:
            continue
        lag_frames, corr = estimate_offset_frames(mic_env, rms_envelope(audio))
        sample_lag, sample_corr = estimate_offset_samples(mic, audio)
        offset_ms = lag_frames * 20
        offsets[name] = {
            "offset_ms": offset_ms,
            "correlation": corr,
            "sample_offset_ms": sample_lag * 1000.0 / 48_000.0,
            "sample_offset_samples": sample_lag,
            "sample_correlation": sample_corr,
            "boundary_jump_ratio": boundary_jump_ratio(audio),
            "suggested_reference_delay_ms": max(0, offset_ms) if name in ("system_reference", "reference_aligned", "reference_matched") else None,
            "suggested_mic_delay_ms": max(0, -sample_lag * 1000.0 / 48_000.0) if name in ("system_reference", "reference_aligned", "reference_matched") else None,
            "likely_output_delay_ms": max(0, -offset_ms) if name == "cleaned_output" else None,
        }
    report["offsets_vs_mic_raw"] = offsets
    ref = streams.get("system_reference")
    after_echo = streams.get("after_echo")
    if ref is not None and after_echo is not None and len(ref) == len(after_echo):
        warmup = min(len(ref), 48_000 * 4)
        ref_seg = ref[warmup:]
        mic_seg = mic[warmup:] if mic is not None and len(mic) > warmup else None
        echo_seg = after_echo[warmup:]
        mic_ref = abs(estimate_offset_samples(mic_seg, ref_seg)[1]) if mic_seg is not None else 0.0
        residual = abs(estimate_offset_samples(echo_seg, ref_seg)[1])
        echo_rms = float(np.sqrt(np.mean(echo_seg * echo_seg)) + 1e-12)
        mic_rms = float(np.sqrt(np.mean(mic_seg * mic_seg)) + 1e-12) if mic_seg is not None else echo_rms
        report["echo_metrics"] = {
            "mic_vs_reference_correlation": mic_ref,
            "after_echo_vs_reference_correlation": residual,
            "echo_reduction_db": 20.0 * float(np.log10(echo_rms / max(mic_rms, 1e-12))),
        }
    return report


def rms_envelope(audio: np.ndarray, hop_samples: int = 960) -> np.ndarray:
    usable = len(audio) - (len(audio) % hop_samples)
    if usable <= 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[:usable].reshape(-1, hop_samples)
    env = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)
    return env - float(np.mean(env))


def estimate_offset_frames(reference_env: np.ndarray, candidate_env: np.ndarray) -> tuple[int, float]:
    count = min(len(reference_env), len(candidate_env))
    if count < 3:
        return 0, 0.0
    a = reference_env[:count]
    b = candidate_env[:count]
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-count + 1, count)
    index = int(np.argmax(np.abs(corr)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return int(lags[index]), float(corr[index] / denom)


def estimate_offset_samples(reference_audio: np.ndarray, candidate_audio: np.ndarray, max_lag_ms: int = 100) -> tuple[int, float]:
    count = min(len(reference_audio), len(candidate_audio))
    if count < 32:
        return 0, 0.0
    a = np.asarray(reference_audio[:count], dtype=np.float32)
    b = np.asarray(candidate_audio[:count], dtype=np.float32)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    max_lag = min(count - 1, int(48_000 * max_lag_ms / 1000))
    best_lag = 0
    best_corr = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: len(aa)]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-8
        corr = float(np.dot(aa, bb) / denom)
        if abs(corr) > abs(best_corr):
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr


def boundary_jump_ratio(audio: np.ndarray, frame_samples: int = 960) -> float:
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < frame_samples * 2:
        return 0.0
    boundary: list[float] = []
    inside: list[float] = []
    for index in range(frame_samples, len(x) - 1, frame_samples):
        boundary.append(abs(float(x[index] - x[index - 1])))
        center = index - frame_samples // 2
        inside.append(abs(float(x[center] - x[center - 1])))
    if not boundary or not inside:
        return 0.0
    return float(np.mean(boundary) / (np.mean(inside) + 1e-12))


def db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-12)))


def fmt_db(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} dBFS"


def fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def fmt_pct_ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def stage_db(stages: dict[str, object], name: str) -> object:
    stage = stages.get(name, {})
    if isinstance(stage, dict):
        return stage.get("rms_dbfs")
    return None


def main() -> int:
    app = CleanSpeechTestbed()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
