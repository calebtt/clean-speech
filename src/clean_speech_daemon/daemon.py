from __future__ import annotations

from contextlib import ExitStack
from queue import Empty, Queue
import signal
import time

from .control import ControlServer
from .diagnostics import DiagnosticsLogger, StageWavWriter, audio_metrics, correlation
from .audio import InputStreamReader, PulseMonitorReader
from .config import Config
from .outputs import DebugWavOutput, FifoPcmOutput, MultiStreamSocketOutput, PulsePipeSource, SocketPcmOutput
from .processing import ProcessingPipeline
from .sync import DriftCompensatingReference


class CleanSpeechDaemon:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stop_requested = False
        self.pipeline = ProcessingPipeline(config)
        self.socket_output = SocketPcmOutput(
            config.output.socket_path,
            config.input.sample_rate,
            config.input.channels,
        )
        self.streams_output = MultiStreamSocketOutput(
            config.output.streams_socket_path,
            config.input.sample_rate,
            int(config.input.sample_rate * config.input.frame_ms / 1000),
        )
        self.fifo_output = FifoPcmOutput(config.output.fifo_path)
        self.wav_output = DebugWavOutput(config.output.debug_wav_path, config.input.sample_rate)
        self.pulse_source = PulsePipeSource(
            config.output.pulse_source_name,
            config.output.pulse_source_description,
            config.output.fifo_path,
            config.input.sample_rate,
            config.input.channels,
        )
        self.diagnostics = DiagnosticsLogger(
            config.diagnostics.log_path,
            config.diagnostics.status_path,
            int(config.diagnostics.interval_ms),
            int(config.diagnostics.print_interval_ms),
        )
        self.stage_wavs = StageWavWriter(config.diagnostics.stage_wav_dir, config.input.sample_rate)
        frame_samples = int(config.input.sample_rate * config.input.frame_ms / 1000)
        self.reference_source = "off"
        self.reference_reader = None
        # Lock the monitor reference to the mic clock at constant latency, instead
        # of pairing whatever frame is in the queue and reusing it on underrun.
        self.ref_sync = DriftCompensatingReference(
            frame_samples,
            target_latency_frames=float(config.processing.reference_sync_latency_frames),
            drift_compensation=bool(config.processing.reference_drift_compensation),
        )
        self._control_queue: Queue[dict[str, object]] = Queue()
        self._control_server = ControlServer(config.output.control_socket_path, self._control_queue)

    def request_stop(self, *_args) -> None:  # noqa: ANN002
        self.stop_requested = True

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        self.socket_output.start()
        self.streams_output.start()
        self.fifo_output.start()
        self.wav_output.start()
        if self.config.diagnostics.enabled and self.config.diagnostics.enable_stage_wavs:
            self.stage_wavs.start()
        if self.config.output.pipewire_virtual_source and self.config.output.auto_load_pulse_pipe_source:
            self.pulse_source.start()

        print(f"socket output: {self.config.output.socket_path}", flush=True)
        print(f"multi-stream socket output: {self.config.output.streams_socket_path}", flush=True)
        print(f"fifo output: {self.config.output.fifo_path}", flush=True)
        print(f"control socket: {self.config.output.control_socket_path}", flush=True)
        print("starting microphone processing", flush=True)

        self._control_server.start()
        try:
            self._run_audio_loop()
        finally:
            self._control_server.stop()
            self.pulse_source.stop()
            self.socket_output.stop()
            self.streams_output.stop()
            self.fifo_output.stop()
            self.wav_output.stop()
            self.stage_wavs.stop()
            self.diagnostics.stop()

    def _run_audio_loop(self) -> None:
        ref_selector = self.config.input.system_audio_reference
        use_ref = ref_selector not in ("", "off", "none", "false")
        last_stats = time.monotonic()

        with ExitStack() as stack:
            mic = stack.enter_context(
                InputStreamReader(
                    self.config.input.microphone,
                    self.config.input.sample_rate,
                    self.config.input.channels,
                    self.config.input.frame_ms,
                )
            )
            reference = None
            if use_ref:
                if ref_selector in ("auto", "default", "running") or ".monitor" in ref_selector or not ref_selector.isdigit():
                    reference = stack.enter_context(
                        PulseMonitorReader(
                            ref_selector,
                            self.config.input.sample_rate,
                            self.config.input.frame_ms,
                        )
                    )
                    self.reference_reader = reference
                    self.reference_source = reference.resolved_source
                else:
                    reference = stack.enter_context(
                        InputStreamReader(
                            ref_selector,
                            self.config.input.sample_rate,
                            self.config.input.channels,
                            self.config.input.frame_ms,
                        )
                    )

            while not self.stop_requested:
                self._poll_control_commands()
                mic_frame = mic.read(timeout=0.5)
                ref_data = None
                if reference is not None:
                    # Drain everything the monitor produced since the last mic frame
                    # into the drift-compensating buffer, then pull exactly one
                    # mic-clock-aligned frame back out (None while priming/underrun).
                    while True:
                        ref_frame = reference.read_next_or_none()
                        if ref_frame is None:
                            break
                        self.ref_sync.push(ref_frame.data)
                    ref_data = self.ref_sync.pull()

                cleaned = self.pipeline.process(mic_frame.data, ref_data)
                stages = self.pipeline.last_stages
                self.socket_output.write(cleaned)
                self.streams_output.write(
                    stages.get("mic_raw", mic_frame.data),
                    ref_data,
                    stages.get("reference_aligned"),
                    stages.get("reference_matched"),
                    stages.get("after_echo"),
                    cleaned,
                )
                self.fifo_output.write(cleaned)
                self.wav_output.write(cleaned)
                if self.config.diagnostics.enabled and self.config.diagnostics.enable_stage_wavs:
                    self.stage_wavs.write(self.pipeline.last_stages)

                if self.config.diagnostics.enabled:
                    self._write_diagnostics(stages, ref_data, cleaned)

                now = time.monotonic()
                if now - last_stats >= 5.0:
                    stats = self.pipeline.stats
                    print(
                        f"frames={stats.frames} speech_frames={stats.speech_frames} "
                        f"vad={stats.vad_score:.2f} noise_floor={stats.noise_floor:.5f} "
                        f"echo_gain={stats.echo_gain:.4f} ref={stats.ref_present}",
                        flush=True,
                    )
                    last_stats = now

    def _poll_control_commands(self) -> None:
        while True:
            try:
                command = self._control_queue.get_nowait()
            except Empty:
                break
            self._apply_control_command(command)

    def _apply_control_command(self, command: dict[str, object]) -> None:
        tuning = self.pipeline.apply_tuning(
            reference_delay_ms=_as_optional_float(command.get("reference_delay_ms")),
            reference_delay_mode=_as_optional_str(command.get("reference_delay_mode")),
            mic_delay_ms=_as_optional_float(command.get("mic_delay_ms")),
        )
        sync_frames = _as_optional_float(command.get("reference_sync_latency_frames"))
        if sync_frames is not None:
            self.ref_sync.set_target_latency_frames(sync_frames)
            self.config.processing.reference_sync_latency_frames = sync_frames
            tuning["reference_sync_latency_frames"] = sync_frames
        if bool(command.get("reset_echo_filter")):
            self.pipeline.reset_echo_filter()
            tuning["reset_echo_filter"] = True

    def _write_diagnostics(self, stages, reference, cleaned) -> None:  # noqa: ANN001
        stats = self.pipeline.stats
        mic = stages.get("hardware_mic")
        mic_to_aec = stages.get("mic_raw")
        after_echo = stages.get("after_echo")
        self.diagnostics.maybe_write(
            {
                "config": {
                    "microphone": self.config.input.microphone,
                    "system_audio_reference": self.config.input.system_audio_reference,
                    "resolved_reference_source": self.reference_source,
                    "reference_reader": self._reference_reader_status(),
                    "sample_rate": self.config.input.sample_rate,
                    "frame_ms": self.config.input.frame_ms,
                    "enable_highpass": self.config.processing.enable_highpass,
                    "enable_reference_delay_align": self.config.processing.enable_reference_delay_align,
                    "reference_delay_mode": self.config.processing.reference_delay_mode,
                    "reference_delay_ms": self.config.processing.reference_delay_ms,
                    "reference_max_delay_ms": self.config.processing.reference_max_delay_ms,
                    "reference_delay_smoothing": self.config.processing.reference_delay_smoothing,
                    "delay_window_ms": self.config.processing.delay_window_ms,
                    "delay_update_min_ref_rms": self.config.processing.delay_update_min_ref_rms,
                    "delay_min_confidence": self.config.processing.delay_min_confidence,
                    "delay_median_frames": self.config.processing.delay_median_frames,
                    "delay_calibrate_seconds": self.config.processing.delay_calibrate_seconds,
                    "delay_cancellation_aware": self.config.processing.delay_cancellation_aware,
                    "delay_fine_tune_ms": self.config.processing.delay_fine_tune_ms,
                    "delay_target_residual_corr": self.config.processing.delay_target_residual_corr,
                    "enable_reference_level_match": self.config.processing.enable_reference_level_match,
                    "enable_echo_cancellation": self.config.processing.enable_echo_cancellation,
                    "echo_canceller": self.config.processing.echo_canceller,
                    "echo_filter_taps": self.config.processing.echo_filter_taps,
                    "echo_step_size": self.config.processing.echo_step_size,
                    "echo_filter_leak": self.config.processing.echo_filter_leak,
                    "echo_boundary_smoothing_samples": self.config.processing.echo_boundary_smoothing_samples,
                    "reference_sync_latency_frames": self.config.processing.reference_sync_latency_frames,
                    "reference_drift_compensation": self.config.processing.reference_drift_compensation,
                    "mic_delay_ms": self.config.processing.mic_delay_ms,
                    "enable_noise_suppression": self.config.processing.enable_noise_suppression,
                    "enable_vad": self.config.processing.enable_vad,
                    "enable_speech_enhancement": self.config.processing.enable_speech_enhancement,
                    "noise_reduction": self.config.processing.noise_reduction,
                    "spectral_floor": self.config.processing.spectral_floor,
                    "reference_gain_min": self.config.processing.reference_gain_min,
                    "reference_gain_max": self.config.processing.reference_gain_max,
                    "reference_gain_smoothing": self.config.processing.reference_gain_smoothing,
                    "reference_target_ratio": self.config.processing.reference_target_ratio,
                },
                "mic": audio_metrics(mic),
                "mic_to_aec": audio_metrics(mic_to_aec),
                "reference": audio_metrics(reference),
                "output": audio_metrics(cleaned),
                "stages": {name: audio_metrics(frame) for name, frame in stages.items()},
                "reference_correlation": correlation(mic_to_aec, reference),
                "residual_ref_correlation": correlation(after_echo, reference),
                "pipeline": {
                    "frames": stats.frames,
                    "speech_frames": stats.speech_frames,
                    "speech_ratio": stats.speech_ratio,
                    "vad_score": stats.vad_score,
                    "pre_vad_score": stats.pre_vad_score,
                    "noise_floor": stats.noise_floor,
                    "echo_gain": stats.echo_gain,
                    "reference_gain": stats.reference_gain,
                    "reference_delay_ms": stats.reference_delay_ms,
                    "reference_delay_correlation": stats.reference_delay_correlation,
                    "reference_delay_confidence": stats.reference_delay_confidence,
                    "reference_cancellation_score": stats.reference_cancellation_score,
                    "residual_ref_correlation": stats.residual_ref_correlation,
                    "ref_present": stats.ref_present,
                    "clipped_output_pct": stats.clipped_output_pct,
                },
            }
        )

    def _reference_reader_status(self) -> dict[str, object]:
        sync = {
            "frames_emitted": self.ref_sync.samples_pulled // self.ref_sync.frame_samples,
            "latency_frames": round(self.ref_sync.latency_frames, 2),
            "resample_ratio": round(self.ref_sync.ratio, 5),
            "underruns": self.ref_sync.underruns,
            "overflow_drops": self.ref_sync.overflow_drops,
            "resyncs": self.ref_sync.resyncs,
        }
        reader = self.reference_reader
        if reader is None:
            return {"type": "none", "sync": sync}
        process = reader.process
        return {
            "type": "parec",
            "resolved_source": reader.resolved_source,
            "frames_read": reader.frames_read,
            "bytes_read": reader.bytes_read,
            "frames_dropped": getattr(reader, "frames_dropped", 0),
            "last_read_age_ms": None if reader.last_read_time <= 0 else int((time.monotonic() - reader.last_read_time) * 1000),
            "process_returncode": None if process is None else process.poll(),
            "last_error": reader.last_error,
            "sync": sync,
        }


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
