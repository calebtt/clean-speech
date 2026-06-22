from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "clean-speech-daemon" / "config.toml"


@dataclass(slots=True)
class InputConfig:
    microphone: str = "auto"
    system_audio_reference: str = "auto"
    sample_rate: int = 48_000
    channels: int = 1
    frame_ms: int = 20


@dataclass(slots=True)
class ProcessingConfig:
    mode: str = "quality"
    max_latency_ms: int = 1500
    enable_highpass: bool = True
    enable_reference_delay_align: bool = True
    # Manual is the validated default: the auto GCC-PHAT aligner wanders frame to
    # frame and resets the NLMS filter on every change (see ProcessingPipeline),
    # which prevents convergence. With a generous mic_delay (below) the reference
    # leads the echo and the adaptive filter's taps absorb the residual delay, so a
    # fixed delay of 0 cancels across sessions. Re-enable auto only once the aligner
    # no longer resets/diverges.
    reference_delay_mode: str = "manual"
    reference_delay_ms: int = 0
    # Clamp the auto search band near the real echo delay (~19 ms measured) so it
    # cannot lock onto spurious 100-400 ms peaks. Only used in auto mode.
    reference_max_delay_ms: int = 40
    reference_delay_smoothing: float = 0.85
    delay_window_ms: float = 300.0
    delay_update_min_ref_rms: float = 0.003
    delay_min_confidence: float = 0.15
    delay_median_frames: int = 15
    delay_calibrate_seconds: float = 10.0
    delay_cancellation_aware: bool = True
    delay_fine_tune_ms: float = 15.0
    delay_target_residual_corr: float = 0.15
    enable_reference_level_match: bool = False
    enable_echo_cancellation: bool = True
    # "scalar" (legacy) | "nlms" (adaptive FIR) | neural 16 kHz models:
    #   "dtln"   deep non-linear cancellation, can warble on speech
    #   "nkf"    linear neural Kalman, artifact-free but shallow
    #   "hybrid" NKF->DTLN, deep cancellation with the voice kept clear (recommended)
    echo_canceller: str = "nlms"
    dtln_mask_smoothing: float = 0.6  # neural DTLN temporal mask smoothing (less warble)
    echo_filter_taps: int = 4096  # ~85 ms at 48 kHz; long enough for the room tail
    # 0.3 adapts fast enough to remove most of an echo-dominant capture (validated:
    # ~6 dB / 77% of the energy on a real recording, near the ceiling set by the
    # mic's non-linear distortion). Lower values (0.05) under-adapt and leave the
    # echo in. The cost is that a high step also subtracts near-end speech during
    # double-talk -- the proper fix for that is double-talk detection (freeze
    # adaptation while the user speaks), not a permanently low step.
    echo_step_size: float = 0.3
    echo_filter_leak: float = 1e-4
    echo_boundary_smoothing_samples: int = 64
    echo_step_size_warmup: float = 0.3
    echo_warmup_frames: int = 150
    # Reference jitter-buffer latency. Must be SMALLER than the acoustic echo delay
    # (monitor tap -> speaker -> mic), otherwise the reference arrives later than
    # the echo and a causal filter cannot cancel it. The adaptive filter then
    # delays the reference the rest of the way, so keep this small.
    reference_sync_latency_frames: float = 1.5
    # Resample the reference to a separate capture clock. Off by default: for
    # monitor-based AEC on one audio graph (PipeWire/Pulse) the mic and monitor are
    # synchronous, and resampling only warps the reference and breaks cancellation.
    reference_drift_compensation: bool = False
    # Delay the mic into the canceller so the (jitter-buffered) reference reliably
    # leads the echo, keeping the adaptive filter causal. This is the primary AEC
    # fix: measured sessions showed the parec monitor reference arriving 16-29 ms
    # AFTER the echo in the mic (capture latency exceeds the acoustic delay), making
    # cancellation non-causal -- a causal FIR cannot subtract a sound it has not yet
    # received. 60 ms covers the observed lateness plus margin; the NLMS taps then
    # model the residual lead. Adds this much output latency.
    mic_delay_ms: int = 60
    enable_noise_suppression: bool = True
    enable_vad: bool = True
    enable_speech_enhancement: bool = True
    noise_reduction: float = 1.65
    spectral_floor: float = 0.035
    reference_gain_min: float = 0.05
    reference_gain_max: float = 20.0
    reference_gain_smoothing: float = 0.92
    reference_target_ratio: float = 1.0
    output_when_no_speech: str = "silence"
    silence_attenuation: float = 0.02


@dataclass(slots=True)
class VadConfig:
    threshold: float = 0.52
    pre_roll_ms: int = 300
    post_roll_ms: int = 600
    noise_learn_ms: int = 800


@dataclass(slots=True)
class OutputConfig:
    pipewire_virtual_source: bool = True
    auto_load_pulse_pipe_source: bool = True
    pulse_source_name: str = "clean_speech_microphone"
    pulse_source_description: str = "Clean Speech Microphone"
    fifo_path: str = "/tmp/clean-speech-daemon.pcm"
    socket_path: str = "/tmp/clean-speech-daemon.sock"
    streams_socket_path: str = "/tmp/clean-speech-daemon-streams.sock"
    control_socket_path: str = "/tmp/clean-speech-daemon-control.sock"
    debug_wav_path: str = ""


@dataclass(slots=True)
class DiagnosticsConfig:
    enabled: bool = True
    log_path: str = "/tmp/clean-speech-daemon-diagnostics.jsonl"
    status_path: str = "/tmp/clean-speech-daemon-status.json"
    interval_ms: int = 1000
    enable_stage_wavs: bool = False
    stage_wav_dir: str = "/tmp/clean-speech-daemon-stages"
    print_interval_ms: int = 5000


@dataclass(slots=True)
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


def _merge_dataclass(instance: object, values: dict[str, object]) -> None:
    field_names = getattr(instance, "__dataclass_fields__", {})
    for key, value in values.items():
        if key in field_names:
            setattr(instance, key, value)


def load_config(path: Path | None = None) -> Config:
    config = Config()
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return config

    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    if isinstance(raw.get("input"), dict):
        _merge_dataclass(config.input, raw["input"])
    if isinstance(raw.get("processing"), dict):
        _merge_dataclass(config.processing, raw["processing"])
    if isinstance(raw.get("vad"), dict):
        _merge_dataclass(config.vad, raw["vad"])
    if isinstance(raw.get("output"), dict):
        _merge_dataclass(config.output, raw["output"])
    if isinstance(raw.get("diagnostics"), dict):
        _merge_dataclass(config.diagnostics, raw["diagnostics"])

    return config


def default_config_text() -> str:
    return """[input]
microphone = "auto"
system_audio_reference = "auto"
sample_rate = 48000
channels = 1
frame_ms = 20

[processing]
mode = "quality"
max_latency_ms = 1500
enable_highpass = true
enable_reference_delay_align = true
reference_delay_mode = "manual"
reference_delay_ms = 0
reference_max_delay_ms = 40
reference_delay_smoothing = 0.85
delay_window_ms = 300
delay_update_min_ref_rms = 0.003
delay_min_confidence = 0.15
delay_median_frames = 15
delay_calibrate_seconds = 10
delay_cancellation_aware = true
delay_fine_tune_ms = 15
delay_target_residual_corr = 0.15
enable_reference_level_match = false
enable_echo_cancellation = true
echo_canceller = "nlms"
echo_filter_taps = 4096
echo_step_size = 0.3
echo_filter_leak = 0.0001
echo_boundary_smoothing_samples = 64
echo_step_size_warmup = 0.3
echo_warmup_frames = 150
reference_sync_latency_frames = 1.0
reference_drift_compensation = false
mic_delay_ms = 60
enable_noise_suppression = true
enable_vad = true
enable_speech_enhancement = true
noise_reduction = 1.65
spectral_floor = 0.035
reference_gain_min = 0.05
reference_gain_max = 20.0
reference_gain_smoothing = 0.92
reference_target_ratio = 1.0
output_when_no_speech = "silence"
silence_attenuation = 0.02

[vad]
threshold = 0.52
pre_roll_ms = 300
post_roll_ms = 600
noise_learn_ms = 800

[output]
pipewire_virtual_source = true
auto_load_pulse_pipe_source = true
pulse_source_name = "clean_speech_microphone"
pulse_source_description = "Clean Speech Microphone"
fifo_path = "/tmp/clean-speech-daemon.pcm"
socket_path = "/tmp/clean-speech-daemon.sock"
streams_socket_path = "/tmp/clean-speech-daemon-streams.sock"
control_socket_path = "/tmp/clean-speech-daemon-control.sock"
debug_wav_path = ""

[diagnostics]
enabled = true
log_path = "/tmp/clean-speech-daemon-diagnostics.jsonl"
status_path = "/tmp/clean-speech-daemon-status.json"
interval_ms = 1000
enable_stage_wavs = false
stage_wav_dir = "/tmp/clean-speech-daemon-stages"
print_interval_ms = 5000
"""
