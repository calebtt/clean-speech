"""Regression tests for live AEC pipeline fixes."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from clean_speech_daemon.config import Config
from clean_speech_daemon.processing import ProcessingPipeline, normalized_correlation


SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
RECORDINGS = Path.home() / "clean-speech-recordings"


def frames(signal: np.ndarray) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % FRAME_SAMPLES)
    return [frame.astype(np.float32) for frame in signal[:usable].reshape(-1, FRAME_SAMPLES)]


def best_lag_correlation(a: np.ndarray, b: np.ndarray, step: int = 48, max_lag_ms: int = 100) -> float:
    n = min(len(a), len(b))
    if n < FRAME_SAMPLES * 4:
        return 0.0
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    max_lag = min(n - 1, int(SAMPLE_RATE * max_lag_ms / 1000))
    best = 0.0
    for lag in range(-max_lag, max_lag + 1, step):
        if lag < 0:
            aa, bb = a[-lag:], b[: len(a) + lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 1e-9:
            continue
        best = max(best, abs(float(np.dot(aa, bb) / denom)))
    return best


def nlms_config() -> Config:
    config = Config()
    config.input.sample_rate = SAMPLE_RATE
    config.input.frame_ms = FRAME_MS
    config.processing.enable_highpass = True
    config.processing.enable_reference_delay_align = True
    config.processing.reference_delay_mode = "manual"
    config.processing.reference_delay_ms = 25
    config.processing.enable_reference_level_match = False
    config.processing.enable_echo_cancellation = True
    config.processing.echo_canceller = "nlms"
    config.processing.echo_filter_taps = 4096
    config.processing.echo_step_size = 0.1
    config.processing.echo_step_size_warmup = 0.3
    config.processing.mic_delay_ms = 0
    config.processing.enable_noise_suppression = False
    config.processing.enable_speech_enhancement = False
    config.processing.enable_vad = False
    return config


class AecPipelineFixTests(unittest.TestCase):
    def test_highpass_runs_after_echo_not_before(self) -> None:
        config = nlms_config()
        pipeline = ProcessingPipeline(config)
        t = np.arange(FRAME_SAMPLES, dtype=np.float32) / SAMPLE_RATE
        reference = (0.08 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        mic = (reference + 0.02 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)

        pipeline.process(mic, reference)
        stages = pipeline.last_stages

        self.assertIn("after_echo", stages)
        self.assertIn("after_highpass", stages)
        self.assertGreater(
            float(np.max(np.abs(stages["after_highpass"] - stages["after_echo"]))),
            1e-6,
            "highpass should modify the signal after echo cancellation",
        )

    def test_mic_raw_equals_hardware_mic_without_mic_delay(self) -> None:
        config = nlms_config()
        config.processing.mic_delay_ms = 0
        pipeline = ProcessingPipeline(config)
        frame = np.linspace(-0.05, 0.05, FRAME_SAMPLES, dtype=np.float32)
        pipeline.process(frame, None)
        stages = pipeline.last_stages
        np.testing.assert_allclose(stages["mic_raw"], stages["hardware_mic"], atol=1e-6)

    def test_saved_fixture_echo_is_reduced_with_fixed_pipeline(self) -> None:
        stamp = "20260621-081654"
        mic_path = RECORDINGS / f"{stamp}-mic_raw.wav"
        ref_path = RECORDINGS / f"{stamp}-system_reference.wav"
        if not mic_path.exists() or not ref_path.exists():
            self.skipTest("saved fixture recordings are not available")

        mic, _ = sf.read(mic_path, dtype="float32")
        reference, _ = sf.read(ref_path, dtype="float32")
        config = nlms_config()
        pipeline = ProcessingPipeline(config)
        after_echo_frames = []
        for mic_frame, ref_frame in zip(frames(mic), frames(reference)):
            pipeline.process(mic_frame, ref_frame)
            after_echo_frames.append(pipeline.last_stages["after_echo"])
        after_echo = np.concatenate(after_echo_frames)

        warmup = FRAME_SAMPLES * 80
        mic_seg = mic[warmup:]
        ref_seg = reference[warmup:]
        echo_seg = after_echo[warmup:]
        mic_best = best_lag_correlation(mic_seg, ref_seg)
        residual_best = best_lag_correlation(echo_seg, ref_seg)
        self.assertLess(
            residual_best,
            mic_best * 0.85,
            f"lagged residual corr {residual_best:.3f} vs mic {mic_best:.3f}",
        )


if __name__ == "__main__":
    unittest.main()