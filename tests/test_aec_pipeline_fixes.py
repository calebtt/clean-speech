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


def nlms_config(*, auto_delay: bool = False) -> Config:
    config = Config()
    config.input.sample_rate = SAMPLE_RATE
    config.input.frame_ms = FRAME_MS
    config.processing.enable_highpass = True
    config.processing.enable_reference_delay_align = True
    config.processing.reference_delay_mode = "auto" if auto_delay else "manual"
    config.processing.reference_delay_ms = 25
    if auto_delay:
        config.processing.delay_window_ms = 300
        config.processing.delay_update_min_ref_rms = 0.002
        config.processing.delay_min_confidence = 0.08
        config.processing.delay_median_frames = 10
        config.processing.reference_delay_smoothing = 0.5
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

    def test_recording_cancels_most_of_the_system_audio(self) -> None:
        """An echo-dominant capture (mostly system audio, near-silent room) must
        have most of that energy removed by the SHIPPED echo config.

        This guards the shipped ``echo_step_size`` / ``echo_filter_taps`` defaults:
        a step too low (e.g. 0.05) under-adapts and leaves the echo in. The ceiling
        is ~6 dB here because this USB webcam mic adds non-linear distortion (onboard
        AGC) the linear NLMS cannot model -- it already beats the ~4 dB linear-FIR
        ceiling, so the test targets the achievable, not the ideal.
        """
        stamp = "20260622-082629"
        mic_path = RECORDINGS / f"{stamp}-mic_raw.wav"
        ref_path = RECORDINGS / f"{stamp}-system_reference.wav"
        if not mic_path.exists() or not ref_path.exists():
            self.skipTest("echo-dominant fixture recording not available")

        mic, _ = sf.read(mic_path, dtype="float32")
        reference, _ = sf.read(ref_path, dtype="float32")

        # Build from shipped defaults so this guards echo_step_size/taps/canceller.
        config = Config()
        config.input.sample_rate = SAMPLE_RATE
        config.input.frame_ms = FRAME_MS
        # The saved mic_raw is already post-mic-delay; isolate the echo stage.
        config.processing.mic_delay_ms = 0
        config.processing.reference_delay_mode = "manual"
        config.processing.reference_delay_ms = 0
        config.processing.enable_reference_delay_align = True
        config.processing.enable_reference_level_match = False
        config.processing.enable_highpass = False
        config.processing.enable_noise_suppression = False
        config.processing.enable_speech_enhancement = False
        config.processing.enable_vad = False

        pipeline = ProcessingPipeline(config)
        after_echo_frames = []
        for mic_frame, ref_frame in zip(frames(mic), frames(reference)):
            pipeline.process(mic_frame, ref_frame)
            after_echo_frames.append(pipeline.last_stages["after_echo"])
        after_echo = np.concatenate(after_echo_frames)

        warmup = FRAME_SAMPLES * 100
        end = len(after_echo)
        mic_seg = mic[warmup:end]
        echo_seg = after_echo[warmup:end]
        ref_seg = reference[warmup:end]

        def rms(signal: np.ndarray) -> float:
            return float(np.sqrt(np.mean(signal * signal)) + 1e-12)

        reduction_db = 20.0 * float(np.log10(rms(mic_seg) / rms(echo_seg)))
        mic_corr = best_lag_correlation(mic_seg, ref_seg)
        residual_corr = best_lag_correlation(echo_seg, ref_seg)

        # Most of the energy gone (>= ~5 dB == >2/3 of the energy).
        self.assertGreaterEqual(reduction_db, 5.0, f"only {reduction_db:.1f} dB of system audio removed")
        # The reference-correlated echo is essentially gone.
        self.assertLess(residual_corr, 0.12, f"residual echo corr {residual_corr:.3f} still high")
        self.assertLess(residual_corr, 0.4 * mic_corr, f"residual {residual_corr:.3f} vs mic {mic_corr:.3f}")

    def test_saved_fixture_auto_delay_locks_near_measured_lag(self) -> None:
        stamp = "20260621-090130"
        mic_path = RECORDINGS / f"{stamp}-mic_raw.wav"
        ref_path = RECORDINGS / f"{stamp}-system_reference.wav"
        if not mic_path.exists() or not ref_path.exists():
            self.skipTest("saved fixture recordings are not available")

        mic, _ = sf.read(mic_path, dtype="float32")
        reference, _ = sf.read(ref_path, dtype="float32")

        from clean_speech_daemon.delay_align import median_frame_lag_estimate

        expected_lag, expected_corr, _ = median_frame_lag_estimate(
            mic,
            reference,
            SAMPLE_RATE,
            FRAME_SAMPLES,
        )
        self.assertGreater(expected_corr, 0.35, "fixture should have measurable ref-active echo")

        auto = nlms_config(auto_delay=True)
        auto_pipe = ProcessingPipeline(auto)
        delay_track: list[float] = []
        for mic_frame, ref_frame in zip(frames(mic), frames(reference)):
            auto_pipe.process(mic_frame, ref_frame)
            delay_track.append(auto_pipe.stats.reference_delay_ms)

        settled = delay_track[-max(20, len(delay_track) // 4) :]
        settled_median = float(np.median(settled))
        expected_ms = expected_lag * 1000.0 / SAMPLE_RATE
        self.assertAlmostEqual(settled_median, expected_ms, delta=5.0)
        self.assertGreater(auto_pipe.stats.reference_delay_confidence, 0.05)
        self.assertNotAlmostEqual(settled_median, 25.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()