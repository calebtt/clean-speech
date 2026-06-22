"""Tests for GCC-PHAT delay estimation."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import lfilter

from clean_speech_daemon.delay_align import (
    GccPhatDelayEstimator,
    cancellation_aware_fine_tune,
    cancellation_residual_score,
    gcc_phat_estimate_delay,
    read_delayed_frame,
)
from clean_speech_daemon.processing import ProcessingPipeline, ReferenceDelayAligner


SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def broadband_signal(samples: int, seed: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = rng.randn(samples)
    x = lfilter([1.0], [1.0, -0.85], x)
    return (x / (np.std(x) + 1e-9) * 0.08).astype(np.float32)


def delayed(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    out = np.zeros_like(signal)
    if delay_samples <= 0:
        return signal.copy()
    out[delay_samples:] = signal[:-delay_samples]
    return out


def frames(signal: np.ndarray) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % FRAME_SAMPLES)
    return [frame.astype(np.float32) for frame in signal[:usable].reshape(-1, FRAME_SAMPLES)]


class GccPhatDelayTests(unittest.TestCase):
    def test_gcc_phat_finds_known_delay(self) -> None:
        delay_samples = int(SAMPLE_RATE * 0.018)
        total = SAMPLE_RATE * 2
        ref = broadband_signal(total)
        mic = 0.65 * delayed(ref, delay_samples) + 0.01 * broadband_signal(total, seed=9)

        lag, peak, conf = gcc_phat_estimate_delay(mic, ref, SAMPLE_RATE, max_lag_ms=80.0)
        self.assertGreater(peak, 0.1)
        self.assertGreater(conf, 0.0)
        self.assertAlmostEqual(lag, delay_samples, delta=SAMPLE_RATE * 0.002)

    def test_estimator_tracks_delay_over_frames(self) -> None:
        delay_samples = int(SAMPLE_RATE * 0.022)
        total = FRAME_SAMPLES * 120
        ref = broadband_signal(total)
        mic = 0.70 * delayed(ref, delay_samples)

        estimator = GccPhatDelayEstimator(
            SAMPLE_RATE,
            FRAME_SAMPLES,
            max_delay_ms=120.0,
            window_ms=250.0,
            min_ref_rms=0.001,
            min_confidence=0.05,
            median_frames=8,
            smoothing=0.5,
            mode="auto",
            cancellation_aware=False,
        )
        for mic_frame, ref_frame in zip(frames(mic), frames(ref)):
            estimator.update(mic_frame, ref_frame)

        self.assertAlmostEqual(estimator.delay_ms, 22.0, delta=4.0)
        self.assertGreater(estimator.confidence, 0.05)

    def test_estimator_holds_delay_on_silence(self) -> None:
        delay_samples = int(SAMPLE_RATE * 0.020)
        ref = broadband_signal(FRAME_SAMPLES * 80)
        mic = 0.70 * delayed(ref, delay_samples)
        estimator = GccPhatDelayEstimator(
            SAMPLE_RATE,
            FRAME_SAMPLES,
            max_delay_ms=100.0,
            window_ms=200.0,
            min_ref_rms=0.001,
            min_confidence=0.05,
            median_frames=5,
            smoothing=0.3,
            mode="auto",
        )
        for mic_frame, ref_frame in zip(frames(mic), frames(ref)):
            estimator.update(mic_frame, ref_frame)
        learned = estimator.delay_ms

        for _ in range(20):
            estimator.update(np.zeros(FRAME_SAMPLES, dtype=np.float32), np.zeros(FRAME_SAMPLES, dtype=np.float32))

        self.assertAlmostEqual(estimator.delay_ms, learned, delta=0.5)

    def test_auto_aligner_beats_wrong_manual_delay(self) -> None:
        from clean_speech_daemon.config import Config

        delay_samples = int(SAMPLE_RATE * 0.024)
        total = FRAME_SAMPLES * 160
        ref = broadband_signal(total)
        mic = 0.60 * delayed(ref, delay_samples)

        wrong = Config()
        wrong.input.sample_rate = SAMPLE_RATE
        wrong.input.frame_ms = FRAME_MS
        wrong.processing.enable_highpass = False
        wrong.processing.enable_reference_delay_align = True
        wrong.processing.reference_delay_mode = "manual"
        wrong.processing.reference_delay_ms = 8
        wrong.processing.enable_echo_cancellation = True
        wrong.processing.echo_canceller = "nlms"
        wrong.processing.echo_filter_taps = 2048
        wrong.processing.echo_step_size = 0.2
        wrong.processing.enable_noise_suppression = False
        wrong.processing.enable_vad = False

        auto = Config()
        auto.input.sample_rate = SAMPLE_RATE
        auto.input.frame_ms = FRAME_MS
        auto.processing.enable_highpass = False
        auto.processing.enable_reference_delay_align = True
        auto.processing.reference_delay_mode = "auto"
        auto.processing.reference_delay_ms = 8
        auto.processing.delay_window_ms = 250
        auto.processing.delay_update_min_ref_rms = 0.001
        auto.processing.delay_min_confidence = 0.05
        auto.processing.delay_median_frames = 8
        auto.processing.reference_delay_smoothing = 0.4
        auto.processing.delay_cancellation_aware = True
        auto.processing.enable_echo_cancellation = True
        auto.processing.echo_canceller = "nlms"
        auto.processing.echo_filter_taps = 2048
        auto.processing.echo_step_size = 0.2
        auto.processing.enable_noise_suppression = False
        auto.processing.enable_vad = False

        wrong_pipe = ProcessingPipeline(wrong)
        auto_pipe = ProcessingPipeline(auto)
        wrong_out = []
        auto_out = []
        for mic_frame, ref_frame in zip(frames(mic), frames(ref)):
            wrong_out.append(wrong_pipe.process(mic_frame, ref_frame))
            auto_out.append(auto_pipe.process(mic_frame, ref_frame))

        warmup = FRAME_SAMPLES * 40
        wrong_echo = np.concatenate(wrong_out)[warmup:]
        auto_echo = np.concatenate(auto_out)[warmup:]
        ref_seg = ref[warmup:]

        def residual(a: np.ndarray, b: np.ndarray) -> float:
            n = min(len(a), len(b))
            aa = a[:n] - np.mean(a[:n])
            bb = b[:n] - np.mean(b[:n])
            return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-9))

        self.assertAlmostEqual(auto_pipe.stats.reference_delay_ms, 24.0, delta=8.0)
        self.assertLess(abs(auto_pipe.stats.reference_delay_ms - 24.0), abs(wrong.processing.reference_delay_ms - 24.0))

    def test_cancellation_aware_fine_tune_finds_better_delay(self) -> None:
        true_delay = int(SAMPLE_RATE * 0.020)
        wrong_delay = int(SAMPLE_RATE * 0.035)
        total = FRAME_SAMPLES * 24
        ref = broadband_signal(total)
        mic = 0.70 * delayed(ref, true_delay)
        ref_concat = ref
        mic_frame = mic[-FRAME_SAMPLES:]

        wrong_ref = read_delayed_frame(ref_concat, FRAME_SAMPLES, float(wrong_delay))
        true_ref = read_delayed_frame(ref_concat, FRAME_SAMPLES, float(true_delay))
        assert wrong_ref is not None and true_ref is not None
        self.assertGreater(cancellation_residual_score(mic_frame, wrong_ref), cancellation_residual_score(mic_frame, true_ref))

        tuned_delay, tuned_score = cancellation_aware_fine_tune(
            mic_frame,
            ref_concat,
            FRAME_SAMPLES,
            float(wrong_delay),
            float(SAMPLE_RATE * 0.12),
            20.0,
            SAMPLE_RATE,
        )
        self.assertLess(tuned_score, cancellation_residual_score(mic_frame, wrong_ref))
        self.assertAlmostEqual(tuned_delay, true_delay, delta=SAMPLE_RATE * 0.004)

    def test_manual_aligner_still_returns_delayed_reference_frame(self) -> None:
        aligner = ReferenceDelayAligner(
            SAMPLE_RATE,
            FRAME_SAMPLES,
            FRAME_MS * 3,
            FRAME_MS * 8,
            smoothing=0.0,
            mode="manual",
        )
        reference_frames = frames(broadband_signal(FRAME_SAMPLES * 8))
        mic = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        outputs = [aligner.process(mic, frame) for frame in reference_frames]
        np.testing.assert_allclose(outputs[5], reference_frames[2], atol=1e-5)
        self.assertAlmostEqual(aligner.delay_ms, FRAME_MS * 3, delta=0.5)


if __name__ == "__main__":
    unittest.main()