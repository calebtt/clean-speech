from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from clean_speech_daemon.config import Config
from clean_speech_daemon.processing import AdaptiveEchoReducer, ProcessingPipeline, ReferenceDelayAligner, ReferenceLevelMatcher


SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


def voice_like_signal(samples: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    envelope = 0.45 + 0.35 * np.sin(2.0 * np.pi * 2.0 * t) + 0.20 * np.sin(2.0 * np.pi * 3.7 * t)
    signal = (
        0.055 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.028 * np.sin(2.0 * np.pi * 360.0 * t + 0.2)
        + 0.018 * np.sin(2.0 * np.pi * 540.0 * t + 0.4)
    )
    return (signal * envelope).astype(np.float32)


def system_reference_signal(samples: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    signal = (
        0.060 * np.sin(2.0 * np.pi * 733.0 * t + 0.1)
        + 0.040 * np.sin(2.0 * np.pi * 1189.0 * t + 0.5)
        + 0.020 * np.sin(2.0 * np.pi * 1777.0 * t + 0.9)
    )
    gate = 0.7 + 0.3 * np.sin(2.0 * np.pi * 1.3 * t)
    return (signal * gate).astype(np.float32)


def delayed(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    out = np.zeros_like(signal)
    if delay_samples <= 0:
        return signal.copy()
    out[delay_samples:] = signal[:-delay_samples]
    return out


def frames(signal: np.ndarray) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % FRAME_SAMPLES)
    return [frame.astype(np.float32) for frame in signal[:usable].reshape(-1, FRAME_SAMPLES)]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def base_echo_config() -> Config:
    config = Config()
    config.input.sample_rate = SAMPLE_RATE
    config.input.frame_ms = FRAME_MS
    config.processing.enable_highpass = False
    config.processing.enable_reference_delay_align = False
    config.processing.enable_reference_level_match = False
    config.processing.enable_echo_cancellation = True
    config.processing.enable_noise_suppression = False
    config.processing.enable_speech_enhancement = False
    config.processing.enable_vad = False
    # Synthetic mic/reference are sample-aligned here, so cancel the shipped
    # default mic delay (which exists to restore causality on live captures).
    config.processing.mic_delay_ms = 0
    # Strong step to exercise full convergence on clean synthetic echo (matches the
    # shipped default; pinned so these tests are independent of default changes).
    config.processing.echo_step_size = 0.3
    config.processing.echo_step_size_warmup = 0.3
    return config


def load_testbed_module():  # noqa: ANN201
    path = Path("/home/caleb/clean-speech-testbed/clean_speech_testbed.py")
    spec = importlib.util.spec_from_file_location("clean_speech_testbed_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EchoCancellationTests(unittest.TestCase):
    def test_adaptive_echo_reducer_improves_known_mixture(self) -> None:
        total_samples = FRAME_SAMPLES * 120
        clean = voice_like_signal(total_samples)
        reference = system_reference_signal(total_samples)
        mic = clean + 0.65 * reference

        reducer = AdaptiveEchoReducer()
        cleaned_frames = [reducer.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))]
        cleaned = np.concatenate(cleaned_frames)

        # Ignore early adaptation frames and compare to the known perfect clean signal.
        start = FRAME_SAMPLES * 25
        before = mse(mic[start:], clean[start:])
        after = mse(cleaned[start:], clean[start:])
        self.assertLess(after, before * 0.20)

    def test_adaptive_echo_reducer_passes_through_without_reference(self) -> None:
        clean = voice_like_signal(FRAME_SAMPLES * 3)
        reducer = AdaptiveEchoReducer()

        output = np.concatenate([reducer.process(frame, None) for frame in frames(clean)])

        np.testing.assert_allclose(output, clean, atol=1e-7)

    def test_processing_pipeline_echo_stage_improves_known_mixture(self) -> None:
        config = base_echo_config()

        total_samples = FRAME_SAMPLES * 120
        clean = voice_like_signal(total_samples)
        reference = system_reference_signal(total_samples)
        mic = clean + 0.55 * reference

        pipeline = ProcessingPipeline(config)
        cleaned = np.concatenate([pipeline.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))])

        start = FRAME_SAMPLES * 25
        before = mse(mic[start:], clean[start:])
        after = mse(cleaned[start:], clean[start:])
        self.assertLess(after, before * 0.25)

    def test_pipeline_manual_delay_alignment_improves_delayed_echo(self) -> None:
        delay_frames = 5
        delay_samples = delay_frames * FRAME_SAMPLES
        total_samples = FRAME_SAMPLES * 160
        clean = voice_like_signal(total_samples)
        reference = system_reference_signal(total_samples)
        mic = clean + 0.70 * delayed(reference, delay_samples)

        no_align_config = base_echo_config()
        no_align_pipeline = ProcessingPipeline(no_align_config)
        no_align = np.concatenate(
            [no_align_pipeline.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))]
        )

        align_config = base_echo_config()
        align_config.processing.enable_reference_delay_align = True
        align_config.processing.reference_delay_mode = "manual"
        align_config.processing.reference_delay_ms = delay_frames * FRAME_MS
        align_config.processing.reference_max_delay_ms = 300
        align_pipeline = ProcessingPipeline(align_config)
        aligned = np.concatenate(
            [align_pipeline.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))]
        )

        start = FRAME_SAMPLES * 40
        no_align_error = mse(no_align[start:], clean[start:])
        aligned_error = mse(aligned[start:], clean[start:])
        self.assertLess(aligned_error, no_align_error * 0.35)
        self.assertEqual(align_pipeline.stats.reference_delay_ms, delay_frames * FRAME_MS)

    def test_pipeline_wrong_manual_delay_is_worse_than_correct_delay(self) -> None:
        delay_frames = 6
        delay_samples = delay_frames * FRAME_SAMPLES
        total_samples = FRAME_SAMPLES * 160
        clean = voice_like_signal(total_samples)
        reference = system_reference_signal(total_samples)
        mic = clean + 0.60 * delayed(reference, delay_samples)

        correct_config = base_echo_config()
        correct_config.processing.enable_reference_delay_align = True
        correct_config.processing.reference_delay_mode = "manual"
        correct_config.processing.reference_delay_ms = delay_frames * FRAME_MS
        correct_config.processing.reference_max_delay_ms = 300
        correct = ProcessingPipeline(correct_config)
        correct_output = np.concatenate([correct.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))])

        wrong_config = base_echo_config()
        wrong_config.processing.enable_reference_delay_align = True
        wrong_config.processing.reference_delay_mode = "manual"
        wrong_config.processing.reference_delay_ms = FRAME_MS
        wrong_config.processing.reference_max_delay_ms = 300
        wrong = ProcessingPipeline(wrong_config)
        wrong_output = np.concatenate([wrong.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(reference))])

        start = FRAME_SAMPLES * 40
        self.assertLess(mse(correct_output[start:], clean[start:]), mse(wrong_output[start:], clean[start:]) * 0.50)

    def test_reference_level_matcher_scales_quiet_reference_towards_mic(self) -> None:
        total_samples = FRAME_SAMPLES * 40
        reference = system_reference_signal(total_samples)
        mic = 0.40 * reference
        quiet_reference = 0.10 * reference
        matcher = ReferenceLevelMatcher(min_gain=0.05, max_gain=20.0, smoothing=0.0, target_ratio=1.0)

        matched = np.concatenate([matcher.process(mic_frame, ref_frame) for mic_frame, ref_frame in zip(frames(mic), frames(quiet_reference))])

        self.assertAlmostEqual(matcher.gain, 4.0, delta=0.05)
        self.assertLess(mse(matched, mic), mse(quiet_reference, mic) * 0.05)

    def test_pipeline_with_silent_reference_does_not_damage_clean_signal(self) -> None:
        config = base_echo_config()
        total_samples = FRAME_SAMPLES * 30
        clean = voice_like_signal(total_samples)
        silence = np.zeros_like(clean)
        pipeline = ProcessingPipeline(config)

        output = np.concatenate([pipeline.process(clean_frame, ref_frame) for clean_frame, ref_frame in zip(frames(clean), frames(silence))])

        np.testing.assert_allclose(output, clean, atol=1e-5)

    def test_manual_delay_aligner_returns_delayed_reference_frame(self) -> None:
        aligner = ReferenceDelayAligner(
            SAMPLE_RATE,
            FRAME_SAMPLES,
            FRAME_MS * 3,
            FRAME_MS * 8,
            smoothing=0.0,
            mode="manual",
        )
        reference_frames = frames(system_reference_signal(FRAME_SAMPLES * 8))
        mic = np.zeros(FRAME_SAMPLES, dtype=np.float32)

        outputs = [aligner.process(mic, frame) for frame in reference_frames]

        np.testing.assert_allclose(outputs[5], reference_frames[2], atol=1e-5)
        self.assertAlmostEqual(aligner.delay_ms, FRAME_MS * 3, delta=0.5)

    def test_alignment_report_detects_known_reference_offset(self) -> None:
        testbed = load_testbed_module()
        total_samples = FRAME_SAMPLES * 80
        mic = system_reference_signal(total_samples)
        reference = delayed(mic, FRAME_SAMPLES * 4)

        report = testbed.alignment_report({"mic_raw": mic, "system_reference": reference})
        offset = report["offsets_vs_mic_raw"]["system_reference"]

        self.assertEqual(offset["offset_ms"], -80)
        self.assertGreater(abs(offset["correlation"]), 0.75)


if __name__ == "__main__":
    unittest.main()
