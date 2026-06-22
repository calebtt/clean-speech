"""Runtime delay tuning via ProcessingPipeline.apply_tuning."""

from __future__ import annotations

import unittest

import numpy as np

from clean_speech_daemon.config import Config
from clean_speech_daemon.processing import ProcessingPipeline


SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class RuntimeTuningTests(unittest.TestCase):
    def test_apply_tuning_updates_manual_reference_delay(self) -> None:
        config = Config()
        config.input.sample_rate = SAMPLE_RATE
        config.input.frame_ms = FRAME_MS
        config.processing.enable_reference_delay_align = True
        config.processing.reference_delay_mode = "manual"
        config.processing.reference_delay_ms = 10
        config.processing.enable_echo_cancellation = False
        config.processing.enable_vad = False
        pipeline = ProcessingPipeline(config)

        applied = pipeline.apply_tuning(reference_delay_ms=33.5, reference_delay_mode="manual")
        frame = np.linspace(-0.04, 0.04, FRAME_SAMPLES, dtype=np.float32)
        reference = np.sin(np.linspace(0.0, 12.0, FRAME_SAMPLES)).astype(np.float32) * 0.08
        for _ in range(8):
            pipeline.process(frame, reference)

        self.assertEqual(applied["reference_delay_ms"], 33.5)
        self.assertAlmostEqual(pipeline.stats.reference_delay_ms, 33.5, delta=0.5)

    def test_reset_echo_filter_clears_nlms_gain(self) -> None:
        config = Config()
        config.input.sample_rate = SAMPLE_RATE
        config.input.frame_ms = FRAME_MS
        config.processing.echo_canceller = "nlms"
        config.processing.enable_echo_cancellation = True
        config.processing.enable_vad = False
        pipeline = ProcessingPipeline(config)
        mic = np.ones(FRAME_SAMPLES, dtype=np.float32) * 0.05
        ref = np.ones(FRAME_SAMPLES, dtype=np.float32) * 0.08
        for _ in range(30):
            pipeline.process(mic, ref)
        self.assertGreater(pipeline.stats.echo_gain, 0.0)
        pipeline.reset_echo_filter()
        self.assertEqual(pipeline.echo.gain, 0.0)


if __name__ == "__main__":
    unittest.main()