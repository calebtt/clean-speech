from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import tomllib
import unittest

import numpy as np

from clean_speech_daemon.aec import NlmsEchoCanceller
from clean_speech_daemon.config import Config, load_config
from clean_speech_daemon.outputs import MULTI_STREAM_NAMES
from clean_speech_daemon.processing import ProcessingPipeline


def load_testbed_module():  # noqa: ANN201
    path = Path("/home/caleb/clean-speech-testbed/clean_speech_testbed.py")
    spec = importlib.util.spec_from_file_location("clean_speech_testbed_runtime_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeConfigRegressionTests(unittest.TestCase):
    def test_default_echo_canceller_is_hybrid_localvqe(self) -> None:
        config = Config()

        self.assertEqual(config.processing.echo_canceller, "hybrid_localvqe")
        self.assertTrue(config.processing.enable_reference_delay_align)
        self.assertFalse(config.processing.enable_reference_level_match)
        # 60 ms is the shipped default: it restores AEC causality (the parec monitor
        # reference arrives after the echo on live captures).
        self.assertEqual(config.processing.mic_delay_ms, 60)
        self.assertGreaterEqual(config.processing.echo_filter_taps, 512)

    def test_old_config_without_echo_canceller_uses_shipped_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """[input]
sample_rate = 48000
frame_ms = 20

[processing]
enable_echo_cancellation = true
""",
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.processing.echo_canceller, "hybrid_localvqe")

    def test_example_config_uses_hybrid_localvqe_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = tomllib.loads((root / "config.example.toml").read_text(encoding="utf-8"))
        processing = profile["processing"]
        self.assertEqual(processing["echo_canceller"], "hybrid_localvqe")
        self.assertGreaterEqual(processing["echo_filter_taps"], 4096)

    def test_nlms_profiles_remain_explicitly_nlms(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in ("profiles/echo-vad.toml", "profiles/stage-wav-debug.toml", "profiles/nlms-aec.toml"):
            with self.subTest(profile=relative):
                profile = tomllib.loads((root / relative).read_text(encoding="utf-8"))
                processing = profile["processing"]
                self.assertEqual(processing["echo_canceller"], "nlms")
                self.assertGreaterEqual(processing["echo_filter_taps"], 4096)

    def test_after_echo_stage_and_stream_are_exposed_before_vad_delay(self) -> None:
        config = Config()
        config.input.sample_rate = 48_000
        config.input.frame_ms = 20
        config.processing.enable_highpass = False
        config.processing.enable_echo_cancellation = False
        config.processing.enable_noise_suppression = False
        config.processing.enable_speech_enhancement = False
        config.processing.enable_vad = True
        config.processing.mic_delay_ms = 0  # after_echo must equal the input frame
        config.vad.pre_roll_ms = 300
        frame = np.full(960, 0.05, dtype=np.float32)

        pipeline = ProcessingPipeline(config)
        output = pipeline.process(frame)

        self.assertIn("after_echo", pipeline.last_stages)
        self.assertIn("after_echo", MULTI_STREAM_NAMES)
        np.testing.assert_allclose(pipeline.last_stages["after_echo"], frame, atol=1e-7)
        np.testing.assert_allclose(output, np.zeros_like(frame), atol=1e-7)

    def test_nlms_boundary_smoothing_removes_block_step(self) -> None:
        canceller = NlmsEchoCanceller(960, taps=512, step_size=0.0, boundary_smoothing_samples=64)
        reference = np.ones(960, dtype=np.float32) * 0.1

        first = canceller.process(np.zeros(960, dtype=np.float32), reference)
        second = canceller.process(np.ones(960, dtype=np.float32), reference)

        self.assertAlmostEqual(float(first[-1]), 0.0, places=7)
        self.assertAlmostEqual(float(second[0]), float(first[-1]), places=7)
        self.assertGreater(float(second[63]), 0.95)

    def test_testbed_reports_sample_offset_and_boundary_ratio(self) -> None:
        testbed = load_testbed_module()
        rng = np.random.RandomState(17)
        reference = rng.randn(48_000).astype(np.float32) * 0.05
        candidate = np.zeros_like(reference)
        candidate[321:] = reference[:-321]
        artifact = candidate.copy()
        artifact[960::960] += 0.1

        sample_lag, corr = testbed.estimate_offset_samples(reference, candidate)

        self.assertEqual(sample_lag, 321)
        self.assertGreater(corr, 0.99)
        self.assertGreater(testbed.boundary_jump_ratio(artifact), testbed.boundary_jump_ratio(candidate) * 2.0)


if __name__ == "__main__":
    unittest.main()
