from __future__ import annotations

import numpy as np

from clean_speech_daemon.config import Config
from clean_speech_daemon.processing import ProcessingPipeline


def test_pipeline_processes_frame() -> None:
    config = Config()
    frame_samples = int(config.input.sample_rate * config.input.frame_ms / 1000)
    pipeline = ProcessingPipeline(config)
    t = np.arange(frame_samples, dtype=np.float32) / config.input.sample_rate
    frame = 0.05 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    cleaned = pipeline.process(frame)
    assert cleaned.shape == frame.shape
    assert cleaned.dtype == np.float32
    assert np.all(np.isfinite(cleaned))


if __name__ == "__main__":
    test_pipeline_processes_frame()
    print("smoke tests passed")
