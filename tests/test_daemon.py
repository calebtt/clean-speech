"""Regression: an underrun (no mic frame within the read timeout) used to raise
an uncaught queue.Empty out of the audio loop, crashing the whole daemon instead
of just skipping that iteration."""

from __future__ import annotations

from pathlib import Path
from queue import Empty
import tempfile
import unittest

import numpy as np

import clean_speech_daemon.daemon as daemon_module
from clean_speech_daemon.audio import AudioFrame
from clean_speech_daemon.config import Config
from clean_speech_daemon.daemon import CleanSpeechDaemon


class _StopSentinel(Exception):
    """Raised by the fake mic reader once the test has seen enough calls, so we
    can assert the loop reached this point without needing a real audio device
    or a separate thread to flip stop_requested."""


class _FakeMicReader:
    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0

    def __enter__(self) -> "_FakeMicReader":
        return self

    def __exit__(self, *exc_info) -> bool:  # noqa: ANN001
        return False

    def read(self, timeout: float = 0.5) -> AudioFrame:
        self.calls += 1
        if self.calls <= 2:
            # Simulate a couple of underrun cycles: no frame arrived in time.
            raise Empty()
        if self.calls == 3:
            return AudioFrame(np.zeros(960, dtype=np.float32), None)
        raise _StopSentinel()


def _make_isolated_config(directory: str) -> Config:
    config = Config()
    config.input.system_audio_reference = "off"
    config.processing.echo_canceller = "nlms"  # avoid pulling in neural models for this test
    config.diagnostics.enabled = False
    config.output.pipewire_virtual_source = False
    config.output.auto_load_pulse_pipe_source = False
    config.output.socket_path = str(Path(directory) / "out.sock")
    config.output.streams_socket_path = str(Path(directory) / "streams.sock")
    config.output.fifo_path = str(Path(directory) / "fifo.pcm")
    config.output.control_socket_path = str(Path(directory) / "control.sock")
    config.output.debug_wav_path = ""
    return config


class MicUnderrunTests(unittest.TestCase):
    def test_mic_read_timeout_does_not_crash_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _make_isolated_config(directory)
            fake_reader = _FakeMicReader()
            original = daemon_module.InputStreamReader
            daemon_module.InputStreamReader = lambda *a, **k: fake_reader
            try:
                daemon = CleanSpeechDaemon(config)
                with self.assertRaises(_StopSentinel):
                    daemon._run_audio_loop()
            finally:
                daemon_module.InputStreamReader = original

            # Two Empty()s were swallowed (not raised out of the loop), and
            # processing reached the real frame delivered on the 3rd call.
            self.assertEqual(fake_reader.calls, 4)
            self.assertGreaterEqual(daemon.pipeline.stats.frames, 1)


if __name__ == "__main__":
    unittest.main()
