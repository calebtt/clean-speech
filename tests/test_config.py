"""Config loading: bad TOML, unknown keys, and unrecognized enum values should
all be surfaced clearly instead of failing silently or with a raw traceback."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest

from clean_speech_daemon.config import (
    Config,
    ConfigError,
    default_config_text,
    load_config,
)


@contextlib.contextmanager
def _capture_stderr():
    original = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield sys.stderr
    finally:
        sys.stderr = original


class DefaultConfigTextTests(unittest.TestCase):
    def test_written_defaults_match_dataclass_defaults(self) -> None:
        # Regression: default_config_text() hardcoded reference_sync_latency_frames
        # = 1.0 while the ProcessingConfig dataclass default was 1.5, so a config
        # written by `write-config` silently behaved differently than no config
        # file at all.
        written = tomllib.loads(default_config_text())
        implicit = Config()
        self.assertEqual(
            written["processing"]["reference_sync_latency_frames"],
            implicit.processing.reference_sync_latency_frames,
        )

    def test_default_config_text_round_trips_to_the_same_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(default_config_text(), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(loaded.processing.echo_canceller, Config().processing.echo_canceller)
        self.assertEqual(
            loaded.processing.reference_sync_latency_frames,
            Config().processing.reference_sync_latency_frames,
        )


class MalformedConfigTests(unittest.TestCase):
    def test_bad_toml_raises_a_clean_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text("this is not [ valid toml", encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
        self.assertIn(str(path), str(ctx.exception))


class UnknownKeyWarningTests(unittest.TestCase):
    def test_unknown_key_warns_and_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """[processing]
eco_canceller = "hybrid_localvqe"
""",
                encoding="utf-8",
            )
            with _capture_stderr() as stderr:
                config = load_config(path)
        # The typo'd key must not silently apply, and the real default is untouched.
        self.assertEqual(config.processing.echo_canceller, Config().processing.echo_canceller)
        self.assertIn("eco_canceller", stderr.getvalue())


class EnumValidationTests(unittest.TestCase):
    def test_unrecognized_echo_canceller_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """[processing]
echo_canceller = "not-a-real-backend"
""",
                encoding="utf-8",
            )
            with _capture_stderr() as stderr:
                config = load_config(path)
        self.assertEqual(config.processing.echo_canceller, "not-a-real-backend")
        self.assertIn("echo_canceller", stderr.getvalue())
        self.assertIn("not-a-real-backend", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
