"""Realtime neural echo canceller (DTLN / NKF / hybrid) integration tests.

Skipped automatically when the optional neural deps (torch / onnxruntime / soxr)
or the model files / fixture recording are not present.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

try:
    import onnxruntime  # noqa: F401
    import soxr  # noqa: F401
    import torch  # noqa: F401
    _HAVE_DEPS = True
except Exception:  # noqa: BLE001
    _HAVE_DEPS = False

from clean_speech_daemon.config import Config

MODELS = Path(__file__).resolve().parents[1] / "models"
RECORDINGS = Path.home() / "clean-speech-recordings"
FIXTURE = "20260622-092844"  # echo-dominant double-talk capture
SR = 48_000
FRAME = 960


def _models_present() -> bool:
    return all((MODELS / m).exists() for m in ("dtln_aec_512_1.onnx", "dtln_aec_512_2.onnx", "nkf_epoch70.pt"))


def _best_lag_corr(a: np.ndarray, b: np.ndarray, step: int = 24, max_lag_ms: int = 100) -> float:
    k = min(len(a), len(b))
    a = a[:k] - a[:k].mean()
    b = b[:k] - b[:k].mean()
    m = int(SR * max_lag_ms / 1000)
    best = 0.0
    for lag in range(-m, m + 1, step):
        aa, bb = (a[-lag:], b[: len(a) + lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag > 0 else (a, b))
        d = np.linalg.norm(aa) * np.linalg.norm(bb)
        if d > 1e-9:
            best = max(best, abs(float(aa @ bb / d)))
    return best


@unittest.skipUnless(_HAVE_DEPS and _models_present(), "neural AEC deps/models not available")
class NeuralAecTests(unittest.TestCase):
    def _load_fixture(self):
        import soundfile as sf

        mic_path = RECORDINGS / f"{FIXTURE}-mic_raw.wav"
        ref_path = RECORDINGS / f"{FIXTURE}-system_reference.wav"
        if not mic_path.exists() or not ref_path.exists():
            self.skipTest("fixture recording not available")
        mic, sr = sf.read(mic_path, dtype="float32")
        ref, _ = sf.read(ref_path, dtype="float32")
        self.assertEqual(sr, SR)
        n = min(len(mic), len(ref))
        return mic[:n], ref[:n]

    def test_factory_builds_neural_cancellers(self) -> None:
        from clean_speech_daemon.aec import make_echo_canceller
        from clean_speech_daemon.neural_aec import NeuralEchoCanceller

        for kind in ("dtln", "nkf", "hybrid"):
            config = Config()
            config.input.sample_rate = SR
            config.processing.echo_canceller = kind
            canceller = make_echo_canceller(config, FRAME)
            self.assertIsInstance(canceller, NeuralEchoCanceller)

    def test_each_frame_returns_exactly_frame_samples(self) -> None:
        from clean_speech_daemon.neural_aec import NeuralEchoCanceller

        aec = NeuralEchoCanceller("hybrid", FRAME, SR)
        rng = np.random.RandomState(0)
        for _ in range(20):
            out = aec.process(rng.randn(FRAME).astype(np.float32) * 0.05,
                              rng.randn(FRAME).astype(np.float32) * 0.05)
            self.assertEqual(len(out), FRAME)
        # reference=None must also return a full frame (passthrough, in sync)
        self.assertEqual(len(aec.process(rng.randn(FRAME).astype(np.float32) * 0.05, None)), FRAME)

    def test_hybrid_cancels_echo_dominant_recording(self) -> None:
        from clean_speech_daemon.neural_aec import NeuralEchoCanceller

        mic, ref = self._load_fixture()
        aec = NeuralEchoCanceller("hybrid", FRAME, SR, mask_smooth=0.6)
        out = np.zeros(len(mic), np.float32)
        for i in range(0, len(mic) - FRAME, FRAME):
            out[i:i + FRAME] = aec.process(mic[i:i + FRAME], ref[i:i + FRAME])

        w = int(2.5 * SR)
        def rms(x):
            return float(np.sqrt(np.mean(x * x)) + 1e-12)
        reduction = 20.0 * np.log10(rms(mic[w:]) / rms(out[w:]))
        residual = _best_lag_corr(out[w:], ref[w:])
        mic_corr = _best_lag_corr(mic[w:], ref[w:])

        self.assertGreaterEqual(reduction, 12.0, f"hybrid removed only {reduction:.1f} dB")
        self.assertLess(residual, 0.15, f"residual echo corr {residual:.3f}")
        self.assertLess(residual, 0.4 * mic_corr, f"residual {residual:.3f} vs mic {mic_corr:.3f}")

    def test_hybrid_localvqe_blend_builds_and_cancels(self):
        from clean_speech_daemon.neural_aec import NeuralEchoCanceller
        try:
            from clean_speech_daemon.neural_aec import StreamingLocalVQE
            StreamingLocalVQE()  # probe: needs lib/liblocalvqe.so + GGUF model
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"LocalVQE engine/model not available: {exc}")

        mic, ref = self._load_fixture()
        aec = NeuralEchoCanceller("hybrid_localvqe", FRAME, SR, mask_smooth=0.6, localvqe_blend=0.7)
        out = np.zeros(len(mic), np.float32)
        for i in range(0, len(mic) - FRAME, FRAME):
            out[i:i + FRAME] = aec.process(mic[i:i + FRAME], ref[i:i + FRAME])
        w = int(2.5 * SR)
        def rms(x):
            return float(np.sqrt(np.mean(x * x)) + 1e-12)
        reduction = 20.0 * np.log10(rms(mic[w:]) / rms(out[w:]))
        self.assertGreaterEqual(reduction, 12.0, f"hybrid_localvqe removed only {reduction:.1f} dB")
        self.assertLess(_best_lag_corr(out[w:], ref[w:]), 0.15)


if __name__ == "__main__":
    unittest.main()
