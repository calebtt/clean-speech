"""Red-baseline tests that exercise the pipeline under *realistic* conditions.

The existing suite (test_echo_cancellation.py) only ever models echo as a
perfectly time-aligned, unfiltered, scaled copy of the reference -- which is the
single case the scalar AdaptiveEchoReducer can handle. These tests instead model
what actually reaches the microphone when desktop speakers play music/TV into the
room:

  * a *broadband* reference (real system audio, not pure tones),
  * convolved with a multi-tap *room impulse response* (direct path + reflections
    + speaker/mic colouring + a non-frame-aligned bulk delay),
  * optionally with slow *clock drift* between the mic ADC and the monitor tap.

They assert the behaviour a usable voice-agent front end needs. Status:

  A1  test_pipeline_cancels_realistic_room_echo      -> requires a real adaptive
  drift test_pipeline_cancels_echo_under_clock_drift     canceller (echo_canceller="nlms")
  A3  test_vad_state_updates_once_per_frame          -> requires the single-VAD-call fix
  A2  test_spectral_suppressor_does_not_modulate...  -> requires overlap-add (still red after step b)

Run:  python -m unittest tests.test_realistic_echo -v
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import lfilter, resample

from clean_speech_daemon.config import Config
from clean_speech_daemon.processing import EnergyVad, ProcessingPipeline, SpectralNoiseSuppressor


SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


# --------------------------------------------------------------------------- #
# Realistic signal generators (deterministic).
# --------------------------------------------------------------------------- #
def voice_signal(samples: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    envelope = np.clip(0.45 + 0.35 * np.sin(2.0 * np.pi * 2.0 * t) + 0.20 * np.sin(2.0 * np.pi * 3.7 * t), 0.0, None)
    signal = (
        0.060 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.030 * np.sin(2.0 * np.pi * 360.0 * t + 0.2)
        + 0.020 * np.sin(2.0 * np.pi * 540.0 * t + 0.4)
    )
    return (signal * envelope).astype(np.float32)


def broadband_reference(samples: int, seed: int = 7) -> np.ndarray:
    """System audio is music/TV: broadband, dynamic -- not a pure tone."""
    rng = np.random.RandomState(seed)
    x = rng.randn(samples)
    x = lfilter([1.0], [1.0, -0.9], x)      # colour it (low-pass-ish, like music)
    x = lfilter([1.0, -0.97], [1.0], x)     # tilt up the highs a little
    x = x / (np.std(x) + 1e-9) * 0.08
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    x = x * (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.7 * t))  # musical dynamics
    return x.astype(np.float32)


def room_impulse_response() -> np.ndarray:
    """Direct path + a handful of reflections at non-frame-aligned offsets."""
    ir = np.zeros(400, dtype=np.float64)
    ir[5] = 0.60
    ir[37] = 0.25
    ir[80] = 0.15
    ir[150] = 0.10
    ir[240] = 0.06
    ir[330] = 0.04
    return ir


def echo_from_reference(reference: np.ndarray, ir: np.ndarray | None = None) -> np.ndarray:
    ir = room_impulse_response() if ir is None else ir
    return np.convolve(reference, ir)[: len(reference)].astype(np.float32)


def frames(signal: np.ndarray) -> list[np.ndarray]:
    usable = len(signal) - (len(signal) % FRAME_SAMPLES)
    return [f.astype(np.float32) for f in signal[:usable].reshape(-1, FRAME_SAMPLES)]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def fraction_removed(mic: np.ndarray, cleaned: np.ndarray, clean: np.ndarray, start: int) -> float:
    n = min(len(mic), len(cleaned), len(clean))
    before = mse(mic[start:n], clean[start:n])
    after = mse(cleaned[start:n], clean[start:n])
    return 1.0 - after / (before + 1e-20)


def echo_config(canceller: str = "scalar") -> Config:
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
    # New knob introduced in step (b). Setting it is harmless if ignored.
    config.processing.echo_canceller = canceller
    return config


def run_pipeline(config: Config, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pipeline = ProcessingPipeline(config)
    return np.concatenate(
        [pipeline.process(m, r) for m, r in zip(frames(mic), frames(reference))]
    )


# --------------------------------------------------------------------------- #
# A1 -- the canceller must cope with real (filtered, non-aligned) echo.
# --------------------------------------------------------------------------- #
class RealisticEchoTests(unittest.TestCase):
    def test_pipeline_cancels_realistic_room_echo(self) -> None:
        total = FRAME_SAMPLES * 150
        clean = voice_signal(total)
        reference = broadband_reference(total)
        mic = (clean + echo_from_reference(reference)).astype(np.float32)

        cleaned = run_pipeline(echo_config("nlms"), mic, reference)

        removed = fraction_removed(mic, cleaned, clean, start=FRAME_SAMPLES * 60)
        # A real adaptive canceller clears most of the echo; the scalar one ~0%.
        self.assertGreaterEqual(
            removed, 0.70, f"only {removed * 100:.0f}% of realistic echo removed (need >=70%)"
        )

    def test_pipeline_cancels_echo_under_clock_drift(self) -> None:
        total = FRAME_SAMPLES * 150
        clean = voice_signal(total)
        reference = broadband_reference(total)

        # The mic ADC and the monitor tap run on independent clocks. Model that
        # drift: the echo path sees a slightly time-stretched reference (100 ppm,
        # a realistic upper bound for consumer audio hardware), while the canceller
        # is fed the un-stretched monitor reference. A static filter / fixed manual
        # delay cannot follow this; a continuously adapting filter can.
        drifted = resample(reference, int(len(reference) * (1.0 + 1e-4)))[: len(reference)].astype(np.float32)
        mic = (clean + echo_from_reference(drifted)).astype(np.float32)

        config = echo_config("nlms")
        # Faster adaptation to track drift without the sync layer. In the daemon
        # the DriftCompensatingReference handles clock drift, so the canceller
        # itself runs at the conservative default step.
        config.processing.echo_step_size = 1.0
        cleaned = run_pipeline(config, mic, reference)

        removed = fraction_removed(mic, cleaned, clean, start=FRAME_SAMPLES * 60)
        self.assertGreaterEqual(
            removed, 0.55, f"only {removed * 100:.0f}% removed under clock drift (need >=55%)"
        )


    def test_canceller_does_not_diverge_on_intermittent_reference_and_double_talk(self) -> None:
        # The real-world failure: system audio that goes silent for long stretches
        # (music/TV with gaps) plus loud near-end speech that is uncorrelated with
        # the reference. Naive NLMS normalization blows the filter up to ~1e73 on
        # the silent/quiet passages. The canceller must stay bounded AND still
        # cancel during the passages where the reference is active.
        rng = np.random.RandomState(5)
        total = FRAME_SAMPLES * 400
        t = np.arange(total) / SAMPLE_RATE
        near_end = (0.05 * np.sin(2.0 * np.pi * 200.0 * t) * (np.sin(2.0 * np.pi * 1.5 * t) > 0)).astype(np.float32)
        reference = (0.02 * rng.randn(total)).astype(np.float32)
        gate = np.repeat((rng.rand(total // FRAME_SAMPLES) > 0.5).astype(np.float32), FRAME_SAMPLES)[:total]
        reference = (reference * gate).astype(np.float32)  # long silent stretches
        echo = lfilter([0, 0, 0, 0, 0, 0.6, 0, 0, 0.25], [1.0], reference).astype(np.float32)
        mic = (near_end + echo).astype(np.float32)

        cleaned = run_pipeline(echo_config("nlms"), mic, reference)

        self.assertTrue(np.all(np.isfinite(cleaned)), "canceller produced non-finite output (divergence)")
        self.assertLess(float(np.max(np.abs(cleaned))), 2.0, "output exploded -> filter diverged")
        active = gate.astype(bool)[: len(cleaned)]
        before = float(np.mean((mic[: len(cleaned)] - near_end[: len(cleaned)])[active] ** 2))
        after = float(np.mean((cleaned - near_end[: len(cleaned)])[active] ** 2))
        self.assertGreater(1.0 - after / (before + 1e-20), 0.5, "echo not reduced during active-reference passages")


# --------------------------------------------------------------------------- #
# A2 -- spectral suppressor must not amplitude-modulate the signal it passes.
# Stays RED after step (b); fixed by the overlap-add rework (option c).
# --------------------------------------------------------------------------- #
class SpectralSuppressorEnvelopeTests(unittest.TestCase):
    def test_spectral_suppressor_does_not_modulate_steady_tone(self) -> None:
        # Configure the suppressor to be a no-op (no spectral subtraction) so we
        # isolate the framing/reconstruction path from the subtraction itself.
        suppressor = SpectralNoiseSuppressor(
            FRAME_SAMPLES, noise_learn_frames=1, mode="quality", reduction=0.0, floor=1.0
        )
        t = np.arange(FRAME_SAMPLES * 20, dtype=np.float64) / SAMPLE_RATE
        tone = (0.1 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

        out = np.concatenate([suppressor.process(f, speech_likely=True) for f in frames(tone)])
        out = out.reshape(-1, FRAME_SAMPLES)[5:]  # skip warmup

        # A steady tone must keep a steady amplitude across each 20 ms frame.
        # With proper overlap-add, samples near the frame edges carry the same
        # energy as samples in the middle. With per-frame Hann windowing and no
        # overlap-add, the edges fade to zero -> a 50 Hz warble.
        edge = np.concatenate([out[:, :48], out[:, -48:]], axis=1)
        center = out[:, FRAME_SAMPLES // 2 - 48 : FRAME_SAMPLES // 2 + 48]
        edge_rms = float(np.sqrt(np.mean(edge ** 2)))
        center_rms = float(np.sqrt(np.mean(center ** 2)))
        ratio = edge_rms / (center_rms + 1e-12)
        self.assertGreaterEqual(
            ratio, 0.80, f"frame edges are {ratio:.2f}x the center energy (warble); need >=0.80"
        )


# --------------------------------------------------------------------------- #
# A3 -- the VAD adapts its noise floor, so it must run exactly once per frame.
# Fixed in step (b).
# --------------------------------------------------------------------------- #
class VadCallCountTests(unittest.TestCase):
    def test_vad_noise_floor_adapts_once_per_frame(self) -> None:
        # With suppression off, the pre- and post-suppression frames are identical,
        # so one frame must move the noise floor by exactly one adaptation step --
        # the same as a single direct EnergyVad.score() call. The original pipeline
        # adapted on both of its per-frame score() calls, moving the floor twice.
        config = Config()
        config.input.sample_rate = SAMPLE_RATE
        config.input.frame_ms = FRAME_MS
        config.processing.enable_highpass = False  # so the VAD sees the raw frame
        config.processing.enable_echo_cancellation = False
        config.processing.enable_noise_suppression = False
        config.processing.enable_speech_enhancement = False

        pipeline = ProcessingPipeline(config)

        rng = np.random.RandomState(3)
        quiet = (0.003 * rng.randn(FRAME_SAMPLES)).astype(np.float32)  # below VAD threshold

        reference_vad = EnergyVad(float(config.vad.threshold))
        reference_vad.score(quiet)  # exactly one adaptation
        expected_noise_rms = reference_vad.noise_rms

        pipeline.process(quiet)

        self.assertAlmostEqual(
            pipeline.vad.noise_rms,
            expected_noise_rms,
            places=9,
            msg="noise floor adapted more than once per frame (double VAD update)",
        )


if __name__ == "__main__":
    unittest.main()
