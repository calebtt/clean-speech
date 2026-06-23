from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np

from .aec import make_echo_canceller
from .config import Config
from .delay_align import GccPhatDelayEstimator


@dataclass(slots=True)
class ProcessingStats:
    frames: int = 0
    speech_frames: int = 0
    vad_score: float = 0.0
    pre_vad_score: float = 0.0
    noise_floor: float = 0.0
    echo_gain: float = 0.0
    reference_gain: float = 1.0
    reference_delay_ms: float = 0.0
    reference_delay_correlation: float = 0.0
    reference_delay_confidence: float = 0.0
    reference_cancellation_score: float = 1.0
    residual_ref_correlation: float = 0.0
    ref_present: bool = False
    clipped_output_pct: float = 0.0

    @property
    def speech_ratio(self) -> float:
        if self.frames == 0:
            return 0.0
        return self.speech_frames / self.frames


class OnePoleHighPass:
    def __init__(self, cutoff_hz: float, sample_rate: int) -> None:
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        self.alpha = rc / (rc + dt)
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        prev_x = self.prev_x
        prev_y = self.prev_y
        alpha = self.alpha
        for i, sample in enumerate(x):
            prev_y = alpha * (prev_y + float(sample) - prev_x)
            prev_x = float(sample)
            y[i] = prev_y
        self.prev_x = prev_x
        self.prev_y = prev_y
        return y


class AdaptiveEchoReducer:
    def __init__(self) -> None:
        self.gain = 0.0

    def process(self, mic: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        if reference is None or len(reference) != len(mic):
            return mic
        ref_energy = float(np.dot(reference, reference)) + 1e-8
        corr_gain = float(np.dot(mic, reference)) / ref_energy
        corr_gain = float(np.clip(corr_gain, -1.5, 1.5))
        self.gain = 0.96 * self.gain + 0.04 * corr_gain
        cleaned = mic - self.gain * reference
        return np.asarray(cleaned, dtype=np.float32)


class ReferenceLevelMatcher:
    def __init__(self, min_gain: float, max_gain: float, smoothing: float, target_ratio: float) -> None:
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
        self.target_ratio = float(target_ratio)
        self.gain = 1.0

    def process(self, mic: np.ndarray, reference: np.ndarray | None) -> np.ndarray | None:
        if reference is None or len(reference) != len(mic):
            return reference
        mic_rms = float(np.sqrt(np.mean(mic * mic)) + 1e-8)
        ref_rms = float(np.sqrt(np.mean(reference * reference)) + 1e-8)
        if ref_rms <= 1e-7:
            return reference * self.gain
        target_gain = np.clip((mic_rms * self.target_ratio) / ref_rms, self.min_gain, self.max_gain)
        self.gain = self.smoothing * self.gain + (1.0 - self.smoothing) * float(target_gain)
        return np.asarray(reference * self.gain, dtype=np.float32)


class SampleReferenceDelayAligner:
    """Fractional-sample reference delay for echo cancellation.

    Manual mode uses a fixed delay. Auto and calibrate modes run a rolling
    GCC-PHAT estimator (see :mod:`delay_align`) on ref-active windows, then read
    the delayed reference with ``np.interp`` for sub-frame precision.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        initial_delay_ms: float,
        max_delay_ms: int,
        smoothing: float,
        mode: str,
        window_ms: float = 300.0,
        min_ref_rms: float = 0.003,
        min_confidence: float = 0.15,
        median_frames: int = 15,
        calibrate_seconds: float = 10.0,
        cancellation_aware: bool = True,
        fine_tune_ms: float = 15.0,
        target_residual_corr: float = 0.15,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.max_delay_samples = max(0, int(self.sample_rate * max_delay_ms / 1000))
        self.manual_delay_samples = max(0.0, float(self.sample_rate * initial_delay_ms / 1000))
        self.delay_samples = self.manual_delay_samples
        self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
        self.mode = mode
        self.history: deque[np.ndarray] = deque()
        self.max_history_samples = self.max_delay_samples + self.frame_samples + 1
        self.correlation = 0.0
        self.confidence = 0.0
        self._estimator = GccPhatDelayEstimator(
            sample_rate,
            frame_samples,
            float(max_delay_ms),
            window_ms=window_ms,
            min_ref_rms=min_ref_rms,
            min_confidence=min_confidence,
            median_frames=median_frames,
            smoothing=smoothing,
            calibrate_seconds=calibrate_seconds,
            mode=mode,
            cancellation_aware=cancellation_aware,
            fine_tune_ms=fine_tune_ms,
            target_residual_corr=target_residual_corr,
        )
        self._estimator.delay_samples = self.manual_delay_samples

    @property
    def cancellation_score(self) -> float:
        return self._estimator.cancellation_score

    def observe_residual(self, after_echo: np.ndarray, aligned_ref: np.ndarray) -> bool:
        return self._estimator.observe_residual(after_echo, aligned_ref)

    def set_tuning(self, *, delay_ms: float | None = None, mode: str | None = None) -> None:
        if delay_ms is not None:
            self.manual_delay_samples = max(
                0.0,
                min(float(self.sample_rate * delay_ms / 1000), float(self.max_delay_samples)),
            )
            self.delay_samples = self.manual_delay_samples
            self._estimator.delay_samples = self.manual_delay_samples
        if mode is not None:
            self.mode = mode
            self._estimator.mode = mode
            if mode == "manual":
                self.delay_samples = self.manual_delay_samples

    @property
    def delay_ms(self) -> float:
        return 1000.0 * self.delay_samples / self.sample_rate

    def _push_reference(self, reference: np.ndarray) -> None:
        self.history.append(np.asarray(reference, dtype=np.float32).copy())
        total = sum(len(frame) for frame in self.history)
        while self.history and total > self.max_history_samples:
            total -= len(self.history[0])
            self.history.popleft()

    def _read_delayed(self, delay_samples: float) -> np.ndarray | None:
        if not self.history:
            return None
        concat = np.concatenate(list(self.history))
        n = self.frame_samples
        if len(concat) < n:
            return None
        max_start = len(concat) - n
        start = float(len(concat) - n) - float(delay_samples)
        if start < 0.0:
            start = 0.0
        if start > max_start:
            start = float(max_start)
        idx = start + np.arange(n, dtype=np.float64)
        return np.interp(idx, np.arange(len(concat)), concat).astype(np.float32)

    def process(self, mic: np.ndarray, reference: np.ndarray | None) -> np.ndarray | None:
        if reference is None or len(reference) != len(mic):
            return reference
        self._push_reference(reference)

        if self.mode == "manual":
            self.delay_samples = self.manual_delay_samples
            selected = self._read_delayed(self.delay_samples)
            if selected is None:
                return reference
            self.correlation = normalized_correlation(mic, selected)
            self.confidence = abs(self.correlation)
            return selected

        delay, corr, conf = self._estimator.update(mic, reference)
        self.delay_samples = delay
        self.correlation = corr
        self.confidence = conf
        selected = self._read_delayed(self.delay_samples)
        return selected if selected is not None else reference


# Back-compat alias for tests and docs that still refer to the old name.
ReferenceDelayAligner = SampleReferenceDelayAligner


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered)) + 1e-8
    return float(np.dot(a_centered, b_centered) / denom)


class SpectralNoiseSuppressor:
    """Spectral-subtraction suppressor with weighted overlap-add (WOLA).

    Each call ingests one hop (``frame_samples``) and emits one hop. Analysis and
    synthesis both use a sqrt-Hann window over a 2-hop frame; at 50% overlap the
    product of the two windows is a periodic Hann that sums to unity (the COLA
    condition), so a signal passed through unchanged is reconstructed exactly --
    no per-frame amplitude fade / 50 Hz warble. This adds one hop (frame_ms) of
    latency, the standard cost of overlap-add.
    """

    def __init__(self, frame_samples: int, noise_learn_frames: int, mode: str, reduction: float | None, floor: float | None) -> None:
        self.hop = int(frame_samples)
        self.win_len = 2 * self.hop
        # Periodic Hann (not np.hanning's symmetric one) so sqrt-Hann pairs are COLA.
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.win_len) / self.win_len)
        self.window = np.sqrt(hann).astype(np.float32)
        self.in_buffer = np.zeros(self.win_len, dtype=np.float32)
        self.ola = np.zeros(self.win_len, dtype=np.float32)
        self.noise_mag = np.zeros(self.win_len // 2 + 1, dtype=np.float32)
        self.learn_frames = max(1, noise_learn_frames)
        self.frames = 0
        self.reduction = float(reduction if reduction is not None else (1.15 if mode == "realtime" else 1.65))
        self.floor = float(floor if floor is not None else (0.08 if mode == "realtime" else 0.035))

    def process(self, frame: np.ndarray, speech_likely: bool) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        # Slide the new hop into the 2-hop analysis buffer.
        self.in_buffer = np.concatenate([self.in_buffer[self.hop :], frame])
        spectrum = np.fft.rfft(self.in_buffer * self.window)
        mag = np.abs(spectrum).astype(np.float32)
        phase = np.exp(1j * np.angle(spectrum))

        if self.frames < self.learn_frames:
            rate = 1.0 / float(self.frames + 1)
            self.noise_mag = (1.0 - rate) * self.noise_mag + rate * mag
        elif not speech_likely:
            self.noise_mag = 0.98 * self.noise_mag + 0.02 * mag
        self.frames += 1

        clean_mag = np.maximum(mag - self.reduction * self.noise_mag, self.floor * mag)
        cleaned = np.fft.irfft(clean_mag * phase, n=self.win_len).astype(np.float32)
        cleaned *= self.window  # synthesis window

        # Overlap-add: complete the oldest hop, emit it, then advance one hop.
        self.ola = self.ola + cleaned
        out = self.ola[: self.hop].copy()
        self.ola = np.concatenate([self.ola[self.hop :], np.zeros(self.hop, dtype=np.float32)])
        return np.clip(out, -1.0, 1.0)


class EnergyVad:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.noise_rms = 0.006
        self.speech_rms = 0.03

    def score(self, frame: np.ndarray, adapt: bool = True) -> float:
        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame))))) if len(frame) > 1 else 0.0
        snr = 20.0 * np.log10((rms + 1e-8) / (self.noise_rms + 1e-8))
        energy_score = 1.0 / (1.0 + np.exp(-(snr - 5.0) / 3.0))
        zcr_score = 1.0 - min(1.0, abs(zcr - 0.08) / 0.25)
        score = float(np.clip(0.82 * energy_score + 0.18 * zcr_score, 0.0, 1.0))
        # Adapt the noise/speech tracker exactly once per frame. The pre-suppression
        # call passes adapt=False (it only needs a read to gate the suppressor);
        # adapting on both calls would update the floor at ~2x rate on two different
        # versions of the frame and corrupt the estimate.
        if adapt:
            if score < self.threshold:
                self.noise_rms = 0.995 * self.noise_rms + 0.005 * rms
            else:
                self.speech_rms = 0.99 * self.speech_rms + 0.01 * rms
        return score


class VadGate:
    def __init__(self, pre_roll_frames: int, post_roll_frames: int, threshold: float, no_speech_mode: str, attenuation: float) -> None:
        self.delay_frames: deque[tuple[np.ndarray, bool]] = deque()
        self.pre_roll_frames = max(1, pre_roll_frames)
        self.post_roll_frames = max(1, post_roll_frames)
        self.post_remaining = 0
        self.threshold = threshold
        self.no_speech_mode = no_speech_mode
        self.attenuation = attenuation

    def process(self, frame: np.ndarray, score: float) -> tuple[np.ndarray, bool]:
        speech = score >= self.threshold
        if speech:
            self.post_remaining = self.post_roll_frames
            self.delay_frames = deque((past_frame, True) for past_frame, _ in self.delay_frames)
            pass_frame = True
        elif self.post_remaining > 0:
            self.post_remaining -= 1
            pass_frame = True
        else:
            pass_frame = False

        self.delay_frames.append((frame.copy(), pass_frame))
        if len(self.delay_frames) <= self.pre_roll_frames:
            return np.zeros_like(frame), False

        delayed_frame, delayed_pass = self.delay_frames.popleft()
        if delayed_pass:
            return delayed_frame, True

        if self.no_speech_mode == "attenuate":
            return delayed_frame * self.attenuation, False
        return np.zeros_like(delayed_frame), False


class ProcessingPipeline:
    def __init__(self, config: Config) -> None:
        sample_rate = int(config.input.sample_rate)
        frame_samples = int(sample_rate * config.input.frame_ms / 1000)
        noise_learn_frames = max(1, int(config.vad.noise_learn_ms / config.input.frame_ms))
        pre_roll_frames = max(1, int(config.vad.pre_roll_ms / config.input.frame_ms))
        post_roll_frames = max(1, int(config.vad.post_roll_ms / config.input.frame_ms))

        self.config = config
        self.highpass = OnePoleHighPass(85.0, sample_rate)
        self.reference_aligner = SampleReferenceDelayAligner(
            sample_rate,
            frame_samples,
            float(config.processing.reference_delay_ms),
            config.processing.reference_max_delay_ms,
            config.processing.reference_delay_smoothing,
            config.processing.reference_delay_mode,
            window_ms=float(config.processing.delay_window_ms),
            min_ref_rms=float(config.processing.delay_update_min_ref_rms),
            min_confidence=float(config.processing.delay_min_confidence),
            median_frames=int(config.processing.delay_median_frames),
            calibrate_seconds=float(config.processing.delay_calibrate_seconds),
            cancellation_aware=bool(config.processing.delay_cancellation_aware),
            fine_tune_ms=float(config.processing.delay_fine_tune_ms),
            target_residual_corr=float(config.processing.delay_target_residual_corr),
        )
        self.reference_matcher = ReferenceLevelMatcher(
            config.processing.reference_gain_min,
            config.processing.reference_gain_max,
            config.processing.reference_gain_smoothing,
            config.processing.reference_target_ratio,
        )
        self.echo = make_echo_canceller(config, frame_samples)
        self.vad = EnergyVad(float(config.vad.threshold))
        self.noise = SpectralNoiseSuppressor(
            frame_samples,
            noise_learn_frames,
            config.processing.mode,
            config.processing.noise_reduction,
            config.processing.spectral_floor,
        )
        self.gate = VadGate(
            pre_roll_frames,
            post_roll_frames,
            float(config.vad.threshold),
            config.processing.output_when_no_speech,
            float(config.processing.silence_attenuation),
        )
        self.mic_delay_samples = max(0, int(sample_rate * config.processing.mic_delay_ms / 1000))
        self.mic_delay_buffer = np.zeros(self.mic_delay_samples, dtype=np.float32)
        self.stats = ProcessingStats()
        self.last_stages: dict[str, np.ndarray] = {}
        self.echo_swap_status = f"active: {config.processing.echo_canceller}"

    def process(self, mic: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        frame = np.asarray(mic, dtype=np.float32)
        frame = np.nan_to_num(frame, copy=False)
        stages: dict[str, np.ndarray] = {"hardware_mic": frame.copy()}

        # Mic delay line: pushes the mic back in time so the jitter-buffered
        # reference leads the echo (keeps the adaptive canceller causal).
        if self.mic_delay_samples > 0:
            combined = np.concatenate([self.mic_delay_buffer, frame])
            frame = combined[: len(frame)].copy()
            self.mic_delay_buffer = combined[len(frame):]

        # mic_raw is the signal the canceller sees (post mic-delay, pre HPF).
        stages["mic_raw"] = frame.copy()
        if reference is not None:
            stages["reference"] = np.asarray(reference, dtype=np.float32).copy()

        aligned_reference = reference
        if self.config.processing.enable_reference_delay_align:
            aligned_reference = self.reference_aligner.process(frame, reference)
        if aligned_reference is not None:
            stages["reference_aligned"] = np.asarray(aligned_reference, dtype=np.float32).copy()

        matched_reference = aligned_reference
        if self.config.processing.enable_reference_level_match:
            matched_reference = self.reference_matcher.process(frame, aligned_reference)
        if matched_reference is not None:
            stages["reference_matched"] = np.asarray(matched_reference, dtype=np.float32).copy()

        if self.config.processing.enable_echo_cancellation:
            frame = self.echo.process(frame, matched_reference)
        stages["after_echo"] = frame.copy()

        if (
            self.config.processing.enable_reference_delay_align
            and self.config.processing.reference_delay_mode in ("auto", "calibrate")
            and matched_reference is not None
        ):
            if self.reference_aligner.observe_residual(stages["after_echo"], matched_reference):
                self.reset_echo_filter()

        if self.config.processing.enable_highpass:
            frame = self.highpass.process(frame)
        stages["after_highpass"] = frame.copy()

        pre_score = self.vad.score(frame, adapt=False)
        speech_likely = pre_score >= self.config.vad.threshold

        if self.config.processing.enable_noise_suppression or self.config.processing.enable_speech_enhancement:
            frame = self.noise.process(frame, speech_likely)
        stages["after_noise"] = frame.copy()

        score = self.vad.score(frame)
        if self.config.processing.enable_vad:
            frame, is_speech = self.gate.process(frame, score)
        else:
            is_speech = True

        clipped = np.clip(frame, -1.0, 1.0).astype(np.float32)
        stages["output"] = clipped.copy()
        self.last_stages = stages
        self.stats.frames += 1
        self.stats.speech_frames += int(is_speech)
        self.stats.vad_score = score
        self.stats.pre_vad_score = pre_score
        self.stats.noise_floor = self.vad.noise_rms
        self.stats.echo_gain = self.echo.gain
        self.stats.reference_gain = self.reference_matcher.gain
        if self.config.processing.enable_reference_delay_align:
            self.stats.reference_delay_ms = self.reference_aligner.delay_ms
            self.stats.reference_delay_correlation = self.reference_aligner.correlation
            self.stats.reference_delay_confidence = self.reference_aligner.confidence
            self.stats.reference_cancellation_score = self.reference_aligner.cancellation_score
        else:
            self.stats.reference_delay_ms = 0.0
            self.stats.reference_delay_correlation = 0.0
            self.stats.reference_delay_confidence = 0.0
            self.stats.reference_cancellation_score = 1.0
        after_echo = stages.get("after_echo")
        if after_echo is not None and matched_reference is not None and len(after_echo) == len(matched_reference):
            self.stats.residual_ref_correlation = normalized_correlation(after_echo, matched_reference)
        else:
            self.stats.residual_ref_correlation = 0.0
        self.stats.ref_present = reference is not None
        self.stats.clipped_output_pct = float(np.mean(np.abs(frame) >= 0.98) * 100.0)
        return clipped

    def apply_tuning(
        self,
        *,
        reference_delay_ms: float | None = None,
        reference_delay_mode: str | None = None,
        mic_delay_ms: float | None = None,
    ) -> dict[str, float | str]:
        applied: dict[str, float | str] = {}
        if reference_delay_ms is not None:
            self.config.processing.reference_delay_ms = int(round(reference_delay_ms))
            self.reference_aligner.set_tuning(delay_ms=float(reference_delay_ms))
            applied["reference_delay_ms"] = float(reference_delay_ms)
        if reference_delay_mode is not None:
            self.config.processing.reference_delay_mode = reference_delay_mode
            self.reference_aligner.set_tuning(mode=reference_delay_mode)
            applied["reference_delay_mode"] = reference_delay_mode
        if mic_delay_ms is not None:
            sample_rate = int(self.config.input.sample_rate)
            samples = max(0, int(sample_rate * float(mic_delay_ms) / 1000))
            self.config.processing.mic_delay_ms = int(round(mic_delay_ms))
            if samples != self.mic_delay_samples:
                self.mic_delay_samples = samples
                self.mic_delay_buffer = np.zeros(samples, dtype=np.float32)
            applied["mic_delay_ms"] = float(mic_delay_ms)
        return applied

    def set_echo_canceller(self, kind: str | None = None, mask_smooth: float | None = None) -> None:
        """Hot-swap the echo canceller (e.g. nlms -> hybrid) without restarting.

        Neural cancellers load ONNX/torch models (~1-2 s), so the new canceller is
        built on a background thread and swapped in atomically when ready; the
        realtime loop keeps using the current one meanwhile (no audio stall).
        """
        if kind is not None:
            self.config.processing.echo_canceller = kind
        if mask_smooth is not None:
            self.config.processing.dtln_mask_smoothing = float(mask_smooth)
        frame_samples = int(self.config.input.sample_rate * self.config.input.frame_ms / 1000)
        target = self.config.processing.echo_canceller
        self.echo_swap_status = f"loading {target}…"

        def _build() -> None:
            try:
                new_canceller = make_echo_canceller(self.config, frame_samples)
                self.echo = new_canceller  # atomic reference swap (GIL)
                self.echo_swap_status = f"active: {target}"
            except Exception as exc:  # noqa: BLE001
                self.echo_swap_status = f"error loading {target}: {exc}"

        threading.Thread(target=_build, name="echo-canceller-swap", daemon=True).start()

    def reset_echo_filter(self) -> None:
        reset = getattr(self.echo, "reset", None)
        if callable(reset):
            reset()
        else:
            self.echo.gain = 0.0
