"""Delay estimation for reference/microphone alignment."""

from __future__ import annotations

from collections import deque

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n - 1)).bit_length()


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def frame_lag_estimate(
    mic_frame: np.ndarray,
    ref_frame: np.ndarray,
    frame_samples: int,
    max_lag_samples: float,
    search_step: int | None = None,
) -> tuple[float, float, float]:
    """Estimate positive reference delay from paired mic/ref frames.

    Echo in *mic_frame* trails the monitor tap by ``lag`` samples, so the best
    match is ``mic[lag:]`` vs ``ref[:n-lag]``. Returns
    ``(delay_samples, correlation, confidence)``.
    """
    mic = np.asarray(mic_frame, dtype=np.float64).ravel()
    ref = np.asarray(ref_frame, dtype=np.float64).ravel()
    n = int(frame_samples)
    if len(mic) != n or len(ref) != n:
        return 0.0, 0.0, 0.0

    max_lag = min(int(max_lag_samples), n - 1)
    step = max(1, int(search_step if search_step is not None else max(1, n // 40)))

    best_lag = 0.0
    best_corr = -1.0
    second_corr = -1.0
    for lag in range(0, max_lag + 1, step):
        if lag == 0:
            corr = _normalized_correlation(mic, ref)
        else:
            corr = _normalized_correlation(mic[lag:], ref[: n - lag])
        if corr > best_corr:
            second_corr = best_corr
            best_corr = corr
            best_lag = float(lag)
        elif corr > second_corr:
            second_corr = corr

    if best_corr < 0.0:
        return best_lag, best_corr, 0.0

    if best_lag > 0.0 and step > 1:
        left = max(0, int(best_lag) - step)
        right = min(max_lag, int(best_lag) + step)
        refine_lag = best_lag
        refine_corr = best_corr
        for lag in range(left, right + 1):
            if lag == 0:
                corr = _normalized_correlation(mic, ref)
            else:
                corr = _normalized_correlation(mic[lag:], ref[: n - lag])
            if corr > refine_corr:
                refine_corr = corr
                refine_lag = float(lag)
        best_lag = refine_lag
        best_corr = refine_corr

    ratio = best_corr / (max(second_corr, 0.0) + 1e-12)
    confidence = float(np.clip(min(best_corr, 1.0), 0.0, 1.0) * np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
    if best_corr > 0.2:
        confidence = max(confidence, float(np.clip((best_corr - 0.15) / 0.5, 0.0, 1.0)))
    return best_lag, best_corr, confidence


def cross_frame_lag_estimate(
    mic_frame: np.ndarray,
    ref_history: np.ndarray,
    frame_samples: int,
    max_lag_samples: float,
    search_step: int | None = None,
) -> tuple[float, float, float]:
    """Estimate delay using the current mic frame and prior reference samples."""
    mic = np.asarray(mic_frame, dtype=np.float64).ravel()
    ref = np.asarray(ref_history, dtype=np.float64).ravel()
    n = int(frame_samples)
    if len(mic) != n or len(ref) < n:
        return 0.0, 0.0, 0.0

    max_lag = int(max_lag_samples)
    step = max(1, int(search_step if search_step is not None else max(1, n // 40)))
    end = len(ref)

    best_lag = 0.0
    best_corr = -1.0
    second_corr = -1.0
    for lag in range(0, max_lag + 1, step):
        start = end - n - lag
        if start < 0:
            break
        corr = _normalized_correlation(mic, ref[start : start + n])
        if corr > best_corr:
            second_corr = best_corr
            best_corr = corr
            best_lag = float(lag)
        elif corr > second_corr:
            second_corr = corr

    if best_corr < 0.0:
        return best_lag, best_corr, 0.0

    if best_lag > 0.0 and step > 1:
        left = max(0, int(best_lag) - step)
        right = min(max_lag, int(best_lag) + step)
        refine_lag = best_lag
        refine_corr = best_corr
        for lag in range(left, right + 1):
            start = end - n - lag
            if start < 0:
                continue
            corr = _normalized_correlation(mic, ref[start : start + n])
            if corr > refine_corr:
                refine_corr = corr
                refine_lag = float(lag)
        best_lag = refine_lag
        best_corr = refine_corr

    ratio = best_corr / (max(second_corr, 0.0) + 1e-12)
    confidence = float(np.clip(min(best_corr, 1.0), 0.0, 1.0) * np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
    if best_corr > 0.2:
        confidence = max(confidence, float(np.clip((best_corr - 0.15) / 0.5, 0.0, 1.0)))
    return best_lag, best_corr, confidence


def read_delayed_frame(ref_concat: np.ndarray, frame_samples: int, delay_samples: float) -> np.ndarray | None:
    """Read one delayed reference frame from a concatenated history buffer."""
    concat = np.asarray(ref_concat, dtype=np.float64).ravel()
    n = int(frame_samples)
    if len(concat) < n:
        return None
    max_start = len(concat) - n
    start = float(len(concat) - n) - float(delay_samples)
    start = max(0.0, min(float(max_start), start))
    idx = start + np.arange(n, dtype=np.float64)
    return np.interp(idx, np.arange(len(concat)), concat).astype(np.float32)


def cancellation_residual_score(mic: np.ndarray, delayed_ref: np.ndarray) -> float:
    """Score how much reference remains after optimal scalar subtraction.

    Lower is better. Uses the same gain model as the legacy scalar reducer.
    """
    mic_v = np.asarray(mic, dtype=np.float64).ravel()
    ref_v = np.asarray(delayed_ref, dtype=np.float64).ravel()
    if len(mic_v) != len(ref_v) or len(mic_v) == 0:
        return 1.0
    denom = float(np.dot(ref_v, ref_v)) + 1e-12
    gain = float(np.dot(mic_v, ref_v)) / denom
    residual = mic_v - gain * ref_v
    return abs(_normalized_correlation(residual, ref_v))


def cancellation_aware_fine_tune(
    mic: np.ndarray,
    ref_concat: np.ndarray,
    frame_samples: int,
    center_delay_samples: float,
    max_delay_samples: float,
    fine_tune_ms: float,
    sample_rate: int,
) -> tuple[float, float]:
    """Search around *center_delay_samples* for the delay with lowest residual score."""
    fine_range = int(sample_rate * fine_tune_ms / 1000)
    center = int(round(center_delay_samples))
    fine_step = max(1, sample_rate // 4000)
    best_delay = float(center_delay_samples)
    best_score = float("inf")
    for offset in range(-fine_range, fine_range + 1, fine_step):
        delay = float(center + offset)
        if delay < 0.0 or delay > max_delay_samples:
            continue
        delayed_ref = read_delayed_frame(ref_concat, frame_samples, delay)
        if delayed_ref is None:
            continue
        score = cancellation_residual_score(mic, delayed_ref)
        if score < best_score:
            best_score = score
            best_delay = delay
    if not np.isfinite(best_score):
        return float(center_delay_samples), 1.0
    return best_delay, best_score


def hybrid_lag_estimate(
    mic_frame: np.ndarray,
    ref_frame: np.ndarray,
    ref_history: np.ndarray,
    frame_samples: int,
    max_lag_samples: float,
    search_step: int | None = None,
) -> tuple[float, float, float]:
    """Prefer same-frame lag for sub-frame delays; cross-frame when it wins clearly."""
    same_lag, same_corr, same_conf = frame_lag_estimate(
        mic_frame, ref_frame, frame_samples, max_lag_samples, search_step=search_step
    )
    cross_lag, cross_corr, cross_conf = cross_frame_lag_estimate(
        mic_frame, ref_history, frame_samples, max_lag_samples, search_step=search_step
    )
    if cross_lag > frame_samples and cross_corr > same_corr * 0.85:
        return cross_lag, cross_corr, cross_conf
    if same_corr >= cross_corr * 0.9:
        return same_lag, same_corr, same_conf
    return cross_lag, cross_corr, cross_conf


def gcc_phat_estimate_delay(
    mic: np.ndarray,
    ref: np.ndarray,
    sample_rate: int,
    max_lag_ms: float = 500.0,
    min_lag_ms: float = 0.0,
) -> tuple[float, float, float]:
    """Offline GCC-PHAT delay estimate for saved-stream analysis."""
    mic = np.asarray(mic, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(mic), len(ref))
    if n < 64:
        return 0.0, 0.0, 0.0

    mic = mic[:n] - float(np.mean(mic[:n]))
    ref = ref[:n] - float(np.mean(ref[:n]))

    fft_n = _next_pow2(2 * n)
    spec_mic = np.fft.rfft(mic, n=fft_n)
    spec_ref = np.fft.rfft(ref, n=fft_n)
    cross = spec_mic * np.conj(spec_ref)
    denom = np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross / denom, n=fft_n)

    min_lag = max(0, int(sample_rate * min_lag_ms / 1000))
    max_lag = min(n - 1, int(sample_rate * max_lag_ms / 1000))
    if max_lag <= min_lag:
        return 0.0, 0.0, 0.0

    search = np.abs(cc[min_lag : max_lag + 1])
    if search.size == 0:
        return 0.0, 0.0, 0.0

    peak_idx = int(np.argmax(search))
    peak_f = float(min_lag + peak_idx)
    if 0 < peak_idx < len(search) - 1:
        y0, y1, y2 = float(search[peak_idx - 1]), float(search[peak_idx]), float(search[peak_idx + 1])
        denom_parab = y0 - 2.0 * y1 + y2
        if abs(denom_parab) > 1e-12:
            peak_f = min_lag + peak_idx + 0.5 * (y0 - y2) / denom_parab

    lag_i = int(peak_f)
    frac = peak_f - lag_i
    if lag_i < n:
        mic_seg = mic[lag_i:]
        ref_seg = ref[: n - lag_i]
        if len(mic_seg) == len(ref_seg) and len(mic_seg) > 0:
            corr = _normalized_correlation(mic_seg, ref_seg)
        else:
            corr = 0.0
    else:
        corr = 0.0

    peak_val = float(search[peak_idx])
    if search.size > 1:
        runner_up = float(np.partition(search, -2)[-2])
        ratio = peak_val / (runner_up + 1e-12)
        confidence = float(np.clip((ratio - 1.0) / 4.0, 0.0, 1.0))
    else:
        confidence = 0.0

    return peak_f, corr, confidence


def median_frame_lag_estimate(
    mic: np.ndarray,
    ref: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    max_lag_ms: float = 120.0,
    ref_energy_percentile: float = 70.0,
) -> tuple[float, float, float]:
    """Median per-frame lag on ref-active segments (offline reports)."""
    hop = frame_samples
    nframes = min(len(mic), len(ref)) // hop
    if nframes < 3:
        return 0.0, 0.0, 0.0

    energies = [float(np.mean(ref[i * hop : (i + 1) * hop] ** 2)) for i in range(nframes)]
    threshold = float(np.percentile(energies, ref_energy_percentile))
    lags: list[float] = []
    corrs: list[float] = []
    max_lag_samples = float(sample_rate * max_lag_ms / 1000)
    for index in range(nframes):
        if energies[index] < threshold:
            continue
        mic_frame = mic[index * hop : (index + 1) * hop]
        ref_frame = ref[index * hop : (index + 1) * hop]
        ref_prefix = ref[: (index + 1) * hop]
        lag, corr, _ = hybrid_lag_estimate(
            mic_frame,
            ref_frame,
            ref_prefix,
            hop,
            max_lag_samples,
        )
        if corr > 0.05:
            lags.append(lag)
            corrs.append(corr)

    if not lags:
        return 0.0, 0.0, 0.0

    median_lag = float(np.median(lags))
    median_corr = float(np.median(corrs))
    spread = float(np.std(lags))
    confidence = float(np.clip(median_corr, 0.0, 1.0) * np.clip(1.0 - spread / max(1.0, sample_rate * 0.01), 0.0, 1.0))
    return median_lag, median_corr, confidence


class GccPhatDelayEstimator:
    """Rolling delay tracker with correlation + cancellation-aware fine tuning."""

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        max_delay_ms: float,
        window_ms: float = 300.0,
        min_ref_rms: float = 0.003,
        min_confidence: float = 0.15,
        min_correlation: float = 0.08,
        median_frames: int = 15,
        smoothing: float = 0.85,
        calibrate_seconds: float = 10.0,
        mode: str = "auto",
        cancellation_aware: bool = True,
        fine_tune_ms: float = 15.0,
        target_residual_corr: float = 0.15,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(frame_samples)
        self.max_delay_samples = max(0.0, float(self.sample_rate * max_delay_ms / 1000))
        self.window_samples = max(self.frame_samples * 4, int(self.sample_rate * window_ms / 1000))
        self.min_ref_rms = float(min_ref_rms)
        self.min_confidence = float(min_confidence)
        self.min_correlation = float(min_correlation)
        self.median_frames = max(1, int(median_frames))
        self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
        self.calibrate_seconds = float(calibrate_seconds)
        self.mode = mode
        self.cancellation_aware = bool(cancellation_aware)
        self.fine_tune_ms = float(fine_tune_ms)
        self.target_residual_corr = float(target_residual_corr)
        self._active_fine_tune_ms = self.fine_tune_ms
        self._search_step = max(1, self.sample_rate // 2000)
        self._delay_reset_threshold_samples = max(1.0, float(sample_rate) / 500.0)

        history_cap = int(self.window_samples + self.max_delay_samples + self.frame_samples)
        self._mic_history: deque[np.ndarray] = deque()
        self._ref_history: deque[np.ndarray] = deque()
        self._history_cap = history_cap
        self._recent_estimates: deque[float] = deque(maxlen=self.median_frames)

        self.delay_samples = 0.0
        self.correlation = 0.0
        self.confidence = 0.0
        self.cancellation_score = 1.0
        self.live_residual_corr = 1.0
        self.delay_changed = False
        self.frames_seen = 0
        self._calibrate_frames = max(1, int(self.calibrate_seconds * sample_rate / frame_samples))
        self._frozen = False

    @property
    def delay_ms(self) -> float:
        return 1000.0 * self.delay_samples / self.sample_rate

    def _append_history(self, mic: np.ndarray, ref: np.ndarray) -> None:
        self._mic_history.append(np.asarray(mic, dtype=np.float32).copy())
        self._ref_history.append(np.asarray(ref, dtype=np.float32).copy())
        mic_total = sum(len(x) for x in self._mic_history)
        while self._mic_history and mic_total > self._history_cap:
            mic_total -= len(self._mic_history[0])
            self._mic_history.popleft()
            self._ref_history.popleft()

    def update(self, mic: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
        """Feed one frame; return ``(delay_samples, correlation, confidence)``."""
        self.frames_seen += 1
        self._append_history(mic, ref)

        if self.mode == "manual":
            return self.delay_samples, self.correlation, self.confidence

        if self.mode == "calibrate" and self._frozen:
            return self.delay_samples, self.correlation, self.confidence

        ref_rms = float(np.sqrt(np.mean(ref * ref)) + 1e-12)
        if ref_rms < self.min_ref_rms:
            return self.delay_samples, self.correlation, self.confidence

        ref_concat = np.concatenate(list(self._ref_history))
        lag, corr, conf = hybrid_lag_estimate(
            mic,
            ref,
            ref_concat,
            self.frame_samples,
            self.max_delay_samples,
            search_step=self._search_step,
        )
        if conf < self.min_confidence or corr < self.min_correlation:
            return self.delay_samples, self.correlation, self.confidence

        self._recent_estimates.append(lag)
        if len(self._recent_estimates) < max(3, self.median_frames // 3):
            return self.delay_samples, self.correlation, self.confidence

        coarse_target = float(np.median(self._recent_estimates))
        target = coarse_target
        cancel_score = self.cancellation_score
        if self.cancellation_aware and self.mode in ("auto", "calibrate"):
            mic_win = np.concatenate(list(self._mic_history)[-min(5, len(self._mic_history)) :])
            if len(mic_win) >= self.frame_samples:
                if self.live_residual_corr > self.target_residual_corr and self.frames_seen > self.median_frames:
                    center = 0.5 * coarse_target + 0.5 * self.delay_samples
                else:
                    center = coarse_target
                tuned_delay, cancel_score = cancellation_aware_fine_tune(
                    mic_win[-self.frame_samples :],
                    ref_concat,
                    self.frame_samples,
                    center,
                    self.max_delay_samples,
                    self._active_fine_tune_ms,
                    self.sample_rate,
                )
                if cancel_score < self.cancellation_score or self.frames_seen <= self.median_frames:
                    target = tuned_delay
                self.cancellation_score = cancel_score

        if self.live_residual_corr > self.target_residual_corr:
            smooth = min(self.smoothing, 0.35)
        elif self.live_residual_corr < self.target_residual_corr * 0.6:
            smooth = self.smoothing
        elif self.mode == "calibrate":
            smooth = 0.5 if self.frames_seen < self._calibrate_frames else self.smoothing
        else:
            smooth = min(self.smoothing, 0.55)

        old_delay = self.delay_samples
        self.delay_samples = float(np.clip(smooth * self.delay_samples + (1.0 - smooth) * target, 0.0, self.max_delay_samples))
        if abs(self.delay_samples - old_delay) > self._delay_reset_threshold_samples:
            self.delay_changed = True
        self.correlation = corr
        self.confidence = conf
        if cancel_score < self.target_residual_corr:
            self.confidence = max(conf, float(np.clip(1.0 - cancel_score, 0.0, 1.0)))

        if self.mode == "calibrate" and self.frames_seen >= self._calibrate_frames:
            self._frozen = True

        return self.delay_samples, self.correlation, self.confidence

    def observe_residual(self, after_echo: np.ndarray, aligned_ref: np.ndarray) -> bool:
        """Track live AEC residual; widen search when reference is still audible."""
        if len(after_echo) != len(aligned_ref) or len(after_echo) == 0:
            return False
        self.live_residual_corr = abs(_normalized_correlation(after_echo, aligned_ref))
        if self.mode not in ("auto", "calibrate"):
            return False
        if self.live_residual_corr > self.target_residual_corr:
            self._active_fine_tune_ms = min(self.fine_tune_ms * 2.0, self._active_fine_tune_ms + 0.25)
        else:
            self._active_fine_tune_ms = max(self.fine_tune_ms, self._active_fine_tune_ms * 0.995)
        return self.consume_delay_changed()

    def consume_delay_changed(self) -> bool:
        changed = self.delay_changed
        self.delay_changed = False
        return changed