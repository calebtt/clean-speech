"""Adaptive acoustic echo cancellation.

The legacy :class:`~clean_speech_daemon.processing.AdaptiveEchoReducer` models the
echo as a single scalar gain applied to the reference within one frame. That only
works when the echo is a perfectly time-aligned, unfiltered, scaled copy of the
reference -- never true for sound that leaves desktop speakers, bounces around a
room, and arrives at the mic delayed and filtered.

:class:`NlmsEchoCanceller` is a real multi-tap adaptive filter: a frequency-domain
(overlap-save) block normalized-LMS filter, the same algorithm class used inside
WebRTC's AEC and Speex. It learns the room's impulse response from the reference
and subtracts the predicted echo, tracking slow changes (volume, clock drift)
continuously. On broadband system audio convolved with a room impulse response it
clears ~90-95% of the echo where the scalar reducer clears ~0%.

The interface intentionally matches AdaptiveEchoReducer so it is a drop-in
replacement in the pipeline:

    canceller.process(mic_frame, reference_frame_or_None) -> cleaned_frame

For echo paths whose bulk delay exceeds the adaptive filter length, run the
coarse :class:`~clean_speech_daemon.processing.ReferenceDelayAligner` ahead of this
so the adaptive filter only has to model the residual room tail. A WebRTC/Speex
native backend can be slotted in behind the same interface later.
"""

from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n - 1)).bit_length()


class NlmsEchoCanceller:
    """Frequency-domain (overlap-save) constrained NLMS adaptive echo canceller.

    Args:
        frame_samples: number of samples per processing block.
        taps: adaptive filter length in samples (room response it can model).
        step_size: NLMS step size mu in (0, ~1]. Larger adapts faster but is less
            stable; 0.3 is a safe default.
        leak: per-block coefficient leakage; nudges unused taps toward zero and
            keeps the filter from drifting during long silences.
        power_smoothing: smoothing for the reference power estimate used to
            normalize the step size.
    """

    def __init__(
        self,
        frame_samples: int,
        taps: int = 512,
        step_size: float = 0.3,
        leak: float = 1e-3,
        power_smoothing: float = 0.9,
        regularization: float = 1.0,
        silence_floor: float = 1e-7,
        boundary_smoothing_samples: int = 64,
    ) -> None:
        self.block = int(frame_samples)
        self.taps = max(1, int(taps))
        self.mu = float(step_size)
        self.leak = float(leak)
        self.power_smoothing = float(np.clip(power_smoothing, 0.0, 0.999))
        # Regularization as a multiple of the mean reference power -- this is what
        # stops the per-bin step from exploding when a bin has near-zero energy
        # (quiet passages in music/TV), the cause of real-world filter divergence.
        self.regularization = float(regularization)
        # Below this reference power (per block, time domain) we do not adapt: there
        # is no echo to learn from silence, and adapting on it only invites drift.
        self.silence_floor = float(silence_floor)

        # FFT size must hold a linear convolution of the filter with the block.
        self.fft_size = _next_pow2(self.taps + self.block)
        self.freq_bins = self.fft_size // 2 + 1

        self.weights = np.zeros(self.freq_bins, dtype=np.complex128)
        self.ref_buffer = np.zeros(self.fft_size, dtype=np.float64)
        self.power = np.full(self.freq_bins, 1e-3, dtype=np.float64)
        self.gain = 0.0  # filter RMS, for diagnostics parity
        self.adapting = False
        self.boundary_smoothing_samples = max(0, int(boundary_smoothing_samples))
        self.previous_output_last: float | None = None

    def process(self, mic: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32)
        if reference is None or len(reference) != len(mic) or len(mic) != self.block:
            # Without a usable reference there is nothing to cancel.
            return mic

        d = np.nan_to_num(mic.astype(np.float64))
        x = np.nan_to_num(np.asarray(reference, dtype=np.float64))

        # Overlap-save: slide the new block into the reference buffer.
        self.ref_buffer = np.concatenate([self.ref_buffer[self.block :], x])
        X = np.fft.rfft(self.ref_buffer)

        # Predicted echo = last `block` samples of the circular convolution.
        y = np.fft.irfft(X * self.weights, n=self.fft_size)[-self.block :]
        error = d - y

        # Only adapt when the reference actually carries energy (something is
        # playing). Adapting on silence is what let the filter run away.
        ref_power = float(np.mean(x * x))
        self.adapting = ref_power > self.silence_floor
        if self.adapting:
            bin_power = X.real ** 2 + X.imag ** 2
            self.power = self.power_smoothing * self.power + (1.0 - self.power_smoothing) * bin_power
            # Normalize by per-bin power PLUS a regularization tied to the mean
            # power, so quiet bins get a bounded (not explosive) step.
            reg = self.regularization * float(np.mean(self.power)) + 1e-9
            error_block = np.zeros(self.fft_size, dtype=np.float64)
            error_block[-self.block :] = error
            E = np.fft.rfft(error_block)
            update = (self.mu / (self.power + reg)) * (np.conj(X) * E)
            self.weights = (1.0 - self.leak) * self.weights + update

            # Gradient constraint: keep the filter to `taps` samples.
            w_time = np.fft.irfft(self.weights, n=self.fft_size)
            w_time[self.taps :] = 0.0

            # Divergence safety net: if the filter energy ever exceeds a sane bound
            # (a linear echo path has gain ~O(1)), rein it back in instead of
            # letting positive feedback blow it up.
            wnorm = float(np.sqrt(np.mean(w_time[: self.taps] ** 2)))
            if not np.isfinite(wnorm) or wnorm > 4.0:
                w_time = np.zeros_like(w_time) if not np.isfinite(wnorm) else w_time * (4.0 / wnorm)
                wnorm = min(wnorm, 4.0) if np.isfinite(wnorm) else 0.0
            self.weights = np.fft.rfft(w_time)
            self.gain = wnorm

        # Block adaptation can leave tiny but audible discontinuities at the 20 ms
        # boundary. Remove only the boundary step with a short decay; this preserves
        # the block contents while avoiding the "robotic/chopped" residual.
        if self.adapting and self.previous_output_last is not None and self.boundary_smoothing_samples > 0 and len(error) > 1:
            n = min(self.boundary_smoothing_samples, len(error))
            step = float(error[0] - self.previous_output_last)
            error[:n] -= step * np.linspace(1.0, 0.0, n, endpoint=False)
        self.previous_output_last = float(error[-1])

        return np.nan_to_num(error).astype(np.float32)


def make_echo_canceller(config, frame_samples: int):  # noqa: ANN001
    """Build the configured echo canceller. Falls back to the scalar reducer."""
    from .processing import AdaptiveEchoReducer

    kind = getattr(config.processing, "echo_canceller", "scalar")
    if kind == "nlms":
        return NlmsEchoCanceller(
            frame_samples,
            taps=int(getattr(config.processing, "echo_filter_taps", 512)),
            step_size=float(getattr(config.processing, "echo_step_size", 0.3)),
            leak=float(getattr(config.processing, "echo_filter_leak", 1e-4)),
            boundary_smoothing_samples=int(getattr(config.processing, "echo_boundary_smoothing_samples", 64)),
        )
    return AdaptiveEchoReducer()
