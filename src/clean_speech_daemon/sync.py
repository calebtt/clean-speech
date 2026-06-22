"""Drift-compensated reference synchronisation.

The microphone (PortAudio) and the system-audio monitor (``parec``) run on two
independent clocks. The original daemon loop paired "one mic frame" with
"whatever reference frame happened to be in the queue", reusing the last frame
for up to 350 ms on underrun. That makes the mic/reference offset jitter frame to
frame and lets latency grow without bound when the monitor produces samples
faster than the mic consumes them -- which defeats any echo canceller downstream,
because the echo path it is trying to model keeps sliding.

:class:`DriftCompensatingReference` replaces that with a continuous reference
sample buffer feeding a fractional resampler. A proportional controller nudges
the resampling ratio to hold the buffer fill (latency) at a fixed target: if the
producer runs fast the ratio rises slightly and consumes the backlog; if it runs
slow the ratio drops. The result is a reference stream locked to the mic clock at
constant latency, with sub-sample alignment, so the adaptive filter only has to
track the (static) room response rather than a continuously drifting delay.

This handles the *sample-rate* mismatch (clock drift). The gross acoustic/bulk
delay is still applied downstream by ReferenceDelayAligner, and the residual room
impulse response by the NLMS filter.
"""

from __future__ import annotations

import numpy as np


class DriftCompensatingReference:
    def __init__(
        self,
        frame_samples: int,
        target_latency_frames: float = 3.0,
        max_buffer_frames: int = 50,
        rate_correction: float = 0.05,
        max_rate_deviation: float = 0.005,
        resync_slack_frames: float = 6.0,
        backlog_smoothing: float = 0.99,
        drift_compensation: bool = False,
    ) -> None:
        # When False (the default for monitor-based AEC on a single audio graph,
        # e.g. PipeWire, where the mic and monitor are effectively synchronous),
        # the reference is NOT resampled at all -- it is emitted sample-for-sample
        # at a fixed delay. Any continuous resampling time-warps the reference and
        # destroys the fixed echo delay the canceller relies on. Slow real drift is
        # still bounded by the hard re-center below. Enable only for genuinely
        # independent capture clocks.
        self.drift_compensation = bool(drift_compensation)
        self.frame_samples = int(frame_samples)
        self.target_latency = float(target_latency_frames) * self.frame_samples
        self.max_buffer = int(max_buffer_frames) * self.frame_samples
        self.rate_correction = float(rate_correction)
        # Resampling authority. True hardware clock drift is well under 0.1%; the
        # resampler only needs to chase that. Capping it small is critical: a large
        # ratio continuously time-warps the reference and DESTROYS the fixed echo
        # delay the canceller depends on (the bug that made cancellation fail even
        # though the raw mic and monitor correlated at ~0.3). Bursty buffering is
        # handled by the hard re-center below, NOT by the resampler.
        self.max_rate_deviation = float(max_rate_deviation)
        # Drive the resampler from a slow average of the backlog, so it tracks real
        # drift and ignores the frame-to-frame jitter of bursty parec/PortAudio
        # delivery (which previously railed the ratio).
        self.backlog_smoothing = float(np.clip(backlog_smoothing, 0.0, 0.9999))
        self.avg_backlog = float(target_latency_frames) * self.frame_samples
        # The gentle resampler tracks steady clock drift, but real parec/PortAudio
        # delivery is bursty: a scheduling hiccup dumps several frames at once and
        # the 2%-capped ratio cannot drain that before the next burst, so latency
        # ratchets up and alignment slides. If the backlog drifts more than this
        # slack past target, snap it back by dropping the oldest samples -- one
        # brief glitch the adaptive filter re-converges through, versus a
        # permanently wrong (and wandering) echo delay.
        self.resync_slack = float(resync_slack_frames) * self.frame_samples
        self.resyncs = 0

        self.buffer = np.zeros(0, dtype=np.float32)
        self.read_pos = 0.0  # fractional read index into self.buffer
        self.ratio = 1.0     # input samples consumed per output sample
        self.primed = False

        # Diagnostics.
        self.samples_pushed = 0
        self.samples_pulled = 0
        self.underruns = 0
        self.overflow_drops = 0

    # -- producer side ---------------------------------------------------- #
    def push(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float32).ravel()
        if f.size == 0:
            return
        self.buffer = np.concatenate([self.buffer, f])
        self.samples_pushed += f.size
        # Hard cap on memory / latency: drop oldest samples if wildly over budget
        # (e.g. the consumer stalled). Keeps the controller in its linear range.
        excess = int(self.available() - self.max_buffer)
        if excess > 0:
            self.buffer = self.buffer[excess:]
            self.read_pos = max(0.0, self.read_pos - excess)
            self.overflow_drops += excess

    def available(self) -> float:
        return len(self.buffer) - self.read_pos

    @property
    def latency_frames(self) -> float:
        return self.available() / self.frame_samples

    def set_target_latency_frames(self, target_latency_frames: float) -> None:
        self.target_latency = float(target_latency_frames) * self.frame_samples
        self.avg_backlog = self.target_latency

    # -- consumer side ---------------------------------------------------- #
    def pull(self) -> np.ndarray | None:
        """Return one mic-clock-aligned reference frame, or None on underrun."""
        n = self.frame_samples

        # Prime: build up to the target latency before emitting, so the controller
        # has headroom in both directions.
        if not self.primed:
            if self.available() < self.target_latency + n:
                return None
            self.primed = True

        # Need at least n+1 samples spanned to interpolate the whole frame.
        if self.available() < n + 1:
            self.underruns += 1
            self.primed = False  # re-prime once the producer recovers
            return None

        # Hard re-center: if a burst pushed the backlog well past target, drop the
        # oldest excess so latency (and thus the echo delay) stays bounded.
        excess = int(self.available() - (self.target_latency + self.resync_slack))
        if excess > 0:
            self.buffer = self.buffer[excess:]
            self.read_pos = max(0.0, self.read_pos - excess)
            self.resyncs += 1

        if self.drift_compensation:
            # Proportional controller on the SMOOTHED backlog: tracks slow drift,
            # not per-frame jitter, so the ratio stays within a small band.
            self.avg_backlog = self.backlog_smoothing * self.avg_backlog + (1.0 - self.backlog_smoothing) * self.available()
            error = (self.avg_backlog - self.target_latency) / self.target_latency
            ratio = 1.0 + self.rate_correction * error
            ratio = float(np.clip(ratio, 1.0 - self.max_rate_deviation, 1.0 + self.max_rate_deviation))
        else:
            # Fixed-delay FIFO: no resampling, so the reference keeps its exact
            # alignment with the mic. read_pos stays integer -> np.interp returns
            # the original samples verbatim.
            ratio = 1.0

        # Don't let the last interpolation index run past the buffer.
        max_index = len(self.buffer) - 1
        if self.read_pos + ratio * (n - 1) > max_index:
            ratio = (max_index - self.read_pos) / (n - 1)
        self.ratio = ratio

        idx = self.read_pos + ratio * np.arange(n)
        out = np.interp(idx, np.arange(len(self.buffer)), self.buffer).astype(np.float32)
        self.read_pos += ratio * n
        self.samples_pulled += n

        # Trim fully-consumed samples; keep the fractional remainder.
        consumed = int(self.read_pos)
        if consumed > 0:
            self.buffer = self.buffer[consumed:]
            self.read_pos -= consumed
        return out
