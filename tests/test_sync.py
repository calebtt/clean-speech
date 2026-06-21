"""Tests for drift-compensated reference synchronisation (daemon-loop A5/A6).

These exercise the pure DriftCompensatingReference component with synthetic
producer/consumer schedules so the clock-sync behaviour is testable without audio
hardware:

  * no drift           -> faithful passthrough at constant latency
  * producer too fast  -> latency stays bounded (no runaway), no underruns
  * producer too slow  -> latency stays bounded, controller slows consumption
  * a hard stall       -> underruns are reported, then it recovers and re-primes
"""

from __future__ import annotations

import unittest

import numpy as np

from clean_speech_daemon.sync import DriftCompensatingReference


SAMPLE_RATE = 48_000
FRAME = 960


# A long, deterministic broadband stream the producer reads from, so that the
# emitted reference can be aligned back to a unique delay (a pure tone would
# correlate at every period and make the faithfulness check meaningless).
_SOURCE = np.cumsum(np.random.RandomState(11).randn(4_000_000).astype(np.float32))
_SOURCE = (_SOURCE - _SOURCE.mean()) / (np.abs(_SOURCE).max() + 1e-9) * 0.3


def source(n: int, start: int) -> np.ndarray:
    return _SOURCE[start : start + n].copy()


class DriftCompensatingReferenceTests(unittest.TestCase):
    def test_no_drift_is_faithful_passthrough(self) -> None:
        sync = DriftCompensatingReference(FRAME, target_latency_frames=3.0)
        produced = 0
        outputs = []
        for _ in range(400):
            sync.push(source(FRAME, produced))
            produced += FRAME
            out = sync.pull()
            if out is not None:
                outputs.append(out)

        self.assertEqual(sync.underruns, 0)
        # Latency is bounded and stable around the target (it oscillates by one
        # frame across the push/pull cycle, so the measured value sits a bit below
        # the 3-frame target rather than exactly on it).
        self.assertGreater(sync.latency_frames, 1.5)
        self.assertLess(sync.latency_frames, 3.5)

        # The emitted stream is the input delayed by the (constant) latency. Find
        # that integer delay and confirm the aligned streams match almost exactly.
        out = np.concatenate(outputs)
        produced_signal = _SOURCE[:produced]
        seg = slice(FRAME * 8, FRAME * 8 + FRAME * 40)
        best = max(
            float(np.corrcoef(out[seg], produced_signal[seg.start - lag : seg.stop - lag])[0, 1])
            for lag in range(0, 6 * FRAME)
            if seg.start - lag >= 0
        )
        self.assertGreater(best, 0.99)

    def test_fast_producer_does_not_accumulate_latency(self) -> None:
        # Monitor clock 0.3% fast: it pushes slightly more than one frame per pull.
        sync = DriftCompensatingReference(FRAME, target_latency_frames=3.0, drift_compensation=True)
        drift = 1.003
        carry = 0.0
        produced = 0
        latencies = []
        for _ in range(3000):
            carry += FRAME * drift
            push_n = int(carry)
            carry -= push_n
            sync.push(source(push_n, produced))
            produced += push_n
            sync.pull()
            latencies.append(sync.latency_frames)

        warm = latencies[500:]
        self.assertEqual(sync.underruns, 0)
        # Naive 1:1 pairing would grow latency by 0.3% * 3000 frames ~= 9 frames.
        # The controller holds it bounded near target instead.
        self.assertLess(max(warm), 6.0, "latency ran away despite drift compensation")
        self.assertGreater(min(warm), 1.0)
        self.assertGreater(sync.ratio, 1.0, "controller should consume faster than nominal")

    def test_slow_producer_stays_bounded(self) -> None:
        sync = DriftCompensatingReference(FRAME, target_latency_frames=3.0, drift_compensation=True)
        drift = 0.997  # monitor clock 0.3% slow
        carry = 0.0
        produced = 0
        latencies = []
        for _ in range(3000):
            carry += FRAME * drift
            push_n = int(carry)
            carry -= push_n
            sync.push(source(push_n, produced))
            produced += push_n
            out = sync.pull()
            if out is not None:
                latencies.append(sync.latency_frames)

        warm = latencies[500:]
        self.assertLess(max(warm), 6.0)
        self.assertLess(sync.ratio, 1.0, "controller should consume slower than nominal")

    def test_drift_is_tracked_and_alignment_stays_constant(self) -> None:
        # The monitor tap runs on the (drifting) sink clock; the sync layer must
        # resample it onto the mic clock and hold a *constant* latency, so the
        # canceller downstream sees a static echo path instead of a sliding one.
        sink_per_mic = 1.0003  # sink clock 300 ppm fast relative to mic
        sync = DriftCompensatingReference(FRAME, target_latency_frames=3.0, drift_compensation=True)
        carry = 0.0
        produced = 0
        latencies = []
        for _ in range(3000):
            carry += FRAME * sink_per_mic
            push_n = int(carry)
            carry -= push_n
            sync.push(source(push_n, produced))
            produced += push_n
            sync.pull()
            latencies.append(sync.available())

        self.assertEqual(sync.underruns, 0)
        # The resampling ratio converges to the true drift factor -- it is actively
        # consuming the reference at the sink rate to stay locked to the mic clock.
        self.assertAlmostEqual(sync.ratio, sink_per_mic, delta=5e-5)
        # And once converged, the held latency is constant (a naive 1:1 pairing
        # would instead grow latency by ~0.3% * 3000 frames ~= 9 frames).
        settled = np.array(latencies[1500:], dtype=np.float64)
        self.assertLess(settled.std(), 2.0, "alignment latency is not constant under drift")

    def test_default_is_fixed_delay_fifo_with_no_warping(self) -> None:
        # The default (drift_compensation=False) must NOT resample: ratio stays
        # exactly 1.0 and the emitted reference is the input verbatim at a fixed
        # delay, even under bursty delivery. Continuous resampling here is what
        # smeared the real-world echo correlation from ~0.3 down to ~0.0.
        sync = DriftCompensatingReference(FRAME, target_latency_frames=4.0)
        produced = 0
        outputs = []
        for i in range(500):
            burst = 2 if i % 2 == 0 else 0  # bursty (2,0,2,0...) but exactly balanced
            for _b in range(burst):
                sync.push(source(FRAME, produced)); produced += FRAME
            out = sync.pull()
            if out is not None:
                outputs.append(out)

        self.assertEqual(sync.ratio, 1.0, "default path resampled the reference (warps alignment)")
        self.assertEqual(sync.resyncs, 0)
        self.assertEqual(sync.underruns, 0)
        out = np.concatenate(outputs)
        # Output must be an exact, un-warped delayed copy: find the integer delay
        # and confirm a bit-for-bit match (resampling would make this impossible).
        seg = slice(FRAME * 20, FRAME * 20 + FRAME * 20)
        matched = False
        for lag in range(0, 8 * FRAME):
            if seg.start - lag < 0:
                continue
            if np.allclose(out[seg], _SOURCE[seg.start - lag : seg.stop - lag], atol=1e-6):
                matched = True
                break
        self.assertTrue(matched, "no fixed delay reproduces the reference exactly -> it was warped")

    def test_stall_reports_underrun_then_recovers(self) -> None:
        sync = DriftCompensatingReference(FRAME, target_latency_frames=3.0)
        produced = 0
        # Prime normally.
        for _ in range(20):
            sync.push(source(FRAME, produced))
            produced += FRAME
            sync.pull()
        self.assertTrue(sync.primed)

        # Producer stalls: keep pulling with nothing arriving.
        stall_outputs = [sync.pull() for _ in range(20)]
        self.assertTrue(any(o is None for o in stall_outputs))
        self.assertGreater(sync.underruns, 0)

        # Producer recovers: it re-primes and resumes emitting frames.
        for _ in range(40):
            sync.push(source(FRAME, produced))
            produced += FRAME
            sync.pull()
        self.assertIsNotNone(sync.pull())


if __name__ == "__main__":
    unittest.main()
