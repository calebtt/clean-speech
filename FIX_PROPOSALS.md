# Echo Cancellation Bugfix Notes

Observed problem: unit tests pass, but live recordings still contain audible system audio in `cleaned_output`, which also sounds robotic/choppy.

Latest analyzed session: `20260621-093440` (~09:34 local, 12 s saved from testbed).

## What the last run showed

| Metric | Value | Meaning |
|--------|-------|---------|
| `after_echo` vs `mic_raw` correlation | 0.49 | AEC barely changed the signal |
| `after_echo` RMS vs `mic_raw` | 1.80× (+5.1 dB) | Echo stage amplified instead of cancelling |
| Echo reduction | −5.1 dB | Worse than passthrough |
| `after_echo` boundary jump ratio | 46.9× (20 ms frames) | Severe block-edge steps |
| GCC-PHAT recommended delay | ~19 ms | Auto delay often far off during run |
| `reference_aligned` offset vs mic | +380 ms | Reference badly mis-timed to mic |

Journal/diagnostics during that window:

- `echo_gain` stuck near zero (often frozen at `0.0093`); NLMS not building a useful model.
- Brief divergence spikes (`echo_gain` up to 0.37) then collapse; `after_echo` louder than mic at those times.
- `ref_corr = 0.0` in 142/660 diagnostic samples — delay estimator blind.
- VAD passing almost everything (`speech_ratio` ~98%, `vad` ~0.96) while reference plays.
- Many `output_rms = -240` silence bursts mixed with loud gated passages.
- Reference sync degraded: **79 underruns**, **1,656 resyncs**, `latency_frames` crept to **6.0** (config target **1.0** ≈ 120 ms buffer vs 20 ms target).
- Delay wandered: 18–20 ms → 120 ms → 360–380 ms → 38.6 ms with `ref_corr = 0.57` but cancel score ≈ 0.

`cleaned_output` is a poor stream for judging AEC alone: 300 ms VAD pre-roll plus gating on top of a broken `after_echo` stage.

## Current hypotheses (replace prior root-cause list)

### 1. Reference sync instability is the primary blocker

`DriftCompensatingReference` is not holding `reference_sync_latency_frames = 1.0`. Underruns (79) and constant resyncs (1,656) jitter the mic/reference pairing frame-to-frame. NLMS cannot learn a stable echo path when the reference timeline slides underneath it.

**When revisiting:** investigate parec delivery, resync triggers, and enforcement of target latency; confirm underrun count drops before tuning NLMS.

### 2. Auto delay alignment locks wrong values

GCC-PHAT on saved audio recommends ~19 ms, but live auto mode wandered to 120 ms, 360–380 ms, and 38.6 ms. Cancellation-aware fine-tune reported success (`cancel score ≈ 0`) while `ref_corr = 0.0` — optimizing against silence or mis-paired frames, not real echo.

**When revisiting:** do not update delay when `ref_corr = 0` or reference RMS is below threshold; hold last good delay; clamp search band around GCC-PHAT / cross-frame estimate (~15–25 ms); ignore cancellation score when reference is not present.

### 3. NLMS is not cancelling (and sometimes diverges)

`echo_gain ≈ 0` for most of the session while `after_echo ≈ mic_raw`. Occasional gain spikes (0.12–0.37) correlate with louder, warped output. Filter resets on delay jumps may be preventing adaptation from converging.

**When revisiting:** judge on `after_echo` stream, not `cleaned_output`; consider smaller step size after warmup, scalar pre-subtract once delay is stable, or alternative AEC backend; log when `after_echo` RMS exceeds `mic_raw` RMS as a divergence alarm.

### 4. Block-edge artifacts cause the robotic sound

`after_echo` boundary jump ratio was 46.9× live (5.3× in alignment report). Existing `echo_boundary_smoothing_samples = 64` is insufficient when the filter is misaligned or diverging. VAD silence bursts (`output_rms = -240`) add choppy gating on top.

**When revisiting:** increase or fix boundary smoothing; add divergence guard to freeze NLMS adaptation when residual grows; use echo-debug profile (VAD off, `pre_roll_ms = 0`) to isolate AEC artifacts from VAD artifacts.

### 5. VAD passes echo-heavy audio as speech

With loud system reference playing, VAD stayed ~0.96 and `speech_ratio` ~98%. Gating does not remove echo — it only decides what reaches `cleaned_output`, so uncancelled reference plus block artifacts reach the final stream.

**When revisiting:** tune VAD threshold or add reference-aware gating; for AEC tuning sessions, disable VAD and compare `after_echo` directly in the testbed.

## Implemented so far (historical)

1. Default `echo_canceller` to `nlms`; expose `after_echo` on multi-stream socket separately from VAD-delayed output.
2. Sample-level delay aligner with fractional delay; high-pass after NLMS.
3. NLMS warmup, boundary smoothing (`echo_boundary_smoothing_samples = 64`), reduced step size (`0.1`).
4. Hybrid auto delay alignment (`delay_align.py`) with cancellation-aware fine tune and closed-loop residual observation.
5. Live delay tuning via control socket and testbed panel.
6. Testbed alignment report: VAD output delay, sample offsets, `boundary_jump_ratio`.

## Current runtime config (as of last session)

```toml
echo_canceller = "nlms"
echo_filter_taps = 4096
echo_step_size = 0.1
echo_boundary_smoothing_samples = 64
enable_reference_delay_align = true
reference_delay_mode = "auto"
reference_sync_latency_frames = 1.0
reference_drift_compensation = false
mic_delay_ms = 0
enable_vad = true
vad.pre_roll_ms = 300
```

## Suggested revisit order

1. Fix reference sync — stop latency creep and underruns.
2. Stabilize auto delay — no updates on `ref_corr = 0`; clamp near ~19 ms.
3. Re-test with VAD off, `pre_roll_ms = 0`, listening to `after_echo` only.
4. Address NLMS divergence and boundary artifacts once timing is stable.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Success criteria for a fixed run:

- `after_echo` vs `mic_raw` correlation drops well below 0.3 during reference playback.
- `after_echo` RMS ≤ `mic_raw` RMS (no divergence).
- Residual ref correlation < 0.1 with stable delay near GCC-PHAT recommendation.
- Reference sync: underruns near zero, `latency_frames` ≈ `reference_sync_latency_frames`.
- Audible: system audio largely absent from `after_echo`; `cleaned_output` natural once VAD re-enabled.