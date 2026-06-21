# Echo Cancellation Session Notes

Observed problem: the unit tests pass, but real recordings still contain the system reference in `cleaned_output`.

Root causes found:

1. Runtime config can silently use the legacy scalar canceller when `echo_canceller` is omitted.
2. The saved `cleaned_output` stream includes VAD pre-roll delay, so alignment reports compare delayed final output against raw mic.
3. Real-time echo settings can delay and level-match the reference before NLMS, making the adaptive filter less effective.
4. Diagnostics do not expose enough AEC/timing settings to confirm what backend is actually active.

Implemented fixes:

1. Default `echo_canceller` to `nlms` and make echo profiles explicit.
2. Prefer wide-tap NLMS with no frame-quantized reference delay or scalar level matching for the main AEC profile.
3. Expose `after_echo`/post-AEC audio on the multi-stream socket separately from final VAD-delayed output.
4. Report AEC backend and timing knobs in diagnostics.
5. Teach the testbed alignment report about VAD output delay and negative reference lag.
6. Add regression tests covering defaults, profile config, and post-AEC stage visibility.

Additional artifact fixes from live stream analysis:

1. Latest saved streams showed `after_echo` had large 20 ms frame-boundary discontinuities, not an every-other-sample dropout.
2. Added short NLMS boundary smoothing (`echo_boundary_smoothing_samples = 64`) to remove block-edge steps while the reference is active.
3. Reduced live NLMS adaptation from `echo_step_size = 0.5` to `0.1` to avoid aggressive frame-to-frame filter jumps.
4. Increased live `mic_delay_ms` from `40` to `60` after sample-level analysis showed sub-frame reference/mic timing mismatch.
5. Added sample-level offset and `boundary_jump_ratio` to saved testbed alignment reports.
6. Restarted the daemon and testbed so the active runtime now uses the updated settings.

Current runtime-relevant settings:

1. `echo_canceller = "nlms"`
2. `echo_filter_taps = 4096`
3. `echo_step_size = 0.1`
4. `echo_boundary_smoothing_samples = 64`
5. `enable_reference_delay_align = false`
6. `enable_reference_level_match = false`
7. `reference_sync_latency_frames = 1.0`
8. `reference_drift_compensation = false`
9. `mic_delay_ms = 60`

Verification:

1. Test suite command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v`
2. Result after changes: `26 tests OK`.
3. Daemon status confirmed the new NLMS and smoothing/timing settings were loaded.
4. Updated testbed GUI was restarted; new saved reports will include sample-level offset and boundary metrics.
