# Clean Speech Testbed

Separate GUI client for testing the background `clean-speech-daemon` output.

Run it with:

```bash
cd /home/caleb/clean-speech-testbed
python3 clean_speech_testbed.py
```

The GUI connects to:

- `/tmp/clean-speech-daemon-streams.sock` — live audio streams (read).
- `/tmp/clean-speech-daemon-control.sock` — runtime tuning (write).

Use it to:

1. Connect/disconnect from the cleaned stream.
2. Watch live RMS/peak levels.
3. View separate waveforms for `mic_raw`, `system_reference`, `reference_aligned`, `reference_matched`, `after_echo`, and `cleaned_output`.
4. Play the cleaned stream — or `after_echo` — through speakers/headphones.
5. Keep the last several seconds of all streams in memory.
6. Save all buffered streams to `~/clean-speech-recordings` only when `Save Streams` is pressed.
7. View daemon diagnostics from `/tmp/clean-speech-daemon-status.json`.
8. Generate an `alignment_report.json` when saving streams, with estimated stream offsets against `mic_raw`.

## Echo Alignment panel

The **Echo Alignment** panel tunes the echo canceller live (over the control
socket) and tells you, in plain language, whether the echo is cancellable.

Background: the adaptive (NLMS) filter can only subtract echo it has already
received, so the system **reference must lead the echo** in the mic. On many
machines the `parec` monitor reference arrives 15–30 ms *late*, which makes
cancellation impossible until you delay the mic to compensate (`mic_delay_ms`).

The live readout (updated ~1×/s from the in-memory buffer) shows:

- **mic vs reference** — echo strength. Near zero ⇒ no echo to cancel.
- **lag** — `reference leads` (causal, good) or `reference LATE` (non-causal).
- **after_echo vs reference** — residual echo; should drop well below *mic vs reference*.
- **after_echo / mic level** — divergence guard; should stay ≤ 0 dB (no boost).
- a colour-coded **Verdict** with the next action.

Controls (each pushes to the daemon control socket):

- **Mic delay (ms)** — primary causality knob. Raise it until the verdict says causal.
- **Ref delay (ms) / mode** — keep `manual` / `0`; the filter taps absorb the lead.
  (`auto` currently wanders and resets the filter, preventing convergence.)
- **Reset Echo Filter** — re-converge after any change; wait ~2 s.

Click **Workflow / Help** for the full step-by-step procedure. To iterate offline
instead, press **Save Streams** and replay with `testbed/replay_session.py`.

The GUI is intentionally separate from the daemon. It consumes the daemon socket exactly like any other test client would.

The daemon should already be running as:

```bash
systemctl --user status clean-speech-daemon.service
```

If needed, restart it:

```bash
systemctl --user restart clean-speech-daemon.service
```

The diagnostics panel shows mic level, reference level, output level, clipping, VAD score, speech ratio, echo gain, reference correlation, and active processing stages.

The waveform panels are labeled explicitly:

1. `Mic Raw`: microphone input before processing.
2. `System Reference`: system playback monitor used as the echo/reference signal.
3. `Reference Aligned`: system reference after adaptive delay alignment to the microphone signal.
4. `Reference Matched`: aligned system reference after adaptive level matching.
5. `After Echo Cancellation`: mic after the echo canceller, before noise/VAD — the stream to judge cancellation on.
6. `Cleaned Output`: final daemon output sent to the virtual microphone and cleaned socket.
