# Clean Speech Testbed

Separate GUI client for testing the background `clean-speech-daemon` output.

Run it with:

```bash
cd /home/caleb/clean-speech-testbed
python3 clean_speech_testbed.py
```

The GUI connects to:

`/tmp/clean-speech-daemon-streams.sock`

Use it to:

1. Connect/disconnect from the cleaned stream.
2. Watch live RMS/peak levels.
3. View separate waveforms for `mic_raw`, `system_reference`, `reference_aligned`, `reference_matched`, and `cleaned_output`.
4. Play the cleaned stream through speakers/headphones.
5. Keep the last several seconds of all three streams in memory.
6. Save all buffered streams to `~/clean-speech-recordings` only when `Save Streams` is pressed.
7. View daemon diagnostics from `/tmp/clean-speech-daemon-status.json`.
8. Generate an `alignment_report.json` when saving streams, with estimated stream offsets against `mic_raw`.

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
5. `Cleaned Output`: final daemon output sent to the virtual microphone and cleaned socket.
