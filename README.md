# Clean Speech Daemon

Standalone Ubuntu microphone cleanup service for GUI testbeds and other local clients.

It captures webcam microphone audio, applies high-pass filtering, adaptive echo/reference reduction when a reference input is configured, spectral noise suppression, speech activity gating, and publishes the cleaned stream through:

1. A PulseAudio/PipeWire virtual source named `Clean Speech Microphone`
2. A Unix socket at `/tmp/clean-speech-daemon.sock`
3. A multi-stream Unix socket at `/tmp/clean-speech-daemon-streams.sock`
4. A raw PCM FIFO at `/tmp/clean-speech-daemon.pcm`
5. An optional debug WAV file

## Quick Start

From this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
clean-speech-daemon write-config
clean-speech-daemon devices
clean-speech-daemon run
```

Select `Clean Speech Microphone` in your GUI testbed. If the virtual source cannot be created, use the socket stream instead.

## Socket Stream

The socket sends one JSON metadata line, then raw little-endian signed 16-bit mono PCM at 48 kHz by default.

Record ten seconds from the cleaned stream:

```bash
clean-speech-daemon record-socket /tmp/cleaned.wav --seconds 10
```

The multi-stream socket is for diagnostics and GUI testbeds. It sends `mic_raw`, `system_reference`, `reference_aligned`, `reference_matched`, `after_echo`, and `cleaned_output` frames without writing them to disk. Use `after_echo` to judge AEC efficacy; `cleaned_output` is the final published stream and may include VAD pre-roll delay:

`/tmp/clean-speech-daemon-streams.sock`

## Configure Devices

List inputs:

```bash
clean-speech-daemon devices
```

Edit:

```bash
~/.config/clean-speech-daemon/config.toml
```

Set `input.microphone` to a device index or a substring of the webcam microphone name. On Linux, system audio is exposed through Pulse/PipeWire monitor sources, and the correct source is often the currently `RUNNING` monitor rather than the default sink. `input.system_audio_reference = "auto"` now prefers the active monitor source and falls back to the default sink monitor.

List Pulse/PipeWire sources and monitor states:

```bash
clean-speech-daemon pulse-sources
```

If auto-selection chooses the wrong output device, set `input.system_audio_reference` to the exact `.monitor` source name from that list.

Example:

```toml
[input]
microphone = "HD Webcam"
system_audio_reference = "auto"
```

## systemd User Service

The daemon should normally be run manually while testing. Do not leave it always listening unless you explicitly need that behavior.

Manual run:

```bash
clean-speech-daemon run
```

Install the optional user service without starting it:

```bash
scripts/install-user-service.sh
```

Start it manually for a test session:

```bash
systemctl --user start clean-speech-daemon.service
```

Stop it when done:

```bash
systemctl --user stop clean-speech-daemon.service
```

If it was previously enabled at login, disable it:

```bash
systemctl --user disable --now clean-speech-daemon.service
```

View logs:

```bash
journalctl --user -u clean-speech-daemon.service -f
```

## Virtual Microphone

The daemon can automatically run `pactl load-module module-pipe-source` so GUI applications see `Clean Speech Microphone`. This works on typical Ubuntu PipeWire-Pulse and PulseAudio setups.

If auto-loading fails, create it manually:

```bash
scripts/create-virtual-source.sh
```

## Quality Notes

This implementation is a local, testable daemon with no GUI dependency. It uses built-in DSP so it runs without heavyweight model downloads. For higher quality, replace or extend `ProcessingPipeline` with WebRTC Audio Processing, DeepFilterNet, RNNoise, or Silero VAD. The I/O, daemon, socket, FIFO, config, and service interfaces are already separated for that upgrade path.

## Echo Cancellation (`echo_canceller`)

Real desktop-speaker echo reaches the mic delayed and filtered by the room, so a single per-frame gain cannot remove it. Two backends are available via `processing.echo_canceller`:

- `"nlms"` (recommended default): a frequency-domain (overlap-save) multi-tap **adaptive filter** (`aec.py`) that learns the room impulse response and tracks slow changes. On broadband system audio convolved with a room response it clears ~90–95% of the echo. Tunables: `echo_filter_taps` (filter length in samples), `echo_step_size` (raise toward `0.6` to track clock drift faster), `echo_filter_leak`.
- `"scalar"`: the legacy single-gain reducer. Only effective when the echo is a perfectly time-aligned, unfiltered copy of the reference; kept for back-compat.

`make_echo_canceller` is the seam for dropping in a native WebRTC/Speex AEC behind the same interface. The current default gives NLMS a wide tap window and avoids frame-quantized reference delay and scalar reference level matching before the adaptive filter. Try `profiles/nlms-aec.toml`.

## Clock-Drift Synchronisation

The mic ADC and the Pulse/PipeWire monitor run on independent clocks. The daemon feeds the monitor reference through `DriftCompensatingReference` (`sync.py`), a continuous sample buffer with a proportional rate controller that resamples the reference onto the mic clock and holds latency constant. This replaces the old "reuse the last frame for 350 ms" pairing, which let latency drift unbounded and broke alignment. The controller's `resample_ratio`, `latency_frames`, and `underruns` are reported under `config.reference_reader.sync` in the diagnostics status.

## Noise Suppression

The spectral suppressor uses weighted overlap-add (sqrt-Hann analysis/synthesis at 50% overlap, a COLA pair), so signal it passes is reconstructed without the per-frame amplitude warble of naive framing. It adds one frame (`frame_ms`) of latency.

## Diagnostics

The daemon writes diagnostics that can be inspected without listening to the audio:

1. Latest status: `/tmp/clean-speech-daemon-status.json`
2. JSONL history: `/tmp/clean-speech-daemon-diagnostics.jsonl`
3. Optional per-stage WAV files: `/tmp/clean-speech-daemon-stages`

Print the current status:

```bash
clean-speech-daemon status
```

Watch status updates:

```bash
clean-speech-daemon status --watch
```

Useful fields:

1. `mic.rms_dbfs` shows microphone input level.
2. `reference.rms_dbfs` shows captured system playback reference level.
3. `reference_correlation` shows whether system playback resembles the mic signal.
4. `output.rms_dbfs` and `output.clipped_pct` show output level and clipping.
5. `pipeline.vad_score` shows speech activity confidence.
6. `pipeline.echo_gain` shows adaptive reference subtraction strength.
7. `config` shows which processing stages are enabled.
8. `pipeline.reference_gain` shows the adaptive gain applied to the system reference before echo cancellation.
9. `pipeline.reference_delay_ms` shows the current delay applied to the system reference before echo cancellation.
10. `pipeline.reference_delay_correlation` shows the correlation score used to select the delay.

Reference timing alignment is controlled by:

```toml
[processing]
enable_reference_delay_align = true
reference_delay_mode = "manual"
reference_delay_ms = 0
reference_max_delay_ms = 500
reference_delay_smoothing = 0.85
```

The raw system monitor remains available as `system_reference`; the delay-adjusted version is `reference_aligned`; the delay-adjusted and gain-adjusted version used by echo cancellation is `reference_matched`. For NLMS profiles, reference delay alignment and level matching are normally disabled, so these reference streams should match the raw reference.

Recommended timing workflow:

1. Open the GUI testbed while system audio is playing through speakers.
2. Press `Save Streams`.
3. Open the generated `*-alignment_report.json` in `~/clean-speech-recordings`.
4. Copy `offsets_vs_mic_raw.system_reference.suggested_reference_delay_ms` into `reference_delay_ms`.
5. Restart with `systemctl --user restart clean-speech-daemon.service`.

Use `reference_delay_mode = "manual"` for repeatable testing. Use `"auto"` only when the reference correlation is strong and stable.

Reference auto-leveling is controlled by:

```toml
[processing]
enable_reference_level_match = true
reference_gain_min = 0.05
reference_gain_max = 20.0
reference_gain_smoothing = 0.92
reference_target_ratio = 1.0
```


To capture stage WAVs for offline inspection, edit `~/.config/clean-speech-daemon/config.toml`:

```toml
[diagnostics]
enable_stage_wavs = true
stage_wav_dir = "/tmp/clean-speech-daemon-stages"
```

Then restart:

```bash
systemctl --user restart clean-speech-daemon.service
```

Stage files include `mic_raw.wav`, `reference.wav`, `after_highpass.wav`, `after_echo.wav`, `after_noise.wav`, and `output.wav` when available.

For baseline testing, the active config currently disables the prototype spectral suppressor because it can produce robotic artifacts. Re-enable it only when comparing stages:

```toml
[processing]
enable_noise_suppression = true
enable_speech_enhancement = true
noise_reduction = 1.15
spectral_floor = 0.08
```

## Diagnostic Profiles

Ready-to-copy profiles are in `profiles/`:

1. `profiles/passthrough.toml`: raw microphone path, no cleanup. Use this to confirm I/O quality.
2. `profiles/echo-vad.toml`: high-pass, adaptive reference reduction, and VAD only. This is the current safest cleanup baseline.
3. `profiles/spectral-experiment.toml`: enables the prototype spectral suppressor with milder settings and stage WAVs.
4. `profiles/stage-wav-debug.toml`: writes per-stage WAV files without spectral suppression.

Apply a profile:

```bash
cp profiles/passthrough.toml ~/.config/clean-speech-daemon/config.toml
systemctl --user restart clean-speech-daemon.service
clean-speech-daemon status
```

## Important Paths

1. Project: `/home/caleb/clean-speech-daemon`
2. Config: `~/.config/clean-speech-daemon/config.toml`
3. Socket: `/tmp/clean-speech-daemon.sock`
4. Multi-stream socket: `/tmp/clean-speech-daemon-streams.sock`
5. FIFO: `/tmp/clean-speech-daemon.pcm`
6. systemd unit: `systemd/clean-speech-daemon.service`
