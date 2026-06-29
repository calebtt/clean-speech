#!/usr/bin/env python3
"""Deterministic offline replay of a saved session through the processing pipeline.

Why this exists
---------------
``reference_drift_compensation`` is off (the reference is a fixed-delay FIFO) and
the NLMS canceller is deterministic, so replaying the saved ``mic_raw`` +
``system_reference`` frames through :class:`ProcessingPipeline` reproduces
``after_echo`` bit-for-bit. That makes this the right tool for iterating the
delay-align / NLMS / boundary fixes: change a knob,
re-run, read the success-criteria metrics -- no re-recording, no live mic.

It does NOT exercise the reference jitter buffer (sync): the saved
reference is already post-sync. Validate sync live via the underrun/resync
counters; validate everything downstream here.

Usage
-----
    # Latest session in ~/clean-speech-recordings, current config:
    PYTHONPATH=src python testbed/replay_session.py

    # A specific session:
    PYTHONPATH=src python testbed/replay_session.py --stamp 20260621-093440

    # A/B a knob without editing the toml (repeatable --set):
    PYTHONPATH=src python testbed/replay_session.py --stamp 20260621-093440 \
        --set processing.reference_max_delay_ms=40

    # Write the replayed after_echo for listening:
    PYTHONPATH=src python testbed/replay_session.py --out /tmp/after_echo.replay.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from clean_speech_daemon.config import Config, load_config
from clean_speech_daemon.processing import ProcessingPipeline, normalized_correlation


RECORDINGS = Path.home() / "clean-speech-recordings"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def _db(ratio: float) -> float:
    return 20.0 * float(np.log10(max(ratio, 1e-12)))


def best_lag_correlation(a: np.ndarray, sample_rate: int, b: np.ndarray, step: int = 48, max_lag_ms: int = 200) -> float:
    """Max |correlation| over lags -- echo strength independent of alignment."""
    n = min(len(a), len(b))
    if n < sample_rate // 10:
        return 0.0
    a = a[:n].astype(np.float64) - float(np.mean(a[:n]))
    b = b[:n].astype(np.float64) - float(np.mean(b[:n]))
    max_lag = min(n - 1, int(sample_rate * max_lag_ms / 1000))
    best = 0.0
    for lag in range(-max_lag, max_lag + 1, step):
        if lag < 0:
            aa, bb = a[-lag:], b[: len(a) + lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom > 1e-9:
            best = max(best, abs(float(np.dot(aa, bb) / denom)))
    return best


def boundary_jump_ratio(x: np.ndarray, frame_samples: int) -> float:
    """Mean step across block boundaries vs mean intra-block sample step."""
    n = frame_samples
    nframes = len(x) // n
    if nframes < 2:
        return 0.0
    trimmed = x[: nframes * n].astype(np.float64)
    boundary_steps = np.abs(trimmed[n::n] - trimmed[n - 1 : -1 : n])
    intra = np.abs(np.diff(trimmed.reshape(nframes, n), axis=1))
    mean_intra = float(np.mean(intra)) + 1e-12
    return float(np.mean(boundary_steps)) / mean_intra


def apply_overrides(config: Config, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects section.field=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        section_name, _, field = dotted.partition(".")
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, field):
            raise SystemExit(f"unknown config field: {dotted}")
        current = getattr(section, field)
        if isinstance(current, bool):
            value: object = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(section, field, value)


def load_session(stamp: str | None, mic_path: Path | None, ref_path: Path | None) -> tuple[np.ndarray, np.ndarray, str]:
    if mic_path and ref_path:
        mic, sr1 = sf.read(mic_path, dtype="float32")
        ref, sr2 = sf.read(ref_path, dtype="float32")
        return mic, ref, f"{mic_path.name} / {ref_path.name}"
    if stamp is None:
        stamps = sorted({p.name[:15] for p in RECORDINGS.glob("*-mic_raw.wav")})
        if not stamps:
            raise SystemExit(f"no *-mic_raw.wav sessions found in {RECORDINGS}")
        stamp = stamps[-1]
    mic_p = RECORDINGS / f"{stamp}-mic_raw.wav"
    ref_p = RECORDINGS / f"{stamp}-system_reference.wav"
    if not mic_p.exists() or not ref_p.exists():
        raise SystemExit(f"missing mic_raw/system_reference for session {stamp}")
    mic, _ = sf.read(mic_p, dtype="float32")
    ref, _ = sf.read(ref_p, dtype="float32")
    return mic, ref, stamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stamp", help="session stamp under ~/clean-speech-recordings (default: latest)")
    parser.add_argument("--mic", type=Path, help="explicit mic_raw wav path")
    parser.add_argument("--ref", type=Path, help="explicit reference wav path")
    parser.add_argument("--config", type=Path, help="config.toml to load (default: user config)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.field=value")
    parser.add_argument("--warmup-frames", type=int, default=80, help="frames to skip before scoring")
    parser.add_argument("--out", type=Path, help="write the replayed after_echo wav here")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    apply_overrides(config, args.overrides)

    sample_rate = int(config.input.sample_rate)
    frame_samples = int(sample_rate * config.input.frame_ms / 1000)
    mic, ref, label = load_session(args.stamp, args.mic, args.ref)

    usable = min(len(mic), len(ref))
    usable -= usable % frame_samples
    mic, ref = mic[:usable], ref[:usable]

    pipeline = ProcessingPipeline(config)
    after_echo = np.empty_like(mic)
    aligned = np.empty_like(ref)
    delays = []
    for i in range(0, usable, frame_samples):
        pipeline.process(mic[i : i + frame_samples], ref[i : i + frame_samples])
        stages = pipeline.last_stages
        after_echo[i : i + frame_samples] = stages["after_echo"]
        aligned[i : i + frame_samples] = stages.get("reference_aligned", ref[i : i + frame_samples])
        delays.append(pipeline.stats.reference_delay_ms)

    w = min(args.warmup_frames * frame_samples, usable // 2)
    mic_s, echo_s, ref_s, aligned_s = mic[w:], after_echo[w:], ref[w:], aligned[w:]
    delays_s = np.asarray(delays[args.warmup_frames :] or delays)

    mic_ref = best_lag_correlation(mic_s, sample_rate, ref_s)
    echo_ref = best_lag_correlation(echo_s, sample_rate, ref_s)
    same_lag_echo_mic = abs(normalized_correlation(echo_s.astype(np.float64), mic_s.astype(np.float64)))
    rms_ratio = _rms(echo_s) / _rms(mic_s)
    reduction_db = _db(_rms(mic_s) / _rms(echo_s))

    print(f"session            : {label}")
    print(f"config             : taps={config.processing.echo_filter_taps} "
          f"step={config.processing.echo_step_size} "
          f"max_delay_ms={config.processing.reference_max_delay_ms} "
          f"delay_mode={config.processing.reference_delay_mode}")
    print(f"duration           : {usable / sample_rate:.1f}s ({usable // frame_samples} frames)")
    print("-" * 60)
    print(f"aligner delay_ms   : {np.mean(delays_s):6.1f} mean  {np.median(delays_s):6.1f} median  "
          f"[GCC-PHAT truth ~19 ms]")
    print(f"echo vs ref (lag)  : {echo_ref:6.3f}   (mic vs ref {mic_ref:.3f}; lower = more echo removed)")
    print(f"echo vs mic (0-lag): {same_lag_echo_mic:6.3f}   (high = filter barely changed mic)")
    print(f"after_echo RMS/mic : {rms_ratio:6.3f}  ({reduction_db:+.1f} dB; want <=0 dB, no boost)")
    print(f"boundary jump ratio: {boundary_jump_ratio(echo_s, frame_samples):6.2f}")
    print("-" * 60)
    checks = {
        "echo removed (echo_ref < 0.6*mic_ref)": echo_ref < 0.6 * mic_ref,
        "no divergence (RMS ratio <= 1.0)": rms_ratio <= 1.0,
        "delay near truth (<60 ms)": float(np.median(delays_s)) < 60.0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    if args.out:
        sf.write(args.out, after_echo, sample_rate, subtype="PCM_16")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
