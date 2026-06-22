#!/usr/bin/env python3
"""A/B-test echo cancellers on a saved recording.

Runs the neural cancellers (dtln / nkf / hybrid) on a saved mic_raw +
system_reference pair, writes cleaned 16 kHz wavs, and prints a metrics table so
you can pick a method by ear and by number.

    python tools/aec_compare.py --session 20260622-092844
    python tools/aec_compare.py --mic a.wav --ref b.wav --methods hybrid dtln
    python tools/aec_compare.py --session 20260622-092844 --out-dir /tmp/aec_ab

Metrics:
  reduction      total echo energy removed (dB; higher = more echo gone)
  residual       leftover correlation with the reference (lower = cleaner of echo)
  gain-jitter    short-time level fluctuation (proxy for warble/musical noise;
                 lower = smoother voice). Trust your ears over this one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neural_aec import METHODS, run_method  # noqa: E402

SR = 16_000
RECORDINGS = Path.home() / "clean-speech-recordings"


def _load16(path: Path) -> np.ndarray:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def _best_lag_corr(a: np.ndarray, b: np.ndarray, step: int = 8, max_lag_ms: int = 80) -> float:
    k = min(len(a), len(b))
    a = a[:k] - a[:k].mean()
    b = b[:k] - b[:k].mean()
    m = int(SR * max_lag_ms / 1000)
    best = 0.0
    for lag in range(-m, m + 1, step):
        aa, bb = (a[-lag:], b[: len(a) + lag]) if lag < 0 else ((a[:-lag], b[lag:]) if lag > 0 else (a, b))
        d = np.linalg.norm(aa) * np.linalg.norm(bb)
        if d > 1e-9:
            best = max(best, abs(float(aa @ bb / d)))
    return best


def _gain_jitter(x: np.ndarray, hop_ms: int = 50) -> float:
    h = int(SR * hop_ms / 1000)
    e = np.array([_rms(x[i * h:(i + 1) * h]) for i in range(len(x) // h)])
    return float(np.std(np.diff(np.log(e + 1e-6))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="stamp under ~/clean-speech-recordings")
    ap.add_argument("--mic", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--mask-smooth", type=float, default=0.6, help="DTLN temporal mask smoothing")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/aec_ab"))
    ap.add_argument("--warmup-s", type=float, default=2.0)
    args = ap.parse_args()

    if args.session:
        mic = _load16(RECORDINGS / f"{args.session}-mic_raw.wav")
        ref = _load16(RECORDINGS / f"{args.session}-system_reference.wav")
        label = args.session
    elif args.mic and args.ref:
        mic, ref = _load16(args.mic), _load16(args.ref)
        label = f"{args.mic.name}/{args.ref.name}"
    else:
        ap.error("provide --session or both --mic and --ref")

    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    w = int(args.warmup_s * SR)

    print(f"session: {label}  ({n / SR:.1f}s @ {SR} Hz)")
    print(f"mic vs reference (echo present): {_best_lag_corr(mic[w:], ref[w:]):.3f}\n")
    print(f"{'method':10} | reduction | residual | gain-jitter | file")
    print("-" * 78)
    for method in args.methods:
        out = run_method(method, mic, ref, mask_smooth=args.mask_smooth)
        m = min(len(out), n)
        red = 20.0 * np.log10(_rms(mic[w:m]) / _rms(out[w:m]))
        res = _best_lag_corr(out[w:m], ref[w:m])
        jit = _gain_jitter(out[w:m])
        path = args.out_dir / f"{label}_{method}.wav"
        sf.write(path, out, SR, subtype="PCM_16")
        print(f"{method:10} |  {red:+5.1f} dB | {res:.3f}    |   {jit:.3f}     | {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
