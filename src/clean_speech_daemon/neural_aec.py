"""Realtime neural echo cancellers for the live daemon (48 kHz in/out).

Wraps 16 kHz neural models behind the daemon's per-frame canceller interface
(``process(mic_frame, reference_frame) -> cleaned_frame`` + ``reset()`` + ``gain``).
Internally: stream-resample 48->16 kHz (soxr), run a stateful 16 kHz canceller,
stream-resample 16->48 kHz, and return exactly ``frame_samples`` per call from an
output FIFO (priming latency ~100 ms).

Methods (config ``echo_canceller``):
- "dtln"   : DTLN-aec (ONNX). Deep non-linear cancellation; can warble on speech.
- "nkf"    : NKF-AEC (neural Kalman, torch). Linear -> artifact-free but shallow.
- "hybrid" : NKF -> DTLN. NKF removes the clean linear echo first so DTLN only
             suppresses the weak non-linear residual; deep cancellation, voice
             stays clear. The recommended neural setting.

Streaming pieces are verified to match the whole-signal models within rounding.
Models live in ``<repo>/models`` (dtln_aec_512_{1,2}.onnx, nkf_epoch70.pt).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soxr

MODELS = Path(__file__).resolve().parents[2] / "models"
NFFT = 1024
HOP = 256
NKF_L = 4
_WIN = np.hanning(NFFT + 1)[:-1].astype(np.float64)  # periodic Hann (matches torch)
_WOLA_C = 1.5  # sum of squared periodic-Hann at 4x overlap (hop=256, win=1024)
DTLN_BLOCK = 512
DTLN_SHIFT = 128
# Divergence guard: clamp the NKF filter-state magnitude. Normal operation keeps
# it around 7-21 (even on loud input), while a runaway shoots to 1e5+. Clamping at
# 50 is well above any normal value -- so it never alters normal output -- yet reins
# in a runaway before it becomes a (clipped) full-scale buzz. A soft clamp is far
# less disruptive than resetting the filter (which would drop cancellation entirely).
NKF_STATE_BOUND = 50.0


# --------------------------------------------------------------------------- #
# Streaming DTLN-aec (ONNX)
# --------------------------------------------------------------------------- #
def _single_thread_session(path: Path) -> ort.InferenceSession:
    # Per-frame inferences are tiny; multi-threading them only adds sync overhead
    # and contends with torch (NKF). Single-threaded is ~6x faster end to end here.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class StreamingDtln:
    def __init__(self, mask_smooth: float = 0.6, mask_floor: float = 0.0) -> None:
        self.s1 = _single_thread_session(MODELS / "dtln_aec_512_1.onnx")
        self.s2 = _single_thread_session(MODELS / "dtln_aec_512_2.onnx")
        self.mask_smooth = float(mask_smooth)
        self.mask_floor = float(mask_floor)
        self.reset()

    def reset(self) -> None:
        self._st1 = np.zeros((1, 2, 512, 2), np.float32)
        self._st2 = np.zeros((1, 2, 512, 2), np.float32)
        self._inb = np.zeros(DTLN_BLOCK, np.float32)
        self._lpb = np.zeros(DTLN_BLOCK, np.float32)
        self._outb = np.zeros(DTLN_BLOCK, np.float32)
        self._prev = None
        self._qi = np.zeros(0, np.float32)
        self._ql = np.zeros(0, np.float32)

    def process(self, mic: np.ndarray, lpb: np.ndarray) -> np.ndarray:
        self._qi = np.concatenate([self._qi, np.asarray(mic, np.float32)])
        self._ql = np.concatenate([self._ql, np.asarray(lpb, np.float32)])
        out = []
        while len(self._qi) >= DTLN_SHIFT and len(self._ql) >= DTLN_SHIFT:
            mc, self._qi = self._qi[:DTLN_SHIFT], self._qi[DTLN_SHIFT:]
            lc, self._ql = self._ql[:DTLN_SHIFT], self._ql[DTLN_SHIFT:]
            self._inb[:-DTLN_SHIFT] = self._inb[DTLN_SHIFT:]; self._inb[-DTLN_SHIFT:] = mc
            self._lpb[:-DTLN_SHIFT] = self._lpb[DTLN_SHIFT:]; self._lpb[-DTLN_SHIFT:] = lc
            fft = np.fft.rfft(self._inb).astype(np.complex64)
            mag = np.abs(fft).reshape(1, 1, -1).astype(np.float32)
            lmag = np.abs(np.fft.rfft(self._lpb)).reshape(1, 1, -1).astype(np.float32)
            mask, self._st1 = self.s1.run(["Identity", "Identity_1"],
                                          {"input_3": mag, "input_4": lmag, "input_5": self._st1})
            if self.mask_smooth > 0.0 and self._prev is not None:
                mask = self.mask_smooth * self._prev + (1.0 - self.mask_smooth) * mask
            if self.mask_floor > 0.0:
                mask = np.maximum(mask, self.mask_floor)
            self._prev = mask
            est = np.fft.irfft(fft * mask).reshape(1, 1, -1).astype(np.float32)
            ob, self._st2 = self.s2.run(["Identity", "Identity_1"],
                                        {"input_6": est, "input_7": self._lpb.reshape(1, 1, -1).astype(np.float32),
                                         "input_8": self._st2})
            self._outb[:-DTLN_SHIFT] = self._outb[DTLN_SHIFT:]; self._outb[-DTLN_SHIFT:] = 0.0
            self._outb += np.squeeze(ob)
            out.append(self._outb[:DTLN_SHIFT].copy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


# --------------------------------------------------------------------------- #
# Streaming NKF-AEC (neural Kalman filter, torch KGNet)
# --------------------------------------------------------------------------- #
def _build_kgnet(ckpt: Path):
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)  # tiny per-frame calls; avoid thread-sync overhead + ORT contention

    class ComplexGRU(nn.Module):
        def __init__(s, i, h, num_layers=1):
            super().__init__(); s.gru_r = nn.GRU(i, h, num_layers, batch_first=True); s.gru_i = nn.GRU(i, h, num_layers, batch_first=True)

        def forward(s, x, h_rr=None, h_ir=None, h_ri=None, h_ii=None):
            Frr, h_rr = s.gru_r(x.real, h_rr); Fir, h_ir = s.gru_r(x.imag, h_ir)
            Fri, h_ri = s.gru_i(x.real, h_ri); Fii, h_ii = s.gru_i(x.imag, h_ii)
            return torch.complex(Frr - Fii, Fri + Fir), h_rr, h_ir, h_ri, h_ii

    class ComplexDense(nn.Module):
        def __init__(s, i, o):
            super().__init__(); s.linear_real = nn.Linear(i, o); s.linear_imag = nn.Linear(i, o)

        def forward(s, x):
            return torch.complex(s.linear_real(x.real), s.linear_imag(x.imag))

    class ComplexPReLU(nn.Module):
        def __init__(s):
            super().__init__(); s.prelu = nn.PReLU()

        def forward(s, x):
            return torch.complex(s.prelu(x.real), s.prelu(x.imag))

    class KGNet(nn.Module):
        def __init__(s, L, fc_dim=18, rnn_layers=1, rnn_dim=18):
            super().__init__(); s.L = L; s.rnn_layers = rnn_layers; s.rnn_dim = rnn_dim
            s.fc_in = nn.Sequential(ComplexDense(2 * L + 1, fc_dim), ComplexPReLU())
            s.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers)
            s.fc_out = nn.Sequential(ComplexDense(rnn_dim, fc_dim), ComplexPReLU(), ComplexDense(fc_dim, L))

        def init_hidden(s, b, d):
            s.h = [torch.zeros(s.rnn_layers, b, s.rnn_dim, device=d) for _ in range(4)]

        def forward(s, f):
            f = s.fc_in(f).unsqueeze(1)
            o, *s.h = s.complex_gru(f, *s.h)
            return s.fc_out(o).permute(0, 2, 1)

    class NKF(nn.Module):
        def __init__(s, L=4):
            super().__init__(); s.L = L; s.kg_net = KGNet(L)

    model = NKF(L=NKF_L)
    model.load_state_dict(torch.load(str(ckpt)))
    model.eval()
    return torch, model.kg_net


class StreamingNkf:
    def __init__(self, ckpt: str = "nkf_epoch70.pt") -> None:
        self._torch, self.kg = _build_kgnet(MODELS / ckpt)
        self.bins = NFFT // 2 + 1
        self.reset()

    def reset(self) -> None:
        self._xt = np.zeros((self.bins, NKF_L), np.complex128)
        self._hpri = np.zeros((self.bins, NKF_L), np.complex128)
        self._hpos = np.zeros((self.bins, NKF_L), np.complex128)
        self._ola = np.zeros(NFFT, np.float64)
        self._qi = np.zeros(0, np.float32)
        self._qr = np.zeros(0, np.float32)
        self.kg.init_hidden(self.bins, "cpu")

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._qi = np.concatenate([self._qi, np.asarray(mic, np.float32)])
        self._qr = np.concatenate([self._qr, np.asarray(ref, np.float32)])
        out = []
        while len(self._qi) >= NFFT and len(self._qr) >= NFFT:
            Y = np.fft.rfft(self._qi[:NFFT] * _WIN)
            X = np.fft.rfft(self._qr[:NFFT] * _WIN)
            self._xt = np.concatenate([self._xt[:, 1:], X[:, None]], axis=1)
            if np.mean(np.abs(X)) < 1e-5:
                S = Y
            else:
                e = Y - np.sum(self._xt * self._hpri, axis=1)
                dh = self._hpos - self._hpri
                self._hpri = self._hpos.copy()
                feat = np.concatenate([self._xt, e[:, None], dh], axis=1).astype(np.complex64)
                with torch.no_grad():
                    kg = self.kg(torch.from_numpy(feat)).squeeze(-1).numpy()
                self._hpos = self._hpri + kg * e[:, None]
                # Divergence guard: clamp the filter-state magnitude (and scrub any
                # non-finite values). Above-normal only -> normal cancellation is
                # untouched; a runaway is reined in continuously, no reset needed.
                self._hpos = np.nan_to_num(self._hpos)
                hmag = float(np.max(np.abs(self._hpos)))
                if hmag > NKF_STATE_BOUND:
                    self._hpos *= NKF_STATE_BOUND / hmag
                S = Y - np.sum(self._xt * self._hpos, axis=1)
            self._ola += np.fft.irfft(S) * _WIN
            out.append((self._ola[:HOP] / _WOLA_C).astype(np.float32).copy())
            self._ola[:-HOP] = self._ola[HOP:]; self._ola[-HOP:] = 0.0
            self._qi = self._qi[HOP:]; self._qr = self._qr[HOP:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)


# --------------------------------------------------------------------------- #
# Streaming LocalVQE (reference-aware AEC+NS+dereverb) via the GGML C engine
# --------------------------------------------------------------------------- #
LIBDIR = Path(__file__).resolve().parents[2] / "lib"
LOCALVQE_GGUF = "localvqe-v1.3-4.8M-f32.gguf"
_LV_LIB = None


def _localvqe_lib():
    """Load liblocalvqe.so (and its ggml deps) once; return the configured handle.

    The PyTorch LocalVQE model is not chunk-streamable, but the GGML engine's
    localvqe_process_frame_f32() is bit-identical batch-vs-frame, so it drives the
    realtime loop. The CPU backend variants are discovered by ggml in its own
    directory (GGML_BACKEND_PATH), so the vendored lib/ is self-contained.
    """
    global _LV_LIB
    if _LV_LIB is not None:
        return _LV_LIB
    import ctypes
    import os

    os.environ.setdefault("GGML_BACKEND_PATH", str(LIBDIR))
    for dep in ("libggml-base.so", "libggml.so"):
        ctypes.CDLL(str(LIBDIR / dep), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(str(LIBDIR / "liblocalvqe.so"))
    cfloat_p = ctypes.POINTER(ctypes.c_float)
    lib.localvqe_new.restype = ctypes.c_void_p
    lib.localvqe_new.argtypes = [ctypes.c_char_p]
    for fn in ("localvqe_process_f32", "localvqe_process_frame_f32"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p, cfloat_p, cfloat_p, ctypes.c_int, cfloat_p]
    for fn in ("localvqe_hop_length", "localvqe_sample_rate"):
        f = getattr(lib, fn)
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p]
    lib.localvqe_reset.argtypes = [ctypes.c_void_p]
    lib.localvqe_free.argtypes = [ctypes.c_void_p]
    _LV_LIB = (lib, cfloat_p)
    return _LV_LIB


class StreamingLocalVQE:
    def __init__(self, gguf: str = LOCALVQE_GGUF) -> None:
        self._lib, self._cfp = _localvqe_lib()
        self._ctx = self._lib.localvqe_new(str(LIBDIR.parent / "models" / gguf).encode())
        if not self._ctx:
            raise RuntimeError(f"LocalVQE failed to load model {gguf}")
        self.hop = self._lib.localvqe_hop_length(self._ctx)
        self._qi = np.zeros(0, np.float32)
        self._qr = np.zeros(0, np.float32)

    def _cf(self, a: np.ndarray):
        return np.ascontiguousarray(a, np.float32).ctypes.data_as(self._cfp)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        self._qi = np.concatenate([self._qi, np.asarray(mic, np.float32)])
        self._qr = np.concatenate([self._qr, np.asarray(ref, np.float32)])
        out = []
        while len(self._qi) >= self.hop and len(self._qr) >= self.hop:
            mc = self._qi[: self.hop].copy(); self._qi = self._qi[self.hop :]
            rc = self._qr[: self.hop].copy(); self._qr = self._qr[self.hop :]
            o = np.zeros(self.hop, np.float32)
            self._lib.localvqe_process_frame_f32(self._ctx, self._cf(mc), self._cf(rc), self.hop, self._cf(o))
            out.append(o)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def reset(self) -> None:
        self._lib.localvqe_reset(self._ctx)
        self._qi = np.zeros(0, np.float32)
        self._qr = np.zeros(0, np.float32)

    def __del__(self) -> None:
        try:
            if getattr(self, "_ctx", 0):
                self._lib.localvqe_free(self._ctx)
        except Exception:
            pass


# Streaming hybrid leads LocalVQE by ~128 samples @16 kHz; delay LocalVQE to align
# the two before blending (measured by cross-correlation, low but consistent).
LOCALVQE_BLEND_ALIGN = 128


# --------------------------------------------------------------------------- #
# Daemon-facing 48 kHz canceller
# --------------------------------------------------------------------------- #
class NeuralEchoCanceller:
    def __init__(self, method: str, frame_samples: int, sample_rate: int = 48_000,
                 mask_smooth: float = 0.6, localvqe_blend: float = 0.7) -> None:
        if method not in ("dtln", "nkf", "hybrid", "hybrid_localvqe"):
            raise ValueError(f"unknown neural method {method!r}")
        self.method = method
        self.frame_samples = int(frame_samples)
        self.sample_rate = int(sample_rate)
        self.mask_smooth = float(mask_smooth)
        self.localvqe_blend = float(np.clip(localvqe_blend, 0.0, 1.0))
        self.gain = 0.0  # diagnostics: recent suppression (1 - out/mic)
        self._build()

    def _build(self) -> None:
        sr = self.sample_rate
        self._rs_mic = soxr.ResampleStream(sr, 16_000, 1, dtype="float32", quality="HQ")
        self._rs_ref = soxr.ResampleStream(sr, 16_000, 1, dtype="float32", quality="HQ")
        self._rs_out = soxr.ResampleStream(16_000, sr, 1, dtype="float32", quality="HQ")
        hybrid = self.method in ("hybrid", "hybrid_localvqe")
        self._nkf = StreamingNkf() if self.method == "nkf" or hybrid else None
        self._dtln = StreamingDtln(self.mask_smooth) if self.method == "dtln" or hybrid else None
        # hybrid_localvqe: run LocalVQE alongside the hybrid and blend (LocalVQE
        # scrubs the residual; the hybrid keeps the near-end voice prominent).
        self._localvqe = StreamingLocalVQE() if self.method == "hybrid_localvqe" else None
        self._hyb_fifo = np.zeros(0, np.float32)
        # delay LocalVQE so its output time-aligns with the hybrid before blending
        self._lv_fifo = np.zeros(LOCALVQE_BLEND_ALIGN, np.float32) if self.method == "hybrid_localvqe" else np.zeros(0, np.float32)
        self._ref16 = np.zeros(0, np.float32)   # ref delay-line aligned to NKF residual
        self._out48 = np.zeros(0, np.float32)    # output FIFO @ daemon rate
        self._primed = False

    def reset(self) -> None:
        self._build()

    def process(self, mic: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
        n = len(mic)
        mic = np.asarray(mic, dtype=np.float32)
        # Feeding silence when the reference is absent keeps every stream in sync
        # (and passes the mic through, since there is nothing to cancel).
        ref = np.zeros(n, np.float32) if reference is None or len(reference) != n else np.asarray(reference, np.float32)

        mic16 = self._rs_mic.resample_chunk(mic)
        ref16 = self._rs_ref.resample_chunk(ref)

        if self.method == "dtln":
            clean16 = self._dtln.process(mic16, ref16)
        elif self.method == "nkf":
            clean16 = self._nkf.process(mic16, ref16)
        else:  # hybrid / hybrid_localvqe: NKF residual -> DTLN, ref delay-aligned to it
            self._ref16 = np.concatenate([self._ref16, ref16])
            residual = self._nkf.process(mic16, ref16)
            m = len(residual)
            ref_aligned, self._ref16 = self._ref16[:m], self._ref16[m:]
            hyb16 = self._dtln.process(residual, ref_aligned) if m else np.zeros(0, np.float32)
            if self.method == "hybrid":
                clean16 = hyb16
            else:  # blend hybrid (keeps voice) with LocalVQE (scrubs residual)
                lv16 = self._localvqe.process(mic16, ref16)
                self._hyb_fifo = np.concatenate([self._hyb_fifo, hyb16])
                self._lv_fifo = np.concatenate([self._lv_fifo, lv16])
                k = min(len(self._hyb_fifo), len(self._lv_fifo))
                if k:
                    b = self.localvqe_blend
                    clean16 = (b * self._lv_fifo[:k] + (1.0 - b) * self._hyb_fifo[:k]).astype(np.float32)
                    self._hyb_fifo, self._lv_fifo = self._hyb_fifo[k:], self._lv_fifo[k:]
                else:
                    clean16 = np.zeros(0, np.float32)

        if len(clean16):
            # Final NaN/Inf defense before the audio leaves the canceller.
            clean16 = np.clip(np.nan_to_num(clean16, nan=0.0, posinf=1.5, neginf=-1.5), -1.5, 1.5)
            self._out48 = np.concatenate([self._out48, self._rs_out.resample_chunk(clean16)])
            mr = float(np.sqrt(np.mean(mic16 * mic16)) + 1e-9) if len(mic16) else 1.0
            cr = float(np.sqrt(np.mean(clean16 * clean16)) + 1e-9)
            self.gain = float(np.clip(1.0 - cr / mr, 0.0, 1.0))

        # Prime once enough output is buffered, then always return exactly n samples.
        if not self._primed:
            if len(self._out48) < n:
                return np.zeros(n, np.float32)
            self._primed = True
        if len(self._out48) >= n:
            frame, self._out48 = self._out48[:n], self._out48[n:]
            return frame
        frame = np.concatenate([self._out48, np.zeros(n - len(self._out48), np.float32)])
        self._out48 = np.zeros(0, np.float32)
        return frame
