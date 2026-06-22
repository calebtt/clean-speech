"""Neural acoustic echo cancellers for the clean-speech-daemon (16 kHz).

Three methods under one interface, selectable for A/B testing and (later) for the
live daemon's `echo_canceller` setting:

- "dtln"   : DTLN-aec (ONNX). Deep non-linear cancellation; can warble on speech.
- "nkf"    : NKF-AEC (neural Kalman filter, torch). Linear -> artifact-free but
             shallow (only the linear echo).
- "hybrid" : NKF -> DTLN. NKF removes the clean linear echo first, so DTLN only
             suppresses the weak non-linear residual and barely touches near-end
             speech. Best of both: deep cancellation, voice stays clear.

All operate on 16 kHz mono float32. Use tools/aec_compare.py to run them on saved
recordings. Realtime factors: DTLN ~0.3x, NKF ~0.11x, hybrid ~0.41x.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(__file__).resolve().parents[1] / "models"
BLOCK_LEN = 512
BLOCK_SHIFT = 128
SR = 16_000


# --------------------------------------------------------------------------- #
# DTLN-aec (ONNX)
# --------------------------------------------------------------------------- #
class DtlnAec:
    def __init__(self, model_base: str = "dtln_aec_512", providers=None) -> None:
        providers = providers or ["CPUExecutionProvider"]
        self.s1 = ort.InferenceSession(str(MODELS / f"{model_base}_1.onnx"), providers=providers)
        self.s2 = ort.InferenceSession(str(MODELS / f"{model_base}_2.onnx"), providers=providers)

    def process(self, mic: np.ndarray, lpb: np.ndarray, mask_floor: float = 0.0, mask_smooth: float = 0.0) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32)
        lpb = np.asarray(lpb, dtype=np.float32)
        n = min(len(mic), len(lpb))
        mic, lpb = mic[:n], lpb[:n]
        pad = np.zeros(BLOCK_LEN - BLOCK_SHIFT, dtype=np.float32)
        mic = np.concatenate([pad, mic, pad])
        lpb = np.concatenate([pad, lpb, pad])
        s1 = np.zeros((1, 2, 512, 2), np.float32)
        s2 = np.zeros((1, 2, 512, 2), np.float32)
        in_buf = np.zeros(BLOCK_LEN, np.float32)
        lpb_buf = np.zeros(BLOCK_LEN, np.float32)
        out_buf = np.zeros(BLOCK_LEN, np.float32)
        out = np.zeros(len(mic), np.float32)
        prev_mask = None
        for k in range((mic.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT):
            i = k * BLOCK_SHIFT
            in_buf[:-BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:]; in_buf[-BLOCK_SHIFT:] = mic[i:i + BLOCK_SHIFT]
            lpb_buf[:-BLOCK_SHIFT] = lpb_buf[BLOCK_SHIFT:]; lpb_buf[-BLOCK_SHIFT:] = lpb[i:i + BLOCK_SHIFT]
            in_fft = np.fft.rfft(in_buf).astype(np.complex64)
            in_mag = np.abs(in_fft).reshape(1, 1, -1).astype(np.float32)
            lpb_mag = np.abs(np.fft.rfft(lpb_buf)).reshape(1, 1, -1).astype(np.float32)
            mask, s1 = self.s1.run(["Identity", "Identity_1"], {"input_3": in_mag, "input_4": lpb_mag, "input_5": s1})
            if mask_smooth > 0.0 and prev_mask is not None:
                mask = mask_smooth * prev_mask + (1.0 - mask_smooth) * mask
            if mask_floor > 0.0:
                mask = np.maximum(mask, mask_floor)
            prev_mask = mask
            est = np.fft.irfft(in_fft * mask).reshape(1, 1, -1).astype(np.float32)
            ob, s2 = self.s2.run(["Identity", "Identity_1"],
                                 {"input_6": est, "input_7": lpb_buf.reshape(1, 1, -1).astype(np.float32), "input_8": s2})
            out_buf[:-BLOCK_SHIFT] = out_buf[BLOCK_SHIFT:]; out_buf[-BLOCK_SHIFT:] = 0.0
            out_buf += np.squeeze(ob)
            out[i:i + BLOCK_SHIFT] = out_buf[:BLOCK_SHIFT]
        return out[(BLOCK_LEN - BLOCK_SHIFT):(BLOCK_LEN - BLOCK_SHIFT) + n]


# --------------------------------------------------------------------------- #
# NKF-AEC (neural Kalman filter, torch) -- vendored from fjiang9/NKF-AEC (BSD-3)
# --------------------------------------------------------------------------- #
def _gcc_phat(sig, refsig, fs=1, interp=1):
    n = sig.shape[0] + refsig.shape[0]
    R = np.fft.rfft(sig, n=n) * np.conj(np.fft.rfft(refsig, n=n))
    cc = np.fft.irfft(R / (np.abs(R) + 1e-15), n=interp * n)
    ms = int(interp * n / 2)
    cc = np.concatenate((cc[-ms:], cc[:ms + 1]))
    return (np.argmax(np.abs(cc)) - ms) / float(interp * fs)


class NkfAec:
    """Lazy torch wrapper so importing this module never requires torch unless used."""

    def __init__(self, ckpt: str = "nkf_epoch70.pt") -> None:
        import torch
        import torch.nn as nn

        # Attribute names MUST match the published checkpoint (fjiang9/NKF-AEC).
        class ComplexGRU(nn.Module):
            def __init__(s, i, h, num_layers=1):
                super().__init__()
                s.gru_r = nn.GRU(i, h, num_layers, batch_first=True)
                s.gru_i = nn.GRU(i, h, num_layers, batch_first=True)

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

            def stft(s, x):
                return torch.stft(x, 1024, 256, 1024, torch.hann_window(1024), return_complex=True)

            def istft(s, X):
                return torch.istft(X, 1024, 256, 1024, torch.hann_window(1024), return_complex=False)

            def forward(s, x, y):
                x = s.stft(x.unsqueeze(0)); y = s.stft(y.unsqueeze(0))
                B, F, T = x.shape; d = x.device
                hpri = torch.zeros(B * F, s.L, 1, dtype=torch.complex64, device=d)
                hpos = torch.zeros(B * F, s.L, 1, dtype=torch.complex64, device=d)
                s.kg_net.init_hidden(B * F, d)
                x = x.reshape(B * F, T); y = y.reshape(B * F, T)
                echo = torch.zeros(B * F, T, dtype=torch.complex64, device=d)
                for t in range(T):
                    if t < s.L:
                        xt = torch.cat([torch.zeros(B * F, s.L - t - 1, dtype=torch.complex64, device=d), x[:, :t + 1]], -1)
                    else:
                        xt = x[:, t - s.L + 1:t + 1]
                    if xt.abs().mean() < 1e-5:
                        continue
                    dh = hpos - hpri; hpri = hpos
                    e = y[:, t] - torch.matmul(xt.unsqueeze(1), hpri).squeeze()
                    kg = s.kg_net(torch.cat([xt, e.unsqueeze(1), dh.squeeze()], 1))
                    hpos = hpri + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))
                    echo[:, t] = torch.matmul(xt.unsqueeze(1), hpos).squeeze()
                return s.istft((y - echo).reshape(B, F, T)).squeeze()

        self._torch = torch
        self.model = NKF(L=4)
        self.model.load_state_dict(torch.load(str(MODELS / ckpt)))
        self.model.eval()

    def process(self, mic: np.ndarray, ref: np.ndarray, align: bool = True) -> np.ndarray:
        torch = self._torch
        y = torch.from_numpy(np.asarray(mic, dtype=np.float32))
        x = torch.from_numpy(np.asarray(ref, dtype=np.float32))
        if align:
            tau = _gcc_phat(y[:SR * 10].numpy(), x[:SR * 10].numpy(), fs=SR, interp=1)
            tau = max(0, int((tau - 0.001) * SR))
            x = torch.cat([torch.zeros(tau), x])[:y.shape[-1]]
        with torch.no_grad():
            return self.model(x, y).cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
METHODS = ("dtln", "nkf", "hybrid")


def run_method(method: str, mic: np.ndarray, ref: np.ndarray, mask_smooth: float = 0.6) -> np.ndarray:
    """Run one canceller on 16 kHz mic/ref, return 16 kHz cleaned audio."""
    if method == "dtln":
        return DtlnAec().process(mic, ref, mask_smooth=mask_smooth)
    if method == "nkf":
        return NkfAec().process(mic, ref)
    if method == "hybrid":
        residual = NkfAec().process(mic, ref)
        return DtlnAec().process(residual, ref, mask_smooth=mask_smooth)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
