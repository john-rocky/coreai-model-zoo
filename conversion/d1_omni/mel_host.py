#!/usr/bin/env python3
"""d1-omni-600M's audio front end on the host: a 16 kHz mono clip -> the normalised log-mel [128, F] the audio graph
(d1_omni_audio.py) reads, and the valid frame count.

Two forms of one computation (the publisher's `audio.py` @ 414f8d64: `waveform()` and `MelFrontend`, always fp32):

  mel_torch   the publisher's code written out with the same torch calls on the same shapes (nothing imported from
              the publisher): equal to its MelFrontend bit for bit (audio_check.py --stage mel). The Python host.
  mel_numpy   NumPy only, the steps spelled out: the form a Swift host copies. Default arithmetic float64 (the DFT,
              the mel projection, the log and the statistics; Swift: Double), cast to float32 at the end.

The steps (both forms; F = 1 + n // 160 columns, `frames` = n // 160 valid columns):

  1  waveform    1-D samples; cut to the first 480,000 (30 s); int16 -> float32 x / 32768 (exact), float -> float32;
                 zero-padded at the end to 8,000 samples (0.5 s) when shorter. n = the length after this step.
  2  preemphasis y[0] = x[0]; y[i] = x[i] - 0.97 * x[i-1]
  3  framing     y zero-padded by 256 samples on both sides (torch.stft center=True, pad_mode "constant"); column t =
                 the 512 padded samples from t * 160, times the window: a Hann of 400 points, periodic=False
                 (w[k] = 0.5 - 0.5 cos(2 pi k / 399), k = 0..399), centred in 512 (56 zeros on each side)
  4  spectrum    the 512-point real DFT of each column, bins 0..256; power = (sqrt(re^2 + im^2))^2 (the publisher's
                 |X| squared; in float64 it is re^2 + im^2 to rounding)
  5  mel         filterbank [128, 257] (slaney_filterbank: librosa's Slaney mel, 0 to 8 kHz, computed in float64 and
                 stored as float32; dumped as mel_filters_128x257_f32.bin beside the bundle) @ power -> [128, F]
  6  log         log(mel + 2^-24)
  7  normalise   per mel row over the valid columns t < frames only: mean; std = sqrt(sum((x - mean)^2) / (frames - 1))
                 (nan when frames = 1 -> 0); x = (x - mean) / (std + 1e-5); the columns t >= frames are set to 0.0.
                 (frames >= 50 for every clip the publisher's waveform() lets through: n >= 8,000.)

The four time masks the audio graph also reads (ConvSubsampling's `_time_mask` before each of its layers) come from
`frames` alone: host.audio_inputs.

How close the two forms are (audio_check.py --stage mel, round 6, 22 clips): mel_torch equals the publisher bit for
bit; mel_numpy in float64 equals mel_torch run in float64 to 4.5e-13 (the same function), and sits from the publisher's
float32 mel exactly as far as that mel sits from its own float64 run: up to 3.1e-4 on speech (14 of 22 clips within
1e-4; the largest gaps are mel rows 0-3, whose few FFT bins near 31-94 Hz carry 1e-8 of the frame's power, where the
float32 FFT keeps 3 or 4 digits). Digital silence is the extreme: every row's std is 0, so float64 gives 0.0 while the
publisher's float32 mean of 200 equal values rounds and its normalised rows are that rounding times 1e5 (up to 0.16).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE, MIN_SAMPLES, MAX_SECONDS = 16000, 8000, 30
N_FFT, WINDOW, HOP, FEATURES = 512, 400, 160, 128
PREEMPHASIS = 0.97
LOG_GUARD = 2.0 ** -24
NORM_EPS = 1e-5
FILTERBANK_FILE = "mel_filters_128x257_f32.bin"


# =========================================================================== audio.py @ 414f8d64, verbatim
def slaney_filterbank(sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = 128) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, norm="slaney") in float32, computed as librosa computes it."""
    f_sp, min_log_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)[()]
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = f_sp * m
        high = m >= min_log_mel
        f[high] = min_log_hz * np.exp(logstep * (m[high] - min_log_mel))
        return f

    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, np.fft.rfftfreq(n=n_fft, d=1.0 / sr))
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, np.newaxis]
    return weights
# =========================================================================== end of the slaney_filterbank copy


def waveform_numpy(audio) -> np.ndarray:
    """waveform() in NumPy: mono 16 kHz int16 PCM or float samples -> float32 [n], cut to 30 s, padded to 0.5 s."""
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be mono: a 1-D array of 16 kHz samples")
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    x = x.astype(np.float32) / np.float32(32768.0) if x.dtype == np.int16 else x.astype(np.float32)
    if len(x) < MIN_SAMPLES:
        x = np.pad(x, (0, MIN_SAMPLES - len(x)))
    return x


def frame_counts(n_raw: int) -> dict:
    """For n_raw input samples: n after waveform(), the STFT columns F and the valid frames."""
    n = max(min(n_raw, MAX_SECONDS * SAMPLE_RATE), MIN_SAMPLES)
    return {"samples": n, "columns": 1 + n // HOP, "frames": (n + N_FFT // 2 * 2 - N_FFT) // HOP}


def filterbank_bytes() -> bytes:
    """mel_filters_128x257_f32.bin: slaney_filterbank() as float32 little-endian, row-major [128, 257]."""
    return np.ascontiguousarray(slaney_filterbank(n_fft=N_FFT, n_mels=FEATURES), dtype="<f4").tobytes()


# --------------------------------------------------------------------------- the torch form (= the publisher's code)
def mel_torch(audio, float64: bool = False) -> tuple[np.ndarray, int]:
    """waveform() + MelFrontend.forward written out (torch, CPU) -> (mel [128, F] float32, frames). float64=True is a
    diagnostic only: the same ops in float64 (the window and the filterbank's float32 values cast up)."""
    import torch

    # waveform(), verbatim but for the torch import
    x = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be mono: a 1-D array of 16 kHz samples")
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    x = x.astype(np.float32) / np.float32(32768.0) if x.dtype == np.int16 else x.astype(np.float32)
    if len(x) < MIN_SAMPLES:
        x = np.pad(x, (0, MIN_SAMPLES - len(x)))
    x = torch.from_numpy(x)[None]
    # MelFrontend.forward (no_grad, CPU)
    with torch.no_grad():
        x = x.double() if float64 else x.float()
        n = torch.tensor([x.shape[1]], device=x.device)
        frames = torch.floor_divide(n + N_FFT // 2 * 2 - N_FFT, HOP)
        x = torch.cat((x[:, :1], x[:, 1:] - PREEMPHASIS * x[:, :-1]), dim=1)
        window = torch.hann_window(WINDOW, periodic=False, dtype=x.dtype).to(x.device)
        x = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=WINDOW, center=True,
                       window=window, return_complex=True, pad_mode="constant")
        x = torch.sqrt(torch.view_as_real(x).pow(2).sum(-1)).pow(2.0)
        fb = torch.from_numpy(slaney_filterbank(n_fft=N_FFT, n_mels=FEATURES))[None].to(x.device).to(x.dtype)
        x = torch.log(torch.matmul(fb, x) + 2**-24)
        valid = torch.arange(x.shape[2], device=x.device)[None] < frames[:, None]
        count = valid.sum(1)
        mean = torch.where(valid[:, None], x, 0.0).sum(2) / count[:, None]
        std = torch.sqrt(torch.where(valid[:, None], x - mean[:, :, None], 0.0).pow(2).sum(2) / (count[:, None] - 1.0))
        x = (x - mean[:, :, None]) / (std.masked_fill(std.isnan(), 0.0) + 1e-5)[:, :, None]
        x = x.masked_fill(~valid[:, None], 0.0)
    return x[0].to(torch.float32).numpy().copy(), int(frames[0])


# --------------------------------------------------------------------------- the NumPy form (the Swift host's spec)
def hann_window_numpy(dtype=np.float64) -> np.ndarray:
    """The 512-point frame window: Hann(400, periodic=False) centred, 56 zeros each side."""
    k = np.arange(WINDOW, dtype=np.float64)
    w = np.zeros(N_FFT, dtype=np.float64)
    offset = (N_FFT - WINDOW) // 2
    w[offset:offset + WINDOW] = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (WINDOW - 1))
    return w.astype(dtype)


def mel_numpy(audio, float64: bool = True) -> tuple[np.ndarray, int]:
    """The module docstring's seven steps in NumPy -> (mel [128, F] float32, frames). float64=False runs the same steps
    in float32 (NumPy's float32 rfft), recorded beside the float64 form."""
    ft = np.float64 if float64 else np.float32
    x = waveform_numpy(audio)
    n = len(x)
    frames = (n + N_FFT // 2 * 2 - N_FFT) // HOP
    columns = 1 + n // HOP
    xf = x.astype(ft)
    y = np.empty_like(xf)
    y[0] = xf[0]
    y[1:] = xf[1:] - ft(PREEMPHASIS) * xf[:-1]
    padded = np.concatenate([np.zeros(N_FFT // 2, dtype=ft), y, np.zeros(N_FFT // 2, dtype=ft)])
    index = (np.arange(columns) * HOP)[:, None] + np.arange(N_FFT)[None]
    spectrum = np.fft.rfft(padded[index] * hann_window_numpy(ft), n=N_FFT, axis=-1)   # [F, 257]
    re, im = spectrum.real.astype(ft), spectrum.imag.astype(ft)
    power = np.sqrt(re * re + im * im) ** 2
    mel = slaney_filterbank(n_fft=N_FFT, n_mels=FEATURES).astype(ft) @ power.T          # [128, F]
    x = np.log(mel + ft(LOG_GUARD))
    valid = x[:, :frames]
    mean = valid.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(((valid - mean) ** 2).sum(axis=1, keepdims=True) / ft(frames - 1))
    std = np.where(np.isnan(std), ft(0.0), std)
    out = (x - mean) / (std + ft(NORM_EPS))
    out[:, frames:] = 0.0
    return out.astype(np.float32), int(frames)
