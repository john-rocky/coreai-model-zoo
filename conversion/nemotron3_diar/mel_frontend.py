"""NumPy mirror of the Nemotron-3-Diarization host front end: the log-mel features of the
transformers `NemotronAsrStreamingFeatureExtractor`, the low-latency streaming chunker of the model
card, and the embedder (8-frame stacking + Linear 1024->512) that runs on the host. NumPy only; this
is the algorithm the Swift host implements, gated in gate_frontend.py against transformers.

log_mel(audio, center) -> [n_valid, 128] float32
  1. preemphasis 0.97 on the chunk, its first sample kept: y[0] = x[0], y[n] = x[n] - 0.97 x[n-1]
  2. frames of 512 samples every 160: center=True (first chunk, offline) pads 256 zeros on both
     sides; center=False (later chunks) does not pad
  3. window: symmetric Hann(400) centered in the 512 frame, i.e. zeros at [0, 56) and [456, 512) --
     torch.stft's padding of a short window. The 400 values are read from hann_window_400.f32le,
     the bits of torch.hann_window(400, periodic=False) (export_n3d.py asserts them equal to the
     feature extractor's window); computing them here in float64 put 9 entries 1 ulp apart
  4. power = sqrt(re^2 + im^2)^2 of the 512-point rFFT (257 bins), float32, as the extractor computes it
  5. mel = filters [128, 257] @ power (librosa slaney filterbank, fmin 0, fmax 8000);
     log(mel + 2**-24); no normalization
  6. valid frames: center=True: floor(len / 160); center=False: floor((len - 512) / 160) + 1

Streaming (low_latency = 9 chunk + 4 look-ahead encoder frames = 104 mel frames per chunk):
  chunk 0 = audio[:16680] (center=True); chunk k>=1 starts at 160 * 72k - 256 and holds 17040
  samples (center=False); the first chunk that would run past the audio instead takes everything
  left and is the last one (no look-ahead). A chunk's own frames = its first 72 mel frames, the last
  32 are look-ahead that the next chunk starts with. The other modes (STREAMING_MODES) follow the
  same rule with their own (chunk, look-ahead): stream_chunks(n, *STREAMING_MODES[mode]).

embed(mel, projection) -> [ceil(n/8), 512]: zero-pad the frame count to a multiple of 8 (zeros in
the log-mel domain, as transformers does), stack 8 frames -> 1024, @ projection.T.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SR, N_FFT, WIN, HOP, N_MELS, SUB = 16000, 512, 400, 160, 128, 8
PREEMPH = np.float32(0.97)
LOG_GUARD = np.float32(2.0 ** -24)

CHUNK_FRAMES, LOOKAHEAD_FRAMES = 9, 4                     # low_latency, encoder frames
MEL_PER_CHUNK = (CHUNK_FRAMES + LOOKAHEAD_FRAMES) * SUB    # 104
MEL_PER_STEP = CHUNK_FRAMES * SUB                          # 72
SAMPLES_FIRST = (MEL_PER_CHUNK - 1) * HOP + WIN // 2       # 16680
SAMPLES_LATER = MEL_PER_CHUNK * HOP + WIN                  # 17040

# the processor's streaming modes: (chunk, look-ahead) in encoder frames (processor_config.json)
STREAMING_MODES = {"low_latency": (9, 4), "very_low_latency": (6, 2), "ultra_low_latency": (3, 1)}

ARTIFACTS = Path(__file__).resolve().parent / "_work" / "artifacts"


def load_f32le(path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    a = np.fromfile(path, dtype="<f4")
    assert a.size == int(np.prod(shape)), (path, a.size, shape)
    return a.reshape(shape).astype(np.float32)


@lru_cache(maxsize=4)
def hann_window(path: str | None = None) -> np.ndarray:
    """Symmetric Hann(400), float32: the shipped bits of torch.hann_window(400, periodic=False)."""
    return load_f32le(path or ARTIFACTS / "hann_window_400.f32le", (WIN,))


def frame_window(path: str | None = None) -> np.ndarray:
    w = np.zeros(N_FFT, np.float32)
    left = (N_FFT - WIN) // 2
    w[left:left + WIN] = hann_window(path)
    return w


def mel_filters(path: str | Path | None = None) -> np.ndarray:
    return load_f32le(path or ARTIFACTS / "mel_filters_128x257.f32le", (N_MELS, N_FFT // 2 + 1))


def n_valid_frames(n_samples: int, center: bool) -> int:
    if center:
        return n_samples // HOP
    return (n_samples - N_FFT) // HOP + 1


def log_mel(audio: np.ndarray, center: bool, filters: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    y = np.empty_like(x)
    y[:1] = x[:1]
    y[1:] = x[1:] - PREEMPH * x[:-1]
    if center:
        y = np.pad(y, (N_FFT // 2, N_FFT // 2))
    n_frames = 1 + (y.shape[0] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx] * frame_window()[None, :]                                 # [F, 512] float32
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)                              # complex64 (numpy >= 2)
    re, im = spec.real.astype(np.float32), spec.imag.astype(np.float32)
    power = np.sqrt(re * re + im * im) ** 2                                   # [F, 257] float32
    mel = (filters @ power.T).T                                               # [F, 128]
    out = np.log(mel + LOG_GUARD).astype(np.float32)
    return out[: n_valid_frames(len(x), center)]


def stream_chunks(n_samples: int, chunk_frames: int = CHUNK_FRAMES, lookahead_frames: int = LOOKAHEAD_FRAMES):
    """(start, end, is_first, is_last) of every streaming chunk, as the model card's generator.
    Defaults = low_latency; other modes pass STREAMING_MODES[mode]."""
    mel_per_chunk = (chunk_frames + lookahead_frames) * SUB
    mel_per_step = chunk_frames * SUB
    samples_first = (mel_per_chunk - 1) * HOP + WIN // 2
    samples_later = mel_per_chunk * HOP + WIN
    yield 0, samples_first, True, False
    mel_idx = mel_per_step
    start = mel_idx * HOP - N_FFT // 2
    while start + samples_later <= n_samples:
        yield start, start + samples_later, False, False
        mel_idx += mel_per_step
        start = mel_idx * HOP - N_FFT // 2
    yield start, n_samples, False, True


def embed(mel: np.ndarray, projection: np.ndarray) -> np.ndarray:
    pad = -mel.shape[0] % SUB
    m = np.pad(mel, ((0, pad), (0, 0))) if pad else mel
    stacked = m.reshape(-1, SUB * N_MELS).astype(np.float32)
    return (stacked @ projection.T).astype(np.float32)
