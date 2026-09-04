# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.1",
#     "torch==2.9.0",
#     "numpy",
#     "soundfile",
#     "huggingface-hub",
#     "kokoro",
#     "misaki",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Export Kokoro-82M (StyleTTS2 + iSTFTNet) text-to-speech to Core AI.

Kokoro is the zoo's first TTS. It is NOT autoregressive: phonemes + a voice/style
vector go in, a 24 kHz waveform comes out. The acoustic graph
(`kokoro.KModel.forward_with_tokens`) has ONE data-dependent length -- the
duration->alignment expansion `L = sum(pred_dur)` -- so the model is cut into
three fixed-shape `.aimodel` bundles with two host steps between them:

    text --(misaki G2P, host)--> phoneme ids
    1. predictor.aimodel : ids[1,T], ref_s[1,256], attn_mask[1,T]
                           -> duration[1,T], d[1,T,640], t_en[1,512,T]
    host: pred_dur = round(duration); alignment one-hot aln[1,T,L]; frame_mask[1,L]
    2. prosody.aimodel   : d, t_en, aln, ref_s, frame_mask -> asr[1,512,L], F0[1,2L], N[1,2L]
    host: har = STFT(SineGen(f0_upsamp(F0)))   (the hn-nsf excitation, ~a windowed FFT)
    3. vocoder.aimodel   : asr, F0, N, har, ref_s, frame_mask -> audio[L*600]; host trims

The bundles are voice-INDEPENDENT (the voice is the `ref_s` input -- one of the
shipped `voices/*.pt`). Token length T and frame length L are fixed BUCKETS
(default 128 / 512); the host left-pads to the bucket and trims the output. Longer
text is split into sentences host-side (as `KPipeline` does), each <= the bucket.

Why three bundles + masking + host source (every workaround is load-bearing):
  * nn.LSTM specializes the sequence length under torch.export (dynamic fails),
    and the decoder's manual conv ops specialize L too -> fixed buckets.
  * The 6 bidirectional LSTMs are unrolled as a MASKED bi-LSTM that carries state
    through right-padding (`masked_bilstm`) -- fused nn.LSTM leaks pad tokens into
    the backward pass and wrecks the prosody (audio corr 0.02).
  * The 58 AdaIN InstanceNorms normalize over L, so bucket pad frames poison the
    statistics; every decoder AdaIN takes the frame mask, normalizes over real
    frames only, and zeros pad output so the convs see exact-like zeros.
  * The hn-nsf source's STFT phase (atan2) flips by 2*pi at the F0->0 pad boundary
    under fp32 on the engine -> compute that one source STFT on the host (`compute_har`).
  * Mask resize uses `arange < real_count*N/Lb`, NOT F.interpolate(nearest), which
    rounds the boundary differently on the engine.
  * weight_norm MUST be folded (`torch.nn.utils.remove_weight_norm`): kokoro uses
    the old hook-based weight_norm, so `module.weight` stays at its random init
    value until a forward fires -- and the manual conv stand-ins read it directly.
  * ConvTranspose1d is replaced by a bit-exact zero-insertion + conv1d. The original
    reasons (a symbolic output length from output_padding; conv_transpose1d returning
    all zeros for the iSTFT) no longer reproduce as of coreai-torch 0.4.1/0.4.2, but
    the iSTFT rewrite still has to stay -- see its docstring.
  * input_ids are int32; atan2 -> 2*atan; the `%1` in SineGen is a no-op for speech
    (f0/sr<1) and dropped.

Run on the Core AI CPU compute unit (the unrolled LSTM is fast there, ~8 ms;
on the GPU it is dispatch-bound). Numerics gate = magspec-corr vs the torch
reference (the bounded pad-boundary effect lowers raw waveform corr to ~0.98,
but the spectrum -- what is perceived -- matches at 0.999).

Deps: pip install kokoro misaki soundfile (+ the coreai stack; torch <= 2.11).

  python export_kokoro.py --out-dir exports            # export the 3 bundles
  python export_kokoro.py --verify --voice af_heart \
      --text "Hello, world."                            # engine vs torch gate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO = "hexgrad/Kokoro-82M"
TB, LB = 128, 512               # default token / frame buckets
MASKED = True                   # ship bundles use the masked unrolled LSTMs


# ===========================================================================
# masked bidirectional LSTM (unrolled; carries state through right-padding)
# ===========================================================================
def masked_bilstm(lstm, x, real_mask):
    B, T, _ = x.shape
    H = lstm.hidden_size

    def cell(xt, h, c, Wi, Wh, bi, bh):
        g = xt @ Wi.T + bi + h @ Wh.T + bh
        i, f, gg, o = g.chunk(4, -1)
        c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        return torch.sigmoid(o) * torch.tanh(c2), c2

    def run(Wi, Wh, bi, bh, order):
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)
        out = [None] * T
        for t in order:
            m = real_mask[:, t:t + 1]
            h2, c2 = cell(x[:, t], h, c, Wi, Wh, bi, bh)
            h = m * h2 + (1 - m) * h
            c = m * c2 + (1 - m) * c
            out[t] = h
        return out

    fwd = run(lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0, range(T))
    bwd = run(lstm.weight_ih_l0_reverse, lstm.weight_hh_l0_reverse,
              lstm.bias_ih_l0_reverse, lstm.bias_hh_l0_reverse, range(T - 1, -1, -1))
    return torch.stack([torch.cat([fwd[t], bwd[t]], -1) for t in range(T)], 1)


def _run_lstm(lstm, x, real_mask):
    if MASKED:
        return masked_bilstm(lstm, x, real_mask)
    return lstm(x)[0]


# ===========================================================================
# export-friendly, numerically-faithful monkeypatches
# ===========================================================================
def _manual_depthwise_ctranspose(x, ct):
    """Depthwise ConvTranspose1d (k3 s2 p1 op1) as zero-insertion + conv1d.

    Written for a symbolic coreai output length under output_padding, which no longer
    reproduces: this config converts and runs clean on 0.4.1 and 0.4.2, gpu and
    cpu_only, to 2.4e-07 through a downstream concat. Kept because it is bit-exact and
    verified in the shipped graph; dropping it needs a full kokoro re-export first.
    """
    C = ct.in_channels
    B, _, L = x.shape
    up = x.new_zeros(B, C, 2 * L)
    up[:, :, ::2] = x
    return F.conv1d(up, torch.flip(ct.weight, dims=[-1]), bias=ct.bias,
                    padding=ct.kernel_size[0] - 2, groups=C)


def _manual_convT_general(x, weight, bias, stride, padding=0):
    """General conv_transpose1d as zero-insertion + conv1d.

    DO NOT remove this. The original symptom (all zeros) is gone, so the old comment
    invited exactly that. The rewrite is still required: at the iSTFT shape (11 -> 1,
    k=20, s=5) coreai's conv_transpose1d is correct on gpu (4.8e-07) and wrong on
    cpu_only (max|d| 4.86) on both 0.4.1 and 0.4.2, and kokoro ships
    GraphModel(computeUnits: .cpu). That is FB24322424, which is still open -- any
    ConvTranspose with kernel >= 8 is exposed on cpu_only.
    """
    B, Cin, L = x.shape
    K = weight.shape[-1]
    up = x.new_zeros(B, Cin, stride * L - (stride - 1))
    up[:, :, ::stride] = x
    w = torch.flip(weight, dims=[-1]).transpose(0, 1).contiguous()
    return F.conv1d(up, w, bias=bias, padding=K - 1 - padding)


def _resize_mask(fm, length):
    if fm is None:
        return None
    if fm.shape[-1] == length:
        return fm
    real = fm.sum()
    idx = torch.arange(length, device=fm.device, dtype=fm.dtype)
    return (idx < real * (length / fm.shape[-1])).to(fm.dtype).view(1, 1, length)


def _textencoder_forward(self, x, input_lengths, m):
    x = self.embedding(x).transpose(1, 2)
    mm = m.unsqueeze(1)
    x = x.masked_fill(mm, 0.0)
    for c in self.cnn:
        x = c(x)
        x = x.masked_fill(mm, 0.0)
    x = x.transpose(1, 2)
    x = _run_lstm(self.lstm, x, (~m).float())
    x = x.transpose(-1, -2)
    return x.masked_fill(mm, 0.0)


def _durationencoder_forward(self, x, style, text_lengths, m):
    from kokoro.modules import AdaLayerNorm
    masks = m
    x = x.permute(2, 0, 1)
    s = style.expand(x.shape[0], x.shape[1], -1)
    x = torch.cat([x, s], axis=-1)
    x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    x = x.transpose(0, 1).transpose(-1, -2)
    for block in self.lstms:
        if isinstance(block, AdaLayerNorm):
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
        else:
            x = x.transpose(-1, -2)
            x = _run_lstm(block, x, (~masks).float())
            x = x.transpose(-1, -2)
            x = x.masked_fill(masks.unsqueeze(1), 0.0)
    return x.transpose(-1, -2)


def _adain1d_forward(self, x, s, fm=None):
    h = self.fc(s).view(s.size(0), -1, 1)
    gamma, beta = torch.chunk(h, 2, dim=1)
    if fm is None:
        return (1 + gamma) * self.norm(x) + beta
    mask = _resize_mask(fm, x.shape[-1])
    n = mask.sum(-1, keepdim=True).clamp(min=1.0)
    mean = (x * mask).sum(-1, keepdim=True) / n
    var = (((x - mean) ** 2) * mask).sum(-1, keepdim=True) / n
    xn = (x - mean) / torch.sqrt(var + self.norm.eps)
    out = (1 + gamma) * (xn * self.norm.weight.view(1, -1, 1) + self.norm.bias.view(1, -1, 1)) + beta
    return out * mask


def _adainresblk1d_residual(self, x, s, fm=None):
    x = self.norm1(x, s, fm)
    x = self.actv(x)
    x = _manual_depthwise_ctranspose(x, self.pool) if self.upsample_type != "none" else self.pool(x)
    x = self.conv1(self.dropout(x))
    x = self.norm2(x, s, fm)
    x = self.actv(x)
    return self.conv2(self.dropout(x))


def _adainresblk1d_forward(self, x, s, fm=None):
    out = (self._residual(x, s, fm) + self._shortcut(x)) * 0.7071067811865476
    return out * _resize_mask(fm, out.shape[-1]) if fm is not None else out


def _adainresblock1_forward(self, x, s, fm=None):
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1,
                                      self.adain2, self.alpha1, self.alpha2):
        xt = n1(x, s, fm)
        xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
        xt = c1(xt)
        xt = n2(xt, s, fm)
        xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
        x = c2(xt) + x
    return x * _resize_mask(fm, x.shape[-1]) if fm is not None else x


def _upsample1d_forward(self, x):
    if self.layer_type == "none":
        return x
    return F.interpolate(x, size=int(2 * x.shape[-1]), mode="nearest")


def _sinegen_f02sine(self, f0_values):
    rad = f0_values / self.sampling_rate            # `% 1` is a no-op for speech
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale,
                        mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                          scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
    return torch.sin(phase)


def _sinegen_forward(self, f0):
    fn = torch.multiply(f0, torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype,
                                         device=f0.device).view(1, 1, -1))
    sine = self._f02sine(fn) * self.sine_amp
    uv = self._f02uv(f0)
    return sine * uv, uv, torch.zeros_like(sine)    # deterministic (no rand/noise)


def _sourcemodule_forward(self, x):
    with torch.no_grad():
        sine, uv, _ = self.l_sin_gen(x)
    return self.l_tanh(self.l_linear(sine)), torch.zeros_like(uv), uv


def _customstft_transform(self, waveform):
    if self.center:
        waveform = F.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
    x = waveform.unsqueeze(1)
    re = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
    im = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)
    mag = torch.sqrt(re ** 2 + im ** 2 + 1e-14)
    phase = 2.0 * torch.atan(im / (mag + re + 1e-12))   # atan2 via half-angle
    return mag, torch.where((im == 0) & (re < 0), torch.full_like(phase, torch.pi), phase)


def _customstft_inverse(self, magnitude, phase, length=None):
    re = _manual_convT_general(magnitude * torch.cos(phase), self.weight_backward_real, None, self.hop_length)
    im = _manual_convT_general(magnitude * torch.sin(phase), self.weight_backward_imag, None, self.hop_length)
    w = re - im
    if self.center:
        w = w[..., self.n_fft // 2:-(self.n_fft // 2)]
    return w[..., :length] if length is not None else w


def _generator_forward_har(self, x, s, har, fm=None):
    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = self.noise_res[i](self.noise_convs[i](har), s, fm)
        up = self.ups[i]
        x = _manual_convT_general(x, up.weight, up.bias, up.stride[0], up.padding[0])
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(self.num_kernels):
            blk = self.resblocks[i * self.num_kernels + j]
            xs = blk(x, s, fm) if xs is None else xs + blk(x, s, fm)
        x = xs / self.num_kernels
    x = self.conv_post(F.leaky_relu(x))
    spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
    phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
    return self.stft.inverse(spec, phase)


def apply_patches():
    from kokoro import modules as kmod
    from kokoro import istftnet as kist
    from kokoro import custom_stft as kstft
    kmod.TextEncoder.forward = _textencoder_forward
    kmod.DurationEncoder.forward = _durationencoder_forward
    kist.AdaIN1d.forward = _adain1d_forward
    kist.AdaINResBlock1.forward = _adainresblock1_forward
    kist.AdainResBlk1d._residual = _adainresblk1d_residual
    kist.AdainResBlk1d.forward = _adainresblk1d_forward
    kist.UpSample1d.forward = _upsample1d_forward
    kist.SineGen._f02sine = _sinegen_f02sine
    kist.SineGen.forward = _sinegen_forward
    kist.SourceModuleHnNSF.forward = _sourcemodule_forward
    kstft.CustomSTFT.transform = _customstft_transform
    kstft.CustomSTFT.inverse = _customstft_inverse


def build_model():
    from kokoro import KModel
    from torch.nn.utils import remove_weight_norm
    model = KModel(repo_id=REPO, disable_complex=True).eval()
    apply_patches()
    for m in model.modules():               # fold old-style hook weight_norm (mandatory)
        try:
            remove_weight_norm(m)
        except (ValueError, RuntimeError):
            pass
    return model


# ===========================================================================
# the three exported sub-models
# ===========================================================================
class Predictor(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.bert, self.bert_encoder = k.bert, k.bert_encoder
        self.predictor, self.text_encoder = k.predictor, k.text_encoder

    def forward(self, input_ids, ref_s, attn_mask):
        T = input_ids.shape[-1]
        lengths = torch.full((1,), T, dtype=torch.long)
        text_mask = attn_mask < 0.5
        bert = self.bert(input_ids, attention_mask=attn_mask.long())
        d_en = self.bert_encoder(bert).transpose(-1, -2)
        d = self.predictor.text_encoder(d_en, ref_s[:, 128:], lengths, text_mask)
        x = _run_lstm(self.predictor.lstm, d, attn_mask.float())
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(axis=-1)
        t_en = self.text_encoder(input_ids, lengths, text_mask)
        return duration, d, t_en


class Prosody(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.predictor = k.predictor

    def forward(self, d, t_en, aln, ref_s, frame_mask):
        s = ref_s[:, 128:]
        p = self.predictor
        fm = frame_mask.unsqueeze(1) if MASKED else None
        en = d.transpose(-1, -2) @ aln
        sh = _run_lstm(p.shared, en.transpose(-1, -2), frame_mask).transpose(-1, -2)
        F0 = sh
        for b in p.F0:
            F0 = b(F0, s, fm)
        N = sh
        for b in p.N:
            N = b(N, s, fm)
        return t_en @ aln, p.F0_proj(F0).squeeze(1), p.N_proj(N).squeeze(1)


class Vocoder(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.decoder = k.decoder

    def forward(self, asr, F0_curve, N, har, ref_s, frame_mask):
        s = ref_s[:, :128]
        d = self.decoder
        fm = frame_mask.unsqueeze(1) if MASKED else None
        F0 = d.F0_conv(F0_curve.unsqueeze(1))
        Nn = d.N_conv(N.unsqueeze(1))
        x = d.encode(torch.cat([asr, F0, Nn], axis=1), s, fm)
        asr_res = d.asr_res(asr)
        res = True
        for block in d.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nn], axis=1)
            x = block(x, s, fm)
            if block.upsample_type != "none":
                res = False
        return _generator_forward_har(d.generator, x, s, har, fm).reshape(-1)


# ===========================================================================
# host steps + helpers
# ===========================================================================
def get_vocab():
    return json.load(open(hf_hub_download(REPO, "config.json")))["vocab"]


def text_to_ids(text, vocab, british=False):
    from misaki import en
    ps, _ = en.G2P(trf=False, british=british, fallback=None)(text)
    ids = [vocab[p] for p in ps if p in vocab]
    return torch.LongTensor([[0, *ids, 0]]), ps


def load_voice(name):
    return torch.load(hf_hub_download(REPO, f"voices/{name}.pt"), weights_only=True)


def build_alignment(duration, speed=1.0, T_bucket=None, L_bucket=None):
    pred_dur = torch.round(duration.squeeze(0) / speed).clamp(min=1).long().reshape(-1)
    T = pred_dur.shape[0]
    L = int(pred_dur.sum().item())
    Tb, Lb = T_bucket or T, L_bucket or L
    aln = torch.zeros((Tb, Lb))
    aln[torch.repeat_interleave(torch.arange(T), pred_dur), torch.arange(L)] = 1
    frame_mask = torch.zeros((1, Lb))
    frame_mask[:, :L] = 1
    return aln.unsqueeze(0), frame_mask, L


def compute_har(F0_pred, model):
    """Host hn-nsf source: f0_upsamp + SineGen + STFT (a windowed FFT) -> har[1,22,frames]."""
    gen = model.decoder.generator
    with torch.no_grad():
        f0 = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        src, _, _ = gen.m_source(f0)
        mag, phase = gen.stft.transform(src.transpose(1, 2).squeeze(1))
    return torch.cat([mag, phase], dim=1)


# ===========================================================================
# convert / verify
# ===========================================================================
def _convert(mod, args, in_names, out_names, out_path, license_="Apache-2.0"):
    import coreai.runtime as rt
    from coreai_torch import TorchConverter, get_decomp_table
    with torch.no_grad():
        ep = torch.export.export(mod.eval(), args)
    ep = ep.run_decompositions(get_decomp_table())
    prog = TorchConverter().add_exported_program(
        exported_program=ep, input_names=in_names, output_names=out_names).to_coreai()
    prog.optimize()
    shutil.rmtree(out_path, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = rt.AIModelAssetMetadata()
    meta.author = "hexgrad (Kokoro-82M); Core AI export: coreai-model-zoo"
    meta.license = license_
    meta.model_description = "Kokoro-82M TTS (StyleTTS2 + iSTFTNet). https://huggingface.co/hexgrad/Kokoro-82M"
    prog.save_asset(out_path, meta)
    sz = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"[convert] {out_path.name} ({sz:.0f} MB)")


def export_bundles(out_dir: Path, tb=TB, lb=LB):
    model = build_model()
    vocab = get_vocab()
    predictor, prosody, vocoder = Predictor(model), Prosody(model), Vocoder(model)
    ids, _ = text_to_ids("Hello, world. This is a test of on device speech generation.", vocab)
    T = ids.shape[-1]
    ref_s = load_voice("af_heart")[T - 1]
    ids_b = torch.cat([ids, torch.zeros((1, tb - T), dtype=torch.long)], dim=1)
    attn = torch.cat([torch.ones((1, T)), torch.zeros((1, tb - T))], dim=1)
    with torch.no_grad():
        dur, d, t_en = predictor(ids_b, ref_s, attn)
        aln, fm, _ = build_alignment(dur[:, :T], 1.0, tb, lb)
        asr, F0, N = prosody(d, t_en, aln, ref_s, fm)
        har = compute_har(F0, model)
    _convert(predictor, (ids_b, ref_s, attn), ["input_ids", "ref_s", "attn_mask"],
             ["duration", "d", "t_en"], out_dir / "kokoro_predictor.aimodel")
    _convert(prosody, (d, t_en, aln, ref_s, fm), ["d", "t_en", "aln", "ref_s", "frame_mask"],
             ["asr", "F0", "N"], out_dir / "kokoro_prosody.aimodel")
    _convert(vocoder, (asr, F0, N, har, ref_s, fm), ["asr", "F0", "N", "har", "ref_s", "frame_mask"],
             ["audio"], out_dir / "kokoro_vocoder.aimodel")
    print(f"[done] 3 bundles in {out_dir}  (token bucket {tb}, frame bucket {lb})")


async def _synth_engine(out_dir, text, voice, tb=TB, lb=LB):
    import coreai.runtime as rt
    model = build_model()
    vocab = get_vocab()
    ids, ps = text_to_ids(text, vocab)
    T = ids.shape[-1]
    ref_s = load_voice(voice)[T - 1]

    async def load(name):
        m = await rt.AIModel.load(out_dir / name, rt.SpecializationOptions.cpu_only())
        return m.load_function("main")
    fp, fpr, fv = await load("kokoro_predictor.aimodel"), await load("kokoro_prosody.aimodel"), await load("kokoro_vocoder.aimodel")

    ids_b = torch.cat([ids, torch.zeros((1, tb - T), dtype=torch.long)], dim=1)
    attn = torch.cat([torch.ones((1, T)), torch.zeros((1, tb - T))], dim=1)
    o1 = await fp({"input_ids": rt.NDArray(ids_b.numpy().astype(np.int32)),
                   "ref_s": rt.NDArray(ref_s.numpy()), "attn_mask": rt.NDArray(attn.numpy())})
    dur = torch.from_numpy(o1["duration"].numpy())
    aln, fm, L = build_alignment(dur[:, :T], 1.0, tb, lb)
    o2 = await fpr({"d": rt.NDArray(o1["d"].numpy()), "t_en": rt.NDArray(o1["t_en"].numpy()),
                    "aln": rt.NDArray(aln.numpy()), "ref_s": rt.NDArray(ref_s.numpy()),
                    "frame_mask": rt.NDArray(fm.numpy())})
    har = compute_har(torch.from_numpy(o2["F0"].numpy()), model)
    o3 = await fv({"asr": rt.NDArray(o2["asr"].numpy()), "F0": rt.NDArray(o2["F0"].numpy()),
                   "N": rt.NDArray(o2["N"].numpy()), "har": rt.NDArray(har.numpy()),
                   "ref_s": rt.NDArray(ref_s.numpy()), "frame_mask": rt.NDArray(fm.numpy())})
    audio = o3["audio"].numpy()[: L * 600]

    with torch.no_grad():
        ref, _ = model.forward_with_tokens(ids, ref_s, 1.0)
    return audio, ref.numpy(), ps


def _magspec(x):
    X = torch.stft(torch.from_numpy(x.astype(np.float32)), 512, 128, 512,
                   window=torch.hann_window(512), return_complex=True)
    return X.abs().numpy()


def verify(out_dir, text, voice):
    import soundfile as sf
    audio, ref, ps = asyncio.run(_synth_engine(out_dir, text, voice))
    n = min(len(audio), len(ref))
    a, b = audio[:n].astype(np.float64), ref[:n].astype(np.float64)
    A, B = _magspec(a), _magspec(b)
    m = min(A.shape[1], B.shape[1])
    mcorr = float(np.corrcoef(A[:, :m].flatten(), B[:, :m].flatten())[0, 1])
    wcorr = float(np.corrcoef(a, b)[0, 1])
    sf.write("kokoro_engine.wav", audio.astype(np.float32), 24000, subtype="FLOAT")
    print(f"phonemes: {ps!r}")
    print(f"[verify:{voice}] {len(audio) / 24000:.2f}s  wav-corr={wcorr:.4f}  magspec-corr={mcorr:.4f}  "
          f"-> {'PASS' if mcorr > 0.99 else 'FAIL'}  (wrote kokoro_engine.wav)")
    return mcorr > 0.99


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="exports/kokoro")
    ap.add_argument("--token-bucket", type=int, default=TB)
    ap.add_argument("--frame-bucket", type=int, default=LB)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--text", default="Hello, world. This is a test of on device speech generation.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if args.verify:
        raise SystemExit(0 if verify(out_dir, args.text, args.voice) else 1)
    export_bundles(out_dir, args.token_bucket, args.frame_bucket)


if __name__ == "__main__":
    main()
