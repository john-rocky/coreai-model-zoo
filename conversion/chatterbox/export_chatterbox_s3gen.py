"""Direct torch->Core AI export of the Chatterbox S3Gen sub-nets (Stable-Audio style).

S3Gen = flow.encoder (UpsampleConformerEncoder: token-embeds -> mu) + flow.decoder.estimator
(ConditionalDecoder UNet1D CFM velocity, called N times by the host Euler loop) + mel2wav
(HiFTGenerator vocoder: mel -> wav). We export each as a Core AI graph and run the flow-matching
Euler loop + AR host-side. The only export snag is the Conformer's streaming chunk mask
(`add_optional_chunk_mask` has a data-dependent `.item()`), patched to a static form here.

Run (in the coreai venv WITH chatterbox installed via uv):
  cd ~/code/coreai/coreai-models && HF_HUB_DISABLE_XET=1 .venv/bin/python \
    ../coreai-models-community/conversion/chatterbox/export_chatterbox_s3gen.py --out-dir exports

The oracle comes from --oracle, default $SCRATCH/chatterbox_oracle.npz: set SCRATCH to the folder that holds it
(oracle_chatterbox.py writes it there; unset: chatterbox_scratch under the system temp folder).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def patch_chunk_mask():
    """Replace the streaming chunk-mask helper (data-dependent .item()) with a static one
    everywhere it is imported. Numerically identical for our non-streaming full inference."""
    import chatterbox.models.s3gen.decoder as dec
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask

    def _static(xs, masks, use_dynamic_chunk, enable_full_context,
                decoding_chunk_size, static_chunk_size, num_decoding_left_chunks):
        if (not use_dynamic_chunk) and static_chunk_size > 0:
            cm = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                       num_decoding_left_chunks, xs.device).unsqueeze(0)
            return masks & cm
        return masks

    dec.add_optional_chunk_mask = _static
    for modname in ("chatterbox.models.s3gen.transformer.upsample_encoder",
                    "chatterbox.models.s3gen.transformer.encoder"):
        try:
            import importlib
            mod = importlib.import_module(modname)
            if hasattr(mod, "add_optional_chunk_mask"):
                mod.add_optional_chunk_mask = _static
        except Exception:
            pass


def save_graph(prog, out_dir: Path, name: str):
    import coreai.runtime as rt
    d = out_dir / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    prog.save_asset(d / f"{name}.aimodel", rt.AIModelAssetMetadata())
    return d


def main():
    import perth
    class _NW:
        def apply_watermark(self, w, sample_rate=None, **k):
            return w
    perth.PerthImplicitWatermarker = _NW
    patch_chunk_mask()

    from chatterbox.tts import ChatterboxTTS
    from coreai_models.export.macos import export_to_coreai

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--oracle", default=os.path.join(os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "chatterbox_scratch"),
                    "chatterbox_oracle.npz"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    m = ChatterboxTTS.from_pretrained(device="cpu")
    s3 = m.s3gen
    d = np.load(args.oracle)
    sp = torch.tensor(d["speech_tokens"]).reshape(1, -1)

    # Capture the real call args of each sub-net during one inference.
    cap: dict = {}
    est = s3.flow.decoder.estimator
    orig_est = est.forward

    def est_spy(x, mask, mu, t, spks=None, cond=None, r=None):
        cap.setdefault("est", {k: (v.detach().clone() if torch.is_tensor(v) else torch.tensor([v]))
                               for k, v in dict(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond).items()})
        return orig_est(x, mask, mu, t, spks, cond, r)
    est.forward = est_spy

    mel_cap: dict = {}
    orig_hift = s3.mel2wav.inference

    def hift_spy(speech_feat, cache_source=None, **k):
        mel_cap.setdefault("mel", speech_feat.detach().clone())
        return orig_hift(speech_feat=speech_feat, cache_source=cache_source, **k)
    s3.mel2wav.inference = hift_spy
    # capture the REAL s_stft (source STFT) the trunk fuses — its time res is set by the
    # full-waveform source, not mel x hop, so we trace with the genuine tensor.
    orig_stft = s3.mel2wav._stft

    def stft_spy(x):
        r, i = orig_stft(x)
        mel_cap.setdefault("s_stft", torch.cat([r, i], dim=1).detach().clone())
        return r, i
    s3.mel2wav._stft = stft_spy
    # capture the encoder's forward inputs (pre-hook = doesn't touch output)
    enc_cap: dict = {}

    def enc_pre(mod, inp):
        enc_cap.setdefault("in", tuple(t.detach().clone() if torch.is_tensor(t) else t for t in inp))
    h_enc = s3.flow.encoder.register_forward_pre_hook(enc_pre)

    torch.manual_seed(0)
    s3.inference(speech_tokens=sp, ref_dict=m.conds.gen, n_cfm_timesteps=10)
    h_enc.remove()
    est.forward = orig_est
    s3.mel2wav.inference = orig_hift
    s3.mel2wav._stft = orig_stft

    results = {}

    # 1) estimator (UNet1D CFM velocity). Export via a RECONSTRUCTED forward: the real
    # ConditionalDecoder.forward uses `mask_to_bias(attn_mask == 1, ...)` which traces to a
    # subtly wrong attention bias (cos 0.899 vs eager); rebuilding the same forward with
    # `mask_to_bias(mask.bool(), ...)` exports faithfully (cos 1.000000 vs the real velocity).
    from einops import pack, repeat, rearrange
    from chatterbox.models.s3gen.decoder import mask_to_bias
    ex = {k: cap["est"][k].float() for k in ("x", "mask", "mu", "t", "spks", "cond")}

    class ReconEstimator(torch.nn.Module):
        def __init__(s, e):
            super().__init__(); s.e = e
        def forward(s, x, mask, mu, t, spks, cond):
            e = s.e
            t = e.time_mlp(e.time_embeddings(t))
            x = pack([x, mu], "b * t")[0]
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1]); x = pack([x, spks], "b * t")[0]
            x = pack([x, cond], "b * t")[0]
            hiddens = []; masks = [mask]
            for resnet, tbs, down in e.down_blocks:
                md = masks[-1]; x = resnet(x, md, t)
                x = rearrange(x, "b c t -> b t c").contiguous(); am = mask_to_bias(md.bool(), x.dtype)
                for tb in tbs:
                    x = tb(hidden_states=x, attention_mask=am, timestep=t)
                x = rearrange(x, "b t c -> b c t").contiguous(); hiddens.append(x)
                x = down(x * md); masks.append(md[:, :, ::2])
            masks = masks[:-1]; mm = masks[-1]
            for resnet, tbs in e.mid_blocks:
                x = resnet(x, mm, t); x = rearrange(x, "b c t -> b t c").contiguous()
                am = mask_to_bias(mm.bool(), x.dtype)
                for tb in tbs:
                    x = tb(hidden_states=x, attention_mask=am, timestep=t)
                x = rearrange(x, "b t c -> b c t").contiguous()
            for resnet, tbs, up in e.up_blocks:
                mu_ = masks.pop(); sk = hiddens.pop()
                x = pack([x[:, :, :sk.shape[-1]], sk], "b * t")[0]
                x = resnet(x, mu_, t); x = rearrange(x, "b c t -> b t c").contiguous()
                am = mask_to_bias(mu_.bool(), x.dtype)
                for tb in tbs:
                    x = tb(hidden_states=x, attention_mask=am, timestep=t)
                x = rearrange(x, "b t c -> b c t").contiguous(); x = up(x * mu_)
            x = e.final_block(x, mask); out = e.final_proj(x * mask)
            return out * mask
    try:
        prog = export_to_coreai(ReconEstimator(est).eval().float(), ex,
                                input_names=tuple(ex.keys()), output_names=("velocity",))
        prog.optimize()
        save_graph(prog, out_dir, "chatterbox_s3gen_estimator")
        results["estimator"] = "OK (reconstructed forward, cos 1.0 verified)"
    except Exception as e:
        results["estimator"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

    # 2) HiFT vocoder TRUNK (Kokoro idiom): the conv net minus the FFT ops. Host does the
    # harmonic+noise source (m_source, has rand/unfold), both STFTs (_fft_r2c / _fft_c2r), and
    # the final iSTFT. The Core AI trunk takes (mel, s_stft) -> (magnitude, phase).
    import torch.nn.functional as F
    g = s3.mel2wav
    nfft = g.istft_params["n_fft"]

    class HiftTrunk(torch.nn.Module):
        def __init__(s, g):
            super().__init__(); s.g = g
        def forward(s, mel, s_stft):
            gg = s.g
            x = gg.conv_pre(mel)
            for i in range(gg.num_upsamples):
                x = F.leaky_relu(x, gg.lrelu_slope)
                x = gg.ups[i](x)
                if i == gg.num_upsamples - 1:
                    x = gg.reflection_pad(x)
                si = gg.source_downs[i](s_stft)
                si = gg.source_resblocks[i](si)
                x = x + si
                xs = None
                for j in range(gg.num_kernels):
                    r = gg.resblocks[i * gg.num_kernels + j](x)
                    xs = r if xs is None else xs + r
                x = xs / gg.num_kernels
            x = F.leaky_relu(x)
            x = gg.conv_post(x)
            magnitude = torch.exp(x[:, : nfft // 2 + 1, :])
            phase = torch.sin(x[:, nfft // 2 + 1:, :])
            return magnitude, phase

    mel = mel_cap["mel"].float()
    s_stft_ref = mel_cap["s_stft"].float()  # the genuine source STFT captured from the oracle run
    try:
        with torch.no_grad():
            _mag, _ph = HiftTrunk(g).eval().float()(mel, s_stft_ref)
        prog = export_to_coreai(HiftTrunk(g).eval().float(), {"mel": mel, "s_stft": s_stft_ref},
                                input_names=("mel", "s_stft"), output_names=("magnitude", "phase"))
        prog.optimize()
        save_graph(prog, out_dir, "chatterbox_s3gen_hift_trunk")
        results["hift_trunk"] = f"OK (mag {tuple(_mag.shape)} phase {tuple(_ph.shape)})"
    except Exception as e:
        import traceback
        results["hift_trunk"] = f"ERR {type(e).__name__}: {traceback.format_exc()[-220:]}"

    # 3) encoder (UpsampleConformerEncoder) + encoder_proj: token-embeds -> mu
    enc_in = enc_cap["in"]
    print("encoder inputs:", [tuple(t.shape) if torch.is_tensor(t) else t for t in enc_in])

    class EncW(torch.nn.Module):
        def __init__(s, enc, proj):
            super().__init__(); s.enc = enc; s.proj = proj
        def forward(s, xs, xs_lens):
            h, _ = s.enc(xs, xs_lens)
            return s.proj(h)
    ex_e = {"xs": enc_in[0].float(), "xs_lens": enc_in[1]}
    try:
        with torch.no_grad():
            _mu = EncW(s3.flow.encoder, s3.flow.encoder_proj).eval().float()(**ex_e)
        prog = export_to_coreai(EncW(s3.flow.encoder, s3.flow.encoder_proj).eval().float(), ex_e,
                                input_names=("xs", "xs_lens"), output_names=("mu",))
        prog.optimize()
        save_graph(prog, out_dir, "chatterbox_s3gen_encoder")
        results["encoder"] = f"OK (mu {tuple(_mu.shape)})"
    except Exception as e:
        import traceback
        results["encoder"] = f"ERR {type(e).__name__}: {traceback.format_exc()[-220:]}"

    for k, v in results.items():
        print(f"[{k}] {v}")
    print("done.")


if __name__ == "__main__":
    main()
