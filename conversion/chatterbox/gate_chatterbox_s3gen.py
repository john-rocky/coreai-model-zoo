"""S3Gen Mac-engine gate: each exported Core AI graph vs its eager output (export fidelity),
+ the patched-eager full pipeline vs the oracle wav (patch is lossless). Run in the coreai venv
(chatterbox installed via uv).

Reads chatterbox_oracle.npz from $SCRATCH: set SCRATCH to the folder that holds it (oracle_chatterbox.py
writes it there; unset: chatterbox_scratch under the system temp folder).
"""
import asyncio
import os
import numpy as np
import torch
import torch.nn.functional as F

import coreai.runtime as rt
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import exports_dir  # noqa: E402

EXPORTS = str(exports_dir())
ORACLE = os.path.join(os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "chatterbox_scratch"), "chatterbox_oracle.npz")


def patch():
    import perth
    class _NW:
        def apply_watermark(self, w, sample_rate=None, **k):
            return w
    perth.PerthImplicitWatermarker = _NW
    import chatterbox.models.s3gen.decoder as dec
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask
    def _static(xs, masks, a, b, c, scs, ndlc):
        if (not a) and scs > 0:
            return masks & subsequent_chunk_mask(xs.size(1), scs, ndlc, xs.device).unsqueeze(0)
        return masks
    dec.add_optional_chunk_mask = _static


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


async def run_graph(name, inputs: dict):
    bundle = f"{EXPORTS}/{name}/{name}.aimodel"
    m = await rt.AIModel.load(bundle, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.cpu()))
    fn = m.load_function("main")
    res = await fn(inputs={k: rt.NDArray(np.ascontiguousarray(v)) for k, v in inputs.items()})
    return res


async def main():
    patch()
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cpu")
    s3 = m.s3gen
    d = np.load(ORACLE)
    sp = torch.tensor(d["speech_tokens"]).reshape(1, -1)

    # capture eager inputs+outputs of the 3 nets during one patched inference
    cap = {}
    est = s3.flow.decoder.estimator
    oe = est.forward
    def est_spy(x, mask, mu, t, spks=None, cond=None, r=None):
        out = oe(x, mask, mu, t, spks, cond, r)
        cap.setdefault("est", (dict(x=x, mask=mask, mu=mu, t=t if torch.is_tensor(t) else torch.tensor([t]), spks=spks, cond=cond), out))
        return out
    est.forward = est_spy
    os_ = s3.mel2wav._stft
    def stft_spy(x):
        rr, ii = os_(x); cap.setdefault("s_stft", torch.cat([rr, ii], 1)); return rr, ii
    s3.mel2wav._stft = stft_spy
    enc_in = {}
    h = s3.flow.encoder.register_forward_pre_hook(lambda mod, inp: enc_in.setdefault("in", inp))
    def mel_spy(speech_feat, cache_source=None, **k):
        cap.setdefault("mel", speech_feat); return omw(speech_feat=speech_feat, cache_source=cache_source, **k)
    omw = s3.mel2wav.inference; s3.mel2wav.inference = mel_spy
    torch.manual_seed(0)
    wav_patched, _ = s3.inference(speech_tokens=sp, ref_dict=m.conds.gen, n_cfm_timesteps=10)
    est.forward = oe; s3.mel2wav._stft = os_; s3.mel2wav.inference = omw; h.remove()

    # 1) patched-eager pipeline vs oracle wav (lossless patch check)
    wav_oracle = torch.tensor(d["wav"])
    n = min(wav_patched.shape[-1], wav_oracle.shape[-1])
    print(f"[patch lossless] patched-pipeline wav vs oracle wav: cos={cos(wav_patched[...,:n], wav_oracle[...,:n]):.6f}")

    # 2) estimator graph vs eager
    ei, eo = cap["est"]
    ein = {k: ei[k].detach().numpy().astype(np.float32) for k in ("x", "mask", "mu", "t", "spks", "cond")}
    r = await run_graph("chatterbox_s3gen_estimator", ein)
    vg = torch.from_numpy(r["velocity"].numpy())
    print(f"[estimator] engine vs eager velocity: cos={cos(vg, eo):.6f}")

    # 3) encoder graph vs eager
    xs, xs_lens = enc_in["in"][0], enc_in["in"][1]
    with torch.no_grad():
        mu_eager = s3.flow.encoder_proj(s3.flow.encoder(xs, xs_lens)[0])
    r = await run_graph("chatterbox_s3gen_encoder",
                        {"xs": xs.detach().numpy().astype(np.float32), "xs_lens": xs_lens.detach().numpy()})
    mu_g = torch.from_numpy(r["mu"].numpy())
    print(f"[encoder] engine vs eager mu: cos={cos(mu_g, mu_eager):.6f}")

    # 4) hift_trunk graph vs eager
    g = s3.mel2wav; mel = cap["mel"]; s_stft = cap["s_stft"]
    nfft = g.istft_params["n_fft"]
    def trunk_eager(mel, s_stft):
        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, g.lrelu_slope); x = g.ups[i](x)
            if i == g.num_upsamples - 1: x = g.reflection_pad(x)
            si = g.source_resblocks[i](g.source_downs[i](s_stft)); x = x + si
            xs2 = None
            for j in range(g.num_kernels):
                rr = g.resblocks[i * g.num_kernels + j](x); xs2 = rr if xs2 is None else xs2 + rr
            x = xs2 / g.num_kernels
        x = g.conv_post(F.leaky_relu(x))
        return torch.exp(x[:, :nfft // 2 + 1]), torch.sin(x[:, nfft // 2 + 1:])
    mag_e, ph_e = trunk_eager(mel.float(), s_stft.float())
    r = await run_graph("chatterbox_s3gen_hift_trunk",
                        {"mel": mel.detach().numpy().astype(np.float32), "s_stft": s_stft.detach().numpy().astype(np.float32)})
    mag_g = torch.from_numpy(r["magnitude"].numpy()); ph_g = torch.from_numpy(r["phase"].numpy())
    print(f"[hift_trunk] engine vs eager: mag cos={cos(mag_g, mag_e):.6f} phase cos={cos(ph_g, ph_e):.6f}")


asyncio.run(main())
