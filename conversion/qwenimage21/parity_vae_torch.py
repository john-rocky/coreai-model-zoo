"""Gate: the VAE decode wrapper (``qi21_vae.QI21VAEDecode``) vs the pipeline's own decode, fp32 CPU torch.

Per oracle (``oracle/256``, ``oracle/512``):

  (a) ``vae.decode(vae_in)`` (the pipeline call, feature cache on)  vs the oracle's ``vae_out``
      — the reference reproduces on this machine.
  (b) ``clamp(decoder(post_quant_conv(vae_in), first_chunk=True))`` with NO feature cache vs (a)
      — one frame decodes the same with and without the cache.            bar: bit-exact
  (c) (b) with ``patch_nearest_upsample`` (repeat_interleave for nearest-exact x2) vs (a)
                                                                          bar: bit-exact
  (d) the wrapper on ``final_latents`` (unpack + ``*std+mean`` in-graph): its un-normalised latent
      vs ``vae_in`` and its image vs ``vae_out[:, :, 0]``                  bar: bit-exact
  red controls (must FAIL the engine bar corr >= 0.9999 / max|d| <= 1e-2 — shows the bar sees them):
      r1 ``first_chunk=False`` (the DupUp3D time axis is not sliced back to one frame)
      r2 no un-normalisation (normalised latents straight into decode)
      r3 un-normalisation applied twice (the Z-Image Swift bug)
      r4 unpack without the transpose (``reshape`` of the token-major layout)

Also prints the smallest channel L2 norm any ``QwenImage21RMS_norm`` sees (``F.normalize`` clamps
the denominator at 1e-12; the export keeps ``F.normalize`` as long as that clamp is never active).

``--synthetic 1024`` instead writes a torch reference for a size with no oracle, in the oracle's
file layout (``_work/vae_ref_1024/``: meta.json, final_latents.f32, vae_out.f32): the 512 oracle's
``final_latents`` nearest-upsampled 2x on the token grid, decoded by the pipeline's own
``vae.decode`` (un-normalised the pipeline's way). NOT an oracle image — a realistic latent for the
engine gate at that size. ``--synthetic 512 --latents <final_latents.f32> --name <n>`` does the same
for any saved sampler output (e.g. ``pipeline_engine.py``'s RGBA sticker, whose alpha reaches 0 —
the oracle images are opaque) -> ``_work/vae_ref_<n>/``.

Run (``.venv-qi21``, from conversion/qwenimage21/):
  python parity_vae_torch.py [--oracle oracle/256 oracle/512]
  python parity_vae_torch.py --synthetic 1024
  python parity_vae_torch.py --synthetic 512 --latents _work/samples/<stem>_final_latents.f32 --name sticker512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qi21_vae import QI21VAEDecode, load_vae, patch_nearest_upsample  # noqa: E402

BAR_CORR, BAR_MAXD = 0.9999, 1e-2


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    if tuple(a.shape) != tuple(b.shape):
        return dict(shape=[list(a.shape), list(b.shape)], exact=False, corr=float("nan"), maxd=float("nan"))
    x = a.detach().double().numpy().ravel()
    y = b.detach().double().numpy().ravel()
    return dict(exact=bool(torch.equal(a, b)), maxd=float(np.abs(x - y).max()), corr=float(np.corrcoef(x, y)[0, 1]))


def fmt(r: dict) -> str:
    if "shape" in r:
        return f"shape mismatch {r['shape'][0]} vs {r['shape'][1]}"
    return f"bit-exact {r['exact']}  max|d| {r['maxd']:.3e}  corr {r['corr']:.9f}"


def load(od: Path, name: str, shape) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(od / f"{name}.f32", "<f4").reshape(shape))


def synthetic(vae, size: int, latents: str | None = None, name: str | None = None) -> int:
    if latents:
        h = w = size // 16
        final = torch.from_numpy(np.fromfile(latents, "<f4").reshape(1, h * w, 64))
        source = f"{latents} (saved sampler output); torch fp32 vae.decode (reference for the engine VAE, not an oracle)"
    else:
        src = HERE / "oracle" / "512"
        m = json.load(open(src / "meta.json"))
        _, h0, w0 = (int(v) for v in m["img_shapes"][-1])
        f = size // (16 * h0)
        assert f >= 1 and h0 * 16 * f == size, (size, h0)
        h, w = h0 * f, w0 * f
        grid = load(src, "final_latents", (1, h0, w0, 64))
        final = grid.repeat_interleave(f, dim=1).repeat_interleave(f, dim=2).reshape(1, h * w, 64).contiguous()
        source = (f"oracle/512 final_latents nearest-upsampled x{f} on the token grid; torch fp32 vae.decode "
                  f"(synthetic reference, not an oracle)")
    z = final.transpose(1, 2).reshape(1, 64, 1, h, w)
    z_dim = vae.config.z_dim
    z = (z * torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
         + torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1))
    t1 = time.time()
    with torch.no_grad():
        out = vae.decode(z, return_dict=False)[0]
    sec = time.time() - t1
    od = HERE / "_work" / f"vae_ref_{name or size}"
    od.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(final.numpy(), "<f4").tofile(od / "final_latents.f32")
    np.ascontiguousarray(out.numpy(), "<f4").tofile(od / "vae_out.f32")
    json.dump(dict(size=size, img_shapes=[[1, h, w]], vae_out_shape=list(out.shape), decode_s=sec, source=source),
              open(od / "meta.json", "w"), indent=2)
    print(f"[vae-torch] synthetic {size}: latent {h}x{w}, vae.decode {sec:.1f}s, out {list(out.shape)} "
          f"range {float(out.min()):+.3f}..{float(out.max()):+.3f}, alpha min {float(out[:, 3].min()):+.4f} -> {od}",
          flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", nargs="+", default=[str(HERE / "oracle" / "256"), str(HERE / "oracle" / "512")])
    ap.add_argument("--json", default=str(HERE / "_work" / "parity_vae_torch.json"))
    ap.add_argument("--synthetic", type=int, default=0, help="write a torch reference for this size (see above)")
    ap.add_argument("--latents", default=None, help="with --synthetic: a saved final_latents [1,N,64] f32 file")
    ap.add_argument("--name", default=None, help="with --synthetic: _work/vae_ref_<name>/ (default: the size)")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    vae = load_vae()
    print(f"[vae-torch] loaded fp32 in {time.time() - t0:.1f}s", flush=True)
    if args.synthetic:
        return synthetic(vae, args.synthetic, args.latents, args.name)
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21RMS_norm

    min_norm = [float("inf")]

    def norm_hook(mod, inp, out):
        x = inp[0]
        min_norm[0] = min(min_norm[0], float(x.float().norm(dim=1 if mod.channel_first else -1).min()))

    res, ok, unpatched = {}, True, {}
    for o in args.oracle:
        od = Path(o)
        meta = json.load(open(od / "meta.json"))
        _, h, w = (int(v) for v in meta["img_shapes"][-1])
        N, C = h * w, 64
        vae_in = load(od, "vae_in", meta["vae_in_shape"])
        vae_out = load(od, "vae_out", meta["vae_out_shape"])
        final = load(od, "final_latents", (1, N, C))
        r = dict(size=meta["size"], lat=[h, w])
        with torch.no_grad():
            t1 = time.time()
            a = vae.decode(vae_in, return_dict=False)[0]
            r["decode_s"] = time.time() - t1
            r["a_pipeline_decode_vs_oracle"] = cmp(a, vae_out)
            hooks = [m.register_forward_hook(norm_hook) for m in vae.decoder.modules()
                     if isinstance(m, QwenImage21RMS_norm)]
            b = torch.clamp(vae.decoder(vae.post_quant_conv(vae_in), first_chunk=True), -1.0, 1.0)
            for hk in hooks:
                hk.remove()
            r["b_nocache_vs_pipeline"] = cmp(b, a)
            r["rms_norm_min_channel_l2"] = min_norm[0]
            try:
                r1 = torch.clamp(vae.decoder(vae.post_quant_conv(vae_in), first_chunk=False), -1.0, 1.0)
                r["r1_first_chunk_false"] = cmp(r1, a)
            except Exception as e:                        # the doubled time axis reaches a Conv2d
                r["r1_first_chunk_false"] = dict(error=f"{type(e).__name__}: {str(e)[:160]}")
            wrap = QI21VAEDecode(vae, h, w).eval()
            zn = wrap.unnormalize(final)
            r["d_unnormalize_vs_vae_in"] = cmp(zn, vae_in)
            img0 = wrap(final)
            unpatched[od.name] = img0
            r["d0_wrapper_unpatched_vs_vae_out"] = cmp(img0, vae_out[:, :, 0])
            ctrl = {}
            z_norm = final.transpose(1, 2).reshape(1, C, 1, h, w)
            dec = lambda z: torch.clamp(vae.decode(z, return_dict=False)[0], -1.0, 1.0)[:, :, 0]  # noqa: E731
            ctrl["r2_no_unnormalize"] = cmp(dec(z_norm), vae_out[:, :, 0])
            ctrl["r3_unnormalize_twice"] = cmp(dec(zn * wrap.std + wrap.mean), vae_out[:, :, 0])
            ctrl["r4_unpack_without_transpose"] = cmp(dec(final.reshape(1, C, 1, h, w) * wrap.std + wrap.mean),
                                                      vae_out[:, :, 0])
            r["controls"] = ctrl
        res[od.name] = r
        print(f"[vae-torch] {od.name}: lat {h}x{w} -> {meta['vae_out_shape']}  pipeline decode {r['decode_s']:.1f}s",
              flush=True)
        print(f"  (a) pipeline decode vs oracle vae_out:           {fmt(r['a_pipeline_decode_vs_oracle'])}", flush=True)
        print(f"  (b) no feature cache, first_chunk=True vs (a):   {fmt(r['b_nocache_vs_pipeline'])}", flush=True)
        print(f"  (d) wrapper unnormalize(final_latents) vs vae_in: {fmt(r['d_unnormalize_vs_vae_in'])}", flush=True)
        print(f"  (d) wrapper image (unpatched) vs vae_out[:,:,0]: {fmt(r['d0_wrapper_unpatched_vs_vae_out'])}", flush=True)
        print(f"  RMS_norm smallest channel L2 norm seen: {r['rms_norm_min_channel_l2']:.3e} (clamp at 1e-12)", flush=True)
        r1 = r["r1_first_chunk_false"]
        print(f"  r1 first_chunk=False: {r1['error'] if 'error' in r1 else fmt(r1)}  (red)", flush=True)
        for k, v in ctrl.items():
            red = "shape" in v or not (v["corr"] >= BAR_CORR and v["maxd"] <= BAR_MAXD)
            v["red"] = red
            print(f"  {k}: {fmt(v)}  {'RED (bar sees it)' if red else 'NOT RED'}", flush=True)
        ok &= (r["a_pipeline_decode_vs_oracle"]["exact"] or r["a_pipeline_decode_vs_oracle"]["maxd"] <= 1e-6)
        ok &= r["b_nocache_vs_pipeline"]["exact"] and r["d_unnormalize_vs_vae_in"]["exact"]
        ok &= all(v["red"] for v in ctrl.values()) and ("error" in r1 or not r1.get("exact", True))

    # (c) the nearest-exact -> repeat_interleave patch, then the wrapper again on every oracle
    n = patch_nearest_upsample(vae.decoder)
    print(f"[vae-torch] patched {n} QwenImage21Upsample modules -> repeat_interleave", flush=True)
    for o in args.oracle:
        od = Path(o)
        meta = json.load(open(od / "meta.json"))
        _, h, w = (int(v) for v in meta["img_shapes"][-1])
        vae_out = load(od, "vae_out", meta["vae_out_shape"])
        final = load(od, "final_latents", (1, h * w, 64))
        with torch.no_grad():
            img = QI21VAEDecode(vae, h, w).eval()(final)
        r = res[od.name]
        r["c_wrapper_patched_vs_vae_out"] = cmp(img, vae_out[:, :, 0])
        r["c_patched_vs_unpatched"] = cmp(img, unpatched[od.name])
        print(f"  {od.name} (c) wrapper, patched upsample vs unpatched: {fmt(r['c_patched_vs_unpatched'])}", flush=True)
        print(f"  {od.name} (c) wrapper, patched upsample vs vae_out[:,:,0]: {fmt(r['c_wrapper_patched_vs_vae_out'])}",
              flush=True)
        ok &= r["c_patched_vs_unpatched"]["exact"] and r["c_wrapper_patched_vs_vae_out"]["exact"]
        ok &= r["d0_wrapper_unpatched_vs_vae_out"]["exact"]
    res["ok"] = bool(ok)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.json, "w"), indent=2)
    print(f"\n[vae-torch] {'PASS' if ok else 'FAIL'} (bit-exact (b)(c)(d), controls red) -> {args.json}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
