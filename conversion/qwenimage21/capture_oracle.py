"""Capture the Qwen-Image-2.1 ground-truth oracle for the Core AI port.

Runs the fp32 diffusers ``QwenImage21Pipeline`` (diffusers main, see meta.json for
the commit) once on CPU and records, at the three graph boundaries, everything the
Core AI graphs must reproduce:

  encoder   enc_input_ids   [1,Lfull] int32  t2i chat template, batch 1 => no padding
            enc_hidden_full [1,Lfull,4096]   last decoder layer output BEFORE the
                                             text encoder's final RMSNorm
            prompt_embeds   [1,L,4096]       = enc_hidden_full[:, drop_idx:]  (what
                                             the DiT reads; L = Lfull - drop_idx)
  DiT       per step s (teacher forcing):
            latent_s [1,N,64]  packed latents fed to the transformer at step s
            t_s      scalar    timestep / 1000 as the transformer sees it
            vel_s    [1,N,64]  transformer output over the N target-image tokens
  VAE       final_latents [1,N,64] -> vae_in [1,64,1,h,w] (un-normalised, what
            vae.decode receives) -> vae_out [1,4,1,H,W] fp32 in [-1,1]
  image_ref.png (RGBA) / image_ref_rgb.png (alpha composited on white)
  noise0 [1,N,64], sigmas [steps+1], timesteps [steps], meta.json

N = (size/16)^2 image tokens (one token per 16x16 pixel tile; the 2x2 grouping the
transformer does internally is invisible at this boundary). ``--no-kv-cache`` runs
the reference with the prefix KV cache off, which must give the same tensors — the
exported graph recomputes the prefix every step, so that equality is part of what
this oracle proves.

Run (qi21 venv, from conversion/qwenimage21/):
  python capture_oracle.py --size 256 --steps 40
  python capture_oracle.py --size 512 --steps 40
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot  # noqa: E402

MODEL = "Qwen/Qwen-Image-2.1"
DEFAULT_PROMPT = "a red apple on a wooden table, studio lighting"


def savef(t, name, out):
    a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)
    np.ascontiguousarray(a, "<f4").tofile(out / f"{name}.f32")
    return list(a.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tag", default=None, help="oracle/<tag>/ (default: the size)")
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="run the reference with the prefix KV cache off (must match)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--snapshot", default=None,
                    help="pipeline directory override (smoke tests on a tiny random pipeline)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tag = args.tag or (f"{args.size}" + ("_nocache" if args.no_kv_cache else ""))
    out = HERE / "oracle" / tag
    out.mkdir(parents=True, exist_ok=True)

    import diffusers
    import transformers
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    from diffusers.utils.torch_utils import randn_tensor

    snap = Path(args.snapshot) if args.snapshot else Path(hf_snapshot(MODEL))
    print(f"[oracle] snapshot {snap.name}  diffusers {diffusers.__version__}  "
          f"transformers {transformers.__version__}  torch {torch.__version__}", flush=True)
    print("[oracle] loading pipeline (fp32, cpu) ...", flush=True)
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(str(snap), torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    print(f"[oracle] loaded in {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- encoder
    prompts = [pipe.prompt_template_t2i.format(args.prompt)]
    mi = pipe.processor(text=prompts, padding=True, padding_side="left", return_tensors="pt")
    ids, am = mi.input_ids, mi.attention_mask
    assert bool(am.all()), "batch 1 must not be padded"
    te = pipe.text_encoder
    text_model = getattr(te.model, "language_model", te.model)
    fwd_kwargs = dict(input_ids=ids, attention_mask=am, output_hidden_states=True)
    if hasattr(mi, "mm_token_type_ids"):
        fwd_kwargs["mm_token_type_ids"] = mi.mm_token_type_ids
    handle = text_model.norm.register_forward_hook(lambda module, a, o: a[0])
    with torch.no_grad():
        h_full = te(**fwd_kwargs).hidden_states[-1]
    handle.remove()
    with torch.no_grad():
        pe, pem, ipm = pipe.encode_prompt(prompt=args.prompt, device=torch.device("cpu"))
    drop = int(pipe._drop_idx)
    assert pe.shape[1] == h_full.shape[1] - drop, (pe.shape, h_full.shape, drop)
    assert torch.equal(pe, h_full[:, drop:]), "pipeline embeds != last-layer pre-norm hidden[drop:]"
    assert pem is None and not bool(ipm.any())
    L, Lfull = int(pe.shape[1]), int(h_full.shape[1])
    np.ascontiguousarray(ids.numpy().astype("<i4")).tofile(out / "enc_input_ids.i32")
    savef(h_full, "enc_hidden_full", out)
    savef(pe, "prompt_embeds", out)
    print(f"[oracle] encoder: Lfull={Lfull} drop_idx={drop} L={L}  "
          f"|h| max {float(h_full.abs().max()):.1f}", flush=True)

    # ------------------------------------------------------------------ hooks
    tr = pipe.transformer
    real_fwd = tr.forward
    rec = []

    def fwd_hook(hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask, **kw):
        t1 = time.time()
        r = real_fwd(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                     timestep=timestep, img_shapes=img_shapes, img_mask=img_mask, **kw)
        sample = r[0] if isinstance(r, tuple) else r.sample
        n = hidden_states.shape[1]
        rec.append(dict(latent=hidden_states.detach().clone(),
                        t=float(timestep.reshape(-1)[0]),
                        vel=sample[:, -n:].detach().clone(),
                        mode=kw.get("kv_cache_mode"), sec=time.time() - t1,
                        img_shapes=img_shapes, img_mask=img_mask.detach().clone(),
                        ehs=tuple(encoder_hidden_states.shape),
                        ehs_mask=kw.get("encoder_hidden_states_mask")))
        return r

    tr.forward = fwd_hook

    real_step = pipe.scheduler.step
    traj = []

    def step_hook(model_output, timestep, sample, *a, **k):
        r = real_step(model_output, timestep, sample, *a, **k)
        traj.append((r[0] if isinstance(r, tuple) else r.prev_sample).detach().clone())
        return r

    pipe.scheduler.step = step_hook

    real_dec = pipe.vae.decode
    vrec = {}

    def dec_hook(z, return_dict=True):
        vrec["in"] = z.detach().clone()
        r = real_dec(z, return_dict=return_dict)
        vrec["out"] = (r[0] if isinstance(r, tuple) else r.sample).detach().clone()
        return r

    pipe.vae.decode = dec_hook

    # -------------------------------------------------------------- generate
    lat = 2 * (args.size // 32)                      # latent side = size/16 (even)
    N = lat * lat
    C = int(pipe.transformer.config.in_channels)     # 64 for the real model
    D = int(pipe.transformer.config.context_in_dim)  # 4096 for the real model
    g = torch.Generator("cpu").manual_seed(args.seed)
    noise = randn_tensor((1, 1, C, lat, lat), generator=g, device=torch.device("cpu"),
                         dtype=torch.float32)
    noise0 = pipe._pack_latents(noise, 1, C, lat, lat)            # [1,N,C]
    savef(noise0, "noise0", out)
    print(f"[oracle] generating size={args.size} lat={lat} N={N} steps={args.steps} "
          f"kv_cache={not args.no_kv_cache} ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        res = pipe(prompt=args.prompt, height=args.size, width=args.size,
                   num_inference_steps=args.steps, latents=noise0, generator=g,
                   true_cfg_scale=1.0, use_kv_cache=not args.no_kv_cache)
    gen_s = time.time() - t0
    img = res.images[0]
    img.save(out / "image_ref.png")
    if img.mode == "RGBA":
        from PIL import Image
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        white.save(out / "image_ref_rgb.png")
    print(f"[oracle] image {img.mode} {img.size}, {len(rec)} transformer calls in {gen_s:.0f}s "
          f"(mean {np.mean([r['sec'] for r in rec]):.2f}s/forward)", flush=True)

    # ------------------------------------------------------------------ save
    assert len(rec) == args.steps and len(traj) == args.steps
    m0 = rec[0]["img_mask"][0]
    assert m0.shape[0] == L + N // 4 and not bool(m0[:L].any()) and bool(m0[L:].all())
    assert rec[0]["ehs"] == (1, L, D) and rec[0]["ehs_mask"] is None
    for s, r in enumerate(rec):
        savef(r["latent"], f"latent_{s}", out)
        savef(r["vel"], f"vel_{s}", out)
    savef(traj[-1], "final_latents", out)
    savef(vrec["in"], "vae_in", out)
    savef(vrec["out"], "vae_out", out)
    sig = pipe.scheduler.sigmas.detach().cpu().numpy()
    ts = pipe.scheduler.timesteps.detach().cpu().numpy()
    savef(sig, "sigmas", out)
    savef(ts, "timesteps", out)
    sc = pipe.scheduler.config
    mu = calculate_shift(N, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                         sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
    meta = dict(model=MODEL, snapshot=snap.name, size=args.size, lat=lat, N=N, steps=args.steps,
                seed=args.seed, prompt=args.prompt, kv_cache=not args.no_kv_cache,
                Lfull=Lfull, drop_idx=drop, L=L, img_shapes=[list(x) for x in rec[0]["img_shapes"][0]],
                mu=float(mu), t=[round(r["t"], 7) for r in rec],
                modes=[r["mode"] for r in rec],
                sigmas=[float(x) for x in sig], timesteps=[float(x) for x in ts],
                sec_per_forward=float(np.mean([r["sec"] for r in rec])), gen_s=gen_s,
                image_mode=img.mode, vae_in_shape=list(vrec["in"].shape),
                vae_out_shape=list(vrec["out"].shape),
                versions=dict(diffusers=diffusers.__version__, transformers=transformers.__version__,
                              torch=torch.__version__))
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"[oracle] meta: {json.dumps({k: v for k, v in meta.items() if k not in ('t', 'sigmas', 'timesteps', 'modes')})}",
          flush=True)
    print(f"[oracle] saved to {out}/", flush=True)


if __name__ == "__main__":
    main()
