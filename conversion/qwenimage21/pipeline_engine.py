"""End-to-end Qwen-Image-2.1 text-to-image on the three Core AI bundles: prompt -> RGBA PNG.

  prompt -> ``qi21_tokenize`` (t2i template, ``tokenizers`` only) -> encoder bundle ``input_ids [1,Lfull]``
         -> ``hidden [1,Lfull,4096]`` -> ``[drop_idx:]`` = prompt_embeds [1,L,4096]
  noise0 [1,N,64] -> 40 x { DiT bundle(latents, prompt_embeds, t_i, RoPE tables) -> vel;
                            latents += (sigma_{i+1} - sigma_i) * vel }    (``qi21_sched``, ``qi21_host``)
  latents -> VAE bundle (unpack + ``*std+mean`` in the graph) -> image [1,4,H,W] in [-1,1]
          -> uint8 RGBA (``(x*0.5+0.5).clamp(0,1)*255`` rounded, the pipeline's postprocess) -> PNG
             + the alpha composited on white (``_rgb.png``), as ``capture_oracle.py`` writes it.

No CFG (the pipeline's default ``true_cfg_scale`` 1.0). Bundles (``SpecializationOptions.default()``,
AOT ``--expect-frequent-reshapes``, the recipe every graph of this port runs on):
  encoder  qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc   (dynamic L 16..512)
  DiT      qi21_dit_full_bf16_dyn_iofp32.h16c.aimodelc          (dynamic L 8..512, N 64..4096)
  VAE      qi21_vae_<size>_fp32.h16c.aimodelc                   (one per size)

Modes:
  --oracle oracle/256     noise0 and prompt from the oracle; prints the latent corr vs the oracle's
                          ``latent_{i+1}`` after every step (``final_latents`` after the last) and scores
                          the image against ``image_ref.png`` (RGBA 4-channel PSNR) and
                          ``image_ref_rgb.png`` (white-composited RGB PSNR), alpha max|d| in bytes.
                          ``--encoder oracle`` feeds the oracle's own ``prompt_embeds`` instead of the
                          encoder bundle (isolates the encoder).
  --prompt "..." --size 512 --seed N    free prompt; noise0 = ``torch.randn((1,1,64,h,w))`` on a CPU
                          generator seeded N, packed — what the reference pipeline draws for that seed.

``--turbo`` (Qwen-Image-2.1-Turbo @ d65dbc9): bundles from ``exports/qwenimage21_turbo/`` — the Turbo DiT
``qi21t_dit_full_bf16_dyn_iofp32.h16c.aimodelc`` next to copies of the encoder and VAE bundles above (the Turbo
text encoder is byte-identical to the base one; the VAE bundles hold the base fp32 weights, of which the Turbo
``vae/`` is a bf16 copy) — the checkpoint's 8-step grid (``qi21_sched.schedule_fixed``, no ``mu``) and the
Turbo ``processor/tokenizer.json``. ``--tokenizer-json`` overrides the tokenizer in either mode.

Sensitivity probe: ``--embed-perturb REL --perturb-seed K`` adds ``g * REL * ||pe|| / ||g||`` to the
prompt_embeds (``g`` ~ N(0, I) from numpy seed K, so the global relative L2 change is exactly REL) —
the size of the gap between the encoder bundle and the fp32 oracle is ~1.25e-5.

Images go to ``_work/samples/``; the run summary (per-step corr, PSNR, per-stage seconds) to
``_work/pipeline_engine_<tag>.json``.

Run (coreai-models base venv, from conversion/qwenimage21/; timing runs under the GPU lock):
  python pipeline_engine.py --oracle oracle/256
  python3 ~/code/coreai-kit/scripts/with-gpu-lock.py -- ~/code/coreai/coreai-models/.venv/bin/python \\
      pipeline_engine.py --prompt "a red apple on a wooden table" --size 512 --seed 7
  python pipeline_engine.py --turbo --oracle oracle/turbo256
  python pipeline_engine.py --turbo --prompt "a red apple on a wooden table" --size 1024 --seed 7 --tag turbo_apple1024
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
from qi21_host import build_inputs  # noqa: E402
from qi21_sched import SAMPLE_SIGMAS_TURBO, model_t, schedule, schedule_fixed, step  # noqa: E402
from qi21_tokenize import QI21Tokenizer  # noqa: E402

ROOT = exports_dir() / "qwenimage21"
ENCODER = ROOT / "qi21_encoder_dynL_w16a32_ids_iofp32_aot_efr" / "qi21_encoder_dynL_w16a32_ids_iofp32.h16c.aimodelc"
DIT = ROOT / "qi21_dit_full_bf16_dyn_iofp32_aot_efr" / "qi21_dit_full_bf16_dyn_iofp32.h16c.aimodelc"
ROOT_TURBO = exports_dir() / "qwenimage21_turbo"
ENCODER_TURBO = ROOT_TURBO / ENCODER.relative_to(ROOT)
DIT_TURBO = ROOT_TURBO / "qi21t_dit_full_bf16_dyn_iofp32_aot_efr" / "qi21t_dit_full_bf16_dyn_iofp32.h16c.aimodelc"
MODEL_TURBO = "Qwen/Qwen-Image-2.1-Turbo"
REVISION_TURBO = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
ORDER = ("img_tokens", "txt_feats", "timestep", "txt_cos", "txt_sin", "img_cos", "img_sin")
C = 64


def vae_bundle(size: int, root: Path = ROOT) -> Path:
    return root / f"qi21_vae_{size}_fp32_aot_efr" / f"qi21_vae_{size}_fp32.h16c.aimodelc"


def alpha_stats(a: np.ndarray) -> dict:
    """uint8 alpha [H,W]: min / mean, and the four corners (pixel, and mean of one 16x16 px latent tile)."""
    t = 16
    tiles = [a[:t, :t], a[:t, -t:], a[-t:, :t], a[-t:, -t:]]                    # TL, TR, BL, BR
    return dict(min=int(a.min()), mean=float(a.mean()), frac_below_128=float((a < 128).mean()),
                frac_255=float((a == 255).mean()), frac_0=float((a == 0).mean()),
                corner_px=[int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])],
                corner_tile16_mean=[float(x.mean()) for x in tiles])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def to_u8(img: np.ndarray) -> np.ndarray:
    """[1,4,H,W] in [-1,1] -> [H,W,4] uint8 (``VaeImageProcessor.postprocess`` + ``numpy_to_pil``)."""
    x = np.clip(img[0].astype(np.float32) * 0.5 + 0.5, 0.0, 1.0).transpose(1, 2, 0)
    return (x * 255).round().astype(np.uint8)


def save_pngs(rgba: np.ndarray, stem: Path):
    from PIL import Image
    img = Image.fromarray(rgba)                              # 4 channels -> RGBA
    img.save(f"{stem}.png")
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=img.getchannel("A"))               # as capture_oracle.py composites
    white.save(f"{stem}_rgb.png")
    return np.asarray(white)


async def load(path: Path):
    import coreai.runtime as rt
    t0 = time.perf_counter()
    model = await rt.AIModel.load(Path(path), rt.SpecializationOptions.default())
    return model, model.load_function("main"), time.perf_counter() - t0   # keep `model` alive while calling


async def run(args) -> dict:
    import coreai.runtime as rt
    od = Path(args.oracle) if args.oracle else None
    meta = json.load(open(od / "meta.json")) if od else None
    size = int(meta["size"]) if od else args.size
    prompt = meta["prompt"] if od else args.prompt
    steps = int(meta["steps"]) if od else args.steps
    assert size % 32 == 0, size
    H = W = size // 16
    N = H * W
    vae_path = args.vae or vae_bundle(size, args.vae_root)
    res = dict(mode="oracle" if od else "prompt", size=size, N=N, steps=steps, prompt=prompt,
               seed=int(meta["seed"]) if od else args.seed, encoder=str(args.encoder), dit=str(args.dit),
               vae=str(vae_path), turbo=bool(args.turbo), loadavg_start=os.getloadavg()[0], sec={})
    sec = res["sec"]
    t_all = time.perf_counter()

    # ------------------------------------------------------------------ encoder
    tok = QI21Tokenizer(args.tokenizer_json)
    res["tokenizer_json"] = str(tok.path)
    ids = tok.encode_np(prompt)
    drop = tok.drop_idx
    if od:
        assert ids.tolist() == [np.fromfile(od / "enc_input_ids.i32", "<i4").tolist()], "host ids != oracle ids"
        assert drop == int(meta["drop_idx"]), (drop, meta["drop_idx"])
        pe_ref = np.fromfile(od / "prompt_embeds.f32", "<f4").reshape(1, -1, 4096)
    if args.encoder == "oracle":
        assert od, "--encoder oracle needs --oracle"
        pe = pe_ref.copy()
        print(f"[engine] encoder: oracle prompt_embeds {list(pe.shape)}", flush=True)
    else:
        m, enc, sec["encoder_load"] = await load(args.encoder)
        it = torch.from_numpy(ids).contiguous()
        t0 = time.perf_counter()
        r = await enc({"input_ids": rt.NDArray(it)})
        hidden = np.array(r["hidden"].numpy(), dtype=np.float32)
        sec["encoder_call"] = time.perf_counter() - t0
        del enc, m
        pe = np.ascontiguousarray(hidden[:, drop:])
        print(f"[engine] encoder: Lfull {ids.shape[1]} drop_idx {drop} -> prompt_embeds {list(pe.shape)}; "
              f"load {sec['encoder_load']:.1f}s call {sec['encoder_call']:.2f}s", flush=True)
        if od:
            tc = [corr(pe[0, i], pe_ref[0, i]) for i in range(pe.shape[1])]
            res["prompt_embeds_vs_oracle"] = dict(min_token_corr=min(tc), argmin=int(np.argmin(tc)),
                                                  rel_l2=float(np.linalg.norm(pe - pe_ref) / np.linalg.norm(pe_ref)))
            print(f"[engine] prompt_embeds vs oracle: min token corr {min(tc):.9f} (tok {int(np.argmin(tc))}), "
                  f"|d|/|ref| {res['prompt_embeds_vs_oracle']['rel_l2']:.2e}", flush=True)
    if args.embed_perturb:
        g = np.random.default_rng(args.perturb_seed).standard_normal(pe.shape)
        d = g * (args.embed_perturb * np.linalg.norm(pe.astype(np.float64)) / np.linalg.norm(g))
        pe = (pe.astype(np.float64) + d).astype(np.float32)
        res["embed_perturb"] = dict(rel=args.embed_perturb, seed=args.perturb_seed)
        print(f"[engine] prompt_embeds perturbed: rel L2 {args.embed_perturb:g} (numpy seed {args.perturb_seed})",
              flush=True)
    res["L"] = int(pe.shape[1])

    # ---------------------------------------------------------------- sampler
    if args.turbo:
        assert steps == len(SAMPLE_SIGMAS_TURBO), (steps, len(SAMPLE_SIGMAS_TURBO))
        sigmas, timesteps = schedule_fixed(SAMPLE_SIGMAS_TURBO)
        mu = None
        res["sampler"] = "fixed grid (model_index.json sample_sigmas, shift 1.0, no mu)"
    else:
        sigmas, timesteps, mu = schedule(steps, N)
        res["sampler"] = "linspace + exponential shift(mu) + terminal 0.02"
    res["mu"] = mu
    g = torch.Generator("cpu").manual_seed(res["seed"])          # the reference pipeline's draw for this seed
    noise = torch.randn((1, 1, C, H, W), generator=g, dtype=torch.float32)
    x_seed = noise.view(1, C, H * W).transpose(1, 2).contiguous().numpy()
    if od:
        x = np.fromfile(od / "noise0.f32", "<f4").reshape(1, N, C)
        ref_sig = np.fromfile(od / "sigmas.f32", "<f4")
        assert np.array_equal(sigmas, ref_sig), "sampler schedule != oracle sigmas"
        res["noise0_equals_seed_draw"] = bool(np.array_equal(x, x_seed))   # the --prompt path's noise is the oracle's
        print(f"[engine] oracle noise0 == torch.randn draw for seed {res['seed']}: {res['noise0_equals_seed_draw']}",
              flush=True)
    else:
        x = x_seed
    m, dit, sec["dit_load"] = await load(args.dit)
    ins = build_inputs(torch.from_numpy(x), torch.from_numpy(pe), torch.from_numpy(model_t(timesteps, 0)), H, W)
    feed = {k: rt.NDArray(ins[k].contiguous()) for k in ORDER}   # RoPE tables + text: fixed for the whole run
    step_s, rows, nan_total = [], [], 0
    mu_s = f"mu {mu:.6f}" if mu is not None else "fixed grid"
    print(f"[engine] DiT: {steps} steps, N {N} ({H}x{W}), L {pe.shape[1]}, {mu_s}; load {sec['dit_load']:.1f}s",
          flush=True)
    for i in range(steps):
        xt = torch.from_numpy(x).contiguous()                    # held until the call returns
        tt = torch.from_numpy(model_t(timesteps, i)).contiguous()
        feed["img_tokens"], feed["timestep"] = rt.NDArray(xt), rt.NDArray(tt)
        t0 = time.perf_counter()
        r = await dit(feed)
        vel = np.array(r["vel"].numpy(), dtype=np.float32).reshape(1, N, C)
        step_s.append(time.perf_counter() - t0)
        nan = int(np.isnan(vel).sum())
        nan_total += nan
        x = step(x, vel, sigmas, i)
        if od:
            ref = np.fromfile(od / (f"latent_{i + 1}.f32" if i + 1 < steps else "final_latents.f32"), "<f4")
            ref = ref.reshape(1, N, C)
            row = dict(step=i, t=float(model_t(timesteps, i)[0]), corr=corr(x, ref),
                       rel_l2=float(np.linalg.norm(x - ref) / np.linalg.norm(ref)), nan=nan, sec=step_s[-1])
            rows.append(row)
            print(f"  step {i:2d} t={row['t']:.6f}: latent corr vs oracle {'final' if i + 1 == steps else i + 1:>5} "
                  f"{row['corr']:.6f}  |d|/|ref| {row['rel_l2']:.3e}  NaN {nan}  ({step_s[-1]:.2f}s)", flush=True)
        elif nan:
            print(f"  step {i:2d}: NaN {nan} in the DiT output", flush=True)
    del dit, m, feed
    sec[f"dit_{steps}_steps"] = float(sum(step_s))
    sec["dit_first_step"] = step_s[0]
    sec["dit_step_median_warm"] = statistics.median(step_s[1:]) if len(step_s) > 1 else step_s[0]
    res["steps_rows"], res["dit_nan"], res["dit_step_s"] = rows, nan_total, step_s
    print(f"[engine] DiT: {steps} steps {sec[f'dit_{steps}_steps']:.1f}s (first {step_s[0]:.2f}s, warm median "
          f"{sec['dit_step_median_warm']:.3f}s), NaN {nan_total}", flush=True)

    # -------------------------------------------------------------------- VAE
    m, vae, sec["vae_load"] = await load(vae_path)
    xt = torch.from_numpy(x).contiguous()
    t0 = time.perf_counter()
    r = await vae({"latents_packed": rt.NDArray(xt)})
    img = np.array(r["image"].numpy(), dtype=np.float32)
    sec["vae_call"] = time.perf_counter() - t0
    del vae, m
    sec["total_wall"] = time.perf_counter() - t_all
    res["loadavg_end"] = os.getloadavg()[0]
    rgba = to_u8(img)
    out_dir = HERE / "_work" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.tag
    rgb_white = save_pngs(rgba, stem)
    np.ascontiguousarray(x, "<f4").tofile(f"{stem}_final_latents.f32")
    a = rgba[..., 3]
    res["alpha"] = alpha_stats(a)
    res["image"] = f"{stem}.png"
    print(f"[engine] VAE: load {sec['vae_load']:.1f}s call {sec['vae_call']:.2f}s -> {stem}.png (+ _rgb.png); "
          f"alpha min {a.min()} mean {a.mean():.1f} ({100 * res['alpha']['frac_below_128']:.1f}% < 128, "
          f"{100 * res['alpha']['frac_0']:.1f}% = 0); corners px {res['alpha']['corner_px']} 16px-tile mean "
          f"{[round(v, 2) for v in res['alpha']['corner_tile16_mean']]}", flush=True)

    if od:
        from PIL import Image
        ref_rgba = np.asarray(Image.open(od / "image_ref.png"))
        ref_rgb = np.asarray(Image.open(od / "image_ref_rgb.png"))
        vae_out = np.fromfile(od / "vae_out.f32", "<f4").reshape(meta["vae_out_shape"])[:, :, 0]
        da = np.abs(rgba[..., 3].astype(np.int16) - ref_rgba[..., 3].astype(np.int16))
        res["vs_oracle"] = dict(
            psnr_rgba=psnr(rgba, ref_rgba), psnr_rgb_white=psnr(rgb_white, ref_rgb),
            psnr_rgb_raw=psnr(rgba[..., :3], ref_rgba[..., :3]), alpha_maxd_u8=int(da.max()),
            ref_alpha_min=int(ref_rgba[..., 3].min()), image_float_corr=corr(img, vae_out),
            final_latent_corr=rows[-1]["corr"], min_step_corr=min(r_["corr"] for r_ in rows))
        v = res["vs_oracle"]
        ok = (v["psnr_rgb_white"] >= 30.0 and v["alpha_maxd_u8"] <= 2 and v["final_latent_corr"] >= 0.999
              and nan_total == 0)
        res["ok"] = bool(ok)
        print(f"[engine] vs oracle: RGBA PSNR {v['psnr_rgba']:.2f} dB | white-composited RGB PSNR "
              f"{v['psnr_rgb_white']:.2f} dB | raw RGB PSNR {v['psnr_rgb_raw']:.2f} dB | alpha max|d| "
              f"{v['alpha_maxd_u8']}/255 (ref alpha min {v['ref_alpha_min']}, engine {a.min()}) | image float corr "
              f"{v['image_float_corr']:.6f} | final latent corr {v['final_latent_corr']:.6f}", flush=True)
        print(f"[engine] {'PASS' if ok else 'FAIL'} (bar: white-composited RGB PSNR >= 30 dB, alpha max|d| <= 2/255, "
              f"final latent corr >= 0.999, NaN 0)", flush=True)
    print(f"[engine] seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in sec.items())
          + f"  (loadavg {res['loadavg_start']:.1f} -> {res['loadavg_end']:.1f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=None, help="oracle/<size>: noise0 + prompt from the oracle, scored against it")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="default 40; --turbo: the grid's 8 (fixed)")
    ap.add_argument("--turbo", action="store_true", help="Qwen-Image-2.1-Turbo: its DiT, 8-step grid, tokenizer.json")
    ap.add_argument("--encoder", default=None, help="encoder bundle, or 'oracle' (the oracle's prompt_embeds)")
    ap.add_argument("--dit", default=None)
    ap.add_argument("--vae", default=None, help="default: qi21_vae_<size>_fp32 AOT efr bundle")
    ap.add_argument("--tokenizer-json", default=None, help="default: the base snapshot's; --turbo: the Turbo snapshot's")
    ap.add_argument("--embed-perturb", type=float, default=0.0, help="relative L2 perturbation of prompt_embeds")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output stem in _work/samples/ and the json name")
    args = ap.parse_args()
    assert bool(args.oracle) != bool(args.prompt), "pass exactly one of --oracle / --prompt"
    args.encoder = args.encoder or str(ENCODER_TURBO if args.turbo else ENCODER)
    args.dit = args.dit or str(DIT_TURBO if args.turbo else DIT)
    args.vae_root = ROOT_TURBO if args.turbo else ROOT
    if args.turbo:
        assert args.steps in (None, len(SAMPLE_SIGMAS_TURBO)), f"--turbo runs the {len(SAMPLE_SIGMAS_TURBO)}-step grid"
        args.steps = len(SAMPLE_SIGMAS_TURBO)
        args.tokenizer_json = args.tokenizer_json or str(
            Path(hf_snapshot(MODEL_TURBO, revision=REVISION_TURBO)) / "processor" / "tokenizer.json")
    args.steps = args.steps or 40
    if args.tag is None:
        if args.oracle:
            args.tag = f"engine_oracle{Path(args.oracle).name}" + ("_encoracle" if args.encoder == "oracle" else "")
            if args.embed_perturb:
                args.tag += f"_pert{args.embed_perturb:g}_s{args.perturb_seed}"
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:32]
            args.tag = f"engine_{args.size}_seed{args.seed}_{slug}"
    res = asyncio.run(run(args))
    js = HERE / "_work" / f"pipeline_engine_{args.tag}.json"
    json.dump(res, open(js, "w"), indent=2)
    print(f"[engine] -> {js}", flush=True)
    return 0 if res.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
