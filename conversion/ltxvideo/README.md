# LTX-Video 2B → Core AI (zoo's first VIDEO)

[Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video), config
`ltxv-2b-0.9.6-distilled` — **text → video** flow-matching DiT, 8 distilled steps (code Apache-2.0;
the 0.9.6-distilled weights are under the [LTXV Open Weights License 0.X](https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt),
under which commercial use by entities with $10M or more annual revenue needs a paid license).
The zoo's first video model: all three neural nets run as Core AI `.aimodel` bundles; only the
FlowMatch sampler loop stays on host.

Pipeline: T5-XXL text encoder → 8-step flow-matching DiT denoiser (host sampler) → causal
spatiotemporal video VAE decoder (also does the final denoise at `decode_timestep`).

## What runs on Core AI

3 neural nets converted (each gated converted-vs-eager **cos = 1.000000**):

| net | shape (demo 512×768×49f) | fp32 | fp16/bf16 | bundle |
|---|---|---|---|---|
| T5-XXL text encoder | ids(1,256)+mask(1,256) → (1,256,4096) | 18 G | **8.9 G (bf16)** | `t5` |
| DiT denoiser (one step) | latent(1,N,128)+grid(1,3,N)+text(1,256,4096)+mask(1,256)+t(1,1) → (1,N,128) | 7.2 G | **3.6 G (fp16)** | `dit` |
| Causal video VAE decoder | latent(1,128,lf,lh,lw)+t(1,) → pixels(1,3,F,H,W) | 2.1 G | **1.0 G (fp16)** | `vae` |

`N = lf·lh·lw`, `lf=(F-1)//8+1`, `lh=H/32`, `lw=W/32` (VAE spatial 32×, temporal 8×). The DiT is
fixed-shape, so re-convert it + the VAE per target resolution; T5 is resolution-independent (seq 256).
At `guidance_scale=1` (distilled) CFG is off and `stg_scale=0`, so `num_conds=1` (batch 1 through the DiT).

The single checkpoint `ltxv-2b-0.9.6-distilled-04-25.safetensors` packs **both** the DiT
(`model.diffusion_model.*`) and the VAE (`vae.*`); the native `Transformer3DModel.from_pretrained` /
`CausalVideoAutoencoder.from_pretrained` load both from it. T5 is the separate
`PixArt-alpha/PixArt-XL-2-1024-MS` `text_encoder/` subfolder (~19 G fp32).

## The one model patch (vs TripoSplat's six — LTX is converter-friendly)

LTX needs **a single edit** to `ltx_video/models/transformers/attention.py:245`:

```python
# norm_hidden_states.squeeze(1)  is a no-op unless dim 1 == 1, but coreai-torch
# treats squeeze(dim) as a HARD shrink and aborts when the dim != 1.
if norm_hidden_states.shape[1] == 1:
    norm_hidden_states = norm_hidden_states.squeeze(1)   # resolved away at trace time
```

Everything else converts cleanly because LTX avoids the usual traps:
- **RoPE is already real** — `precompute_freqs_cis` returns `(cos, sin)`; `apply_rotary_emb` is
  `x·cos + rotate_half(x)·sin`. No complex ops to rewrite.
- **Norms use explicit `rsqrt(var+eps)`** (diffusers `RMSNorm`, T5 `LayerNorm`) — no `F.normalize`
  eps-clamp blow-up.
- **The RoPE/timestep `sin·cos` are input-driven** (from `indices_grid` / `timestep`), so they're
  computed at runtime, not constant-folded.
- `optimize=False` (the big-attention `optimize()` hang from TripoSplat still applies; AOT
  `coreai-build` optimizes for the device).

## Scripts

- `_conv_dit.py [H W F seq]` / `_conv_vae.py [H W F]` / `_conv_t5.py [seq]` — convert + gate each net (fp32).
- `_conv_fp16.py {dit|vae|t5} [H W F seq] [--bf16]` — half-size bundles (fp16 weights, fp32 IO).
- `_run_coreai.py [H W F seed] [--t5] [--fp16] [--t5bf16] [--cpu]` — end-to-end: monkeypatches
  `transformer.forward` + `vae.decoder.forward` (+ `text_encoder.forward` with `--t5`) onto Core AI
  bundles, reuses the real pipeline's FlowMatch sampler / patchify / indices_grid / decode-noise /
  RNG, writes `coreai.mp4`.
- `_baseline.py` — pure-torch reference. `_gate_step.py` — captures real per-step DiT inputs from a
  torch run and checks the bundle reproduces every step (the meaningful gate; see below).
- `_common.py` — shared helpers + `build_pipeline()` (replicates `create_ltx_video_pipeline` without
  importing `ltx_video.inference`, which hard-imports `imageio`).

Setup: clone LTX-Video, download the distilled checkpoint to `ckpts/` and the PixArt T5 to
`ckpts/pixart/`, apply the one patch, then run with the coreai venv (`coreai_kit` on `sys.path`).

## Gating a stochastic diffusion port (lesson)

End-to-end pixel cosine vs a reference is **~0.93 — and that is sampler variance, not error**:
two legit torch runs (MPS vs CPU, same seed) are *also* cos ≈ 0.9325. The right gate is **per-step
DiT cos** (`_gate_step.py` → 1.000000 on all 8 real steps) plus a **visual** check. Don't judge a
stochastic video port by its end-to-end pixel cosine.

## Runner gotchas (not in the conversion — see ../../knowledge/conversion-guide.md)

- **Pass an explicit `SpecializationOptions`** to `AIModel.load`, never `None` — `load(path, None)`
  trips `RuntimeError: MPSGraph Unresolved symbol (prepare/initialize)` on the GPU path;
  `SpecializationOptions.default()` (GPU) / `.cpu_only()` work.
- **Keep the `AIModel` reference alive** in a persistent runner (not just `load_function`); a GC'd
  model returns garbage (no crash) → washed-out output.
- **T5 overflows in fp16** (well-known T5-encoder issue) → washed-out video. Use **bf16** (same
  exponent range as fp32, half the size) or fp32 for T5; DiT + VAE are fine in fp16.
- Run the host glue on CPU when driving Core AI on GPU — PyTorch's MPS Metal context vs Core AI's
  MPSGraph can contend.

## Numbers (Mac, M-series GPU, fp16 DiT/VAE + bf16 T5)

- 256×256×25f, 8 steps: ~2 s. 512×768×49f, 8 steps: **~14 s.** Output is coherent, photoreal video
  (per-step DiT cos 1.000000 vs torch).
- Ship set total **13.5 G** (DiT 3.6 + VAE 1.0 + T5 8.9) vs 27 G fp32.

## On-device note

iPhone is a stretch (same class of limit as TripoSplat): the DiT's video-latent full-attention
working set grows with token count and T5-XXL is 4.76 B params. Mac is the shipped path.
