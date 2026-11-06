# GLM-Image port — AR+diffusion hybrid on Core AI

zai-org/GLM-Image (16B, MIT) → the zoo's first autoregressive + diffusion hybrid image
model. Text → 9B GLM-4 AR writes discrete *visual prior tokens* → 7B flow-matching DiT
denoises conditioned on them → 16ch VAE. Shipped as
[mlboydaisuke/GLM-Image-CoreAI](https://huggingface.co/mlboydaisuke/GLM-Image-CoreAI)
(1024 native + 512 fast, Mac), driven by `apps/CoreAIImageGen`'s bespoke `GlmImagePipeline`
(the high-level `CoreAIDiffusionPipeline` auto-detect path can't host a hybrid).

## Architecture → export mapping

| Piece | What it is | How it exported |
| --- | --- | --- |
| AR (`vision_language_encoder`) | dense GLM-4-9B, 40L/4096/32h/GQA-2, sandwich 4-norm, q/k/v bias, fused gate_up, partial rotary 0.5 (rotate_half), 3D mRoPE [8,12,12], lm_head → vision vocab 16 512 | standard zoo S=1 decode graph; mRoPE cos/sin host-precomputed as graph inputs (FLUX precomputed-RoPE idiom); int8lin per-block-32 |
| DiT (`transformer`) | 30-block MMDiT, patch 2, prior-token conditioning, flow matching | wrapper: precomputed rope + float `prior_scale` (1/0) replacing the export-hostile boolean `prior_token_drop`; **text seq static T=1** (dynamic `split([text,image])` kills the converter — fine because glyph-free prompts always embed `""`); int8lin |
| VAE | 16ch AutoencoderKL decoder | unscale (z·std+mean) baked in + `_patch_nearest_upsample`; **fp32 CPU** — fp16 overflows the activations → NaN black frames |
| Text encoder (T5/ByT5) | glyph (text-in-image) encoder | **not exported** — glyph-free prompts embed the empty string, a prompt-independent constant `[1,1,1472]` shipped as `ehs.f32` |

Drivers: `conversion/export_glm_image_ar.py`, `export_glm_image_dit.py` (`--size 512|1024`),
`export_glm_image_vae_fp32.py`. Both DiT sizes share weights — the exports differ only in
static shapes, so 512 buys speed, not memory.

## The one bug that ate a day: DiT timestep conditioning

Symptom: composition perfect, colors drifted (mild softness at 512, strong
orange/pink over-saturation at 1024). Every component checked clean in isolation — DiT
single-forward cosine 0.9999 vs bf16, VAE byte-exact, AR prior injected from diffusers
reproduced the reference composition — yet the loop output drifted, and per-step latent
traces diverged geometrically (~×1.4/step).

Root cause: diffusers' `FlowMatchEulerDiscreteScheduler.set_timesteps` keeps
`self.timesteps` = the **raw (unshifted) integer schedule** when you pass BOTH
`timesteps=` and `sigmas=` (which `GlmImagePipeline` does), but sets
`timesteps = shifted_sigmas × 1000` when you pass sigmas only (which our port did). So the
reference conditions the DiT on `[999, 949, 899, …]` while stepping the latent with
mu-shifted sigmas; we fed the DiT the shifted values `[999, 983.06, 965.9, …]`. The adaLN
time embedding is systematically wrong from step 1, and the error is proportional to the
shift μ (1.75 @512, 3.25 @1024) — exactly the observed resolution dependence.

Fix (loop-side, no re-export): condition the DiT on
`trunc(linspace(1000, 1, steps+1)[i]) − 1` (integers → also fp16-exact), and keep the euler
update on the shifted sigmas `σ' = μ/(μ + (1/σ − 1))`, μ = `0.75·side/256 + 0.25`.
Teacher-forced per-step error dropped 0.036 → 0.0016; output is now visually at parity
with the reference at the same step count. The debugging playbook that found it (worth
reusing): reference prior+noise injection → per-component isolation → transformer-hook
step-0 dump/replay → per-step latent trace diff → teacher-forced transitions → inferred
sigma forensics. Scripts live in the working tree as `_glmimg_*` (coreai root).

## Other gotchas

- **AOT `.aimodelc` must load with `SpecializationOptions.default`** — forcing
  `preferredComputeUnitKind: .gpu` re-specializes (JIT) and wedges the 9 GB graphs on OS 27
  (`failedToSpecialize`). GUI apps hit this reliably; CLIs sometimes got away with it.
  `.cpuOnly`, once listed here as an alternative, fails with `failedToSpecialize` on a Mac with macOS 27.0
  (26A428) on small conv test graphs (2026-11-06; the GLM-Image bundles were not re-run).
- **Swift `AIModel(contentsOf:)` did not follow symlinked bundles** to their
  arch-specific delegate dir (`main-h16c-delegates`) → same `failedToSpecialize`. Resolve
  with `url.resolvingSymlinksInPath()` (the Python runtime follows symlinks fine).
  That was July 2026. On macOS 27.0 (26A428) a Swift load through a symlinked `.aimodelc` runs on the Mac (2026-11-06; iPhone not
  re-checked).
- **AR must be sampled** (temp 0.9 / top-p 0.75 per upstream `generation_config`); greedy
  collapses visual tokens to flat regions. HF's top-p keeps the boundary token
  (inclusive cumsum) — an exclusive nucleus is subtly greedier.
- **Prompt → AR inputs is a fixed recipe** (glyph-free): `tokenize(prompt)` + constant
  suffix `<sop>H/32 W/32<eop><sop>16 16<eop><|dit_token_16384|>`; prefill positions =
  plain ramp on all 3 mRoPE axes; decode positions = per-grid (t=base, h=base+row,
  w=base+col), grids processed in reverse. Verified byte-exact vs `GlmImageProcessor`
  across en/zh/ja prompts.
- Timings (M4 Max): AR ~36 tok/s (513 tok @512 / 1281 @1024), DiT 20-step CFG ~40 s @512 /
  ~172 s @1024, VAE seconds. 12 steps is visually near-parity and ~40% faster.

## Follow-ups

Glyph (text-in-image) = export the ByT5 encoder + a dynamic-T DiT variant. Editing/i2i =
`GlmImageKVCache` write/read path. iPhone = int4 + sequential component loading (memory,
not resolution, is the wall: AR+DiT ≈ 19 GB int8).
