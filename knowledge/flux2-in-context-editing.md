# FLUX.2 in-context editing on Core AI

How the zoo runs FLUX.2 [klein] 4B **instruction editing** and **multi-reference composition**
on Apple Silicon via Core AI — without a separate editing model, a ControlNet, or any
non-commercial weights. The whole thing rides on the Apache-2.0 klein base.

## Why this exists

FLUX.2 [klein] is a *unified* model: the same 4B DiT does text-to-image, image-to-image, and
in-context editing. The zoo already shipped T2I. The editing capability was blocked only by the
runtime path — Apple's stock `CoreAIDiffusionPipeline` exposes `startingImage` + `strength`
(SDEdit), which **re-renders the whole frame** and can't do "add a hat, keep everything else."
Third-party FLUX editing assets don't help: FLUX.2 [dev] ControlNets and klein-9B RefControl
LoRAs are all **non-commercial** (FLUX NCL). The clean answer was to unlock klein's own native
editing, which is Apache and already in the weights we ship.

## The mechanism (in-context / "Kontext"-style)

FLUX.2 editing is **not** a separate transformer input. The reference image's VAE-latent tokens
are **concatenated into the image sequence** the transformer denoises, distinguished only by a
time coordinate `T` in the model's 4-axis `(T, H, W, L)` rotary positions:

```
image sequence = [ output latent (T=0) ; reference 0 (T=10) ; reference 1 (T=20) ; … ]
```

Per denoising step:
1. Build `hidden_states = concat(output_latents, clean_reference_latents…)`.
2. Run the transformer over the joint sequence (+ text).
3. **Slice the prediction back to the output tokens** and step only those. The reference tokens
   are re-supplied clean every step and their predictions are discarded — they are context, never
   denoised.

The output starts from **pure noise** (`sigmaMax = 1.0`), not a noised copy of an input — so the
instruction can add / replace / relight / combine content while attention to the clean references
keeps the subjects. `guidance_embeds = false` on klein, so there is no CFG (single forward).

This is the same DiT graph as text-to-image. Only the **sequence length** grows and the reference
blocks carry a nonzero `T`.

## Export (Core AI graphs are fixed-shape)

Because Core AI graphs are static-shape, each *number of references* is its own export. The zoo's
export recipe (`coreai_models/diffusion/flux2.py` + `components.py`) adds:

| component | sequence (1024 / 512) | contents |
|---|---|---|
| `transformer_edit` / `_512` | 8192 / 2048 | output + 1 reference |
| `transformer_edit_2ref` / `_512` | 12288 / 3072 | output + 2 references |

```bash
uv run coreai.diffusion.export flux2-klein-4b --components transformer_edit transformer_edit_2ref
```

The exported wrapper (`Flux2TransformerPrecomputedRoPEWrapper`) is **token-content-agnostic** — it
takes `hidden_states [1, seq, C]` plus **precomputed RoPE** (cos/sin), so all of the `(T,H,W,L)` id
layout is decided by the runtime, not baked into the graph. That is the single reason this worked
with zero graph surgery: the shipped T2I graph's RoPE already has the `T` axis (T2I just uses `T=0`
for every image token). `coreai-build` compiles the long edit sequences (8192, 12288) at int4.

## Runtime

`Flux2Pipeline.editImages(referenceImages:instruction:editTransformer:…)`:
- VAE-encodes each reference to clean latent tokens (encode → scale → patchify → batch-norm → pack)
  — the img2img encode **minus the noise blend**.
- Builds the concatenated latents + the `(T,H,W,L)` RoPE for `[text ; output(T=0) ; ref_i(T=10·(i+1))]`.
- Runs the edit-sequence transformer, slices the output tokens, steps them with the discrete-flow
  scheduler, VAE-decodes.
- Picks the 1- or 2-reference transformer by `referenceImages.count`.

The `diffusion-runner` CLI exposes it as `--edit-image` / `--edit-image2` / `--instruction`.

## Numbers (M-class Mac GPU, int4, 4 steps)

- 1 reference, 1024: **~25 s** (seq 8192)
- 2 references, 1024: **~43 s** (seq 12288)

Parity confirmed against the diffusers `Flux2KleinPipeline` edit path (same edit semantics; int4
introduces the expected minor pixel differences). Mac-first — at 4B the peak footprint overruns a
12 GB iPhone; the 512 edit transformers (seq 2048 / 3072) are the iPhone path (not yet gated).

## Gotchas

- **Runtime base matters.** apple/coreai-models `02a8edd` has a working Qwen3 text-encode path;
  some other revisions crash with `77 vs 512` (a CLIP-77 tokenizer fallback). Base the fork on a
  verified revision.
- **Fixed shape = one graph per reference count.** Variable references would need more exports or
  AOT reshape.
- **Edit weights are separate ~2 GB transformers.** Fetch them on demand, not in the base download.

## On the stock runtime (2026-10-10)

apple/coreai-models `97a14be` runs this mechanism itself. Its FLUX.2 image-to-image
(`FlowTransformerPipeline`, `PipelineConfiguration.startingImage` + `referenceGrid`) VAE-encodes the
reference to clean latent tokens, appends them after the noise tokens with `T = 10` on RoPE axis 0, starts
the output from pure noise (`sigmaMax = 1.0`), runs the transformer over the joint sequence and keeps only
the noise tokens' prediction: the one-reference layout above. The exporter traces it as the
`img2img_full` / `_half` / `_quarter` entrypoints of the multi-function transformer; `full` is 4096 noise +
4096 reference tokens = 8192 at 1024, the length `transformer_edit` had. The RoPE comes from position ids
inside the graph instead of host-computed cos/sin.

- `apps/CoreAIImageGen` runs its Edit tab on it since 2026-10-10: `macos-int8/` of the Hub repo,
  `referenceGrid: .full`, one reference. `macos-fp16/` has no image-to-image transformer.
- Multi-reference compose (`transformer_edit_2ref`) has no stock counterpart. It stays on the fork's
  branch and left the app.
- Measured on the stock path (2026-10-10, M4 Max, `macos-int8/`, 1024, 4 steps, one reference, full grid):
  24.39 and 24.31 s per edit in the app, against 11.60 / 11.59 s for text-to-image in the same window
  (`apps/CoreAIImageGen/README.md`).

## Where

- Weights + docs: [`mlboydaisuke/FLUX.2-klein-4B-CoreAI`](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)
- Runtime: apple/coreai-models `97a14be` (`FlowTransformerPipeline` image-to-image) since 2026-10-10; the
  2026-07 edit-sequence export recipe and the 2-reference path: `john-rocky/coreai-models` branch
  `flux2-in-context-edit`
- App: `apps/CoreAIImageGen` (the **Edit** tab)
