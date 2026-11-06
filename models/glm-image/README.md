# GLM-Image (AR + diffusion hybrid) — Core AI

[`zai-org/GLM-Image`](https://huggingface.co/zai-org/GLM-Image) (16B, MIT) — the zoo's
**first autoregressive + diffusion hybrid** image model. The image is generated twice:
first as *language* — a 9B GLM-4 AR model samples a grid of discrete visual prior tokens
left-to-right like an LLM (~36 tok/s) — then as *pixels* — a 7B flow-matching DiT denoises
conditioned on those tokens, and a 16ch VAE decodes. Composition comes from the AR stage
(re-rolling the seed changes the scene), texture from the DiT.

Bundle: [🤗 mlboydaisuke/GLM-Image-CoreAI](https://huggingface.co/mlboydaisuke/GLM-Image-CoreAI)
— macOS, int8 AR (9.6 GB) + int8 DiT (9.2 GB, 1024² and 512² variants) + fp32 VAE.
Weights are resolution-independent, so 512 buys speed (~1 min per image at 512² on an M4 Max, 2026-07-07; no end-to-end timing of 1024² is on record), not memory.
Mac-only today (iPhone needs int4 + sequential loading — memory, not resolution, is the
wall).

## Use it

▶️ **Run it** — [CoreAIImageGen (macOS)](../../apps/CoreAIImageGen):

```
open apps/CoreAIImageGen/CoreAIImageGen.xcodeproj
# → run the CoreAIImageGenMac scheme
# → MODEL: "GLM-Image 1024 (AR+diffusion)" → Download & Load → prompt → Generate
```

Steps 20 (default) is reference quality; 12 is ~40% faster and nearly indistinguishable.
Guidance 1.5. The Seed dice meaningfully re-rolls composition (the AR is sampled at
temperature 0.9 / top-p 0.75).

The hybrid can't ride Apple's high-level `CoreAIDiffusionPipeline`, so the app ships a
bespoke ~300-line host loop (`GlmImagePipeline` in
[`DiffusionEngine.swift`](../../apps/CoreAIImageGen/Sources/DiffusionEngine.swift)):
S=1 AR decode with host 3D-mRoPE + top-p sampling → 2× prior upsample → 20-step CFG
flow-match euler (raw-integer timestep conditioning + mu-shifted sigma stepping) →
fp32 CPU VAE.

## Prompting

- Glyph-free only: prompts with quoted text (`'…'`, `"…"`, `「…」`) route through a T5
  glyph encoder that isn't exported — text-in-image will come out garbled.
- Negative prompts are a no-op (GLM's CFG drops the prior instead).
- For photorealism, lead with camera vocabulary (`RAW photo`, `85mm f/1.8`,
  `natural skin texture`) and keep to a single subject; composite scenes skew painterly.

## Port notes

Conversion drivers: [`conversion/export_glm_image_ar.py`](../../conversion/export_glm_image_ar.py),
[`export_glm_image_dit.py`](../../conversion/export_glm_image_dit.py) (`--size 512|1024`),
[`export_glm_image_vae_fp32.py`](../../conversion/export_glm_image_vae_fp32.py).
Full findings — including the DiT timestep-conditioning bug (raw vs mu-shifted schedule)
that cost a day and the debugging playbook that found it — in
[`knowledge/glm-image-port.md`](../../knowledge/glm-image-port.md).
