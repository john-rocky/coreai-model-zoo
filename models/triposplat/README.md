# TripoSplat — Core AI

[VAST-AI/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) (MIT) — **single image →
3D Gaussian splats** (`.ply`/`.splat`), the zoo's **first 3D model**. Outputs drop straight
into a Gaussian-splat viewer (RealityKit on visionOS, or
[MetalSplatter](https://github.com/scier/MetalSplatter) on iOS/macOS).

Bundle: [🤗 mlboydaisuke/TripoSplat-CoreAI](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI)
— **5 nets converted** (each gated converted-vs-eager **cos = 1.000000**): DINOv3 ViT-H
encoder + Flux2-VAE encoder (fp16), 20-step flow-matching DiT denoiser (fp16), octree
probability decoder + Gaussian decode with `.ply` activations baked (fp32). The
flow-matching sampler and the octree systematic resampling stay host-side (data-dependent
control flow). ~1 min per image on a Mac GPU.

<!-- gen-cards:use-it begin id=triposplat (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.1](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.1/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

▶️ **Run it (source)** — [`apps/TripoSplatMac`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/TripoSplatMac),
the zoo app that ships this model (single image → 3D Gaussian splats on Mac: 5 converted nets + host flow sampler and octree resampling; build & run steps in its README).

<!-- gen-cards:use-it end -->

## Pipeline

```
image ──(bg removal, host)──▶ 1024²
  DINOv3 ViT-H  (1,3,1024,1024) → (1,4101,1280)   feat1
  Flux2-VAE enc (1,3,1024,1024) → (1,4096,128)    feat2
  DiT ×20 steps: latent(1,8192,16) + cam + t + feat1 + feat2 → latent   (host FlowEuler sampler)
  octree decoder: positions + cond → occupancy logits   (host systematic resampling)
  decode: points + cond → (262144,14) splats → .ply / .splat
```

Conversion + runner scripts: [`conversion/triposplat/`](../../conversion/triposplat/).

## Run

The zoo's [TripoSplatMac](../../apps/TripoSplatMac) app: drop an image, get a `.ply`/`.splat`
you can open in a splat viewer. (The 5-net + host-sampler pipeline is an engine showcase —
a kit `threeD` surface is future work.)
