# LTX-Video 2B (distilled) — Core AI

The zoo's **first video model**. Lightricks'
[LTX-Video](https://github.com/Lightricks/LTX-Video), config `ltxv-2b-0.9.6-distilled`
(MIT / OpenRAIL-M) — **text → video** via an 8-step distilled flow-matching DiT.
All three neural nets run as Core AI `.aimodel` bundles; only the FlowMatch sampler
loop stays on host (the standard zoo pattern — convert the heavy nets, keep the
data-dependent control flow in Python).

Architecture: **T5-XXL** text encoder (4.76 B, the size bottleneck — bigger than the
DiT) → **Transformer3DModel** denoiser (28 blocks, inner 2048, 1.9 B; real-cos/sin
RoPE, rms-norm q/k, AdaLayerNorm-single timestep) run for **8 distilled steps** by a
host FlowMatch-Euler sampler → **causal spatiotemporal video VAE decoder** (1:192
compression, 32× spatial / 8× temporal; also performs the final denoise via timestep
conditioning at `decode_timestep=0.05`). At `guidance_scale=1` (distilled) CFG is off
and `stg_scale=0`, so the DiT runs batch-1, single-conditioning.

<!-- gen-cards:use-it begin id=ltx-video-2b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

**New to Core AI? [Start with CoreAIKit 0.7.3](https://github.com/john-rocky/coreai-kit#readme).** Follow its requirements and first-run steps for `qwen3-0.6b`, then open the same release's [ChatDemo](https://github.com/john-rocky/coreai-kit/tree/0.7.3/Examples/ChatDemo). The README records the tested OS/SDK and download size; model and device coverage is stated per example.

▶️ **Run it (source)** — [`apps/CoreAIVideo`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/CoreAIVideo),
the zoo app that ships this model (text → video on Mac: T5 + 8-step flow-matching DiT + causal video VAE, host FlowMatch sampler; build & run steps in its README).

<!-- gen-cards:use-it end -->

## Sample

512×768 · 49 frames · 8 steps · **~14 s on a Mac GPU**. Prompt: *"A clear glass of water on a
wooden table, slow motion droplet falling into it creating ripples, cinematic."*
([`ltxvideo_sample.mp4`](ltxvideo_sample.mp4) · also on the
[🤗 repo](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI))

<video controls autoplay loop muted src="https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI/resolve/main/sample.mp4"></video>

## Graph contracts (demo 512×768, 49 frames)

```
T5   "input_ids"(1,256) i32 + "attention_mask"(1,256) f32 → "text_embeds"(1,256,4096)
DiT  "hidden_states"(1,N,128) + "indices_grid"(1,3,N) + "encoder_hidden_states"(1,256,4096)
     + "encoder_attention_mask"(1,256) + "timestep"(1,1) → "sample"(1,N,128)
     N = lf·lh·lw ; lf=(F-1)//8+1, lh=H/32, lw=W/32  (here 7·16·24 = 2688)
VAE  "latent"(1,128,lf,lh,lw) + "timestep"(1,) → "pixels"(1,3,F,H,W)   [un-normalize on host]
```

DiT + VAE are fixed-shape → re-convert per target resolution; **T5 is
resolution-independent** (seq 256) so its bundle is reused.

## Numerics gate (vs torch fp32 eager)

Per-net **converted-vs-eager cosine = 1.000000** (T5, DiT, VAE decoder). The DiT also
reproduces torch **on every one of the 8 real sampler steps** (capture real rollout
inputs → bundle vs torch: cos 1.000000, max|Δ| ~1e-3).

End-to-end pixel cosine vs a reference is **~0.93 — that is stochastic-sampler
variance, not error**: two legit torch runs (MPS vs CPU, same seed) are *also* cos
0.9325. Output is verified by per-step cos + **visual** (coherent photoreal video).

| net | fp32 | shipped | per-net cos |
|---|---|---|---|
| T5-XXL encoder | 18 G | **bf16 8.9 G** | 1.000000 |
| DiT (1 step) | 7.2 G | **fp16 3.6 G** | 1.000000 |
| Video VAE decoder | 2.1 G | **fp16 1.0 G** | 1.000000 |

**T5 must be bf16 or fp32** (the encoder overflows in fp16 → washed-out video; bf16
has fp32's exponent range). DiT + VAE are fp16-clean. Ship set **13.5 G** (vs 27 G fp32).

## Speed (Mac, M-series GPU, fp16 DiT/VAE + bf16 T5, 8 steps)

| resolution × frames | time |
|---|---|
| 256×256 × 25f | ~2 s |
| 512×768 × 49f | **~14 s** |

## Running it

`conversion/ltxvideo/_run_coreai.py [H W F seed] --t5 --fp16 --t5bf16` — monkeypatches
`transformer.forward` + `vae.decoder.forward` + `text_encoder.forward` onto the Core AI
bundles and reuses the real pipeline's FlowMatch sampler, patchify, `indices_grid`,
decode-noise mixing and RNG, then writes `coreai.mp4`. Load bundles with an explicit
`SpecializationOptions.default()` (GPU) — `AIModel.load(path, None)` trips an MPSGraph
error; keep the `AIModel` refs alive in the runner.

## On-device note

iPhone is a stretch (same class of limit as [`triposplat`](../triposplat/README.md)): T5-XXL is
4.76 B params and the DiT's video-latent full-attention working set grows with token
count. **Mac is the shipped path.**

Source: `Lightricks/LTX-Video` (`ltxv-2b-0.9.6-distilled-04-25.safetensors`) + T5 from
`PixArt-alpha/PixArt-XL-2-1024-MS`. Convert: `conversion/ltxvideo/`.
