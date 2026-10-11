# Z-Image-Turbo — Core AI

[`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B,
Apache-2.0) — a Single-Stream Diffusion Transformer (S3-DiT). A Qwen3-4B text encoder
conditions a 34-block DiT that denoises in 8 FlowMatchEuler steps with classifier-free
guidance, and a 16-channel VAE decodes. Photoreal by default.

Bundle: [🤗 mlboydaisuke/Z-Image-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/Z-Image-Turbo-CoreAI)
— macOS, bf16 DiT (11 GB) + bf16 text encoder (7.3 GB) + fp32 VAE (256 / 512 / 1024) + 2.4 MB
of glue. Weights are bf16, graph boundaries are fp32 — Swift cannot fill a bfloat16 `NDArray`,
and the cast costs ~15 % while *raising* fidelity to 42.6 dB.

**One DiT graph covers 256², 512² and 1024² and any prompt length** — both the image-token
and caption axes are dynamic, at a ~5–9 % cost. No per-resolution bundles.

Mac-only. iOS is blocked, and not for the usual memory reason: the DiT goes all-NaN in fp16
(sampler step 2, at every resolution), bf16 is exact — and `coreai-build compile` refuses a
bf16 module. See [port notes](../../knowledge/zimage-port.md).

## Use it

▶️ **Run it** — [CoreAIImageGen (macOS)](../../apps/CoreAIImageGen):

```
open apps/CoreAIImageGen/CoreAIImageGen.xcodeproj
# → run the CoreAIImageGenMac scheme
# → MODEL: "Z-Image-Turbo 512" → Download & Load → prompt → Generate
```

The host loop is ~250 lines (`Sources/ZImagePipeline.swift`) and is gated against the Python
reference by `conversion/zimage/ZImageRunner`, which compiles the *same file* into a CLI: same bundle,
same noise, images agree to 42.9 dB.

Or drive it from Python (Core AI runtime, no app needed):

```
cd conversion/zimage
PY=../../coreai-models/.venv/bin/python

# one-time: record the fp32 reference + teacher-forcing tensors
$PY capture_oracle.py --size 512 --steps 8 --guidance 1.0

# generate
$PY pipeline_engine.py exports/zimage_dit_512_cap32_full_native_bf16_dyncap_dynimg_iofp32/*.aimodel \
    --dtype fp32 \
    --encoder exports/zimage_encoder_seq64_full_bf16_ids_iofp32/*.aimodel \
    --vae exports/zimage_vae_512_fp32/*.aimodel \
    --prompt "a lighthouse on a rocky cliff at sunset, dramatic clouds"
```

## Which image model should I use?

The zoo's Mac text-to-image models are **not** ranked — pick by trade-off
(all timings M4 Max, 1024²):

| | params | sampler | time @1024 | precision |
| --- | --- | --- | --- | --- |
| **FLUX.2 klein** | 4B | 4 steps, guidance-distilled (no CFG) | **~11 s** | fp16 or int8 |
| **Z-Image-Turbo** | 6B | 8 steps + CFG (16 forwards) | ~70 s | **bf16, near-lossless** |
| **GLM-Image** | 16B (9B AR + 7B DiT) | AR prior + 20-step DiT | ~208 s | int8 |
| **RGBA-Image-2.1** | 7B | 40 steps, no CFG | ~190 s | bf16 |
| **RGBA-Image-2.1-Turbo** | 7B | 8 steps, distilled, no CFG | ~38 s | bf16 or int8 |

FLUX.2 klein is the fast one, and it also does in-context editing. Z-Image trades that speed
for a non-distilled sampler you can steer with CFG and a port that is **numerically near the
fp32 reference** (PSNR 42.3–42.6 dB at 512²/1024², per-step velocity correlation ≥ 0.9997). GLM-Image buys
composition from its autoregressive prior. No head-to-head *aesthetic* comparison has been
run — the table above is speed and fidelity-to-reference, not taste.

## Prompting

- The shipped text-encoder graph is fixed at **64 chat-templated tokens** (≈ 35–40 words of
  prompt). Longer prompts need `export_encoder.py --L 128`.
- `guidance=1.0` is the reference setting. `guidance=0` skips CFG — half the work, still a
  clean image at 256², a different composition.
- Negative prompts work (the uncond branch is a real forward).

## Port notes

Full write-up, gates and every dead end: [`knowledge/zimage-port.md`](../../knowledge/zimage-port.md).
Working state: `conversion/zimage/ZIMAGE_STATE.md`.

Three details each cost a wrong image, and all three were named first by the
**device-proven LiteRT/Android port** of this model:

1. the DiT conditions on the encoder's **penultimate** hidden state (`hidden_states[-2]`);
2. the CFG is **negated** — `noise_pred = -(pos + g·(pos − neg))`;
3. captions are padded to a multiple of 32 with a *learned* pad token that is **real
   attention context** — so `n_cap = round_up(L, 32)` must match exactly, and cond/uncond
   generally have different `n_cap`.

Two Core AI lessons worth carrying elsewhere:

- **int8 is a size trade, not a speed win, on a compute-bound graph.** Weight-only int8
  dequantizes back to 16-bit and runs the same matmul: 2.35 s/fwd vs bf16's 0.89 at 512².
  Activation dtype and quant granularity make no difference. It wins on LLM decode (S=1,
  bandwidth-bound) and on footprint — not here.
- **fp32 activations make the AOT compiler constant-fold quantized weights** into fp32
  (a 1.9 GB int8 graph becomes an 8.7 GB `.aimodelc`); fp16 activations keep them int8.
  Same mechanism, three symptoms: the fp32 graph was the *fastest* int8 config, the largest,
  and the reason a shipped iOS bundle like `glm_ocr_decode_int8hu_s1` stays 764 MB.

License: Apache-2.0 (inherited).
