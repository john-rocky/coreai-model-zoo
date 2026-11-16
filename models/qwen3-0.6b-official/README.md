# Qwen3-0.6B (Apple's official recipe) — Neural Engine bundle

[🤗 mlboydaisuke/qwen3-0.6b-CoreAI-official](https://huggingface.co/mlboydaisuke/qwen3-0.6b-CoreAI-official) · Apache-2.0 · source [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

This is **not a zoo port**: every bundle in the repo is Apple's own `coreai.llm.export qwen3-0.6b`
recipe, unmodified (the `official/` line — see [`../../official/README.md`](../../official/README.md) and
[`../../knowledge/apple-models-bench.md`](../../knowledge/apple-models-bench.md) for the bench matrix).
This directory exists because the repo gained a **Neural Engine bundle** on 2026-09-15 that the zoo
gated on the phone, and a gated bundle needs a recipe entry and a transcript like any other.

## Bundles

| subtree | what | size | provenance |
|---|---|---:|---|
| `macos/` | macOS dynamic, int4 per-block (Apple's macOS `4bit` preset) | 335 MB | 2026-06-11 export, re-uploaded 2026-09-08 |
| `macos-26-export/` | same recipe, macOS-26-era artifact (2.2× faster lowering; cannot be re-created) | 360 MB | 2026-06-11 |
| `ios/` | **iOS static IR**, Apple's iOS preset for `qwen3-0.6b` = `models/qwen3/qwen3_0_6b_mixed_4bit_8bit.yaml` (k-means **4-bit, per-grouped-channel group 8** on the body; layers 0/2/4/7/15/16 at **8-bit per-tensor**; embeddings int8 by the iOS exporter), ctx 4096, static graphs `prompt_opt`/`extend` × {256, 512, 1024, 2048, 4096} × {8, 16, 64} + `load_embeddings` | 440 MB | 2026-06-11 export (coreai-torch 0.4.0), re-serialized 2026-07-21 by `strip_debug_info` (weights/graph unchanged; the 0.4.0 debug locations broke OS 27 beta 2+ loads) |
| `ios-ane-h18p/` | **the same `ios/` IR, AOT-compiled for the Neural Engine** (`xcrun coreai-build compile --platform iOS --preferred-compute neural-engine --architecture h18p`, coreai-build 3600.83.1 / Xcode 27.0 RC) — **31/31 ANE regions**; `metadata.json` points at the `.h18p.aimodelc`, tokenizer included; loads through Apple's unmodified `EngineFactory` → `StaticShapeEngine` on iPhone 17-class devices | 557 MB (`main-h18p.mlirb` sha256 `95c85f6c6b8970f9a31dda2b90eac15fee1f3494c316fdfdf7f6b28315811f8d`) | 2026-09-15 (`conversion/export_ane_stock.py --aot-only`) |

`ios/` is the portable IR for the ANE bundle (what other repos call `ios-static/`): compile it yourself for
another chip. The 0.4.0-era IR carried no `author`/`license`/`description`; the AOT step wrote them
(Qwen Team / Apache-2.0) before compiling, so the `.aimodelc` metadata is complete.

## Device gate (iPhone 17 Pro, iOS 27.0 24A435, `ondevice/_ane_gate`)

Pending. The fixture is `Qwen/Qwen3-0.6B` fp32: alphabet
list (24 steps), no-think chat "What is the capital of France?" (10 steps incl. the stop), 396-id
counting prompt (16 steps). The tool goes RED on a poisoned fixture first. Rules and the transcript
format: [`../../knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md) §2026-09-15.

## Measured

Pending (same-day interleaved A/B vs the repo's `macos/` dynamic int4 bundle on the GPU, p128 / g256 n5).
The iOS 27 **beta** numbers on the HF card (decode 69.6 / 54.1 tok/s, prefill 5,325, warm load 0.045 s,
512p / 1024g, 2026-06) predate the GA toolchain and this AOT; they are not the ANE bundle's numbers.

## Reproduce

```bash
# the shipped bundle: AOT of the published IR (download ios/ from the HF repo first)
python3 conversion/zoo_convert.py run qwen3-0.6b-official-ane

# a fresh export of the same recipe with today's toolchain (a different IR, not what is published):
python3 conversion/export_ane_stock.py qwen3-0.6b --devbundle qwen3_0_6b_ane_fresh
```
