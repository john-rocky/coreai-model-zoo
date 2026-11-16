# Qwen3-4B (Apple's official recipe) — Neural Engine bundle

[🤗 mlboydaisuke/qwen3-4b-CoreAI-official](https://huggingface.co/mlboydaisuke/qwen3-4b-CoreAI-official) · Apache-2.0 · source [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

This is **not a zoo port**: every bundle in the repo is Apple's own `coreai.llm.export qwen3-4b` recipe,
unmodified (the `official/` line — [`../../official/README.md`](../../official/README.md),
[`../../knowledge/apple-models-bench.md`](../../knowledge/apple-models-bench.md)). This directory exists
because the repo gained a **Neural Engine bundle** on 2026-09-15 that the zoo gates on the phone; the
[qwen3-0.6b-official](../qwen3-0.6b-official/README.md) card is the same story one size down.

## Bundles

| subtree | what | size | provenance |
|---|---|---:|---|
| `macos/` | macOS dynamic, int4 per-block (Apple's macOS `4bit` preset) | 2.26 GB | 2026-06-11 export, re-uploaded 2026-09-08 |
| `ios/` | **iOS static IR**, Apple's iOS preset for `qwen3-4b` = `models/qwen3/qwen3_4b_mixed_4bit_8bit.yaml` (k-means **4-bit, per-grouped-channel group 32** on the body; layers 6/8/11/33/34 at **8-bit per-tensor**; embeddings int8 by the iOS exporter), ctx 4096, static graphs `prompt_opt`/`extend` × {256, 512, 1024, 2048, 4096} × {8, 16, 64} + `load_embeddings` | 2.49 GB | 2026-06-11 export (coreai-torch 0.4.0), re-serialized 2026-07-21 by `strip_debug_info` (weights/graph unchanged) |
| `ios-ane-h18p/` | **the same `ios/` IR, AOT-compiled for the Neural Engine** (`xcrun coreai-build compile --platform iOS --preferred-compute neural-engine --architecture h18p`, coreai-build 3600.83.1 / Xcode 27.0 RC, 413 s) — **31/31 ANE regions**; `metadata.json` points at the `.h18p.aimodelc`, tokenizer included | 2.66 GB (`main-h18p.mlirb` 391 MB, sha256 `c5dd0b05d8278cd5c42a4f02e60642e2df94d5fbb95c47ccc1325d571aa0a279`; delegates 2.1 GB) | 2026-09-15 (`conversion/export_ane_stock.py --aot-only`) |

`ios/` is the portable IR for the ANE bundle (what other repos call `ios-static/`). The 0.4.0-era IR
carried no `author`/`license`/`description`; the AOT step wrote them (Qwen Team / Apache-2.0) before
compiling. At 2.66 GB the AOT bundle is above the zoo's conservative ~1.5 GB iPhone ship line; whether it
loads and runs on an iPhone 17 Pro under iOS 27.0 GA is exactly what the device gate below answers (the
iOS 27 **beta** bench loaded it: 13.2 tok/s decode, 3.3 GB footprint, 194 s cold specialization).

## Device gate (iPhone 17 Pro, iOS 27.0 24A435, `ondevice/_ane_gate`)

Pending. Fixture = `Qwen/Qwen3-4B` fp32: alphabet list
(24 steps, min margin 0.930), no-think chat "What is the capital of France?" (10 steps incl. the stop,
min 0.664, EOS 0.999), 396-id counting prompt (16 steps, min 0.958); red twin poisons natural[5]. Rules
and transcript format: [`../../knowledge/minicpm5-1b.md`](../../knowledge/minicpm5-1b.md) §2026-09-15.

## Measured

Pending (same-day interleaved A/B vs the repo's `macos/` dynamic int4 bundle AOT-compiled for the h18p
GPU, p128 / g256 n5). The beta-era HF-card numbers (decode 13.2 / 12.2 tok/s, prefill 546, 512p / 1024g,
2026-06) are not this bundle's numbers.

## Reproduce

```bash
python3 conversion/zoo_convert.py run qwen3-4b-official-ane     # AOT of the published ios/ IR (download it first)
python3 conversion/export_ane_stock.py qwen3-4b                  # a fresh export of the same Apple recipe (a different IR)
```
