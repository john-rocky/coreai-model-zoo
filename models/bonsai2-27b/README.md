# Bonsai 2 27B (ternary Qwen3.8) — Core AI

[🤗 `rahulrachuri/ternary-bonsai-2-27b-coreai`](https://huggingface.co/rahulrachuri/ternary-bonsai-2-27b-coreai)
· Apache-2.0 · base
[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)

A Core AI port of PrismML's ternary Bonsai 2 27B checkpoint, with a native Swift host at
[`RahulRachuri/bonsai-swift`](https://github.com/RahulRachuri/bonsai-swift). The published
bundle runs greedy generation at 23.6 tok/s on an M4 Pro and uses approximately 7.5–8 GB of
resident memory.

The model is Qwen3.8-27B's 64-layer hybrid decoder: gated full attention interleaved with
GatedDeltaNet linear-attention layers, dense SwiGLU feed-forward blocks, and a 248,320-token
untied language-model head. PrismML quantization-aware-trained the matrix weights to ternary
values and stores them in PQ2_0 group-128 packing with signed blockwise Walsh-Hadamard activation
transforms.

## Bundle

The published repository contains one three-function bundle and its tokenizer:

```text
bonsai2_27b_decode_pq2_0_pf64/
  metadata.json
  bonsai2_27b_decode_pq2_0_pf64.aimodel/
  aot_mac/
    bonsai2_27b_decode_pq2_0_pf64.h16s.aimodelc/
  tokenizer/
```

| function | input | output |
|---|---|---|
| `main` | `input_ids [1,1]`, dynamic `position_ids` | `logits [1,1,V]`, `next_token [1,1]` |
| `prefill` | `input_ids [1,64]`, dynamic `position_ids` | `logits [1,64,V]` |
| `prefill16` | `input_ids [1,16]`, dynamic `position_ids` | `logits [1,16,V]` |

All three functions share four mutable states: key and value caches for the full-attention
layers, plus convolutional and recurrent states for the GatedDeltaNet layers. The export traces
a 4,096-token maximum context and a 2,048-token minimum KV allocation.

The portable graph is 6.7 GB. The repository also includes an `h16s` AOT compile for the M4 Pro,
bringing the published directory to approximately 14 GB. The artifact used for the measurements
below is pinned at revision
[`0391d83323b76e58fe3167caef5a3ce994e5d4aa`](https://huggingface.co/rahulrachuri/ternary-bonsai-2-27b-coreai/tree/0391d83323b76e58fe3167caef5a3ce994e5d4aa).

## Conversion

The exporter reads PrismML's PQ2_0 GGUF directly. It does not dequantize and re-quantize the
matrix weights. The Core AI graph keeps packed ternary words and fp16 group scales, then applies
four custom Metal kernel families:

- packed group-128 ternary matvec and tiled GEMM;
- signed Walsh-Hadamard transforms and packed embedding gather;
- fused GatedDeltaNet single-token update; and
- fused residual/norm/transform, SwiGLU, and paired-projection sites.

The decode matvec converts four 2-bit codes per GPU instruction through the unorm unpack path.
Measured packed-weight throughput is approximately 220–240 GB/s on the M4 Pro. Prompt processing
uses the largest available chunk first (64, then 16) and walks any remainder through `main`.

The host uses the low-level Core AI runtime because the validation gate needs logits at every
prompt position. The graph-provided `next_token` output is also checked against a host argmax over
the same fp16 logits.

## Validation

The cross-runtime gate teacher-forces the Core AI bundle against greedy reference decodes from
PrismML's MLX runtime. Every comparison uses the same token context even after an argmax mismatch.

| reference prompt | prefill | argmax positions | continuation |
|---|---|---:|---|
| "The capital of France is", 24 new | 16+16+16, 9 walked | 77/80 | 24/24 identical |
| train-time arithmetic, 48 new | 64+16, 3 walked | 127/130 | 48/48 identical |
| lighthouse passage, 40 new | 64+16+16+16, 13 walked | 161/164 | 40/40 identical |
| lighthouse passage, S=1 prompt walk | walk | 161/164 | 40/40 identical |

The three position mismatches occur inside the chat template's system prompt, where the MLX
reference has fp16-scale top-two margins of 0.002–0.024. Graph-provided and host-computed argmax
agree on all 134 walked fixture steps.

A separate ten-prompt chunk-vs-walk battery covers 4,405 positions across prose, code, numbers,
newlines, CJK, mixed scripts, emoji, and repeated-token stress cases. Chunked prefill and an S=1
walk agree at 4,347 positions and produce **80/80 identical continuation tokens**. Each case runs
in a fresh process; resident memory remains near 7.5 GB without additional swap growth.

## Performance

M4 Pro, macOS 27.0 build `26A428`, Xcode 27.0 build `27A5237l`, Swift 6.4, release build,
Python-free generation loop:

| workload | result |
|---|---:|
| decode | **23.6 tok/s** |
| 114-token prompt, end to end | **48 tok/s** |
| S=64 prefill chunk | **59 tok/s** |
| S=1 prompt walk | **23.6 tok/s** |

One decoded token takes approximately 42 ms: about 30 ms in ternary matvecs, about 2 ms in fused
kernels, approximately 3.5 ms at the tail of Core AI's CPU-side encode work, and the remainder in
runtime and command-buffer overhead.

## Reproduce

The recipe downloads the source GGUF from Hugging Face unless `--gguf` supplies a local copy:

```bash
python3 conversion/zoo_convert.py show bonsai2-27b
python3 conversion/zoo_convert.py run bonsai2-27b
```

Equivalent direct export:

```bash
.venv/bin/python conversion/export_bonsai2_27b_decode_pipelined.py \
  --chunk 64 --chunks 16
```

The M4 Pro asset was compiled from the portable graph with:

```bash
xcrun coreai-build compile \
  exports/bonsai2_27b_decode_pq2_0_pf64/bonsai2_27b_decode_pq2_0_pf64.aimodel \
  --platform macOS --architecture h16s --preferred-compute gpu \
  --expect-frequent-reshapes
```

The Swift host provides the release build, generation loop, MLX fixtures, and bounded parity
battery: [`bonsai-swift v0.1.0`](https://github.com/RahulRachuri/bonsai-swift/releases/tag/v0.1.0).

## Limitations

- The export is text-only; PrismML's optional vision projection is not included.
- The exported context limit is 4,096 tokens.
- The native host currently implements greedy generation.
- The included AOT asset targets the M4 Pro. The portable graph is included for producing an
  architecture-matched AOT asset on another Apple silicon Mac.

## Provenance and license

- Language weights: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, file
  `Ternary-Bonsai-2-27B-PQ2_0.gguf`
- Tokenizer and chat-template lineage: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Export tools: `coreai-core 1.0.0b2`
- M4 Pro AOT compiler: `coreai-build-3600.82.1`

The converted artifact inherits the Apache License 2.0 terms of the PrismML and Qwen sources.
Created using Bonsai by Prism ML. The Hugging Face repository includes the license, attribution
notice, and a SHA-256 manifest for every bundle file.
