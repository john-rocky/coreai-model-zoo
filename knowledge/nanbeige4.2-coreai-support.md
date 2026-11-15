# Nanbeige4.2-3B on Core AI

This report records the final architecture, runtime, verification, and optimization decisions for
[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) at checkpoint revision
`5ff54fb7ed86ce8e216d78bff5417ab9981de3d4`.

The verified int8 LanguageBundle is published at
[`ukint-vs/Nanbeige4.2-3B-CoreAI@5864ec7`](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e).
Port by [@ukint-vs](https://github.com/ukint-vs); tracked in
[`john-rocky/coreai-model-zoo#5`](https://github.com/john-rocky/coreai-model-zoo/issues/5).

## Architecture

Nanbeige4.2 is not a 44-weight-layer Llama. It owns **22 physical Llama blocks** and executes the same blocks
twice, applying the final RMS norm after each pass. Each pass has independent attention history:

```text
embedding
  → physical blocks 0…21 → norm       (cache slots 0…21)
  → physical blocks 0…21 → norm       (cache slots 22…43)
  → untied language-model head
```

The Core AI authoring overlay therefore retains:

- 22 trainable block instances and 111 physical linear modules;
- 44 logical KV-cache layers;
- one serialized and quantized copy of every physical weight;
- execution order `22 blocks → norm → same 22 blocks → norm`.

The implementation reuses the existing Llama embedding, fused QKV attention, MLP, RMSNorm, RoPE, head, weight
loader, and quantization traversal. Nanbeige-specific code adds released-config validation, recurrent execution,
and a cache-offset wrapper. Unsupported loop counts, skipped loop norms, loop-loss metadata, biases, QK norm,
shared loop KV, N-gram features, split loops, hyper-connections, mHC, and depth attention fail explicitly.

The semantics follow [mlx-lm commit `308bc1f`](https://github.com/ml-explore/mlx-lm/commit/308bc1f68fafd41756a5973e446215729f5ca7fe);
mlx-lm is a reference, not a runtime dependency. The checkpoint config SHA-256 is
`f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`.

## Conversion and runtime

`conversion/export_nanbeige41_decode_pipelined.py` remains compatible with Nanbeige4.1. It selects Llama or
Nanbeige authoring from checkpoint `model_type`, forwards `--revision` to model and tokenizer downloads, and
records the revision in bundle metadata.

Nanbeige4.2 uses K/V state shape `[44, 1, 8, max_context, 128]` and the existing static-S=1
`coreai-pipelined` GPU contract:

```text
input_ids, position_ids, mutable k_cache, mutable v_cache → logits
```

The shipping configuration is `int8hu --head-sym --static-ids`. The runtime patch supplies descriptor-driven
single-token prompt chunking and warmup plus static logits capacity. No recurrence-specific runtime or new
dependency is required.

## Accepted artifact and correctness

| Check | Result |
|---|---|
| Bundle | 4.59 GiB (`4,815,288 KiB`), `coreai-core 1.0.0b2` |
| Physical/logical structure | 22 unique blocks, 111 physical linears, 44 cache layers |
| Synthetic float32 | Full and cached logits pass at `rtol=1e-4`, `atol=1e-4`; one-layer truncation still executes twice |
| Official checkpoint | Full max error `1.01566e-4`, cached max error `2.09808e-5`, identical 32-token greedy continuation |
| Int8 authoring | Prompt top-1 8/8, greedy 32/32, cosine 0.9997768, deterministic |
| Int8 Core AI | Token-exact on two factual prompts and the 64-token `9.11` versus `9.8` reasoning smoke |
| Bundle smoke | Loads, produces logits, and mutates all 44 cache layers |

The bundled vendor chat template renders with `enable_thinking=true` and `enable_thinking=false`.

## Performance and memory

Measurements used an M4 Max with 36 GB RAM, macOS 27.0, Xcode 27 beta 4, Core AI runtime `aff0bb2`, AC power,
High Power Mode, and Release builds.

| Workload | Prefill | Decode | Peak memory |
|---|---:|---:|---:|
| prompt 128 / generation 256, 3-run average | **47.37 tok/s** | **46.35 tok/s** | — |
| prompt 3,840 / generation 256 | **29.83 tok/s** | **32.80 tok/s** | **9.17 GiB**, zero swaps |

The loop changes storage and numerical sensitivity, but not dependency order:

| Resource | Complexity |
|---|---:|
| Weight storage and quantization | `O(22 physical layers)` |
| Weight reads and transformer compute | `O(2 × 22)` per token |
| Attention work | `O(44 × context)` per token |
| KV-cache storage | `O(44 × context)` |
| Full-prompt attention | `O(44 × context²)` |

With 8 KV heads, head dimension 128, and fp16 K/V, cache storage is 176 KiB per token: 704 MiB at 4,096 tokens
and 44 GiB at 262,144 tokens. The published model is therefore verified to 4K; the advertised 262K context is
not claimed.

## Optimization decisions

| Candidate | Result | Decision |
|---|---|---|
| int8hu block-32 symmetric | Pass; 4.59 GiB, 46.35 decode tok/s | **Ship** |
| int4hu block-32 symmetric | 3.14 GiB, 56.07 decode tok/s, but decisive multi-token quality failures | No-go |
| mixed physical-layer or projection-role int4/int8 | Smaller bundles, but decisive Core AI reasoning divergence | No-go |
| custom SDPA kernels | Slower than externalized SDPA and recurrent quality failures | Keep stock SDPA |
| TensorOps FlashAttention probe | 22% slower at 257 tokens and 57% slower at 4K | Keep stock SDPA |
| custom symmetric-int8 GEMV and fused gate/up | 0.964–0.999× stock throughput across real model shapes | Keep stock linears |
| pass-0 self-draft | 41/104 token agreement (39.4%); no calibrated intermediate exit | No-go |

The final path is the existing single two-pass graph with fused QKV, compiler-recognized SDPA, stock int8
linears, shared physical weights, and 44 disjoint cache slots. No custom Nanbeige kernel remains in the overlay.

For contexts beyond 4K, quantized KV is the only current kernel target with a meaningful capacity benefit:
int8 KV would halve cache storage and long-context traffic. It requires a quantization-aware attention path and
must still beat stock SDPA through every recurrent quality gate. Pass-0 speculation is only worth revisiting
with a distilled intermediate head or loop-loss training, which would be a new model artifact.

## Device status

Mac execution and Mac-side iOS compile acceptance pass. Xcode 27 (`27A5228h`) with Metal Toolchain `27A5228f`
compiled the int8 bundle for iOS 27 GPU `h18p` in 11.53 seconds. The resulting 4,809,424 KiB `.aimodelc`
records the published `.aimodel` source hash.

The iOS 27 `h18p` hardware gate passed on 2026-07-24 (iPhone 17 Pro, maintainer-run): nat 24/24 and
oracle 24/24 greedy tokens identical to the M4 Max engine reference across two full runs, cold-specialization
load 31.7 s (warm 10.8 s), and p=128/g=256 throughput of 6.9/5.7 tok/s first-run → 8.5 prefill / 6.4 decode
tok/s settled. The two-pass loop reads the 22 shared blocks twice per token, so the bandwidth-bound phone
lands near half a same-size single-pass model — consistent with theory, no conversion issue. CoreAIKit
enrollment (`nanbeige4.2-3b`) and the generated "Use it" block shipped with the gate.

## Reproduction

Use a `coreai-models` checkout at the commit pinned by `conversion/overlay/BASE`, apply the overlay, then run:

```sh
python3 ../coreai-model-zoo/conversion/zoo_convert.py run nanbeige4.2-3b --dry-run
python3 ../coreai-model-zoo/conversion/export_nanbeige41_decode_pipelined.py \
  int8hu --head-sym --static-ids \
  --hf-id Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4

python3 ../coreai-model-zoo/_smoke/verify_nanbeige42_checkpoint.py \
  --official-python /path/to/nanbeige-oracle/bin/python

python3 ../coreai-model-zoo/conversion/coreai_gate.py \
  exports/nanbeige4_2_3b_decode_int8hu_block32_sym_s1 \
  Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4 \
  --arch nanbeige -n 24
```

Model weights are Apache-2.0. Conversion code is BSD-3-Clause.
