# Granite-Embedding-97M-Multilingual-R2 — Core AI

IBM's 97M-parameter **multilingual text embedder** — a ModernBERT encoder, 384-d CLS-pooled
unit vectors, Japanese and English among its languages — as a static `.aimodel` for macOS 27
and, ahead-of-time compiled, for the iPhone 17 Pro.
[`ibm-granite/granite-embedding-97m-multilingual-r2`](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
(Apache-2.0, revision `835ad1408…`) is the **smallest embedder in this catalog** (390 MB fp32,
against 1.2 GB for EmbeddingGemma-300m and 1.1 GB for Qwen3-Embedding-0.6B) and its **first
encoder-architecture one** — every other embedder here is a causal decoder run as an encoder.
Its retrieval quality relative to those three was **not** measured here: the fixture set below
is a parity instrument (35 texts, 4 queries, 12 documents), not a benchmark.

**This is an encoder, not a generator** — one forward over the right-padded grid returns one
unit vector. No autoregressive loop, no KV cache, no LM head. It runs as a plain `.aimodel`
through raw `AIModel.run` (like the vision encoders), not the pipelined generate engine.

Architecture (`model_type: modernbert`): 12 layers, hidden 384, 12 heads × 32, GLU MLP 1536
(SiLU), vocabulary 180,000, biasless everything (attention, MLP, LayerNorm ε 1e-5). Global
attention at layers **0, 3, 6, 9** (RoPE θ 150,000); the other eight are **local**, a sliding
window of inclusive radius 64 (129 keys per interior query, RoPE θ 160,000). Layer 0 has no
attention pre-norm (the embedding LayerNorm serves). Pooling is CLS → L2 normalize, both in the
graph.

## Graph contract

```
input  "input_ids"       [1, S]    int32   right-padded to the grid S with 179935
input  "attention_mask"  [1, S]    int32   1 over real tokens, 0 over padding
output "embedding"       [1, 384]  fp32    CLS-pooled, L2-normalized
S = 128 or 512 (export-time choice); batch = 1
```

**Host recipe** — the tokenizer is the whole contract, and the stock one is not enough:
- **No prefix, no stripping, no normalization.** Query and document prompts are both empty in
  the checkpoint. Raw whitespace is kept: sentence-transformers strips text before tokenizing,
  the upstream README's `AutoTokenizer` path does not, and the two disagree on `"  東京駅から…\n"`.
  The reference is the raw path.
- Tokenize with the pinned `tokenizer.json`: regex `Split(Isolated)` → `ByteLevel` (no prefix
  space) → byte BPE with **`ignore_merges = true`** (a whole pre-token that is in the vocabulary
  wins; ` ક` is token 2999, not three). A BPE that ignores the flag tokenizes differently.
- Truncate the **body to S−2**, then wrap: `[CLS 179934] body… [SEP 179938]`, right-pad with
  **PAD 179935** and mask 0. Truncating after adding the specials loses SEP; padding with 0 is a
  different token. Both are silent.
- Similarity = dot product (unit vectors). Dimension truncation is not a property of this model.

`conversion/granite_embedding/_granite_tokenizer.py` is that recipe with no HF import, and
`host/GraniteTokenizer.swift` in the HF repo the same recipe in Foundation-only Swift; the gate
holds both to `AutoTokenizer` exactly (ids and masks) over **681 texts × 2 grids = 1,362 cases**
including every added token in five boundary contexts, and proves four mutations are caught
(pad 0 / lose SEP / strip / `ignore_merges=false`).

## Measured

**iPhone 17 Pro** (iPhone18,1), iOS 27.0 build **24A437**, the compiled `h18p` bundles loaded by
the native `AIModel` loader, GPU-preferred (MPSGraph/Metal plan). Every row: 35 HF texts, the
gate below, 105 warm samples, thermal state fair before and after, caches retained (so "first"
is process-first, not cache-cold). Peak footprint is the whole app process, tokenizer and file
hashing included. Measured 2026-09-19.

| Variant | S | Gate | Min cosine vs HF | Max \|err\| | Load | First after load | **Warm median** | Peak footprint |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| fp32 | 128 | 35/35 | 0.999999999999407 | 1.97e-7 | 81 ms | 23.1 ms | **5.54 ms** | 640 MB |
| fp32 | 512 | 35/35 | 0.999999999999486 | 2.38e-7 | 586 ms | 39.9 ms | **20.99 ms** | 640 MB |
| w8 / fp32 table | 128 | 35/35 | 0.999410 | 5.67e-3 | 61 ms | 25.3 ms | 6.64 ms | 555 MB |
| w8 / fp32 table | 512 | 35/35 | 0.999410 | 5.67e-3 | 447 ms | 138.7 ms | 23.07 ms | 553 MB |

Each row matched 4/4 retrieval top-1s with 0 clear-pair flips and 0 repeat drift.

**Mac** (M4 Max, Mac16,9), macOS 27.0 build 26A428, the JIT `.aimodel`, GPU-preferred, fp32.
The driver refused to run while any foreign accelerator job was present; 105 warm samples.

| S | Gate | Min cosine vs HF | Max \|err\| | Load | First after load | **Warm median** |
|---:|---|---:|---:|---:|---:|---:|
| 128 | 35/35 | 0.99999999999967 | 2.98e-7 | 481 ms | 642 ms | **4.14 ms** |
| 512 | 35/35 | 0.99999999999887 | 2.98e-7 | 470 ms | 264 ms | **4.89 ms** |

The Mac h16c AOT twin also passed 70/70 (same numerics), but its timings were taken with another
lane's GPU evaluation running and are not reported. The w8 variant on Mac is gated on **CPU
only** (min cosine 0.999410, max |err| 5.67e-3, ranking exact); Mac GPU for w8 was not run.

The fixed grid computes every position, so pick the smallest grid that covers the text: S=128
for queries and short notes, S=512 for passages. **fp32 is the default.** w8 is a storage
option only — 22% smaller, not faster here — because the 180,000×384 fp32 vocabulary table is
276 MB of the bundle and palettization touches the 48 linear weights alone.

**Mac ANE (base M4, Mac mini), fp16, `--preferred-compute neural-engine`.** The fp32 bundle above
cannot reach the Neural Engine at all: `coreai-build compile --preferred-compute neural-engine`
forms **0 ANE regions** on it, which is a silent GPU fallback. fp16 forms 13; removing two fp32
ops from the graph forms **1** — the whole 12-layer body in a single ANE program — and that is
the configuration that beats the GPU path. 4000 warm iterations per row.

| Variant | S | Gate | Min cosine vs HF | Max \|err\| | Load | **Warm median** | ANE regions |
|---|---:|---|---:|---:|---:|---:|---:|
| fp32 (published), GPU | 128 | 35/35 | 0.999999881 | 2.98e-7 | 797 ms | 4.31 ms | **0** |
| fp16, as exported | 128 | 35/35 | 0.999957561 | 1.46e-3 | — | 13.19 ms | 13 |
| **fp16 + two fp32-op fixes** | 128 | 35/35 | **0.999958634** | 1.58e-3 | **257 ms** | **2.14 ms** | **1** |

The two fixes are one line each, both fp32 ops that Apple's authoring rules say create an f32
buffer the engine cannot execute — the attention softmax (`F.softmax(…, dtype=torch.float32)`) and
the pooling head's `.float()` cast. The progression is the point: **13 regions cost 13.19 ms,
1 region costs 4.60 ms** for the same graph at the same precision, because each boundary is a GPU
dispatch inside the layer loop (a (1,128,384) fp16 boundary tensor is ~98 KB ≈ 1 µs, so the cost
is latency, not transfer).

The embedding + retrieval gate is run on **runtime** embeddings: 4/4 queries keep their exact
top-1, 0 clear pair flips, worst score error 2.20e-03 (≤ 0.01), cosine 24× over the floor. The
**layer** gate (per-hidden-state ≤ 1e-4) is the one it fails — the gate whose documented purpose is
catching graph bugs (a window of 63 instead of 64 passes the embedding gate at cos 0.99995) — so
this is a uniform precision reduction, not a structural error. **Not a shippable variant**: the
published fp32 bundle stays the default.

Also measured, and negative: **w8 palettes with fp16 compute compile and pass the gate but are
4.8× slower** (int8-affine weights *fold* to dense fp16 before the data-movement step, so they save
storage and not bandwidth); **w6** compiles (14 regions) but fails the gate; **w4** forms 0 ANE
regions *and* inverts 16 document pairs. `--preferred-compute neural-engine` is baked into an AOT
bundle: the fp16 `.aimodelc` uses the ANE under both `--compute neuralEngine` and `--compute gpu`,
and only `cpuOnly` drops it to zero.

## Numerics gate

One gate at every stage, the oracle being official HF eager CPU fp32 (transformers 4.57.6):
per text cosine ≥ 0.999, max element error ≤ 0.02, L2-norm error ≤ 0.002; per query exact
top-1 over the 12 documents, retrieval-score error ≤ 0.01, and no inversion of any document
pair the oracle separates by ≥ 0.001; repeat drift ≤ 1e-6. A wrong-pairing control (every vector
matched to the wrong text) must FAIL.

- **Authoring** (`gate_granite_authoring.py`): the re-authored graph against every one of the 13
  saved hidden states, max |err| ≤ **1e-4** at fp32, both grids. Five mutations must trip it:
  all-global, all-local, ignore-padding and mean-pooling fail the embedding gate; a local radius of
  **63 instead of 64** passes the embedding gate (cos 0.99995) and fails only the layer gate —
  which is why the layer gate exists. Whole-model **fp16 fails** this layer gate on both grids.
- **Export**: the torch-exported, decomposed graph is gated before conversion, on both grids.
- **Runtime**: Mac CPU and GPU (JIT), Mac h16c AOT, iPhone h18p AOT — the tables above.
- **w8**: the same gate at prepared, finalized and decomposed stages, 48 `lut_to_dense` ops
  counted, palettes hashed; the iOS w8 export reuses the Mac palettes byte for byte.

`gate-granite-embedding-97m.json` beside this card is the transcript: the eight runtime rows
(min cosine, max error, retrieval, timings, device/OS build), the tokenizer gate and the
authoring gate, each with the sha256 of the full record it summarizes.

## ⬇️ Bundle

**[mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI](https://huggingface.co/mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI)**
— one folder per variant, each self-contained: the bundle, `tokenizer/`, `reference.json` (the
35 HF fixtures with ids, masks and embeddings — the parity test) and `provenance/` (export
manifest with per-file sha256, the runtime gate record). `coreai-kit.json` at the root maps
platform → folder.

| Folder | Platform | Format | Bundle | Bytes |
|---|---|---|---|---:|
| `macos/fp32-s512/` **(default)** | macOS 27 | JIT `.aimodel` | `granite97m_fp32_s512_bound.aimodel` | 390,431,506 |
| `macos/fp32-s128/` | macOS 27 | JIT `.aimodel` | `granite97m_fp32_s128_bound.aimodel` | 389,989,146 |
| `ios/fp32-s512/` **(default)** | iOS 27, **h18p only** | AOT `.aimodelc` | `granite97m_fp32_s512_bound.h18p.aimodelc` | 390,308,788 |
| `ios/fp32-s128/` | iOS 27, h18p only | AOT `.aimodelc` | `granite97m_fp32_s128_bound.h18p.aimodelc` | 390,081,410 |
| `macos/w8-fp32table-s512/` | macOS 27 (CPU-gated) | JIT `.aimodel` | `granite97m_w8_fp32table_s512.aimodel` | 305,569,358 |
| `macos/w8-fp32table-s128/` | macOS 27 (CPU-gated) | JIT `.aimodel` | `granite97m_w8_fp32table_s128.aimodel` | 305,126,985 |
| `ios/w8-fp32table-s512/` | iOS 27, h18p only | AOT `.aimodelc` | `granite97m_w8_fp32table_s512_r02.h18p.aimodelc` | 305,479,184 |
| `ios/w8-fp32table-s128/` | iOS 27, h18p only | AOT `.aimodelc` | `granite97m_w8_fp32table_s128_r02.h18p.aimodelc` | 305,251,774 |

The `ios/` bundles are compiled for one device architecture (`h18p`, the iPhone 17 Pro) with
`xcrun coreai-build compile --platform iOS --min-deployment-version 27.0 --preferred-compute gpu
--architecture h18p` (coreai-build 3600.83.1). **Never load an iOS bundle on a Mac.** Other
phones need their own compile from the recipe; the source IR is reproducible, not shipped.

Convert yourself: [`conversion/granite_embedding/`](../../conversion/granite_embedding/README.md)
— five staged scripts, `recipe.toml` here names the commands.

## CoreAIKit (Swift)

**Not enrolled** in the kit catalog. The kit's `TextEmbedder` pads with 0, truncates after adding
the special tokens (losing SEP), applies its own BPE without `ignore_merges`, discovers a single
`.aimodel`, and has no grid / architecture selection — every one of those is wrong for this
model. Running it today means: the Swift tokenizer from the HF repo's `host/` folder, a fixed
grid, `AIModel` on the platform's folder. Enrolling it needs a `textEmbedding` driver that takes
the pad id, a SEP-preserving truncation, a per-platform variant path and an AOT-aware loader —
tracked as maintainer work, not a blocker on the bundle.

## The port in one lesson: gate the layers, not just the vector

ModernBERT's alternating local/global attention is the whole risk. The config says
`local_attention: 128`; the executed window is inclusive `|i − j| ≤ 64` — 129 keys — and a
window of 63 reproduces the final embedding to cos 0.99995 while every hidden state past layer 1
is wrong. Only a per-layer oracle catches it. Three more things the raw checkpoint settles that
the modeling file hides: layer 0 has no attention norm (adding one loads a missing weight),
the two RoPE thetas are per-layer-kind, and the CLS/L2 head needs an explicit `clamp_min`
epsilon because the converter's `F.normalize` decomposition drops it.

## License and limits

Apache-2.0 at the pinned upstream revision; the HF repo carries IBM's unmodified card as
`UPSTREAM_README.md` and a `LICENSE-NOTE.md` listing the changes (static graph, in-graph
pooling, optional w8 palettes, h18p compile). Not tested: other phones or OS builds, the Mac GPU
with w8, dynamic or batched shapes, S > 512, languages beyond the JA/EN
fixtures, retrieval quality on a benchmark, sustained thermals, true cache-cold load.


The Neural Engine is **no longer untested** — see the Mac ANE block above. Short version: the fp32
bundle cannot use it (0 ANE regions), fp16 can, and two fp32-op removals make it worth having
(2.14 ms / 257 ms load against 4.31 ms / 797 ms for the published fp32 export, on a base M4). It
fails the layer gate, so it is a measured result and not a shipped variant. One observation is
left open: the ANE shows a burst state (~1.85 ms) and a sustained state (~4.75 ms) with a one-way
transition about 830 inferences into a run that 4 minutes of idle does not restore;
`powermetrics`' ANE power rail was not reliable enough to identify the cause.