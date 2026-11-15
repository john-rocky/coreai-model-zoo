# Porting Granite-Embedding-97M (ModernBERT): the window is 129 keys, and only a layer gate knows

Written 2026-09-19 from the port of `ibm-granite/granite-embedding-97m-multilingual-r2`
(revision `835ad140…`, 97,441,152 parameters, 74 tensors, Apache-2.0) to a static Core AI
graph for macOS 27 (JIT) and the iPhone 17 Pro (h18p AOT). Card and numbers:
[`models/granite-embedding-97m/`](../models/granite-embedding-97m/README.md); scripts:
[`conversion/granite_embedding/`](../conversion/granite_embedding/README.md). The general
embedder lessons are in [`porting-text-embedders.md`](porting-text-embedders.md); this note is
what ModernBERT adds.

## Re-author from the weights: four things the modeling file hides

- **Local attention is inclusive radius 64, not "128".** `config.local_attention: 128` is the
  window width; the executed mask is `|i − j| ≤ 64`, so an interior query sees **129** keys.
  Global attention is on layers 0, 3, 6, 9 (`global_attn_every_n_layers: 3`), local on the rest.
- **Two RoPE thetas, by layer kind.** Global layers use `global_rope_theta` 150,000, local
  layers `local_rope_theta` 160,000; half-split rotation (`cat(-x2, x1)`), positions `arange(S)`
  (not mask-cumsum), trig computed in fp32 exactly as HF does. Getting the theta per kind right
  matters more than it looks: the mutation that makes every layer global still reaches cos 0.995.
- **Layer 0 has no attention pre-norm.** The embedding LayerNorm serves; adding one produces a
  missing weight under `strict=True`. Every LayerNorm is biasless (`norm_bias: false`), ε 1e-5.
- **`position_embedding_type: absolute` in config.json is not what runs.** There is no absolute
  position table; positions are rotary. A port that reads the config field would add a
  nonexistent embedding.

Everything above came from `model.safetensors` and `AutoModel` under `strict=True` with a
per-layer oracle, not from reading the modeling file — which also carries branches (SDPA,
flash, unpadding) this checkpoint never takes on CPU eager.

## Gate the layers, not just the vector

A local radius of **63 instead of 64** — one key per side — reproduces the final embedding to
**cos 0.99995** on the 35-text fixture set. That passes the runtime gate (cos ≥ 0.999) and would
have shipped. It fails the per-layer gate (max |err| ≤ 1e-4 against every one of the 13 saved
hidden states) at layer 1 by a wide margin. So the authoring gate keeps every hidden state from
the oracle and holds the re-authored graph to 1e-4 at fp32, with five mutations that must go
red: all-global, all-local, ignore-padding and mean-pooling fail the embedding gate; `window63`
must fail the layer gate specifically. `gate_granite_authoring.py --negative-controls`.

The same gate says **whole-model fp16 is not shippable** here: it fails the 1e-4 layer bar on
both grids (the embedding-level cosine still looks fine). fp32 is the ship dtype; at 97M
parameters that is 390 MB and 5.5 ms per S=128 embedding on the phone, so nothing was lost.

## The tokenizer is the contract, and the stock one gets three things wrong

The model is small; the host recipe is where an integration fails silently.

| Rule | Why | What the wrong version does |
|---|---|---|
| Pad with **179935**, mask 0 | `pad_token_id` is not 0 in a 180k vocabulary | 0 is a real token; padded positions attend |
| Truncate the **body to S−2**, then add CLS 179934 / SEP 179938 | `truncation_side: right` applies to the body | truncating after wrapping drops SEP on every long text |
| **No stripping**, no normalization, no prefix | the upstream README path is raw `AutoTokenizer`; prompts are empty | sentence-transformers strips text first — `"  東京駅から…\n"` tokenizes differently |
| BPE with **`ignore_merges = true`** | a whole pre-token in the vocabulary is emitted as one id | ` ક` is id 2999 with the flag, three ids without it |
| A missing initial byte symbol is **dropped**, not UNK | `unk_token: null`, `byte_fallback: false`; HF's `merge_word` skips it | a NUL byte (mapped `Ā`, absent from this vocabulary) raises instead of vanishing |

The independent tokenizer (`_granite_tokenizer.py`, Python; `host/GraniteTokenizer.swift` in
the HF repo, Foundation only) is gated against `AutoTokenizer` over 681 texts × 2 grids
(1,362 id/mask comparisons: the model fixtures, every added token in five boundary contexts,
CJK/Indic/Arabic/emoji/control cases, 300 seeded random strings). The first version of the gate
passed with the `ignore_merges` mutation undetected — the corpus had no text that separated the
two behaviours — and was only accepted after scanning the vocabulary for one that does (token
2999). A gate that cannot go red for a mutation is not gating it.

CoreAIKit's `TextEmbedder` does the first two wrong and its BPE ignores the flag, which is why
this model is published without kit enrollment; the requirements are on the card.

## Compression: the table is 71% of the bundle

8-bit scalar k-means palettes on the 48 linear weights (per-tensor, 256-entry fp32 LUT,
`KMeansPalettizerConfig.presets.w8()`) pass the same gate at every stage — prepared, finalized,
torch-decomposed, Mac CPU, iPhone — at min cosine **0.999410**, max |err| 5.7e-3, ranking
exact. The bundle shrinks **390 → 305 MB (22%)** and is **not faster** (6.64 vs 5.54 ms at
S=128 on the phone), because palettization is `F.linear`/`F.conv` only and the 180,000 × 384
fp32 embedding table is 276 MB of the 390. `porting-text-embedders.md` already recorded this
for Nemotron (24% of parameters in the table); here it is 71% of the bytes.

The table lever is real: a **w8 + fp16-table** build is 167 MB and passed the same CPU gates
(the checkpoint is bf16, so an fp16 table represents every value exactly — the storage-only
change preserved every hidden state to the 1e-4 bar). It was not device-qualified and is not
published; it is the first thing to run when a smaller Granite is wanted.

One pin to know: `coreai-opt 0.2.1` requires `safetensors <= 0.7.0`, and the fp32 exporter's
environment resolved 0.8.0. The w8 script therefore declares its own environment; do not
loosen the fp32 one to fit it.

## Two runtime facts

- **The CLS/L2 head needs an explicit epsilon.** `F.normalize` decomposes without its `eps`
  clamp under `coreai-torch 0.4.1`; the graph computes `x * rsqrt(clamp_min(sum(x²), 1e-24))`
  itself. A zero vector cannot occur after LayerNorm, but a graph that divides by an unclamped
  norm is one export away from NaN.
- **A Python `NDArray` constructor SIGTRAP inside a sandbox is the sandbox.** The runtime gate
  in Python (`coreai.runtime`) trapped on a model-free `NDArray(...)` inside a workspace-write
  sandbox that also blocked `ps` and `devicectl`; the same script outside it loads, runs and
  gates the bundle in 1.4 s. If the trap appears before `INPUT_CREATED`, check the process
  boundary before the model.

## Numbers that travelled

iPhone 17 Pro (iOS 27.0 24A437, h18p AOT, GPU-preferred): fp32 **5.54 / 20.99 ms** warm median at
S=128 / 512, load 81 / 586 ms, min cosine 0.9999999999994, peak footprint 640 MB whole-process.
Mac M4 Max (26A428, JIT, GPU-preferred): **4.14 / 4.89 ms**, load 481 / 470 ms. The Mac h16c
AOT twin gated 70/70 with identical numerics; its timings were taken alongside another
GPU job and were discarded rather than labelled — a timing measured under contention is not a
timing with a caveat, it is a different measurement.
