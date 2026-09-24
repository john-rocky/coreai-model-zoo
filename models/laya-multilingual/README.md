# laya multilingual — Core AI

A **typed decision model** that answers in one forward pass: give it a state (a message, an email, a
JSON record, a conversation) and a typed question — `choice` (pick one of named options), `score`
(an ordered scale) or `noul` (yes/no) — and it returns a probability for every option, no generation.
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), subfolder `multilingual/`
(Apache-2.0, revision `1c5edc17…`, laya 0.3.4), as a static `.aimodel` for macOS 27 and for the iPhone
17 Pro (run there 2026-09-23: the same 201 rows pass, 47 ms per decision on the GPU). It is the catalog's first **encoder-type
decision model**: the other
decision models here are language models read out after a prefill; this one scores every option at its
own marker position in a single encoder call.

Architecture (from `model.safetensors` and `encoder/config.json`; 321,908,998 parameters, all stored in F16
but the three-value temperature buffer):
an mmBERT-base encoder — ModernBERT layout, 22 layers, hidden 768, 12 heads × 64, GLU MLP 1152 with exact
GELU, biasless attention/MLP/LayerNorm (ε 1e-5), vocabulary 256,000 — with global attention on layers
0, 3, …, 21 and a sliding window of inclusive radius 64 elsewhere, RoPE θ 160,000 for both kinds; then a
type embedding (choice / score / noul), two pre-norm transformer layers (768, 12 heads, ReLU 3072), a scorer (LayerNorm → Linear → GELU → Linear) at every position, and a small act head.

**This is an encoder, not a generator.** One question is one forward over a right-padded window; no
KV cache, no sampling loop. It runs through raw `AIModel` calls, not the generate engine.

## Graph contract

```
function "main"
  input  "input_ids"       [1, S]    int32    right-padded with PAD 0
  input  "attention_mask"  [1, S]    int32    1 over real tokens, 0 over padding
  input  "qtype_onehot"    [1, 3]    fp32     choice / score / noul
  output "token_logits"    [1, S]    fp32     the scorer at every position — read at the option markers
  output "pooled_cls"      [1, 768]  fp32     input of "act"
function "act"                                  fp32 in every variant
  input  "pooled_cls"      [1, 768]  fp32
  input  "feats"           [1, 4]    fp32     top1, top1 − top2, entropy / ln max(K,2), max(K,2) / 255
  output "act_logits"      [1, 2]    fp32     class 0 = answer directly
S = 256 or 512 (export-time choice); batch = 1; one bundle holds both functions
```

**Host recipe** — the sequence and the readout are the publisher's, unchanged:
- Build `[CLS 2] <type> question: <instructions> [SEP 1] [MASK 4] option … [MASK 4] option … [SEP 1]
  <state> [SEP 1]` with laya's `build_sequence` (option text ≤ 48 tokens, head budget 256, state
  right-truncated to the window; the marker positions are where the `[MASK]` ids sit). Every text piece is
  tokenized without special tokens by the checkpoint's `tokenizer.json` (Gemma-style BPE, byte fallback).
- Gather `token_logits` at the markers (K logits), softmax them **raw** for the four `feats`, run `act`;
  then divide the K logits by the question's temperature (bucket by type and K first, then per type) and
  softmax for the answer. `metadata.json` carries the checkpoint's temperatures (T = 1) and the fitted
  calibration the LiteRT port published (`temperature`, `temperature_by_options`).
- `reference.json` in every folder is the parity instrument: 201 question rows for that window with token
  ids, marker positions, the publisher's answers at T = 1 and the official model's raw logits.

## Measured

**Mac** (Apple M4 Max), macOS 27.0 build **26A428**, coreai-build 3600.83.1, the JIT `.aimodel`,
2026-09-23. Every row: the 201 rows of its window through the whole pipeline (main → host features →
act), 3 runs per row, then 40 warm questions; the warm time is one whole question, NumPy I/O included.
The GPU rows ran alone on the GPU (the machine-wide GPU lock held, no other GPU job visible during the
run). `cpu_only` is the parity option, and its milliseconds were taken while other jobs used the machine's
CPU: reference values, not a speed claim.

| Variant | S | Compute | Argmax (choice + score) | Max \|Δp\| at T=1 | Marker max \|Δ\| | Act relative | Repeat drift | Load | Warm median |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| wfp16 | 256 | GPU | 81/81 | 4.47e-6 | 5.53e-5 | 2.60e-6 | 0 | 570 ms | **12.1 ms** |
| wfp16 | 512 | GPU | 81/81 | 4.59e-6 | 1.00e-4 | 3.47e-6 | 0 | 589 ms | **19.5 ms** |
| fp32 | 256 | GPU | 81/81 | 4.47e-6 | 5.53e-5 | 2.60e-6 | 0 | 853 ms | 10.4 ms |
| fp32 | 512 | GPU | 81/81 | 4.59e-6 | 1.00e-4 | 3.47e-6 | 0 | 982 ms | 17.2 ms |
| wfp16 | 256 | cpu_only | 81/81 | 9.48e-6 | 7.49e-5 | 1.85e-6 | 0 | 720 ms | 45.1 ms |
| wfp16 | 512 | cpu_only | 81/81 | 9.48e-6 | 6.68e-5 | 1.94e-6 | 0 | 600 ms | 86.2 ms |
| fp32 | 256 | cpu_only | 81/81 | 9.48e-6 | 7.49e-5 | 1.85e-6 | 0 | 762 ms | 63.0 ms |
| fp32 | 512 | cpu_only | 81/81 | 9.48e-6 | 6.68e-5 | 1.94e-6 | 0 | 720 ms | 84.2 ms |

On the CPU, wfp16 and fp32 return bit-identical marker and act logits on all 402 rows: wfp16 stores the
weights in fp16 — exact, the checkpoint is F16 — and computes in fp32. On this Mac's GPU both stay within
1e-4 of the CPU run as well. **Use the GPU, and request it explicitly**: with the Neural Engine preference
the Mac returned the GPU's results bit for bit for fp32, but for wfp16 it returned different results that
changed from run to run (next section). **iPhone 17 Pro** (iOS 27.0 24A437, the `ios/wfp16-s256` JIT bundle,
2026-09-23, a headless harness over CoreAIKit's `EncoderDecider.decideRow` on the 201 fixture rows, the
fixture's own token ids, T = 1): on the GPU **201/201 rows within 1e-3, argmax 81/81, max |Δp| 8.1e-6**,
act probability delta 0, no drift over three passes; CPU-only the same verdict at 9.4e-6. The kit's builder
renders every row identically on the phone (201/201 tokens and markers). One decision on the phone's GPU:
**53 ms median** (p90 57 ms, 30 warm-up calls, 603 timed calls) with the row's tokens given, 69–70 ms
through `TypedDecisions` with the state's tokens kept (the question tokenized on the phone, 5 ms median),
73 ms CPU-only; the bundle loads in 1.2–2.5 s, the process peaks at 357 MB (670 MB CPU-only). With a Neural
Engine preference the phone misses the bar the way the Mac does — 196/201 within 1e-3, argmax 80/81, max
|Δp| 0.33 on five rows — at 82 ms and 1.4 GB, so the kit keeps refusing that preference for this bundle.
Thermal state during those runs: "fair" for the GPU rows, "serious" for the CPU-only and Neural Engine
rows; the GPU gate run again at "nominal" (2026-09-24) gave the same verdict at 54.5 ms median (p90 59.6),
so the row cost is not thermal-bound. Through the kit's own `decide-cli bench --repeat 3` on the cool phone
(a 109-token state, eight questions, the measure of the Mac's 11.5 ms): **47.1 ms per decision** with the
state shared, 46.4 from scratch, 390 / 458 ms for the state and its eight. `gate-laya-multilingual-iphone-gpu.json`
is the first GPU run's record. The AOT `ios-h18p/wfp16-s256` bundle, run the same way on the cool phone
(2026-09-24): the same verdict (201/201, argmax 81/81, max |Δp| 8.1e-6) at **51.0 ms** median (p90 54.4), first call
68.6 ms after a 1.7 s load, peak 357 MB — no faster to load or to run than the JIT bundle the catalog ships.

The publisher's own model on the SemIf `authored144` fixture (144 three-option evidence / rule /
candidate questions it was not trained for), as `laya.load(...).predict` answers them: mean family
balanced accuracy **0.6114**, accuracy 0.5903 (the same at T = 1 and with the fitted calibration, and at
both windows). That is the reference a port's decisions are compared with, not a claim about the port.

### JevBench public 231 (Mac, 2026-09-24)

| easy 48 | standard 72 | hard 111 | ECE hard | p50 | p95 | hard max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.896 | 0.403 | 0.342 | 0.273 | 0.02 s | 0.20 s | 0.3 s |

The benchmark's own harness ([fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)
`2fa63fa`, v1.4.0, `typesafe` adapter) ran the 231 public items against coreai-kit `adbc755`
`decide-cli serve` with the `macos/wfp16-s256` bundle, one question per request. The bundle's input
window is 256 tokens: 95 of the 111 hard items were truncated to it (hard states are up to 3,677 tokens
long), and no easy or standard item was (the longest is 107 tokens). Accuracy per tier and the hard
tier's ECE are JevBench's own scoring (argmax of the returned probabilities); p50 and p95 are per-request
latency over all 231 requests, hard max the maximum over the hard tier. Latency was measured without an
exclusive GPU window (contended), with this model's server running alone. JevBench's published scores
(Intelligence and the rest) are chance-corrected over 534 items, sealed ones included, and are not
comparable to these accuracies.

## Numerics gate

The oracle is the publisher's package itself (`laya.load(<pinned snapshot>, subfolder="multilingual")`,
transformers 5.17.0, CPU fp32): it reproduces the frozen fixture of the LiteRT port exactly (token ids
402/402; batched answers and logits bit-identical), and each row alone, right-padded to its window, is the
tensor reference. Bars, at every stage: argmax identical on every choice and score row, max |Δp| ≤ 1e-3 at
T = 1 over the options, |Δ act probability| ≤ 1e-3; marker logits ≤ 1e-3 and act logits ≤ 1e-4 relative
to the official batch-1 run; repeat drift ≤ 1e-6 on CPU; a wrong-pairing control (every row judged
against another same-shape row's outputs) must fail — it does, on 162–166 of 200 pairs.

- **Authoring** (`gate_laya_authoring.py`): the 201 rows of each window, plus every hidden state of 14
  rows (embeddings, 22 layers, final norm, after the type embedding, both head layers) against the
  official model — absolute max |err| ≤ 1e-4 on the embeddings and layers 0–9, relative max |err| / max
  |ref| ≤ 2e-4 on layer 10 through the head (measured: 5.0e-5 absolute, 5.7e-5 relative at S=256, 7.8e-6
  at S=512). Five mutations must be caught and are, by the layer, tensor and answer gates: a local radius
  of 63 (layer 1, 1.24 / 0.84 at S=256 / 512), all-global, all-local, ignoring padding, and dropping the
  type embedding (after the type embedding, 6.6e-2 relative). A pad-isolation check replaces every pad id
  with a random token: real positions stay bit-identical.
- **Export**: the torch-exported, decomposed `main` and `act` pass the same 201-row gate before conversion.
- **Runtime**: the tables above.

`gate-laya-multilingual.json` beside this card is the transcript: every stage's summary with the sha256
of the full record it came from; the records keep each row's raw marker and act logits, unrounded.

## Neural Engine and fp16

The checkpoint is F16, so fp16 **storage** is exact. fp16 **compute** is not good enough here. The recipe
that runs everything in fp16 except the RoPE application and the attention softmax misses the answer bar
everywhere it was measured — torch max |Δp| 3.4e-3 / 5.0e-3 (S=256 / 512); Mac CPU 4.38e-2 with one choice
flipped (80/81); Mac GPU 5.84e-3; Mac Neural Engine 1.85e-2 with one choice flipped (both windows) — and keeping
the residual stream and LayerNorm in fp32 as well still leaves 4.9e-3 in torch (S=256).

Placement follows the compute precision here. Compiled for the iPhone 17 Pro with `--preferred-compute
neural-engine`, the fp16 recipe gets 47 Neural Engine regions; wfp16, which computes in fp32, gets a single
region of 5.7 KB of IR and everything else stays in the GPU package. On the Mac, the Neural Engine
preference gave these results:

| Variant | S | Argmax | Max \|Δp\| | Marker max \|Δ\| | Repeat drift | Load | Warm median | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 | 256 / 512 | 81/81 | 4.47e-6 / 4.59e-6 | 5.53e-5 / 1.00e-4 | 0 | 986 / 1,026 ms | 10.5 / 20.4 ms | identical to the GPU run on every row: it ran on the GPU |
| wfp16 | 256 | 81/81, 81/81 | 0.290, 0.164 | 0.82, 1.66 | **90, 90** | 1,186 ms | 12.7 ms | two runs, different answers each time: a defect of this specialization, do not use |
| wfp16 | 512 | 81/81, **80/81** | 0.343, 0.238 | 7.76, 11.4 | **202, 212** | 1,188 ms | 20.1 ms | the same, and the second run flipped a choice |
| fp16 recipe | 256 | 80/81 | 1.85e-2 | 0.211 | 0 | 5,919 ms | 42.9 ms | deterministic, but outside the bar |
| fp16 recipe | 512 | 80/81 | 1.85e-2 | 0.681 | 0 | 5 ms (cached) | 67.7 ms | the same |

`coreai.runtime` reports no placement, so the reading column is inferred from the numbers. The iPhone
folder is therefore the GPU compile, named for what it is (`ios-h18p/`); the neural-engine compile's
region count is recorded in its manifest.

## ⬇️ Bundle

**[mlboydaisuke/Laya-Multilingual-CoreAI](https://huggingface.co/mlboydaisuke/Laya-Multilingual-CoreAI)**
(revision `1175a4e6231fdfe8946e6566276f8d71eb8f02ef`, 2026-09-23; every file's sha256 and size checked against the
staging manifest after the upload). One folder per variant, each
self-contained: the bundle, `tokenizer/` (the checkpoint's files, unmodified), `metadata.json` (the
decision contract), `reference.json` and `provenance/` (export manifest with per-file sha256, the export
and runtime gate records).

| Folder | Platform | Format | Bundle | Bytes |
|---|---|---|---|---:|
| `macos/wfp16-s256/` | macOS 27 | JIT `.aimodel` | `laya_ml_wfp16_s256.aimodel` | 644,855,189 |
| `macos/wfp16-s512/` | macOS 27 | JIT `.aimodel` | `laya_ml_wfp16_s512.aimodel` | 645,772,737 |
| `macos/fp32-s256/` | macOS 27 (reference) | JIT `.aimodel` | `laya_ml_fp32_s256.aimodel` | 1,288,156,432 |
| `macos/fp32-s512/` | macOS 27 (reference) | JIT `.aimodel` | `laya_ml_fp32_s512.aimodel` | 1,289,073,976 |
| `ios/wfp16-s256/` | iOS 27 (portable JIT) | JIT `.aimodel` | `laya_ml_wfp16_s256.aimodel` | 644,855,189 |
| `ios/wfp16-s512/` | iOS 27 (portable JIT) | JIT `.aimodel` | `laya_ml_wfp16_s512.aimodel` | 645,772,737 |
| `ios-h18p/wfp16-s256/` | iOS 27, **h18p only** | AOT `.aimodelc` | `laya_ml_wfp16_s256.h18p.aimodelc` | 645,117,160 |
| `ios-h18p/wfp16-s512/` | iOS 27, **h18p only** | AOT `.aimodelc` | `laya_ml_wfp16_s512.h18p.aimodelc` | 646,034,970 |

The `ios-h18p/` bundles are compiled for one device architecture (`h18p`, the iPhone 17 Pro) with
`xcrun coreai-build compile --platform iOS --min-deployment-version 27.0 --preferred-compute gpu
--architecture h18p` (coreai-build 3600.83.1). **Never load an iOS bundle on a Mac.**

Convert yourself: [`conversion/laya/`](../../conversion/laya/README.md) — staged scripts, and
`recipe.toml` here names the commands.

## CoreAIKit (Swift)

Catalog id `laya-multilingual` (active once the kit's encoder backend is merged). `TypedDecisions` loads
the bundle as an encoder backend — the same `decide` / `prefill` calls as the kit's language-model
decision models — and runs it on the GPU; a Neural Engine preference is refused at load.

Measured through the kit on the same Mac (M4 Max, macOS 27.0 26A428, the wfp16 bundles above, GPU lock
held, 2026-09-23):

- Parity on the 201 rows of each window: token ids and marker positions 201/201, argmax 81/81, max |Δp|
  5e-6 on the GPU and 9e-6 with `cpuOnly`.
- `decide-cli bench` (a 109-token state, 8 questions, warm): **11.5 ms** per decision at S=256 when the
  state is tokenized once and shared, 12.1 ms when every decision tokenizes its row; 18.9 / 19.4 ms at S=512.
- Load 1.06–1.16 s, of which the tokenizer (the 256,000-entry `tokenizer.json`) takes about 1.0 s and
  145–162 MB; the whole process is 474–498 MB after the first decision.
- SemIf authored144 at the kit's default temperature (the fitted calibration): argmax 144/144 with the
  publisher's model at the same temperature, max |Δp| 2.9e-6; mean family balanced accuracy 0.6114,
  accuracy 0.5903.
- With a Neural Engine preference (measured before it was refused): 168–183 of 201 rows within the 1e-3
  bar over four runs, max |Δp| 0.61.

`measurements-coreai-kit.json` beside this card holds these numbers and the records they came from.

## The port in one lesson: a massive activation moves the layer bar

From layer 11 the encoder parks a value of about 14,000 on a few dimensions of the first token. One
fp32 step there is about 0.001, so the publisher's own two attention paths (SDPA, which the package uses,
and eager) already disagree by up to 0.048 on those layers while their marker logits agree to 3.5e-5. The
publisher's own eager path does not meet a per-layer bar of 1e-4 above layer 10, so it cannot be the bar
for a reimplementation there; the gate keeps 1e-4 where it still separates a correct graph from a wrong
one (the embeddings and layers 0–9, where a window of 63 instead of 64 shows up at 0.8–1.2) and switches
to a relative bar of twice the official spread above. In the fp16 recipe the largest relative error sits
at the final norm (0.13 at S=256 in torch), the state that normalizes those large values.

## License and limits

Apache-2.0 at the pinned upstream revision. Not tested: the 512 window on the phone (the 256 window was, JIT
and AOT h18p, on one iPhone 17 Pro), other Macs or OS builds, dynamic or batched shapes, windows other than 256 and 512, more than 20 options, the
fitted calibration's quality at S = 512, languages beyond the fixture's English, Japanese and mixed rows,
sustained thermals. The act probability is saturated at 1.0 on every fixture row, in the publisher's
model as here: it is carried through the graph, not evidence of when to escalate.
