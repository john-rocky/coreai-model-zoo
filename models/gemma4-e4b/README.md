# Gemma 4 E4B (text decoder) — Core AI

Gemma 4 E4B text decoder on the pipelined-engine fast path, ported 2026-06-11 **directly
from Google's QAT release**
[`google/gemma-4-E4B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized)
(bf16 weights *trained* for q4_0 rounding = per-block-32 absmax symmetric int4 — exactly
the `int4lin` recipe class, so **int4 ≈ bf16 quality by design**, per Google "preserving
similar quality to bfloat16").

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI)** —
`gpu-pipelined/` provider (both platforms) + precompiled `_aotc_h18p` + tbl (Mac-fastest),
plus the QAT gather tables under `ios-frontend/gemma4_e4b_qat_gather_raw/`.

<!-- gen-cards:use-it begin id=gemma-4-e4b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let tldr = try await CoreAI.summarize(text, options: .model("gemma-4-e4b"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [ChatDemo runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo)
(GUI + CLI, one app for every chat model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj
# → Run, then pick "Gemma 4 E4B" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model gemma-4-e4b --prompt "What can you do, offline?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let chat = try await ChatSession(catalog: "gemma-4-e4b")
let reply = try await chat.respond(to: prompt)
// reply: the answer, generated fully on-device
```

Also runs behind **Apple's FoundationModels API** — CoreAIKit's [`KitGemmaModel`](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api) plugs this bundle into the system `LanguageModelSession`; capabilities (tool calling, guided generation) auto-detect per model.

The take-home is [`Examples/ChatDemo/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/ChatDemo/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `ChatSession` across turns for its transcript.
Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps the
conversation history; `streamResponse(to:)` yields tokens as they decode.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed (macOS)
- First run downloads the model — 7.6 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Architecture (config + checkpoint verified — corrects the old "E4B adds MoE" note)

**Clean dense model, no MoE** (`moe_intermediate_size: null`). vs E2B: 42 layers
(full attention every 6th: 5/11/17/23/29/35/41), hidden 2560, intermediate **uniform
10240** (`use_double_wide_mlp: false` — E2B's shared-layer double-wide is a config flag),
**2 KV heads** (E2B: 1), 18 KV-shared layers → **24 unified KV slots** (20 sliding + 4
full), same dual head_dim 256/512, dual RoPE, PLE (ld 256 → table [262144, 10752]),
softcap 30, vocab 262144. The QAT checkpoint prunes the dead shared-layer KV projections
(k_proj/v_proj/k_norm — base checkpoints carry them unused).

**The port needed ZERO model-code changes** — `gemma4_text.py`/`gemma4_pipelined.py` are
config-driven and every E4B difference arrives through the HF config; the only session
code change was tolerating the pruned dead weights in the loader.

## Verification (the full ladder, first run each)

| check | result |
|---|---|
| fp32 oracle (margin rule ≥ 0.1) | min decision margin **0.624** PASS ("1, 2, 3," — E4B counts in numerals where E2B spells words) |
| torch ladder: pipelined wrapper S=1 vs stateless full-pass | **27/27 positions**, full-pass + wrapper both 8/8 vs HF |
| `int4lin` bundle (3.7 GB) python GPU gate | **8/8 PASS** |
| `int4lin --tbl` bundle python GPU gate | **8/8 PASS** |
| ENGINE path oracle (both bundles) | **8/8 PASS** |

## Throughput (M4 Max, release, `COREAI_CHUNK_THRESHOLD=1`, p128 g256 n3)

| config | bundle | prefill | decode |
|---|---:|---:|---:|
| `int4lin` — PLE rows per-token (host mmap provider) | 3.7 GB | 62.6 | 53.2 |
| `int4lin --tbl` — PLE table as static input | 3.7 GB + 2.7 GB tables | 61.0 | **55.8** |

Per-token reads ≈ 2.1 GB int4 — decode scales sub-linearly vs E2B (74.7/78.9) because the
fixed per-step cost amortizes over the bigger matmuls. AOT h18p compiles clean even at
3.7 GB graph constants (the on-device specializer ceiling is far lower — AOT is mandatory
on iPhone).

## iPhone 17 Pro — measured (AOT h18p, provider mode, settled)

| | result |
|---|---|
| Numerics | **mac-seq 24/24 + hf-oracle 8/8 PASS** (token-identical to M4 Max, both runs; all 24 are real decisions — no EOS inside the window) |
| Decode / prefill | **15.1 / 21.3 tok/s** (p128 g256; install-adjacent r1 ≈ settled r2, the number is stable) |
| Engine load | 19.3 s cold (3.7 GB AOT ingest) / 10.1 s warm |
| Memory | footprint peaks **2.2 GB**, headroom **4.2 GB** — provider-mode mmap PLE + evictable AOT pages make jetsam a non-issue even at E4B size |

Both phases land exactly on the known cost model: ~2.1 GB int4/tok ÷ ~47 GB/s effective
≈ 45 ms BW floor (= prefill 21.3) + the ~13 ms/tok provider sampler round-trip (= decode
15). Only **provider mode** is arithmetically possible on a 12 GB phone (~6.4 GB entitled
jetsam limit): the tbl variant would need the 3.7 GB graph + 2.7 GB of owned table bytes.

**Verdict: ships BOTH platforms** — the first 4B-class Gemma on iPhone in this project,
with the official QAT int4 quality guarantee. (Needs the `increased-memory-limit`
entitlement only as headroom insurance; measured footprint stays low.)

## Reproduce

```
# oracle + PLE dump are CHECKPOINT-derived — regenerate for any new weights:
~/.venv-coreml-llm-py312/bin/python ondevice/gen_gemma4_prompt.py 8 \
    --hf-id google/gemma-4-E4B-it-qat-q4_0-unquantized --tag e4b_qat
.venv/bin/python ondevice/export_gemma4_gather_raw.py \
    --hf-id google/gemma-4-E4B-it-qat-q4_0-unquantized --out …/gemma4_e4b_qat_gather_raw
# export (conversion/export_gemma4_decode_pipelined.py is --hf-id aware):
.venv/bin/python conversion/export_gemma4_decode_pipelined.py int4lin \
    --hf-id google/gemma-4-E4B-it-qat-q4_0-unquantized            # + [--tbl --raw-dir …]
# gate (E4B unified-KV geometry):
GEMMA_REF=…/_gen_ref_e4b_qat.json GEMMA_RAW=…/gemma4_e4b_qat_gather_raw \
GEMMA_SLOTS=24 GEMMA_NKV=2 .venv/bin/python _smoke/probe_gemma4_decode_parity.py <bundle>
```

Method, run contract, and every trap: [`../knowledge/pipelined-engine.md`](../../knowledge/pipelined-engine.md);
shared gemma4 details (PLE riding, engine patches, AOT): [`gemma4-e2b.md`](../gemma4-e2b/README.md).
