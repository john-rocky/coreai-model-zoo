# MiniCPM5-1B / MiniCPM5-2B — Core AI conversion notes (reusable techniques)

MiniCPM5-1B (OpenBMB, Apache-2.0) is a plain `LlamaForCausalLM` (1.08B, GQA 16:2, RoPE θ=5e6,
RMSNorm, SiLU, explicit `head_dim` 128, untied head, vocab 130560, 128K, hybrid Think/No-Think).
The port produced no model-specific code — it's three reusable levers worth keeping.

## 1. Plain-Llama → the stock exporter via a `llama → mistral` remap

The stock `coreai.llm.export` graph registry has families for qwen2/qwen3/gemma/mistral/… but **no
`llama`**. A plain `LlamaForCausalLM` is architecturally identical to the **Mistral** builder minus
the sliding window: GQA + RoPE + RMSNorm + SiLU, **no qkv bias** (qwen2 has it), **no qk-norm**
(qwen3 has it), and the Mistral builder already honors an explicit `config.head_dim`. So:

```python
# coreai_models/models/registry.py — MODEL_TYPE_REMAPPING
"llama": "mistral",
```

is a one-line unlock for *any* plain-Llama checkpoint. Unregistered HF ids also need
`--experimental --compute-precision float16`. Validate with greedy parity vs HF (token-exact).

## 2. Clean weight-only INT8 without a custom decode-pipelined export

The macOS `--compression int8` preset is **iOS-palettization-only** (`AssertionError: palettization
is only supported for iOS variant`), and the zoo's int8 LLM bundles come from per-model
`export_*_decode_pipelined.py` scripts (none exists for plain Llama). But `coreai.llm.export
--compression-config <yaml>` accepts a **`quantization_config`** (macOS torch-pre-export via
coreai-opt `quantize_pytorch_model`) — write **symmetric per-channel int8, absmax (NO clipping;
clipping craters the big-vocab LM head — absmax keeps it lossless), SDPA/RoPE/RMSNorm excluded**
(see `conversion/minicpm5_int8sym.yaml`). coreai-opt has only PTQ/palettization/pruning — **no
GPTQ/AWQ** — so symmetric-per-channel int8 is the clean ceiling; int4 hits the non-QAT cliff.

## 3. Ship a DYNAMIC-shape bundle for the iPhone (pipelined engine), not a static iOS export

`EngineFactory.autoDetectVariant`: **dynamic structure → pipelined engine** / **chunkedStatic →
staticShape engine**. A `coreai.llm.export --platform iOS` static bundle is detected as
chunkedStatic and routed to the staticShape engine, which expects `extend_*` / `load_embeddings`
multi-graph functions an FM-format bundle doesn't provide → `NSPOSIXError 2` at engine-create on
device. The **dynamic** FM-format bundle (the macOS default export) routes to the **pipelined
engine** and runs unchanged on both macOS and iPhone. So for the CoreAIChat / pipelined path, ship
the dynamic bundle (sideload to `Documents/models/<name>`; `LanguageBundle` + `EngineFactory`
load it like any pipelined model). Cold first-load is one-time (~45 s JIT spec); the cache persists
→ warm loads ~2–5 s, so AOT (`.aimodelc`) is unnecessary.

Chat EOS: base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073) —
set the bundle's tokenizer `eos_token` to `<|im_end|>` (as Qwen ships) or generation never halts.

## Result

iPhone 17 Pro (`PipelinedBench`): **int8 decode 66.8 / prefill 68.0 tok/s, 24/24 token-exact vs HF
fp32 (lossless), 1.0 GB** — ~2.2× fp16 (decode is bandwidth-bound → half the weight read ≈ double
throughput) at no quality cost. 🤗 `mlboydaisuke/MiniCPM5-1B-CoreAI`.

**Mac is the opposite — for PER-CHANNEL int8.** On a compute-rich M4 Max int8 is ~59 tok/s vs fp16's
~208 — when bandwidth isn't the bottleneck the per-channel dequant overhead dominates. **This is a
property of the granularity, not of int8**: the 2B section below measured per-block-32 int8 at
1.6× *fp16's* Mac decode. The per-channel bundle is an iPhone win only; per-block-32 wins on both.

## App integration (CoreAIChat — applies to any Think-mode model)

- **Generation budget.** MiniCPM5 is hybrid Think/No-Think and the `<think>` trace alone can run
  several hundred tokens, so a small `maxNew` cap (the app shipped 1024) truncates the answer
  mid-stream. Cap generously (4096) — the model emits its own eos well before that.
- **Decode-rate measurement.** The chat loop must NOT re-decode the whole token list + push a
  SwiftUI update on every token (O(n²)) *inside* the decode timer — it drags both the measured and
  the experienced rate below the true model rate (in-app read ~59 vs the 66.8 bench). Throttle the
  live refresh to ~25 fps and **exclude the UI-callback time from the decode timer**; after that the
  in-app rate matches `PipelinedBench`. Full write-up: `int8-head-and-decode-measurement.md`.

## MiniCPM5-2B (2026-09-06 release) — the recipe, one YAML apart

`openbmb/MiniCPM5-2B` is the same `LlamaForCausalLM` family scaled up — 42 layers × hidden 2048,
intermediate 6144, GQA 16:2, `head_dim` 128, RoPE θ 5e6, no scaling, untied 130560-vocab head,
128K, the same `</s>` vs `<|im_end|>` (130073) chat-EOS split — so the port is the three levers
above with **zero new model code**: the `llama → mistral` remap already in the overlay, one
`AIModelMetadataFields` entry for the new id, and `export_minicpm5.py --hf-id openbmb/MiniCPM5-2B
--qconfig minicpm5_int8sym_b32.yaml` (the wrapper grew both flags; the 1B and its per-channel
yaml are the defaults). Shipped bundle: int8 per-block-32, **2.67 GB** (`main.mlirb`
2,674,882,267 bytes — a single 2.49 GiB file); the per-channel sibling is 2.52 GB.

What the re-run taught:

- **Choose the device-gate prompt by margin, not by habit.** The 1B's `PipelinedBench` spec
  free-runs `"The capital of France is"` (no BOS) and scored 24/24 there. On the 2B that same
  prompt has **six knife-edge positions** (fp32 top-2 margin 0.022–0.089; the continuation
  wanders into Python code), so a device mismatch on it would tell you nothing — exactly the
  coin-flip `cli/coreai_verify.py` refuses (`DEFAULT_PROMPT`, margin floor 0.1). Swept eight
  candidates in fp32 first (`prompt_sweep`: raw/BOS × five prompts + a chat-templated one):
  `"The alphabet begins A, B, C, D, E, F,"` clears every one of 24 positions at **min margin
  0.916** (0.996 through the chat template with `enable_thinking=False`); "count to twenty" and
  "days of the week" both carry ties. The 2B spec uses the alphabet prompt; nat == oracle.
- **Mac gate:** `cli/coreai_verify.py` stock backend, fp32 oracle, **16/16 token-exact**
  (`models/minicpm5-2b/gate-minicpm5-2b.json`). The four-prompt free-run
  (`verify_minicpm5.py --hf-id openbmb/MiniCPM5-2B`, 30 tokens) is 2/4 per-channel and 3/4
  per-block-32; see the granularity table below for what the misses are.
- **iPhone 17 Pro (`PipelinedBench`, Release):** **engine ready 28.9 s cold** (a 2.67 GB single-file dynamic bundle JIT-specializes on the phone — no AOT; needs `com.apple.developer.kernel.increased-memory-limit`, and ~3 GB of free storage for the compile cache: the first attempt died with `LLVM ERROR: IO failure on output stream: No space left on device` on a full phone), **nat 24/24 + oracle 24/24**, **decode 22.4 / prefill 27.3 tok/s** (two trials 24.5→20.3; the per-channel bundle measured 22.7 / 28.9 on the same phone the same night — the phone is bandwidth-bound either way, so the block scales cost nothing there). The old "capital of France" spec prompt would have been unusable here (six ties).
- **Granularity decides the Mac number, and it is not subtle.** `llm-benchmark` (512p/1024g,
  5 trials, M4 Max, dynamic bundle so the 512-token prefill runs as one graph):

  | bundle | size | decode | prefill | 4-prompt free-run vs fp32 (30 tok) |
  |---|---:|---:|---:|---|
  | int8 **per-channel** absmax (the 1B's yaml) | 2.52 GB | **25.6 tok/s** | 2018 | 2/4 — one 0.006 tie, one **0.245-margin flip** (`,`→` and` at +10) |
  | fp16 (no compression; control) | 5.03 GB | 80.0 | 2775 | 4/4 |
  | int8 **per-block-32** (`minicpm5_int8sym_b32.yaml`) | 2.67 GB | **127.6 tok/s** | 2654 | 3/4 — only the 0.006 tie remains |

  Per-channel int8 lowers to a slow dequant path on the Mac GPU (the 1B's "int8 is slower than
  fp16" observation); per-block-32 lands on the fast quantized-matmul path the zoo's `int8hu
  block32` bundles use, and comes out **5× the per-channel decode and 1.6× fp16's**, at +155 MB
  (6%). It also removes the one non-tie greedy divergence: with block scales every position
  whose fp32 margin clears 0.1 matches the fp32 oracle, which is the standard the margin-aware
  gate holds a bundle to. The YAML delta is three lines (`granularity: {type: per_block,
  block_size: 32}`, axis resolved per module by the quantizer). **Lead, not measured here:** the
  published 1B bundle is per-channel; the same three-line change should lift its Mac decode by
  the same mechanism.
- **Free-run divergences are read by fp32 margin, not counted.** `verify_minicpm5.py` scores
  exact 30-token text on four prompts. The fp16 control export scores 4/4, so any int8 miss is
  quantization, not the engine; then the margin at the first divergent token says whether it
  matters: `Emma`/`Lily` (0.2126 vs 0.2065, margin 0.006) is a tie any precision may flip;
  `,`/` and` (0.612 vs 0.367, margin 0.245) is a real per-channel error, gone at per-block-32.
- **Publish record.** The per-block bundle was exported with the bare CLI for the A/B, and that
  skipped the wrapper's chat-EOS rewrite: the first Hub commit shipped `</s>` as eos. Tier-1 verify
  against `verify.toml` flagged it (`eos: '</s>', declared expectation '<|im_end|>'`) and the two
  tokenizer files were re-published (weights unchanged; kit re-pinned). The engine still stopped
  cleanly on the wrong file in a chat-template run, so this is the kind of drift only the
  declared-expectation check sees.
- **Spec-decode test bed.** OpenBMB ships `openbmb/MiniCPM5-2B-DSpark`: a 5-layer, 324M
  DSpark draft (7 draft tokens per pass, `num_target_layers` 42, the target's tokenizer) trained
  for exact pairing with this checkpoint. Not ported here; a dense target with an official
  drafter is the cleanest place to measure the verify-cost staircase from
  `spec-decode-ngram-dense.md`, and Apple's `coreai-models` just grew a drafter path
  (DFlash for Muse Glimmer, upstream #228, 2026-09-05).
