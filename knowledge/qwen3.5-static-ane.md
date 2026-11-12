# Qwen3.5 (GDN hybrid) on the Neural Engine via Apple's static iOS path — measured 2026-09-15

Companion to [`pipelined-engine.md`](pipelined-engine.md) (the shipped GPU paths for qwen3.5) and to
`minicpm5-1b.md` §2026-09-15 (the stock static export + AOT + on-device gate that works for plain dense
models). This note records what the same toolchain does with the hybrid linear-attention decoder, on
the release toolchain, so nobody re-derives it. Mac only; no device run (no bundle reached the ANE).

Toolchain: coreai-torch 0.4.2, transformers 5.12.1, coreai-build 3600.83.1 (Xcode 27.0 RC), macOS 27.0.
Checkpoint `Qwen/Qwen3.5-0.8B` (24 layers: 18 gated-delta linear + 6 full attention, vocab 248320, tied).

## 1. The stock exporter refuses the family

`coreai.llm.export Qwen/Qwen3.5-0.8B --platform iOS --compression 4bit_weight_palettized_group32
--max-context-length 4096 --experimental --compute-precision float16` stops in 2.6 s with
`ValueError: Model 'qwen3_5' does not support iOS variant`: the registry entry `qwen3_5_text` carries a
`macos_class` only (`models/registry.py`), and `export/pipeline.py` requires an `ios_class` for
`--platform iOS`. Apple's `models/ios/` builders remain mistral / olmo2 / qwen2 / qwen3 (+ the overlay's
phi3 / smollm3); none is a hybrid.

## 2. The community builder is not on Apple's contract

The overlay's `models/ios/qwen3_5_ios.py` (June 2026, the source of the `ios-gpu/` static bundles on
[mlboydaisuke/qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI)) is a
**single `main` graph, query length 1 only**: `forward(input_ids, position_ids, in_step, causal_mask,
key_cache, value_cache, conv_state, rec_state) → logits`, exported by `ondevice/export_qwen3_5_ios_fast.py`
(not the CLI). It has no `load_embeddings` / `gather_embeddings` / `extend` / `prompt_opt` split, no
query buckets {8, 16, 64}, no `export_static_shape_configs` / `export_hardware_constraints`, and Apple's
`StaticShapeEngine` binds exactly two states (`key_cache`, `value_cache`) — the SSM `conv_state` /
`rec_state` would be left unbound. Under coreai-torch 0.4.2 it still exports: 15 s, 1506 MB fp16.

## 3. AOT to the Neural Engine: 0 regions in every arm

`xcrun coreai-build compile <x>.aimodel --platform iOS --preferred-compute neural-engine --architecture h18p`,
regions counted with `find <x>.aimodelc -name '*ANE_region*' | wc -l` (0 = the silent GPU fallback,
`coreai-error-index.md` § "ANECCompileOffline() failed").

| arm (all fp16 weights, ctx 256, q=1) | AOT | ANE regions | ANEC failures | `Incompatible element type` warnings |
|---|---|---|---|---|
| A — builder as-is, conv/rec as plain I/O | exit 0, 6 s, 2.3 GB GPU delegate | **0** | 1 (`aneCompileStatus=1`, empty ErrorList) | 42 |
| B — conv/rec as Core AI states (`--state`) | exit 0, 6 s | **0** | 1 | 12 |
| C — 6 layers only (`--layers 6`) | exit 0, 3 s, 966 MB | **0** | 1 | 30 |
| D — `ANE_DIAG_TRIVIAL=1` (recurrence, per-head SDPA and RoPE removed; projections, state writes, norms, lm_head kept) | exit 0, 6 s | **0** | 1 | 0 (only "Reached maxReorderingDistance 50" ×7 at residual adds) |
| E — control: Apple's stock `Qwen/Qwen3-0.6B --platform iOS --compression none --max-context-length 1024` (uncompressed fp16 weights, dense) | exit 0, 1 min 44 s, 1.0 GB | **19 / 19** | 0 | 0 |

The compiler's own words (arms A–C, `warning:` lines with debug-info source locations):

```
"ane_validation_message"("Incompatible element type for ANE: expected fp16, f8E4M3, si8, ui8, si16, or ui16")
… failed: ANE I/O op isn't a squeeze / expandDims
```

Op ids are `_to_copy_*` (dtype casts), `unsqueeze_*` and `add_*`; the source lines are
`qwen3_5_ios.py:290` — the call into `_gated_delta_step`, which casts q/k/v/β/g and the recurrent state
to fp32 (`models/macos/qwen3_5.py:199`) — and the residual adds `:330-333`. The builder's local fp32
norms and its in-graph fp32 RoPE are never cited.

Reading, one variable at a time:

- A = B: whether conv/rec are states or I/O does not matter.
- A = C: not an ANEC capacity limit (the June "chunk into 6 layers and ANECompile passes" was the
  on-device JIT on a beta; the release AOT compiler rejects 6 layers for the same reason as 24).
- D: with the fp32 recurrence gone the element-type complaints vanish, and the ANE compile **still**
  fails with an empty error list — the same non-diagnostic signature as linear-INT4
  ([apple/coreai-models#205](https://github.com/apple/coreai-models/issues/205)). So the fp32 recurrence
  is one blocker and there is at least one more the compiler does not name (candidates, untested: the
  248320×1024 tied head as an in-graph constant in a single-`main` layout, the inline fp32 norm chains,
  the `mutable_slice_update` state writes).
- Apple's dense static graphs are fp16 end to end (`primitives/ios/rms_norm.py` casts to fp32 only
  under `USE_HF_IMPL`), which is why they land 31/31 regions palettized.

This is the compiler-side face of the June 2026 device result (`ondevice/_qwen_ios_fast_RESULTS.md`,
beta OS, on-device JIT): an fp16-only gated-delta recurrence produced multilingual garbage on the ANE
while the same chunk answered " Paris." on the GPU, and the fp32 version could not be compiled for the
ANE. The release toolchain confirms the second half from the other side: fp32 is not an ANE element type.

## 4. Control (fp16 weights, Apple dense builder)

Apple's own `Qwen3ForCausalLMForiOS` exported uncompressed (`--compression none`, fp16 weights,
contexts {256, 512, 1024} × queries {8, 16, 64}) compiles to **19 ANE regions out of 19 static
functions** with zero ANEC failures and zero warnings, 1 min 44 s. So uncompressed fp16 weights are not
what keeps a graph off the ANE (the palettization requirement in `coreai-error-index.md` is specific
to linear INT4), and the 0-region result above is the hybrid graph's own.

## 5. What a port would have to add (facts, not a plan)

registry `ios_class`; the four-entrypoint contract with four states and query buckets (a chunked
gated-delta for q>1 exists in `models/macos/qwen3_5.py` but is fp32 too); an fp16-only recurrence that
keeps numerics (unverified on the release OS; it failed on the beta); a Swift static engine that binds
`conv_state`/`rec_state` (Apple main does not); palettization wired into the trace; and the unnamed
second blocker of arm D. Until the fp16-recurrence question is answered (cheapest instrument: the Mac
GPU, fp32 vs fp16 recurrence under the 24/24 oracle gate — no device needed), the family stays on the
GPU paths in `pipelined-engine.md` (iPhone 17 Pro 69.7–74.0 tok/s decode-only pipelined, 42.5–45.4
static fused-int8).

Evidence: `~/code/coreai/coreai-models/exports/qwen3_5_0_8b_ios_fast_ct042/` (`RESULTS.md`, probe
script, `aot_*.log`), `…/exports/qwen3_0_6b_ios_fp16_control/`, handoff
`2026-09-15-ane-s3-qwen35-static.md` (local, not published).
