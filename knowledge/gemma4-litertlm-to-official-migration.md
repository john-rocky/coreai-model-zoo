# Gemma-4 E2B mixed-bit QAT — migration from `.litertlm` extraction to the official HF checkpoint

**Status: source migrated + proven bit-exact (2026-07-17). Provenance switched to official.
Pack regeneration + external push are the remaining mechanical steps.**

Companion docs:
`knowledge/gemma4-mixedbit-qat-transplant.md` ("P6 DONE" = the full test log),
`knowledge/gemma4-raw-metal-port.md` (the ship engine), `../../GEMMA4_OFFICIAL_SHIP_PREP.md`
(ship checklist + open decisions).

## Why this migration exists

The Gemma-4-E2B mobile mixed-bit QAT weights (int2/int4/int8, per-layer-embedding tables) are
what make the raw-Metal engine hit LiteRT-LM parity on iPhone. Until now we obtained them by
**reverse-engineering Google's LiteRT binary**: download `litert-community/gemma-4-E2B-it-litert-lm`,
crack the `.litertlm` container with `litert_lm_builder.litertlm_peek`, and pull the quantized
tensors out of the embedded TFLite subgraphs (`litertlm-convert/scripts/extract_gemma4_mixedbit.py`).
It worked and was bit-exact, but shipping that recipe meant publishing "how to crack Google's
binary" — awkward provenance for a public port.

**2026-07-15 Google published the same weights as a plain 🤗 Transformers checkpoint:**
`google/gemma-4-E2B-it-qat-mobile-transformers` (Apache-2.0, `quant_method: "gemma"`, the wNa8o8
mobile schema). This is the same QAT run, released in a standard, redistributable format — so the
extraction can be replaced by a normal `safe_open` read of an official checkpoint.

## What we proved before switching (so the swap carries zero risk)

The official checkpoint is **bit-exact identical** to our validated `.litertlm` extract. Evidence
(scripts under `litertlm-convert/scripts/`, full log in the transplant doc's "P6 DONE" section):

| check | result |
|---|---|
| 13 weight families, official codes vs our extract (dequantized) | `max\|Δ\|=0.0`, code-exact=1.0, cos=+1.0 — **every value identical** |
| converter output vs the litertlm extract, packed bytes | **312/313 tensors byte-identical** |
| the 1 differing tensor, `per_layer_model_projection` | official ships it BF16 (higher precision) vs our int8; after the pipeline's int8 requant the delta is ~1 int8 LSB (`max\|Δ\|≈5e-5` on 0.006-magnitude weights) — **greedy-lossless** |
| norm-gate: MLX fp16, official bf16 norms vs our fp32 norms, 3 prompts | greedy **3/3 full match** (40/40, 8/8, 35/35) |
| **raw-Metal P1 token gate from the OFFICIAL extract** (the ship engine's own greedy gate) | **3/3 PASS, EXACT==fp16** |

Packing detail worth knowing: the official checkpoint stores int4/int2 as **unsigned code minus
midpoint** (int4 −8, int2 −2); our kernels use two's-complement. Different bytes, identical values.
int8 (PLE gate/proj) is genuinely signed. Norms are used directly (no `(1+w)` shift). The official
checkpoint also carries `input/output_activation_scale` and `k/v_cache_scale` (the wNa8o8 int8
activation path) — **unused** by our fp16-activation engine.

## What changed in the pipeline

**Old:** `litert-community/gemma-4-E2B-it-litert-lm` → `litertlm_peek` → `extract_gemma4_mixedbit.py`
→ `gemma4e2b_mixedbit_weights.safetensors` + manifest + fp32 norms (reverse-attributed from the
anonymized graph) + `final_norm.f32.npy`.

**New:** `google/gemma-4-E2B-it-qat-mobile-transformers` → `official_extract.py` → the **same four
artifacts**, byte-compatible with the pack pipeline. Norms come straight from the checkpoint's
plain bf16 tensors — this also **eliminates the fragile fp32-norm graph-attribution step** (the
old path reverse-derived norms by consumer-scope attribution across 976 anonymized subgraphs).

New/changed files:
- `litertlm-convert/scripts/gemma4_official_to_mixedbit_extract.py` — the converter (canonical),
  also copied to `conversion/gemma4_raw_metal/official_extract.py`.
- `litertlm-convert/scripts/gemma4_official_vs_litertlm_equiv.py` — the equivalence test
  (copy: `conversion/gemma4_raw_metal/official_equiv_check.py`).
- `litertlm-convert/scripts/gemma4_official_norm_gate.sh` — the norm greedy gate.
- `conversion/gemma4_raw_metal/p0_sdpa_parity.py` — `EXTRACT` is now `GEMMA4_EXTRACT`-overridable
  (default still the litertlm dir). Set `GEMMA4_EXTRACT=…/gemma4e2b_extract_official` to source
  the whole pack pipeline (P0/P1/P2) from official.
- Built extract: `litertlm-convert/out/gemma4e2b_extract_official` (2.0 GB).

Reproduce the ship-engine gate from official:
```
cd coreai-models-community/conversion/gemma4_raw_metal
GEMMA4_EXTRACT=/Users/USER/code/litertlm-convert/out/gemma4e2b_extract_official \
  /Users/USER/code/litertlm-convert/.venv/bin/python p1_chain.py --gate
```

## Provenance decision (2026-07-17)

**Drop the "extracted from LiteRT-LM / `.litertlm`" narrative from all ship-facing text; the source
is now the official `google/gemma-4-E2B-it-qat-mobile-transformers` checkpoint.** The weights are
still Google's official Gemma-4 mobile QAT release (credit to Google DeepMind stays); only the
container and the extraction method change. Files updated: the raw-Metal conversion README, the
staging README + HF model card drafts, `gemma4-raw-metal-port.md`, the community README raw-Metal
row, and both `Gemma4MetalEngine.swift` header comments.

## What is left (mechanical)

1. **Regenerate the shipped pack from official** so the binary matches the new provenance. The
   existing `gemma4_pack.bin` is byte-≈identical (only `model_proj` ~1 LSB), but for a truly
   official-sourced artifact run `GEMMA4_EXTRACT=…_official` through P2/P2b. **Blocker/decision:**
   the official main checkpoint has no MTP drafter (that's a separate repo,
   `google/gemma-4-E2B-it-assistant`, non-quantized bf16). Since MTP is A19-negative and the engine
   ships S=1, the clean move is to regenerate WITHOUT the drafter (needs a `--no-drafter` path in
   `p2_export_pack.py`, which today requires `RawLoopMTP`). See `GEMMA4_OFFICIAL_SHIP_PREP.md` D1/D2.
2. External push: GitHub `john-rocky/gemma4-metal` → HF
   `mlboydaisuke/gemma-4-E2B-metal` → kit/zoo commits.

## One-line summary

Gemma-4-E2B mixed-bit QAT is now sourced from the official HF checkpoint
`google/gemma-4-E2B-it-qat-mobile-transformers` (bit-exact vs the old `.litertlm` extraction,
raw-Metal P1 gate 3/3 from official); use `official_extract.py`, not the `.litertlm` peeker.
