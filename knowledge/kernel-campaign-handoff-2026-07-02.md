# Kernel campaign — state (2026-07-02)

> One-page state + next steps. Detail: `dense-int4km-flagship-
> session-findings.md`, `spec-decode-design.md`, `flagship-full-tuning-stack.md`, `tensorops-quantized-
> kernels.md`.

## Net result of this session
- ✅ **Dense-path int4km lever** (lm_head + attn q/o, on top of gather_qmm): **LFM-8B on-device 1.23–1.43×
  + quality PASS**; **flagship Qwen3.6-35B decode 2.18×** (Mac GPU, ratio-valid). Export variants +
  findings committed.
- ⛔ **BUT 2.18× is a 4-bit speed win WITH a quality cost.** Flagship lm_head int4 flips **104/512** vs
  int8's **32**. **fp4 does NOT rescue it** — direct 35B measurement: fp4(e8m0)=int4=**104**, fp4(fp16-scale)
  **117** (worse). The LFM-1.2B "fp4≈int8" did NOT transfer to the 35B. **fp4 = no-op** (kernel built,
  no quality edge). ⇒ no post-hoc 4-bit gives 2.18× at int8 quality.
- ✅ **tree-attention verify kernel** built + GPU-validated (cos 1.000000) — the machinery for the LOSSLESS
  lever (spec-decode tree verify). `_specdecode_proto/tree_attn_verify.py` (e591499).
- Corrections made: prefill/FlashAttention is A19-refuted but **OPEN on Mac** (M4 SDPA ~22% peak = headroom,
  no custom Mac prefill kernel run yet); gather-family read-reduction is **largely DONE** (gather_qmm +
  BatchedMetalSwitchGLU grouped-GEMM + per-slot KV-write).

## Honest strategic picture
Quality-safe *dramatic* flagship speedup does NOT come from 4-bit (int4/fp4 both degrade, capped). It comes from:
1. **spec-decode (LOSSLESS, ~3×)** — the real quality-safe lever. Verify K tokens/forward, distribution-exact.
2. **QAT-int4** — the only quality-safe route to the 4-bit *bandwidth* win (train to recover the 104→~32 flip gap).
3. int8-dense bandwidth ≈ ~1.6× (but experts-int4 also degrades → partial).

## NEXT per lever
- **Stream C (spec-decode) — HIGHEST value (lossless ~3×, quality-safe).** Tree-attn verify kernel READY
  (`_specdecode_proto/tree_attn_verify.py`). Next: (a) train an EAGLE-3 head for Qwen3.6 (Red Hat
  `Speculators`, external GPU, ~1–2 days) or start with vanilla-draft (qwen3.5-0.8B, no training); (b) wire
  draft→verify→accept/rollback in the pipelined engine; (c) use the tree-attn kernel for the tree verify.
  Design + feasibility: `spec-decode-design.md`. (Note: prompt-lookup spec-decode already shipped in
  CoreAIChat; this is the tree/EAGLE-3 upgrade.)
- **Stream D (quant) — re-scoped fp4 → QAT.** fp4 disproven (§8b of findings doc). Next: **QAT-int4** on the
  flagship dense path (OS26-shippable) to recover flagship 4-bit quality. fp4 kernel exists
  (`gemma4_metal_mlp_fp4.py`) if bandwidth-only ever wanted. Alt: **mixed int4+int8-outlier matvec** (bridge
  the int4 cliff at ~int4 BW without QAT — un-built, extend `MetalInt4KMLinear`).
- **Mac prefill FlashAttention — OPEN.** Scaffold `_tensorops_proto/m4_speed_ab.py` (matmul2d
  vs MPSGraph matmul on M4). ⚠️ hit `Program load failure` under Stream-D GPU contention + may need an
  `asset.executable()` API fix — run on a CLEAN Mac-GPU window. If matrix path beats default → build fused
  FlashAttention, A/B vs `ondevice/_prefill_sdpa_baseline.py`.
- **KV-quant in decode-attn (#3)** — un-built, quality-safe decode-BW for long-ctx (int8 KV dequant in the
  flash-decode SDPA). General, moderate.
- **flagship absolute tok/s** — the 2.18× is ratio-valid; absolute 2.79/6.08 are Mac-driver-overhead-inflated
  (~10×). Real numbers need the proper engine (35B jetsams the phone; fix `_qwen36_mac_bench.py` driver or
  use a fitting model).

## Coordination / method notes
- **GPU-SOLO via `_GPU_LOCK`** (Mac GPU shared with Stream D
  fp4/QAT — serialize; A19 shared with Stream B TensorOps/LLaDA — serialize).
- Mac-GPU method for raw decode bundles (macOS 27): `from_preferred_compute_unit_kind(gpu())`; **ANECompile
  errors are NON-FATAL (GPU fallback)** — don't kill on them. Phone RAM ceiling ≈ 5–6 GB (int4-8B) → 35B jetsams.
- **HF downloads: `HF_HUB_DISABLE_XET=1`** (hf_xet stalls; not a rate-limit — 5-min window, never hit).
- Stream B: matmul2d refuted on A19 (prefill); FP4-via-matmul2d irrelevant for decode (BW = matvec).
- No coreml bundles / build files in commits.

## Artifacts (committed, coreai-models-community, committer john-rocky)
bab5fa7 LFM #2 export · 200cb2b flagship plan · 3eb4a5f spec-decode design · 5b790f5 Qwen3.6 export+findings ·
69c65ea fp4-framing fix · 6436d8a prefill-Mac-open fix + m4 scaffold · e591499 tree-attn verify kernel.
(§8/§8b fp4-disproven added by Stream D.) Uncommitted by design: coreai-models `macos/` arsenal
(Apple clone — incl. the `MetalInt4KMLinear.weight` fix + `gemma4_metal_mlp_fp4.py`), `ondevice/` scripts
(non-git: `_qwen36_mac_bench.py`, `_dense_int4km_microbench.py`, `_flagship_dense_coverage_audit.py`,
`_prefill_sdpa_baseline.py`), coreml bundles in `exports/`.
