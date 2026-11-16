# Flagship dramatic-speedup plan — the full stackable tuning set (2026-07-01)

> Target: **Qwen3.6-35B-A3B** (primary) and **GLM-4.7-Flash** (MLA sibling). Goal: stack EVERY
> applicable lever so the gains **multiply**, not just add. Grounded in this session's on-device
> measurements (`_flagship_dense_coverage_audit.py`, LFM #2 A/B) + the accel-levers survey
> ([`accel-levers-survey-and-plan.md`](accel-levers-survey-and-plan.md)). All ⚗️ numbers are targets/estimates unless marked MEASURED.

## The core idea — three orthogonal axes that MULTIPLY

Decode tok/s ≈ **1 / (per-forward bytes)** × **1 / (forwards per token)**. The two factors are
independent, so byte-reduction and forward-reduction **compound**. Prefill (TTFT) is a third,
compute-bound axis. Attack all three:

```
  decode speedup ≈ (Axis-1 byte cut) × (Axis-2 spec-decode)      [× kernel micro-opts]
  prefill speedup ≈ (Axis-3 TensorOps matmul/flash)
```

## Baseline byte model (Qwen3.6-35B-A3B, per-token decode read, MEASURED via config audit)

Current shipped recipe = experts on `gather_qmm` **int8**, dense path **int8**:

| bucket | int8 now | notes |
|---|---:|---|
| routed experts (gather_qmm, top-8/256) | 1007 MB | biggest single chunk |
| attn q/o (k/v small) | 755 MB | dense, un-kernelized fused |
| lm_head (vocab 248k × 2048) | 509 MB | single biggest matvec |
| shared expert | 126 MB | always-on |
| router (fp16) | 42 MB | stays |
| **TOTAL** | **~2438 MB/tok** | ← BW-bound decode floor |

GLM-4.7-Flash = ~3586 MB/tok (MLA attn 1023 + experts 1736 + shared 434 + lm_head 317) — cross-checked
to its known 3.58 GB/tok.

---

## Axis 1 — per-forward BYTE reduction (weight-bound decode)

| # | lever | bytes | est. | status |
|---|---|---|---|---|
| 1a | **dense-path int4km** (lm_head + attn q/o + shared → int4) | −695 MB | ~1.40× | ✅ **MEASURED on LFM-8B: 1.23× sustained / 1.43× avg, quality PASS** (33/48 greedy match, correct) — same kernel, config-general |
| 1b | **experts int8→int4km** (gather_qmm km4) | 1007→504 = −503 MB | stacks to ~1.97× | int4-experts benched COHERENT on LFM; needs a 35B multi-token reasoning gate |
| 1c | **FP4 (E2M1) via TensorOps** (experts+dense, OS27/A19) | 4-bit @ near-FP8 quality | quality-safe 4-bit | the answer to the int4-RTN cliff; native `matmul2d` dequant on A19. Depends on Stream B TensorOps path |
| 1d | **KV-quant in the decode-attn kernel** (int8/int4 KV, on-the-fly dequant) | cuts KV BW at ctx | long-ctx | general across all models; kernel work |

**Axis-1 stacked (1a+1b, int4 everything)**: 2438 → **~1240 MB/tok ≈ 1.97× decode from bytes alone.**
FP4 (1c) makes that quality-safe on OS27; KV-quant (1d) protects it at long ctx.

## Axis 2 — forward-COUNT reduction: speculative decoding (the bandwidth-wall breaker)

The ONLY lever that beats the decode bandwidth wall — verify K tokens per forward. **Multiplies on top
of Axis 1** (Axis-1 cuts per-forward cost; spec-decode cuts forward count).

| # | lever | est. | status |
|---|---|---|---|
| 2a | **n-gram / prompt-lookup** (training-free) | 2–4× on code/RAG/structured | zero-cost first win, no training |
| 2b | **vanilla draft** = shipped qwen3.5-0.8B / Qwen3-0.6B | ~2× | no training |
| 2c | **EAGLE-3 head** (train via Red Hat Speculators, Qwen3 1.7B–235B supported) | **3–5×** (accept 0.80–0.88) | ~1–2 days / 8×GPU training |

**Gating prereq (do FIRST)**: prove the pipelined engine can do a **verify-forward (S=K batch)** +
draft→verify→rollback wiring. This is new ENGINE work (Swift), not a kernel — the single highest-leverage
build in the whole plan. Prove with n-gram + vanilla draft (no training) before EAGLE-3.

## Axis 3 — prefill / TTFT (compute-bound, matrix-unit)

Prefill is matrix-MATRIX = compute-bound. **MEASURED this session**: MPSGraph prefill SDPA plateaus at
**~22% of the M4 fp16 ceiling** (clean S² scaling, no crash) — big headroom, but scalar-MSL kernels can't
beat a matrix-tuned baseline (the documented 16% loss). The win needs **matrix units**:

| # | lever | est. | status |
|---|---|---|---|
| 3a | **TensorOps `matmul2d`** (native int4/int8/fp4 dequant) on prefill projections+FFN | 3.3–4.06× (Apple MLX/M5) | Stream B, M5/A19 + OS27 |
| 3b | **TensorOps FlashAttention** via cooperative tensors | prefill attn | Stream B |

## Axis 4 — kernel micro-opts (stack on all of the above)

- Fused RoPE + RMSNorm + SwiGLU (fewer launches / intermediate BW).
- **Absorbed-MLA cross-head staging** (GLM-4.7 only): 1.12×@4K MEASURED, long-ctx-only; revisit at 8K+.
- FlashDecoding-style seq-split occupancy on the decode-attn kernel (already in the arsenal).

---

## The combined target (why "dramatic")

- **Decode**: Axis-1 (~1.97× bytes) × Axis-2 (~3× spec-decode, conservative) ≈ **~5–6×**.
  Qwen3.6-27B dense 15.9 t/s → **~90+ t/s** target; 35B-A3B similarly.
- **Prefill/TTFT**: Axis-3 ~**3–4×**.
- **Quality preserved**: FP4/QAT (1c) holds 4-bit quality; spec-decode (Axis 2) is **lossless** (verify).

## Sequencing (impact × de-risk), with gates

1. **Axis-1a+1b now (proven kernels, Mac-GPU)** — flagship dense-int4 + experts-int4, one export, on-device
   A/B + multi-token reasoning gate. Immediate ~2×. *(Blocked only on Mac disk for the 70 GB fp16 flagship,
   or a macOS-27 runtime wheel; on-device path proven on LFM this session.)*
2. **Axis-2 engine verify-forward (biggest new lever)** — prove n-gram + vanilla draft on the pipelined
   engine, then EAGLE-3. This is where the ~3× multiplier comes from.
3. **Axis-1c FP4 (OS27/A19)** — de-risks the int4 quality question; needs Stream B TensorOps.
4. **Axis-3 prefill TensorOps** (M5/A19) — TTFT.
5. **Axis-1d KV-quant** — long-ctx protection.

**Gate every quant lever on multi-token reasoning** (chat-formatted prompt token-match + PPL, not a single
token — the Nanbeige lesson; validated this session: LFM #2 dense-int4 PASSED with a real arithmetic prompt).

## Dependencies / conventions
- Axis-1 = Mac GPU (`_GPU_LOCK`, SOLO). Axis-2 EAGLE-3 training = external GPU box. Axis-1c/Axis-3 = A19
  device + OS27 (fp4/fp8) / OS26 (int4/int8). Serialize the single A19 across device runs.
- On-device decode bundles: raw `.aimodel` (main.mlirb) JITs on A19 for ≤8B decode-only graphs (MEASURED,
  engine ready ~47 s); needs **>15 GB device free** for the on-device compile scratch. Flagship 35B may
  need AOT (`coreai-build compile --platform iOS --architecture h18p`) — verify.
- Don't commit coreml bundles/build files.
```
