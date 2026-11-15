# Spec-decode C1 — DONE (2026-07-03): the loop now beats shipped 15.9 tok/s

> Full detail: `spec-decode-hybrid-verify-design.md`.
> Code: `apps/CoreAIChatMac/Sources/SpecDecodeEngine.swift` (uncommitted, SHIP-GATE).

## RESULT

Qwen3.6-27B ⚡Spec decode = **~18 tok/s** (cool, single response) vs shipped 15.9 tok/s
decode = **1.12×**. LOSSLESS proven byte-identical (spec on == off, 870/870 bytes, coherent
code). α 2.62, tok/fwd 3.37 on a code prompt. Reproducible across two clean runs
(17.79 / 18.26 tok/s, 256 tokens in ~14.0–14.4 s).

Chosen shape = **(b)** from the original plan (route the validated outer loop's forwards
through owned buffers), NOT (a) engine-internal — the GDN snapshot/tail/re-anchor rollback
discipline already lives correct-and-lossless in `WindowedModel`, so only the forward
mechanism changed.

## WHAT ACTUALLY MOVED THE NEEDLE (the handoff's premise was wrong)

The old handoff blamed the 10.9 tok/s on "raw MPSGraph JIT forward 135 ms vs engine 63 ms"
and projected 25–30 tok/s from engine-grade forwards. Measurement says otherwise:

1. **The real bottleneck was the CPU logits flatten, not the forward.** The old code did
   `flattenAsFloat` = `[1,S,vocab]`→`[Float]` (2.2 M fp16→Float, +9 MB alloc) on EVERY forward
   (390×/gen), then argmaxed. That CPU pass was **~38 s of a 73 s decode (52%)**. Fixing it —
   argmax directly over the fp16 output buffer for the 1–7 rows actually needed
   (`WindowedModel.argmax(row:)`) — dropped decode **73 s → 14 s (3.5 → 18 tok/s)** by itself.
   THIS was the win. Owned-buffer `encode` was necessary plumbing but not the lever.
2. **Owned-buffer encode did NOT make the verify forward 63 ms.** Measured target verify GPU
   time (drain) = **105–108 ms median**, i.e. **c_v ≈ 1.67** vs the shipped S=1 decode's 63 ms —
   the S=9 dynamic-query GDN graph is inherently ~1.7× a decode step. The "63 ms / c_v≈1"
   projection came from the small iOS *dense* model and does not transfer to the 27B GDN
   hybrid on Mac. 25–30 tok/s is NOT reachable without a faster verify graph.
3. **Per-anchor specialization is NOT the dominant cost** (only 2 forwards >400 ms in a full
   gen). position_ids length varies per anchor (`forward_stateful_core` derives the KV write
   `offset = position_ids.shape[1] - query_len`, so length IS the window position — can't bind
   `[1,S]` to collapse shapes), but under `expectFrequentReshapes=true` the repeat-shape cost
   is small, not ~1 s. Ignore this lever.

## PER-FORWARD BREAKDOWN (256 tok, 14 s, M4 Max, cool)

| role   | n   | GPU drain (med / total) | encode/CPU-build (med / total) |
|--------|-----|-------------------------|--------------------------------|
| target | 76  | 105 ms / 7.9 s          | 30 ms / 2.6 s                  |
| draft  | 314 | 2.9 ms / 0.9 s          | 8.9 ms / 2.9 s                 |

Residual overhead = **enc (Core AI command-building) 5.5 s of 14 s (39%)**, inherent to the
sequential encode path (can't pipeline — each round needs the prior logits). Target drain
(7.9 s) is the model physics floor. Draft GPU is now trivial (0.9 s); draft's *enc* (2.9 s)
is its cost. Further speedups would need: (i) cheaper per-forward encode, or (ii) a faster
S=9 verify graph (close the 1.67× kernel gap — the real 27B "custom kernel" work), or (iii)
fewer draft forwards (K tuning under the NEW cost model — K=6 was tuned pre-flatten-fix).

## THERMAL NOTE

M4 Max throttles under sustained GPU load: a single response from cool = ~18 tok/s; ~6
back-to-back 40 GB runs drop it to ~8 tok/s (forwards ~2.2× slower, decode 30 s). For a
demo/recording, capture a single cool-start response. In-run spec-vs-baseline stays
favorable regardless (8.3 on vs 6.0 off in the throttled A/B).

## CODE CHANGES (SpecDecodeEngine.swift, uncommitted)

- `WindowedModel.forward()`: owned persistent MTLBuffers (ids/pos/logits/4 states) +
  `fn.encode(...to: computeStream)` + `currentWorkCompleted()` drain, replacing fresh-NDArray
  `fn.run()`. conv/rec snapshot/restore now raw `memcpy` on the owned buffers.
- `WindowedModel.argmax(row:)`: fp16-direct argmax over one logits row (no flatten). `peek`/
  `verifyRound` return the row `base`; caller argmaxes only needed rows. **This is the win.**
- `nonisolated(unsafe) let computeStream` — non-Sendable ComputeStream, `currentWorkCompleted()`
  suspends off @MainActor; access is serialized so it's safe.
- Bench instrumentation (all gated on `SPEC_BENCH` env, zero cost in production): `SPEC STATS`
  decode line, per-forward `FWD role=… enc=… drain=…`, `SPEC TEXT` dump, `SPEC_OFF=1` forces
  the no-draft baseline for the lossless A/B. `role` param on WindowedModel for labelling.

## PROFILE — where the verify-vs-decode gap lives (2026-07-03, SPEC_OFF target-only)

Per-forward on the 27B (M4 Max, warm, GPU drain + command-build):

| shape | per-forward | source |
|-------|-------------|--------|
| S=1 decode (shipped)  | 63 ms  | shipped 15.9 tok/s |
| S=9 verify (warm)     | ~134 ms (drain 104 + build 30) | measured |
| S=13 verify (warm)    | ~140 ms | measured |

**Slope S9→S13 = 1.6 ms/token — S-scaling is negligible.** So the 9-query attention (scales
with S) is NOT the cost. Extrapolated S=1 verify ≈ 121 ms vs decode 63 ms = a **~58 ms FIXED
gap**, independent of how many tokens are verified. That fixed gap = the verify graph's GDN
**chunk-inverse** formulation (`use_loopfree_chunk`, matrix/doubling inverse, 48 GDN layers)
vs decode's single-step recurrence (`use_loopfree_step`). Both bundles use the plain generic
engine (`export_to_coreai`, NO custom kernels) — decode is fast because it's S=1 + step-form
GDN, not because of special kernels (in fact the generic engine beats the hand qmv kernels 3.5×
here, per the decode export docstring). ⚠️ The S=13 *cold-cache* first run mis-measured (drain
6 ms / 242 ms total) — the enc/drain split is unreliable when encode runs synchronously; use
the WARM total-per-forward (140 ms).

**=> Custom kernels CAN close most of the gap — specifically the existing GDN chunk kernel,
not new matmul kernels.** `models/macos/qwen3_5_gdn_metal.py` already runs the sequential
gated-delta recurrence (decode-exact math, no matrix inverse) for a whole chunk in one dispatch
— "flat to S=32 (~20 ms)". Wiring it into the verify export (switch `export_to_coreai` →
`export_to_coreai_with_kernels`) should drop verify toward decode. Upper bound: recover the
~58 ms → verify ~70–75 ms ≈ decode → spec-decode **~24–28 tok/s (MLX-class)**. Second, independent
lever: verify is S-flat, so verifying MORE tokens/forward is ~free → raising acceptance α
(EAGLE-3 draft, C3) multiplies tokens/forward directly (α 2.6→5 ≈ doubles). DECISIVE TEST =
re-export S=9 verify WITH the GDN kernel, re-prove lossless (the kernel changes numerics:
sequential vs chunk-inverse), measure. Heavy (27B re-export + wiring) but the kernel exists.

### GDN-kernel attempt — DONE, BLOCKED BY OS BUILD (2026-07-03)

Wired it: added `--metal` to `export_qwen3_5_verify_pipelined.py` (additive, off by default;
mirrors `export_minicpmv46_chunked_prefill.py`'s `metalize_gdn_chunk` + `export_to_coreai_with_kernels`).
0.8B verify `--metal` **exports cleanly** (`[metal] GDN fp32 sequential-scan kernel on 18 linear
layers`). But it **fails to RUN on this Mac OS build** (macOS 27.0, runtime 26A5353q):

```
GPUCustomMetalKernelOps.mm:77: failed assertion 'Failed to create MTLTensor from NDArray buffer:
Tensor Descriptor Validation' — [tensor.strides extentAtDimensionIndex:0] (16) should be 1
```

The kernel forward already `.contiguous()`s every input, so this is the RUNTIME NDArray→MTLTensor
bridge (Metal 4) rejecting the kernel op's tensor layout. ⚠️ CORRECTION (was "OS wall"): this is NOT
a blanket OS block — other custom kernels (the gemma4 gather/head-argmax) DO run on this build. It is
a **documented Apple Core AI beta known-issue**, confirmed against the iOS/macOS 27 beta1 release notes:

- **178056451** "Models with custom Metal kernels will fail to load." ← our GDN kernel (its reversed-axis
  DSL layout `torch[h,S,dk]→DSL[dk,S,h]` trips the buggy MTLTensor stride validation; simpler kernels
  like the gather don't). ROOT CAUSE pinned: the failing `strides[0]=16 should be 1` — the **16 is
  `linear_num_key_heads=16`** (the GDN head count); the reversed-3D-axis binding puts h at a stride the
  buggy Metal-4 validator rejects. gather works because it's a 1D/2D matvec ([1,N]), no reversed 3D axes.
  A beta1 layout workaround was ruled out (2026-07-08): it'd mean rewriting the kernel's axis convention
  against a validator that is itself the bug — not tractable/worth it. Needs the 178056451 fix (beta2).
- **177729331** "Ahead-of-time (AOT) compilation might fail unexpectedly for certain models." ← our AOT
  wall (`invalidCompiledModel`), same root: a beta framework bug, not our recipe.
- **177354777** "Inference might fail/crash for control-flow over dynamic-shape tensors (e.g. Qwen3.5/3.6)."
  ← why the verify graph avoids the GDN while_loop in the first place.
- **175789258** encode blocks until compute done UNLESS specialized preferred-compute GPU (we set `.gpu`).
- **177991751** Metal-API-Validation → CoreAI fail-to-execute — reportedly FIXED in beta2 (not our issue;
  we launch standalone, validation off).

So BOTH accelerator paths (AOT 177729331, custom kernels 178056451) are blocked by **documented beta
bugs on build 26A5353q**, not by our code or a fundamental limit. current build = beta1 (macOS 27.0
26A5353q / Xcode 27A5194q). **beta2 (26A5368g) is out** — user reports the fixes are published; could
not independently confirm the beta2 *resolved* list (Apple dev-docs is a JS SPA; releasebot only carries
beta1). NEXT after a beta2 update: **retest AOT (177729331) FIRST** — it's the cleaner, complete fix
(precompiles the WHOLE verify graph → verify≈decode → ~25-28 tok/s, no per-op wiring). GDN custom kernel
(178056451) is the fallback. Kept for retest: `--metal` flag in the export + bundle
`exports/qwen3_5_0_8b_verify_s9_int4lin_metal`. Did NOT run the 27B `--metal` export (same beta wall).

## STILL OPEN / NEXT

- Frozen config unchanged: target `qwen3_6_27b_verify_s9_int8hu_block32_sym` × draft
  `qwen3_5_0_8b_verify_s9_int4lin`, K=6, S=9 (both symlinked into
  `~/Library/Application Support/CoreAIChatMac/models`). JIT path (AOT still dead on this OS).
- ⛔ Outward (HF push / X / screen recording) = user-gated (SHIP-GATE). The number to sell is
  "lossless + faster than the shipped 27B", not the old 25–30 fantasy.
- If more speed wanted: re-tune K under the new cost model, or attack the 1.67× verify-graph
  kernel gap (heavier, export-level).

Conventions: GPU-SOLO `_GPU_LOCK`; no "claude" in commits; UI/comments English; disk ≥120 G
before 27B runs; build = xcodebuild(beta) Release `-derivedDataPath build`, launch binary
with `CHATMAC_MODEL`/`SPEC_BENCH` env hooks.
