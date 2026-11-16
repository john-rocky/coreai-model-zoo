# Spec-decode on GDN hybrids (Qwen3.5/3.6 family) — static-S verify design (2026-07-02)

> Stream C, C2 (draft-model) continuation toward the **Qwen3.6-27B dense** target on
> CoreAIChatMac. Companion to [`spec-decode-design.md`](spec-decode-design.md) /
> [`spec-decode-c-feasibility-and-plan.md`](spec-decode-c-feasibility-and-plan.md).

## ⚠️ Premise correction (important)

The C2 target rationale said "Qwen3.6-27B dense = pure attention → rollback is a
counter rewind". **That is wrong.** The 27B is *dense* only in the no-MoE sense — it is
a **GDN hybrid**: 64 layers, `full_attention_interval 4` → **48 linear (GatedDeltaNet)
+ 16 full-attention** (see `QWEN36_27B_STATE.md`). Same family as the 0.8B/2B/35B. So:

1. **No dynamic-query verify graph.** The GDN `GatedDeltaUpdate` while_loop does not
   lower on the GPU delegate, and JIT dynamic-query dies `ANECompile → MTL4` anyway.
2. **conv/rec states do NOT roll back by an integer** — the KV-counter rollback that
   made pure-attention spec-decode trivial does not cover the SSM states.

Both problems are solved below **without touching the engine or the model overlay**.

## Design

### Verify graph = STATIC S=K+1 + chunked GDN scan

`conversion/export_qwen3_5_verify_pipelined.py`: `input_ids` **static [1, S]** (same
fixed-query class as the shipped decode-only bundles — the class that runs everywhere),
`position_ids`/KV-seq dynamic (unchanged runtime contract), every linear layer routed
through `use_loopfree_chunk` (`_gated_delta_chunk`). Quantization configs mirror the
decode script, so a verify bundle is **weight-identical** to its shipped decode sibling.

- The known fp16 chunked-inverse NaN is a **chunk≥64** problem; verify chunks are ≤17.
  Gated at S∈{2,9,17}: `_smoke/test_qwen35_verify_chunk_parity.py` — chunk == step-looped
  (cos 1.000000, no NaN) and 0.8B model-level verify-forward == sequential decode
  (argmax 9/9, end-state parity). (`use_metal_chunk` fp32 kernel exists as a fallback;
  not needed.)
- **SDPA must NOT be externalized** in this export: its lower-right causal mask emits a
  `k_len ≥ S` guard that the externalize pipeline's auto-Dim (min=2) violates at static
  S>1 (`ConstraintViolationError d_73`). Decomposed in-graph SDPA builds the identical
  mask from plain ops; at S≤17 unfused attention cost is noise vs the weight read.
- All-position logits `[1, S, vocab]` are native (no `last_token_only`).

### Hybrid state discipline: snapshot + tail + exact-S re-anchor

KV rolls back for free (stale rows are always rewritten by a later forward before any
read — position `i` is only attended after a forward re-writes it). conv/rec cannot.
Rule: **device state only ever holds a fully-committed prefix.**

- Keep a host-side snapshot (conv/rec copy) at committed position `m`; the committed
  tokens not yet in device state are the **tail** (`m + |tail| = n` = committed length).
- Verify round (window S): feed `tail + [a0] + drafts(+pad)` — exactly S tokens from
  offset `m`. a0 = the pending greedy token from the previous round (fused scheme:
  always correct, committed immediately). Accept the longest draft prefix; the row
  after the last accepted token is the next round's a0 (correction/bonus unified).
- **State commit rule:** if every fed token ended up committed (all drafts accepted,
  no pad) → the post-forward state is valid: snapshot it, `m += S` (**all-accept
  rounds commit for free**). Otherwise restore the snapshot (tail rides along again).
- Tail grows by (1 + j) per non-committing round, bounded by S (j ≤ K − |tail|);
  when |tail| = S, one **re-anchor forward** (all-committed feed) moves `m` by S.
  Amortized tax ≈ (1+j̄)/S extra forwards — and its last row refreshes a0 for free.
- **Draft model** (same-vocab small hybrid, S=1 decode bundle): snapshot before each
  speculative burst; on the next propose, restore + replay committed tokens (S=1
  catch-up steps — cheap because the draft is cheap, ~3% of a 27B step each).

Prefill uses the same S-window (full chunks commit; the remainder is the initial tail).
One extra probe forward bootstraps a0 when the prefill remainder leaves it unknown.

Reference implementation (runs on Mac GPU today): `ondevice/_spec_mac_two_model.py`
(NgramDrafter + ModelDrafter), gate: `ondevice/_spec_verify_runtime_gate.py`.

## Validated so far (2026-07-02, M4 Max, macOS 27)

| gate | result |
|---|---|
| GDN chunk vs step, S=2/9/17 (eager fp16) | cos 1.000000, no NaN — PASS |
| 0.8B model verify-forward vs sequential (eager) | argmax 9/9 + end-state parity — PASS |
| 0.8B verify bundle export (int8lin, S=9) | exports/qwen3_5_0_8b_verify_s9_int8lin |
| Runtime: verify bundle S=9 vs decode bundle S=1, Mac GPU | **argmax 45/45 — PASS** (ANE errors non-fatal, GPU fallback) |
| Mac GPU c_v(9), 0.8B via python driver | 18.7 ms vs 10.6 ms = **1.77** (small-model + driver-overhead confounded; 27B is BW-bound → expect ≈1.0–1.2) |
| **Spec loop E2E, n-gram (0.8B)** | **LOSSLESS 64/64 PASS**, coherent code output |
| **Spec loop E2E, draft-model (0.8B draft == 0.8B target, free-form)** | **LOSSLESS 64/64 PASS; accepted/round 3.57 ≈ 100% of proposals; tokens/target-forward 4.00** (vs 0.97 greedy) — limiter is the S=9 window room, not acceptance |

### Two hard-won harness lessons (python driver, macOS 27)

1. **Respecialization is ~1.5–2 s per NEW position length** (with an ANECompile
   failure + GPU fallback each time), and hundreds of unique shapes also make a
   `MTL3On4CommandBuffer`-never-completes hang likely (an S=1-per-token harness
   hung reproducibly; the asyncio loop parks in kevent — completion never fires).
   **Fix = shape-stable design**: drive EVERYTHING (greedy ref, draft proposals,
   catch-up) through the S-window verify graph with the tail discipline, so a new
   shape appears only when the anchor m advances (once per S committed tokens).
   In the Swift engine `expectFrequentReshapes=true` covers this; in python it is
   the difference between working and hanging.
2. **Drafters must propose PAST the anchor a0** (the token the fused round commits
   first). Proposing continuations of C (not C+[a0]) off-by-ones every draft: the
   drafter re-proposes a0 itself → acceptance 0. With the fix, a perfect drafter
   accepts ~100%. (State-NDArray replacement between forwards is safe — repro'd
   clean in `ondevice/_spec_state_swap_repro.py`.)

## Economics (27B target, engine-grade)

Draft 0.8B int8 ≈ 0.9 GB/step vs 27B int8 ≈ 28 GB/step → r ≈ 0.03. With c_v ≈ 1.1–1.2,
K = 8, free-form accepted/round 1.4–1.9 (measured 1.7B→8B on device; 0.8B→27B TBD):
speedup ≈ (1+j̄)/(c_v + K·r + (1+j̄)/S) ≈ **1.6–1.9× free-form**; n-gram fuses in for
extractive/code (2–4×). Baseline 15.9 tok/s (shipped int8hu) → **~25–30 tok/s** target.

## MEASURED — 27B × 0.8B two-model run (Mac GPU, 2026-07-02)

Setup: target `qwen3_6_27b_verify_s9_int8hu_block32_sym` (S=9), draft
`qwen3_5_0_8b_verify_s9_int8lin`, K≤6, gen 48, `--reload-every 3` (reload the
target between shape buckets to dodge the macOS27 `MTL4CommandQueueErrorDomain`
death; a straight GPU run dies after ~12 tokens, CPU specialization won't load).

| prompt | α (accepted/round) | tok/target-fwd (greedy) | target fwds | draft fwds (med ms) | LOSSLESS | engine-eq speedup |
|---|---|---|---|---|---|---|
| free | 1.40 | 2.18 (0.96) | 22 | 74 (18) | 48/48 PASS | **~1.56×** |
| code | 3.55 | 2.40 (0.84) | 20 | 61 (23) | 48/48 PASS | **~1.89×** |
| rag  | 2.71 | 2.09 (0.84) | 23 | 76 (20) | 48/48 PASS | **~1.67×** |

- Engine-eq speedup = (target_fwds·135ms + draft_fwds·draft_ms) / greedy(fwds·135ms),
  using clean per-forward medians (134–137 ms target; reload-polluted medians in the
  code/rag spec phases excluded). Prediction band 1.6–1.9× free-form: **confirmed**.
- Measured r = draft/target per-forward ≈ 20/136 ≈ **0.15 in the python harness**
  (byte model said 0.03 — small-model fixed overhead dominates; Swift engine should
  land in between, so these speedups are a floor).
- Rejection tax is real on high-α prompts: forwards/round = 1.1 (free) but 1.8 (code)
  / 1.6 (rag) — snapshot-restore + exact-S re-anchor costs an extra forward on most
  rejected rounds. S=17 window + K tuning is the obvious next dial.
- Ops note: reload-every re-specialization writes tens of GB to
  `~/Library/Caches/coreai-cache`; the first attempt died on a FULL DISK (which also
  took down the CLI session). Keep ≳100 GB free for 27B reload runs.

## S/K SWEEP (2026-07-02 late) — window cost curve + optimal configs

15 runs total, S∈{9,13,17} × K∈{6,8,12,16} (S=13 at K=8 only), all **LOSSLESS 48/48
PASS** — the snapshot/tail/exact-S re-anchor discipline is solid at every window size.

**c_v(S) is NOT linear — compute cliff between S=13 and S=17** (27B int8hu, M4Max GPU,
greedy-ref medians): **135 ms @ S=9 → 143 ms @ S=13 (+6%) → 197 ms @ S=17 (+38%)**.
Growing the window is nearly free up to S=13, then hits a kernel/tiling cliff.

Acceptance does rise with the window (code acc/round 3.55 @ S9K6 → 6.71 @ S17K8 →
9.33 @ S17K16) but K>8 never pays: α saturates while draft forwards keep costing
~21-24 ms each (r≈0.15 in this harness).

Cost per token = (target_fwds × greedy-med + draft_fwds × draft-med)/48, baseline-free
config comparison (lower = better; S=1 greedy decode proxy ≈ 135 ms/tok):

| config | free | code | rag |
|---|---|---|---|
| S=9 K=6 | **89.6** | 86.3 | **97.3** |
| S=13 K=8 | 105.6 | **72.7** | 103.3 |
| S=17 K=6 | 112.4 | 92.3 | 116.5 |
| S=17 K=8 | 111.3 | 75.9 | 111.3 |
| S=17 K=12 | 124.8 | 80.3 | 122.3 |
| S=17 K=16 | 130.3 | 80.6 | 130.7 |

**Verdict:**
- **General/chat default = S=9, K=6** (~1.5× free, ~1.4× rag vs S=1 proxy). Bigger
  windows lose on low-α prompts: the extra verify cost + wasted draft forwards
  outweigh the acceptance gain.
- **Code/structured = S=13, K=8** (72.7 ms/tok ≈ **1.86×**, 16% faster than S9K6).
  S=17 is dominated by S=13 everywhere (the c_v cliff) — drop S=17.
- **Next bottleneck is the draft, not the window**: at 22 ms/forward the 0.8B draft
  is ~15% of a target forward in python (roofline says ~2 ms). In the Swift engine
  a ~7 ms draft would put free-form at ~73 ms/tok (**~1.85×**) with S=9 K=6 alone.
  Window tuning is done; the remaining upside is engine wiring + a better drafter
  (EAGLE-3, C3).
- Bundles on disk: `qwen3_6_27b_verify_s{9,13,17}_int8hu_block32_sym` +
  `qwen3_5_0_8b_verify_s{9,13,17}_int8lin`. Harness: `--ref-cache` added (greedy-ref
  is K-independent; cached per prompt+S).

## DRAFT SELECTION A/B (2026-07-02 night) — 0.8B-int4 wins; no training needed to wire

Question: is the 0.8B-int8 draft the best off-the-shelf choice? Constraint: the draft
must share the target's tokenizer (vocab 248320) — Qwen3.5-0.8B is the smallest such
model; Qwen3.5-2B is the next size. Draft quality only affects α (lossless is
verify-guaranteed), so quantizing the draft can only cost acceptance, not correctness.
6 runs (2 drafts × 3 prompts, S=9 K=6, same 27B target), all **LOSSLESS 48/48 PASS**:

| draft | free α / draft-med | code α / draft-med | rag α / draft-med |
|---|---|---|---|
| 0.8B int8lin (baseline) | 1.40 / 18 ms | 3.55 / 23 ms | 2.71 / 20 ms |
| **0.8B int4lin** | **1.53 / 16 ms** | **4.00 / 18 ms** | **2.86 / 18 ms** |
| 2B int8lin | 1.82 / 25 ms | 4.00 / 25 ms | 3.33 / 25 ms |

- **int4 holds α completely** (matches or edges int8 on all three prompts — within
  single-run noise, i.e. no measurable degradation) at half the bytes (0.45 GB).
  The int4-RTN quality collapse that matters for a *shipping* model is irrelevant
  for a draft: the 27B verify filters every token.
- **2B raises α as expected (free 1.40→1.82) but its per-forward cost eats the
  gain**: in-harness cost/tok — free 88.1 / code 84.4 / rag 94.8 vs int4's
  **83.3 / 77.9 / 95.6**. Swift-projection (target ≈63 ms engine decode, draft
  ≈4 ms int4 vs ≈7 ms 2B) keeps int4-0.8B ahead everywhere:
  **free ~1.9× / code ~2.1× / rag ~1.7×**. 2B also costs 2.5 GB resident next to a
  28 GB target. → 2B dropped.
- **Training verdict: NOT needed for wiring.** Lossless never depends on the draft;
  the only training-worthy lever is **EAGLE-3 (C3)** to lift free-form α beyond the
  off-the-shelf ceiling (~1.5-1.8 accepted/round) — separate work, does not
  block the engine wiring. Cheaper fine-tunes (distilling 0.8B toward 27B outputs,
  layer-truncated drafts) are dominated by EAGLE-3 and not worth a session.
- **Wiring config frozen: target 27B int8hu S=9 (+S=13 code-mode option) ×
  draft 0.8B int4lin** (`exports/qwen3_5_0_8b_verify_s9_int4lin`).

Ops addendum: the mid-run disk-full class is TRANSIENT — in-run respecialization temp
(mpsgraph under /var/folders) that APFS reclaims minutes after process exit; df lags
the reclaim, so disk guards must retry (see `_spec_draft_ab3.sh` wait_disk). Persistent
cache = one ~28 GB entry per 27B verify graph in `~/Library/Caches/coreai-cache`
(S=17's entry deleted with the config).

## Status / next

- [x] 0.8B mini pipeline: exports + parity + runtime gates (above).
- [x] 0.8B n-gram spec loop E2E on Mac GPU — LOSSLESS 64/64.
- [x] 0.8B draft-mode loop — LOSSLESS 64/64, tokens/target-forward 4.00 with a
      perfect drafter (catch-up/restore/window-room paths all exercised).
- [x] 27B weights re-download (55.6 GB) → verify+decode int8hu `--head-sym` exports.
- [x] 27B × 0.8B two-model run: **LOSSLESS 48/48 on all three prompts**, α and
      economics above — table is the deliverable.
- [x] S/K sweep: S∈{9,13,17} × K∈{6..16} — c_v cliff after S=13; configs frozen
      (general S=9 K=6, code S=13 K=8; S=17 dropped). Table above.
- [x] Draft A/B: 0.8B-int4 ≥ 0.8B-int8 on α at half the bytes; 2B dropped (cost eats
      its α gain). Wiring draft = `qwen3_5_0_8b_verify_s9_int4lin`. No training
      needed to wire; EAGLE-3 (C3) is the only training lever.
- [x] CoreAIChatMac wiring (commit f574c75): `SpecDecodeEngine.swift` — two-model
      WindowedModel port (S=9, K=6, snapshot/tail/exact-S re-anchor), draft
      auto-pairs via bundle metadata `spec_draft`, ⚡Spec lossless toggle + α
      stats in the statsbar, catalog entries for the verify pair,
      `expectFrequentReshapes=true` on both loads (replaces the python
      harness's reload-every crutch). Bundles symlinked into
      `~/Library/Application Support/CoreAIChatMac/models`. UNTESTED on the
      real app yet (build + a code-prompt ON/OFF byte-compare are the
      acceptance gate).
      Found & fixed on the way: the harness anchor bootstrap read a PAD row
      whenever prompt length ≡ 0 (mod S) — rag (72 tok) was affected; lossless
      gate couldn't see it (both sides shared the bootstrap). Harness now
      holds the last prompt token back and peeks it (always-correct anchor);
      stale _spec_ref_* caches deleted.
- [ ] EAGLE-3 head (C3) to lift free-form α — window tuning is
      exhausted, drafter quality is the remaining α lever.

Conventions: GPU-SOLO `_GPU_LOCK`; coreml bundles never committed.
