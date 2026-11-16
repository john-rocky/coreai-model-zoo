# Gemma4 raw-Metal loop on A19 — findings + the remaining (parked) levers

*(2026-07-15, session 2 of the raw-Metal decode loop. Working tree `coreai/_metal_loop/`
(local, uncommitted), full ladder + traps in `GEMMA4_METAL_LOOP_STATE.md`. This doc is
the durable catalogue so the tail-end levers can be resumed later without re-deriving.)*

## Where it landed

iPhone 17 Pro, Gemma4 E2B mixed-bit (783 MB/token), fresh settled trial1, greedy,
p128 g256 — **everything below is LOSSLESS** (device S1 token-exact 3/3 + MTP lossless
3/3, Mac + python gates green at every step):

| config | decode tok/s |
|---|---|
| Core AI pipelined engine (shipped) | 36.5 |
| raw loop, session 1 (M4-tuned kernels) | 47.7 |
| raw loop, session 2 (A19-tuned, below) | **55–56.2** (day noise ±1.4) |
| LiteRT-LM (single historical measurement) | 57 |

56.2 = 43.8 GB/s effective whole-model vs LiteRT's 44.6 → 98% of parity. Next step is a
same-afternoon interleaved A/B vs LiteRT (`GEMMA4_LITERT_AB_KICKOFF.md`), NOT more
kernel work.

## What moved the needle (largest first) — all bit-exact by construction

1. **Constant-memory byte-LUT for the int2 decode (+~4.3 tok/s).** The int2 gateup
   (16 codes/word × 2 matrices, ~6 ops/code) measured **23.4 GB/s CACHE-HOT** on A19 =
   ALU-bound, ~42% of the token from ~24% of the bytes. `constant half4 LUT2[256]`
   (byte → 4 decoded values) → **59.8 GB/s**; int2 down 50.8 → 80.4. Products x·c with
   c ∈ {−2,−1,0,1} are EXACT in fp16 and the fp32 accumulation stays in code order →
   bit-exact. The 2026-07-03 "kernels are BW-bound at 43.5, tuning closed" verdict was a
   probe-mix artifact. NOTE: the *banned* variants (threadgroup byte-LUT, shl-asr) die on
   the tg-gather — the constant cache is the working lane. lm_head int2 is genuinely
   DRAM-bound (40.6 GB/s); LUT is neutral there, it keeps the arithmetic decode.
2. **Dispatch-count fusion, 452 → 253/token (+~3 tok/s).** A19 charges **~8 µs per
   dispatch ON the GPU timeline even with no hazards** (dep chain 8.5 µs, independent
   8.0 µs — issue rate, not fences; M4 ≈ 3 µs; decode wall == gpuBusy, CPU contributes
   zero). Folds: residual-add rmsnorms into the next matvec prologue (`_nxa/_nxaa`,
   sg0-only fold + threadgroup-broadcast inv2), q/k/v single dispatch, q-rope/k-rope/
   v-norm merged into the SDPA dispatch (`flash_sdpa_rope_occ` — key `pos` is the owner
   subgroup's last strided iterate, so the online-softmax order is preserved exactly).
3. **A19-specific tile/G retune (+~2.3).** M4's R=1 matvec lane is an A19 regression →
   R=4; sliding SDPA G 16 → 8. Decode-G and verify-G must MATCH (G changes the fp32
   strided-merge order; MTP losslessness needs verify == S1 bitwise).
4. **char4 int8 weight loads** (PLE + model_proj; 1 B scalar loads are issue-bound) —
   small but free.
5. **Interleave-4 pack (`p2b_interleave_pack.py`, QP words [N/4, kw, 4], uint4 loads):
   ~0 gain** — the flat layout was already 128 B-coalesced per SIMD-group. Kept (device
   pack is il, bit-exact, `IL4` preprocessor macro + `@il` PSOs), but do NOT expect
   speed from wide loads here again.

## Traps (cost us time; will bite again)

- **fp16 op-sequence trap, 2nd sighting**: writing rope as `c*x1 - s*x2` instead of the
  reference's explicit `m1=c*x1; m2=s*x2; … m1-m2` let the Metal compiler contract to
  FMA **even under mathMode .safe** → near-tie drift (MTP tokens/round 1.66 → 1.70).
  RULE: copy the reference kernel's op sequence LITERALLY, temporaries included.
- Thermal protocol: back-to-back device runs droop 5–10%; "20-min-cold" does NOT
  measure faster than mid-session (tested) — day noise is ±1.4 tok/s. Fresh = trial1 of
  a settled run, and cross-config claims need interleaved A/B, not different-day numbers.
- Per-kernel BW probes on small tensors read CACHE-HOT on A19 (SLC swallows 5–10 MB
  tensors × reps) — only ≥100 MB streams (lm_head) give true DRAM numbers.

## The parked levers (resume checklist, expected sizes)

Current wall decomposition at 56: exec ≈ 16.0 ms (≈ DRAM-stream levels everywhere after
the LUT) + **dispatch tax ≈ 1.8 ms (253 × ~7 µs)** + thermal band ±0.5 ms.

| lever | expected | notes |
|---|---|---|
| argmax stage1+2 merge (single-dispatch, keep first-max tie-break) | ~+0.05 tok/s | trivial, do with the next kernel touch |
| embed_gather folded into model_proj prologue (front 4→3) | ~+0.03 | trivial |
| TOKENS_PER_CB retune under 253 dispatches | ~0 (was flat at 452) | one env sweep |
| lm_head stream 40.6 → 45 GB/s question (short 96-word rows ⇒ per-row epilogue density; try R=8 head tile) | +0.3–0.8 IF it's not the true DRAM ceiling | KPROBE first (`G4_KPROBE=1`) |
| **mega-kernel restructure** (LiteRT-style: whole-layer or multi-stage single dispatches to attack the 1.8 ms issue tax) | up to +5, uncertain | a PROJECT, not a lever; only if the goal is beating LiteRT rather than parity |
| MTP on A19 | ⛔ stays OFF | with all session-2 levers: sky 48.0 eff < S=1 55.9 (verify still compute-bound; bandwidth-bound-runtime lever only) |

Env knobs baked into both runners (defaults = ship config): `G4_R1O/G4_R1D` (R=1/4),
`G4_GS/G4_GF` (SDPA G), `G4_FUSE`, `G4_LUT2`, `G4_FSR`, `G4_PROBE` (dispatch-tax probe),
`G4_KPROBE` (per-kernel BW), `G4_SKIP_MTP`, `G4_MATH=fast` (reproduces the losslessness
break). Mac is unaffected by the A19 lanes (123.5 vs 124.1, its wall was never links).
