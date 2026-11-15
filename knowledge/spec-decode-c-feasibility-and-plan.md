# Stream C — Speculative decoding: feasibility verdict + implementation plan (2026-07-01)

> Companion to [`accel-levers-survey-and-plan.md`](accel-levers-survey-and-plan.md) (Part 2, Stream C).
> **STATUS (2026-07-01): greedy n-gram spec-decode RUNS ON THE A19 — lossless 96/96, 4.06× on a
> high-α prompt (Qwen3-4B).** De-risk + first device implementation both done this session.
>
> **REAL α MEASURED (2026-07-01, iPhone 17 Pro, real BPE prompts, lossless all PASS, K=8 gen=96):**
> the synthetic 4.06× was a CEILING. On real code/RAG the speedup is model-dependent and
> COLLAPSES on the stronger model:
>
> | model | code | RAG (grounded) | accepted/round (code / rag) |
> |---|---|---|---|
> | Qwen3-4B | 2.12× | 2.08× | 2.35 / 1.40 |
> | Qwen3-8B | 0.99× | 1.26× | 0.08 / 0.41 |
>
> Prompt-lookup α depends on the model self-repeating input n-grams: the weaker 4B echoes the
> prompt (high α → ~2×); the stronger 8B writes novel text (α≈0 → no gain). At α≈0 the fused
> scheme still costs ~0 (0.99×), so gating is harmless. **⇒ n-gram is a lossless free bonus for
> mid/weak models + extractive workloads, NOT the lever for a strong flagship. The 8B lever is a
> draft model (C2 — bundles qwen3_0_6b_gpu + qwen3_8b_gpu already exist) or EAGLE-3 (C3).**
> Greedy-only (chat sampling needs the rejection-sampling variant). Absolute tok/s are prototype
> host-loop values (the production pipelined engine starts from a higher baseline); the RATIO is
> apples-to-apples (same Driver). Repro: `_smoke/specdecode_pretokenize_realprompt.py` emits
> PB_PROMPT_IDS → `PB_PROMPT_IDS=$(cat specprompt_code.ids) _spec_device_probe.sh spec_4b_code qwen3_4b_gpu`.
> **DECISION (user, 2026-07-01): ship the 4B n-gram win as a gated CoreAIChat "fast" mode.**
>
> **SHIPPED → CoreAIChat (in-app, device-verified 2026-07-01).** New `SpecDecodeBackend.swift`
> (RWKV7Backend-style direct InferenceFunction driver + swift-transformers tokenizer from the
> bundle's tokenizer/ dir), `GemmaMode.qwen3spec` mode + a ⚡Spec/Greedy UI toggle. No-think ChatML
> is pre-seeded (reasoning traces are novel text → α=0; a fast mode wants concise answers anyway).
> A hit-rate gate drops the drafter after `gateAfter` zero-accept rounds. In-app A/B on iPhone 17
> Pro (GEMMA_SPEC=1 vs =0, **byte-identical output both times = lossless**):
> verbatim-reproduce 116.5 vs 14.5 tok/s = **8.0×** (accepted/round 8.0, K=8 ceiling); extractive
> RAG 52.9 vs 15.1 tok/s = **3.5×** (accepted/round 3.4); free-form/thinking α≈0 → ~1× (gated, no
> slowdown). Absolute tok/s is the host-loop driver on MainActor (below the production pipelined
> engine); the ON/OFF RATIO is the spec-decode win. Model sideloaded to the app container
> (`com.daisukemajima.CoreAIChat` → Documents/models/qwen3_4b_gpu); HF bundle upload + commit are
> user-gated. Headless A/B: `GEMMA_ENGINE=spec GEMMA_SPEC=0|1 GEMMA_PROMPT=… GEMMA_N=…`.

## → EAGLE-3 + TREE SESSION: START HERE (handoff 2026-07-02)

**Decision:** spec-decode's outward ship is GATED on your EAGLE-3+tree validation. n-gram is NOT
shipping standalone (marginal for chat — user confirmed on-device: "count 100" felt no faster;
n-gram only wins where output echoes input = extractive/code-only). Ship spec-decode ONCE, as the
version that's broadly fast on free-form. If EAGLE-3+tree lands → ship that. Else fall back to
draft-model (C2, 27B). n-gram stays as plumbing.

**Reusable substrate already built (this session) — plug EAGLE into it, don't rebuild:**
- `apps/CoreAIChat/Sources/SpecDecodeBackend.swift` — the driver you extend. It already does:
  stateful dense-AOT load (`expectFrequentReshapes=true` + `.gpu` — REQUIRED, else ANECompile→MTL4
  death on OS27), a **verify-forward** (S=K+1 → per-position argmax), free KV rollback via
  `processed`, greedy accept/reject (indexing proven lossless 3000/3000 in
  `_smoke/specdecode_reference.py`), no-think ChatML pre-seed, streaming, hit-rate gate. **For
  EAGLE: swap the drafter — instead of n-gram/`draft()`, run the EAGLE head to propose a TREE, and
  replace the linear verify loop with tree-mask verification.** UI: `GemmaMode.qwen3spec` +
  ⚡Spec/Greedy toggle already wired in `Gemma4ChatEngine.swift` + `ChatView.swift`. **App code is
  UNCOMMITTED on purpose (entangled with other sessions' RWKV7/BitVLA); leave it, extend it.**
- `ondevice/PipelinedBench/Sources/SpeculativeDecoder.swift` — the same algorithm as a headless
  probe (`PB_SPEC=1`; `PB_DRAFT=<dir>` = draft-model path; `PB_PROMPT_IDS` = real BPE prompt).
  Fastest place to measure EAGLE α before touching the app.

**De-risk numbers to beat (this session):** draft-model 1.7B→8B on free-form = **accepted/round
1.4–1.9** (n-gram ≈0 there). EAGLE-3+tree should beat this. Cost model: net speedup needs the draft
cheap vs target — on a 27B target a small head/draft (2–6%) → ~1.8–2.3× even at these α; a tree
lifts α further. Real α table (n-gram): 4B code 2.12/RAG 2.08×, 8B collapses.

**Infra you can use here (no 26.4 box needed):** **coreai conversion RUNS on macOS 27** — verified
this session: `export_qwen3_5_decode_pipelined.py int8lin` produced weights BIT-IDENTICAL to the
shipped bundle. So you can build the **EAGLE head + a 27B dense DYNAMIC verify bundle on this
machine** (the shipped Qwen3.6-27B Mac bundle is decode-only static S=1 — you need a multi-shape
prefill/decode/verify export, then `xcrun coreai-build compile … --platform macOS`). ⚠️ Spot-check
the **4bit** path numerically first (int8lin verified; 4bit not — no shipped ref to diff → use a
forward-vs-HF gate). Broken draft trap: `qwen3_0_6b_gpu` is an OLD (06-09, fp16) export whose graph
contract differs → garbage from prefill; use same-generation 4bit/06-18 bundles (`qwen3_1_7b_gpu` OK).

**Caveat that applies to tree too:** "lossless" is exact only in real arithmetic; fp16 batched
verify (S>1) vs sequential decode (S=1) can flip a near-tie argmax → free-form output can differ
from plain greedy by one valid token (measured: 66/96 on free-form; 96/96 on confident/extractive).
Tree verify (even wider S) will see the same. Frame it as "greedy-lossless, fp16 near-tie aside."

**Device note (2026-07-02):** iPhone 17 Pro `--console` capture dropped ~5× this session (device
stayed listed as connected but the stream stalled mid-generation). If probes hang with only load
lines, it's the console tunnel, not your code — retry / reconnect.

## HANDOFF — reproduce + next (read first)

**What exists & works:** a self-contained greedy n-gram spec-decode prototype driving a stateful
dense AOT bundle, wired into PipelinedBench as `PB_SPEC=1`. Lossless-verified on device.

**Files (all uncommitted):**
- Impl: `ondevice/PipelinedBench/Sources/SpeculativeDecoder.swift` (+ `PipelinedBenchApp.swift`
  dispatches `PB_SPEC`). Probe: `ondevice/_spec_device_probe.sh`.
- Algorithm reference / de-risk: `coreai-models-community/_smoke/specdecode_reference.py`
  (losslessness proof), `_ngram_alpha_realtext.py` (α proxy), `specdecode_speedup_model.py`,
  `cv_from_device_bench.py` (device c_v).
- Verify bundle: `coreai-models/exports/qwen3_4b_gpu/` (stateful dense AOT, already sideloaded to
  the A19 at `Documents/models/qwen3_4b_gpu`).

**Reproduce (device = iPhone 17 Pro A6F3E849, Xcode-beta):**
```
# build
cd ondevice/PipelinedBench && DEVELOPER_DIR=~/Downloads/Xcode-beta.app/Contents/Developer \
  xcodegen generate && xcodebuild -project PipelinedBench.xcodeproj -scheme PipelinedBench \
  -configuration Release -destination 'generic/platform=iOS' -derivedDataPath .build_xcode \
  -allowProvisioningUpdates build
# install + sideload (once) + run;  reuse:  SKIP_INSTALL=1 SKIP_PUSH=1
cd ../.. && PB_GEN=96 ./ondevice/_spec_device_probe.sh spec qwen3_4b_gpu
# knobs: PB_K (draft len) PB_NG (n-gram order) PB_GEN (tokens) PB_MAXSEQ (KV cap)
# result: grep 'SPEC LOSSLESS|SPEC STATS' ondevice/_spec_dev_spec.log
```

**Two non-obvious gotchas (both cost a device cycle if forgotten):**
1. The verify graph **must be AOT** (`.h18p.aimodelc`) — JIT dynamic-query dies `ANECompile → MTL4`.
2. Load it with **`SpecializationOptions.expectFrequentReshapes = true`** (+ `.gpu`) — else the
   multi-shape (prefill/decode/verify) runtime specialization tries a fixed-shape ANE compile and
   dies the same way. This flag flip is what turned the failing run green.

**Next, in order:**
1. **Real α — DONE (2026-07-01).** See the REAL α table above: 4B ~2.1× on code/RAG, 8B collapses
   (α≈0). Added `PB_PROMPT_IDS` (real BPE prompt ids) to `SpeculativeDecoder.swift` + probe; host
   pre-tokenizer `_smoke/specdecode_pretokenize_realprompt.py`. Gating (n-gram OFF on low hit-rate)
   is still TODO in the app integration. **Now shipping the 4B win as a CoreAIChat fast mode.**
2. **Vanilla draft (C2) — MECHANISM DE-RISKED GREEN (2026-07-01).** Added `PB_DRAFT=<dir>` to
   `SpeculativeDecoder.swift`: a small draft model proposes K greedy tokens (own KV), the target
   verifies all K in one S=K+1 forward (same accept/reject indexing as the proven n-gram path).
   On a FREE-FORM prompt (where n-gram α≈0), **draft=1.7B → target=8B gives accepted/round 1.4–1.9**
   (r0 accepted 6/6) — model-drafting wins on free-form, the whole point of C2. LOSSLESS 64/64.
   BUT net speedup was only ~1.0× because the draft (1.7B) is 21% of the target (8B): the K draft
   forwards eat the gain (K-sweep 4/6/8 all ~1×, ratio-bound not K-bound). Cost model → on a 27B
   target a small draft (0.6–1.7B = 2–6%) makes draft cost negligible ⇒ ~1.8–2.3× on free-form.
   **⚠️ gotcha found: `qwen3_0_6b_gpu` is a BROKEN draft — an OLD export (compression:null, 2026-06-09)
   whose graph contract differs from the 2026-06-18 4bit dense-dynamic bundles; it produces garbage
   from prefill (draft_arg=33067 vs target 785, then degenerate repeats). Use a same-generation
   bundle (`qwen3_1_7b_gpu` = 4bit 06-18 works: draft_arg=785 matches).** Remaining for the real
   win (Qwen3.6-27B dense on CoreAIChatMac): export+Mac-AOT a dense DYNAMIC verify bundle for the
   27B (current Mac bundle is decode-only static S=1) + a same-generation small draft (needs the
   26.4 convert box), then a two-model SpecDecodeBackend in CoreAIChatMac. Then **EAGLE-3 head**
   (Red Hat Speculators) for even higher α — task C3.
3. Optional perf: batched GPU argmax (currently CPU argmax on `[1,K,vocab]`), and moving off the
   host-driven loop onto the pipelined engine if the sync per-round overhead matters.

**Do NOT** re-sideload qwen3_4b_gpu (it's on the device); use `SKIP_PUSH=1`. Nothing is committed —
push / HF / model card are user-gated.

## TL;DR

- **C1 (gating prereq) = YES.** The pipelined engine already contains the verify-forward
  substrate. The S=K forward that produces **per-position logits `[1, K, vocab]`** is the
  existing chunked-prefill path; KV rollback is **free** (rewind one integer). The missing
  pieces are control logic, not a kernel.
- **Best first bet = n-gram / prompt-lookup, greedy, linear chain.** Draft is free, so it
  wins at almost any positive acceptance (break-even α≥0.01 at c_v=1.0; ≥0.24 even at a
  conservative c_v=1.3). On code/RAG/structured (α≈0.7–0.9) the model predicts **2.5–4.7×**.
- **`c_v` MEASURED (device, 2026-07-01): ≈1.0–1.1 at K≤8.** A K-token verify forward costs
  ~one decode step at small K — verify is nearly free. (Derived from AppleBenchRunner A19 STATS
  for the dense `qwen3_0_6b_dynamic` AOT-GPU bundle: `c_v(2)=1.01–1.04, c_v(4)=1.04–1.11,
  c_v(8)=1.10–1.26`; conservative — small-K is more bandwidth-bound than the fit implies. Larger
  models are MORE bandwidth-bound per token ⇒ c_v stays ~1.) **Economics: GO with wide margin.**
- **Device constraint FOUND: the S=K verify graph must be iOS-AOT-compiled.** A JIT
  dynamic-query (`main.mlirb`) graph fails `ANECompile → MTL4CommandQueueError` on BOTH the
  macOS-27 Mac GPU and the A19 (even fed at S=1).
  The static S=1 decode graphs ship fine; only the dynamic-query
  graph trips it. The AOT `.h18p.aimodelc` form runs — and dense AOT dynamic bundles already
  exist: `coreai-models/exports/qwen3_{0_6b,4b,8b}_gpu/*.h18p.aimodelc`. So spec-decode's
  verify-forward must be exported+AOT'd, not JIT'd.
- **Target = Qwen3.6-27B *dense*** (pure attention, 2 states). Avoids the rollback trap that
  hybrid/GDN models (Qwen3.5, MiniCPM-V) hit — their recurrent extra-states can't roll back
  by an integer rewind. Pick dense on purpose.

---

## C1 — Engine substrate map (what already exists)

File: `coreai-models/swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIPipelinedEngine.swift`

1. **S=K forward with per-position logits — EXISTS.** `_encodeChunk` / `_encodeNextStepGPU`
   build the logits output as `[1, queryLength, vocab]` (lines ~971, ~1328) and run one
   `function.encode`. When `queryLength = K`, the graph writes logits for ALL K positions to
   the `GrowingLogitsBuffer`. This is the production chunked-prefill path → validated for
   **dense attention** (the "S>1 is buggy" caveat in PipelinedBench is specific to the
   4-state **GDN** bundle's chunked inverse overflowing fp16, not dense).

2. **KV cache has NO internal length cursor.** `KVCache+CoreAI.swift`: the cache exposes only
   `currentCapacity`; the actual sequence position lives entirely in `EngineImpl.processedTokenCount`,
   which drives `position_ids` and the token write offset. `reset()` clears state by just
   setting `processedTokenCount = 0` and relies on "attention only reads positions below the
   offset." ⇒ **rollback to accepted length j = set `processedTokenCount = n + j`.** Stale KV
   rows at positions ≥ that are overwritten by the next forward and never attended. Free.

3. **GPU sampler is single-position.** `MPSGraphSamplers.swift`: argmax graph is compiled for
   `[1, vocab]`; `encodeWithSlice` blits only the LAST position before sampling. For verify we
   need argmax at EVERY position → add a batched argmax graph `[K, vocab] → [K]`
   (`reductionArgMaximum(axis:1)`), or issue K single-position argmax dispatches. **Greedy
   verify needs only the K argmax token-ids back on CPU — NOT the full logits.**

4. **Algorithm-level verify primitive EXISTS for a cheap Mac oracle.** `InferenceOptions` has
   `includeLogits` + `forcedContinuation`; the sequential / static-shape engines run the model
   forced along given tokens and return per-position logits, and `ContinuationEvaluation.swift`
   already scores a fixed continuation → `[num_cont, vocab]`. Use this to validate the
   accept/reject ALGORITHM on Mac before touching the pipelined engine. (Slow path; oracle only.)

5. **No spec-decode code exists anywhere** (`grep speculat|eagle|n-gram` = empty). Greenfield.

### The delta — new ENGINE work for greedy n-gram spec-decode
- A `verify-forward` entry on the engine: feed K draft tokens at positions `[n .. n+K-1]`,
  one encode, read **per-position argmax** (K ids) — not just last.
- Batched argmax sampler `[K, vocab] → [K]` (new MPSGraph, ~20 lines, mirrors the existing one).
- Accept/reject loop (greedy): `aᵢ = argmax(L_{i-1})`; accept `dᵢ` iff `dᵢ == aᵢ` and all prior
  accepted; on first miss at j, emit `d₀..d_{j-1}` + correction `a_j`, set `processedTokenCount = n+j`.
  All-accept ⇒ emit `d₀..d_{K-1}` + bonus `a_K`.
- n-gram drafter: pure host hashtable over the generated+prompt token stream (last m tokens →
  most-recent continuation). Zero model cost.
- Bypass the public guards: the pipelined `generate()` throws on `includeLogits` /
  `forcedContinuation` (GPU-side sampling). The verify path is an INTERNAL engine API, not the
  public sampling call.

### Deferred / harder
- **Lossless sampling verify (temp>0):** needs probabilities + residual-distribution resample,
  not argmax. Greedy first.
- **Hybrid/GDN models:** recurrent extra-states can't roll back by an integer → need
  snapshot/restore. Dense Qwen3.6 sidesteps this.
- **EAGLE-3 tree verify:** needs a tree attention mask + a trained head. Linear chain first.

---

## C-value — speedup model (greedy, linear chain)

`E(α,K) = (1 − α^{K+1})/(1 − α)` new tokens per verify round (accepted drafts + 1 bonus).
`speedup = E / (draft_cost + c_v)`, costs in units of one target decode step.
- n-gram: `draft_cost ≈ 0` → `speedup = E / c_v`.
- vanilla draft: `draft_cost = K·r`, `r = target_tps / draft_tps`.

Break-even α for a net win (best K):
| draft source | break-even α |
|---|---|
| n-gram, c_v=1.0 | **0.01** |
| n-gram, c_v=1.3 | **0.24** |
| vanilla, r=0.10 (draft 10× faster) | 0.18 |
| vanilla, r=0.20 (5× faster) | 0.31 |
| vanilla, r=0.32 (0.8B vs 27B-on-iPhone) | 0.45 |

n-gram predicted speedup (c_v=1.0 / 1.3), best K≤8:
| α | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| c_v=1.0 | 2.00 | 2.47 | 3.20 | 4.33 | 6.13 |
| c_v=1.3 | 1.54 | 1.90 | 2.46 | 3.33 | 4.71 |

Read: **n-gram is robust** (free draft ⇒ never a net loss when α>break-even, and α≈0 chat just
costs the wasted verify ≈ c_v−1 per round, which you cap by disabling n-gram when recent hits
are low). **vanilla draft is fragile** — at the realistic r=0.32 it needs α≥0.45 and tops out
~1.8×; only worth it if a much faster/closer draft (r≤0.2) is available. Confirms the plan's
ordering: n-gram → vanilla → EAGLE-3 (trained head buys the high α that makes everything sing).

Model script: `_smoke/specdecode_speedup_model.py`.

### Control-logic proof + empirical α (model-free, run 2026-07-01)

- **Losslessness PROVEN.** `_smoke/specdecode_reference.py` is the greedy spec-decode reference
  (n-gram drafter + accept/reject + bonus + "rollback"=commit-only-accepted). Against 3000
  cases (random vocab/prompt/length, hash + Markov oracles, K∈{1,2,3,4,8}, ngram∈{1,2,3}) the
  spec-decode output is **token-identical to plain greedy (3000/3000)**. The accept/reject
  INDEXING the Swift must mirror is now locked: `aᵢ = argmax(prefix+draft[:i])`; commit accepted,
  emit correction (first miss) or bonus (`argmax(prefix+draft)`). The Swift engine maps `a₀` to
  the cached last-step logits and `a₁..a_K` to positions `0..K-1` of the one verify-forward.
- **Empirical α proxy on REAL text** (`_smoke/specdecode_ngram_alpha_realtext.py`, prompt-lookup
  self-repeat = model-free upper proxy; word-level): per-position expected accepted draft tokens
  from n-gram ALONE — Python code **~2.1**, Swift code **~0.75**, markdown/prose **~0.5**
  (ngram=3). Code/structured ⇒ ~3 tokens/forward incl. bonus ≈ the model's 2.5–4× band; prose
  lower. CAVEAT: this is text self-repetition; real α also needs the target MODEL's greedy output
  to repeat — true for input-grounded tasks (code-completion/RAG/structured), but free-form chat
  generates novel text so real chat α ≪ this proxy. Confirms: n-gram is a code/RAG win, a chat
  no-op (gate it off when recent hits are low).

---

## Plan — staged, each stage gated on the previous

1. **Measure c_v — DONE (2026-07-01).** c_v≈1.0–1.1 at K≤8 from existing AppleBenchRunner A19
   data (`_smoke/cv_from_device_bench.py`). Economics confirmed. En route, found the AOT
   constraint (above): JIT dynamic-query graphs die on ANEC/MTL4 on Mac-27 AND A19; the S=K
   verify graph must be AOT `.h18p.aimodelc`. GDN chunked bundles (`minicpmv46_text_chunked_*`,
   both int8lin and the metal variant) fail this way on device — do NOT use them; dense AOT
   `qwen3_*_gpu` bundles are the ones that run.
2. **Mac algorithm oracle.** Implement n-gram drafter + greedy accept/reject in a small harness
   driving the *sequential* engine via `forcedContinuation`/`ContinuationEvaluation`; verify the
   accepted stream is **token-identical** to plain greedy (losslessness) and log measured α on
   code vs chat prompts. No device, no GPU lock for correctness.
3. **Engine verify-forward + batched argmax.** Add the internal API + `[K,vocab]→[K]` sampler to
   the pipelined engine. Unit-gate: verify-forward argmax stream == K separate S=1 argmaxes.
4. **Wire n-gram greedy into the pipelined decode loop.** Bench on A19 via PipelinedBench
   (`ondevice/PipelinedBench`, `com.coreai.pipelinedbench`) — decode tok/s with output
   distribution preserved, α per domain. **Gate: lossless + net speedup on code/RAG.**
5. **vanilla draft** (second engine instance, qwen3.5-0.8B) only if r turns out ≤0.2.
6. **EAGLE-3 head** via Red Hat `Speculators` (external GPU box) → tree verify. Separate session.

### Implementation status — WORKS ON DEVICE (2026-07-01)
- **Greedy n-gram spec-decode prototype RUNS ON A19, LOSSLESS, 4.06× on a high-α prompt.**
  `ondevice/PipelinedBench/Sources/SpeculativeDecoder.swift` (`PB_SPEC=1`; probe
  `ondevice/_spec_device_probe.sh`). Self-contained host-driven loop (mirrors the chunked host-loop +
  `_smoke/specdecode_reference.py`) — NO pipelined-engine surgery, NO GPU argmax kernel (CPU argmax on
  the `[1,K,vocab]` verify output). Loads the **stateful dense AOT bundle** `exports/qwen3_4b_gpu`
  (input_ids+position_ids, 2 KV states, all-position logits) via `AIModel(contentsOf:)`.
  **FUSED scheme**: each round forwards `[a0]+draft` (a0 = argmax(cachedLogits), always-correct next
  token), per-position argmax verifies drafts, commit accepted + advance `processed` (rollback = don't
  advance past accepted); one forward per round.
- **Device result (iPhone 17 Pro, Qwen3-4B, K=4, ngram=3, gen=96, repetitive prompt):**
  `SPEC LOSSLESS 96/96 PASS` · `greedy=19.9 tok/s  spec=80.9 tok/s  speedup=4.06×
  accepted/round=4.00  forwards=20`. **Lossless verified on real hardware** (fused-scheme indexing
  correct). c_v cross-check: spec forward (5 tok) 59 ms vs greedy step 50 ms ⇒ **c_v(5)≈1.18**,
  matching the AppleBenchRunner-derived c_v(4)≈1.04–1.11 — independent confirmation that verify is
  ~free.
- **⚠️ THE FIX that made it run**: a dynamic multi-shape graph (prefill S=P / decode S=1 / verify
  S=K+1) driven directly MUST load with **`SpecializationOptions.expectFrequentReshapes = true`**
  (+ `.gpu`) = `ModelStructure.dynamic.specializationOptions`. With `false` the runtime attempts a
  fixed-shape **ANE compile that fails on the OS27 beta** (`ANECompile → MTL4CommandQueueError`, the
  same class the JIT/GDN bundles hit). This is why PB_LLADA (single fixed S) worked but multi-shape
  did not until the flag flip.
- **Caveat / honesty**: accepted/round=4.00 is the CEILING (the prompt is maximally repetitive → 100%
  n-gram hit). Real α is per-domain (code/RAG ~0.7–0.9 → ~2.5–3.5×; free chat ~0 → ~1×). The win here
  proves the **implementation is correct + the mechanism delivers the modeled speedup**, not a
  real-workload number. Not wired into any chat app; n-gram only; greedy only.
- Next: measure real α on code/RAG prompts (needs the bundle tokenizer to encode text); gate n-gram
  off when recent hit-rate is low (so chat isn't slowed by wasted verifies); then vanilla-draft /
  EAGLE-3 (C3) for the high-α general case.

Conventions (from the campaign doc): engine work is C's resource; own files by explicit path,
never `git add -A`; no "claude" in commits; English code/UI; don't claim a win before the bench
shows one; push/HF/card user-gated. Build/test on the real device (no local build test).
