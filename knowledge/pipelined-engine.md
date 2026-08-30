# Riding Apple's pipelined GPU engine — 3.5× over a hand-rolled loop, zero custom kernels

> The single highest-leverage Core AI LLM finding in this project (2026-06-10/11, verified on
> M4 Max + iPhone 17 Pro): the same `.aimodel` weights decode **3.5× faster** when Apple's
> `coreai-pipelined` engine drives them instead of a hand-rolled per-token `fn.run()` loop —
> Qwen3.5-0.8B int8: **210 tok/s vs 58.5 on M4 Max, 69.7–74.0 vs 42.5–45.4 (fused-kernel
> monolith) on iPhone 17 Pro** (ship config incl. the per-block-32 absmax int8 head; fp16-head
> figures were 204 / 50.3–51.5). This page is how to put a model on that engine, every trap we
> hit, and what doesn't fit. Working artifacts: the qwen3.5 fast path in
> [`../conversion/export_qwen3_5_decode_pipelined.py`](../conversion/export_qwen3_5_decode_pipelined.py)
> + [`../apps/coreai-pipelined-extra-states.patch`](../apps/coreai-pipelined-extra-states.patch)
> + the [`gpu-pipelined/` HF bundle](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI).

## Why hand-rolled per-token loops lose ~3×

The zoo's first engines (and most sample code) drive low-level `AIModel.load()` →
`await fn.run()` per token. Three structural costs, none of them kernel-fixable:

1. **Synchronous dispatch** — `await fn.run()` blocks on GPU completion every token; CPU prep
   for token N+1 never overlaps GPU work for N. The pipelined engine `encode()`s and returns,
   with a `PipelineGate(capacity: 3)` bounding in-flight steps; the sampler's Metal completion
   handler yields the token.
2. **CPU sampling = a full-vocab logits readback per token** — 250–600 K-vocab fp16 logits
   cross the GPU→CPU bus, then argmax on CPU. The engine samples **on-GPU** (MPSGraph
   argmax/topK); only the sampled int crosses.
3. **Host-cache KV** — re-feeding the whole KV per token (the workaround for the in-graph
   KV-write bug, see [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md))
   plus multi-dispatch per token. The engine grows KV **on-device** (buffer-expand + async blit).

Correction this forced: [`performance-ceiling.md`](performance-ceiling.md)'s "the MLX gap is
structural" verdict was measured on the hand-rolled loop — it was that loop's ceiling, not Core
AI's. The official engine is ~2× **faster** than MLX on the same machine (qwen3-0.6B-4bit:
~1,150 tok/s vs MLX ~535).

### The fork's sampler drain — retired 2026-08-29 (measured redundant after upstream #121)

Fork commit 427a05b (2026-07-23) put `await computeStream.currentWorkCompleted()` before every
GPU sampler encode, against a bogus first token sampled from a not-yet-written logits row
(observed under the composite sampler after the 0.2.0 merge). Upstream #121 (aff0bb2) later
gave the composite sampler a per-call `MPSGraphExecutableExecutionDescriptor` — the shared one
corrupted scratch across in-flight steps, the same class of symptom. Re-measured with #121 in
the tree (macOS, qwen3-0.6b 4-bit dynamic, release, drain vs. none interleaved, GPU lock held):
single turn N=20/arm, 200 tokens — greedy **354.5 → 538.5 tok/s (+52%)**, temperature 0.7
**336.6 → 481.0 (+43%)**, greedy output token-identical across all 40 runs, zero repeated
words / doubled punctuation in the sampled runs; multi-turn (generate → partial `reset(to:)` →
generate, the "post-drain boundary" the original symptom named) 12 turns × 3 reps/arm — greedy
streams identical, sampled runs clean. Removed in zoo-0.4 (`8c6f804`). What the code says: the
model step and the sampler share one `MTLCommandQueue`, and upstream's own constrained path
(#169/#170) commits the sampler straight after `function.encode(to:)` with no drain. Residual:
**not measured on iOS**; the other `currentWorkCompleted()` sites (reset/teardown) stay.
Harness: `~/code/coreai/_zoo04_harness/` (`forkturns` = fork by path, `hybrid2turn` = kit by
path).

## What a model needs to ride the engine

`EngineFactory.createEngine` auto-selects `coreai-pipelined` for **dynamic-shape** bundles
(`chunkedStatic` → the static/ANE engine instead). The checklist:

- **One `main` graph, `input_ids [1,S] → logits`**, embedding and lm_head **in-graph** (the
  engine feeds token ids and reads sampled ids — there is no hook for host-side gathers or a
  separate head dispatch). Embed+head tables must therefore fit in the graph: fine up to a
  few GB fp16, which covers 0.3–4B-class models.
- **A growing KV pair** (`keyCache`/`valueCache`, dynamic seq dim, layers stacked in one
  tensor). Sliding-window layers can ride as plain linear KV (the mask keeps the math right;
  you pay some memory).
- **At most 2 extra fixed-shape states** — with our engine patch
  ([`../apps/coreai-pipelined-extra-states.patch`](../apps/coreai-pipelined-extra-states.patch)).
  Stock, the engine hard-requires *exactly* the 2 KV states; the patch lets it carry e.g. a
  hybrid-SSM's `convState`/`recState`: allocated zero-filled, bound on every encode, zeroed on
  `reset()`. (Patch gotcha for future edits: `AsyncMutableViews.insert` is
  `@_lifetime(self: &value)` — the insert and the consuming `encode` must sit in ONE scope;
  hence the `switch extraStates.count` blocks instead of a loop.)
- **A full LanguageBundle directory** — `metadata.json` (kind `llm`, `assets.main`,
  `language.{tokenizer,vocab_size,max_context_length,function_map}`) + `tokenizer/` +
  the `.aimodel`. A bare `.aimodel` dir is not loadable by `LanguageBundle`/`EngineFactory`.
- **Giant per-token host tables need the per-token-inputs patch**
  ([`../apps/coreai-pipelined-per-token-inputs.patch`](../apps/coreai-pipelined-per-token-inputs.patch)).
  Gemma 4 (3n-style) gathers a 9.4 GB per-layer-embedding table by token id on the host every
  step; stock, the engine has no hook for that. With the patch, inputs beyond
  `input_ids`/`position_ids` (fixed-shape, S=1 — e.g. `ple_tokens [1,1,L,ld]`) are filled each
  step by an `EngineOptions.perTokenInputProvider` callback (sub-ms mmap gather). Prefill stays
  fully pipelined (slot-per-position buffer; prompt tokens are known up front); decode waits
  for each GPU-sampled token before encoding the next step — that round trip measured
  ~2.4 ms/token on M4 Max and ~13 ms/token on iPhone 17 Pro for Gemma-4-E2B (the
  on-GPU-argmax + on-device-KV wins survive; the gather itself is ~0.1 ms). Default-off:
  two-input models run the engine bit-identically to before (qwen 204 tok/s regression intact).
- **Constant gather tables can instead ride as STATIC inputs — the static-inputs patch**
  ([`../apps/coreai-pipelined-static-inputs.patch`](../apps/coreai-pipelined-static-inputs.patch))
  removes that per-token serialization entirely: export the table itself as a graph INPUT
  (`ple_table [V, L*ld] int8` + `ple_scale [V] f32`) gathered in-graph by `index_select` on
  `input_ids` (int8 table input + gather + cast + scale-multiply all lower cleanly on the GPU
  delegate, bit-exact vs a numpy reference), and hand the engine the buffers once via
  `EngineOptions.staticInputBuffers` — bound unchanged on every encode, no provider call, no
  S=1 constraint, no decode wait on the sampled token, so the full 3-deep pipeline survives
  in decode (decode ≈ prefill). Two traps, both invisible until you bench: the runtime
  ingests REGULAR inputs by value, so feeding a 2.35 GB table through a python NDArray costs
  ~2.6 s/step — only the Swift `AsyncValue`-over-MTLBuffer path is viable; and **buffer mode
  decides everything**: a `PROT_READ`-only mmap under `makeBuffer(bytesNoCopy:)` costs
  ~65 ms/GB *per encode* on the macOS delegate (silent, size-proportional — 4.8 tok/s instead
  of 77), a writable COW mmap (`PROT_READ|PROT_WRITE`, `MAP_PRIVATE`) is free on the Mac but
  still pays ~6–7 ms/GB/encode on iPhone (file-backed residency), and an OWNED
  `storageModeShared` buffer (read the file in once) is free on the Mac and cheapest on
  iPhone but is dirty memory against the jetsam limit — budget it (a 12 GB iPhone 17 Pro
  gives an entitled app ~6.4 GB and the gemma4 trials peak ~2 GB above the table bytes) and
  ship the `increased-memory-limit` entitlement. Every statically-bound byte pays the iPhone
  per-encode tax, so bind ONLY what cannot live in-graph (gemma4: the PLE table — moving the
  0.8 GB embed table out as a 3rd input measured strictly worse). Outcome on gemma4-E2B
  (owned buffers, AOT h18p, settled device): **iPhone decode 30.3 / prefill 38.9 vs the
  per-token provider's 26.5 / 40.5 — decode +14%, prefill ≈ par** (M4 Max: 77.0 / 87.1 =
  +8.6% decode). One more measurement rule on top of the install-adjacent one: a
  just-unlocked iPhone under-reads ~35% (19.8 vs 30.3 ten minutes apart) — bench settled.

## The export trick: decode-only, loop-free

Recurrent scans (`torch.ops.higher_order.while_loop`, e.g. Qwen3.5's GatedDeltaUpdate) do
**not lower** on the MPSGraph GPU delegate (`'scf.while'` region type mismatch), and on the
macOS-27 beta the while_loop bundle fails even `cpu_only` (Compiler error 2) — so "it verified
on CPU earlier" proves nothing about the GPU graph. The escape that made the qwen3.5 ride
possible:

- **`input_ids` STATIC `[1,1]`** — at S=1 a scan is one step, so a loop-free single-step
  recurrence (`use_loopfree_step=True` on every linear-attention layer, set BEFORE
  quantization/tracing) is numerically identical and removes `scf.while` from the graph
  entirely.
- `position_ids` and the KV seq dims stay **dynamic** → the factory still classifies the
  bundle as dynamic → pipelined engine.
- **Prefill becomes pipelined S=1 steps**: run with `COREAI_CHUNK_THRESHOLD=1`. Prompt tok/s ≈
  decode tok/s (~51 on iPhone, ~200 on M4 Max). For long prompts a static chunked-prefill
  companion graph (q=16 unrolled scan, 147 tok/s on iPhone) still wins time-to-first-token;
  a chunkwise-parallel (no-scan) prefill formulation is the open fix.

## Run contract (every one of these bit us)

- `COREAI_CHUNK_THRESHOLD=1` **before engine creation** (`ModelConfig.chunkThreshold` reads the
  env var; `setenv` in app init is fine).
- **Never call `engine.warmup()`** on an S=1 bundle — it warms query length **256** and the
  static `[1,1]` graph rejects it (`NDArrayDescriptor` fatal). `llm-benchmark` is safe (warms
  via a real trial); `llm-runner` needs `--warmup exact --warmup-length 1`; in an app, a
  1-token generate after load IS the warmup.
- **Stock `apple/coreai-models` cannot even CONSTRUCT the pipelined engine for an S=1
  decode-only bundle** (logits static `[1,1,vocab]`): `CoreAIPipelinedEngine.swift` sizes
  `GrowingLogitsBuffer` to `averageExpectedPromptSize` (256) before any token is bound, and
  `TensorStorage+CoreAI.swift` resolves `[1,256,vocab]` against the static descriptor →
  `Shape at dimension 1 of 256 is not a valid substitution for source shape 1`. The `256` in
  that message is the constant, not the prompt length (fingerprint). The zoo fork's five-line
  `logitsSeqIsStatic` guard (john-rocky/coreai-models@9e5b605, 2026-06-13) is what makes every
  zoo S=1 bundle load; `CoreAIStaticShapeEngine` is not involved (an S=1 bundle with a `main`
  function and no `extend_*`/`load_embeddings` is `.dynamic` → pipelined, in both trees).
  Reported upstream as apple/coreai-models#212 (2026-08-29, patch inline); a third party had
  misattributed it to "the fork's static-shape engine chunked prefill" (Liquid4All/pipette-clients#13).
  Upstream triage (stikves, 2026-08-29): a feature request outside the pipelined engine's scope, "as
  it is designed for dynamic inputs" — so plan on the fork guard, not an upstream fix. The S=1
  rationale (this section) was answered on-thread 2026-08-30 (issuecomment-5471143196).
  `COREAI_CHUNK_THRESHOLD` itself is upstream (`ModelConfig.swift`), so guard + `=1` is the whole recipe.
- **Benchmark Release builds only** — a Debug engine measures ~3× slow (host-side per-token
  work dominates unoptimized Swift).
- Cold GPU specialization of the 0.8B bundle: ~4.8 s on iPhone, then ~0.2–1.0 s warm loads
  (content-keyed cache) — no AOT needed on this path. The 2.3 GB 2B bundle: 29.1 s cold,
  3.0 s warm; the 2.9 GB 2B ship bundle (int8 head): 22.3 s cold, 5.6 s warm.
- **2B-class bundles (≥~2 GB) need the `com.apple.developer.kernel.increased-memory-limit`
  entitlement on iPhone** — cold specialization dies with `std::bad_alloc` (SIGABRT) at the
  default jetsam limit. With the entitlement the same spec completes.
- **Failed cold specializations leave partial e-caches** in the app container that eat device
  disk (~3.5 GB for the 2B) and make every later attempt fail as `NSPOSIXErrorDomain code=2`
  ("No such file or directory") at engine create — an out-of-disk ENOENT chain, not a payload
  problem. Recovery WITHOUT losing sideloaded bundles: ship a cache-wipe hook that removes
  `Library/Caches/coreai-cache` before engine creation (CoreAIChat's
  `GEMMA_CLEAR_SPEC_CACHE=1` — device-verified: after the wipe the same 2B spec completed in
  20.9 s where every prior attempt ENOENT'd; every model pays one cold re-spec). Last resort:
  uninstall the app (clears the container incl. all sideloads), reinstall, retry with ≥~4 GB
  free. Diagnose by logging `attributesOfFileSystem` free space from the app.
- **Adding a GB-class AOT bundle to a container can break OTHER models' cached
  specializations.** First engine create for the gemma4 `--tbl` `.aimodelc` loaded from
  Documents ingests its ~2 GB executable into the content-keyed coreai-cache (engine load
  ~11 s including the copy, ~6 s warm; one first-ever attempt instead sat silent for
  ~10 min after the table-mapping log and died without a console line — unreproduced, and
  the completed ingest survived: if you see that, relaunch). Right after that ingest, the
  gemma4 ANE chunk set in the SAME container started dying at load with the MPSGraph
  assertion `Unable to use cached specializations and original module not available`
  (signal 6; the first failing attempt exited 1 after chunk load). Recovery = the in-app
  cache wipe (`GEMMA_CLEAR_SPEC_CACHE=1`); every model pays one cold re-spec (ANE chunks
  53.8 s) and the tbl ingest re-runs cleanly. Rule of thumb: after adding a multi-GB
  bundle next to other specialized models, expect one wipe + re-spec cycle.

- **Driving hybrid bundles from an APP (found wiring Ornith-1.0-9B into CoreAIChatMac,
  2026-07-03): the factory's `coreai-sequential` variant hard-requires exactly 2 states**,
  so every 4-state hybrid (the whole qwen3.5 family, Granite) fails engine-create with
  `invalidOutputType("Expected 2 states … got 4")`. Route those to the (extra-states-patched)
  `coreai-pipelined` variant instead, and remember `COREAI_CHUNK_THRESHOLD=1` is read LIVE at
  prefill time — set it for the pipelined load, restore the launch value before sequential
  loads in the same process. CoreAIChatMac's pattern: try sequential, catch that specific
  error, retry pipelined (gather_qmm bundles excluded — their logits shape asserts in the
  pipelined `GrowingLogitsBuffer`).
- **Streaming display is O(n²) if you re-decode the full sequence per token** — a 601-token
  answer displayed 37 tok/s while the engine ran 48 (the stream buffers; only the consumer
  lags). Fix: incremental detokenization (decode a small tail cache, fold into stable text at
  newline boundaries; a trailing U+FFFD means a UTF-8 char is still split across tokens — hold
  the fold) + coalescing UI writes to ~20 Hz. Suffix decode is only valid for byte-level-BPE
  tokenizers — SentencePiece strips a leading space per chunk — so audit the first fold against
  one full decode and fall back. Displayed rate recovered to 48.9 at 601 tokens.

## State & precision traps on the GPU delegate (found by the LFM2.5 port)

Two macOS-27-beta GPU-delegate behaviors that produce *silently wrong* decode (the bundle
loads and runs; only numerics gating catches them). Both bit LFM2.5-1.2B and neither bit
qwen3.5 — pattern- and model-dependent, so treat them as authoring rules:

1. **Don't chain per-slot writes on one fixed-shape state.** N per-layer
   `SSMState.update_states` calls compile to N read_handle → slice_update → write_handle
   round trips on the same state handle; with N > 1 the GPU delegate dropped them ALL (state
   buffer stays zero — the compiled IR is correct and token-chained; 1 slot works, a 3-slot
   repro fails; qwen's 18-slot conv/rec pattern happens to survive). Symptom: position 0 is
   fine (fresh state IS zero), everything after decodes garbage. **Rule: collect each
   layer's new state slice and issue ONE fused full-state `slice_update` per step** (reads
   stay per-layer narrows of the input state; slots are disjoint, so semantics are
   identical). The KV growing pair is unaffected (its written values are re-read in-graph).
2. **fp16 matmuls in the attention prologue lose ~1.3% relative accuracy under a
   dynamic-shape graph** (the same projection in an all-static graph measures 0.07% — the
   delegate appears to pick an fp16-accumulation kernel when dynamic dims are present).
   Whether that matters is model-dependent: LFM2.5's large q/k-norm gains (|k| up to ~14)
   amplify it and the error compounds across the stack into garbage logits (full-stack cos
   0.71 vs eager); qwen3.5's modest activations shrug it off. **Rule: if the oracle gate
   fails with healthy per-position cosines that decay through the stack, keep the four
   attention projections (q/k/v/out) in fp32** — weights fp32, cast in/out around the
   matmul; layer-level error drops to ~1e-5 at +2 bytes/param for those four (LFM2.5-1.2B:
   +126 MB). Conv-mixer and MLP matmuls measured clean in fp16.

## Quantization on the GPU delegate (measured, qwen3.5-0.8B, M4 Max p128/g256)

| config | size | decode tok/s | verdict |
|---|---:|---:|---|
| fp16 | 1.4 GB | 175.8 | baseline |
| int8 k-means g32 (256-entry LUT) | 1.0 GB | 113.2 | **LUT gather is the bottleneck** — slower than fp16 |
| **int8 linear per-block-32 (no LUT)** | 1.0 GB | **204.1** | ship config; ≡ fp16-GPU token-for-token |
| + untied int8 lm_head | 1.3 GB | 201.4 | **no win** — the head matvec isn't the critical path; naive bandwidth models can lie |
| int4 k-means g8 (+int8 rescue variants) | 0.75–0.88 GB | — | **fails the oracle gate** (12–16, 14/16, 12/16); per_tensor int8 rescue is *coarser* than int4-g8 LUT for SSM in_projs |

The gemma4-E2B sweep (all oracle-8/8, engine path with the PLE provider, M4 Max p128/g256)
settles the LUT question — same bytes, 2.25× apart:

| config | bundle | prefill | decode | verdict |
|---|---:|---:|---:|---|
| int4 k-means g32 (16-entry LUT) | 1.9 GB | 41.0 | 31.5 | LUT dequant dominates |
| int8 linear per-block-32 | 3.1 GB | 71.9 | 57.2 | BW-bound (~165 GB/s effective) |
| **int4 linear per-block-32** | 2.0 GB | **85.3** | **70.9** | ship class; +20–25% over the zoo's kernel CLI |
| int4lin `--tbl` (PLE table as static input) | 2.0 GB + 2.35 GB table | 87.1 | **77.0** | +8.6% decode on Mac (no per-token wait); see the static-inputs bullet for the iPhone buffer-tax economics |

Rules of thumb: **eager-palettized k-means LUT dequant is the slow class on this delegate at
ANY entry count** — 256-entry int8 (qwen 113) and 16-entry int4 (gemma 31.5) both lose to
per-block LINEAR (scale-multiply) dequant at the same or even 2× the bytes. (Apple's official
int4-km-g8 models are fast, but that path isn't reproducible via `palettize_pytorch_model`.)
`quantize_pytorch_model` takes `dtype: "int4"` with per-block granularity — **linear int4
per-block-32 was top-1 EXACT on gemma4** (8/8) but **qwen3.5 is int4-NO-GO at every scheme
tried**: k-means g32 and g8+int8-rescue on the 0.8B, and linear per-block-32 on the 2B
(gate 10/16 and it fails even the cache-seeded single step — transformer/SSM-in_proj damage,
not head damage; also only 156 tok/s vs int8hu's 159, int4-linear dequant underuses BW).
Quantization sensitivity is a model property — gate it per model.
**Ornith-1.0-9B is the qwen3.5-family int4 COUNTEREXAMPLE** (2026-07-03): the same
linear per-block-32 int4 that fails the 0.8B and 2B gates **24/24 exact** on this
RL-trained 9B (fp32-oracle eager teacher-forced rig, incl. 0.205/0.058-margin knife
edges; engine greedy 12/12 ≡ oracle) and measures **58.9 tok/s vs int8hu's 48.3 on
M4 Max (+22%, 7.5 vs 9.8 GB)** — this shape sits in the gemma4/LFM
int4-linear-wins class, not the qwen-2B flat class. Short-context instruments only
(48 gated tokens): per the gemma4-VL 272-token lesson, int8hu keeps the quality
claim until a long-context gate clears int4. **LFM2.5-1.2B is also
int4-NO-GO** (3-probe bisect, 2026-06-11): pure int4lin g32 = 14/16; +int8-linear rescue of
the conv-mixer projections fixes the mid-position flip (15/16) but a **short-context flip at
oracle position 1 survives** conv rescue, early-layer-MLP rescue AND per-block-16 (cos climbs
0.90→0.95, argmax stays a special token) — recovering it would need ~all-MLP int8, which IS
int8lin. The forfeited speed was real: int4lin g32 measured **314 tok/s on M4 Max (+24% over
int8lin's 253)** before failing the gate. Two speed rules from the same sweep: **per-block-16
scales are a slow class on this delegate** (97.6 tok/s vs g32's 314 — 3.2×; stay at block 32),
and int4-linear's BW win is shape-dependent (LFM +24%, gemma4 +24%, qwen-2B ~flat).
The eager quantizer
**silently skips tied weights** — clone the embedding table first if you actually want the
head quantized, and measure + gate before believing it helps — **on the surface that is
actually bandwidth-bound**: the untied int8 head looked like a no-win on the 0.8B *on the
Mac* (204→210, +3% — the Mac pipeline hides the head) but the same change is **+40% on
iPhone (50.3–51.5 → 69.7–74.0)** where the fp16 head was 54% of the per-token read; on the
2B it's +26% on BOTH surfaces (127→161 Mac, 19–21→28–30 iPhone). "No win on the Mac" does
NOT mean "no win on the phone" — re-test head quant on every BW-bound surface.
**Big-vocab heads: quantize with absmax `symmetric`, never `symmetric_with_clipping`**
(RESOLVED 2026-06-11): with the default clipping qscheme the 2B head flips 6/16 oracle
top-1s with a tell-tale signature — one sweep position craters to cos 0.62 while neighbors
sit at 0.999x = outlier head rows clipped, not uniform noise. Plain `symmetric` gates 16/16
at identical speed. **Ship shape = per-block-32 + symmetric** (`int8hu --head-sym`; block32
is the script default). Measured: qwen-2B **161 tok/s M4 Max, 28–30 tok/s iPhone 17
Pro (≥ the CoreML-2B port's ~27)**; qwen-0.8B **210 / 69.7–74.0** — greedy rollouts
token-identical to the fp16-head bundles in both cases.
The transformer body is fine WITH clipping (int8lin gates 16/16 everywhere) — this rule is
specifically about fat-tailed embedding/head tables.
**QAT checkpoints are the one GUARANTEED int4 route** (2026-06-11, gemma4): int4
tolerance is a model lottery (gemma4-PTQ ✓ / qwen3.5 ✗ / LFM2.5 ✗), but Google's
`gemma-4-{E2B,E4B}-it-qat-q4_0-unquantized` releases are bf16 weights *trained* for the
q4_0 grid — and q4_0 IS per-block-32 absmax symmetric int4, the exact recipe class this
delegate runs fast. Re-exported `int4lin` bundles from both QAT checkpoints gate 8/8
(python + engine + device) at unchanged speed (E2B 74.7–78.9 / 30.7 settled device;
E4B-from-QAT is the first E4B port: 53.2–55.8 on M4 Max), so the int4 claim upgrades
from "PTQ that happens to gate" to "int4 ≈ bf16 by design" (Google's wording:
"preserving similar quality to bfloat16"). A/B'd `symmetric` (the literal q4_0 grid)
vs the default `symmetric_with_clipping` on the QAT weights: both 8/8, same speed —
qscheme doesn't matter here; clipping stays the default. Two QAT-checkpoint traps:
(1) they **prune dead weights** — KV-shared layers ship without k_proj/v_proj/k_norm
(base checkpoints carry them unused), so a strict loader must tolerate exactly those;
(2) the PLE table and the oracle are checkpoint-derived — regenerate BOTH from the QAT
weights (the swapped checkpoint is a different oracle; same prompt, healthy margins,
near-identical continuations in practice).
**Per-channel (axis-0) int8 weights are BROKEN on this GPU delegate** (found 2026-06-11
replicating the head lever on LFM2.5): the bundle loads and runs but the quantized matmul
returns garbage (full-model gate 0/16 with cos=nan; minimal head-only graph reproduces it
at multiple vocab shapes, `symmetric` and clipping alike, while the SAME minigraph with
per-block-32 is cos 0.9999x vs torch — torch-level numerics gate 16/16, so it is a
delegate lowering bug, not quantization damage). Historical footnote: the qwen A/B
granularity bisect never actually exercised per-channel — the export script parsed
`--head-quant` without applying it, so both probes were per-block-32 (byte-identical
bundle sizes confirm it); the HF bundles named `*_perchan_sym` contain per-block-32
heads, and every published number stands. The lesson stacks with the int8-km-LUT one:
on this delegate, prefer plain per-block-32 linear for everything until a probe proves
otherwise.
**The head lever replicates across models** (2026-06-11, `int8hu --head-sym` ported to
both export scripts, Mac AND iPhone measured): LFM2.5-1.2B **276.5 tok/s decode on M4 Max
(+9% over int8lin's 253.3) and 44.1–46.6 on iPhone 17 Pro (+15–20% over 38.0–39.6,
~94–98% of the ~47 naive ceiling)**, oracle gate 16/16 + decode PASS, greedy rollouts
token-identical to int8lin on both fixed prompts (python runtime and release
`llm-runner`), device numerics 24/24 ≡ Mac-GPU on all 3 runs; bundle 1.62 GB (+0.13 GB
for the untied int8 head). Granite-4.0-h-1b: gate 16/16 + decode PASS, **Mac-flat
(134.2 vs int8lin's 136.5) but +17–21% on iPhone (30.2–31.3 → 35.4–37.1 typical
settled, 24/24 ≡ Mac ×3 runs)** — the THIRD "Mac no-win ≠ device no-win" confirmation
(after qwen-0.8B and qwen-2B; head ≈ 10% of the per-token read on the BW-saturated
surface). Its natural-prompt greedy forks from int8lin at +7 tokens, inside the
post-<|end_of_text|> filler — the oracle rollout is token-identical; judge by the gate,
not rollout identity.

## Numerics gating (how to judge a quantized bundle)

- Gate on **single-step top-1 vs an fp32 HF oracle**: a teacher-forced S=1 sweep over a fixed
  prompt (each position's argmax vs the oracle's per-position argmax — state noise accumulates
  through the bundle's own caches, which is exactly the path quantization corrupts) plus an
  oracle-cache-seeded decode step. 16/16 required.
- **Never gate on long greedy rollouts vs CPU** — GPU and CPU fp16 fork after ~17 tokens on
  natural prompts from accumulated fp16 noise. "≡ fp16-GPU sequence" is meaningful;
  "≠ CPU rollout" is not.
- Verify the **engine path** too, not just the python runtime: the on-GPU argmax sampler
  reproduced the python-probe sequences exactly in our runs, but it's one `llm-runner
  --warmup exact --warmup-length 1` invocation to confirm.
- Cross-device determinism held in our runs: iPhone GPU sequences were 24/24 token-identical
  to M4 Max GPU on both fixed prompts.
- **Validate the oracle prompt itself**: every oracle position must have a healthy fp32
  top-2 logit margin (we require **≥ 0.1**, computable from the oracle alone before any
  bundle exists). LFM2.5's first prompt had two ~0.012-margin near-ties — statistical
  coin-flips that healthy int8 noise (cos 0.9998) flips and that fp16 passes only by luck;
  a 14/16 there gates nothing. Pick a prompt that clears the margin floor (ours: ≥ 0.40)
  and keep the 16/16 criterion strict.

## Measured end state (Qwen3.5-0.8B, 2026-06-11)

| surface | prefill | decode |
|---|---:|---:|
| M4 Max, ship (int8 + per-block-32 absmax int8 head) | 211.6 | **210.0** |
| iPhone 17 Pro, ship, one-shot runner | 72.0–73.9 | **69.7–74.0** |
| M4 Max, fp16-head `int8lin` | 198.8 | 204.1 |
| iPhone 17 Pro, fp16-head `int8lin` | 51.2 | 50.3–51.5 |
| iPhone 17 Pro, chat app (CoreAIChat Qwen mode = the ship bundle, 200-tok turn) | 69.0 | **62.3–67.0** |

vs the previous best iPhone config (fused int8 Metal-kernel static monolith, 42.5–45.4):
the ship config is **~1.6× with zero custom kernels**, and the same bundle runs on macOS
at 210. CoreAIChat downloads the ship bundle since 2026-06-11 (chat-surface decode
62.3–67.0 vs the int8lin default's 47.9, +30–40%).

## VLM rider: Qwen3-VL (the id-space trick)

A text-only engine can carry a full VLM without engine changes — proven by
Qwen3-VL 2B (models/qwen3-vl/README.md, 187.6 tok/s M4 Max, iPhone numerics 24/24):

- **Vision tower = a separate plain `.aimodel`** run once per image; at a
  fixed grid every positional computation (pos-embed interpolation, 2D
  rotary) bakes into constants.
- **Image embeddings ride the static-input hook** as rewritable owned
  buffers (~3 MB); the host rewrites `<|image_pad|>` ids to extension ids
  `V + slot`, and the graph selects `table[ids]` vs `image_embeds[ids-V]`
  per token. DeepStack adds gather the same way at decoder layers 0-2.
- **Interleaved M-RoPE needs NO position plumbing**: image tokens
  self-locate from (ids, pos) — slot = ids-V, image start s0 = pos-slot,
  (t,h,w) = (s0, s0+slot//W, s0+slot%W) — and post-image text uses
  pos - shift with shift = amount·(pos ≥ start), two [1] i32 static inputs.
  The interleave itself is three constant 0/1 masks over head_dim mixing
  three standard RoPE composite calls (per-frequency-pair selection is a
  valid rotation). Text tokens have t=h=w, so the same graph IS a plain
  text LLM when the embeds are zero and start = 1<<30.
- Pure-attention backbones need NO extra states — the engine's native KV
  pair suffices; only the static-inputs patch is required.

Caveats from this port: the engine pads scalar [1] i32 static inputs to a
64-byte minimum buffer; a dynamic-query (`input_ids [1,s]`) twin exports and
gates in torch but crashes the engine at generate on the macOS-27 beta
(`NSArrayM nil insert`) — ship S=1 static and treat chunked prefill as a
future lever; the python runtime cannot execute dynamic-shaped-output graphs
at all (gate via a static-S=1 twin bundle).

### Second rider: Gemma 4 E2B VL (and what it added to the recipe)

Gemma 4 E2B vision (models/gemma4-vl/README.md, 82.4 tok/s M4 Max / 25.5 iPhone) reuses
the id-space trick on an ALREADY-SHIPPED text decoder — same checkpoint, same
PLE tables — and is simpler than Qwen3-VL (no M-RoPE, no DeepStack, standard
positions). New, generalizable findings:

- **Check the mask config before assuming bidirectional images.** Gemma 3
  attends bidirectionally within an image; Gemma 4 gates it on
  `use_bidirectional_attention` — `None` on E2B/E4B (conventional causal,
  engine-compatible as-is), `"vision"` only on 26b/31b. Verify with a
  runtime mask dump, not just the config (`create_masks_for_generate` has
  per-size branches).
- **Models with per-token side tables (PLE) splice cleanly**: HF computes
  the per-layer rows for the PAD id at image positions, so the in-graph
  gather just indexes `where(ids ≥ V, pad, ids)` — side tables stay
  byte-identical to the text ship; the per-layer projection branch reads
  the spliced embedding (matching HF exactly).
- **Gate quantized bundles at REAL context lengths.** QAT-q4_0 weights are
  trained for the per-block-32 absmax grid; `symmetric_with_clipping`
  mismatches it and the error COMPOUNDS — invisible at a 19+8-token gate,
  but at 272 tokens it flips argmaxes whose fp32-oracle top-2 margin is
  1.0+ and drags prompt-end logits cos to 0.75. Ship `--lin-sym` for QAT
  checkpoints, and adopt the **margin rule**: a flip where the oracle's
  top-2 gap is < 0.1 is a knife-edge tie (fp16 class), not a failure.
- **iOS per-encode scratch-heap ceiling (engine bug, 2 reproducers).**
  MPSGraph plans a ~208 KB per-encode temporary heap; one allocation past
  it aborts the first encode (`allocateMTLBufferFromMTLHeap … exceeds heap
  total 212992` + `Failed to acquire the source buffer for the ViewOp`).
  Triggers: 16-head×512 full-attention (12B dense, any bundle) and the
  gemma4 tbl in-graph table-gather + VL splice. Model-side graph squeezing
  is blind — a 36 KB-smaller dequant chain crashed at the byte-identical
  offset (the savings land in a different heap region). macOS is
  unaffected. Escape: move the offending gather OFF-graph — the VL device
  ship is provider mode (`PerTokenInputProvider` PLE rows, extension ids
  mapped to the pad row host-side) with `image_embeds` still a static
  buffer: **static and per-token extra inputs compose in one engine**.
  Status 2026-08-27: still live on iOS 27.0 (24A5418b) — a fresh
  coreai-torch 0.4.2/coreai-core b2 pf32 export re-AOT'd with coreai-build
  3600.82.1 aborts the first S=32 prefill chunk byte-identically
  (`offset 98816 + size 98304 exceeds heap total 145920`, 2/2). Filed
  upstream as apple/coreai-models#201 (iOS face of the closed #27).
- Device-vs-Mac exactness has a context-length limit: at 272 tokens the
  AOT-h18p fusion lands elsewhere within the runtime's own fp16 noise
  (Mac prompt-end logits cos dips to ~0.9), so a 24-token greedy chain can
  fork at a real-margin synonym deterministically. Gate the prefix + the
  margin rule + a decoded-rollout sanity check instead of demanding
  bit-equal chains.

## What fits next / what doesn't

- **Fits with zero engine work**: pure transformers ≤~4B (KV-only states) — stock engine, no
  patch; the work is the `coreai_models`-style re-author + export.
- **Fits with the existing patch**: hybrids with ≤2 fixed-shape extra states — conv+attention
  (LFM2-class — **verified 2026-06-11 on LFM2.5-1.2B-Instruct, Mac AND iPhone**, the first
  non-Qwen rider: no scan anywhere so the decode graph is loop-free by construction, ONE
  extra conv state `[10,1,2048,2]`; int8lin **253 tok/s** / fp16 162 on M4 Max +
  **38.0–39.6 tok/s on iPhone 17 Pro** (~87% of the naive BW ceiling — BW-saturated, a
  higher fraction than qwen-0.8B's ~60%; cold specialization 6.8 s / warm 1.6 s, no AOT),
  oracle gate 16/16 both, engine path ≡ python 24/24, iPhone ≡ Mac-GPU 24/24 on both fixed
  prompts — needed the two GPU-delegate workarounds above, see
  [`../models/lfm2.5/README.md`](../models/lfm2.5/README.md)), Mamba2+attention (Granite-4.0-H /
  Falcon-H1-class: conv + ssm state = exactly 2 — **verified 2026-06-11 on
  Granite-4.0-h-1b + 350m, Mac AND iPhone, the first SSM-scan rider**: at S=1 the Mamba2
  selective scan is a single recurrence step (the HF `use_precomputed_states` branch),
  loop-free like the GDN step; 1b int8lin **136.5 tok/s** / fp16 103.6 on M4 Max +
  **30.2–31.3 tok/s on iPhone 17 Pro** (~84% of the naive BW ceiling ~37 — BW-saturated
  like LFM2.5's ~87%; cold specialization 5.7 s / warm 1.9 s, no AOT, no memory
  entitlement needed at 1.6 GB; numerics 24/24 ≡ Mac-GPU on both fixed prompts, both
  runs), oracle gate 16/16 both bundles on a margin-clean oracle; 350m ships fp16 at 191 —
  int8 there FAILS the gate (real per-block-32 MLP damage + a 0.0102-margin tie decode
  step, see the margin rule below) *and* is no faster (overhead-bound at 0.7 GB); neither
  LFM2.5 delegate workaround needed — per-layer `SSMState` writes compile fine and NoPE
  attention shows no fp16 amplification, see
  [`../models/granite-4.0-h/README.md`](../models/granite-4.0-h/README.md)), GDN hybrids (qwen3.5 family incl. 2B —
  **verified 2026-06-11, Mac AND iPhone**: same script via `--hf-id Qwen/Qwen3.5-2B`, zero new
  work; ship config adds the per-block-32 absmax int8 head → **161 tok/s M4 Max,
  28–30 tok/s iPhone** (≥ the CoreML-2B port's ~27), oracle gate 16/16, 24/24 ≡ Mac-GPU
  on device; fp16-head int8lin = 127/19–21; needs the increased-memory entitlement on
  phone).
- **Fits with the per-token-inputs patch**: per-token host-gathered inputs — **verified
  2026-06-11 on Gemma-4-E2B, Mac AND iPhone**: PLE rows ride as a `ple_tokens [1,1,35,256]`
  input filled by an mmap provider; unified single KV pair (15 non-shared slots padded to
  head_dim 512, sliding layers as linear KV + SDPA window mask); embed + softcapped head
  in-graph; int4-linear bundle = oracle 8/8 and **70.9 tok/s decode / 85.3 prefill on M4
  Max** vs the zoo's int4km-kernel CLI at 56.6–59 (+20–25%, zero custom kernels). iPhone 17
  Pro (AOT `.aimodelc` — the 2.0 GB-constants graph crashes the ON-DEVICE specializer with
  `LLVM ERROR: Failed to allocate mmap'd buffer`, so compile with `coreai-build … --platform
  iOS --architecture h18p --expect-frequent-reshapes` and point `assets.main` at the
  `.aimodelc`): numerics 24/24 ≡ Mac-GPU, **decode 26.5 avg (25.8–27.6) = +10–20% over the
  22–24 tok/s kernel monolith, prefill 40.5 = +71%**, warm engine load 2.6 s (measure on an
  idle device — install-adjacent runs under-read ~15%). Device decode is BW-bound at the
  prefill floor (1.15 GB ÷ ~47 GB/s effective ≈ 25 ms) + ~13 ms/tok sampler round-trip in
  per-token mode (the PLE gather itself is ~0.1 ms — the serialization fear was mis-aimed);
  next levers are fusing the sampler into the model command buffer or a smaller head.
  **Gemma-4-E4B rides the same path with ZERO model-code changes** (2026-06-11, from the
  QAT checkpoint): the re-author is config-driven and the HF config carries everything
  that differs — 42 layers (full attention every 6th), hidden 2560, **2 KV heads**,
  18 KV-shared → 24 unified slots, `use_double_wide_mlp: false` (E4B's MLP is uniform
  10240 — E2B's shared-layer double-wide is a config flag, and **E4B is clean dense, no
  MoE**, `moe_intermediate_size: null`). Torch ladder 27/27 first run; int4lin 3.7 GB
  gates 8/8; M4 Max 53.2 decode provider / 55.8 `--tbl`; AOT h18p compiles clean even at
  3.7 GB constants. On iPhone only provider mode is arithmetically possible (3.7 GB graph
  + 2.7 GB owned tables would exceed the ~6.4 GB entitled limit).
- **Doesn't fit**: pure-RNN models (no growing KV — the engine's state model assumes one),
  models whose embed+head can't live in-graph, big MoE (different problem).

Apple's `coreai-models` repo is **issues-only** (no PRs), so engine capabilities ship from this
repo as a **patch stack** in [`../apps/`](../apps/), applied in order on a fresh upstream clone
(`git -C coreai-models apply <p1> <p2> <p3> <p4>` — see [`../apps/README.md`](../apps/README.md)):
`coreai-shared-product.patch` → `coreai-pipelined-extra-states.patch` →
`coreai-pipelined-per-token-inputs.patch` → `coreai-pipelined-static-inputs.patch` (each
applies cleanly after the previous and the full stack builds with `swift build -c release`).
If Apple opens the engine, the three capabilities worth upstreaming are: N extra fixed-shape
states, the per-token host-input hook, and the static-input buffer hook (the Gemma-4/PLE
enablers — `EngineOptions.perTokenInputProvider` / `EngineOptions.staticInputBuffers`).
