# Bonsai port: research and adversarial review reports (2026-09-18)

Two agent reports, kept verbatim; the digest and the fixes are in `bonsai-ternary-hadamard.md` §8.8.

---

# Decode-speed research: what could move us off 42.4 ms/token (23.6 tok/s)

Scope: single-stream (batch=1) greedy decode of Ternary-Bonsai-2-27B through Apple's Core AI
(MPSGraph-based) runtime on an M4 Pro, `bonsai-swift`'s sync loop, one graph call per token.
Current budget: ~30 ms ternary matvec (near the chip's ~223–240 GB/s practical ceiling), ~2 ms
fused GDN/norm/Hadamard kernels, ~3.5–4 ms fixed per-step CPU encode tail inside
`MPSRuntime::evaluateOps` (independent of layer count, not reachable from the graph), remainder
~7 ms of small-dispatch gaps. Pure bandwidth floor is 27.8 ms (36 tok/s). Dead levers per the
task brief: static-shape contract, dispatch-count reduction (small), streamed/async pipelining,
GPU argmax, head truncation, 8-heads/threadgroup GDN kernel.

## Ranked table

| # | Technique | Realistic gain on 42.4 ms | Cost / effort | Keeps token-identity | Risk it's dead on this runtime | Sources |
|---|---|---|---|---|---|---|
| 1 | **Fold Hadamard transform into the matvec threadgroup prologue** (the untested lever named in the brief) | 42.4 → ~40.4 ms (**23.6 → ~24.8 tok/s, +5%**). Ceiling is the ~2 ms the fused Hadamard/norm kernels already cost; folding removes that dispatch entirely rather than shrinking it, so realistic gain is most of that 2 ms, not more. | Medium: one new MSL kernel that does sign+FWHT₁₀₂₄ in registers/threadgroup memory right before the matvec's K-loop, replacing `HadamardSite.transform` + `bonsai_tern_mv128(_x2/_x3)` with a single dispatch. Numerics must still gate bit-for-bit against the existing 4-stage fp32 FWHT → fp16 chain. | Yes — it's a pure kernel-fusion refactor of already-gated arithmetic, no precision change if the fp32 accumulation order is preserved. | Low — this is the one lever the brief explicitly hasn't tried; round 3 already validated that fusing kernel calls (GDN step, norm+FWHT) saved real time (44.4→42.4 ms) by the same mechanism (fewer dispatches / fewer copy-in-copy-out boundaries), so the pattern is proven to work on this runtime. | Internal: `knowledge/bonsai-ternary-hadamard.md` §8.7 (round 3 fusion results), §9. |
| 2 | **Speculative / self-speculative decoding via the existing S=16/64 GEMM as a "verify" entrypoint**, drafted by prompt-lookup (n-gram match against context, zero extra model) or a small paired drafter | Conditional and workload-split. PrismML's own dspark speculative decoding for this exact model family, measured on **Apple Silicon (M5 Max, Metal)**, reports **~1.2× on ternary code/math workloads only; chat/reasoning and the 1-bit family come out *slower* than plain decode, "not recommended on Apple Silicon."** (CUDA/L40S gets 1.8–2.4×, tensor-core GEMM the Metal path doesn't have.) On our M4 Pro, going through Core AI/MPSGraph (extra per-call host overhead vs. their raw llama.cpp-Metal binary) and a slower chip than the M5 Max they measured, expect **≤1.15× (42.4 → ~37 ms, ~27 tok/s) on code/agentic-edit sessions with prompt-lookup acceptance, and roughly break-even-to-worse on freeform chat.** Reasoning below. | High: new static-S GEMM entrypoint(s) for small verify batches (S=4/8, reusing `TernaryLinear128`'s existing `gemm_kernels` dict and `bonsai_tern_gemm128_m{bm}` — kernel already exists for BM∈[16,128]; a true S=4 needs a new BM), a rollback path for the GDN recurrent state (below), and a host-side draft/verify loop in `BonsaiEngine`. | Not automatically — verify-and-accept against the model's own argmax preserves it by construction (same as prefill-vs-walk gating today), but any drafter mismatch on tie-break rounding needs the same care as the current 3-position system-prompt ties. | **Medium-high.** The GEMM kernel (`bonsai_ternary_metal.py`'s tiled loop, plain scalar FMA, no `simdgroup_matrix`) is compute-bound at large S: the measured S=64 chunk is 61 tok/s-equivalent = **~16.4 ms of *compute* per verified row**, not bandwidth-amortized to near-zero. That means a rejected tail token still costs real GPU time, so speedup only appears once average accepted-tokens-per-call is high (a back-of-envelope model: effective tok/s ≈ a / (32 ms fixed + 16 ms·S), which is *worse* than 23.6 tok/s unless acceptance α ≳ 0.9 at S≈16). This matches PrismML's own real-world 1.2×-on-code-only number closely enough to trust it as the calibration point. | PrismML `Bonsai-demo/SPECULATIVE.md` (via GitHub); Apple Silicon (M5 Max) numbers quoted there; internal `bonsai_ternary_metal.py` GEMM source and §8.3 chunk timing (554→61 tok/s-equivalent) in the knowledge note; EAGLE-3 acceptance 0.60–0.88 workload-dependent — [Spheron EAGLE-3 blog](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/); prompt-lookup decoding — [apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding), [zolotukhin.ai on PLD for coding agents](https://zolotukhin.ai/blog/2026-07-24-a-coding-agents-speculative-draft-is-the-context-it-already-read-back/); GDN/Mamba state-rollback problem — [ReplaySSM](https://tridao.me/blog/2026/replayssm/), [TreeWY (arXiv 2608.20961)](https://arxiv.org/abs/2608.20961). |
| 3 | **PTQ1_0 repack (1.75 bpw base-3 trits) instead of PQ2_0 (2.125 bpw) for decode** | If the unpack cost stays free: 30 ms matvec × (1.75/2.125) ≈ 24.7 ms → total ≈ 37.1 ms (**23.6 → ~27 tok/s, +14%**). If the unpack reintroduces an ALU wall (see risk), gain could be **zero or negative**. | High: new trit codec (non-positional bit layout, §3.2 of the knowledge note — 5 trits/byte with a non-trivial byte→trit map), a fresh unpack kernel, a re-export of the whole weight pack, and re-gating token-identity end to end. | Yes, in principle — same ternary values, same scales, exact lossless transcode of PQ2_0 (`mlx:codec.py` already does this both ways), so no new rounding is introduced by the packing itself. | **Medium-high.** Round 1's entire finding was that the *int→float conversion*, not bandwidth or ALU count, was the wall at 2-bit (132→242 GB/s only after switching to `unpack_unorm4x8_to_float`, a hardware unpack instruction that exists for byte-of-8-bit-code, not for 5-trits-per-byte). PTQ1_0's byte layout (`q → trit_n = ((q·3ⁿ)&0xFF)·3>>8` or a 256-entry LUT) has no equivalent single hardware instruction; the knowledge note itself already concludes "the trit unpack is arithmetic the bandwidth-bound decode would pay for nothing on a GPU" (written before round 1 confirmed *why* extra per-code arithmetic is expensive on this GPU). | Internal: `knowledge/bonsai-ternary-hadamard.md` §3.2, §8.5 (round-1 unorm-unpack fix and its cause); general bit-packing-vs-unpack-cost trade-off — [Intel BITCOS / "Breaking the 1.58-bit Barrier"](https://arxiv.org/abs/2609.16338) (their own CPU numbers: on bandwidth-sufficient cores the "fixed 2-bit kernel beat BITCOS on every model," i.e. unpack cost dominated once bandwidth wasn't the bottleneck — exactly our GPU's situation post round-1 fix). |
| 4 | **`MTLIndirectCommandBuffer` replay / captured-command-buffer reuse for the per-step CPU encode** | None reachable by us. The 3.5–4 ms tail is Apple's `MPSRuntime::evaluateOps` re-encoding the graph's ~500 kernel calls each step (confirmed by Metal System Trace + Time Profiler in round 3) — this is *inside* the closed-source Core AI/MPSGraph runtime, not something our export or Swift host can touch. | N/A from our side — this would be an Apple engineering change (Core AI shipping executable/ICB caching across steps of a static-shape graph on a future OS). | N/A | **Dead, definitively**, for the reason above (already confirmed by round 3's own trace, not just an outside claim). ICB replay is a real and effective technique *for hand-written Metal decode loops* — general-Metal-ecosystem reports describe cutting host dispatch from ~0.8–1.5 ms/step to <0.05 ms/step this way — but it requires owning the command encoding, which MPSGraph does not expose. | Apple's own API — [`MTLIndirectCommandBuffer`](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer), [`MTLResidencySet`](https://developer.apple.com/documentation/metal/mtlresidencyset) (real, current Apple Developer Docs); general technique reports (treat as directional, not verified against our exact runtime) describing 24–50× host-dispatch reductions in hand-rolled Metal LLM backends; llama.cpp-ecosystem reports of 26–29% of Metal decode wall time being host-side command encoding/barriers, consistent in kind with our own 3.5–4 ms / 42.4 ms (~9%) — the smaller fraction here is expected since our graph already runs far fewer, larger command buffers (~15 of ~2.5 ms) than a naive per-op encode. |
| 5 | **GDN kernel restructuring beyond the current per-head design (e.g. 8-value-rows-per-simdgroup with register-local reduction instead of one simdgroup per value row)** | Effectively 0. Total GDN budget is already ~2 ms of 42.4 ms; even a 75% reduction in intra-warp communication latency (as claimed for this exact idea in the general Metal-LLM literature) caps out around 1–1.5 ms, and **we already tried the equivalent restructuring** (8-heads-per-threadgroup) and measured it *slower*. | Low to try (a kernel variant), but expected-value is near zero. | Yes if gated. | **High — already dead**, per the brief's own list. The literature's "8-row SIMDgroup packing" proposal is structurally the same idea (fewer, fatter simdgroups per value-row group instead of one simdgroup per row) as our tried-and-rejected 8-heads/threadgroup kernel; nothing found suggests a different mechanism that would flip the result on this chip. | General-Metal-LLM GitHub issue describing the same restructuring idea (cited for pattern-matching only, not verified against our kernel) — treat as corroboration that this shape of optimization is a known idea, not new evidence it works here. |
| 6 | **Vocab pruning / top-k logits shortcut for the 248k-row head** | ~0. Already effectively tested: truncating the head to 8192 rows saved only the head matvec's own ~1.3 ms — the runtime isn't touching the 318 MB head weight beyond that matvec, so a smaller *logical* head wouldn't remove work the runtime is doing elsewhere. Even a perfect prune caps at 1.3 ms (**23.6 → ~24.0 tok/s, +1.7%**), and a prune large enough to matter risks ties resolving differently than the MLX reference. | Medium (need a keep-list, and a guarantee the true argmax is always in it — otherwise it isn't token-identical) if pursued at all. | Only if the keep-list always contains the true top-1, which is workload-dependent and hard to guarantee at 248k vocab; described techniques (vLLM's pruned lm_head) get bit-identical logits only for *kept* tokens, i.e. this needs a proof for our head, not an assumption. | Already effectively dead by our own round-3 probe. | Internal: knowledge note §8.7 (`--head-rows` probe); general technique — [vLLM-Ascend pruned lm_head PR](https://github.com/vllm-project/vllm-ascend/pull/15788) (bit-identical-if-kept design, useful as a reference for *how* to prove correctness if ever revisited). |
| 7 | fp8 / lower-precision head that keeps *greedy argmax* identical but not full logits | Would remove ~half the head's 1.3 ms bandwidth-bound matvec time at best — same tiny ceiling as #6. | Medium | **No** — explicitly breaks bit-for-bit logits even if argmax often survives; disqualified by the current token-identity gate, noted separately per the task's own instruction. | N/A (out of scope while the identity gate holds) | Not separately sourced; same ceiling reasoning as #6 (bandwidth already spent on the head is only 1.3 ms of 42.4 ms). |

## Top 3, mapped onto our graph/host structure

**#1 — Hadamard-in-matvec-prologue.** Today `HadamardSite.transform` runs `bonsai_fwht1024` as its own dispatch (sign flip + FWHT₁₀₂₄ + 1/32 scale, output fp16), and the transformed activation is handed to `TernaryLinear128.forward`, which for S=1 calls `bonsai_tern_mv128`/`_x2`/`_x3` (`bonsai_ternary_metal.py`). The fusion is: extend `_MV_SRC`/`_MV_MULTI_SET`'s per-`kb`-block load (`xr[j] = float(A[k0+j,0])`) to instead read the *un-rotated* normed residual for that K-slice, apply the K-width's sign vector and the relevant 1024-block's Hadamard butterfly (5 simd-shuffle stages + up to 3 threadgroup-memory passes, same math as `bonsai_hadamard_metal.py`'s `kernel_fwht_tg<1024,256>`) in registers/`threadgroup float` before the existing 4-codes-per-instruction unorm-unpack loop, and drop the separate `HadamardSite` dispatch and its site-memo bookkeeping in `bonsai2.py`'s `seed_site`. On the export side this touches `bonsai_ternary_metal.py` (new kernel variant, e.g. `bonsai_tern_mv128_fwht`) and `bonsai2.py`'s `TernaryLinear128`/`BonsaiAttention`/`BonsaiMLP`/`BonsaiGDN` wiring to pass the un-rotated `x` and the site's sign vector straight into the matvec instead of through a `HadamardSite`; nothing changes in `bonsai-swift` (same `main`/`prefill` I/O contract, same states). Gate with `_smoke/bonsai/gate_fused_sites.py`-style comparison against the current two-dispatch chain before trusting it, since the K-loop's per-`kb` structure (512-wide chunks of 32 lanes × 16 codes) doesn't align 1:1 with the Hadamard's 1024-wide blocks, so the threadgroup needs to either process two `kb` chunks per Hadamard block or restructure the K-loop to 1024-wide steps — the real engineering risk is here, not in the numerics.

**#2 — Speculative decoding via the GEMM path.** The pieces already exist: `prefill16`/`prefill64` are static-S GEMM entrypoints, and `TernaryLinear128.gemm_kernels` is already a dict keyed by BM, so adding a small verify size (S=4 or 8) is "export one more chunk size" rather than new architecture — `conversion/export_bonsai2_27b_decode_pipelined.py`'s `--chunks` flag already generalizes this (`prefill16` was added exactly this way in round 2). The hard part is host-side and stateful: `BonsaiEngine` would need (a) a drafter — cheapest is a host-side prompt-lookup hash over `processed` tokens (zero GPU cost, matches `zolotukhin.ai`'s coding-agent PLD result of reusing context the model already read), (b) a verify call through the new small-S entrypoint that returns logits for all S draft positions in one graph call, (c) argmax-compare against the drafted tokens to find the first rejection index `j`, and (d) — the real cost — a rollback of `recState`/`convState` (the GDN chunk-scan state, `[48,1,48,128,128]` fp16 ≈ 75 MB, `[48,1,10240,3]` fp16 ≈ 2.95 MB) to their pre-verify values, since GDN's recurrence (unlike the position-indexed `keyCache`/`valueCache`, which self-heals because future calls simply overwrite stale slots) has already irreversibly folded all S draft tokens into one final state and cannot be sliced back to "as of position j." The practical fix is cheap relative to the ~30 ms matvec budget: snapshot `recState`/`convState` (a plain NDArray copy, sub-millisecond at ~245 GB/s for ~78 MB) before each verify call, and on any rejection, restore the snapshot and re-run a `prefillChunk`/`step` walk of just the accepted prefix + corrected token. Given PrismML's own Apple Silicon number (~1.2× code-only, chat/reasoning slower) as the calibration point, this is worth a half-day spike (build the S=4 GEMM entrypoint, wire a prompt-lookup drafter, measure on the existing chat/code fixtures) before committing to the state-rollback plumbing — if our GEMM kernel's per-row compute cost (~16 ms/row, estimated from the 61 tok/s-equivalent S=64 chunk number) doesn't come down, this lever tops out well short of the naive "S× speedup" intuition.

**#3 — PTQ1_0 repack.** This only makes sense as a follow-on to round 1's unorm-unpack trick, not a replacement: the whole finding of round 1 was that `int→float` (or here, `byte→trit`) conversion, not bandwidth, is the wall, and PQ2_0's win came from finding a *hardware* unpack instruction (`unpack_unorm4x8_to_float`) that maps directly onto its byte layout. PTQ1_0's layout (§3.2 of the knowledge note: element `16n+m` = trit `n` of byte `m`, decoded via `((q·3ⁿ)&0xFF)·3>>8` or a 256-entry LUT) has no equivalent single GPU instruction, so the first step — before touching `bonsai2.py`'s loader or the export pipeline at all — is a standalone microbench of just the trit-unpack loop (same shape as round 1's "standalone Metal microbench of the exact loop" methodology) to see whether it lands above or below the 132 GB/s wall round 1 measured for the *naive* 2-bit path. Only if that microbench beats ~200 GB/s is it worth building `pq1_words_and_scales` (a `PTQ1_0`-analog of `pq2_words_and_scales` in `bonsai_ternary_metal.py`), a new matvec/GEMM kernel pair, and re-exporting the whole 6.7 GB bundle. Given the knowledge note already flagged this exact risk before round 1 even ran, and round 1 then empirically confirmed per-code ALU cost is the dominant term at 2-bit, this is a "test the premise in an afternoon before investing a week" item, not a next sprint.

## Things we probably can't do

- **Any technique that changes the runtime's own per-step encode.** Core AI / MPSGraph is closed-source; round 3's own trace already localized the 3.5–4 ms tail to `MPSRuntime::evaluateOps` / `GPURegionCallOpHandler::encodeOp`, which is Apple's code, not ours. ICB replay, residency-set consolidation, and command-buffer-reuse are all real, well-documented Metal techniques ([`MTLIndirectCommandBuffer`](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer), [`MTLResidencySet`](https://developer.apple.com/documentation/metal/mtlresidencyset)) — but only for code that owns the command encoder, which a Core AI graph does not expose to us. This can only be fixed by Apple, e.g. in a future Core AI release that caches/replays a static-shape graph's dispatch sequence.
- **Cache-resident weights / SLC tricks.** The M4 Pro's system-level cache is tens of MB; the model's ~6.8 GB/token weight traffic can't be made cache-resident by any packing choice, and round 1's own microbench notes flag a 23.7 MB buffer re-read every rep as a *false-positive* "270 GB/s" result from hitting system cache — i.e. we already know real weight traffic doesn't get this for free, and there's no legitimate way to make ~7 GB of ternary weights SLC-resident on this chip.
- **A trained draft model.** The brief allows a small (≤1 GB) Qwen3-tokenizer-compatible draft, but PrismML's own paired `dspark` drafter (~0.6 GB, purpose-trained for this exact target) only reaches ~1.2× on Apple Silicon for code/math and is *slower* on chat — training or fine-tuning an equivalent drafter ourselves would be a multi-week project competing against a result PrismML already shipped and measured as marginal on this hardware class; not worth pursuing ahead of the cheap prompt-lookup spike in row 2.
- **fp8/quantized head, aggressive vocab pruning without a correctness proof.** Both are disqualified while the token-identity gate against the MLX reference holds (explicitly out of scope per the task), and both would only move ≤1.3 ms even if allowed (row 6/7) — not worth the identity risk for the size of the prize.
- **Retraining anything (Medusa/EAGLE-style extra heads, layer-skip self-speculation tuned on Bonsai's own weights).** EAGLE-class methods get their 0.6–0.88 acceptance by training the draft head on the target model's own generations; we have neither the training pipeline nor (per the perf-round rule) budget for that here, and self-speculative layer-skip is reported to degrade sharply past skipping ~half the layers on a comparable-depth model, which would then need per-workload tuning we have no infrastructure for.

Sources consulted (beyond the internal knowledge note and kernel/Swift sources already cited inline):
- [EAGLE-3 acceptance rates / production status](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/)
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding), [PLD for coding agents on Apple/consumer GPUs](https://zolotukhin.ai/blog/2026-07-24-a-coding-agents-speculative-draft-is-the-context-it-already-read-back/)
- [ReplaySSM — SSM state rollback for speculative decoding](https://tridao.me/blog/2026/replayssm/), [TreeWY — speculative verification for Gated DeltaNet hybrids (arXiv 2608.20961)](https://arxiv.org/abs/2608.20961)
- [PrismML Bonsai-demo SPECULATIVE.md](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) — the load-bearing calibration point for row 2
- [Breaking the 1.58-bit Barrier for Ternary LLMs / BITCOS (arXiv 2609.16338)](https://arxiv.org/abs/2609.16338) — unpack-cost-vs-bandwidth trade-off, calibrates row 3's risk
- [`MTLIndirectCommandBuffer`](https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer), [`MTLResidencySet`](https://developer.apple.com/documentation/metal/mtlresidencyset) — Apple Developer Documentation, confirms these are real current APIs (row 4, "can't do" list)
- [vLLM-Ascend pruned lm_head PR #15788](https://github.com/vllm-project/vllm-ascend/pull/15788) — bit-identical vocab pruning design reference (row 6)
- Weschera/Bonsai-2-27B-Mac README (GitHub) and PrismML's own Bonsai-demo/GGUF model cards — corroborate the whitepaper's M4 Pro/M4 Max llama.cpp numbers already cited in the internal knowledge note (§5); treated as directional corroboration only, not as authoritative as the pinned whitepaper/runtime sources the knowledge note already cites.

Note on source quality: several GitHub issue trackers surfaced by search (e.g. an `anthony-chaudhary/fak` repo, an `antonellof/ferrox` repo) read like active performance-engineering backlogs for hand-rolled Metal LLM runtimes and contain numbers consistent with well-known Apple APIs and with our own measurements, but they are not established, widely-cited projects — I've used them only as directional pattern-matching (confirming a technique is a known idea and roughly what order of magnitude it moves), never as the sole basis for a claim in the ranked table above.


---

# Adversarial review — Bonsai 2 27B ternary Core AI port + Swift host

Scope: zoo overlay kernels/loader, the export script, the Swift host. No files modified, nothing
downloaded, no model run. Numeric claims marked **[snippet]** were checked with standalone numpy in
the scratchpad (`chk1.py`, `chk2.py`, `chk3.py`); the rest are **[read]** or **[speculative]**.

---

## Findings, most severe first

### 1. The MLX token-identity gate never exercises the prefill plan the product actually uses
`Sources/BonsaiKit/Gate.swift:58-71` vs `Sources/BonsaiKit/BonsaiEngine.swift:315-323`

`runGate` re-implements its own chunk plan and uses **only the largest entrypoint**:
`let whole = (ids.count - 1) / chunk` with `chunk = engine.chunk` (64). `BonsaiEngine.prefill`,
which `generate` / `generateStreamed` / `chat` / `bench` all call, uses the greedy multi-size plan
`while let size = chunkSizes.first(where: { $0 <= tokens.count - 1 - i })`.

Concrete divergence on the shipped bundle (`metadata.json`: `prefill_chunks: [64, 16]`) and the
125-token lighthouse prompt:
* `parity` runs 64 chunked + 61 walked → touches `bonsai_tern_gemm128_m64` only.
* `chat` runs 64 + 16 + 16 + 16 chunked + 13 walked → also runs `bonsai_tern_gemm128_m16`
  (`BM=16`, `TM=1`, a separately generated MSL instance) and the GDN chunk kernel at S=16 with a
  chunk_max=64 output buffer.

So the claim the whole project rests on — "token-identical to the MLX reference" — is only
established for a path `chat` does not take. The only thing covering `prefill16` is
`parity --self` (self-consistency against the bundle's own walk), and the zoo's own note records
that at **124/125**, not 125/125. Nothing gates `prefill16` against the reference.

Confidence: **verified by reading both code paths + the bundle metadata.**
Fix: make `runGate` call `engine.prefill(...)` (scoring per chunk) instead of its own loop, or add
a `--plan` flag so `parity` can reproduce the shipping plan.

---

### 2. `generateStreamed`'s error path permanently destroys the four state buffers
`Sources/BonsaiKit/BonsaiEngine.swift:382-387, 436-445`

```swift
var k = NDArray(shape: [1], scalarType: .float16), v = k, c = k, r = k
swap(&k, &keyCache); swap(&v, &valueCache); swap(&c, &convState); swap(&r, &recState)
...
} catch { thrown = error }
await stream.currentWorkCompleted()
if let nd = try? await kA.ndArray { keyCache = nd }      // <- nil silently accepted
```

If any `try? await …ndArray` yields nil (the stream faulted, the async value never resolved), the
corresponding slot keeps the **1-element fp16 placeholder** that was swapped in at line 383. Nothing
marks the engine unusable:
* `reset()` calls `zeroFloat16(&keyCache)` whose `count = shape.reduce(1,*) = 1` → succeeds.
* the next `step()` inserts a `[1]` fp16 array as `keyCache` for a `[16,1,4,2048,256]` state.

Aggravating: `v = k, c = k, r = k` on line 382 — unlike `run()` (lines 271-274) which allocates four
distinct placeholders — so if `NDArray` is a handle-carrying value type, all four slots end up
aliasing **one** 1-element buffer.

Also note `processed += encoded` runs even on the error path, so the engine's idea of where it is in
the sequence survives while the state does not.

Confidence: **verified by reading**; triggering needs an encode/await failure mid-stream.
Fix: on any restore failure, re-allocate the states from their descriptors and force `processed = 0`
(or set a `poisoned` flag that every entrypoint checks).

---

### 3. `BonsaiEngine` is `public`, `@unchecked Sendable`, and memory-unsafe under concurrency
`Sources/BonsaiKit/BonsaiEngine.swift:24, 232-246`

The doc comment says "Everything is single-threaded and strictly ordered", but nothing enforces it:
no actor, no lock, `@unchecked Sendable` on a class with six mutable `NDArray` properties plus
`processed` and `lastNextToken`. Worse, `prefillChunk` holds an `inout` into a dictionary element
across a suspension:

```swift
try await withLocal(&prefills[n]!.logits) { out in
    try await run(function, input: input, output: &out, total: processed + n)
}
```

That keeps the `Dictionary` subscript `_modify` coroutine (and the `Optional` force-unwrap `_modify`)
open for the whole GPU run. A second concurrent touch of `prefills` is an exclusivity violation —
a trap in the best case, silent corruption otherwise. Two concurrent `step()` calls would also
interleave the state swap/`defer`-swap-back pairs and leave the placeholders in the slots (the same
failure as finding 2, but without any error to catch).

Confidence: **verified by reading.** It's a library type, so a consumer will hit this eventually.
Fix: make it an `actor`, or drop `Sendable` and document it as non-Sendable.

---

### 4. `--kv` is never validated against the graph's traced bounds or `max_context_length`
`Sources/BonsaiKit/BonsaiEngine.swift:139, 248-252`; `conversion/export_bonsai2_27b_decode_pipelined.py:82-84`

The export pins:
* `position_ids` Dim `min = max(2, query)`, `max = max_ctx - 1 = 4095`
* `k_cache`/`v_cache` seq Dim `min = TRACE_KV_CACHE_SEQ_LEN = 2048`, `max = max_ctx = 4096`

The host:
* resolves **every** negative dim of every state to `options.kvCapacity`
  (`d.shape.map { $0 < 0 ? options.kvCapacity : $0 }`) — fine today, wrong the moment a state gains a
  second dynamic axis;
* `ensureCapacity` only checks `total <= kvCapacity`;
* never reads `bundle.maxContextLength` (4096 in the shipped `metadata.json`) at all.

Two concrete failures from `main.swift:55` (`intOption("kv", 2048)`, no range check):
* `--kv 512` → states allocated **below** the traced KV minimum of 2048.
* `--kv 4096` → `ensureCapacity(4096)` passes, but `position_ids` of length 4096 exceeds the traced
  max of 4095. A one-off at exactly the context limit.

Also `intOption` silently falls back to 2048 for a non-numeric or missing value.

Confidence: **verified by reading both sides + metadata.json.**

---

### 5. Resolved: the S=1 trace must accept a length-1 `position_ids`
`export_bonsai2_27b_decode_pipelined.py:82`; `BonsaiEngine.swift:269`

The original trace used `torch.export.Dim("seq_pos", min=max(2, query))`, which set a minimum of 2
for `main`, while `run()` builds `[1, total]` with `total = 1` on the first walked token. Reached
by a one-token prompt, `--walk`, `parity --walk`, and `selfCheck`'s walk arm. The exporter now uses
`min=query`, so the S=1 contract accepts the first length-1 position tensor.

Confidence: **verified in the current exporter and host gate.**

---

### 6. Two different argmax tie-breakers, and the gate only validates one of them
`Sources/BonsaiKit/Runtime.swift:79-88`; `BonsaiEngine.swift:461, 470`

`Logits.top2` uses strict `>`, so the **lowest index** wins a tie. `generate` uses the *host* argmax
for the first generated token (`pre.last.argmax(row: 0)`) and the *graph's* `next_token`
(`logits.argmax(dim=-1)`, Core AI's `reduce_index`) for every token after that. `runGate` and
`selfCheck` — i.e. the entire numerics gate — use only the host scan.

This matters because the logits are fp16. **[snippet]** fp16 ULP at |logit| = 15 is **0.0078**; the
reference's top-2 margins at the three known misses are 0.002 / 0.005 / 0.024. Exact fp16 ties are
therefore routine at exactly the positions the gate already flags, and Core AI's `reduce_index` is
not documented to break them the same way the host loop does. `chat --compare` does not catch this:
both arms use the graph argmax after token 0.

Confidence: tie frequency **[snippet]**; the divergence itself **[read]** (would need a run to observe).
Cheap check: add a `--no-next-token`-style host/graph argmax A/B to `parity`, or have `generate`
assert `lastNextToken == logits.argmax(row: 0)` under a debug flag.

---

### 7. `A_log` is round-tripped through fp16 log space, amplifying the decay error ~2.2×
`bonsai2.py:449` — `g.A_log = param(torch.log(-ssm_a)[perm_h])`, `param` casts to fp16.

The GGUF stores `ssm_a = -exp(A_log)` in F32. The loader takes `log()`, rounds to fp16, and every
step the kernel computes `exp(float(ALOG[hh]))` again. **[snippet]** over A ∈ [1,16]:

| | mean rel err | max rel err |
|---|---|---|
| `exp(fp16(log A))` (what the port does) | 3.7e-4 | 9.8e-4 |
| `fp16(A)` stored directly | 1.7e-4 | 4.9e-4 |

2.2× worse, for nothing. It lands on `ge = exp(-exp(A_log)·softplus(a+dt_bias))`, the one factor that
multiplies the recurrent state **every token**, so the error is systematic and compounds: at |g| ≈ 0.1
it is ~3.7e-3 of drift in the decay over 100 steps. The three gate prompts are ≤165 positions, so
long-context behaviour is untested. The MLX reference keeps these in fp32.

Fix (one line): store `-ssm_a` (i.e. `exp(A_log)`) in fp16 and drop the `exp` in the kernel, or keep
`A_log` at fp32. Same argument applies to `dt_bias` and the `in_proj_a/b` weights, which the note
says are BF16 in the GGUF and fp32 in the reference but are fp16 here.

Confidence: error magnitudes **[snippet]**; the downstream effect on tokens **[speculative]**.

---

### 8. `main` (fused) and `prefill` (chunk) disagree on the dtype of the q/k L2-norm reduction
`bonsai2.py:114-115` vs `bonsai_gdn_step_metal.py:114-118`

```python
# BonsaiGDNChunk (prefill)
def l2norm(x): return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)   # x is fp16
```
```c
// _GDN_STEP_SRC (decode)
qsh[c] = act[0];                       // fp32 copy of the fp16 conv output
for (uint d = 0; d < DK; ++d) sq += qsh[d]*qsh[d];   // fp32 reduction over 128
```

The chunk path emits an `aten.sum` over a **fp16** tensor; whatever Core AI lowers that to is not the
kernel's fp32 tree. Combined with the other known fused-vs-chunk differences (the chunk kernel keeps
the recurrent state in fp32 across all 64 tokens while the walk rounds it to fp16 every token; the
q-scale is fp32 in `_gated_delta_step` but fp16 in both the chunk path and the fused kernel), this is
a plausible contributor to round 3's prefill-vs-walk worst |Δ| jumping 0.055 → 0.32.

Note the zoo's `_gated_delta_step` ("step" mode) is a *third* variant here — it does the q-scale in
fp32 — so the three GDN modes are not mutually bit-comparable. `gate_gdn_step.py` compares the
kernel against `gdn_step_reference`, which shares the kernel's choices, so the gate cannot see this.

Follow-up on 2026-09-19: the Swift host's bounded ten-prompt battery made the difference live.
Chunked prefill and the S=1 fused walk agreed on 4,347/4,405 teacher-forced prompt argmaxes;
newline, emoji and mixed-script repetition produced the most drift. The same final states still
agreed on 80/80 continuation tokens (eight per prompt, teacher-forced after a miss), and all three
MLX fixtures retained identical 24-, 48- and 40-token generations. This is measured internal
numeric drift, not yet a demonstrated generation failure; matching the reduction dtype remains a
candidate for a future re-export, not a release claim.

Follow-up on 2026-09-20: q/k reduction was not the whole explanation. The chunk GDN kernel keeps
its recurrent state in fp32 until the end of a chunk, while the S=1 path writes fp16 state after
every token. On a 4-layer same-GDN-path control, deliberately rounding chunk state after each token
changed S=64 from 418/419 to 419/419. On the full 64-layer model, state rounding changed the
shipping fused-main result from 399/419 to 407/419; aligning the GDN path as well reached 413/419
at S=64 and 410/419 at S=16. Every control retained 8/8 continuation agreement. The non-monotonic
S=16/S=64 residual points to shape-dependent reduction order elsewhere in the chunk graph; q/k
normalization is one known contributor, not an isolated root cause. Per-token state rounding is a
diagnostic, not a recommended shipping change: it discards the chunk path's higher-precision
recurrence and adds work without producing full-model exactness.

Confidence: **verified by reading and exercised by the host battery**; user-visible impact **not observed**.

---

### 9. The host trusts positional ordering for inputs/outputs while checking states by name
`Sources/BonsaiKit/BonsaiEngine.swift:105-117, 152`

```swift
names.inputIds  = descriptor.inputNames[0]
names.positionIds = descriptor.inputNames[1]
names.logits = descriptor.outputNames[0]
...
self.nextTokenName = descriptor.outputNames.count == 2 ? descriptor.outputNames[1] : nil
```

States are validated by name a few lines below, inputs and outputs are not. `metadata.json` carries
`next_token_output: "next_token"` and `BonsaiBundle.nextTokenOutput` parses it — but the engine never
uses it. If the runtime ever reports outputs in a different order the host reads the 248320-wide
logits tensor as the `[1,1]` `next_token` (`readInt32` takes `p[0]`) and decodes garbage without an
error. One `guard` on the actual names would close it.

Confidence: **verified by reading.**

---

### 10. No bounds check on token ids into the packed embedding
`bonsai_hadamard_metal.py:195` (`const uint row = uint(IDS[s]);`), `PackedEmbedding.forward:278-286`

`row` indexes a 248320-row table with no clamp anywhere in the chain (tokenizer → `[Int32]` →
`input_ids`). An out-of-range or negative id is an out-of-bounds GPU read. This is reachable in
practice through the `--head-rows` debug path (`bonsai2.py:403-406`), which truncates `lm_head` and
rewrites `config.vocab_size` but leaves the embedding table and the tokenizer at full vocab — any
generated token ≥ `head_rows` would then be fed straight back in. Also reachable from a
tokenizer/bundle mismatch.

Confidence: **verified by reading.**

---

### 11. Code 3 is silently weight `+2` everywhere, and nothing validates the pack
`bonsai_ternary_metal.py:72-75` (matvec), `:168-169` (GEMM), `bonsai_hadamard_metal.py:203-204` (embed),
`:217` (`dequant_reference`), `pq2_words_and_scales:191-204`

Good news first: **[snippet]** `255.0f * unpack_unorm4x8_to_float(c)` is **exact** for c ∈ {0,1,2,3}
(0.0/1.0/2.0/3.0, bit-exact), so the matvec maps code 3 → `3 - 1 = +2`, identical to the GEMM's
`float(q-1)`, the embed kernel, `dequant_reference`, llama.cpp's `(q-1)·d` and MLX's `code·s - s`.
**There is no divergence.** But there is also no detection: `pq2_words_and_scales` validates only the
byte count, so a corrupt or mis-parsed block would produce silently-wrong weights rather than an error.
Given the design note verified "no code 3" on only the first 20,000 rows of `output.weight`, a
one-time histogram over the whole pack (or a cheap `(w & (w>>1) & 0x55555555) == 0` assert at load)
would be worth having.

Confidence: exactness **[snippet]**; consistency **[read]**.

---

### 12. Latent: `TernaryLinear128` silently computes only row 0 under a dynamic query length
`bonsai_ternary_metal.py:324-335`

```python
if isinstance(s, int) and s != 1:   # -> GEMM
    ...
else:                               # -> M=1 matvec
```

A `SymInt` `s` falls into the `else`, whose MSL only ever reads `A[k, 0]` and writes `C[n, 0]`, while
`result_shapes=[[s, self.N]]` claims `s` rows. Rows 1..s-1 would be uninitialised — no error, wrong
logits. Today `decode_spec` sets `"input_ids": None` so `s` is always a Python int, so it cannot
trigger; any future dynamic-query export would hit it. A `raise` in the non-int branch costs nothing.

Confidence: **verified by reading.**

---

### 13. Latent: `last_token_only` would double-apply the head Hadamard transform
`qwen3_5.py:636-638`; `bonsai2.py:240-241`

`BonsaiModel.forward_stateful_core` returns the **already-rotated** head activation and seeds
`h_head._memo = (y, y)`. `Qwen3_5ForCausalLMStateful.forward` then does
`if self.last_token_only: h = h[:, -1:, :]` **before** `self.lm_head(h)`. The slice creates a new
tensor, the memo misses on identity, and `TernaryLinear128.forward` runs a **second** signed FWHT on
an already-rotated vector — garbage logits, no error. `last_token_only` is False and never set today.
The `seed_site` contract (identity-keyed memo, `object.__setattr__`) is invisible to anyone editing
that line; an assert on the flag in `load_bonsai2_from_gguf` would document it.

Confidence: **verified by reading.**

---

### 14. Streaming output mangles multi-byte UTF-8
`Sources/bonsai-swift/main.swift:155` — `let printer: (Int32) -> Void = { t in print(tok.decode([t]), …) }`

Byte-level BPE routinely splits a UTF-8 code point across two tokens; decoding each token in
isolation yields replacement characters. Cosmetic, but it makes `chat` unusable for anything
non-ASCII. Buffer the ids and diff the decode of the running prefix instead.

---

### 15. Smaller things
* **EOS is emitted.** `generate:464-466` / `generateStreamed:421-424` append the stop token to
  `gen.tokens` and pass it to `onToken` **before** breaking, so `<|im_end|>` is printed and is part of
  the final `tok.decode(gen.tokens)`.
* **Context overflow aborts the process.** `ensureCapacity` throws → `generate` propagates →
  `main.swift:175` `fail("generation failed: …")` → `exit(2)`, discarding everything generated so far.
  A `stoppedOnCapacity` flag would be kinder.
* **`prefill` traps on an empty prompt.** `precondition(!tokens.isEmpty)` (`:312`) is live in release.
* **Chunk-plan edge cases** (all correct, one suboptimal): 1 → no chunk, 1 walked step at
  `position_ids` length 1 (finding 5); 64 → `16+16+16` + 16 walked, *not* one 64-chunk, because the
  plan reserves the last token; 65 → one 64-chunk + 1 walked; 80 → 64 + 16 walked; 81 → 64 + 16 +
  1 walked; 125 → 64+16+16+16 + 13 walked (matches the note).
* **~257 `HadamardSite` instances** each `register_buffer("signs", …)` — harmless, because
  `signs.to(torch.float32).contiguous()` is a no-op on an already-fp32 contiguous tensor, so all sites
  of one K width share the *same* tensor object and dedup to one constant. Worth a comment; a future
  `.clone()` there would add ~10 MB of duplicated constants.
* **`HadamardSite._memo` holds a strong reference to the last traced tensor pair for the lifetime of
  the model** — keeps a FakeTensor from the `main` trace alive through the `prefill` traces. Harmless
  (and in fact what makes identity-keying safe against id reuse), but unintentional.
* **Stale type annotation**: `hadamard_signs_from_gguf` is annotated `-> tuple[int, dict[...]]` but
  returns three values (`bonsai2.py:299, 320`).

---

## Verified OK (things I checked that hold)

**The unorm-unpack matvec trick**
* `255.0f * unpack_unorm4x8_to_float(c)` is bit-exact for c = 0,1,2,3 **[snippet]**.
* `(packed >> 2j) & 0x03030303` maps byte 0..3 to codes j, j+4, j+8, j+12 for every j = 0..3,
  including j=3 where the top byte is bits 30..31 (bit algebra).
* Full-kernel emulation at K=5120, N=64, realistic fp16 activations and 0.004–0.027 scales
  **[snippet]**: rel err vs fp64 **1.6e-5**, against **1.5e-5** for the plain fp32
  dequant-then-matmul reference — i.e. the `255·Σ(x·c/255) − Σx` cancellation costs essentially
  nothing, and **0 of 64 outputs flip** at the fp16 store. The accumulation-order change is well
  inside the fp16 output rounding (~5e-4).
* One fp16 rounding at the store (`C[...] = TYPE(tot)`), matching the MLX chain (fp16 in, fp32
  accumulate, fp16 out).

**Kernel invariants**
* **16 | 128**: a matvec lane's `k0` is a multiple of 16, so its 16 codes lie entirely inside scale
  group `k0 >> 7`.
* **64 | 128**: the GEMM's `k0` is a multiple of 64, so `k0 >> 7` is constant over `[k0, k0+64)` for
  both the even and odd sub-block — `D[k0>>7, n]` never straddles a group.
* **K % 512** (matvec step) and **K % 64** (GEMM step) for every Bonsai K: 5120 = 10×512,
  6144 = 12×512, 17408 = 34×512. ✓
* **N % 32** (`R·SGY`) and **N % 64** (`BN`) for every Bonsai N: 12288, 1024, 10240, 6144, 17408,
  5120, **248320 = 64 × 3880**. The 248k-row head has **no tail** — every threadgroup is full, and
  `TernaryLinear128.__init__` would have raised otherwise.
* **Paired matvec set boundaries** are multiples of 32 rows in every pairing shipped
  (`10240|6144`, `17408|17408`, `12288|1024|1024`), so no simdgroup (4 rows) or threadgroup (32 rows)
  straddles two weight sets, and the `if (row < N)` branch is simdgroup-uniform (`row` does not depend
  on `lane`) — `simd_sum` inside it is safe.
* GEMM index algebra: `xs`/`ws` tiling, `tm`/`tn`, the `BK*BM` cooperative A-load, and the
  `C[n_base + tn*TN + j, tm*TM + i]` store are all consistent for both BM = 16 (TM=1) and 64 (TM=4);
  the barrier pairs around each K-step are correct.

**Hadamard**
* `hadamard_matrix`'s xor-fold parity equals popcount parity for n = 1024, and `H @ H == 1024·I`
  **[snippet]** — natural (Sylvester) order as the note requires.
* Butterfly index algebra: element-index bits 0..4 in the simd lane (`simd_shuffle_xor`), bits 5..7
  in the threadgroup-memory stage (`t ^ i`), bits 8..9 in the register index (`step = i/NT`) — a
  complete 1024-point transform, with the ±1 sign and the 1/32 (both exact in binary) folded into the
  load and a single fp16 rounding at the store.
* Embedding inverse is transform-then-sign (`Y = TYPE(reg * SG[e])`), the reverse of the forward
  site, matching note §4.2 and `_embed_torch_defn`.
* In the fused norm kernel all 5 threadgroups compute the sum of squares over the *whole* K in the
  *same* order, so every 1024-block is scaled by a bit-identical `inv` — no block-to-block seam.

**GDN**
* Delta-rule algebra matches `_gated_delta_step` exactly: decay first, then `kv = Sᵀk`,
  `δ = (v − kv)·β`, `S += k δᵀ`, `o = Sᵀq` with the *updated* S. State layout `RS[c,d,hh]` = torch
  `S[head, k-dim d, v-dim c]` is consistent between the fused step, the chunk kernel and the torch
  reference, and both kernels agree on "state after token t".
* Softplus threshold 20 matches `F.softplus`'s default; `ge = exp(−exp(A_log)·softplus(a+dt_bias))`
  matches the zoo's `g` then `g.exp()`.
* GVA mapping `kh = hh / REP` matches both `repeat_interleave(rep, 0)` and the zoo's
  `expand+reshape` (value head h ← key head h//rep).
* Conv step: taps 0..2 on the conv state, tap 3 on the new input, reproducing
  `F.conv1d(cat([conv_in, mixed]), …, padding=0, groups=conv_dim)` at s=1; new state `[cs1, cs2, x]`
  reproduces `w[..., -(k-1):]`. q/k conv-state rows are written exactly once (`hh % REP == 0` covers
  key heads 0..15 via hh = 0,3,…,45); v rows once per head. `KEYD = (10240 − 6144)/2 = 2048`,
  `NK = 16`, `REP = 3` all derive correctly from the extents.
* Threadgroup-memory discipline in the fused step is correct: `qsh`/`ksh` are reused three times
  (a/b partials → raw q/k → normalised q/k) and every reuse is fenced by a barrier, including the
  often-missed one after the last *read* of the previous contents (line 108, line 119). The a/b
  `simd_sum` reduction over 4 simdgroups of a 128-thread group is correct (`c & 31`, `c >> 5`).
* `BonsaiGDNChunk`'s `G[hh, t]` / `BETA[hh, t]` rewrite is right for the `[S, h]` layout it hands
  over, and `g[0].transpose(0,1)` really is contiguous given that the zoo produced `g` by
  transposing a contiguous `[b,s,num_v]`.
* The chunk kernel's `OUT` is `[h, chunk_max=64, dv]` for *both* prefill entrypoints; rows ≥ S are
  uninitialised but are removed by a static `out[:, :S, :]` slice before anything reads them.

**Rounding chains (fused kernels vs the graph they replace)**
* `bonsai_add_norm_fwht` / `bonsai_norm_fwht` reproduce `RMSNormImpl` + `RMSNormPlusOne` exactly:
  fp16 residual add, fp32 squares/mean/rsqrt, fp32 `(w + 1)` gain, **one** fp16 rounding
  (`RMSNormImpl` takes the `scale.dtype == float32` branch, which also rounds once, at the end).
* The fused gated RMSNorm matches `RMSNormGated`: round to fp16, multiply by the fp16 gain, round,
  multiply by `silu(z)` computed in fp32, round.
* `bonsai_swiglu_fwht`'s `silu(gate)` → fp16 → `× up` → fp16 matches the graph's materialisation
  points, and `U`/`G` are wired to `up_proj`/`gate_proj` in the right order (MLP is
  `down(up · silu(gate))`).

**Graph plumbing**
* Residual `(h, r)` passing reproduces `Qwen3_5DecoderLayer`'s `h = x + r; return h + mlp(post(h))`;
  the first layer's `r_in is None` correctly treats the embedding as the residual; the final
  `add_norm_fwht` + `seed_site(head_site)` correctly replaces `norm` + the head's site.
* `seed_site` / memo identity is safe: each layer owns its own `h_in` / `h_mlp` / `h_down` /
  `h_out`, the memo holds a strong reference so ids are never recycled, and the `prefill` traces
  (`use_fused_sites = False`) miss the memo on a fresh tensor and recompute.
* `set_next_token_output` patches the *class* forward with `_ORIG_FORWARD.setdefault`, and the
  trace-time switches are flipped inside the lazy `export_fn` closure, not in the entry loop — which
  is the documented trap and is handled.
* `_vperm` is the correct tiled→grouped permutation (grouped `kh·rep + r` → tiled `r·nk + kh`),
  applied to exactly the seven tensors note §6 lists (`attn_qkv` V rows via `conv_perm`,
  `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d` V channels), with
  `ssm_out` and the per-channel `ssm_norm` correctly left alone. Row permutation of a PQ2_0 tensor is
  safe because the fold is along K.
* `norm_gain` subtracts 1 from exactly the five `RMSNormPlusOne` gains and not from `ssm_norm`.
* The GDN's `in_proj_a` / `in_proj_b` read the **un-rotated** normalised vector in both the fused
  path (`nrm` from the site kernel) and the unfused path (the site transform lives inside
  `TernaryLinear128`, not in front of it).
* Signed-int reinterpretation of the uint32 words is harmless: `int64(int32(w))` keeps bits 0..31, and
  every `(p >> 2j) & 3` reads bits below 32.
* SDPA's default `causal_variant` is `lower_right` — the correct alignment for a 64-token query block
  against a longer KV cache.

**Host positions / bookkeeping**
* `position_ids = [0, total)` with `total = processed + n` gives `offset = seq_len − query_len =
  processed` in `forward_stateful_core`; the KV write at `[offset, offset+n)` and the fetch
  `narrow(-2, 0, seq_len)` both stay inside `kvCapacity` whenever `ensureCapacity` passes. **No
  off-by-one in the decode/prefill position math** (the only off-by-one is finding 4, at
  `kv == max_ctx`).
* `BonsaiModel.forward_stateful_core`'s `offset = seq_len - 1` is the correct specialisation of the
  general `seq_len - query_len` at query_len = 1.
* `fillInt32` writes from index 0 regardless of the `ArraySlice`'s `startIndex` — the classic
  slice-index bug is not present.
* The streamed loop produces exactly the same token count and the same `processed` as the sync loop,
  including the one extra encode discarded on stop (traced by hand for maxNew = 2 and 3).
* `Logits.top2`'s runner-up tracking is correct (old best demoted on a new best).
* `Logits` never outlives its buffer: `withLocal`'s `defer` swap-back runs before the `Logits` is
  constructed, and `selfCheck` copies via `floats(row:)` before the next step.
* The `prefills` dictionary, the `prefill` / `prefill16` naming and `metadata.prefill_chunks` line up,
  and `guard pin.shape == [1, size]` would catch a mismatch.
