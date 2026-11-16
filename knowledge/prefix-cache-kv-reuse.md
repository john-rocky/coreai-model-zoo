# Prefix caching (cross-turn KV reuse) — the orthogonal on-device speed lever (2026-07-03)

**Result: turn-2 TTFT 0.23 s vs 23.3 s = 101× at a 4k-token context** (CoreAIChatMac, qwen3-0.6b,
sequential engine, Mac; 99.3% of the context reused). **Lossless — PROVEN**: with greedy (temp 0)
the ON and OFF turn-2 answers are byte-identical. Orthogonal to quant / kernels / MLX-parity — it
beats the "byte floor" because it removes work entirely rather than doing it faster. The win scales
with context length: 15× at 357 tokens, 101× at 4k, and grows further for real RAG/agent contexts.

## Why it exists

Decode tok/s is at MLX parity (byte floor) — that game is over (see `coreai-vs-mlx-speed.md`).
But the zoo's real workload is shifting to **agents / RAG / multi-turn**, where the user-felt
latency is TTFT, and TTFT is dominated by **re-prefilling the shared conversation prefix every
turn**. `CoreAIChatMac/Sources/ChatEngine.swift` was doing exactly the worst thing: `engine.reset()`
+ `applyChatTemplate(full history)` + full re-prefill on EVERY turn — so turn N reprocessed all of
turns 1..N-1 from scratch (system prompt + retrieved docs + history). For a 4k-token RAG context
that is seconds of dead time before the first new token, every turn.

## The mechanism (LCP KV reuse)

The engines already (a) preserve the KV cache across `generate()` calls and (b) prefill only
`input[processedTokenCount...]` (sequential) / the tokens passed (pipelined). The only missing
primitive was a **KV rewind**. `reset()`'s own comment gave the key: *"the KV pair needs no
clearing — attention only reads positions below the new offset."* So a partial trim = just set
`processedTokenCount = length`; positions ≥ length are overwritten before they're ever read.

`InferenceEngine` additions (all uncommitted):
- `func trimKVCache(to length: Int) async -> Int` — rewind toward `length`, return the ACTUAL
  retained prefix (clamped to `processedTokenCount`; the last generated token's KV lags one decode
  step, so retained can be `length-1`). Negative = unsupported (recurrent/SSM state can't be
  reconstructed mid-sequence → caller falls back to `reset()`; guarded on `extraStates.isEmpty`).
  Default (extension): `-1`.
- `var prefixReuseFeedsFullSequence: Bool` — feed contract after trim. Sequential slices
  `input[retained...]` internally → caller passes the FULL running sequence (`true`, the default).
  Pipelined prefills exactly what's passed → caller passes only the un-cached suffix (`false`).

Implemented on `CoreAISequentialEngine` (verified) and `CoreAIPipelinedEngine` (symmetric,
UNVERIFIED — the test model loaded as sequential; pipelined path needs a device gate).

`ChatEngine.send()` now, each turn:
1. `full = applyChatTemplate(history)` (as before).
2. `want = min(commonPrefixLength(full, kvTokens), full.count-1)` — `kvTokens` = the exact tokens
   the engine's KV holds (prompt + streamed generation), tracked across turns.
3. `reused = await engine.trimKVCache(to: want)`; on `< 0`, `reset()` and `reused = 0`.
4. `feed = engine.prefixReuseFeedsFullSequence ? full : full[reused...]`; `engine.generate(with: feed)`.
5. Break at the stop sequence (no drain) so the KV ends at prompt + real answer.

Lossless by construction: KV[0..reused] holds the identical tokens at identical positions whether
reused or recomputed. A/B toggle: `CHATMAC_NO_PREFIX_CACHE=1` forces the old reset path;
`CHATMAC_STATS_LOG=<file>` dumps `PFXCACHE prompt=… reused=… ttft=…` per turn.

## Measured (qwen3-0.6b, Mac, CoreAIChatMac; greedy for the 4k row)

| turn | prompt toks | reused | TTFT ON | TTFT OFF | speedup |
|---|---|---|---|---|---|
| 1 (cold) | 81–3820 | 0 | = OFF | (initial prefill, unavoidable) | 1× |
| 2 @ 357 toks | 357 | 336 | 0.126 s | 1.915 s | **15.2×** |
| 2 @ 4103 toks | 4103 | **4075 (99.3%)** | **0.230 s** | **23.282 s** | **101×** |

**Losslessness proven**: the 4k row ran greedy (`CHATMAC_GREEDY=1`); turn-2 output is byte-identical
ON vs OFF (`PFXANSWER` lines match exactly). The 15× vs 101× gap is the whole point — re-prefill
cost grows with context while reuse cost stays ~flat, so the longer the shared context the bigger
the win. Turn 1 pays the full prefill once (3820 tokens ≈ 22 s on this small model's S=1 sequential
prefill — a separate chunked-prefill lever); turns 2..N no longer re-pay it.

**Multi-turn robustness (3 turns, greedy)**: reuse HOLDS and grows with the conversation —
turn 1 826 tok (cold, 4.40 s), turn 2 reused 826 (TTFT 0.122 s), turn 3 reused 849 (TTFT
0.151 s). Turn 3 reuses turn-2's entire prompt AND turn-2's answer, i.e. prior assistant turns are
reused too (for models whose `reply.content` == the raw generation — most of them; `HarmonyParser`
passes qwen/llama through unchanged). No degradation across turns.

Reuse depth = the longest common prefix. The **system prompt + prior user turns always match**
(template is append-only there), so that prefix — the dominant cost in long RAG/agent contexts —
is always reused. Prior ASSISTANT turns reuse only when the model's raw generation matches the
template's re-render (thinking-stripping / retokenization can diverge); LCP handles the divergence
gracefully (reuse the common part, re-prefill the tail).

## Remaining work / blocked (assessed 2026-07-03)

- **Assistant re-anchoring** (deeper reuse when content is stripped, e.g. gpt-oss harmony): would
  keep reuse O(new tokens) per turn instead of O(distance-to-first-stripped-turn). Needs a
  **prefill-only engine call** (align KV to the canonical rendering without sampling) — the current
  `generate()` always decodes. NOT implemented: narrow benefit (only the harmony/reasoning-strip
  case; qwen/llama/mistral already deep-reuse verified 3-turn) vs a real new engine API + risk to
  the verified path. Do it only if a stripping model becomes a multi-turn priority.
- **iOS CoreAIChat**: currently **single-turn** (`PipelinedBackend.generate(_ prompt:)` templates a
  lone user message; no history accumulation). Prefix caching has nothing to reuse there — it needs
  the app to become multi-turn FIRST, which is a separate product feature, not a caching add.
- **Pipelined engine path**: implemented + symmetric but UNVERIFIED. Can't be exercised on Mac
  (CoreAIChatMac forces sequential; pipelined SIGTRAPs in GrowingLogitsBuffer for these bundles) and
  the iOS pipelined app is single-turn. Verifying it needs either fixing the Mac GrowingLogitsBuffer
  crash or a multi-turn pipelined device harness.

## Scope / honesty

- Verified on the **sequential** engine, which is the ONE **CoreAIChatMac** actually uses —
  `ChatEngine` forces `variant: "coreai-sequential"` (the pipelined variant SIGTRAPs in
  `GrowingLogitsBuffer` for these bundles). So the verified path IS the production Mac-chat path.
- The **pipelined** engine has the symmetric impl but is UNVERIFIED (PipelinedBench / iOS / llm-
  benchmark use it; a device gate with a pipelined bundle would confirm it).
- **SSM/GDN hybrids** (Qwen3.5/3.6 linear-attn) return `-1` from `trimKVCache` → fall back to full
  re-prefill (their recurrent scan state can't be rewound). Pure-attention models get the win.
- The bigger the shared prefix (long system prompt / RAG docs / deep history), the bigger the win.
  Short single-turn chats see nothing (nothing to reuse) — this is a long-context/agent lever.
- All changes uncommitted; ChatView shows a green "N cached" badge when reuse fires.
