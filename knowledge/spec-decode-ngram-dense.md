# n-gram speculative decoding on a DENSE decode bundle — measured (2026-08-15)

> Companion to [`spec-decode-design.md`](spec-decode-design.md) (the loop) and
> [`spec-decode-hybrid-verify-design.md`](spec-decode-hybrid-verify-design.md) (the GDN-hybrid
> case). This note is the dense case, measured end to end on Muse-Glimmer-30B `int4hu`
> (M4 Max 40-core, macOS 27.0). Tool: `coreai-models/swift/Sources/Tools/spec-decode/`.
>
> **Read §1 before tuning anything.** The design note's "verify is nearly free at K ≤ 8" is
> right about the mechanism and wrong about the shape of the cost, and the difference decides
> whether a workload speeds up or slows down.

## 0. The dense case is the easy one

A dense (KV-only) decode bundle needs **no state snapshot and no re-anchor forward**. The
graph writes KV at `offset = position_ids.count - input_ids.count` and attends `[0, offset+q)`,
so rows past the accepted prefix are stale but never read, and the next forward overwrites
them. **Rollback == do not advance `processed`.** That is the entire difference between the
~300-line dense loop and the windowed snapshot/tail/re-anchor discipline a GDN hybrid needs.

Prerequisites, both already true of the shipped decode exports:

* the head runs on all positions (no `last_token_only` slice) → `[1, q, vocab]`;
* `input_ids` is **dynamic** in the query dimension. A `--static-ids` bundle is pinned to S=1
  and cannot verify at all — check this first, it is a one-line reason for the whole idea to
  be dead on a given bundle.

**Gate it before building the loop.** Feed q tokens as q sequential S=1 decodes, then as ONE
S=q forward, and compare per-row argmax. Measured here: 8/8 rows identical, max |Δ| on the top
logit **0.0078** — one fp16 ulp. `spec-decode --mode rows` is that gate.

## 1. Verify cost is a STAIRCASE, not a slope

Single-S resolution, 848-token context (`--mode verify-cost --sweep-s`):

| S | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 36.9 | 37.1 | 37.7 | 53.6 | 54.0 | 54.4 | 54.7 | 54.9 | 84.6 | 84.7 | 81.0 |
| c_v | 1.00 | 1.01 | **1.02** | 1.45 | 1.46 | 1.48 | 1.48 | **1.49** | 2.30 | 2.30 | 2.20 |

**Three flat plateaus — S ≤ 3 at 1.0×, S = 4…8 at ~1.47×, S ≥ 9 at ~2.3×** — with the steps
between them costing more than everything inside them. The forward is bandwidth-bound (16.35
GB of weights per pass, ~37 ms of traffic), so extra query positions are free until a step,
then free again until the next. The same plateaus and the same step positions came back at a
228-token context, and a coarse sweep at 76 tokens agrees everywhere it samples — so this is a
property of the graph's shape specialization, not of context length.

That gives a mechanical tuning rule, and it is not "smaller K is safer":

> **Set K to the TOP of a plateau: K=2 (S=3) or K=7 (S=8). Never the bottom.**

K=3 (S=4) is the single worst choice available — it pays plateau 2's full surcharge while
committing fewer tokens than K=7 does for exactly the same price. Measured, on code:
K=2 → 1.69×, **K=3 → 1.48×**, K=4 → 1.63×, K=7 adaptive → 1.96×. A sweep that only samples
powers of two (1, 2, 4, 8, 16) sees a smooth curve and hides all of this. Sweep every S on the
bundle in hand before picking K; a round number is not a reason.

A round pays iff it commits more than `c_v` tokens. So the draft length K sets the bar:

* **K=2 → bar 1.02 tokens/round.** Any drafter better than useless wins.
* **K=7 → bar 1.49 tokens/round.**
* **K=8 → bar 2.30 tokens/round** — K=8 means S=9, one position over the step, so the bar
  nearly doubles in exchange for one more draft token. A weak drafter now LOSES outright.

This is the whole reason "n-gram is a no-op on free chat" is wrong on this stack. It is a
no-op only at a draft length whose cost the drafter cannot clear.

## 2. Measured acceptance and speed, three workloads

256 generated tokens, greedy, batch 1. `off` is the identical loop with drafting disabled
(one S=1 decode per forward), run seconds before each `on` run so thermals cannot flatter it.

tok/s, with the speedup over that row's own baseline in brackets:

| config | free chat | code rewrite | tool calling (ATEM) |
| --- | ---: | ---: | ---: |
| K=1 | 34.21 (1.26×) | 39.26 (1.44×) | 41.10 (1.53×) |
| **K=2** (top of plateau 1) | **36.65 (1.34×)** | 46.34 (1.69×) | **50.19 (1.84×)** |
| K=3 (bottom of plateau 2) | 31.47 (1.15×) | 40.52 (1.48×) | 42.06 (1.55×) |
| K=4 | 31.86 (1.17×) | 44.30 (1.63×) | 43.89 (1.63×) |
| K=8 (over the step, S=9) | **25.99 (0.95×)** | 38.12 (1.40×) | 37.56 (1.39×) |
| adaptive cap 5 | 34.50 (1.25×) | 47.31 (1.75×) | 48.86 (1.82×) |
| **adaptive cap 7** | 34.50 (1.25×) | **53.71 (1.96×)** | 49.97 (1.83×) |
| adaptive cap 8 | 34.97 (1.28×) | 49.11 (1.81×) | 46.37 (1.73×) |

The K=2 and adaptive-cap-7 rows are two off/on pairs each; the rest are one pair. Every row
is a validated run — the machine throttles under sustained load, and a throttled half
corrupts the ratio in either direction, so rows whose spec-off baseline left the cool-state
band (26.0–28.0 tok/s here) are excluded rather than averaged in. Build that check into the
report script, not into your judgement: one discarded row read a flattering **2.10×** purely
because its baseline had collapsed to 15.55 tok/s.

Note the trap: on every workload, tok/fwd rises monotonically with K (free chat 1.30 → 1.47 →
1.57) while tok/s **falls**. Acceptance is the metric everyone reports; it is not the metric
that pays.

## 3. Adaptive draft length wins everywhere, and needs no tuning

Start at K=1; after a fully accepted round K += 2; after any rejection K -= 1; clamp to
[1, K_max]. Capped at 7 it beat, or matched, every fixed K on all three workloads — and the
cap matters as much as the rule: cap 8 lets it reach S=9 and costs it 8% on code (1.96× → 1.81×).

Why it works is worth stating precisely, because "it accepts more" is the wrong explanation:
on the code workload adaptive ran **107 forwards vs K=8's 104 — and finished in 5.21 s vs
6.71 s.** Same forward count, same tokens, 22% less time. It wins by spending its forwards on
CHEAPER SHAPES (mean draft 3.1), not by drafting better — it accepts *fewer* drafts per round
than fixed K=8 does (1.39 vs 1.46). Its floor is the free S≤3 plateau, which is why, unlike a
fixed K, it cannot lose.

A cruder version of the same idea — `--ngram-min 2`, i.e. only draft off a ≥2-token match —
also recovers most of the loss (code 1.69×, tools 1.56×) by leaving the low-confidence rounds
at the free S=1 shape.

## 4. The host path is NOT the bottleneck here

The 27B GDN work found a CPU `[1,S,vocab]` → `[Float]` flatten eating **half** the decode wall.
That trap is avoidable, not inherent: argmax the fp16 rows in place, only the rows you need.
Measured across all 45 runs of this loop: **≤1.3% of decode time** outside the forwards,
drafting and argmax included (e.g. 0.07 s of 7.5 s). If a dense spec-decode loop is slow, the
forward shape is the suspect, not the host.

## 4b. Check the baseline against the shipped runner, not just against itself

A spec-decode harness reimplements the decode loop, so "spec on == spec off" only proves the
harness is self-consistent. Compare the no-draft path against the shipped engine on the same
prompt: same greedy text, and 27.20 tok/s (harness) vs 27.6 tok/s (`llm-runner`, pipelined
engine) — within 1.5%. Without that check, every speedup ratio in this note could have been
measured against a slow baseline of its own making.

## 4c. GPU-solo is load-bearing for the lossless gate, not just for timing

The only lossless FAIL in 46 A/B runs came from an accidental **concurrent** run — two 16 GB
jobs on one GPU — and carried a 21.40 tok/s baseline against the usual 26.9, plus a FAIL flag
with no divergence line, which a single process cannot emit. Solo, the same config passes 4/4.
So: a contended run can produce a record that looks like a correctness failure. Do not debug a
lossless FAIL until you have re-run it alone; and do not report numbers from a run that shared
the GPU, in either direction.

## 4d. Which other bundles this can apply to

Three properties decide it, and all three are readable off the export script — no measurement
needed to rule a model OUT:

| requirement | fails when |
| --- | --- |
| dense, **KV pair only** (2 states) | GDN/SSM hybrids need the snapshot + re-anchor discipline; the ~1.2–1.3× cap read off c_v ≈ 1.67 did not hold (the 27B's own MTP head measured 1.30× on int4 and 1.71× on int8, on code, M4 Max, 2026-08-17) |
| **dynamic `input_ids`** | a `--static-ids` graph is pinned to S=1 and cannot verify at all |
| head on all positions | no `[1, q, vocab]`, nothing to verify against |

Surveyed against this repo's export scripts (2026-08-15):

* **`--static-ids` in the shipped recipe — cannot verify today**: `bitcpm-8b`,
  `nanbeige4.1-3b`, `nanbeige4.2-3b` (plus muse-glimmer's `_s1` variant). Worth flagging
  because these are the **iPhone-class ship shapes** — on-device speculation needs a
  dynamic-ids export first, i.e. a second bundle, not a runtime flag.
* **Hybrid / SSM (4 states) — capped**: `lfm2.5*`, `granite-4.0-h`, `minicpm-v-4.6`,
  `qwen3.5`, `qwen3.6`, `nemotron-h`.
* **Confirmed KV-only exports**: `muse_glimmer`, `bitcpm8b`, `bitvla_llm`, `nanbeige41`.
  Other dense families (`glm-4.7-flash` MLA, `shieldstral`, `youtu` MLA, `minicpm5`) declare
  their states through a different helper and were **not** resolved — check before assuming.

**Hypothesis, not a measurement:** MoE targets may lose the free plateau entirely. The plateau
exists because the forward is bandwidth-bound on weights, and S > 1 in an MoE activates more
experts per forward, so the "extra positions are free" premise is the first thing to break.
One `--mode verify-cost` run on a qwen3.6-27B-class bundle settles it.

And the workload rule bounds it further: n-gram only pays where the continuation is already in
the context. **OCR / ASR / TTS are structurally out** — the output is not in the prompt tokens,
so prompt-lookup has nothing to find and a real drafter is the only route.

## 5. What this does not cover

* **Acceptance is workload-bound, not model-bound.** These three prompts are not a
  distribution. Anyone quoting a single average across an unpublished prompt mix is quoting a
  mix, not a speedup — publish the split.
* **Acceptance also decays with generation length**, on some workloads sharply. Doubling the
  budget from 256 to 512 tokens took code from 1.96× to **1.40×** (ā 1.37 → 0.61) as the model
  left the phase where it was restating the prompt, and free chat from 1.25× to 1.10×. Tool
  calling held (1.83× → 1.85×), because the ATEM protocol keeps echoing prompt values for the whole
  turn — the advantage there is structural, not a front-loaded artifact. **Quote the token
  budget with the speedup, or the number means nothing.**
* n-gram drafts a single linear continuation. Tree/multi-candidate verification needs a custom
  attention mask, i.e. an export change (`knowledge/_specdecode_proto/tree_attn_verify.py`).
* An EAGLE-shaped head reading hidden states from several target layers is the general 3–5×
  answer and needs multi-layer taps out of the exported graph — a new authoring module.
