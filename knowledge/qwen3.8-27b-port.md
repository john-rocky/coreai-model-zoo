# Qwen3.8-27B — port knowledge

`Qwen/Qwen3.8-27B` (Apache-2.0, weights landed 2026-08-14) is the Qwen3.8 generation's only
open compact model — a dense 27B **native VLM** with the Qwen3.5 hybrid text decoder. This
port ships the **whole VLM**: the text decoder (phase 1, exactly as
[qwen3.6-27b](../models/qwen3.6-27b/README.md) did for the previous generation) plus the
**vision path** (phase 2, below — the family's first vision authoring). The phase-1
content is what a **same-architecture generation bump** costs (nearly nothing, if you
verify it really is one) and what a **release-day download** costs (a lot, unless you
route around the CDN).

## The port was a weight swap, verified

Before writing any code, diff two things against the shipped sibling:

1. **`config.json`** — all 34 `text_config` keys AND the full `vision_config` are
   byte-identical to `Qwen/Qwen3.6-27B`. Same 64L/5120h hybrid 3:1, GDN GVA 48v/16k,
   full-attn 24q/4kv hd256, untied 248320 vocab, 262K ctx.
2. **The safetensors weight map** — the text key SET (`model.language_model.*` +
   root `lm_head.weight`, 851 keys) is identical; only the values changed. (3.6-27B was
   *also* a VLM checkpoint with `model.visual.*` + `mtp.*` — the loader has skipped those
   prefixes since June, so nothing new fires.)

With both identical, the whole conversion is
`export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id Qwen/Qwen3.8-27B` and the
GVA/untied-head/loop-free findings from the 3.6-27B port transfer wholesale. This is the
session-boundary rule paying out: same authoring module ⇒ same recipe ⇒ hours, not days.

## Release-day download: HF crawled, ModelScope did not

The weights were <24 h old with 2 downloads. Measured from this (Tokyo) network:

- HF CDN (`us.aws.cdn.hf.co`): **~0.15 MB/s per connection** — even for the *warm*
   two-month-old 3.6-27B repo — and a per-IP aggregate ceiling of **~2.7 MiB/s** no matter
  how many connections (curl ×3, aria2c ×16, aria2c j5×16 all plateau there). An
  authenticated token changes nothing. 67 GB ⇒ ~7 h.
- XET (`hf download` default): **stalled at 0 bytes for 60 s+** — the known stall bug,
  reproduced live. `HF_HUB_DISABLE_XET=1` remains mandatory on this network.
- **ModelScope** (`modelscope.cn/models/Qwen/...` — Qwen releases publish there
  simultaneously): ~1 MB/s per connection, scales linearly;
  `aria2c -i urls.txt -j3 -x6 -s6` sustained **~20 MB/s ⇒ 55 GB in <1 h**.

Trust is not delegated to the mirror: every shard is **sha256-verified against HF's LFS
oids** (from `/api/models/<id>/tree/main`) before being seeded into the hub cache as
`blobs/<oid>` + snapshot symlinks (`_qwen38_aria/seed_cache.py` in the workspace). The
loaders call `snapshot_download(hf_id)` internally and hit the seeded cache without
re-downloading. Blob name MUST be the LFS oid — it is the etag the client validates.

## Oracle: transformers 4.57 cannot load this family — build a 5.x venv

The export venv (4.57) has no `transformers.models.qwen3_5`; the config shim only covers
config parsing for the overlay. For the HF-side oracle a dedicated venv was created
(`~/.venvs/qwen38-oracle`: transformers 5.12.1 + torch 2.13) rather than touching any
existing lane's venv. Two 5.x-era gotchas:

- `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`, not a tensor —
  take `enc["input_ids"]` (or `.shape` dies with a misleading `KeyError: 'shape'`).
- The default chat template opens a `<think>` span: greedy continuations begin with
  reasoning-register text ("We need answer user: …"). Normal; budget max-tokens in apps.

Oracle = bf16 (fp32 for 27.8B is ~111 GB RAM — off the table on a 128 GB host), greedy 16,
full logits rows saved so the eager gate can report per-position cos.

## Gate results (cleaner than 3.6-27B)

Teacher-forced single-step argmax vs the bf16 oracle, margin≥0.1 rule, eager fake-quant on
CPU (never `AIModel.load(...gpu())` on a multi-GB graph — its fp16/ANE fallback returns
garbage):

- **fp16 control: 16/16 exact.** The 3.6-27B's pos-4 bf16-resolution artifact has no
  analogue here — full precision reproduces the oracle everywhere, so any quant flip would
  have been attributable cleanly.
- **int8hu (ship recipe): 15/16, zero confident flips.** The one miss is a genuine
  knife-edge (margin 0.061 < 0.1), cos range 0.9977–0.9999. Same verdict as 3.6-27B: the
  absmax int8 untied head + int8 per-block-32 body are quality-transparent.

## The vision path (phase 2, same-day follow-up session)

`model.visual.*` (333 tensors, 458M) + an embeddings-input decoder variant ship as the
combined release. Design: fixed-grid one-shot tower (`qwen3_5_vision.py`, 512×512 tile →
256 merged tokens, positional constants baked in the processor's merge-block-major order,
**no deepstack** — `deepstack_visual_indexes: []` makes this strictly simpler than the
Qwen3-VL tower it was patterned on) + `Qwen3_5VLStatefulEmbeds` (same hybrid graph,
`inputs_embeds` input, three host-fed interleaved-mRoPE position planes, multifunction
S=1 decode / S=16 chunked prefill). Host contract in NumPy: `_smoke/qwen38vl_preprocess.py`
(byte-equal to the HF processor) + `_smoke/qwen38vl_host.py` (mRoPE planes + embed splice).
Full gate chain in `models/qwen3.8-27b/gate-qwen3.8-27b-vl-suite.json`: tower fp32 cos
1.000000, eager mixed text+image 32/32, int8hu full chain 5/6 suite cases token-exact
(the miss a 0.055-margin tie). M4 Max: tower 111 ms/image, prefill 80.2 tok/s (5× the S=1
text bundle), decode 14.9 tok/s.

Four lessons that will outlive this port:

1. **A bf16 full-model oracle is NOT a valid vision-tower target.** HF-fp32 vs HF-bf16 on
   the same tower already differs by min-row cos 0.9929 — an authored tower that is
   *numerically identical* to fp32 "fails" a 0.999 per-row bar against the bf16 dump. The
   tower is small enough to run fp32; gate each stage against the strongest oracle that
   stage affords (`_smoke/qwen38vl_tower_fp32_ref.npz`), keep bf16 only where fp32 is
   physically impossible (the 27.8B decoder).
2. **The loop-free chunked GDN scan has an overflow cliff, and "passed the suite" does
   not clear it.** The doubling-inverse's worst-case intermediate grows ~C(S−1, S/2−1):
   ~6·10³ at S=16, ~3·10⁸ at S=32, and real image spans (weak decays, `g ≈ 0`) sit near
   the worst case. The pf32 build passed the 6-case oracle suite and then collapsed to
   "!" spam on the next two real photos — one of them only through the app's CGContext
   resize, i.e. the margin was thinner than a resize filter. In fp32 the same inverse
   dies at S≈300 (first symptom: layer-0 GDN NaN on real embeds while random-tensor
   unit tests pass). Two rules fall out: ship the chunk size the DTYPE can prove
   (S=16 for fp16), and gate the class oracle-free — chunked vs S=1-only prefill must
   produce identical greedy tokens on real images
   (`_smoke/test_qwen38vl_chunk_consistency.py`). Chunked prefill + S=1 remainder is
   *mandatory decoder semantics*, not a speed option; the S=1 entrypoint statically
   short-circuits to the single-step scan.
3. **Multifunction bundles can be un-JIT-able on the python runtime.** Loading the pf32
   bundle asserts in MPSGraph `ANERegionFormationPass` ("operand #0 does not dominate this
   use", on the prefill function's state slice-update); `preferred-compute gpu` at load
   does NOT avoid the pass. The fix is the LFM2.5-VL lesson generalized: AOT
   (`xcrun coreai-build compile … --preferred-compute gpu --expect-frequent-reshapes
   --architecture h16c`) and load the `.aimodelc`.
4. **Capture the oracle's rope positions; don't re-derive them.** A forward hook on the
   text rotary module records the exact 3-plane mRoPE positions the reference used
   (prefill AND every decode step); the host reimplementation is then *asserted equal*
   instead of trusted. This is how "an image consumes only `max(H,W)//merge` rope
   positions, `rope_delta = −240`" became a checked fact rather than a reading of
   modeling code.

**A Swift host for the embeds decoder needs no engine.** The pattern is
`CoreAISpeech/SpeechDecoder.swift`, not the engine-based VL backends: load the AOT
`.aimodelc` with `AIModel(contentsOf:)`, `loadFunction("main"/"prefill")`, build the four
state NDArrays from `stateDescriptor(of:)` + `resolvingDynamicDimensions`, pass them via
`InferenceFunction.MutableViews` every call, argmax the logits view. The fp16 embed table
mmaps straight from `embed_tokens.safetensors` (8-byte LE header length + JSON header +
raw rows — ~20 lines of Swift). A working chat app on this pattern measures within ~15%
of the python driver (prefill 78.6 / decode 12.9 tok/s vs 86.0 / 15.2) — host overhead,
not graph cost. One trap the app run caught that the python gates could not: the app's
CGContext resize produces slightly different patches than the gated PIL path, and that
difference alone was enough to tip the pf32 chunk over its fp16 cliff — host-side image
resampling is part of the numerics surface, not cosmetics.

## The MTP draft head, ported later

- **`mtp.*` (15 tensors, MTP draft head).** First left out on the reading that GDN hybrids cap draft-style
  speculation at ~1.2–1.3× (verify c_v≈1.67). As a lossless drafter pair
  (`conversion/export_qwen3_8_mtp.py`) it then measured 1.30× over int4 decode (22.2 → 29.0 tok/s) and 1.71× over
  int8 (15.6 → 26.7) on code, and 0.79× on prose (int4; M4 Max, macOS 27.0 beta 26A5406e, 2026-08-17).
