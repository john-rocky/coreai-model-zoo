# Qwen3.8-27B (VLM: text decoder + vision path) — Core AI

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) is the **Qwen3.8 generation's
only open compact model** — the flagship line's dense 27B, released 2026-08-14 as a native
VLM (image-text-to-text). This card covers the **whole port**: the text decoder (ported the
day the weights landed) and the **vision path** (tower + embeddings-input decoder, the
zoo's first qwen3.5-family vision authoring). The text side is the generation successor to
[Qwen3.6-27B](../qwen3.6-27b/README.md), and the two are **architecturally
byte-identical**: every `text_config` key and the full 851-key text weight map match — new
weights on the proven graph.

The graph is the Qwen3.5 hybrid decoder run dense: 64 layers on a 3:1 interleave of
**GatedDeltaNet** linear-attention mixers (GVA 48 value / 16 key heads) and **gated full
attention** (24 q / 4 KV, head_dim 256, partial mRoPE θ=1e7, swish output gate), dense
`MLP(17408)` FFNs, untied 248 320-vocab head, 262 144 native context.

One thing the checkpoint carries that this port first left out:

- **`mtp.*` (a built-in MTP draft head).** Left out on the reading that GDN hybrids cap draft-style speculation
  at ~1.2–1.3× (c_v ≈ 1.67). As a lossless drafter pair it then measured 1.30× over int4 decode (22.2 → 29.0
  tok/s) and 1.71× over int8 (15.6 → 26.7) on code, and 0.79× on prose (int4; M4 Max, macOS 27.0 beta 26A5406e, 2026-08-17).
  The [HF repo](https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI) has the bundles.

It is also a **reasoning model**: the vendor chat template opens a `<think>` span, and
generations spend their first tokens thinking. Budget `max-tokens` accordingly.

Source checkpoint: revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0).
Weights were pulled through ModelScope (HF CDN was crawling on release day) and every shard
**sha256-verified against HF's LFS oids** before use — see
[`knowledge/qwen3.8-27b-port.md`](../../knowledge/qwen3.8-27b-port.md).

**⬇️ Converted `.aimodel` bundles:**
`gpu-pipelined/qwen3_8_27b_decode_int8hu_block32_sym/` (28 GB text LanguageBundle incl.
tokenizer + chat template; decode-only loop-free for the
[pipelined engine](../../knowledge/pipelined-engine.md)) ·
`gpu-pipelined/qwen3_8_27b_vision_fp16/` (0.9 GB ViT tower, one shot per image) ·
`gpu-pipelined/qwen3_8_27b_vl_decode_int8hu_block32_sym_pf16/` (28 GB embeddings-input VLM
decoder, multifunction S=1 decode + S=16 chunked prefill, + `embed_tokens.safetensors`).
Added on 2026-08-17, in the same `gpu-pipelined/` directory: `qwen3_8_27b_decode_int4lin/` (18 GB int4 text
decoder) and the lossless MTP drafter set (three bundles: `qwen3_8_27b_verify_s9_int4lin_d4_hpost/`,
`qwen3_8_27b_mtp_s1_int8hu_block32_sym/`, `qwen3_8_27b_mtp_s9_int8hu_block32_sym/`). All seven are in
[mlboydaisuke/Qwen3.8-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.8-27B-CoreAI).

## Verified

Mac numbers below were measured on this machine; **no iPhone numbers are published because
none were measured** — at 28 GB the bundle is far past the iPhone memory ceiling anyway:
this is a Mac-class model.

| bundle | size | prompt tok/s | decode tok/s | eager quant gate | engine bundle gate |
|---|---:|---:|---:|---|---|
| `int8hu --head-sym` (ship) | 28 GB | 16.2 | **15.7** | **PASS** 15/16, 0 confident flips | **PASS** — token-for-token == fp16 oracle |

M4 Max 128 GB, macOS 27 beta, release `llm-benchmark -p 64 -g 128 -n 3` (±0.04 across
trials), `COREAI_CHUNK_THRESHOLD=1`, engine variant `coreai-pipelined` (+ extra-states
patch). Prefill ≈ decode because prefill runs as pipelined S=1 steps.

**Numerics.** fp32 for 27.8B needs ~111 GB RAM, so the oracle is the checkpoint's native
**bf16** (HF eager, transformers 5.12.1); the gate is teacher-forced single-step argmax
under the oracle-margin ≥ 0.1 rule. Result — *cleaner than the 3.6-27B port*:

- **fp16 control: 16/16 exact.** Full precision reproduces the bf16 oracle at every
  position (the 3.6 port had one bf16-resolution artifact; this checkpoint has none), so
  any quantization flip would attribute cleanly.
- **int8hu: 15/16 with zero confident flips.** The single miss is a knife-edge top-2 tie
  (oracle margin 0.061 < 0.1), per-position cos 0.9977–0.9999. The absmax `symmetric`
  per-block-32 head rule (clipping craters big-vocab heads — see the
  [qwen3.5 card](../qwen3.5/README.md)) and the int8 per-block-32 body are
  quality-transparent on this model.

The engine-side transcript (`coreai_gate.py`, greedy, warmup off, fp16 oracle) is in this
directory as `gate-qwen3.8-27b-int8hu.json`.

## The vision path

The tower (`model.visual.*`, 458M, **no deepstack** — `deepstack_visual_indexes: []` makes
this simpler than Qwen3-VL) is authored as a fixed-grid one-shot encoder
(`models/macos/qwen3_5_vision.py` on the overlay): `patches [1024, 1536] → image_embeds
[256, 5120]` at a baked 512×512 tile, everything positional (bilinear pos-embed
interpolation to the 32×32 patch grid, 2D rotary θ=10000) baked as constants in the
processor's merge-block-major patch order. The known cost of a baked square grid applies:
non-square images are stretched, and no gate here can see that (the oracle is captured
through the same tile).

The decoder is the SAME hybrid graph as the text bundle with two changes
(`Qwen3_5VLStatefulEmbeds` in `qwen3_5.py`): `inputs_embeds` replaces `input_ids` (the
host gathers text rows from the shipped fp16 `embed_tokens.safetensors` and splices tower
rows at the 256 `<|image_pad|>` positions), and **real interleaved mRoPE** from three
host-fed int32 position planes — text ramps; image tokens self-locate on the merged grid;
an image consumes only `max(H,W)//merge = 16` rope positions, so post-image text resumes
at start+16, not start+256 (`rope_delta = -240` on every suite prompt). With equal planes
the interleave collapses to the text bundle's plain partial RoPE — verified to 0
difference — so text-only behavior is unchanged. Host contract (mRoPE planes + splice) is
NumPy: `_smoke/qwen38vl_host.py`, asserted against the oracle's **captured** rotary
positions, 6/6 prompts; preprocessing is `_smoke/qwen38vl_preprocess.py`, byte-equal to
the HF processor (max|d| 5.9e-8, 6/6).

It is a `_pf16` **multifunction bundle**: "main" = static S=1 decode, "prefill" = static
S=16 chunk. The chunk size is a safety bound, not a tuning knob: the chunked GDN scan's
doubling-inverse runs in fp16 in-graph, and its worst-case intermediate growth is
~C(S−1, S/2−1) — ~6·10³ at S=16 (inside fp16's 65504) vs ~3·10⁸ at S=32. An S=32 build
passed this 6-case oracle suite and then collapsed to "!" spam on the next two real
photos (weak-decay image spans; one only failed through the app's CGContext resize —
that is how content-marginal S=32 is). The regression gate for this class is
`_smoke/test_qwen38vl_chunk_consistency.py`: chunked vs S=1-only prefill must produce
identical greedy tokens on real images, no oracle needed — pf16 agrees 1.00 on all 5
gate images including both S=32 breakers. Prefill at S=16 chunks measures **86.0 tok/s
vs 16.2** for the S=1 text bundle — the difference between ~4 s and ~20 s to first token
on a ~316-token image prompt.

| stage | gate | result |
|---|---|---|
| NumPy preprocess | vs HF processor `pixel_values`, 6 cases | **exact** (max\|d\| 5.9e-8) |
| tower (fp32 torch) | cos vs HF **fp32** tower, 3 images | **1.000000**, min-row 1.000000 |
| tower (fp16 `.aimodel`, GPU) | same | cos ≥ 0.999996, min-row ≥ 0.999886 |
| host mRoPE planes | vs oracle-captured positions, 6 prompts | **equal**, delta −240 everywhere |
| decoder eager fp16, mixed text+image | teacher-forced 16 steps × 2 cases (image-first + text-first) | **32/32 exact** |
| full chain, int8hu `.aimodelc` (GPU) | 6-case suite, greedy 24 tokens vs bf16 oracle | **5/6 token-exact, 140/144 tokens**; the miss is a 0.055-margin knife-edge tie |
| chunked vs S=1 prefill | 5 real images (3 suite tiles + the 2 S=32 breakers), 48 greedy tokens | **agree 1.00 on all 5**, no degeneration |

Transcript: `gate-qwen3.8-27b-vl-suite.json` in this directory. Two things the gates
taught, worth keeping: the **bf16 full-model oracle is not a valid tower target** (HF-fp32
vs HF-bf16 already differ by min-row cos 0.9929 — the tower gates against a dedicated
fp32 tower reference), and the multifunction bundle **must be AOT-compiled for the python
runtime** (JIT specialization asserts in MPSGraph's `ANERegionFormationPass`, an
operand-dominance bug on the prefill function; `--preferred-compute gpu` at load does not
avoid it):

```bash
xcrun coreai-build compile qwen3_8_27b_vl_decode_int8hu_block32_sym_pf16.aimodel \
    --platform macOS --preferred-compute gpu --expect-frequent-reshapes --architecture h16c
```

Measured on the AOT compile (M4 Max, python-runtime driver
`_smoke/test_qwen38vl_suite_gate.py` — `llm-runner` cannot bind an embeddings input, so
the text bundle covers engine benches): **tower 111 ms/image, prefill 86.0 tok/s, decode
15.2 tok/s.**

**Dense means no MoE speed trick:** decode reads the whole model per token — ~28 GB/token
at int8 → 15.7 tok/s is memory-bandwidth-bound on M4 Max, matching the 3.6-27B (15.9)
within noise. int4 was left out at first (on the byte-identical 3.6-27B it had a
real high-confidence flip), then exported on 2026-08-17 as `qwen3_8_27b_decode_int4lin` (18 GB): 15/16 in the
eager fake-quant check against the bf16 oracle with zero confident flips (the miss is the same 0.061-margin tie as int8), and
22.2 tok/s decode against 15.6 for int8 in the same CoreAIChatMac engine (M4 Max, macOS 27.0 beta 26A5406e).

## Reproduce

```bash
python3 conversion/zoo_convert.py run qwen3.8-27b          # text bundle
python3 conversion/zoo_convert.py run qwen3.8-27b-vl       # vision tower + VLM decoder
# or directly:
cd coreai-models   # with the qwen3_5 model overlay (see ../conversion)
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_5_decode_pipelined.py \
    int8hu --head-sym --hf-id Qwen/Qwen3.8-27B
.venv/bin/python ../coreai-models-community/conversion/export_qwen38vl_pipelined.py int8hu
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/qwen3_8_27b_decode_int8hu_block32_sym -p 64 -g 128 -n 3
```

Model overlay: `models/macos/qwen3_5.py` — the GVA head-repeat is config-driven, the
loader already picks up the untied root `lm_head.weight` and skips `mtp.*`; the vision
path adds `models/macos/qwen3_5_vision.py` (tower) and the `Qwen3_5VLStatefulEmbeds`
class + interleaved-mRoPE masks in `qwen3_5.py`. **Decode-only loop-free** because the
GatedDeltaNet `while_loop` does not lower on the GPU delegate. If the Hugging Face CDN
crawls (release day), the ModelScope + sha256-verify route is documented in the
[port note](../../knowledge/qwen3.8-27b-port.md).
