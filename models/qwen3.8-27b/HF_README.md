---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - qwen3.8
  - hybrid
  - gated-deltanet
  - vlm
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — Apple Core AI (`.aimodel`)

**The Qwen3.8 generation's dense 27B, converted to Apple's Core AI** (the Core ML successor
announced at WWDC26) — ported the day the weights landed. This repo ships the **full VLM**:
the text decoder plus the vision path. The text decoder is the Qwen3.5 hybrid graph run
dense, 64 layers on a 3:1 interleave of **GatedDeltaNet** linear-attention mixers (GVA
48v/16k) and gated full attention (24 q / 4 KV, head_dim 256), untied 248 320-vocab head,
262 K native context. It rides Apple's **`coreai-pipelined` GPU engine** decode-only and
loop-free, with the SSM conv/recurrent states carried as fixed-shape extra states. The
vision path adds the 458M ViT tower and an embeddings-input decoder variant with real
interleaved **mRoPE** (see below).

This is a **reasoning model** — the chat template opens a `<think>` span and generations
spend their first tokens thinking. Budget `max-tokens` accordingly.

**Mac-class, Mac-only:** 28 GB int8 is far past the iPhone memory ceiling. On an M4 Max the
whole 27B is read per token — memory-bandwidth-bound by construction.

> Requires the macOS 27 beta (Core AI ships with the OS). Conversion code, gates and
> knowledge base: **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | prompt tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| `gpu-pipelined/qwen3_8_27b_decode_int8hu_block32_sym` (text) | 28 GB | 16.2 | **15.7** | int8 = 0 confident flips vs bf16 oracle (fp16 control 16/16) |
| `gpu-pipelined/qwen3_8_27b_vision_fp16` (ViT tower) | 0.9 GB | — | 111 ms/image | cos ≥ 0.999996 vs HF fp32 tower |
| `gpu-pipelined/qwen3_8_27b_vl_decode_int8hu_block32_sym_pf16` (VLM decoder) | 28 GB | **86.0** | 15.2 | 5/6 suite cases token-exact (140/144; the miss is a 0.055-margin tie); chunked-vs-S=1 prefill agree 1.00 on 5 real images |

Text row: M4 Max 128 GB, macOS 27 beta, release `llm-benchmark -p 64 -g 128 -n 3`,
`COREAI_CHUNK_THRESHOLD=1`. Eager quant gate: teacher-forced single-step argmax vs the HF
bf16 oracle under the margin ≥ 0.1 rule — 15/16 with the single miss a 0.061-margin
knife-edge tie; the fp16 full-precision control is 16/16. Engine transcript in the
[zoo card directory](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/qwen3.8-27b).

Vision rows: same machine, python runtime on the AOT `h16c` compile (command below).
Prefill is **5× the text bundle's** because the VLM decoder is a `_pf16` multifunction
bundle — a static S=16 "prefill" function chunks the prompt while "main" (S=1) decodes;
image prompts are ~316 tokens, so this is what makes the image path usable. S=16 is a
safety bound, not a tuning knob: the chunked GDN scan's fp16 doubling-inverse overflows
content-dependently at S=32 (passed the oracle suite, then collapsed on the next two
real photos), so the ship chunk stays inside the provable fp16 range. Suite gate:
6 cases (3 COCO images × 2 coarse prompts, one text-before-image) against the bf16 HF
oracle, greedy 24 tokens, full-chain (NumPy preprocess → tower → embed splice → decoder).
The fp16 eager control on the mixed text+image sequences is 32/32 token-exact.

The checkpoint's MTP draft head is included as a lossless drafter pair: on code it measured 1.30× over int4
decode (22.2 → 29.0 tok/s) and 1.71× over int8 (15.6 → 26.7); on prose it is slower (0.79×, int4), so prose
stays on plain decode (M4 Max, macOS 27.0 beta 26A5406e, 2026-08-17).

**No iPhone numbers are published here because none were measured** (28 GB is far past the
iPhone ceiling; the tower alone would fit but has no on-device decoder to feed).

## The vision path, in one paragraph

The tower is a fixed-grid one-shot encoder: `patches [1024, 1536] → image_embeds
[256, 5120]` at a baked 512×512 tile (32×32 patches, 2×2 merge — the fixed square grid
stretches non-square images). The host resizes/normalizes/patchifies in NumPy
(`_smoke/qwen38vl_preprocess.py`, gated exactly against the HF processor), runs the tower
once per image, gathers text-token rows from the shipped `embed_tokens.safetensors`
(2.5 GB, fp16), splices tower rows at the 256 `<|image_pad|>` positions, and feeds the
result to the decoder's `inputs_embeds` input together with three int32 mRoPE position
planes (`pos_t/pos_h/pos_w` — text ramps, image tokens self-locate on the merged grid, an
image consumes only `max(H,W)/2 = 16` rope positions; `_smoke/qwen38vl_host.py` is the
reference host, asserted against the oracle's captured positions). Text-only prompts make
the three planes equal and the graph reduces to plain partial RoPE — i.e. the same
numerics as the text bundle.

`llm-runner`/`llm-benchmark` cannot drive this bundle (embeddings and rope planes are not
engine inputs); the reference driver is
[`_smoke/test_qwen38vl_suite_gate.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/_smoke/test_qwen38vl_suite_gate.py).
Driving it from the python runtime needs the AOT compile (the JIT path asserts in
MPSGraph's ANE region pass on this multifunction graph):

```bash
xcrun coreai-build compile qwen3_8_27b_vl_decode_int8hu_block32_sym_pf16.aimodel \
    --platform macOS --preferred-compute gpu --expect-frequent-reshapes --architecture h16c
```

## Run it

```bash
git clone https://github.com/john-rocky/coreai-kit
cd coreai-kit/Examples/ChatDemo
swift run chat-cli --model qwen3.8-27b --prompt "What can you do, offline?"
```

Or in Swift, via [CoreAIKit](https://github.com/john-rocky/coreai-kit):

```swift
import CoreAIKit
let chat = try await ChatSession(catalog: "qwen3.8-27b")
let reply = try await chat.respond(to: prompt)
```

## Reproduce

```bash
git clone https://github.com/john-rocky/coreai-model-zoo
cd coreai-model-zoo
python3 conversion/zoo_convert.py run qwen3.8-27b
```

Recipe (text): `export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id
Qwen/Qwen3.8-27B` — the same verified recipe as Qwen3.6-27B (the two generations are
architecturally byte-identical; only the weights changed). Recipe (vision path):
`export_qwen38vl_pipelined.py int8hu` — one run emits the fp16 tower AND the pf16 VLM
decoder (+ `embed_tokens.safetensors`). Port write-up:
[`knowledge/qwen3.8-27b-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/qwen3.8-27b-port.md).
