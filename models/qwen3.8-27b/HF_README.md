---
library_name: coreai
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
  - speculative-decoding
pipeline_tag: image-text-to-text
base_model_relation: quantized
---

Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles that run on the GPU or the Neural Engine, e.g. Qwen3-8B 4-bit decodes at 94 tok/s on an M4 Max GPU, MLX 90 under the same protocol ([apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench), macOS 27 beta 26A5353q, 2026-06-11).

<!-- gen-cards:devicemark begin (managed by scripts/gen-cards + tools/devicemark_row.py — edit cards.json, not this block) -->
This model has no row on [DeviceMark](https://devicemark.github.io/), the on-device LLM leaderboard.
<!-- gen-cards:devicemark end -->

# Qwen3.8-27B — Apple Core AI (`.aimodel`)

**The Qwen3.8 generation's dense 27B, converted to Apple's Core AI** (the Core ML successor
announced at WWDC26) — ported the day the weights landed. This repo ships the **full VLM**
(text decoder + vision path), an **int4 text decoder**, and the checkpoint's own **MTP head
as a lossless ⚡Spec drafter pair**. The text decoder is the Qwen3.5 hybrid graph run
dense, 64 layers on a 3:1 interleave of **GatedDeltaNet** linear-attention mixers (GVA
48v/16k) and gated full attention (24 q / 4 KV, head_dim 256), untied 248 320-vocab head,
262 K native context. It rides Apple's **`coreai-pipelined` GPU engine** decode-only and
loop-free, with the SSM conv/recurrent states carried as fixed-shape extra states. The
vision path adds the 458M ViT tower and an embeddings-input decoder variant with real
interleaved **mRoPE** (see below).

This is a **reasoning model** — the chat template opens a `<think>` span and generations
spend their first tokens thinking. Budget `max-tokens` accordingly.

**Mac-class, Mac-only:** 18–28 GB is far past the iPhone memory ceiling. On an M4 Max the
whole 27B is read per token — memory-bandwidth-bound by construction.

> Requires macOS 27 (Core AI ships with the OS). Conversion code, gates and
> knowledge base: **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | prompt tok/s | decode tok/s | numerics |
|---|---:|---:|---:|---|
| `gpu-pipelined/qwen3_8_27b_decode_int8hu_block32_sym` (text) | 28 GB | 16.2 | 15.7 | int8 = 0 confident flips vs bf16 oracle (fp16 control 16/16) |
| `gpu-pipelined/qwen3_8_27b_decode_int4lin` (text, int4) | 18 GB | — | **22.2** | int4 = 0 confident flips vs bf16 oracle; the single miss is the same 0.061-margin knife-edge tie as int8 |
| `gpu-pipelined/qwen3_8_27b_verify_s9_int4lin_d4_hpost` (⚡Spec verify) | 18 GB | — | **29.0 with ⚡Spec, code** | output byte-identical to greedy (see below) |
| `gpu-pipelined/qwen3_8_27b_mtp_s1_int8hu_block32_sym` (⚡Spec drafter) | 4.1 GB | — | — | the checkpoint's own 1-layer MTP head, int8 |
| `gpu-pipelined/qwen3_8_27b_mtp_s9_int8hu_block32_sym` (⚡Spec replay) | 4.1 GB | — | — | same weights as an S=9 replay graph |
| `gpu-pipelined/qwen3_8_27b_vision_fp16` (ViT tower) | 0.9 GB | — | 111 ms/image | cos ≥ 0.999996 vs HF fp32 tower |
| `gpu-pipelined/qwen3_8_27b_vl_decode_int8hu_block32_sym_pf16` (VLM decoder) | 28 GB | **86.0** | 15.2 | 5/6 suite cases token-exact, 140/144 tokens; the one miss is a 0.055-margin knife-edge tie |

int8 text row: M4 Max 128 GB, macOS 27 beta, release `llm-benchmark -p 64 -g 128 -n 3`,
`COREAI_CHUNK_THRESHOLD=1`. int4 and ⚡Spec rows: same machine, measured in the zoo's
CoreAIChatMac engine (cool state, GPU solo, medians over the generation; the engine
reproduces the int8 llm-benchmark number at 15.6). Eager quant gates: teacher-forced
single-step argmax vs the HF bf16 oracle under the margin ≥ 0.1 rule — int8 15/16,
int4 15/16, both missing only the same 0.061-margin knife-edge; the fp16 full-precision
control is 16/16. Engine transcript in the
[zoo card directory](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/qwen3.8-27b).

Vision rows: same machine, python runtime on the AOT `h16c` compile (command below).
Prefill is **5× the text bundle's** because the VLM decoder is a `_pf16` multifunction
bundle — a static S=16 "prefill" function chunks the prompt while "main" (S=1) decodes;
image prompts are ~316 tokens, so this is what makes the image path usable. S=16 is a safety bound: the first upload (`_pf32`,
S=32, 2026-08-15) passed the 6-case suite, then collapsed to a run of `!` on two real photos (fp16 overflow in the chunked
GDN scan), and was replaced the same day. Suite gate:
6 cases (3 COCO images × 2 coarse prompts, one text-before-image) against the bf16 HF
oracle, greedy 24 tokens, full-chain (NumPy preprocess → tower → embed splice → decoder).
The fp16 eager control on the mixed text+image sequences is 32/32 token-exact.

**No iPhone numbers are published here because none were measured** (18 GB is still far
past the iPhone ceiling; the tower alone would fit but has no on-device decoder to feed).

## ⚡Spec — lossless speculative decoding with the checkpoint's own MTP head

The checkpoint ships a trained 1-layer MTP draft head (`mtp.*`, DeepSeek-V3 style:
`fc([emb|hidden]) → decoder layer → norm`, shared lm_head). This repo includes it as a
Core AI drafter pair. The loop is **lossless by construction** — draft tokens are
verified by the target graph and the committed text is byte-identical to plain greedy
in every measured run.

Three bundles cooperate:

- `verify_s9_int4lin_d4_hpost` — static S=9 verify forward: 1 committed + 8 draft
  tokens in, per-position logits **plus the last-layer hidden** out (the MTP head's
  input). Chunk-scan doublings cut to 4 (exact for S=9).
- `mtp_s1_...` — the MTP head as an S=1 stateful drafter with its own 1-layer KV.
- `mtp_s9_...` — the same weights as an S=9 graph sharing that KV, so the host
  replays committed context into the drafter in one forward instead of nine
  (worth +33% on code).

Measured end-to-end (M4 Max, 256-token cap, cool state, GPU solo):

| workload | plain int4 decode | ⚡Spec int4 K=8 | vs decode |
|---|---:|---:|---|
| code continuation | 22.2 tok/s | **29.0 tok/s** | **1.30×** |
| free-form prose | **22.2 tok/s** | 17.5 (K=4) | 0.79× — keep spec OFF |

Spec pays where the head predicts well (code: 4.7 accepted/round) and loses on
free-form (1.5/round): the GDN verify forward carries a fixed ~60 ms chunk-scan cost
that quantization does not shrink (109 ms verify vs 45 ms decode step). So the ship
shape is split: **verify+MTP pair for code-like work, plain decode bundle for prose.**
The int8 pair measures 26.7 tok/s on code (1.71× its 15.6 decode) — int4 wins both
absolute and split. Reference host loop (replay → seed → K draft steps → verify →
commit, snapshot/re-anchor state discipline): the zoo repo's CoreAIChatMac
`SpecDecodeEngine`. The verify bundle's metadata names its drafter pair
(`spec_draft`, `spec_mtp_replay`) for auto-pairing.

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

Recipes (text): `export_qwen3_5_decode_pipelined.py int8hu --head-sym --hf-id
Qwen/Qwen3.8-27B` (int8 ship recipe, identical to Qwen3.6-27B) and
`... int4lin --hf-id Qwen/Qwen3.8-27B` (int4). ⚡Spec set:
`export_qwen3_5_verify_pipelined.py int4lin --hf-id Qwen/Qwen3.8-27B --s 9 --doublings 4
--emit-hidden post` (verify) + `export_qwen3_8_mtp.py` (S=1 drafter; `--s 9` for the
replay sibling). Recipe (vision path): `export_qwen38vl_pipelined.py int8hu` — one run
emits the fp16 tower AND the pf16 VLM decoder (+ `embed_tokens.safetensors`). Port
write-up:
[`knowledge/qwen3.8-27b-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/qwen3.8-27b-port.md).

<!-- funnel:v1 -->

---

**More models in this format:** [Core AI Model Zoo](https://huggingface.co/collections/mlboydaisuke/core-ai-model-zoo-6a7ff330f753e8dcae04671a) — 75 models, each with the recipe that produced it.

**Want a different model on-device?** [Open a request](https://github.com/john-rocky/on-device-requests) — free, open weights only; the export and its measured numbers get published publicly.

<!-- /funnel:v1 -->
