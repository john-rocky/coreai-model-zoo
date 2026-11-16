# North-Micro-Vision (`cohere_compass`) — port knowledge

Cohere's 2.4B VLM: a 400M tower custom-trained from SigLIP2-SO400M plus a 2B Cohere decoder.
The short version of this port is that **half of it already existed and the other half is not
the architecture its config implies**.

Ported: `models/macos/cohere_compass.py` (decoder only),
`conversion/export_northmv_pipelined.py`, gates in `_smoke/*northmv*`.

## The vision tower is Qwen3-VL's, and that is checkable in a minute

The weight names give it away before any modeling file is read:

```
model.visual.patch_embed.proj            [1152, 3, 2, 16, 16]   Conv3d, temporal 2
model.visual.pos_embed.weight            [2304, 1152]           interpolated to the grid
model.visual.blocks.N.attn.qkv           fused                  Qwen-VL naming
model.visual.merger.{norm,linear_fc1,linear_fc2}
model.visual.deepstack_merger_list.{0,1,2}.*
```

That is the Qwen3-VL visual encoder — deepstack mergers and all — at SigLIP2-SO400M's
dimensions (hidden 1152 / MLP 4304 / 27 blocks / 16 heads, deepstack at [8,16,24]). The zoo's
`Qwen3VLVisionEncoder` loads `model.visual.*` with **zero missing and zero unexpected keys**,
and reproduces every seam at **cos 1.000000**. `vision_encoder_from_hf` is therefore a config
shim, not a second implementation.

**The cheap check is worth building the habit around**: instantiate the closest existing
encoder, `load_state_dict(strict=False)`, and print missing/unexpected. It costs a minute and
either saves days or tells you immediately that the resemblance was superficial.

## The decoder is not a Llama with different numbers

Four things, each of which runs and produces fluent text when done wrong:

1. **Parallel block.** One `input_layernorm` per layer — the weight dump has no
   `post_attention_layernorm` at all — and attention and MLP both read it, their outputs summed
   into the residual: `h = x + attn(ln(x)) + mlp(ln(x))`.
2. **Cohere LayerNorm.** The mean *is* subtracted and there is no bias. Reaching for RMSNorm
   because the weight shape matches is the same silent class of error as the parallel block.
3. **A quarter of the layers have no positional encoding.** `layer_types` is `SSSF × 7`; the
   config gives `rope_parameters.full_attention: null`, and the reference hands those layers
   `position_embeddings=None`. The 21 sliding layers carry **interleaved M-RoPE** (sections
   [24,20,20], θ 5e4 — the same `apply_interleaved_mrope` Qwen3-VL uses, so the zoo's
   `mrope_masks` applies verbatim) inside a 4096 window.
4. **`logit_scale` 0.25** multiplies the logits, and the 262 144-entry embedding is **tied** to
   the head.

Position handling, deepstack and the extension-id splice are the Qwen3-VL rider's contract
unchanged, including the rope shift (an image consumes only max(H,W) rope positions).

## Host preprocessing

`_smoke/northmv_preprocess.py`. The patch layout is **Qwen-VL's, not NaFlex's** — the opposite
of the LFM2.5-VL host in the same directory:

- inside one patch vector the order is `[C][T][py][px]` (**channel-major**, still frame
  duplicated across `temporal_patch_size = 2`, so `patch_dim = 3·2·16·16 = 1536`);
- the patches themselves are **block-major**, so each 2×2 merge group is contiguous.

Resampling is `resample: 3` (PIL BICUBIC), clipped like Pillow because Pillow writes uint8.
Fed Pillow's own resize the NumPy host is **bit-exact** against the processor.

## Quantization: 100 % at int8, a cliff at int4

int8lin is token-exact against fp32 on the whole 9-case suite (338/338) and 24/24 on device.
There is no fp16 baseline row on the card because a perfect score does not need one — the
baseline exists to explain a gap.

int4lin is 0/9 and 23.7 % of tokens, and it fails in the way that names itself: a lost sentence
boundary with instruction boilerplate leaking in (`"…a blanket.Answer: Cats.I apologize, but I
cannot provide a detailed description of the image"`) and flat repetition (`"Images of cats
sleeping on a couch are shown."` twice). Not published.

**Three int4 verdicts in one day, and they do not line up with size**: LFM2.5-VL-450M craters,
LFM2.5-VL-3B does not move (7/9 = its own fp16 baseline), this 2.4B craters. Read the
generations of the model in front of you; nothing else predicts it.

## The iOS load wall is further out than this repo thought

This bundle's AOT `resources.bin` is **2.39 GiB** and loads on an iPhone 17 Pro (nat 16/16,
image oracle 24/24, 21.5 prefill / 18.2 decode tok/s, clean at `PB_G=1024`). An earlier note
put the wall at 2 GiB (2^31) on a 1.96 ✅ / 3.92 ❌
bracket; LFM2.5-VL-3B cleared it at 2.03 GiB the same day and this at 2.39. **Measure the
compiled artifact, then ask a phone.** A bracket with a 2 GiB gap in it is not a verdict.

## Environment

The oracle requires **transformers git main** (5.16.0.dev0): the 5.15.0 *release* does not know
`cohere_compass` and raises on `AutoConfig` — which is the good failure. The venv used here is
`~/code/litertlm-convert/.venv-vl0930-t515`.

## Measurements

M4 Max, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`: vision 83.4 ms per
image at cos 0.999996, text core 145.3 prefill / 118.6 decode tok/s, decoder int8lin 2.4 GB,
tower fp16 1.0 GB. iPhone 17 Pro: 21.5 / 18.2 tok/s with the image bound.
