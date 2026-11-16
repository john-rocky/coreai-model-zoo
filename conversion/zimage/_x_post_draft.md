# X post drafts

## A — the port

Z-Image-Turbo (6B, Apache-2.0) running on Core AI, on a Mac GPU.

One graph covers 256/512/1024 and any prompt length — the image-token and caption
axes are both dynamic, ~5-9% cost. bf16, PSNR 42.6 dB vs the fp32 reference.
18s @512, 70s @1024 on an M4 Max.

https://huggingface.co/mlboydaisuke/Z-Image-Turbo-CoreAI

[attach: engine_image.png (apple), engine_prompt.png (lighthouse)]

---

## B — the int8 finding

Porting a 6B diffusion DiT, I assumed int8 would be the small-and-fast option.
It's neither, on this shape:

  bf16                0.89 s/forward
  int8 (weight-only)  2.35 s/forward

Weight-only int8 dequantizes back to 16-bit and runs the same matmul. On a
compute-bound graph (1056 tokens x 6B) that's strictly more work. Activation
dtype and quant granularity make no difference — I checked all of them.

int8 wins on bandwidth-bound shapes (LLM decode, S=1) and on footprint. Not here.

---

## C — the AOT one (niche but real)

Two things I learned trying to get a 6B diffusion DiT onto an iPhone:

1. fp32 activations make the AOT compiler constant-fold the quantized weights
   into fp32. A 1.9 GB int8 graph becomes an 8.7 GB .aimodelc. fp16 activations
   keep them int8. Same mechanism explains why the fp32 graph was also the
   *fastest* int8 config — the dequant is gone.

2. The iOS runtime won't load a resources.bin over 2 GiB. 1.96 GB loads,
   3.92 GB gives ENOENT. The compiler produces the 6.2 GB bundle with zero
   errors, so "it compiled" tells you nothing about whether it loads.

The model stayed on the Mac.

---

## D — the honest one

Spent a day trying to get Z-Image onto an iPhone. fp16 sends the DiT all-NaN at
sampler step 2; bf16 is exact but AOT won't take a bf16 module. I tried five
fixes — hand-rolled fp16-safe norms, two output-exact weight rescalings, both
together, activation quantization. None moved it by one bit.

Then I stopped reasoning from the Mac and put it on the phone. Same NaN, same
count. Plus a 2 GiB load wall I hadn't even suspected.

Should have gone to the device three hypotheses earlier.

---

Notes:
- Numbers above are the shipped --io-fp32 graphs, M4 Max, measured.
- Images: conversion/zimage/oracle/{engine_image,engine_prompt,engine_s1024}.png
