# X post drafts

## A — the port

Nemotron-3-Nano 4B (NVIDIA's Mamba2 + attention + MLP hybrid) running on Core AI,
on an iPhone 17 Pro.

42 blocks: 21 Mamba2 mixers, 17 MLPs, 4 NoPE attention layers. int8, 4.3 GB,
AOT-compiled. 16.0 tok/s decode, 85.2 on an M4 Max. Greedy output is
token-identical to the fp32 transformers rollout.

No custom kernel. At S=1 the selective scan is a single recurrence step, so the
decode graph is loop-free and lowers as-is.

https://huggingface.co/mlboydaisuke/Nemotron-3-Nano-4B-CoreAI

---

## B — the 4-bit trilemma

Wanted this 4B under 3 GB. Tried every 4-bit scheme I had. All three fail, and
each one fails for a *different* reason:

  int4 symmetric      27-29/33 oracle top-1   dequants at 2.20 ms/step (int8: 2.31)
  int4 asymmetric     33/33                   3.0 tok/s on device (int8: 16.0)
  int4 k-means (LUT)  22/33                   — worst, and it's the one the fast kernel reads

Symmetric is fast but wrong. Asymmetric is right — token-identical greedy to int8
over 48 tokens — but the zero-point costs ~0.4 ms per layer, so 42 layers of it
lose 4.6x while reading 1.45x fewer bytes. k-means shares 16 centroids across
32 rows x K columns, no scale along K; that was exact on another model, worst here.

The fast path and the correct path are different schemes. So the lever isn't a
better quantizer, it's a fused *asymmetric* int4 matvec kernel. Nobody has one,
because the LUT format exists precisely to avoid the affine path.

---

## C — the bandwidth model correction (small, and I got it wrong first)

I predicted 10-11 tok/s from the bundle size. It did 16.0.

The embedding table is a one-row gather, not a matmul. It doesn't get read per
token. Drop it and the per-token read is 3.78 GB — at ~60 GB/s that's a 15.9 tok/s
ceiling. Measured 16.0. Saturated.

The trap: on a **tied**-embedding model the lm_head *is* the embedding table, so
it does get read every token, and "bundle size / bandwidth" happens to be right.
That's why the estimate worked on the last model and was 40% low on this one.
Check whether the head is tied before you trust it.

---

## D — the honest one

I benchmarked a fused Metal kernel for a Mamba2 decode step against the compiler's
graph. It came out 1.3x slower. I wrote that up: the optimizer beats the hand kernel.

Then I found the cause — a fp32 cast at the kernel boundary — fixed it, remeasured,
got 1.07x faster, and wrote *that* up. Two causal stories, one day.

Both were noise. The machine drifts 10-15% between runs and I was timing the two
arms in separate passes. Paired properly — same process, alternating, eight reps,
report the median — every variant of the kernel is 3-8% faster, including the one I
had "diagnosed" as broken. The variant I "fixed" is not reliably better than the one
I fixed it from.

A 5% effect cannot be measured by running each side once. I know this. I did it
anyway, twice, and each time the number was decisive enough to explain.
---

Notes:
- All numbers measured: iPhone 17 Pro (A19 Pro) cooled, PipelinedBench, AOT h18p;
  M4 Max via raw AIModel calls. Quality = teacher-forced top-1 vs the fp32 oracle at
  the 33 margin-clean prompt positions.
- D's figures: granite-4.0-h-350m, 32 layers. Paired median stock/fused = 1.060 (v1),
  1.031 (v2), 1.078 (v3), 8 interleaved reps each. Unpaired singles of the same configs
  ranged 0.89x–1.19x.
- Card: models/nemotron-3-nano/README.md
