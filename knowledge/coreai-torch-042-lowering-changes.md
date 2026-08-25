# Which coreai-torch 0.4.2 lowering changes can reach a shipped bundle

Established 2026-08-25 by converting the same minimal modules under coreai-torch 0.4.1 and
0.4.2 in two isolated environments — same `coreai-core 1.0.0b2`, same `torch 2.11.0`, so the
converter is the only variable — and diffing the emitted Core AI graph. Composite bodies are
inlined before the diff and the generated symbol suffixes normalised; without that, every
`batch_norm` reads as changed (the suffix is regenerated each run) and every changed
composite body reads as unchanged.

## Why the obvious scan cannot answer this

Scanning shipped `.aimodel` bundles for the op names in the changelog does not clear them.
The bundle is what the converter *produced*: a lowering that was wrong is already baked into
`sub`/`divide`/`sqrt` and no longer carries the name of the op it came from. Two concrete
traps found doing it:

- **`batch_norm` is emitted as `coreai.invoke @batch_norm_<random suffix>`, not as an op
  named `batch_norm`.** An op histogram over a bundle reports `invoke`. Reading "batch_norm:
  0" as "no batch norm in this bundle" is reading the wrong field.
- **`layer_norm` with `elementwise_affine=False` never appears as itself either** — it is
  reduce/mul/add/rsqrt by the time anything can count it.

Ask the converter instead. A construct whose graph is byte-identical across the two versions
cannot have changed numerically: it is the same program.

## Verdicts

| Fix | Reaches a graph? | Values move? |
|---|---|---|
| #54 fp16 batch norm computed in fp32 | yes, fp16 only | **no — measured** |
| #32 integer true-divide promoted | **yes** | **yes — measured** |
| #40 conv transpose, 1D | yes | no — shapes only |
| #40 conv transpose, 2D without `output_padding` | no | no |
| #55 layer_norm gamma/beta shape | yes, multi-dim `normalized_shape` only | no |
| #42 mean operand cast | no | no |
| #43 `min.dim` argmin | no | no |
| #36 `max_pool2d` default stride | 0.4.1 **cannot convert it at all** | n/a |
| #35 `atan2` | 0.4.1 **cannot convert it at all** | n/a |
| #24 quantize/dequantize negative axis | not from this repo — see below | n/a |

### #54 — changes the graph, and not the output

Under 0.4.1 the whole normalisation is written in fp16, `eps` included: `1e-5` is cast to fp16
before it is added to the variance, and the `sqrt`, the subtract and the divide are all fp16.
Under 0.4.2 the input and all four parameters are cast to fp32, the arithmetic is written
there, and the result is cast back. An fp32 model is unaffected apart from the removal of some
no-op `cast f32 -> f32`.

**Then it was run, and the outputs are bit-identical.** `Conv1d -> BatchNorm1d` in fp16,
executed through `coreai.runtime` under both converters, max-abs against a torch fp32
reference:

| regime | 0.4.1 | 0.4.2 |
|---|---|---|
| trained-looking stats (`var` 0.01–5) | 0.00873 | 0.00873 |
| `var` below fp16's smallest normal (`1e-8`–`1e-6`) | 7.2617 | 7.2617 |
| fp32 control | 1.9e-06 | 1.9e-06 |

The second row is the regime built to defeat it: `eps = 1e-5` is **subnormal in fp16**, so if
the fp32 upcast ever mattered it would matter there. It does not. Nor is it `optimize()`
erasing the difference — running with and without it gives the same digits.

**Why: `batch_norm` is emitted as `coreai.invoke @batch_norm_<suffix>`, and the runtime picks
the kernel for a composite.** The casts the converter writes inside the composite body are a
declaration of intent, not the arithmetic that runs. The fp16 error that is there — and 0.00873
against a signal of 21.8 is real error — comes from fp16 weights and fp16 input, and both
converters inherit it equally.

**The general form, which is the opposite direction of this note's own method:** a graph that is
identical proves the output is identical, because it is the same program. A graph that
*differs* proves nothing about the output, and proves least of all across a composite boundary.
Measure the second case; only the first one is free.

Three recipes here build a `BatchNorm1d` — the Conformer convolution module in
`conversion/parakeet/export_encoder.py`, `conversion/lfm_audio/export_encoder_adapter.py` and
`conversion/sortformer_diar/sortformer_model.py`. (`conversion/nemotron_asr/export_encoder.py`
states in its own header that its conv-module norm is LayerNorm, not BatchNorm.) They are the
models this would have reached, and it reaches none of them: the measurement above is exactly
their shape. `sortformer_float16` needs no re-export on this account.

### #32 — silently truncating division

0.4.1 lowered a true-divide on two integer tensors to an **integer** divide and then cast the
truncated result to float. Run through the runtime on `[7, 8, 9, 10] / 3`:

| | output |
|---|---|
| torch | `2.3333, 2.6667, 3.0, 3.3333` |
| 0.4.1 | `2.0, 2.0, 3.0, 3.0` |
| 0.4.2 | `2.3333, 2.6667, 3.0, 3.3333` |

Wrong by whole units, not by an ulp — which is also the reason to expect none shipped: every
bundle here passes a parity gate against torch eager before publication, and an error that size
does not pass a cosine check. This is the one place in 0.4.2 where a graph difference and an
output difference actually coincide.

### #40 — 1D conv transpose loses its static shape under 0.4.1

Both versions emit the same `conv_transpose2d`; they differ in how the trailing dimension
added for the 1D case is dropped afterwards.

```
0.4.1   get_shape -> slice -> reshape      result type: tensor<?x256x?xf32>
0.4.2   shrink_dims                        result type: tensor<1x256x102xf32>
```

Same elements, same order. What changes is that **0.4.1 makes the output dynamically shaped,
and everything downstream inherits that**. `mimi_decoder_float32_ring272_outer_q_gs` is the
bundle this applies to: three `nn.ConvTranspose1d` (the SEANet upsamplers at `m[2]`, `m[5]`,
`m[8]`, `kernel_size=ratio*2`, `stride=ratio`, no `output_padding`), exported under 0.4.1.
Re-exporting under 0.4.2 should make the graph static from each of them onward — a compile and
speed question, not a correctness one.

2D transposed convs with no `output_padding` are byte-identical across the two versions. SAM 3
upscales with `nn.ConvTranspose2d(kernel_size=2, stride=2)` in
`transformers/models/sam3/modeling_sam3.py`, so `sam3_float16` is clear without re-exporting
it.

`output_padding > 0` is the regime #40 actually rewrote — 0.4.1 emulated it with a pre-pad and
a post-crop, 0.4.2 hands it to the native op — and no model here uses it.

### #24 — cannot reach this repo

The fix is real (`axis + rank - 1` where the eager op resolves `axis + rank`, so a per-channel
`axis=-1` was applied one dimension early, silently wrong wherever the two dims are equal), but
it is in the lowering for `torch.ops.coreai.quantize` / `dequantize`. Nothing in `conversion/`
or `models/` calls those. This repo registers its own
`dequantize_per_tensor -> coreai.blockwise_shift_scale` lowering
(`coreai-models/python/src/coreai_models/export/mlir_ops.py`), which is per-tensor and takes no
axis at all.

## The general shape

Reach for the converter, not the artifact, when the question is "did a converter change affect
us". Two versions, one variable, diff the graph. It is minutes rather than a re-export, it
names the mechanism instead of producing a number to interpret, and it answers for constructs
no shipped model happens to contain yet.

But the diff is only half a method, and the half that is sound runs one way. **Graph identical
⇒ output identical. Graph different ⇒ nothing yet.** Of the two changes here that rewrote a
graph's arithmetic, one moved the output by whole units and the other did not move it by a
single bit, and no amount of reading the diff separated them — running them did, in about ten
minutes. Diff to find the short list; run the short list.
