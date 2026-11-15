# clef-flash: a hidden-state decision model and its head on Core AI

> 2026-10-03. Cloudflare/clef-flash (revision `17f0b0a`, Apache-2.0) answers typed questions about a
> state (text, JSON or an image) with a probability for every option, read by a 121.8M-parameter joint
> schema head from the backbone's final hidden states. The port is a 9B ids-input decoder that returns
> hidden rows, a fixed-grid tower from shipped code, the head re-written as a graph, and the untied
> `lm_head.weight` as a host table. Card: [`models/clef-flash/README.md`](../models/clef-flash/README.md).
> Transcripts below are in `models/clef-flash/`; records outside the repo are marked "local" and live
> under `$ZOO_WORK_ROOT/_clefflash/`. Mac M4 Max, macOS 27.0 26A428, Xcode 27.0 RC.

## What was reused and what is new

| part | source | change |
|---|---|---|
| vision tower | overlay `qwen3_5_vision.Qwen3_5VisionEncoder`, `conversion/export_qwen38vl_pipelined.py --skip-decoder --vision-dtype fp16w32` | none: the HF id and the grid (8×8 / 14×14) by flag |
| decoder | decider-2b-vision's `conversion/decider_vision/qwen3_5_vl_pipelined.py` (ids in, image rows as a static input, M-RoPE in the graph) | subclass `conversion/clef_flash/qwen3_5_clef_decoder.py` at 9B: no vocabulary head, the final-norm hidden state at every position out, one static-S function, a 1,024-row image buffer |
| head | none: the author's `JointSchemaHead` loops over records, questions and options in Python | `conversion/clef_flash/clef_head.py` `ClefHeadGraph`: the author's own submodules, the loops written as matrices, ten tensor inputs |
| lm_head table | none | `export_lm_head_table.py`: the untied table as raw fp16 for the host's gather |
| host | decider-2b-vision's `host.py` form | new `conversion/clef_flash/host.py` (the author's `encode_record()`, Pillow's resize) and `decide.py` (the whole read-out, the SystemOne-compatible response) |
| readout gates | the decider-2b-vision method (AOT `.aimodelc`, fresh states, a reset re-run per process) | new `readout_gate.py`, `gate_head.py`, `int8_bisect_torch.py`, `gate_swift.py` |
| Swift host | `apps/DeciderVision`'s low-level runtime pattern | new `apps/ClefFlash`; the image resize copies Pillow's integer resampler, not DeciderVision's float one |

The checkpoint's `encode_record()` puts the image block right after a fixed 36-token prefix, so the
decider-2b-vision decoder's image-first M-RoPE derivation applies as is (image block at index 36,
`rope_shift_start` = 37 + N).

## The decoder returns hidden rows, not logits

The head reads `last_hidden_state` at every position, not one slot's logits. The decoder graph
therefore has no vocabulary head and outputs `hidden [1, S, 4096]` fp16 for every position of each
call. A record of T ids runs as ⌈T / S⌉ calls of one function from zeroed states; the last call is
padded with `<|endoftext|>` (248044) and its padded rows are dropped. The decoder is causal, so a
padded position cannot reach a real one. There is no S=1 function: nothing is generated. The four
states at a 4,096-token context are 160.6 MB of fp16 (`keyCache` / `valueCache` [8, 1, 4, 4096, 256],
`convState` [24, 1, 8192, 3], `recState` [24, 1, 32, 128, 128]); zeroing them per record takes
1.3 ms in Swift (`gate-clef-flash-timing.json`).

S is a static chunk with the Gated DeltaNet recurrence unrolled in the graph. In fp32 torch the chunk
width changes nothing that matters: S = 32 and 64 against 16 move the hidden state by at most 1.6e-3
and p by 8.6e-7 (`gate-clef-flash-torch-parity.json` P5). On the GPU, S = 64 costs 134.2 ms per call
and 2.26 ms per token against 58.8 ms and 3.77 ms at S = 16 (202 warm runs, a shared GPU); S = 128
gave 2.14 vs 2.19 ms per token on 19 runs and 1.227 vs 1.210 s per run, the longer padding eating
the gain (`gate-clef-flash-readout-fp16_pf64.json` `compare_with`, `gate-clef-flash-variants.json`).
The shipped bundles use S = 64.

The untied `lm_head.weight` is read by the head, not by the decoder: the module's `load_report` lists
it as the one checkpoint key it leaves unread on purpose.

## The fixed grid's price

The author's processor picks an image grid per image (at least 256×256 pixels). A graph needs a fixed
one. `conversion/clef_flash/grid_price.py` priced it with the author's code alone, fp32 on the CPU,
on the fixture's 14 image records and 50 questions: the image resized to a square tile first,
against the processor's own grid (`gate-clef-flash-variants.json` `grid_price`):

| arm | image tokens | argmax = native | max \|Δp\| | mean \|Δp\| | gold correct |
|---|---:|---:|---:|---:|---:|
| native | 412 (mean) | — | — | — | 35/38 |
| 256×256 | 64 | 46/50 | 0.959 | 0.039 | 33/38 |
| 448×448 | 196 | 48/50 | 0.436 | 0.018 | 36/38 |
| 672×672 | 441 | 48/50 | 0.350 | 0.016 | 36/38 |
| 896×896 | 784 | 48/50 | 0.244 | 0.013 | 36/38 |

The flips are not one image: `img_08` `rooms` (a 1024×768 floor plan) flips at 256, 448 and 672 but
not at 896, where `photo_01` `limes` flips instead; `img_01` `clutter`, a near-tie on the processor's
own grid, flips from 448 up. The rule written before the large grids ran: a third tower only if a
larger grid agreed with `native` more often than 448. None did, so 256 and 448 ship.

## fp32 torch parity before any graph

The decoder module was gated in fp32 torch, driven the way the graph runs (chunks from zeroed states,
the unrolled recurrence, M-RoPE in the module, the oracle's own tower rows), read through the
author's head (`gate-clef-flash-torch-parity.json`):

- P1, every oracle run on the CPU: 214 runs, argmax 405/405 (the four near-ties included), max |Δp|
  9.6e-6, lowest per-position hidden cosine 0.99999939; its M-RoPE planes equal the oracle's on
  214/214. The author's head re-run on the oracle's own hidden reproduced the oracle's logits bit for
  bit on 214/214, so every |Δp| is the decoder's.
- P2: with zero image inputs the hidden state equals the overlay's plain Qwen3.5 text decoder
  (`Qwen3_5StatefulForCausalLM.model.forward_stateful`, loaded separately) bit for bit at every
  position, and every chunk × layer output hashes equal (3/3 runs).
- P3, per layer: the layer-0 input is bit-equal; the largest |d| is 0.016 at layer 31, where the
  residual stream reaches 2,269 (`own_t01`) / 2,535 (`img_01` `g256`).
- P4, red arms: zeroing the image rows changes 1–2 answers per image run (max |Δp| 0.53–0.75);
  swapping the (row, col) table moves p by 0.006–0.062, 1-D positions by 0.007–0.075 (one answer on
  `img_01` `g256` flips), the post-image shift off by one by 0.0008–0.0014. Positions are gated by the planes'
  equality, as in decider-2b-vision, because a |Δp| bar would let some of these through.
- CPU vs MPS in fp32, 5 runs: max |Δp| 1.1e-6, hidden max |d| 3.3e-3.

The host spec rebuilt every run's ids, spans, option order and planes with two tokenizer
implementations (242/242 runs, the larger grids included). A one-word change in one question moves
its span (124–131 → 124–135) and turns the run red.

## fp16 on the GPU: the probabilities pass, a few rows do not

The fp16 decoder passes the bar with room (213 runs, max |Δp| 0.0112, mean 0.00054), but some hidden
rows drift. Against the oracle, 50 of 73,343 positions have a cosine below 0.99 (34 of them image
rows), 6 below 0.9, one below 0.5: 0.41 at position 134 of `photo_04` `g448`, an image row of a run
whose max |Δp| is 1.1e-4 (`gate-clef-flash-readout-fp16_pf64.json`). The decisions do not move: the
head averages rows over spans and the image rows enter through memory attention.

What was measured about the cause:

- The fp16 module in torch on MPS (round 3, S = 16, 4 runs) finds the same rows hardest but keeps them
  higher: 0.877 at that position against the graph's 0.413; graph vs fp16 torch 0.321 there
  (`gate-clef-flash-variants.json` `fp16_torch_crosscheck`).
- The full-attention linears and SDPA in fp32 (variant `fp16attn32`, 38 runs) do not lift it: lowest
  0.378 vs 0.407, 53 vs 48 positions below 0.99, max |Δp| 0.0075 in both, 7% slower per call.
- The residual stream's absmax over every fixture run is 2,673.5, far inside fp16's 65,504
  (`activation_absmax`).

Which op in the graph opens the gap is not isolated: the graph has no per-layer outputs, and the next
candidates (the final RMSNorm, the Gated DeltaNet projections in fp32) were not tried.

## int8: the error lives in the first sixteen layers

int8 per block of 32 over every decoder linear (int8lin) fails: max |Δp| 0.0748 on a SemIf record
(fp16: 0.0105 on the same record), 9 of 213 runs above 0.02, a near-tie flipped, 336 positions below
cosine 0.99 against fp16's 50 (`gate-clef-flash-readout-int8lin_pf64.json`).

`conversion/clef_flash/int8_bisect_torch.py` evaluates configurations in fp32 torch with the
exporter's own int8 weights, read back through the finalized module's dequantization, on the six runs
int8lin misses worst. It reproduces the graph (int8lin 0.0842 on the worst run, the graph 0.0748;
all exact 0.0000). The rule was written to `bisect/rule.json` before any result (sha256 `2879ed1e…`):
worst run ≤ 0.010, no run worse than its int8lin value, at most six fp16 layers, the smallest set
(`gate-clef-flash-int8-bisect.json`):

| configuration | worst-run max \|Δp\| |
|---|---:|
| all int8 | 0.0842 |
| layers 0–15 fp16 | 0.0023 |
| layers 16–31 fp16 | 0.0845 |
| only the Gated DeltaNet projections int8 | 0.0752 |
| only the MLP linears int8 | 0.0342 |
| only the attention linears int8 | 0.0230 |
| one fp16 layer, the most helpful (6) | 0.0492 |
| one fp16 layer, layer 3 | 0.1164 |
| the ranked sets of 2–6 fp16 layers | 0.0260–0.0350 |

No set met the rule: every ranked set also made some run worse than int8lin. The map run after the
search, outside the rule and never selecting from it, gave contiguous fp16 blocks: 0–7 0.0568, 0–8
0.0509, 0–9 0.0232, 0–10 0.0116, 0–11 0.0069, 4–15 0.0207, 8–15 0.0296, 12–15 0.0939; with layers
0–15 one kind kept exact, Gated DeltaNet 0.0229, MLP 0.0464, attention 0.1151. The error is spread over
the first layers and partly cancels between kinds of linear.

int8mix keeps layers 0–11 fp16 (12 layers, outside the cap). On the fixture: 397/397 + 4/4, max 0.0153
(the same SemIf record). Its test is the held-out set, written afterwards: 186/186, one of two
near-ties flipped (`ho_t07`, oracle margin 0.0117), max 0.0171 (`gate-clef-flash-heldout.json`). It
saves 4.05 GB against fp16 and no time: 134.4 ms per call against fp16's 134.2 and int8lin's 134.3
on the same 202 runs. Why int8 weights buy nothing at S = 64 is not isolated.

## The head as a graph: buckets, not dynamic shapes

The author's head does not export as is: `torch.export` stops at the record loop
(`GuardOnDataDependentSymNode` at `joint_schema_model.py:355`). `ClefHeadGraph` keeps the author's
submodules (`load_state_dict(strict=True)`) and writes one record's loops as matrices: span means as
`q_avg [Q, T]` / `o_avg [O, T]` rows of 1/len, the global vector as `g_avg [1, T]`, option-to-question
membership as `member [Q, O]`, the attention blocks as plain matmul / softmax with the in_proj rows
split q / k / v. In eager fp32 it equals the author's head within 3.7e-6 logit on 214 runs, 3.5e-6
padded to the buckets (`gate-clef-flash-head.json` `h0`).

A dynamic-shape export (T 64–4,096, Q 1–16, O 2–128) converts in 5 s. Its AOT asset built with
`--expect-frequent-reshapes` loads and then aborts at the first call, at the traced shape too:

```
GPUMemrefOps.mm:164: failed assertion `Failed to resolve dynamic dimensions for memref.alloc'
```

Built without the flag it runs and matches the eager head (≤ 9.5e-7 logit), but every new input shape
pays a first call of 0.27–1.5 s in a process (`export.dynamic_without_efr`). The shipped form is
one asset with four static functions sharing the weights, `t512` / `t1024` / `t2048` / `t4096` at Q 16,
O 128; the host pads T to the smallest that holds it with zero rows and `key_valid` 0. On the oracle's
fp32 hidden: 214 runs, argmax 405/405, max |Δp| 1.8e-7; second calls 6.0 (t512) to 19.0 ms (t4096)
on a shared GPU. Three perturbed arms show the gate can fail: span rows one token late, the lexical
term zeroed and option membership rotated move p by up to 0.056, 0.055 and 0.348 (`h2_fixture.red_arm`).

## Attention over 4,032 keys on the GPU

The first bucket asset matched the author's head in `t512` / `t1024` / `t2048` and missed in `t4096`:
on `own_t01` and `own_t15` by 0.35–0.95 logit, different on every call. Isolated on those two runs and
on random tensors (`gate-clef-flash-head.json` `key_block_probes`):

- Eager fp32 padded to 4,096: exact (1.2e-6). The padding math is not it.
- Bucket functions at 3,072 keys: exact; at 4,032 and 4,096: wrong and changing call to call.
- Intermediates at 4,096: memory, the span means and the global vector exact; the first evidence-routing
  layer's output 10–14% off.
- The bare chain `softmax(q kᵀ · s + b) v` as its own graph: exact at 3,072 and 3,584 keys, wrong at
  4,032 / 4,096 / 8,192 for 16, 8 and 1 heads and 128, 64 and 16 queries, with or without the bias.
- Each op alone (`q kᵀ`, the max and sum over T, softmax over T, `a v` with K = T) is exact at 4,096.
- Keys in blocks of at most 2,048 with one shared max: exact (5.7e-7) and repeatable at 4,032 and 4,096.

`clef_head.py` uses the blocked form when the static key length exceeds 2,048 (`KEY_CHUNK`); only
`t4096` changes (it gains `amax` / `maximum`). Where in the compiled chain the error enters is not
isolated. The decoder's own attention does not show it: a 4,050-token row (a diagnostic, not a gate
set) whose last calls attend 4,032 and 4,096 keys has one position of 4,050 below cosine 0.99 (index
1,300, 0.9585), max |Δp| 6.1e-4, and `decide.py` agrees with the author on it (3/3 choices).

## The lm_head table

The head's lexical term averages `lm_head.weight` rows (untied, `[248320, 4096]` bf16, 2.03 GB in
fp16) over each option span. The host keeps the table as raw fp16 and gathers rows. Every bf16 value
in fp16's normal range is exact in fp16; 3,317,193 elements fall below it and 1,649 become 0. At the
head, over every oracle run (`gate-clef-flash-variants.json` `lm_head_table_effect`):

| table | bytes | max \|Δp\| | argmax |
|---|---:|---:|---:|
| fp32 (re-run floor) | — | 1.8e-7 | 405/405 |
| fp16 (ships) | 2,034,237,440 | 1.8e-7 | 405/405 |
| int8 per row | 1,017,615,360 | 4.4e-4 | 405/405 |
| int8 per block of 32 | 1,080,688,640 | 2.5e-4 | 405/405 |

## Swift host notes

- **Resize.** The processor resizes with Pillow's BICUBIC. DeciderVision's float copy of Pillow's pass
  order (`host.resize_bicubic`) lands 1–2 levels off Pillow on 12 of the 28 fixture tiles (4 at `g256`,
  8 at `g448`; `gate-clef-flash-torch-parity.json` `host.pixel_values`). `ImagePreprocess.swift` copies
  Pillow's integer resampler instead (`Resample.c`: 22-bit fixed-point weights, the horizontal pass
  first, a uint8 intermediate): 48/48 tiles bit-equal to Pillow, the patches equal to the oracle's
  `pixel_values` (`gate-clef-flash-swift.json` `pixels`).
- **The prompt is Python's JSON.** `render()` is `json.dumps(sort_keys=True, ensure_ascii=False,
  separators=(",", ":"))`, so the tokens depend on Python's number formatting. `JSONValue.swift` keeps
  each number literal as written (`1250` ≠ `1250.0`), `Renderer.swift` sorts keys by code point and
  formats floats as Python's `repr`; the response rounds with Python's `round(x, 4)`. Against CPython:
  2,204 rendered values, 17 raw literals and 199,994 doubles equal (`render_check`). A lone surrogate
  escape (`"\ud800"`), which Python keeps in a `str`, is refused; the fixture has none.
- **float32 softmax.** The per-question softmax is float32 in NumPy's order (`expf`, pairwise float32
  sums, float32 division), so the Swift probabilities equal the Python ones recomputed from the same
  logits on every run, and the 4-decimal responses equal `decide.py`'s on 24/24.
- **Borrowed buffers.** `InferenceFunction.MutableViews` borrows what it holds until `run` returns;
  `ClefDecoder` keeps the four states and the hidden buffer in a class property and moves them into
  locals for a record's whole pass, the DeciderVision form.
- **JIT.** The fp16 `.aimodel` specialized by the Swift runtime computes what the AOT `.aimodelc` does,
  bit for bit on 253/253 runs. int8mix's JIT does not match its AOT on any run (max |Δp| 0.0014, max
  |Δlogit| 0.020, argmax 589/589); both pass the bar, and the cause is not isolated. The first
  specialization took 56.1 s for fp16 (the decoder 50.0 s) and grew the runtime's cache by 31.8 GB;
  int8mix 42.3 s and 17.0 GB (`gate-clef-flash-swift.json` `jit_vs_aot`).
- **Not tested:** JPEG input (the fixture is PNG; ImageIO's JPEG decode is not guaranteed to equal
  libjpeg's) and a native-grid image through the Swift tower (the towers are `g256` / `g448`; the
  `native` runs were fed the oracle's rows).

## Numbers of record

Swift, Release CLI, AOT, the machine-wide GPU lock taken with the GPU at 0 %, two passes per bundle
(`gate-clef-flash-timing.json`):

- A decision: 0.41 s at 185 tokens, 0.95 s at 436, 2.17 s at 967, 5.29 s at 2,494, 5.56 s at 2,603;
  an image at `g256` 1.14–1.27 s, at `g448` 1.46–1.59 s. int8mix within 0.01 s of fp16 on every row.
- A decoder call (64 tokens) 134.2 ms fp16, 134.3 ms int8mix; the head 7.4 (t512), 12.1 (t1024),
  27.7 ms (t4096); the tower 40 (`g256`) / 87 ms (`g448`); image decode, resize and patches 9–13 ms.
- Load from the runtime's cache 3.3 s (the decoder 2.3 s, the tokenizer 0.6 s); the first decision of
  a process 3.4–3.7 s, its first decoder call 2.6–2.8 s.
- The Python gates on a shared GPU measured the same decoder call: 134.2 ms (fp16, 1,208 warm calls).
