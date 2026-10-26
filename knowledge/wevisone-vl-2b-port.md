# WeVisOne-VL-2B: a vision-language model with a pixel decoder, on Core AI

> 2026-10-25. tencent/WeVisOne-VL-2B (revision `3dad9ce8`, Apache-2.0; WeChat Vision, Tencent Inc.) is built upon
> Qwen3-VL and answers detection, referring localization, counting, pointing, OCR, referring segmentation and alpha
> matting as structured text, plus a 9,511,460-parameter pixel decoder that turns the hidden state of each `<OBJ>`
> token into a matte. The port is four graphs: the zoo's Qwen3-VL decoder with a second output, its fixed-grid tower
> with a third, and the pixel decoder restructured into two static graphs; an app drives them on Core AI's low-level
> runtime. Card: [`models/wevisone-vl-2b/README.md`](../models/wevisone-vl-2b/README.md); scripts:
> [`conversion/wevisone/`](../conversion/wevisone/); records cited as `record …` live under
> `$ZOO_WORK_ROOT/_wevisone/`, outside the repository.

## What was reused and what is new

| part | source | change |
|---|---|---|
| text decoder | the overlay's `qwen3_vl_pipelined.py` through `conversion/export_qwen3_vl_pipelined.py` | a second output `last_hidden` (`mask/mask_hidden.py`); the exporter's import now finds the classes in `qwen3_vl_pipelined.py` |
| vision tower | the same overlay's fixed-grid `Qwen3VLVisionEncoder` | a third output: the four ViT layer outputs the pixel decoder reads, unshuffled to row-major in the graph (`mask/mask_vision_taps.py`) |
| pixel decoder | the package's `WeVisOneMaskDecoder` (Tencent/WeVisOne-VL `40f735b8`), weights from `mask_decoder.pt`, strict load | graph A (once per answer, 16 query rows) and graph B (once per match) in `mask/mask_graph.py`; `grid_sample` as the gather sampler of `conversion/export_rf_detr.py` |
| host | the package's `engine.py` steps | NumPy copies first (`mask/mask_host.py`, `mask_graph.box_fourier_np`), then Swift in CoreAIKit `Examples/Cutout` |
| export wrapper | the Qwen3-VL exporter | `export_wevisone.py`: the revision pinned through `refs/main`, the checkpoint's tokenizer files copied as they are |

## The hidden state the engine does not hand out

The pixel decoder reads, at every `<OBJ>` the model writes, the last decoder layer's output before the final norm.
The decoder graph returns it as a second output `last_hidden` next to `logits` (logits stay output 0, which the
pipelined engine binds); the engine runs that bundle unchanged, with logits bit-equal to the one-output bundle and
decode at 96.5 against 96.6 tok/s on an M4 Max (record `r2/swift_hidden_pf64_timing.json`). But the engine
samples on the GPU and returns ids only, so the hidden state never reaches an app through it. The app therefore runs
the decoder itself on the low-level runtime (AIModel, loadFunction, MutableViews states) and keeps `last_hidden` from
the call that feeds each `<OBJ>` back.

That generation-time state is the one the package computes in a second teacher-forced pass: cosine ≥ 0.999994 between
the two on Core AI with the int8 decoder, so no second pass is needed (it would cost 639 ms per answer on the Mac;
record `r2/hidden_int8hu-pf64.json`).

## The pixel decoder in two static graphs

- **The queries of one answer attend to each other.** The decoder's self-attention runs over all `<OBJ>` rows of an
  answer at once (`nn.MultiheadAttention`, 16 per batch in the package), so one call per object is a different
  computation. Graph A takes up to 16 query rows with a key-padding mask (`where(keep, scores, fill)`); padding
  contents leave the real rows bit-equal on Core AI, and the arm without the mask moves them (record `ROUND2.md`).
- **The levels' sizes are Python ints** and `F.grid_sample` is the gather sampler: torch.export turns the package's
  `.tolist()` sizes into data-dependent symbols, and `aten.grid_sampler_2d` has no Core AI lowering.
- **The alpha cascade is cut after `semantic_s2`**: graph A computes the shared part once per answer, graph B runs the
  S/4 → S/2 → S/1 stages for one match.
- **fp16 is allowed** by the activation absmax over every row (1,381 at the module outputs; inputs: taps 3,023,
  `<OBJ>` states 1,220), and the fp16 graphs keep the mattes within MAE 2.09e-4 and IoU 0.99208 of the package's fp32
  decoder on 46 instances (record `r2/coreai_afp16_bfp16.json`). The Fourier frequencies (2^0 … 2^31) stay an fp32 buffer:
  2^31 overflows fp16.

## One fp32 ulp in a box moves the matte

The box prior is `FourierPE1D`: sin / cos of 2π · v · 2^k for k = 0 … 31 over v = x1, y1, 0.2 log w + 0.5 and
0.2 log h + 0.5. The phases run to about 1e10 rad (1.12e10 at k = 31 over the fixture's boxes), so one fp32 ulp of
difference in v turns the high-k features into unrelated values (k = 12: 2e-3, k = 20: 0.4, k ≥ 22: 1–2; records
`ROUND2.md`, `r2/fourier_sin_probe_boxes.json`).

- **On the GPU it was the log, not the sine.** With the features made inside graph A, 0.2 log w + 0.5 came out one ulp
  off the CPU's on 28 of 92 box terms and one matte fell to IoU 0.906. The GPU's sin / cos of a given fp32 phase stayed
  within 1.2e-7 of float64 up to 1.1e10 rad (record `r2/fourier_sin_probe.json`). Graph A takes the features as an input
  (`box_pe`), made on the host.
- **On the host, `logf` is not torch's `log` everywhere.** A box with integer per-mille corners has 3,494 distinct fp32
  widths (w = fp32(b/1000) − fp32(a/1000) over the 500,500 pairs, plus the 1e-6 clamp). Darwin's `logf` (and NumPy's
  `np.log`) round 7 of them one or two ulp away from torch's CPU `log` (torch 2.9 and 2.11, any tensor length or
  position): the widths 0.17799997, 0.20900002, 0.217, 0.34499997, 0.38299996, 0.60899997 and 0.7260001, on 926 of
  the 500,500 pairs. With libm's value the matte of such a box moved by up to MAE 6.6e-3 and fell under IoU 0.95 for 5
  of the 7 (record `r4/pe_effect_mac.json`). The Swift host carries torch's 7 values and then equals torch on every
  width and pair (record `r4/boxpe_table_check_torch{29,211}.json`); `host/check_boxpe.py` is the full check.
- Check a host input that a Fourier prior amplifies over its whole domain, not over the fixture's boxes: the 46
  fixture boxes matched torch bit for bit before the 7 were found.

## The decoder's accuracy bar sits on the fp16 floor

The bar was fixed before any number, three parts named A, B and D in the records: A (the last four prompt argmax),
B (16 greedy tokens, or a first difference at the reference's knife edge, top-2 softmax gap < 0.1) and D (16
teacher-forced steps at cosine ≥ 0.9999 each), over 33 rows of the fixed 448 grid. int8 per block of 32 passes A and B
on 33/33 and D on none (lowest 0.99581). fp16 weights in fp32 compute reach 32/33 (lowest 0.9998997, at a knife edge),
so D at 0.9999 is at the edge of fp16
weights themselves (record `GATE.md`). The int8 error is spread over the body: in fp32 torch the body alone gives
0.99912 and the head alone 0.99998; four body qschemes (block 32 / 16, with and without clipping) stay at 0.9936–0.9956,
and the four most sensitive of the 28 layers back in fp16 lift the lowest row only to 0.9977 (records `ROUND5.md`,
`logs/qscheme_B*_g14.json`). On the task level the int8 decoder finds every reference box at IoU ≥ 0.75 and every
count (record `SUITE.md`). On the iPhone 18 Pro the device-specialized int8 decoder decodes 2.1 × as fast as fp16
(53.3 against 24.9 tok/s; the JIT keeps int8 there, while the Mac's AOT adds an fp16 copy of the linears).

## The prompt's tail goes in padded

The `_pf64` decoder has `prefill` (S = 64) and `main` (S = 1). The kit's pipelined engine sends the tail under 65
tokens through `main` one token at a time: the last 45 tokens of a 237-token image prompt. Padding the tail with
`<|endoftext|>` (151643) into one more `prefill` call and reading row r − 1 gives the same next token and the same 16
greedy tokens after it, the pad's KV rows never read. On the iPhone 18 Pro a tail of 13 / 45 / 63 tokens takes 30.0 ms
padded against 236.5 / 816.1 / 1,139.3 ms one by one; on the Mac 28 ms against 127–606 ms (records `r4/tail_dev.json`,
`r4/tail_mac.json`). An earlier note measured the opposite on an iPhone 17 Pro with another model (0.99 s padded
against 0.57 s for 19 tokens); it did not repeat on this graph. The app pads; the engine change for the kit is not
made.

## iOS caps a dynamic KV cache at 1,024

On iOS the pipelined engine stops a dynamic-KV decoder at prompt + output = 1,024
(`contextLengthExceeded(0, 1024)`; `CoreAIPipelinedEngine.swift` L1407-1420, a device-compiler miscompile once the KV
axis reaches 2,048, apple/coreai-models#124). The fixed 448 grid makes a 237-token prompt, which leaves 787 tokens for
the answer; the package's own aspect-keeping form makes prompts of up to 2,041 tokens, which do not fit. The app binds
its KV state at 1,024 slots on both platforms.

## The Neural Engine: measured, not shipped

- The dynamic decoder gets 0 ANE regions under the neural-engine preference (Mac h16c), and the vision tower's ANE
  compile fails (`ANECCompileOffline() failed: aneCompileStatus=1`) on the Mac h16c and the iPhone h19p alike; on the
  Mac both run bit-equal to their GPU AOT.
- Apple's static iOS export of the text decoder runs on the iPhone 18 Pro's ANE and passes AneGateRunner on the text
  rows at 6-bit (60.74 tok/s) and 8-bit (45.24 tok/s) (record `ROUND6.md`).
- **With the image path**: the host fills the image rows of `transformer_input`, adds the three deepstack rows after
  layers 0–2 as inputs, and makes the M-RoPE cos / sin itself. Making the M-RoPE inside the static graph (three gathers
  of the RoPE table merged by 0/1 masks) crashes the h19p neural-engine compile (`coreai-build` rc −11 after 16.7 s,
  `CanonicalizeCopyWithConstraints::matchAndRewrite`, `KERN_INVALID_ADDRESS at 0x18`; two layers reproduce it in 1 s),
  while host-made cos / sin inputs compile with an ANE region in all 18 functions (record `ROUND7.md`). 8-bit weights
  (the 152,680-row head stays fp16: it does not split into groups of 32) pass AneGateRunner on the 6 image rows tried:
  a 237-token prompt in 85.9 ms, decode 42.96 tok/s; 6-bit fails 2 of 6 rows (69.7 ms, 62.88 tok/s). These times, and
  the phone's times in the next item, were measured with a system permission alert on the phone's screen; the effect
  is not measured. D is 0/6 for both (record `ROUND9.md`). These graphs return logits only, not the hidden state the
  pixel decoder needs.
- **The pixel decoder**: on the Mac ANE, `PixelShuffle(2)` and `ConvTranspose2d` (k4 s2 p1) return values unrelated to
  torch's (cos 0.365 / −0.005) while conv 3×3, GroupNorm, bilinear and cat + 1×1 agree; `PixelShuffle` written as a
  rank-5 reshape / permute runs on the ANE within the matte bar, and graph A then runs 1.63 × faster on the phone's ANE
  than on its GPU alone (54.7 against 89.3 ms), but the GPU's graph B after it takes 45–55 ms longer, so the request is
  no faster. Graph B's fp32 islands (its resizes and logit sums) fail the ANE compile from the S/2 stage on, and its
  fp16-only form misses the bar in two rewrites of the transposed convolution (record `ROUND8.md`).

## The bytes that ship

- coreai-torch 0.4.1 writes each op's PyTorch source location into `main.mlirb` (paths under the exporting user's home
  directory, `…/torch/nn/modules/…`), and an AOT compiled from it carries them too: 13 of 50 staged files before the
  strip. A text-only grep reports none
  (it skips binary files); `LC_ALL=C grep -a -F` finds them in about a second over 13.5 GB.
- `strip_debug_info` keeps op counts and `coreai-build inspect` equal, and an AOT compiled from the stripped graph runs
  bit-equal to the unstripped one (ids 10/10, hidden states 10/10, mask logits 23/23 on the Mac). `strip_stage.py` run
  again on the same exports gives the same four `main.mlirb` byte for byte; only `.aimodel/metadata.json` changes (its
  `creationDate`).
- A new export of an unchanged graph is not the same bytes: the exporter names its externalized composites and
  resources anew each time (`layer_norm_cchropiy` / `layer_norm_ngrvkyal`, `blocks.0.attn.sdpa_3981e201` /
  `…_7ea34dac`, `resource_<integer>`). The tower exported again matched the staged one in its first 811,102,801 bytes
  (the weights), in `inspect` and in op counts, and differed in the last 12 KB (record `r10/strip_reexport.json`).

## Traps

- **Rebuilding an AOT asset in one Python process does not give the memory back**: about 5.5–7 GB per rebuild; a gate
  that rebuilt every 16 calls (advice from the JIT era) reached 110 GB resident after 54 rebuilds, and the disk's free
  space went to zero. Without rebuilds 300 calls moved the resident size from 9.51 to 9.60 GB. Load once per process.
- **The checkpoint's tokenizer files are in the transformers-5 layout** (`extra_special_tokens` as a list):
  transformers 4.57 fails with `AttributeError: 'list' object has no attribute 'keys'`, and the exporter saves the
  tokenizer after the asset, so the whole export is lost. Copy the three files byte for byte.
- **A revision downloaded with `--revision <sha>` writes no `refs/main`**, and an exporter that calls
  `snapshot_download(repo_id)` offline then cannot resolve it. Pin through `refs/main` and assert it.
- **JPEG decoding is part of the numerics**: ImageIO and Pillow give different pixels (185,078 of 1,204,224 patch
  values on one photo), which moved a knife-edge coordinate by one bin on one row. Gate a host on PNG input.
- **A system permission alert stays on the iPhone's screen until a person answers it**, over every later app, and an
  iPhone UI test cannot answer the first one unattended (`Timed out while enabling automation mode`: the first
  automation session needs the passcode). On a shared phone, inject the denied state to check the app's sentence.
- **Graph B with K instances per call** is 1.11 × faster per instance on the phone at K = 2 but pays a full call for a
  short last chunk and grows the load-time footprint with K; it was not shipped (record `ROUND8.md`).

## Numbers of record

| what | value | record |
|---|---|---|
| iPhone 18 Pro, one cut-out request (1 match / 7 matches, warm) | 0.988–0.997 s / 4.119–4.138 s | `r4/device_docs/cutout-run-*.json` |
| iPhone 18 Pro, first launch | 6.85 s to READY, 8.14 s first request | `ROUND4.md` |
| iPhone 18 Pro, peak footprint | 2.00 GB (app), 1.932 GB (Instruments) | `r4/xt/frame_footprint.json` |
| Mac M4 Max, one match in a 3744 × 5616 NASA photo (decoded at 2731 × 4096) | 0.545–0.554 s | `r4/bench_mac_guppy.json` |
| decoder A / B / D (int8, 33 rows) | 33/33, 33/33, 0/33 (lowest 0.99581) | `GATE.md` |
| suite (box / count / point / OCR CER) | 100.0 % / 100.0 % / 89.4 % / 0.000 | `SUITE.md` |
| mattes on the phone against the package (5 rows, 6 instances) | MAE ≤ 5.81e-4, IoU ≥ 0.9959 | `r4/fixture_dev_oracle_check.json` |
| box encoding against torch (3,494 widths / 500,500 pairs) | 0 / 0 mismatches | `r4/boxpe_table_check_torch{29,211}.json` |
