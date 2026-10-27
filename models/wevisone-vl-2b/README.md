# WeVisOne-VL-2B — Core AI

[🤗 mlboydaisuke/WeVisOne-VL-2B-CoreAI](https://huggingface.co/mlboydaisuke/WeVisOne-VL-2B-CoreAI) (revision `c370b50f`) · Apache-2.0 · source [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B/tree/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf) (revision `3dad9ce8`)

On an iPhone 18 Pro (iOS 27.2), cutting out what a few words name in a photo — each match's box and its alpha matte —
took 0.99–1.00 s for one match and 4.1 s for seven, measured 2026-10-25 in CoreAIKit's `Examples/Cutout` (warm
requests; record `r4/device_docs/cutout-run-*.json`). WeVisOne-VL-2B on Apple's Core AI: the text decoder, the vision
tower and the 9.5M-parameter pixel decoder as four `.aimodel` graphs, converted by
[`conversion/wevisone/`](../../conversion/wevisone/) on coreai-torch 0.4.1 / coreai-core 1.0.0b2. The records this card
cites as `record …` are the conversion's working records; they live under `$ZOO_WORK_ROOT/_wevisone/`, outside the
repository.

The kit example [`Examples/Cutout`](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Cutout) (CoreAIKit,
pull request [#79](https://github.com/john-rocky/coreai-kit/pull/79)) runs the four graphs on an iPhone and a Mac: pick or take a photo, type what to cut out, and save
the cut-out as a PNG with the background removed.

A recorded run on an iPhone 18 Pro — a photo of three lit jack-o'-lanterns among small uncarved pumpkins, "every carved
pumpkin" typed: 3 found, boxes 0.83 s and matte 1.42 s on the screen, only the three carved ones cut out, then lined up
on one sheet and saved to Photos — is the 14.5 s clip
[`demos/wevisone-vl-2b-pumpkins-iphone-20261027.mp4`](https://github.com/john-rocky/coreai-assets/blob/main/demos/wevisone-vl-2b-pumpkins-iphone-20261027.mp4)
in coreai-assets (its `demos/README.md` has the run's numbers; the photo is CC0).

## What it is

WeVisOne-VL-2B, by WeChat Vision, Tencent Inc., is built upon Qwen3-VL (vision: 24 layers, width 1024, deepstack layers
5 / 11 / 17; text: 28 layers, hidden 2048, vocabulary 152,680, an untied head) and answers open-vocabulary detection,
referring localization, counting, pointing, OCR, referring segmentation and alpha matting as structured text; a
9,511,460-parameter pixel decoder turns each object the model writes into a matte (the source `config.json` and card;
the decoder's configuration `configs/wevisone_vl_v1_decoder.json` in Tencent/WeVisOne-VL `40f735b8`). A segmentation
answer reads `<ref>the lighthouse</ref><box><158><222><415><722><OBJ></box>`: four coordinates in per mille, then
`<OBJ>`, whose hidden state the pixel decoder reads together with the box and the vision tower's layers 5 / 11 / 17 / 23.

## Graphs

| graph | form | inputs → outputs | `.aimodel` bytes | `main.mlirb` sha256 |
|---|---|---|---:|---|
| decoder `wevisone_vl_2b_decode_int8hu_pf64` | `main` S = 1 and `prefill` S = 64 sharing two KV states; linears int8 per block of 32 (body `symmetric_with_clipping`, the untied head absmax) | ids, positions, image rows, deepstack rows, rope shift → `logits`, `last_hidden` | 2,455,690,498 | `f62a162875ab…` |
| vision `wevisone_vl_2b_vision_taps_g14` | fp16, a 448 × 448 image (196 rows) | patches [784, 1536] → `image_embeds`, `deepstack_embeds`, `taps` [4, 1024, 28, 28] | 811,114,774 | `d8dd7a3f69c4…` |
| pixel decoder A `wv_mask_a_g28_n16_fp16_hf` | fp16, 16 matches per call, the box encoding from the host | canvas, taps, `<OBJ>` states, boxes, `box_pe`, valid → queries, coarse logits, `semantic_s2` | 19,540,579 | `ffcf2e679798…` |
| pixel decoder B `wv_mask_b_fp16` | fp16, one call per match | canvas, `semantic_s2`, query, coarse logits → `mask_logits` [1024, 1024] | 851,155 | `13bd5dfef16b…` |

`last_hidden` is the last decoder layer's output before the final norm, the tensor the package's `LastHiddenCapture`
hands to the pixel decoder. Each `.aimodel` is its export with the MLIR debug locations removed
([`strip_stage.py`](../../conversion/wevisone/strip_stage.py), which rebuilds these four `main.mlirb` byte for byte from
the exports; record `r10/strip_check.json`). The Hugging Face repository holds them under `ios/` (3,298,811,031 B) and
the same four compiled ahead of time for the Mac under `macos/` (10,188,197,010 B: the compile adds an fp16 copy of the
decoder's int8 linears).

## Decoder accuracy

The bar the port set itself before any number (record `ROUND1.md`) has three parts, named A, B and D in the records,
against the package in fp32 on the same fixed grid, over 33 rows: A, the argmax at the last four prompt positions; B, a
16-token greedy run that never leaves the reference's tokens, or leaves them only where the reference's own top-2
softmax gap is under 0.1 (a knife
edge); D, the logits of 16 teacher-forced steps within cosine 0.9999 of the reference at every step. The int8 decoder
passes A on 33 of 33 rows and B on 33 of 33 (all 16 tokens equal on 18 rows; the other 15 part at the reference's knife
edges), and misses D on all 33: its lowest step cosine is 0.99581 (record `GATE.md`, `oracle/gate_decoder_int8hu_g14.json`).
No decoder with fp16 weights passes D on every row either: fp16 31 of 33 (lowest 0.999811), fp16 weights with fp32
compute 32 of 33 (lowest 0.9998997, at a knife edge). In torch, the int8 body alone accounts for it (body only
0.99912, head only 0.99998), and putting the 4 most sensitive of the 28 layers back in fp16 still leaves the lowest
row at 0.9977 (record `ROUND5.md`). This port ships the int8 decoder on the A, B and task-level gates below: on the
structured-output suite it finds every one of the reference's boxes at IoU ≥ 0.75 and gives every count exactly.

## How the app drives it

The pipelined engine samples on the GPU and returns ids only, so the `<OBJ>` hidden state never reaches an app through
it. `Examples/Cutout` runs the four graphs on Core AI's low-level runtime (AIModel, loadFunction and MutableViews
states) and does the package's host steps in Swift: Pillow's bicubic resize in integer arithmetic (the same bytes as
Pillow on PNG input), the chat template with the system line "You are a helpful assistant.", greedy decoding that keeps
`last_hidden` at each fed `<OBJ>`, the package's box parser, the box Fourier encoding in fp32, part A once per 16
matches, part B once per match, then sigmoid and a bilinear resize to the photo. The prompt goes in as 64-token
`prefill` calls with the tail padded into one more call: on the iPhone 18 Pro the 237-token prompt takes 137–158 ms
this way, and a 13-, 45- or 63-token tail takes 30 ms padded against 237–1,139 ms one token at a time, with the same
next token and the same 16 greedy tokens after it (record `r4/tail_dev.json`).

The box encoding is sin / cos of 2π · v · 2^k for k = 0 … 31, so one fp32 ulp in an input moves the high-k features to
unrelated values. Inside a GPU graph the term 0.2 · log(w) + 0.5 came out one ulp off the CPU's on 28 of 92 box terms
and one matte fell to IoU 0.906 (record `r2/fourier_sin_probe_boxes.json`), so the host computes it. Darwin's `logf`
and torch's CPU `log` round 7 of the 3,494 widths a box with integer corners can have one or two ulp apart (926 of
500,500 corner pairs); the app carries torch's values for those 7, and its encoding then equals torch's on every width
and every pair (record `r4/boxpe_table_check_torch{29,211}.json`).

## Measured

iPhone 18 Pro (iPhone19,2, iOS 27.2 24B5099f), the app's Release build, the four `.aimodel` graphs specialized on the
phone (GPU), 2026-10-25, thermal state nominal, three warm requests each (records `r4/device_docs/cutout-run-*.json`,
`ROUND4.md`):

| request | total | parts |
|---|---|---|
| the lighthouse example (1024 × 683), "the lighthouse", 1 match, 13 tokens | 0.988–0.997 s | resize to 448 17 ms, tower 89 ms, prompt (237 tokens) 141 ms, answer 233 ms, resize to 1024 and the mask inputs 26 ms, part A 75 ms, part B 403 ms (one request) |
| the chess example (960 × 640), "the light chess pieces", 7 matches, 50 tokens | 4.119–4.138 s | answer 957 ms, part B 2,771 ms for the seven (one request) |
| first launch after installing | 6.85 s to READY, 8.14 s for the first request | the phone specializes each graph once; later launches reached READY in 1.5–2.6 s |

The app's memory footprint peaked at 2.00 GB (its own record; Instruments: 1.932 GB). Mac M4 Max (macOS 27.0 26A428,
the `macos/` graphs): one match in a 3744 × 5616 NASA photo (decoded at 2731 × 4096) took 0.545–0.554 s, the PNG another
0.29 s; three matches ("people") in another NASA photo 0.663 and 0.666 s, with one run of the three at 1.225 s; peak
footprint 2.8 GB (records `r4/bench_mac_guppy.json`, `r4/bench_mac_people.json`).

## Gates

The reference is the package in fp32 on the CPU (transformers 5.12.1, one thread) with the photo resized to 448 × 448
first (the fixed grid of these graphs), greedy with no repetition penalty. The fixture set is 33 rows over 17 images
(two drawn images, four NASA public-domain photos, and eleven images from the authors' repository — crops of its
case-study figures and its demo photo — used for parity only and not redistributed), plus four rows with several
objects. The bars were written before the runs; a matte passes at MAE ≤ 0.01 and IoU(α > 0.5)
≥ 0.95 per instance.

| gate | what | result | record |
|---|---|---|---|
| fp32 torch | the zoo's Qwen3-VL decoder and tower against the reference: M-RoPE positions, tower outputs, 16 decode steps | positions equal everywhere, tower cosine 1.000000, prefill top-1 33/33, decode 486/486; two red arms red on 33/33 | `GATE.md` |
| decoder, Mac GPU | A and B above, on the `main` function (D: [Decoder accuracy](#decoder-accuracy)) | A 33/33, B 33/33 | `oracle/gate_decoder_int8hu_g14.json` |
| decoder, `prefill` | the prompt in 64-token chunks against `main` and the reference | last-four argmax = the reference's 105/105 off knife edges; logits cosine ≥ 0.999956 to `main` | `oracle/gate_prefill_int8hu_g14.json` |
| `last_hidden` | the two-output decoder against the one-output one, and its `<OBJ>` states against the reference's | logits bit-equal on 10 rows at every step; `<OBJ>` state cosine ≥ 0.99863 | `r2/hidden_int8hu-pf64.json` |
| tower | `image_embeds` / `deepstack_embeds` against the reference | cosine ≥ 0.999869 / ≥ 0.999704 (bar 0.999) | `oracle/suite_coreai_g14.json` |
| taps | the tower's third output against the reference's layer maps, and the matte through them | cosine ≥ 0.998621 (39 of 40 maps ≥ 0.999); mattes within MAE 5.15e-5, IoU ≥ 0.99737 | `r2/vision_taps_g14.json` |
| mask graphs | parts A + B in fp16 on the reference's inputs against the package's fp32 decoder | 20 rows / 46 matches: MAE ≤ 2.09e-4, IoU ≥ 0.99208 | `r2/coreai_afp16_bfp16.json` |
| suite | the structured answers of the int8 decoder against the reference | boxes found at IoU ≥ 0.75: 100.0 % (63); counts exact 100.0 % (7); points within 2 %: 89.4 % (47); OCR CER 0.000 | `SUITE.md` |
| Swift host, Mac | the app's code on the PNG rows: input tensors, prompt ids, answers, mattes | patches and canvas byte-identical on 7/7 PNG rows, prompt ids 10/10; with the reference's own tensors the Swift chain gives the Python chain's answers and mattes on 10/10 rows; mattes of the shipped form on the rows whose boxes equal the reference's: MAE ≤ 8.10e-4, IoU ≥ 0.99581 | `r4/fixture_mac_check.json`, `r4/fixture_mac_d1_check.json` |
| iPhone 18 Pro | the same rows in the app on the phone against the Mac | tokens equal on 9/10 rows (the tenth parts where the Mac's top two logits are equal in fp16); mattes against the Mac MAE ≤ 6.67e-4, IoU ≥ 0.99794; against the reference MAE ≤ 5.81e-4, IoU ≥ 0.9959 | `r4/fixture_dev_check.json`, `r4/fixture_dev_oracle_check.json` |

## Limits

- **A fixed 448 × 448 grid.** The package keeps the photo's aspect ratio (smart-resize, up to 2025 × 32² pixels); these
  graphs stretch it to 448 × 448. Against the package's own form, the fixed grid's fp32 answers find 85.5 % of its boxes
  at IoU ≥ 0.75, 71.4 % of its counts and 81.8 % of its points, OCR CER 0.070 (record `SUITE.md`, table ii).
- **1,024 positions on iOS.** Core AI's pipelined engine stops a dynamic-KV decoder at 1,024 positions of prompt and
  answer on iOS (`CoreAIPipelinedEngine.swift` in apple/coreai-models: a device-compiler miscompile once the cache reaches
  2,048 positions, apple/coreai-models#124); the app binds its cache at 1,024 on both platforms: 237 prompt tokens and
  up to 787 answer tokens. The package's own form writes prompts of up to 2,041 tokens, which an iPhone cannot hold.
- **No repetition penalty.** The package's greedy decoding uses 1.05; the app has none. In fp32 the two agree on 26 of
  33 answers, every box and count the same, 93.6 % of points (record `SUITE.md`, table iii).
- **JPEG decoding.** ImageIO and Pillow decode JPEG to slightly different pixels, which moved one box coordinate by a bin
  on one row (`nasa_jungle_mat`: y2 1000 against 994); PNG pixels are identical (record `r4/fixture_mac_check.json`).

## Neural Engine (measured, not in the bundle)

The decoder in Apple's static iOS form, with the image rows, deepstack rows and M-RoPE cos / sin passed in by the host,
runs on the iPhone 18 Pro's Neural Engine: with 8-bit weights (the head left in fp16) it passed the AneGateRunner check
(the zoo's Neural Engine parity check against the fp32 reference) on the 6 image rows tried and took a 237-token image
prompt in 85.9 ms, decoding at 42.96 tok/s; with 6-bit weights it failed 2 of the 6 rows (69.7 ms, 62.88 tok/s). These
times were measured with a system permission alert on the phone's screen; the effect is not measured. D was 0 of 6 for
both (records `SPEED_LADDER.md` rows 13 and 13d, `ROUND7.md`, `ROUND9.md`). These graphs return logits
only, and the pixel decoder needs the last hidden state at each
`<OBJ>`. The mask graphs stay on the GPU: part B's fp32 islands fail the Neural Engine compile, and part A on the
Neural Engine was faster alone but made the request no faster (record `ROUND8.md`).

## Convert / verify

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0,
transformers 4.57.6) for export, AOT and the gates; the reference in its own venv (Python 3.12, transformers 5.12.1,
torch 2.11.0, the package by path at `40f735b8`); the snapshot at `3dad9ce8` in the Hugging Face cache with
`refs/main` naming it and `HF_HUB_OFFLINE=1`; `DEVELOPER_DIR` = Xcode 27.0 RC for `coreai-build`. The recipes are
[`recipe.toml`](recipe.toml); every script, in order: [`conversion/wevisone/README.md`](../../conversion/wevisone/README.md).

Port notes: [`knowledge/wevisone-vl-2b-port.md`](../../knowledge/wevisone-vl-2b-port.md).

## License

Apache-2.0, as the source. The source's `LICENSE` names Qwen3-VL-2B-Instruct (Alibaba Cloud, Apache-2.0) as a component
modified by Tencent; the Hugging Face repository carries that file unchanged.
