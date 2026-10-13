# MiniCPM-V-4.7-1B — Core AI

[🤗 mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI) · Apache-2.0 · source [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/tree/984a89aafcd5160164f4f0d04b4be0f4eec03cb8) (revision `984a89a`)

MiniCPM-V-4.7-1B on Apple's Core AI: a vision tower and a decoder as two `.aimodel` graphs, the decoder driven by
Core AI's pipelined GPU engine, converted by this repository's [`conversion/minicpmv47/`](../../conversion/minicpmv47/)
on coreai-torch 0.4.1 / coreai-core 1.0.0b2. On an iPhone 18 Pro (iOS 27.2) the int8 decoder generates 98.0 tokens
per second, measured 2026-10-13 ([`gate-minicpm-v-4.7-1b-iphone.json`](gate-minicpm-v-4.7-1b-iphone.json)).

<!-- gen-cards:use-it begin id=minicpm-v-4.7-1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
(generated after the kit enrollment)
<!-- gen-cards:use-it end -->

## What it is

OpenBMB's MiniCPM-V-4.7-1B reads an image and a question and answers in text. It follows MiniCPM-V-4.6
([`minicpm-v-4.6`](../minicpm-v-4.6/README.md)) in the same family, and the two share their vision code: a 27-layer
SigLIP-type tower (width 1152, patch 14) with a window-attention merger after layer 6 and a 2×2 downsample MLP, 64
tokens for a 448×448 image, spliced into a Qwen3.5 hybrid text decoder (24 layers: 18 Gated DeltaNet + 6 full
attention, hidden 1024, vocabulary 248,094; the source `config.json`). In transformers 5.19.0 the 4.7 vision classes
are 4.6's line for line after renaming (lane `FEASIBILITY.md` §r1; "lane" records live under
`$ZOO_WORK_ROOT/_minicpmv47/`, outside the repository). Three things change for a port:

- **Positions are canvas M-RoPE.** 4.6 used plain 1-D positions. 4.7 gives every token three planes (T, H, W,
  interleaved sections [11, 11, 10]): the image sits on a canvas, and the text after it resumes past the canvas. The
  decoder here derives the planes inside the graph ([below](#how-the-vlm-rides-the-text-only-engine)).
- **The head is untied.** `lm_head.weight` is its own tensor ([248094, 1024], 0.51 GB in the checkpoint's bf16; lane
  `LANE.md` §1), within max |Δ| 4.9e-4 of the embedding (row cosine ≥ 0.99998). The embedding used as the head moves
  P0's logits by up to 0.196 and reorders the top-5 at 2 of 48 positions (lane `oracle/parity_text.json`), so the
  bundle carries the checkpoint's own head.
- **The prompt side.** The processor also returns `mm_token_type_ids` (not needed here: the decoder reads the image
  from the ids), generation stops on either 248044 or 248046, and the chat template has an `enable_thinking` switch.
  This port runs with thinking off.

## Bundles

[mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.7-1B-CoreAI), three folders
under `gpu-pipelined/`. An answer about an image needs the decoder and one tower. **Recommended: the int8hu decoder +
the int8lin tower**, 1,942,089,859 bytes together. The fp16 tower is the exact one (per-token cosine to the fp32 tower:
min 0.999957) at 474 MB more.

| folder | what | weights | `.aimodel` bytes | main.mlirb sha256 |
|---|---|---|---:|---|
| `minicpmv47_vlm_decode_int8hu/` | decoder (recommended): `.aimodel` + `metadata.json` + `tokenizer/` | linears int8 per block of 32 (body `symmetric_with_clipping`, the untied head `symmetric`); embedding, conv and norms fp16 | 1,308,109,419 | `ae9609fde02d…` |
| `minicpmv47_vision_int8lin/` | tower (recommended): 448×448 → 64 rows, `.aimodel` + `metadata.json` | `nn.Linear` int8 per block of 16; LayerNorm, conv and embedding fp16; fp16 math | 613,967,602 | `40265d0c06d9…` |
| `minicpmv47_vision_fp16/` | tower, exact, same layout | fp16 weights and math | 1,087,749,442 | `0b3ece089d7b…` |

Folder totals: decoder 1,328,120,899 B (its `tokenizer/` is the snapshot's `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json`, byte for byte), int8lin tower 613,968,960 B, fp16 tower
1,087,750,729 B (lane `r6/STAGE.md`, `r6/strip_*.json`). The int8 block size of the tower is 16 because `mlp.fc2`
takes 4,304 inputs, which 32 does not divide.

Each `.aimodel` is the export with its MLIR debug locations removed: coreai-torch 0.4.1 writes the exporting machine's
source paths into `main.mlirb`, and `strip_debug_info` (the method of
[`conversion/d1_omni/strip_ship.py`](../../conversion/d1_omni/strip_ship.py)) takes them out. The stripped graphs have
the export's op counts (3,326 / 1,967 / 1,613) and `coreai-build inspect` summary, their AOT h16c compiles hold the
export's `resources.bin` byte for byte, and the stripped pair re-ran the end-to-end gate on the Mac with the export's
greedy ids on all four image prompts (lane `r6/regate_stripped.json`). The timing and iPhone numbers below were measured
on the exports before the strip. No AOT asset is in the upload: the iPhone numbers are the phone's own specialization
of the `.aimodel`.

## How the VLM rides the text-only engine

The engine runs `ids + positions → logits` over states and knows nothing about images. The image enters as three
static inputs that the host writes once per image and the engine binds on every step
([`apps/coreai-pipelined-static-inputs.patch`](../../apps/coreai-pipelined-static-inputs.patch)); the Gated DeltaNet
conv and recurrent states ride [`apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch).
The compiled `main` function ([`gate-minicpm-v-4.7-1b.json`](gate-minicpm-v-4.7-1b.json) `descriptor`):

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 1] int32; an image token is id V + k (V = 248,094, k = 0..63, row-major over the 8×8 merged grid) |
| | `position_ids` | [1, seq] int32, the ramp 0 .. seq − 1 |
| static inputs | `image_embeds` | [64, 1024] fp16: the tower's rows; zero for a text-only prompt |
| | `rope_shift_start` | [1] int32 |
| | `rope_shift_amount` | [1] int32 |
| states | `keyCache`, `valueCache` | [6, 1, 2, −1, 256] fp16 (the 6 full-attention layers; context up to 4,096) |
| | `convState`, `recState` | [18, 1, 6144, 3], [18, 1, 16, 128, 128] fp16 |
| output | `logits` | [1, 1, 248094] fp16 |

In the graph an id ≥ V reads row id − V of `image_embeds`; any other id reads the embedding table.

**The canvas rule.** For one 448×448 image the frame is `<image>`, the 64 image tokens, `</image>`, and the template's
`"\n"`. With O = the index of `<image>` (the canvas origin) and the host's `start` / `amount`, the graph gives each
position p its planes:

| token | position | (T, H, W) |
|---|---|---|
| text before the frame (`<image_id>0</image_id>` included) | p < O | (p, p, p) |
| `<image>` (248078) | p = O | (O, O − 1, O − 1) |
| image row k | p = O + 1 + k | (O, O + k // 8, O + k % 8) |
| `</image>` (248079) | p = O + 65 | (O, O + 8, O + 8) |
| the `"\n"` after `</image>` | O + 65 < p < start | (O, O, O) |
| everything after the frame, generated tokens included | p ≥ start | (p − amount) on all three |

`start` = the index after `</image>` and the marker tokens that follow it; `amount` = start − (O + 9), where 9 =
max(H, W) + 1 for the 8×8 grid. On the four image prompts O = 6, 6, 6, 17, start = 73, 73, 73, 84 and amount = 58.
This is transformers' `get_rope_index` for `mrope_mode: canvas`: in fp32 torch the rule equals the planes the
original model's rotary received at 433/433 prompt positions and 458/458 generate positions (lane
`oracle/parity_vl_torch.json`). A text-only prompt sends start = 2³⁰ and amount = 0, every position gets (p, p, p),
and in fp32 torch the decoder then equals the plain Qwen3.5 text decoder bit for bit (same record). The host side is
`host_static_inputs` in `conversion/minicpmv47/minicpmv47_vl_pipelined.py`.

## Prompt

The processor's expansion, which the decoder's ids follow (P1 of
[`fixtures-minicpm-v-4.7-1b.json`](fixtures-minicpm-v-4.7-1b.json)):

```
<|im_start|>user\n<image_id>0</image_id><image><|image_pad|>×64</image>\nWhat is in this image?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

- Thinking off: `enable_thinking=False` ends the prompt with `<think>\n\n</think>\n\n`. Every gate here ran that way.
- Stop on 248044 or 248046 (`generation_config.json`). The answers here that ended before the 32-token budget ended
  on `<|im_end|>` (248046).
- The 64 `<|image_pad|>` (248056) become V + k. The bundle tokenizer's encoding of the expanded prompt equals the
  processor's ids on all four image prompts ([`gate-minicpm-v-4.7-1b-e2e.json`](gate-minicpm-v-4.7-1b-e2e.json)
  `tokenizer_ids_equal_processor_ids`).
- **Thumbnail only.** The tower is baked for one 448×448 image (32×32 patches, 64 tokens at the default
  `downsample_mode` 16x). Every gate here fed a 448×448 image, which the original processor neither slices (at any
  `max_slice_nums`) nor resizes again. For other images the original keeps the aspect ratio (`find_best_resize`,
  sides rounded to multiples of 56) and, when the image is larger than 448×448 and `max_slice_nums` > 1, adds slices
  (the source card recommends 36 for images); this port squashes the image to 448×448 and keeps the thumbnail only.
  What that costs in answers is not measured here.
- Host preprocessing: RGB → Pillow BICUBIC resize to 448×448 (aspect ratio not kept) → x / 255 → (x − 0.5) / 0.5 →
  CHW `[1, 3, 448, 448]`, as `preprocess` in `conversion/minicpmv47/host.py`. Against the processor's own pixels:
  max |Δ| 5.9e-8 on the four image prompts (lane `oracle/host_preprocess.json`).

## Measured

Two instruments that do not compare with each other: the Mac numbers time one Python-runtime call at a time, the
iPhone numbers are the Swift pipelined engine's.

### Mac M4 Max (macOS 27.0 26A428, coreai-core 1.0.0b2 Python runtime, AOT h16c, 2026-10-13)

`coreai-build compile … --platform macOS --preferred-compute gpu --architecture h16c` (`--expect-frequent-reshapes`
for the decoder), loaded with `SpecializationOptions.default()`. ms = wall time around one call including the
output read-back, inside a quiet window of the machine-wide GPU lock. The decoder was fed P1's 88-token prompt, then
timed on greedy `main` calls.

| asset | ms per call, all forms in one process, rotated (run 3) | runs 1 / 2 | cold / warm in the tower gate | record |
|---|---:|---|---|---|
| decoder int8hu, S=1 `main` | **7.44** (134.4 tok/s) | 7.89 / 9.04, one process per decoder | — | lane `r3/timing_run3.json`, `r3/timing.json`, `r3/timing_run2.json` |
| decoder fp16 (reference, not uploaded) | 7.42 (134.9 tok/s) | 7.96 / 7.37 | — | same |
| tower int8lin | **43.81** | 44.22 / 43.65 | 921.3 / 44.01 | lane `r3/timing_run3.json`; [`gate-minicpm-v-4.7-1b-tower-int8lin.json`](gate-minicpm-v-4.7-1b-tower-int8lin.json) `timing` |
| tower fp16 | 43.85 | 44.20 / 43.66 | 86.6 (right after int8lin) / 44.40 | lane `r3/timing_run3.json`; [`gate-minicpm-v-4.7-1b-tower-fp16.json`](gate-minicpm-v-4.7-1b-tower-fp16.json) `timing` |
| tower fp16w32 (fp16 storage, fp32 math; not uploaded) | 55.03 | 55.21 / 54.64 | 686.7 / 55.54 | lane `r3/timing_run3.json`, `r3/gate_vision_fp16w32.json` |

Separate processes drifted by up to 20 % (in both runs the second process of the window was the slower one); the
run-3 column is the number to read. int8hu and fp16 take the same time per call on this runtime (7.44 vs 7.42 ms);
how that time splits was not measured.

### iPhone 18 Pro (iPhone19,2, iOS 27.2 24B5099f, PipelinedBench, device JIT, 2026-10-13)

PipelinedBench (Swift, Release, Xcode 27.0 RC, the pipelined engine with both patches, the
`increased-memory-limit` entitlement) loaded the bundles' `.aimodel` files and the phone specialized them; USB
power, screen state not controlled. The decoder runs in PipelinedBench's GLM-OCR form, which binds `image_embeds`,
`rope_shift_start` and `rope_shift_amount` as static buffers. Speed trial: a random prompt (seed 0) of p = 128
tokens prefilled one token per call (`COREAI_CHUNK_THRESHOLD=1`), then greedy g = 256, with the image rows zero and
start = 2³⁰ (text only); decode tok/s = (tokens − 1) / time after the opening streamed token, prefill tok/s = p /
time to it. Every row is in [`gate-minicpm-v-4.7-1b-iphone.json`](gate-minicpm-v-4.7-1b-iphone.json) under `runs`.

| launch | engine ready | prefill tok/s | decode tok/s (trials) | nat P0 / oracle P1 | run |
|---|---:|---:|---:|---|---|
| cold (fresh container) | 6.4 s | 105.7 | 96.0 (94.5, 97.5) | 23/23, 32/32 | `run1_cold` |
| warm | 0.2 s | 105.7 | **98.0** (98.0, 98.0) | 23/23, 32/32 | `run2_warm` |
| warm, g = 1024 asked | 0.2 s | 105.5 | 96.9 (one trial) | 23/23, 32/32 | `run3_long` |
| warm, P1's rows from the fp32 HF tower | 0.2 s | 105.3 | 97.1 (97.1, 97.1) | 23/23, 32/32 | `run4_hfrows` |

The g = 1024 trial ran without an error; the engine's iOS dynamic-KV guard caps prompt + generated at 1,024
positions, so it decoded 896 tokens by the code path (the app does not print the count).

Towers (`PB_VISION`, fp16 in and out, P1's 448×448 image, median of 9 calls after one cold call; cosine over all
65,536 output values against the Mac's output of the same form on the same fp16 patches):

| tower | `.aimodel` download | encode ms, two launches (order reversed) | cos vs the Mac | max \|d\| | runs |
|---|---:|---|---:|---:|---|
| int8lin | 614 MB | 67.8, 68.5 | 0.999997 | 0.007 | `vis_int8lin`, `vis_int8lin_b` |
| fp16 | 1.09 GB | 66.5, 68.6 | 0.999997 | 0.013 | `vis_fp16`, `vis_fp16_b` |

The container's opening tower call took 2,214 ms; later launches' cold calls took 82–91 ms with either form.

## Gates

The oracle is transformers 5.19.0 in fp32 on the CPU (one thread, eager) at the pinned revision. A knife edge is an
oracle step whose softmax top-2 gap is below 0.1 (`conversion/coreai_gate.py`'s rule), and a free-run passes when it
never leaves the oracle's greedy ids or leaves them at a knife edge. The bars were fixed before the runs.

| gate | what | bar | result | record |
|---|---|---|---|---|
| text, fp32 torch | the overlay's Qwen3.5 decoder vs the oracle, P0's last 48 positions | ordered top-5 equal | 48/48, cos min 0.999999999832, max \|Δ\| 2.39e-4 | lane `oracle/parity_text.json` |
| planes, fp32 torch | the canvas rule vs the planes the oracle's rotary received, P0–P4 + generate | equal at every position | 433/433 + 458/458; red arms (amount + 1, 1-D planes) fail 4/4 | lane `oracle/parity_vl_torch.json` |
| tower, fp32 torch | the re-authored tower vs the oracle's rows, P1–P4 | per-token cos min ≥ 0.9999, max \|Δ\| ≤ 1e-3 | min 0.9999999992, max \|Δ\| 3.8e-5; the baked export path = the reference path bit for bit | lane `oracle/parity_vision.json` |
| G1, Mac AOT | int8hu decoder, P0 teacher-forced at positions 19..66 + free-run | argmax equal at every non-knife-edge position; free-run rule | 44/44 (48/48 with the 4 knife edges), logits cos min 0.99458; free-run = the oracle's 23 tokens to `<\|im_end\|>` | [`gate-minicpm-v-4.7-1b.json`](gate-minicpm-v-4.7-1b.json) `G1` |
| G2, Mac AOT | int8hu decoder, P1–P4 free-run on the oracle's tower rows | free-run rule, 4/4 | 4/4, no divergence over 32 / 30 / 2 / 32 tokens; the 1-D-planes red arm red (max \|d\| at the oracle's top-5, from the frame on: 11.98 vs 0.372) | same, `G2`, `red_arms` |
| G3, Mac AOT | towers vs the oracle's rows, P1–P4, per-token cos | int8lin: mean ≥ 0.9995, min ≥ 0.99; the fp16w32 bar: mean ≥ 0.9999, min ≥ 0.999 | int8lin mean ≥ 0.99972, min 0.99411 (P2); fp16 mean ≥ 0.999997, min 0.999957 (meets the fp16w32 bar) | [`…-tower-int8lin.json`](gate-minicpm-v-4.7-1b-tower-int8lin.json), [`…-tower-fp16.json`](gate-minicpm-v-4.7-1b-tower-fp16.json) |
| G4, Mac AOT | image file → host → tower → int8hu decoder → answer, P1–P4 | free-run rule, 4/4 | 4/4 with the int8lin tower and with the fp16 tower; the answers equal the oracle's token for token | [`gate-minicpm-v-4.7-1b-e2e.json`](gate-minicpm-v-4.7-1b-e2e.json), [`…-e2e-fp16.json`](gate-minicpm-v-4.7-1b-e2e-fp16.json) |
| iPhone 18 Pro | int8hu decoder, nat (P0) + oracle (P1) | free-run rule | 23/23 + 32/32 on all 4 launches, with the Mac tower's rows and with the HF rows | [`gate-minicpm-v-4.7-1b-iphone.json`](gate-minicpm-v-4.7-1b-iphone.json) |
| iPhone 18 Pro | towers vs the Mac's output of the same form | cos ≥ 0.999 | 0.999997, both forms, both launches | same |
| staged bytes | the stripped int8lin tower + int8hu decoder, G4 re-run on the Mac | greedy ids = the export's | equal on P1–P4 (and = the oracle's), reset bit-equal | lane `r6/regate_stripped.json` |

P0's step 8 is a knife edge (gap 0.0553); the Mac and the phone took the oracle's token there. The four answers
(G4, either tower), ≤ 32 tokens each: P1 *"This image shows **Earth** — specifically, a vivid, close-up view of the
planet from space. …"*; P2 *"An aerial black-and-white photo captures a Saturn V rocket standing on its launch pad,
surrounded by barren terrain and runway lines under a cloudy sky."*; P3 *"EXIT"*; P4 *"Looking at this Earth image, I
see a rich variety of colors: …"*. The images are two NASA photographs (public domain) and one drawn with Pillow
primitives; the fixture file pins them by sha256 and does not carry them.

## What the Mac and the phone do not separate

The Mac's h16c AOT compile and the phone's own specialization both store the int8lin tower as fp16: its compiled
`resources.bin` is 1,087,609,504 B, the fp16 tower's size (Mac: lane `r3/aot_minicpmv47_vision_{int8lin,fp16}.json`,
contents different, sha256 `261c57c2…` vs `152bc955…`; phone: lane `r4/logs/container_caches.json`). So the two
towers run at the same speed wherever measured (Mac 43.81 vs 43.85 ms, iPhone 67.8 / 68.5 vs 66.5 / 68.6 ms), and
int8lin buys download size only: 614 MB against 1.09 GB. Its cost is tower accuracy against the oracle (per-token cos
min 0.99411 vs 0.999957, G3), which did not reach the answers (G4 identical with either tower). The decoder keeps its
int8 bytes on the phone (its cache `resources.bin` is 1,308,616,980 B, the `.aimodel` 1,308,566,922 B); on the Mac its
AOT with `--expect-frequent-reshapes` is 2,813,727,608 B (lane `r3/FILES.md`).

## Convert / verify

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1, torch 2.9.0,
transformers 4.57.6) for export, AOT and the gates; the oracle in its own venv (Python 3.12, transformers 5.19.0,
torch 2.14.1, torchvision 0.29.1, Pillow 12.3.0); the snapshot at `984a89a` in the Hugging Face cache with
`HF_HUB_OFFLINE=1`; `DEVELOPER_DIR` = Xcode 27.0 RC for `coreai-build`. Every script, in order, with its record:
[`conversion/minicpmv47/README.md`](../../conversion/minicpmv47/README.md).

```bash
PY=../coreai-models/.venv/bin/python     # the overlay venv; run from the repository root
<oracle-venv>/bin/python -I conversion/minicpmv47/make_fixtures.py       # the three images, 448x448
<oracle-venv>/bin/python -I conversion/minicpmv47/oracle_minicpmv47.py   # transformers 5.19.0, fp32, CPU
<oracle-venv>/bin/python -I conversion/minicpmv47/oracle_margins.py      # the knife edges
$PY -I -B conversion/minicpmv47/export_decoder.py int8hu
$PY -I -B conversion/minicpmv47/export_vision.py int8lin                 # and: fp16
$PY -I -B conversion/minicpmv47/aot_compile.py $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --reshapes --record <dir>/aot_decoder.json
$PY -I -B conversion/minicpmv47/aot_compile.py $ZOO_EXPORTS/minicpmv47_vision_int8lin --record <dir>/aot_tower.json
$PY -I -B conversion/minicpmv47/gate_decoder.py $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --text --vl --red --out <dir>/g1g2.json
$PY -I -B conversion/minicpmv47/gate_vision.py $ZOO_EXPORTS/minicpmv47_vision_int8lin --out <dir>/g3.json
$PY -I -B conversion/minicpmv47/gate_e2e.py $ZOO_EXPORTS/minicpmv47_vision_int8lin $ZOO_EXPORTS/minicpmv47_vlm_decode_int8hu --out <dir>/g4.json
```

The gates load only the AOT `.aimodelc` with `SpecializationOptions.default()`, one bundle per process (the Python
runtime leaks an IOSurface per call; each gate stays under 15 sequences). On the M4 Max an export takes 4–41 s, an AOT
compile 3–37 s, a decoder gate about 20 s and a tower gate about 6 s (lane `r3/export_*.json`, `r3/aot_*.json`).

## Try it

CoreAIKit enrollment: pending. Until then the decoder runs in PipelinedBench's GLM-OCR form (above): the host binds
`image_embeds`, `rope_shift_start` and `rope_shift_amount` as static buffers, feeds the ids with the 64 image tokens
written as V + k, and stops on 248044 or 248046.

## Lessons

- **A last-position logits bar does not see the plane scheme.** With 4.6's 1-D positions on 4.7's weights the
  last-position top-5 still matched on all four image prompts (cos 0.9981–0.9996 against the oracle's
  0.99999999999; lane `oracle/parity_vl_torch.json`), so the planes are gated for equality in torch, and on the
  engine the plane check is the 1-D-planes red arm (11.98 vs 0.372). The amount + 1 arm went red on the fp16 decoder
  (0.640 vs 0.140, lane `r3/gate_decoder_fp16.json`) but not on int8hu (0.446 vs 0.372,
  [`gate-minicpm-v-4.7-1b.json`](gate-minicpm-v-4.7-1b.json) `red_arms`): shifting the frame and the text after it by
  one changes only their offset from the six tokens before the frame, which int8's own spread covers.
- **After specialization the int8lin tower is stored as fp16.** The Mac AOT and the phone both hold its weights in
  fp16, so int8 changes the download, not the encode time ([above](#what-the-mac-and-the-phone-do-not-separate)).
- **`devicectl device copy to` skips a destination file of the same size whose mtime is newer than the source.** A
  bytes check passes and the old contents stay; `touch` the source before re-pushing and compare a pulled md5.

Port notes: [`knowledge/minicpm-v-4.7-1b-port.md`](../../knowledge/minicpm-v-4.7-1b-port.md).

## License

Source Apache-2.0 (`license: apache-2.0` in the source card; the snapshot carries no LICENSE file). The bundles
inherit it, and the Hugging Face repository carries the Apache-2.0 text. The fixture photographs are NASA works in
the public domain; the synthetic image is drawn by `make_fixtures.py`.
