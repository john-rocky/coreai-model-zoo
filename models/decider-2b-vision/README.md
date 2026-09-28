# decider-2b-vision — Core AI

[🤗 mlboydaisuke/decider-2b-vision-CoreAI](https://huggingface.co/mlboydaisuke/decider-2b-vision-CoreAI) · Apache-2.0 · source [Mapika/decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision/tree/863e290863655f1d6b69324d77d09ac972d21609) (revision `863e290`) · base Qwen/Qwen3.5-2B-Base

A **decision model that reads an image**. Give it an image (a photo, a diagram, a game frame), a
short context and one or more questions with up to 10 lettered options each. It returns a
probability for every option, read from the letter logits at each question's answer slot in one
forward pass. It never generates text. Text-only questions go through the same weights.

Mapika transplanted the v5 text weights of decider-2b into the Qwen3.5-2B vision-language model
and fine-tuned it for one epoch on game frames labelled by scripted policies, multiple-choice image
tasks from The Cauldron and a replay of the text mixture, then ran PPO from pixels (the author's
card and `decider/vision/` in [Mapika/decider](https://github.com/Mapika/decider)). The text
mixture's teacher-written items come from a locally run Qwen3.5-27B (`decider/data/teacher_*.py`).
The author's numbers, quoted from the source card and not re-measured here: Visual7W (held out)
accuracy 0.89 and ECE 0.03 on 300 items; 0.80–0.95 on six of the mixture's The Cauldron tasks.

This port runs that readout as two Core AI graphs. The **vision tower** is baked at a fixed grid:
`g256` (a 256×256 tile, 64 image tokens) for game frames and speed, `g448` (448×448, 196 tokens)
for photos. It stores fp16 weights and computes in fp32 (660 / 663 MB). The **decoder** is the
Qwen3.5 hybrid (18 Gated DeltaNet + 6 full-attention layers) with token ids in and the tower's rows
as a static input; it derives the three-plane M-RoPE positions inside the graph. Its linears are
int8 per block of 32 except in layers 0, 2 and 5, and the tied embedding / head stays fp16. It is
one bundle with an S=1 `main` and an S=16 `prefill` function, 2.64 GB. The gate is probability
parity with the author's own fp32 code, on a self-made fixture and on 500 held-out photo runs.

## Readout contract

One row per image; every question of the row is answered in the same pass:

```
<|vision_start|><N image tokens><|vision_end|>Context:\n<context>\n\nQuestion 1: <question>\nOptions:\n(A) <option>\n(B) <option>\nAnswer 1: (\n\nQuestion 2: <question>\n…\nAnswer 2: (
```

- The image block sits directly in front of `Context:`: `<|vision_start|>` (248053), N image
  tokens, `<|vision_end|>` (248054). No BOS, no chat template. N = 64 at `g256`, 196 at `g448`.
- The author's processor writes the image tokens as N copies of `<|image_pad|>` (248056). This
  decoder takes them as ids V + k instead (V = 248,320, k row-major over the merged grid), so each
  token names its tower row.
- The text is the author's `build()` at the pinned revision, without shuffling: the context cut
  at 1,536 tokens, then one block per question; the number after `Question` / `Answer` appears
  only when there are several questions. The text is decoded and re-encoded before the ids are
  built, as the author's `prepare()` does.
- A question's slot is the token `" ("` (318) preceded by `:` (25) with `Answer` (15666) among the
  5 tokens before it.
- The letters are bare `A`..`J`, ids 32..41, with no leading space. `p = softmax(logits[32 : 32 + n])`
  over the question's n options, at T = 1.
- Image: RGB → Pillow BICUBIC resize to the grid's square → /255 → (x − 0.5) / 0.5 →
  merge-block-major patches `[4G², 1536]`, the frame repeated at both temporal slots. The aspect
  ratio is not kept: a 256×240 frame becomes 256×256. The author's processor chooses a grid per
  image; [what a fixed grid costs](#what-a-fixed-grid-costs-the-authors-code-only) is measured below.
- Text-only row: no image block, the same bundle, zero image rows and `rope_shift_start` = 2³⁰.

`conversion/decider_vision/oracle_decider_vision.py` runs the checkpoint's own `decider/vision.py`
(`VisionDecisionModel.prepare()` → `slot_logits()`, fp32, CPU) and records the reference:
[`fixtures-decider-2b-vision.json`](fixtures-decider-2b-vision.json). Its 41 requests use images
drawn from Pillow primitives by `make_fixture_images.py` (CC0-1.0): the author's own benchmark
case (a solid-colour 224×224 image), The Cauldron form (shapes, counts, arrows, bar charts,
seven-segment digits), the author's game-frame prompts with the game titles replaced by plain
descriptions, and 6 text-only requests. Each image request runs at `g256`, `g448` and `native`
(the processor's own grid): 111 runs, 157 slots. The smallest oracle top-2 margin is 0.056, so
the argmax gate needs no near-tie exemption.

## Core AI shape

**Tower.** The overlay's `qwen3_5_vision.Qwen3_5VisionEncoder`, the tower code of the zoo's
Qwen3.8-27B vision path, baked at one merged grid by `conversion/export_qwen38vl_pipelined.py
--skip-decoder`: patches `[4G², 1536]` f32 → `image_embeds [G², 2048]` f32, every positional
term a constant. This port added `--vision-dtype fp16w32`: parameters stored fp16 and read
through a cast, the math in fp32. The plain fp16 tower misses the zoo's tower bar on the Mac GPU
(worst row cosine 0.849 at `g256`, 0.528 at `g448`); fp16w32 keeps every row at 0.9999957 or above.

**Decoder.** `conversion/decider_vision/qwen3_5_vl_pipelined.py`, new in this port: the Qwen3.5
hybrid text decoder with the embedding table inside the graph. An id ≥ V reads row id − V of
`image_embeds` instead of the table.

| | name | shape, type |
|---|---|---|
| inputs | `input_ids` | [1, 1] (`main`) or [1, 16] (`prefill`), int32 |
| | `position_ids` | [1, seq], int32, the ramp 0 .. seq − 1 |
| static inputs | `image_embeds` | [256, 2048] fp16: tower rows 0..N−1, the rest zero |
| | `image_rc` | [256, 2] int32: (k // W, k % W) for token k |
| | `rope_shift_start` | [1] int32: 1 + N, the `<\|vision_end\|>` index |
| | `rope_shift_amount` | [1] int32: N − max(H, W) |
| states | `keyCache`, `valueCache` | [6, 1, 2, ctx, 256] fp16, ctx up to 4,096 |
| | `convState`, `recState` | [18, 1, 6144, 3], [18, 1, 16, 128, 128] fp16 |
| output | `logits` | [1, 1, 248320] fp16 (the prefill function: the chunk's last position) |

An image token k at position p gets (t, h, w) = (p − k, p − k + row, p − k + col) from `image_rc`;
text after the image gets p − `rope_shift_amount` on all three planes. That is Hugging Face's
`get_rope_index`: the image spends max(H, W) positions. The row/column table lets one decoder
take `g256`, `g448` and the processor's own rectangles (the fixture's `native` runs), up to 256
image tokens.

The decoder derives the three planes because that is the author's contract: the checkpoint runs
full M-RoPE. Its planes equal the ones the author's model used on 111/111 runs. Positions are gated
by that equality, not by probability: on this fixture, 1-D positions move p by at most 0.0012
(r14 / r16 at `g256`, [`gate-decider-2b-vision-torch-parity.json`](gate-decider-2b-vision-torch-parity.json) P4).

**Two functions.** `main` (S=1) and `prefill` (S=16, Gated DeltaNet unrolled in the graph) share
the weights. Per row, from zeroed states: `prefill` while a whole chunk fits before the next slot,
then `main` one token at a time up to and including the slot. The bundle's `metadata.json` carries
this order, the letter ids, the slot rule and the image contract.

**No engine.** The pipelined engine exposes no logits (the decider-0.8b finding), so both hosts
drive the low-level runtime: Swift `AIModel(contentsOf:)` + `loadFunction` with four zeroed states
per row, Python `coreai.runtime`.

## Measured (Apple M4 Max, macOS 27.0 26A428, 2026-09-28 / 29)

The Python gates load an AOT `.aimodelc` (`coreai-build compile … --platform macOS
--preferred-compute gpu --architecture h16c`, `--expect-frequent-reshapes` for the decoder) with
`SpecializationOptions.default()`. Oracle = the author's fp32 readout above.

### Decoder alone: 111 runs with the oracle's ids and fp32 image rows

| bundle | functions, order | letter argmax | max \|Δp\| | mean of run means | bar |
|---|---|---:|---:|---:|---|
| fp16 | `main`, S=1 | 157/157 | 0.0101 | 0.00016 | PASS |
| int8hu (int8 body + int8 head) | `main`, S=1 | 157/157 | 0.0206 | 0.00040 | FAIL |
| int8lin (int8 body, fp16 head) | `main`, S=1 | 157/157 | 0.0217 | 0.00040 | FAIL |
| int8lin_pf16 | `main` + `prefill`, chunk | 157/157 | 0.0224 | 0.00041 | FAIL |
| fp16_pf16 (reference) | `main` + `prefill`, chunk | 157/157 | 0.0101 | 0.00017 | PASS |
| **int8mix_pf16 (ship)** | `main` + `prefill`, chunk | **157/157** | **0.0094** | **0.00026** | **PASS** |

The bar was fixed before any result: every slot found, letter argmax equal to the oracle's on
every slot, the full-vocabulary top-1 equal to the oracle's letter (157/157 on all six), max |Δp|
≤ 0.02, mean of run means ≤ 0.002, and every process re-running its first run bit for bit.
Red arm: r14 at `g256` with `image_embeds` zeroed moves the argmax from A to D (|Δp| 0.82) on the
five bundles that ran it. Transcripts: `gate-decider-2b-vision-readout-<bundle>.json`.

Before export, the new decoder module in fp32 torch equals the oracle on 157/157 slots (max |Δp|
3.3e-6); on text rows it equals the overlay's plain Qwen3.5 text decoder bit for bit, 467/467 steps
([`gate-decider-2b-vision-torch-parity.json`](gate-decider-2b-vision-torch-parity.json)).

### End to end: image file → decision, 70 image runs

The fixture image → `host.preprocess` (Pillow) → tower `.aimodelc` → decoder, with the ids built
by `host.build_ids` from the bundle's tokenizer (equal to the oracle's on 70/70 runs):

| decoder + tower | letter argmax | max \|Δp\| | mean of run means | bar |
|---|---:|---:|---:|---|
| fp16 + fp16w32 | 98/98 | 0.0111 | 0.00019 | PASS |
| int8hu + fp16w32 | 98/98 | 0.0214 | 0.00038 | FAIL |
| int8hu + fp16 tower | 98/98 | 0.0285 | 0.00045 | values only |
| int8lin_pf16 + fp16w32 | 98/98 | 0.0209 | 0.00036 | FAIL |
| fp16_pf16 + fp16w32 | 98/98 | 0.0085 | 0.00017 | PASS |
| **int8mix_pf16 + fp16w32 (ship)** | **98/98** | **0.0096** | **0.00029** | **PASS** |

[`gate-decider-2b-vision-e2e.json`](gate-decider-2b-vision-e2e.json). Tower alone on the same
images, fp16w32 vs the author's fp32 tower: worst image cosine 0.99999992 (`g256`) / 0.99999997
(`g448`), worst row 0.9999975 / 0.9999957; the host's patches equal the processor's
`pixel_values` exactly on every fixture image ([`gate-decider-2b-vision-tower.json`](gate-decider-2b-vision-tower.json)).

### Held out: 500 photo runs that no choice was made on

The 250 photos of the grid-price table below (the first rows of visual7w 150 + vsr 100 in The
Cauldron, the two subsets the author held out of training), thumbnailed to 768 as the author's
data code does, each at both grids. The bars were registered before any result; the reference is
the author's code (fp32 on the Mac's MPS backend).

| bundle | ids = processor | argmax, author margin ≥ 0.02 | near-ties agreeing | max \|Δp\| | mean of run means | bar |
|---|---:|---:|---:|---:|---:|---|
| **int8mix_pf16 (ship)** | 500/500 | 497/497 | 3/3 | 0.0146 | 0.00144 | PASS |
| fp16_pf16 | 500/500 | 497/497 | 3/3 | 0.0131 | 0.00061 | PASS |

The host's pixels equal the processor's on 500/500 runs. On vsr alone the int8mix mean is 0.00276
(`g256`) / 0.00274 (`g448`), above 0.002; the bar was registered over all 500 runs (fp16_pf16 on
vsr: 0.00107 / 0.00105). The two bundles agree on the argmax of 500/500 runs. Red arm: each
bundle's `g448` probabilities against the author's `g256` reference of the same photo — 233/247
decisive argmax, max |Δp| 0.69, all three probability bars red. The transcript
[`gate-decider-2b-vision-heldout.json`](gate-decider-2b-vision-heldout.json) holds dataset row
indices and numbers only; no image or question text is redistributed.

### What a fixed grid costs (the author's code only)

`conversion/decider_vision/grid_price.py` runs the checkpoint's own code three ways on the same
photos: the processor's dynamic grid (`native`), and the image resized to 256×256 or 448×448
first. Gold accuracy on 150 / 100 items, fp32 on the MPS backend:

| arm | visual7w accuracy | visual7w argmax = native | vsr accuracy | vsr argmax = native | image tokens |
|---|---:|---:|---:|---:|---:|
| native | 0.940 | — | 0.770 | — | 189.5 / 279.1 (mean) |
| `g256` | 0.913 | 0.947 | 0.740 | 0.910 | 64 |
| `g448` | 0.940 | 0.980 | 0.770 | 0.940 | 196 |

On the fixture's 256×240 game frames the processor itself picks the 64-token grid, and `native`
equals `g256` probability for probability (7/7 rows). Use `g256` for frames and `g448` for photos.

### Swift ([`apps/DeciderVision`](../../apps/DeciderVision/), Mac)

| check (fixture, 76 runs: `g256`, `g448`, text) | result |
|---|---|
| ids and slots built in Swift = the oracle's | 76/76 |
| tiles resized in Swift = `host.resize_bicubic` (Pillow's pass order) | 58/58 bit-equal |
| AOT int8mix_pf16 vs the oracle | argmax 108/108, max \|Δp\| 0.0096, mean 0.00028 |
| JIT: the bundle's `.aimodel` specialized by the Swift runtime | argmax 108/108, max \|Δp\| 0.0103, mean 0.00026 |
| AOT fp16_pf16 vs the oracle | argmax 108/108, max \|Δp\| 0.0085 |
| AOT int8mix_pf16, Swift vs Python on the same bundle | max \|Δp\| 2.9e-5; full logits bit-equal on 104/108 slots |

Every Swift/Python difference sits on the 2 runs whose tile Pillow and `resize_bicubic` resize
differently; everything else is bit-equal. Transcript:
[`gate-decider-2b-vision-swift.json`](gate-decider-2b-vision-swift.json).

Time per decision (Release CLI, AOT int8mix_pf16, the machine-wide GPU lock taken with the GPU at
0 %, two passes; PNG decode + resize + tower + prompt + decoder calls + read-out, 1–3 questions
per row):

| | `g256` (median 143.5 tokens) | `g448` (median 275 tokens) | text (median 76 tokens) |
|---|---:|---:|---:|
| decision, median | 0.322 / 0.321 s | 0.584 / 0.579 s | 0.284 / 0.286 s |
| tower | 22.0 ms | 61.3 / 61.2 ms | — |

One `main` call takes 11.6 ms and one `prefill` call (16 tokens) 25.2 / 25.1 ms. Load: 13.1 s in
the first timed process, 2.2–2.3 s in the others; peak footprint 1.01 GB. The JIT asset's first
specialization took 14.35 s, and 10.34 s from the cache.

### iPhone 18 Pro (iOS 27.0 24A437, device JIT, 2026-09-29)

The same `.aimodel` files, specialized on the phone by the Core AI runtime (no AOT asset), driven
by the same Swift library inside a headless gate app ([`apps/DeciderVisionGate`](../../apps/DeciderVisionGate/),
one target for iOS and macOS). The app carries the `com.apple.developer.kernel.increased-memory-limit`
entitlement: without it the decoder's cold specialization dies with `std::bad_alloc` at the default
limit (about 3.5 GB available), the way the zoo's pipelined-engine note records for 2B-class bundles.
One cold run (fresh container) and one warm re-run, the phone on USB power at 80 %:

| check (fixture, 76 runs: `g256`, `g448`, text) | result |
|---|---|
| ids and slots built on the phone = the oracle's | 76/76 |
| vs the oracle, cold run | argmax 108/108, full-vocabulary top-1 108/108, max \|Δp\| 0.0091, mean 0.00027 |
| warm re-run vs cold run | slot logits bit-equal 108/108 |
| vs the Mac Swift JIT run of the same files | argmax 108/108, max \|Δp\| 0.0015 (letter logits bit-equal on 28/108 slots) |

Load: tower alone 0.63 s cold / 0.49 s warm; decoder alone 21.6 s cold (14.7 s of it the model's
specialization, `main` 2.5 s, `prefill` 3.9 s) and 7.9 s from the runtime's cache; decoder plus one
tower 8.0 / 6.6 s (`g256`) and 7.8 / 7.0 s (`g448`). The runtime's cache for the three graphs takes
6.7 GB on the phone. Peak process footprint through the runs 543 MB (the weights are mapped). The
first decision after a cold load took 8.1 s, 0.98 s after a warm load.

Time per decision (1 warm-up + 5 timed, each burst under 5 s, USB power, battery 80 %):

| row | cold container, median (min–max) | warm re-run, median (min–max) | thermal |
|---|---:|---:|---|
| r23 `g256` (a 256×240 frame, 64 image tokens) | 785 ms (767–793) | 754 ms (744–769) | nominal |
| r23 `g448` (196 image tokens) | 1,439 ms (1,427–1,465) | 1,415 ms (1,389–1,415) | fair (cold) / nominal (warm) |
| t06 text (76 tokens) | 735 ms (732–746) | 725 ms (713–731) | nominal |

Transcript: [`gate-decider-2b-vision-iphone.json`](gate-decider-2b-vision-iphone.json), which also
keeps the stopped run without the entitlement.

## Precision

int8 per block of 32 over the whole body misses the 0.02 bar on one fixture question: r29 at `g256`,
a grid-walk frame (its `native` run has the same inputs and the same result), 0.0217 with the fp16
head (int8lin) and 0.0206 with an int8 head (int8hu). The head is not what fails it. An fp32 torch bisect with the exporter's own int8 weights
(`conversion/decider_vision/int8_bisect_torch.py`, 4 fixture runs) spread the error over layers
0–11. No single fp16 layer rescued it, and the smallest set that met its rule was {0, 2, 5}. That
set costs 165 MB over the all-int8 bundle (main.mlirb 2,644,272,749 vs 2,478,952,184 bytes) and
saves 1.12 GB against fp16 (3,765,759,218 bytes).

The set was chosen on the fixture, so the held-out photos above are its test: argmax 497/497 plus
3/3 near-ties, max |Δp| 0.0146. The quantizer's scheme is not the lever: coreai-opt 0.2.1's
`symmetric_with_clipping` did not clip (each block's absmax maps to ±127 on the layer checked), and
plain `symmetric` did not meet the bisect's rule either. Details: [`knowledge/decider-2b-vision-port.md`](../../knowledge/decider-2b-vision-port.md).

## ⬇️ Bundle

[mlboydaisuke/decider-2b-vision-CoreAI](https://huggingface.co/mlboydaisuke/decider-2b-vision-CoreAI),
four folders under `gpu-pipelined/`. A decision needs one decoder and one tower.

| folder | what | main.mlirb bytes | main.mlirb sha256 |
|---|---|---:|---|
| `decider_2b_vision_decode_int8mix_pf16/` | decoder, ship: `.aimodel` + `metadata.json` + `tokenizer/` | 2,644,272,749 | `a82f61ae…bce75` |
| `decider_2b_vision_decode_fp16_pf16/` | decoder, fp16 reference, same layout | 3,765,759,218 | `a0461812…e04d` |
| `decider_2b_vision_g256_vision_fp16w32/` | tower, 256×256 → 64 rows (`.aimodel`) | 660,263,631 | `4a0de698…cede1` |
| `decider_2b_vision_g448_vision_fp16w32/` | tower, 448×448 → 196 rows (`.aimodel`) | 662,696,647 | `9bf6023d…d9bdd` |

Uploaded 2026-09-29 as Hub revision `4948e3231035df85c7b03c90b766eb17a08d9451`; after the upload every file's size and sha256 matched the staged copy (the 6 LFS files by the Hub's own hash, the 18 small files by download), and `conversion/zoo_verify.py` reports the two decoder bundles PASS (the towers carry no tokenizer and are skipped).

No AOT asset ships: the Swift runtime specializes the `.aimodel` correctly on the Mac (the JIT row
above). A Mac AOT compile of the ship decoder is 10.2 GB, and the `prefill` function adds 3.8 GB of
it (int8lin compiles to 6.2 GB with `main` alone, 10.0 GB with both). The source `config.json`
and `decider_config.json` are in the repository verbatim. No runtime patch is involved.

## Use it

Swift, with the [`DeciderVision`](../../apps/DeciderVision/) package (targets macOS 27 and iOS 27,
measured on the Mac; the system CoreAI framework + swift-transformers' tokenizer):

```swift
import CoreAI
import DeciderVision

let root = URL(filePath: "decider-2b-vision-CoreAI/gpu-pipelined")      // a download of the HF repo
var decoderOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
decoderOptions.expectFrequentReshapes = true
let decider = try await VisionDecider(
    bundle: root.appending(path: "decider_2b_vision_decode_int8mix_pf16"),
    towers: [.g256: root.appending(path: "decider_2b_vision_g256_vision_fp16w32/decider_2b_vision_g256_vision_fp16w32.aimodel")],
    decoderOptions: decoderOptions, towerOptions: SpecializationOptions(preferredComputeUnitKind: .gpu))
let probs = try await decider.decide(image: cgImage /* a CGImage */, context: "This is a visual question about the image.",
    questions: [DecisionQuestion(text: "How many circles are in the image?", options: ["1", "2", "3", "4", "5"])],
    grid: .g256)                                                        // [[Double]], one row per question
```

The same from the CLI (`swift build -c release` in `apps/DeciderVision` builds `.build/release/decider-vision`):

```bash
R=decider-2b-vision-CoreAI/gpu-pipelined; decider-vision ask --asset jit --bundle $R/decider_2b_vision_decode_int8mix_pf16 --tower $R/decider_2b_vision_g256_vision_fp16w32/decider_2b_vision_g256_vision_fp16w32.aimodel --grid 256 --image x.png --context "This is a visual question about the image." --question "How many circles are in the image?" --options "1|2|3|4|5"
```

Python, with `conversion/decider_vision/host.py` and `coreai.runtime` (the gates' own code path,
`heldout_eval.py`): `host.preprocess(image, g)` → the tower → rows 0..N−1 of a zero
`image_embeds [256, 2048]` fp16; `host.build_ids(True, context, questions, tokenizer, (g, g))` →
ids, slots, `rope_shift_start`, `rope_shift_amount`; `image_rc[k] = (k // g, k % g)`; zeroed states
from the function's descriptors; the chunk order above; at each slot
`softmax(logits[0, -1, 32 : 32 + n])`. The Python gates ran the AOT `.aimodelc` (the `--aot` flag
below); this port did not use the Python runtime's JIT.

## Reproduce

Environment: the zoo overlay venv (coreai-core 1.0.0b2, coreai-torch 0.4.1, coreai-opt 0.2.1,
torch 2.9.0, transformers 4.57.6, overlay `397b337`), the source snapshot pinned at `863e290` in
`HF_HOME`, `HF_HUB_OFFLINE=1`, Xcode 27.0 RC. The author's code runs in its own venv
(transformers 5.17.0). The steps, in order, with every flag, are in
[`conversion/decider_vision/README.md`](../../conversion/decider_vision/README.md).

```bash
# towers: the Qwen3.8-27B vision exporter with the HF id and the grid swapped
python conversion/export_qwen38vl_pipelined.py --hf-id Mapika/decider-2b-vision --name decider_2b_vision_g256 \
    --grid-h 8 --grid-w 8 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/_decider2bv/exports
python conversion/export_qwen38vl_pipelined.py --hf-id Mapika/decider-2b-vision --name decider_2b_vision_g448 \
    --grid-h 14 --grid-w 14 --skip-decoder --vision-dtype fp16w32 --out-dir <work>/_decider2bv/exports
# decoders (--aot adds the h16c .aimodelc the Python gates load)
python conversion/decider_vision/export_decoder.py int8mix --fp16-layers 0,2,5 --prefill-chunk 16 --aot
python conversion/decider_vision/export_decoder.py fp16 --prefill-chunk 16 --aot
```

Port notes: [`knowledge/decider-2b-vision-port.md`](../../knowledge/decider-2b-vision-port.md).

## Other formats

GGUF conversions of the same checkpoint on the Hub: `mradermacher/decider-2b-vision-GGUF` and
`mindchain/decider-2b-vision-GGUF` (not run here).

## License

Source Apache-2.0 (Mapika/decider-2b-vision declares it in its card and carries no LICENSE file);
the bundles inherit it, and the Hugging Face repository carries the Apache-2.0 text. The author's
`decider/` code runs only in the oracle and reference scripts at gate time; it is not part of the
bundles. The fixture images are generated by this repository's script (CC0-1.0).
