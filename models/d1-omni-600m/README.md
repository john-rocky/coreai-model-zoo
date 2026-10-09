# d1-omni-600M — Core AI

[🤗 mlboydaisuke/d1-omni-600M-CoreAI](https://huggingface.co/mlboydaisuke/d1-omni-600M-CoreAI) · LFM Open License v1.0 ·
source [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M/tree/414f8d6438174f5b2133a9c21a478fc42625e308)
(revision `414f8d6`)

A **decision model** from Liquid AI. Give it a state (text or JSON, with images or one voice clip) and named
questions: `noul` (yes / no), `choice` (named options) or `score` (ordered levels). It returns a probability for every
option of every question, read from one forward pass. It does not generate text.

The checkpoint has 587 M parameters: a bidirectional LFM2.5-Encoder-350M trunk with a decision head, a SigLIP2 vision
tower and a 17-layer FastConformer audio tower. A request carries images or audio, not both. Liquid AI's release blog
calls it an early research release and reports speed for d1-3B only.

This port re-authors the three networks in plain PyTorch from `model.safetensors` and exports each one as a Core AI
graph in fp16. The host builds the token rows, prepares the images and the audio, runs the graphs, and applies the
publisher's temperatures and softmax. Every stage is gated against the publisher's own model in fp32 on the CPU.

## What this repository holds

Three graphs. The decision graph runs at seven static lengths; a host picks the smallest one that holds a row. The
vision graph takes one image crop per call. The audio graph runs at four clip lengths. Their MLIR debug locations were
removed before shipping (the operations are unchanged).

| folder | graph | `.aimodel` bytes | h19p `.aimodelc` bytes |
|---|---|---:|---:|
| `decide-fp16-L64` | decision, 64 positions | 761,631,060 | 761,747,067 |
| `decide-fp16-L128` | decision, 128 positions | 761,647,561 | 761,763,726 |
| `decide-fp16-L256` | decision, 256 positions | 761,680,330 | 761,796,331 |
| `decide-fp16-L512` | decision, 512 positions | 761,745,868 | 761,861,994 |
| `decide-fp16-L1024` | decision, 1,024 positions | 761,876,787 | 761,993,099 |
| `decide-fp16-L2048` | decision, 2,048 positions | 762,139,098 | 762,255,228 |
| `decide-fp16-L4096` | decision, 4,096 positions (attention in two blocks of 2,048 keys) | 762,676,555 | — (not shipped: on the iPhone 18 Pro its first call exceeds the per-process memory limit; the `.aimodel` runs) |
| `vision-fp16` | one image crop → up to 256 prefix rows | 188,218,374 | 188,303,983 |
| `audio-fp16-5s` | a clip of up to 5 s → up to 63 prefix rows | 224,920,212 | 225,226,318 |
| `audio-fp16-10s` | up to 10 s → up to 125 rows | 225,049,263 | 225,355,344 |
| `audio-fp16-20s` | up to 20 s → up to 250 rows | 225,305,277 | 225,611,159 |
| `audio-fp16-30s` | up to 30 s → up to 375 rows | 225,561,282 | 225,867,256 |

Each folder is in `macos/` and `ios/` (the same `.aimodel`; the device specializes it when it loads it) and, except
`decide-fp16-L4096`, in `ios-h19p/` (an ahead-of-time compile for the iPhone 18 Pro GPU, which other phones refuse). Each folder also holds
`metadata.json` (its contract) and its host files: `tokenizer/` for the decision graphs (the publisher's files,
unmodified), `position_table.f32` for the vision graph, `mel_filters_128x257_f32.bin` for the audio graphs. The
repository root holds the shared `metadata.json`, `tokenizer/`, `host/` (the host files and the Swift host's sources),
`reference/` (the public fixture and the publisher's numbers on it), `LICENSE` and `NOTICE.md`.

Source: `metadata.json` `staged`, `MANIFEST.json`.

## Use it

Swift: the `D1Omni` package is in `host/`, a copy of the zoo's
[`apps/D1Omni`](../../apps/D1Omni/) (macOS 27, the system CoreAI
framework, Accelerate, ImageIO and swift-transformers 1.3.3). An app whose package folder holds a download of this
repository depends on it by path. SwiftPM names a path package after its folder, so the package is `host`:

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Ask", platforms: [.macOS("27.0")],
    dependencies: [.package(path: "d1-omni-600M-CoreAI/host")],
    targets: [.executableTarget(name: "Ask", dependencies: [.product(name: "D1Omni", package: "host")])])
```

`Sources/Ask/main.swift`, run from the package folder:

```swift
import Foundation
import D1Omni

let dir = URL(filePath: "d1-omni-600M-CoreAI/macos")
let d1 = try await D1Omni(folders: D1Omni.folders(macos: dir), media: D1Omni.MediaFolders.find(macos: dir))
let questions: JSONValue = try JSONParser.parse(Data(#"""
    {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
     "team": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults"}}}
    """#.utf8))
let text = try await d1.systemOne(state: .string("I was charged twice this month, please refund one of them."),
                                  questions: questions)
let heard = try await d1.systemOne(state: nil, questions: questions,                 // 16 kHz mono 16-bit WAV
                                   audio: URL(filePath: "d1-omni-600M-CoreAI/reference/audio/aud_01.wav"))
let room: JSONValue = try JSONParser.parse(Data(#"{"bed": {"type": "noul", "instructions": "Is there a bed in the room?"}}"#.utf8))
let seen = try await d1.systemOne(state: nil, questions: room,                       // PNG or JPEG files
                                  images: [URL(filePath: "d1-omni-600M-CoreAI/reference/images/img_01.png")])
print(PythonFormat.dumps(text, asciiOnly: false))
```

The same from the Mac command line. `swift build -c release --package-path d1-omni-600M-CoreAI/host` builds `d1omni`
in `d1-omni-600M-CoreAI/host/.build/release/`:

```bash
d1omni ask --bundle-dir d1-omni-600M-CoreAI/macos --state "I was charged twice this month, please refund one of them." \
    --question '{"type": "noul", "instructions": "Is the customer asking for a refund?"}'
d1omni ask --bundle-dir d1-omni-600M-CoreAI/macos --image d1-omni-600M-CoreAI/reference/images/img_01.png \
    --question '{"type": "noul", "instructions": "Is there a bed in the room?"}'
```

Python, with the reference host [`conversion/d1_omni/host.py`](../../conversion/d1_omni/)
from a clone of the zoo (`coreai-model-zoo/`, beside the download) and the Core AI Python runtime (`coreai-core`), on
the Mac. The Python runtime loads compiled assets, so compile the decision graphs before you run it:

```bash
for L in 64 128 256 512 1024 2048 4096; do
  xcrun coreai-build compile d1-omni-600M-CoreAI/macos/decide-fp16-L$L/d1_omni_decide_fp16_L$L.aimodel \
      --output aot --platform macOS --preferred-compute gpu --architecture h16c
done
```

```python
import asyncio, json, sys
sys.path.insert(0, "coreai-model-zoo/conversion/d1_omni")
import host                                   # the publisher's prompt rules and readout, written out
import coreai.runtime as rt

async def ask(state, questions):
    tok = host.RawTokenizer("d1-omni-600M-CoreAI/tokenizer/tokenizer.json")
    host.check_token_ids(tok)
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    rows = host.request_rows(tok, state, questions)                     # one row per question
    probabilities = []
    for row in rows:
        L = host.bucket_for(row.positions, host.ALL_BUCKETS)            # the smallest of 64 ... 4096
        model = await rt.AIModel.load(f"aot/d1_omni_decide_fp16_L{L}.h16c.aimodelc", gpu)
        inputs, markers = host.graph_inputs(row, L)
        out = await model.load_function("main")({k: rt.NDArray(v) for k, v in inputs.items()})
        scores = out["scores"].numpy().reshape(-1)
        probabilities.append(host.probabilities_from_logits(scores[markers], row.question, row.calibrate))
    return host.response(rows, probabilities)

print(json.dumps(asyncio.run(ask(
    "I was charged twice this month, please refund one of them.",
    {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}))))
```

## Contract

**Rows.** One row per question, the publisher's `prompt.py` (`host.py` copies it):

```
[1 <|startoftext|>, 17 <|reserved_7|>] + state + [18 <|reserved_8|>] + instructions
  + for each option: [19 <|reserved_9|>, 16 <|mask|>] + " " + option + [20 <|reserved_10|>]
  + [21 <|reserved_11|>]
```

Every text piece is tokenized without special tokens, after `<|name|>` in caller text is rewritten to `<¦name¦>`. A
JSON state is written with `json.dumps(ensure_ascii=False)`. A state that does not fit is cut at its end. `<|pad|>` is
0 and `<|im_end|>` is 7; a host checks the nine ids against the tokenizer at load. Option text: `choice` →
`name: description`; `score` → `level i: description`; `noul` → `false: …`, `true: …` from the criteria, or
`false: no, the statement does not hold` / `true: yes, the statement holds` without them. Without criteria, an image
request writes `false: no` / `true: yes`. An audio request always does, and writes choices as
`option_000: description`, as those questions were trained.

**Decision graph** (`main`, static length L):

| tensor | shape | dtype | value |
|---|---|---|---|
| `input_ids` | [1, L] | int32 | 0 on the media prefix [0, P), the row's ids on [P, P + n), 0 after |
| `prefix_embeds` | [1, L, 1024] | float32 | the image or audio prefix on [0, P), 0.0 elsewhere |
| `pad_mask` | [1, L] | float32 | 1.0 on [0, P + n) |
| `prefix_mask` | [1, L] | float32 | 1.0 on [0, P) |
| `keep_right` | [1, L] | float32 | 0.0 at P − 1 when P > 0, 1.0 elsewhere |
| `qtype_onehot` | [1, 3] | float32 | choice / score / noul |
| → `scores` | [1, L] | float32 | read at P + each `<|mask|>` position |

L is the smallest of 64, 128, 256, 512, 1,024, 2,048, 4,096 that holds P + n. With an image the text is cut to 896
positions, with audio to 15,360; all positions together stay within 16,384.

**Readout.** The K marker scores of a text row are divided by the temperature of its type and option count, then
softmaxed in fp32. Image and audio rows take the plain softmax. A `noul` is scored as [false, true] and reported as
[yes, no].

| temperature key | T | | temperature key | T |
|---|---:|---|---|---:|
| `choice:2` | 1.7465145587921143 | | `noul:2` | 1.6663223505020142 |
| `choice:3-5` | 1.3998981714248657 | | `score:3-5` | 1.7301132678985596 |
| `choice:6-10` | 1.1751071214675903 | | `score:6-10` | 1.0 |
| `choice:11+` | 1.372515082359314 | | `choice` / `score` / `noul` (any other K) | 1.0 |

The answer is the publisher's: `noul` → P(yes); `choice` → the argmax name, a confidence and every p; `score` → the
expected level, a confidence, every p and the legend. `usage.input_tokens` counts every position the graphs read.

**Vision graph** (`main`, one crop): `pixel_values` [1, 1024, 768] (16-px patches row-major, (x − 127.5) / 127.5),
`pos_embed` [1, 1024, 768] (the 16 × 16 position table, `position_table.f32`, resized to the crop's patch grid,
bilinear with antialias), `patch_mask` [1, 1024], `unshuffle_index` int32 [256, 4] → `prefix` [1, 256, 1024]. An image
whose rounded area exceeds 2 × 512² is cut into the 512-px tiles of the closest grid (2 to 10 tiles) plus a
thumbnail; a smaller one is one crop. A 384 × 384 image is 144 prefix rows.

**Audio graph** (`main`, one clip bucket of 5 / 10 / 20 / 30 s): `mel` [1, 128, F] and four time masks → `prefix`
[1, T, 1024]. The clip is 16 kHz mono, cut at 30 s and padded to 0.5 s, not resampled. The mel: pre-emphasis 0.97; a
centred 512-point STFT, hop 160, Hann(400) centred in 512; power; 128 Slaney mel bins
(`mel_filters_128x257_f32.bin`); log(x + 2⁻²⁴); each mel row normalized over the clip's frames, (x − mean) /
(std + 1e-5). F = 501 / 1,001 / 2,001 / 3,001 columns; a clip takes the smallest bucket whose F holds its
1 + n // 160 frames.

Every value above is in `metadata.json`.

## Measured

**Time per decision.** Apple M4 Max, macOS 27.0 (26A428), the GPU, the Core AI Python runtime on the h16c compiles of
these graphs. One decision = the graph inputs, the graph call(s), the output copy, the marker read, the temperature and
the softmax. Tokenizing, image decoding and the mel are not included. Each form ran 30 decisions after 5 warm-up
rounds, interleaved with the other forms, inside a machine-wide measurement window (no other GPU job).

| request | positions | graphs | Mac M4 Max, median ms (p10–p90) | iPhone 18 Pro, median ms, `.aimodel` / h19p compile |
|---|---|---|---|---|
| one question | 47 | decision L64 | **7.35** (7.20–7.52) | **10.79** / 10.70 |
| three questions, one call each | 47 / 56 / 51 | decision L64 × 3 | **21.97** (21.67–23.36) | **32.52** / 32.07 |
| one question on a 3.4k-token state | 3,454 | decision L4096 | **299.2** (302.53 and 295.87 in two windows) | **758.2** / — (the compile exceeds the memory limit, see below) |
| one question on a 384 × 384 image | 144 + 39 | vision + decision L256 | **38.3** (38.50 and 38.12 in two windows) | **57.67** / 57.72 |
| one question on 9.6 s of audio | 121 + 63 | audio 10 s + decision L256 | **24.94** (24.76–25.06) | **27.02** / 27.30 |
| one question on 2.8 s of audio | 35 + 63 | audio 5 s + decision L128 | **17.72** (17.49–17.96) | not measured |
| one question on 28 s of audio | 350 + 63 | audio 30 s + decision L512 | **43.9** (43.83 and 43.92 in two windows) | not measured |

The single-window rows ran in window `d1d-r12` (2026-10-08 20:09:31–20:09:50 JST) on these stripped graphs. The
two-window rows ran in windows `d1d-r7-run1` and `d1d-r7-run2` (12:39–12:48 JST) on the same graphs before their debug
locations were removed: the operations and the compiled weights (`resources.bin`, by sha256) are the same, and these
three were not timed again. Source: `gate-d1-omni-600m-timing-mac.json` (zoo) ← `results/timing/ranking_r12.json`,
`ranking.json`.

**iPhone 18 Pro** (iPhone19,2, iOS 27.2.0 build 24B5099f, on USB power at 80 % charge, thermal state nominal on every
decision). The Swift host ran in a gate app on the phone: 20 decisions per form after 3 warm-up decisions, each series
under 20 s. The `.aimodel` column is the graph specialized on the phone at load (0.40–1.79 s cold); the h19p column is
the ahead-of-time compile (0.29–1.65 s cold). On a subset of the fixture (78 rows: 60 text, 9 image, 9 audio) every
row passed the gate above with the `.aimodel` graphs (max |Δp| 0.0059, mean 0.00090) and the h19p compiles gave the
same bits on the 74 rows they ran. The L4096 compile loaded but its first call was killed at a footprint of 3,306 MB,
the phone's per-process limit, so it is not shipped: on the phone the 4,096-position graph runs as the `.aimodel`.
The phone's bits differ from the Mac's (max |Δp| 0.0040 between them on the same rows; the host arrays are bit-equal).
Source: `gate-d1-omni-600m-iphone.json` ← `results/iphone_bench.json`, `iphone_parity_jit.json`,
`iphone_parity_aot.json`.

The Swift host (`D1Omni`, Release) gave the same times as the Python runtime with the `.aimodel` specialized at load
(JIT) and with the AOT compile: one question 16.14 / 16.16 ms and three questions 48.46 / 48.59 ms (both at L256, before
L64 / L128 existed), the 3.4k-token state 301.21 / 300.84 ms, the image 37.37 / 37.47 ms, 9.6 s of audio 24.48 /
24.78 ms. Its outputs were bit-equal between JIT and AOT on every timed call. Source: `gate-d1-omni-600m-timing-mac.json`
← `ranking_r8.json`, `ranking_r9.json`.

**Gate.** The oracle is the publisher's model (`AutoModel.from_pretrained(..., trust_remote_code=True)`,
`probabilities()`), fp32 on the CPU. The bar for each row: the argmax equal to the oracle's when the oracle's top-2
margin is above 0.02 (near ties are reported apart), max |Δp| ≤ 0.02 over the options, and the mean of the rows' max
|Δp| ≤ 0.002. Every gate also runs a control that judges each row against another row's oracle; it must fail, and it
does.

| graphs (Mac GPU, these stripped compiles) | rows | argmax | near ties | max \|Δp\| | mean of row max |
|---|---:|---:|---:|---:|---:|
| decision L64 | 64 | 64/64 | — | 0.0058 | 0.00095 |
| decision L128 (278 + 20 shorter rows padded in) | 298 | 290/290 | 8/8 | 0.0117 | 0.00109 |
| decision L256 | 436 | 424/424 | 12/12 | 0.0067 | 0.00098 |
| decision L512 (19 + 20 padded + 10 with a media prefix) | 49 | 46/46 | 3/3 | 0.0044 | 0.00095 |
| decision L1024 (20 padded + 10 with a media prefix) | 30 | 29/29 | 1/1 | 0.0071 | 0.00102 |
| decision L2048 (11 + 5 padded + 10 with a media prefix) | 26 | 25/25 | 1/1 | 0.0071 | 0.00099 |
| decision L4096 (4 + 5 padded + 11 with a media prefix) | 20 | 20/20 | — | 0.0038 | 0.00064 |
| image rows: vision → decision | 46 | 46/46 | — | 0.0062 | 0.00111 |
| audio rows: audio → decision (NumPy mel, the Swift host's) | 46 | 46/46 | — | 0.0036 | 0.00085 |
| audio rows: audio → decision (the publisher's torch mel) | 46 | 46/46 | — | 0.0055 | 0.00067 |

The decision gates ran each row three times (five at L4096) and the scores never changed; the image and audio gates'
repeated calls did not change them either. Source: `gate-d1-omni-600m-runtime-decide.json`,
`-runtime-vision.json`, `-runtime-audio.json` ← `results/ship_runtime_*.json`, `small_runtime_audio_e2e_fp16.json`.

- **Before export.** The decision graph's module in fp32 torch meets a tighter bar on all 470 rows: argmax 470/470,
  max |Δp| 1.45e-5 (bar 2e-5: the fp32 oracle is itself 9.3e-6 from the publisher's code run in float64), marker
  logits 8.6e-5. In float64 it equals the publisher's code to 1.2e-13. Source: `gate-d1-omni-600m-eager.json`.
- **The strip.** Every gate gave the same numbers bit for bit before and after the debug locations were removed.
  Source: `gate-d1-omni-600m-strip.json`, `-runtime-decide.json`.
- **Swift.** The Swift host's token rows equal the Python host's on every fixture row (479 / 551 / 623 rows of three
  fixture versions). Its marker logits equal the Python runtime's bit for bit on 562 rows of the 256–4,096 graphs (470
  text, 46 image, 46 audio) and on 342 text rows of the 64 and 128 graphs, with the AOT compile and the `.aimodel`
  alike. Source: `gate-d1-omni-600m-swift.json`.

**Measured and not shipped.** Decision graph at L256, one question; the same windows as above:

| form | gate (436 rows) | one question, ms |
|---|---|---:|
| fp16 (shipped) | pass, max \|Δp\| 0.0067 | 16.81 |
| fp32 | pass, 1.1e-5 | 17.63 |
| fp16 weights, fp32 compute | pass, 0.0020 | 23.67 |
| int8 weights (every decision linear, per block of 32) | fail: argmax 429/436, max \|Δp\| 0.097 | 16.70 |
| fp16 on the Neural Engine (52 regions) | fail: argmax 409/436, max \|Δp\| 0.95 | 41.77 |
| fp16 weights, fp32 compute, on the Neural Engine (1 region) | fail: a score moves by up to 13.8 between calls | 23.45 |

The int8 `.aimodel` is 468,960,581 bytes, but its compile's `resources.bin` is 761,554,008 bytes, the size of the fp16
compile's: only the download is smaller. The vision graph on the Neural Engine misses the bar (mean 0.0031); the audio
graph on it passes (24 rows) at 44.45 ms per 9.6 s clip. On the iPhone 18 Pro the Neural Engine compiles of the audio
and decision graphs failed to load; the `.aimodel` graphs with the Neural Engine preferred (placement not confirmed)
gave 59.98 ms for the 9.6 s audio request (9 rows passed) and a failing decision graph (argmax 33/55). Source:
`gate-d1-omni-600m-forms.json`, `gate-d1-omni-600m-iphone.json`.

## Other formats

On the Hub on 2026-10-08 (21:17 JST), not run here:

- GGUF from the publisher: [`LiquidAI/d1-omni-600M-GGUF`](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF), for
  llama.cpp, which the publisher's [blog](https://www.liquid.ai/blog/open-d1) names as supported from release day.
  Community GGUF files: `TechnoBaptist/d1-omni-600M-GGUF`, `AtomicChat/d1-omni-600M-GGUF`,
  `webmp3/Sakura-d1-omni-600M-HighQuality-GGUF`.
- ONNX for Transformers.js: `onnx-community/d1-omni-600M-ONNX`.
- MLX: `Nurymanau/d1-omni-600M-MLX-fp16`.
- Core ML: `FluidInference/d1-omni-600m-coreml`, `suryatmodulus/d1-omni-600m-coreml`.

No other Core AI conversion was listed.

## Reference fixture

`reference/records.json` holds the 237 public records of the gate's fixture and `reference/oracle.json` the
publisher's model's numbers on their 360 questions. The gates also ran on measured-only records that are not
published. Their sources:

| records | source | licence |
|---|---|---|
| `card_text`, `card_batch_00`, `card_batch_01` | the publisher's README examples | LFM Open License v1.0 |
| `semif_*` (144) | [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) `authored144` at `ca3ba65f` | MIT, Copyright (c) 2026 TheoLeeCJ (`reference/LICENSE-SemIf`) |
| `tv4_000` … `tv4_059` | MMLU test questions ([cais/mmlu](https://huggingface.co/datasets/cais/mmlu) at `c30699e8`), as lines 1–60 of jaredpalmer/kev's transfer-v4 development file | MIT, Copyright (c) 2020 Dan Hendrycks (`reference/LICENSE-MMLU-MIT.txt`) |
| `own_*`, `long_3400` | written for these ports | as the conversion code (BSD-3-Clause) |
| `img_01` | own FLUX.2 klein 4B output | Apache-2.0 model output |
| `img_02`, `img_03` | photographs from Wikimedia Commons, downscaled | CC0 1.0 |
| `aud_01` … `aud_15` | speech synthesized with Kokoro-82M from scripts written for these ports | Apache-2.0 model output |

The zoo's [`fixtures-d1-omni-600m.json`](fixtures-d1-omni-600m.json)
carries the same records and numbers, with each row's token ids, markers and bucket.

## Reproduce

[`models/d1-omni-600m/recipe.toml`](recipe.toml) in
the zoo lists every step: the oracle in its own environment (transformers 5.19.0), the exports in the zoo's
environment (coreai-torch 0.4.1, coreai-core 1.0.0b2), the strip, the compiles (Xcode 27.0 RC, coreai-build
3600.83.1), the gates and the staging. The scripts are in
[`conversion/d1_omni/`](../../conversion/d1_omni/); the lessons are in
[`knowledge/d1-omni-port.md`](../../knowledge/d1-omni-port.md).

## License

LFM Open License v1.0 (`LICENSE`, the publisher's text, unmodified). The graphs here are converted and modified forms of
the publisher's `model.safetensors`; `NOTICE.md` lists the changes: the networks re-authored in PyTorch and exported in
fp16, the debug locations removed, the decision graph cut into seven static lengths, the iPhone compiles, and the
files written for this repository. Commercial use is licensed only to an entity below the licence's threshold of
10 million US dollars in annual revenue (Section 5).

## Limits

- Liquid AI trained the audio questions on requests from an English speaker to an assistant, and cuts clips at 30 s.
- Measured on one Mac (M4 Max, macOS 27.0 26A428) and one iPhone 18 Pro (iOS 27.2.0, on USB power). The phone ran a
  78-row subset of the fixture and the five request shapes of the table; the two audio rows marked not measured were
  not run there.
- The Swift host was not timed at L64 / L128, and it was not run on the 9 audio rows that the 64 and 128 graphs take
  (the Python runtime was: they pass in the audio rows above).
- The Swift host decodes PNG and baseline 4:4:4 JPEG bit-equal to the publisher's decoder. A JPEG with chroma
  subsampling or progressive coding goes to ImageIO, which was not compared on such files; on a 4:4:4 photo it was up
  to 3 levels off on 19 % of the bytes.
- The Swift tokenizer (swift-transformers 1.3.3) takes about 0.4 s on a 3.4k-token state, longer than that decision.
