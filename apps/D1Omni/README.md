# D1Omni: the Swift host of d1-omni-600M

A Swift package that answers a SystemOne request — a state and typed questions (`noul`, `choice`, `score`) — with the
probabilities of [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) on Core AI, on the system
CoreAI framework, Accelerate, ImageIO and swift-transformers' tokenizer only (pinned to 1.3.3, the version the gate ran). The
package declares macOS 27 and iOS 27; everything below was built and run on a Mac, and the iPhone build is the device
round's gate app ([`apps/D1OmniGate`](../D1OmniGate/)). The executable `d1omni` is the Mac CLI. The bundles are staged
for the Hugging Face repository `mlboydaisuke/d1-omni-600M-CoreAI` (not uploaded yet); the card is
[`models/d1-omni-600m/`](../../models/d1-omni-600m/), the transcript of these gates `gate-d1-omni-600m-swift.json` there.
`conversion/d1_omni/host.py` is the specification every file here copies, and `conversion/d1_omni/gate_swift.py`
checks the copy against the publisher's model and against the Python reference on the same graphs. Text, image and
audio requests run end to end: the images are decoded, cut, resized and patched here and run through the vision graph,
the clip is read, turned into a log-mel and run through the audio graph of its length bucket.

```swift
import D1Omni

let d1 = try await D1Omni(folders: D1Omni.folders(macos: dir),            // dir: macos/ (or ios/, ios-h19p/) of the bundles
                          media: D1Omni.MediaFolders.find(macos: dir))    // vision-fp16/ and audio-fp16-<sec>s/ beside them
let response = try await d1.systemOne(state: .string(log), questions: questions)
let seen = try await d1.systemOne(state: nil, questions: questions, images: [photoURL])    // PNG or JPEG files
let heard = try await d1.systemOne(state: nil, questions: questions, audio: wavURL)       // 16 kHz mono 16-bit WAV
print(PythonFormat.dumps(response, asciiOnly: false))            // as json.dumps(response, ensure_ascii=False) writes it
```

`dir` holds one folder per decision bucket, `decide-fp16-L<L>/` (the Hugging Face layout) or `fp16-L<L>/` (the
conversion's work folder), each with `metadata.json`, the graph `metadata.json` names (`assets.main`: the `.aimodel`,
or the `.h19p.aimodelc` in `ios-h19p/`) and `tokenizer/`. Every such folder in `dir` is a bucket (the shipped folders:
L = 64, 128, 256, 512, 1024, 2048, 4096), and a row goes to the smallest bucket that holds its media prefix and its
ids (host.py `bucket_for(positions, ALL_BUCKETS)`): the card's one-question requests (47–56 positions) run at L64. Keep
a trial bucket out of a shipped folder, or it takes the short rows. The tokenizer is the smallest bucket's
`tokenizer/` (the same file in every bucket). A bucket's graph loads on its first row (`preload: true`
loads them all at init); `assets:` swaps a bucket's file for another one (the CLI's `--asset aot` passes the Mac's
`.h16c.aimodelc`). Both load with `SpecializationOptions(preferredComputeUnitKind: .gpu)`, never `.default`. The media
folders (`MediaFolders`) are `vision-fp16/` (the graph and `position_table.f32`) and `audio-fp16-<sec>s/` for 5 / 10 /
20 / 30 s (each graph and `mel_filters_128x257_f32.bin`); their graphs load on first use, and `assets:` swaps one in
the same way (`"vision"`, `"audio-<sec>"`).

## What each part copies

| file | the contract it reproduces (`conversion/d1_omni/host.py`) |
|---|---|
| `JSONValue.swift` | the request as written: members in order, number literals kept (`1250` ≠ `1250.0` in the text the model reads), `json.loads`' duplicate-key and NaN rules; keys compared by code points (from apps/Kev) |
| `PythonFormat.swift` | Python's `str()` and `repr()` of a value, `json.dumps` (`ensure_ascii` off for the state, `separators=(", ", ": ")` for a criterion), Python 3.12's compensated `sum` (from apps/Kev) |
| `Tokenizer.swift` | `escape()` (`<\|name\|>` → `<¦name¦>` on code points) and `enc` = `tok(escape(s), add_special_tokens=False)`: no `<\|startoftext\|>` from the post-processor, `""` → no ids; the nine token ids looked up by text and checked against `metadata.json` |
| `Prompt.swift` | `as_question` with the publisher's messages, `serialize`, `_criterion`, `render_options` (per mode: the noul default, the audio `option_000:` form), `encode` (budget, per, room, the cut), the per-mode `max_len`, `request_rows`, `temperature_key`; the image and audio prefix lengths (vision.py `layout`, the audio subsampling) |
| `DecisionGraph.swift` | `graph_inputs` (the six inputs at the bucket's L), `bucket_for`, the `main` function's contract checked at load |
| `Readout.swift` | `probabilities_from_logits`: ÷ T on a text row, the float32 softmax with NumPy's pairwise sum, the noul flipped to [yes, no]; `answer()` and `response()` |
| `D1Omni.swift` | the glue, in host.py's order; the buckets' metadata read and checked against each other; `systemOne(…images:)` / `(…audio:)` (round 9) |
| `ImagePreprocess.swift` | the image path of host.py §5 in its NumPy form: the decode as PIL's `convert("RGB")` reads the file (ImageIO's decoded bytes: no EXIF rotation, no ICC profile applied, alpha dropped), vision.py's crops (`layout`: a tiled image's grid resize cut into 512 px tiles, then the thumbnail), torchvision's `resize(uint8, BILINEAR, antialias=True)` as it runs on a CPU without AVX2 (`host.resize_uint8_antialias_numpy`: float32, torch's separable antialias kernel — width pass, then height, each tap a fused multiply-add — rounded half to even), `patchify` ((x − 127.5) / 127.5, [py][px][c] in a patch), the position table resized by the same kernel (`host.position_embeddings_numpy`), `unshuffle_index` |
| `JPEGBaseline.swift` | a baseline JPEG as PIL's libjpeg-turbo decodes it (jidctint.c's integer IDCT, jdcolor.c's fixed-point YCbCr → RGB), for the JPEGs whose components share one sampling factor; other JPEGs (chroma subsampling, progressive) go to ImageIO, which was up to 3 levels off PIL on a 4:4:4 photo and was not compared on these |
| `AudioPreprocess.swift` | the audio path of host.py §6 / `mel_host.mel_numpy`: a 16-bit PCM mono 16 kHz WAV read as int16, `waveform()` (30 s cut, / 32768, 0.5 s pad), the log-mel in float64 (preemphasis, the centred Hann(400) frames, vDSP's double real FFT, the bundle's Slaney filterbank by `cblas_dgemm` as NumPy's matmul calls it, `log(x + 2^-24)`, the per-row mean and std with NumPy's pairwise sums) cast to float32, the clip bucket and the four ConvSubsampling masks |
| `MediaGraphs.swift` | the vision graph (one crop → 256 rows, the crop's (ph/2)(pw/2) kept) and the audio graph of one bucket (mel + masks → T rows, P kept), each `main` checked against its contract at load |

## What the gate checked (Mac M4 Max, macOS 27.0 26A428, 2026-10-08)

| check | result | transcript (`$ZOO_WORK_ROOT/_d1_omni/`) |
|---|---|---|
| rows: every fixture row's ids, markers, the cuts, bucket and the five graph inputs (sha256) against the publisher's `prompt.encode` and host.py | 479 / 551 / 623 rows (fixture v1 / v2 / v3), every field equal; a one-word change in one question flags that row only | `results/swift_rows_gate.json` |
| readout: the publisher's raw marker logits through the Swift readout, the responses through `answer()` | p bit-equal to host.py on 470 + 90 + 72 rows; 402 + 30 + 24 responses byte-equal to `json.dumps` of the publisher's | `results/swift_rows_gate.json` |
| tokenizer: 79 edge strings (whitespace, combining marks, emoji, `<\|…\|>`, the five special tokens that are not `<\|…\|>`) | plain and escaped ids equal to the `tokenizers` library | `results/swift_rows_gate.json` |
| parity: the 470 oracle rows (459 text + 11 with the oracle's media prefix) on the fp16 graphs, AOT and JIT | FACTS §7 PASS both (argmax 470/470, max \|Δp\| 0.0067, mean 0.00098); marker logits and p bit-equal to the Python runtime on every row; JIT = AOT on every row; the wrong-pairing control FAILs | `results/swift_parity_gate.json` |
| L1024 (no fixture row has 513–1024 positions): one request with three rows of 961–1,009 positions | AOT, JIT cold and JIT warm responses byte-equal to the Python runtime's | `results/swift_l1024_probe.json` |
| images (round 9): the 16 fixture images (15 PNG, among them four with an ICC profile and four with EXIF; one baseline 4:4:4 JPEG), 62 crops, against the Python host's dump (`host_dump_media.py`) on the ship form (vision fp16, decision fp16) | the decoded RGB 16/16, the crops 62/62, `pixel_values` / `pos_embed` / `patch_mask` / `unshuffle_index` 62/62 and the prefix 16/16 bit-equal; the 46 image rows' marker logits and p bit-equal to the Python runtime; FACTS §7 PASS (argmax 46/46, max \|Δp\| 0.0062, mean 0.0011); the next image's prefix as a control FAILs; JIT = AOT | `results/swift_media_image.json` |
| audio (round 9): the 16 fixture clips (15 WAV; one measured-only FLAC, which the Python dump reads with soundfile and hands over as samples), against the same dump's NumPy-mel path | the samples, the four masks and the prefix 16/16 bit-equal; the mel bit-equal but for 1 of 3,522,048 values (1 ulp, 4.5e-13: the FFT is vDSP's, NumPy's is pocketfft); the 46 audio rows bit-equal to the Python runtime; FACTS §7 PASS (46/46, 0.0032, 0.00077); the next clip's prefix FAILs; JIT = AOT | `results/swift_media_audio.json` |
| resize (round 9): `host.resize_uint8_antialias_numpy` against torchvision on the 62 crops | 62/62 bit-equal (42 of them resized); a float64 filter differs on 40, PIL's BILINEAR on 42, rounding half up on 17 | `results/resize_numpy_gate.json` |
| stripped bundles (round 10): the 562 rows above (text 470, image 46, audio 46) on the bundles without debug locations, AOT and JIT | bit-equal to the Python runtime on every row, and to the unstripped run | `results/ship_swift_parity.json` |
| small buckets (round 12): the 342 text rows of 128 positions or fewer at L64 (64 rows) and L128 (278), AOT and JIT | bit-equal to the Python runtime on every row; FACTS §7 PASS (argmax 342/342, max \|Δp\| 0.0117, mean 0.0011); JIT = AOT. The 9 audio rows of 128 positions or fewer were run at L64 / L128 by the Python runtime only | `results/swift_parity_L64_128.json` |

FACTS §7 is the bar of every reduced-precision gate here: the argmax equal to the oracle's (the publisher's model in
fp32 on the CPU) on every row whose oracle top-2 margin is above 0.02 (near ties reported apart), max |Δp| ≤ 0.02, and
the mean of the rows' max |Δp| ≤ 0.002.

## Speed (two measurement windows, the rule fixed first: `results/timing/ranking_rule_r8.md`)

One decision = the graph inputs → `InferenceFunction.run` → the scores copied → marker gather + temperature +
softmax, the mean of the two windows' medians (30 decisions each); tokenizing is timed apart.

| workload | JIT | AOT | Python runtime, same AOT, same windows | tokenize (Swift) |
|---|---:|---:|---:|---:|
| W1 one question (47 positions, L256) | 16.14 ms | 16.16 ms | 16.67 ms | 3.4 ms |
| W2 three questions (3 calls, L256) | 48.46 ms | 48.59 ms | 49.55 ms | 3.3 ms |
| W3 a 3.4k-token state (3,454 positions, L4096) | 301.21 ms | 300.84 ms | 301.50 ms | 397–404 ms |

The JIT and AOT outputs were bit-equal on every timed call. Load (`AIModel` + `loadFunction`), L256 / L4096: cold
(no cache entry) AOT 2.00 / 2.43 s, JIT 1.93 / 2.39 s; a second launch AOT 1.04 / 1.11 s, JIT 1.64 / 0.99 s; a second
load in the same process 0.14–0.35 s. On the Mac the `.aimodel` alone is enough. The tokenizer loads in 0.18 s; its
BPE takes 0.4 s on a 3.4k-token state (the Python `tokenizers` library: 2.0 ms), longer than the decision itself —
not addressed yet. The Swift host was not timed at L64 / L128 (the Python runtime was, in round 12: 7.35 ms for the
one question at L64; `gate-d1-omni-600m-timing-mac.json`). The iPhone is measured in the device round.

The media workloads (round 9, one measurement window, the rule fixed first: `results/timing/ranking_rule_r9.md`): one
decision = the media graph on inputs prepared beforehand → its P rows → the decision graph → softmax; the median of 30.

| workload | JIT | AOT | Python runtime, same AOTs, same window | media graph / decision graph (AOT) | preprocessing (Swift) |
|---|---:|---:|---:|---:|---|
| W4 a 384 px image (img_01, 1 crop, 144 + 39 positions, L256) | 37.37 ms | 37.47 ms | 37.98 ms | 21.21 / 16.23 ms | PNG decode 1.91, crops 0.08, patches + positions 0.73, tokenize 0.35 ms |
| W5 9.6 s of audio (aud_01, 10 s bucket, 121 + 63 positions, L256) | 24.48 ms | 24.78 ms | 25.49 ms | 8.60 / 16.16 ms | WAV 0.30, mel + masks 1.80, tokenize 3.58 ms |

The JIT and AOT prefixes and outputs were bit-equal on every timed call. Load (cold, `AIModel` + `loadFunction`):
vision AOT 0.33 s / JIT 0.60 s, audio 10 s AOT 0.39 s / JIT 1.40 s, decision L256 AOT 1.07 s / JIT 2.09 s; a second
load in the same process 0.04–0.13 s.

## Build and run

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer   # the SDK with CoreAI.framework
swift build -c release --package-path apps/D1Omni --scratch-path $ZOO_WORK_ROOT/_d1_omni/swift/.build
B=$ZOO_WORK_ROOT/_d1_omni/swift/.build/out/Products/Release/d1omni
$B ask --bundle-dir <macos> --state "<text>" --question '{"type": "noul", "instructions": "Is the customer asking for a refund?"}'
$B ask --bundle-dir <macos> --image room.png --question '{"type": "noul", "instructions": "Is there a bed in the room?"}'
$B ask --bundle-dir <macos> --audio note.wav --question '{"type": "noul", "instructions": "Does the caller want a refund?"}'
```

`d1omni rows | tokenize | parity | parity-media | ask | time` — the header of `Sources/d1omni-cli/main.swift` lists
their arguments; `conversion/d1_omni/gate_swift.py` runs them (`rows`, `pyref`, `parity`, `window` under
`quiet_hold.py`, `timing`; round 9: `resize`, `media`, `media-window` under `quiet_hold.py`, `media-timing`).

What the media path does not read: a JPEG with chroma subsampling or progressive coding is decoded by ImageIO (on a
4:4:4 baseline photo ImageIO was up to 3 levels off PIL's libjpeg-turbo on 19 % of the bytes; no subsampled or
progressive file is in the fixtures, so neither its pixels nor its effect on the answers was compared); images other than 8-bit RGB / gray / indexed (and a translucent pixel that ImageIO premultiplied)
throw; audio other than 16-bit PCM mono at 16 kHz throws (the publisher does not resample either), and FLAC is not read
here.
