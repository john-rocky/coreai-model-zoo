# DeciderVision: the Swift read-out of decider-2b-vision

A Swift package that answers "image + questions -> the probability of every option" with the decider-2b-vision Core AI
port, on the system CoreAI framework and swift-transformers' tokenizer only (no engine: the pipelined engine exposes
no logits). It is the reference host a device gate links: the library `DeciderVision` builds for macOS 27 and iOS 27,
the executable `decider-vision` is the Mac CLI.

```swift
let decider = try await VisionDecider(bundle: bundleDir, decoder: aimodelcURL,     // decoder: nil = the bundle's .aimodel
                                      towers: [.g256: towerURL])
let probs = try await decider.decide(image: cgImage, context: "This is a visual question about the image.",
                                     questions: [DecisionQuestion(text: "How many circles are in the image?",
                                                                  options: ["1", "2", "3", "4", "5"])],
                                     grid: .g256)                                   // [[Double]], one row per question
```

`decide(image: nil, …)` reads a text-only row through the same decoder. `trace(…)` returns the ids, slots, letter
logits, full-vocabulary top-1, the tower output and every step's time.

## What each part copies

| file | the contract it reproduces |
|---|---|
| `ImagePreprocess.swift` | `host.resize_bicubic` (Pillow's bicubic, horizontal pass first, uint8 intermediate) and `host.patchify`; the file's 8-bit samples as decoded, no color matching |
| `PromptBuilder.swift` | `host.build_ids`: the author's `build()`, decode + re-encode, the image block `V + k`, the slot rule, `rope_shift_start/amount` |
| `VisionTower.swift` | patches f32 `[4 G², 1536]` -> image_embeds f32 `[G², 2048]` (G = 8 / 14) |
| `DecisionDecoder.swift` | the 2-function decoder (`main` S=1, `prefill` S=16), four zeroed states per row, the static inputs, the chunk order of the bundle's `decision.readout` |
| `VisionDecider.swift` | the glue and the read-out: letters `A..` (ids 32..) over the options, softmax at T = 1 in float64 |

The contract checks run at load (input / state / output names, shapes and types of both functions and the tower, the
letters' ids through the tokenizer), so an asset that differs fails there, not in a probability.

## Build and run (Mac)

```sh
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
# $ZOO_WORK_ROOT: the lane work root, by default the parent directory of this repository (conversion/_paths.py)
swift build -c release --scratch-path $ZOO_WORK_ROOT/_decider2bv/swift/.build
BIN=$ZOO_WORK_ROOT/_decider2bv/swift/.build/release/decider-vision
$BIN ask --bundle <bundle dir> --tower <tower .aimodelc> --grid 256 --image x.png --context "…" \
    --question "…" --options "a|b|c" [--question … --options …] [--asset aot|jit] [--json out.json]
$BIN fixture --bundle <bundle dir> --tower-g256 <…> --tower-g448 <…> --rows rows.json --images <dir> --out out.json \
    [--asset aot|jit] [--meta meta.json] [--reload] [--no-dump]
$BIN preprocess --images <dir> --out-dir <dir>      # the host half alone
$BIN make-image --out x.png --size 300 --circles 3  # a fixture-free test image
```

`--asset aot` (default) loads `<bundle>/../../bundles_aotc/<name>.h16c.aimodelc` with `SpecializationOptions.default`, as
the Python gates do; `--asset jit` specializes the bundle's `.aimodel` here, GPU preferred with frequent reshapes (the
exporter's AOT flags). `conversion/decider_vision/gate_swift.py` scores the CLI's JSON against the author's fp32 oracle,
the Python host and the Python read-out of the same bundle; `_time_mac.sh` times the fixture under the machine-wide GPU
lock (the FunASRGate protocol). The results are in `models/decider-2b-vision/gate-decider-2b-vision-swift.json`.

## For the device gate

- The decoder's JIT `.aimodel` read-out is correct in Swift on the Mac (S4 of the transcript), unlike the Python runtime's
  JIT; the iPhone path is device JIT of the `.aimodel` or an `h19p` AOT asset (an `h18p` asset is refused on the 18 Pro).
- The states are allocated once per decoder at `max_context_length` (4096) and zeroed per row (≈ 60 MB of fp16).
- A JPEG decoded by ImageIO is not guaranteed to equal Pillow's libjpeg decode; the fixture is PNG, where the decoded
  RGB equals Pillow's byte for byte.
