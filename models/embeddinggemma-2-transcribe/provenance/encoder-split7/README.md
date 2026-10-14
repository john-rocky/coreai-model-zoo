# Provenance — the second encoder form

The records behind `macos-encoder-split7-30s/` in the Hugging Face repository of this model: the EmbeddingGemma 2
audio encoder cut in two after conformer layer 7, as two JIT `.aimodel` bundles for the Mac. They are copied from the
conversion's work directory with its paths replaced by placeholders (`<work>` = the work directory,
`<eg2-transcribe>` = the reader / app repository, `<venv>`, `<caches>`, `<home>`). `SOURCES.tsv` lists each copy, the
record it was copied from, the speed-ladder rows it backs, the sha256 of the record as written and the sha256 of the
copy.

| File | Record |
|---|---|
| `mac-gate.json` | the gate of the pair on the 20 fixture clips (Mac M4 Max, the AOT h16c compile of each graph): one window at a time, twice (drift 0), the red arm, the front alone and the body alone against PyTorch float32, and the two-in-flight order against the sequential pass |
| `mac-timing.json` | the timing of the same pair: one window at a time (front, host cast and body per call) and two windows in flight |
| `mac-timing-window.json` | the measurement window it ran in and the forms compared in it |
| `mac-one-bundle-gpu-same-window-timing.json` | the one-bundle fp16 encoder on the GPU in that window (40.04 ms per window) |
| `strip_eg2_front_layer7_fp32_T3000.json`, `strip_eg2_body_layer7_fp16_T3000.json` | the two published bundles are the gated `.aimodel` bundles with their MLIR debug locations removed (coreai-torch `strip_debug_info`): the op counts and the `coreai-build inspect` summaries are unchanged, no local path is left, `main.hash` is new |
| `compare_mac.json` | the published pair on the Mac, the 20 fixture clips: hidden states and soft tokens bit-identical to the gated `.aimodel` pair and to the AOT h16c pair of the gate, its own gate table equal to `mac-gate.json`'s, two windows in flight bit-identical to one at a time |

## Contract

- `eg2_front_layer7_fp32_T3000.aimodel` (subsampling + conformer layers 0–7, float32): `features [1, 3000, 128]` +
  `half_mask [1, 1500]`, float32 → `hidden [1, 750, 1024]`, float32. Load it with the GPU preferred.
- The host casts `hidden` to float16.
- `eg2_body_layer7_fp16_T3000.aimodel` (conformer layers 8–11 + `embed_audio`, float16): `hidden [1, 750, 1024]`,
  float16 → `soft_tokens [1, 750, 512]`, float16. Load it with the Neural Engine preferred.
- The log-mel, `half_mask` and the 30 s windows are those of the one-bundle encoder (the model card).
- Two windows in flight: issue the front of window i + 1 together with the body of window i (two bundles: the GPU and
  the Neural Engine at once). Never issue two calls of one bundle's function together: on the one-bundle encoder that
  returns wrong soft tokens (max |Δ| 8.04 against the sequential pass).
- When it is faster: long recordings on a Mac. 20 windows in a row take 30.96 ms per window (the wall time divided by
  20) against 40.04 ms for the one-bundle encoder in the same measurement window; one window alone takes 57.40 ms.
- Not for the iPhone: on the iPhone 18 Pro the body is placed on the GPU with the Neural Engine preferred, its Neural
  Engine AOT h19p fails to load (`failedToSpecialize`), and the float32 front alone takes 124.4 ms per 30 s window;
  two windows in flight take 144.0–144.6 ms per window against 58.4–60.9 ms for the one-bundle fp16 encoder in the
  same launch. `ios/` and `ios-h19p/` do not have it.
- The app of the eg2-transcribe repository does not download it: the app fetches `macos/`, and this folder sits at the
  root of the repository.
