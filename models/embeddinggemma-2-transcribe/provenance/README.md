# Provenance

The records behind the numbers of the card, copied from the conversion's work directory with its paths replaced by
placeholders (`<work>` = the work directory, `<eg2-transcribe>` = the reader / app repository, `<venv>`, `<home>`).
`SOURCES.tsv` lists each copy, the record it was copied from, the speed-ladder row it backs, the sha256 of the record
as written and the sha256 of the copy.

| Folder | Records |
|---|---|
| `encoder/` | the fp16 encoder: Mac M4 Max GPU gate and timing; iPhone 18 Pro GPU gate and timing, AOT h19p and JIT |
| `reader/` | reader v1 + word table, fp16: Mac GPU gate, timing and the zero-half red arm; Mac Neural Engine gate; the iPhone 18 Pro GPU timing of the same compiled graph with an interim checkpoint's weights |
| `pipeline/` | the 3,257.4 s recording end to end on the Mac (encoder GPU ∥ reader Neural Engine with two windows in flight; encoder GPU → reader GPU one window at a time) and the float32 PyTorch path on the same windows |
| `fixture-20.md` | the transcripts of the 20 fixture utterances (reader v1's PyTorch path) and the one the Core AI fp16 reader writes differently |
| `strip/` | `strip_<name>.json`: the bundles of the HF repo's `macos/` and `ios/` are the gated bundles with their MLIR debug locations removed (coreai-torch `strip_debug_info`); `aot_h19p_<name>.json`: the bundles of `ios-h19p/` are compiled from those stripped `.aimodel` bundles and hold the same `resources.bin` as the compiles of the gated graphs; `compare_mac.json`: on the Mac the stripped bundles return outputs bit-identical to the gated ones |
| `encoder-split7/` | the second encoder form (the HF repo's `macos-encoder-split7-30s/`, Mac only): its gate, timing and measurement window, the one-bundle encoder's timing in that window, the strip records of its two bundles and the Mac compare; its own `README.md` gives the contract |
| `app/` | the eg2-transcribe app with reader v1 on the HF repo's files: on the Mac M4 Max its default form and both bundles on the GPU, on the iPhone 18 Pro its default form and the timing record of its passes (the 3,257.4 s recording, the 20 fixture utterances, launch to transcript after a Core AI cache clear and from the cache); its own `SOURCES.tsv` |

The `ios-h19p/` bundles were compiled from the stripped `.aimodel` bundles; the app ran them on the iPhone 18 Pro with
reader v1 (`app/`: fixture transcripts 20/20 = PyTorch, long-recording WER 5.77 %). The iPhone encoder and reader rows
of the card were measured on the compiles of the unstripped graphs.
