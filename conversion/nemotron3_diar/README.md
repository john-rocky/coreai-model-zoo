# nemotron3_diar — Nemotron-3-Diarization to Core AI

Exporter, gates and hosts for [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(8-speaker streaming Sortformer, OpenMDW-1.1), pinned at revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`.
The reference is transformers `Nemotron3DiarizationForAudioFrameClassification` in fp32. Card:
[`models/nemotron-3-diarization`](../../models/nemotron-3-diarization/README.md); porting notes:
[`knowledge/nemotron3-diarization-port.md`](../../knowledge/nemotron3-diarization-port.md).

## Files

| file | what it does |
|---|---|
| `make_reference.py` | oracle venv: transformers references (logits, probs, turns) and a capture of every encoder step, per clip and mode |
| `n3d_model.py` | the per-chunk network re-authored in plain PyTorch, loaded strictly from `model.safetensors` (sha256 checked) |
| `gate_reauthor.py` | re-authored graph vs the captures, padding zero and random, plus three negative controls |
| `export_n3d.py` | fp32 / fp16 bundles (T = 541 streaming, 684 offline), the four `.f32le` host constants, `metadata.json` (`--ship`: the published one, four bundles) |
| `gate_engine.py` | Core AI bundles vs the captures, one step at a time (fp32 on the CPU, fp16 on the GPU or Neural Engine) |
| `mel_frontend.py` | NumPy log-mel and the model card's streaming chunker |
| `gate_frontend.py` | oracle venv: the NumPy mel and embeds vs the transformers feature extractor |
| `host_loop.py` | NumPy host: packing, graph call, pooling, speaker cache (transformers' rules, float64 scores) |
| `gate_closed_loop.py` | whole-clip loops vs transformers: parts t (teacher-forced compressions), a, b, c, d, e, f |
| `aot_compile.py` | `xcrun coreai-build compile` for iOS h18p and macOS h16c, GPU and Neural Engine, with ANE regions counted by id |
| `bench_mac.py` | ms per chunk and the whole 97.6 s loop on the Mac GPU and Neural Engine |
| `export_golden.py` | golden files for the Swift and device gates (`_work/golden/`) |
| `stage_ship.py` | the Hugging Face repo laid out in `_work/ship/`, checked against `metadata.json`, with `SHA256SUMS` |
| `swift/` | `NemotronDiarizer` (the Swift host), `N3DGateSupport` (gate metrics), `n3d-selftest` (Mac CLI), `gate_swift.sh` |

The iPhone gate app is [`apps/N3DGate`](../../apps/N3DGate); it links `swift/` by path.
Intermediate files go to `_work/` (git-ignored).

## Reproduce

cwd = this directory. The fixture audio is in `<work root>/_n3d/fixtures/`: `diarization_example_16k.wav`
(97.6 s, the transformers integration-test clip) and `test_multispk_16k.wav` (21.5 s).

```bash
O=~/code/coreai/_n3d/venv-oracle/bin/python     # Python 3.12, torch 2.9.0, transformers git main @ 4b28d51
E=~/code/coreai/coreai-models/.venv/bin/python  # Python 3.11, torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2

# one chunk at a time
$O make_reference.py
$E gate_reauthor.py
$E export_n3d.py --dtype float32
$E export_n3d.py --dtype float16 --skip-assets
$E gate_engine.py --runs float32:cpu,float16:gpu
$E export_n3d.py --assets-only                  # hann_window_400.f32le
$O gate_frontend.py --downstream

# whole clips, offline, AOT, Mac speed
$O make_reference.py --mode very_low_latency,ultra_low_latency --skip-expected
$E export_n3d.py --dtype float32 --profile offline --skip-assets
$E export_n3d.py --dtype float16 --profile offline --skip-assets
$E gate_engine.py --profile offline --runs float32:cpu,float16:gpu
$E gate_closed_loop.py --parts t,a,b,f,c,d,e
$E aot_compile.py
$E gate_closed_loop.py --parts c --unit ane --label ane_jit
$E gate_engine.py --runs float16:ane
$E bench_mac.py

# Swift host
$E export_golden.py
$E export_n3d.py --metadata
cd swift
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
swift build -c release
./gate_swift.sh                                 # units, mel, s1, s2, s2chunks, s3, s4
./gate_swift.sh bench
cd ..

# ship staging and the device gate
$E aot_compile.py --bundle _work/artifacts/n3d_offline_float16.aimodel --tag offline --targets ios_gpu
$E export_n3d.py --metadata --ship              # metadata.ship.json: the four shipped bundles
$E stage_ship.py
../../apps/N3DGate/_build.sh && ../../apps/N3DGate/_stage.sh
../../apps/N3DGate/_gate.sh <udid>              # iPhone 17 Pro, unlocked, on USB
```

The safe-LayerNorm probe of the Neural Engine error (`--safe-ln` on `export_n3d.py`, `gate_reauthor.py`
and `gate_engine.py --tag _safeln`) is optional; its bundles are not shipped.
