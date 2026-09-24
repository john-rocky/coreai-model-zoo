# Nemotron-3-Diarization — Core AI

Speaker diarization ("who spoke when") for up to **8 speakers** at 10 ms resolution, streaming or
offline, on-device. [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(OpenMDW-1.1, 99.2M parameters, a streaming Sortformer) has its per-chunk network exported as one
Core AI graph. The host keeps the state: the log-mel front end, the 8-frame projection and the speaker
cache (AOSC compression and a FIFO), ported line by line from transformers' `Nemotron3DiarizationSpeakerCache`.

It succeeds the 4-speaker [Streaming Sortformer port](../sortformer-diar/README.md). Another Core AI
conversion of this checkpoint exists:
[smdesai/Nemotron-3-Diarization-CoreAI](https://huggingface.co/smdesai/Nemotron-3-Diarization-CoreAI).

- 🤗 [mlboydaisuke/Nemotron-3-Diarization-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3-Diarization-CoreAI) (revision `8dc6258`, 2026-09-24; mirror:
  [coreai-community/Nemotron-3-Diarization-CoreAI](https://huggingface.co/coreai-community/Nemotron-3-Diarization-CoreAI)).
- Conversion, gates and the Swift host: [`conversion/nemotron3_diar`](../../conversion/nemotron3_diar).
- iPhone gate app: [`apps/N3DGate`](../../apps/N3DGate).
- Porting notes: [`knowledge/nemotron3-diarization-port.md`](../../knowledge/nemotron3-diarization-port.md).

## How it works

The graph takes the rows transformers feeds its encoder and returns speaker logits:

```
packed [1, T, 512]    float32  [speaker cache | FIFO | chunk + look-ahead] rows, left-packed in [0, L); rows [L, T) zero
valid  [1, T]         float32  1.0 for rows < L, 0.0 after
  -> logits [1, T*8, 8]  float32  speaker logits at 10 ms; rows [0, L*8) are real
```

Inside the graph: input LayerNorm, 31 pre-LN layers (RoPE on all 64 head dims with positions
0..T−1 baked in as constants, key mask `(1 − valid) · (−1e4)`, erf GELU), the final LayerNorm, the
512→192 projection, a multiply by `valid`, the sub-pixel Conv1d 192→1536 and the classifier. The fp16
bundles cast inside, so the host passes float32 either way.

The host does the rest:

1. Log-mel: pre-emphasis 0.97, 512-point STFT (hop 160, the 400-sample window centered), 128 slaney
   mel bins, `log(x + 2⁻²⁴)`. Each streaming chunk takes its mel from its own audio slice, like the
   source card's `inputs_generator`.
2. 8 mel frames are stacked and projected (`embedder_projection.f32le`, float32): one 512-d row per 80 ms.
3. Per step: pack the rows, run the graph, average-pool `sigmoid(logits[:L*8])` 8× to the cache rate,
   emit the chunk's frames.
4. The speaker cache follows transformers' `update` / `_compress`. Its compression scores are float64.
5. Turns: per speaker, every run of probability > 0.5; overlaps stay.

`metadata.json` in the bundle directory holds every host constant.

## Profiles

One graph at T = 541 serves the three streaming modes: the rows are left-packed and `valid` marks how many
are real. Offline needs the larger T = 684. The speaker cache holds 264 rows in every profile.

| profile | chunk + look-ahead (80 ms frames) | FIFO / update period | graph | 97.6 s agreement, Mac GPU |
|---|---|---|---|---|
| low latency | 9 + 4 (0.72 s + 0.32 s) | 264 / 222 | `n3d_streaming_float16`, T = 541 | 99.9987 % |
| very low latency | 6 + 2 (0.48 s + 0.16 s) | 264 / 222 | `n3d_streaming_float16`, T = 541 | 99.9424 % |
| ultra low latency | 3 + 1 (0.24 s + 0.08 s) | 264 / 222 | `n3d_streaming_float16`, T = 541 | 99.9949 % |
| offline | 340 + 40, over the recording's mel | 40 / 300 | `n3d_offline_float16`, T = 684 | 99.9987 % |

## Verification

The reference is transformers `Nemotron3DiarizationForAudioFrameClassification` in fp32 (git main
`4b28d51`, 5.18.0.dev0) on `nvidia/Nemotron-3-Diarization@f667ed7`. The 97.6 s clip is the transformers
integration-test audio; there the cache compresses 4 times in low latency. The 21.5 s clip is the previous
generation's fixture. The score is speaker-activity agreement at 0.5 over every 10 ms frame × 8 speakers,
with a bar of 99.9 %. Offline leaves out the last 16 frames, as the transformers integration test does.

Closed loop, whole clip, fp16 graph on the Mac GPU:

| clip | profile | agreement@0.5 (differing / elements) | max \|Δp\| | turns: reference / ours / matched | verdict |
|---|---|---|---|---|---|
| 97.6 s | low latency | 99.9987 % (1 / 78,072) | 0.025 | 37 / 38 / 36 | PASS |
| 97.6 s | very low latency | 99.9424 % (45 / 78,072) | 0.362 | 43 / 43 / 42 | PASS |
| 97.6 s | ultra low latency | 99.9949 % (4 / 78,072) | 0.011 | 61 / 60 / 59 | PASS |
| 97.6 s | offline | 99.9987 % (1 / 77,952) | 0.029 | 29 / 29 / 29 | PASS |
| 21.5 s | low latency | 100 % (0 / 17,192) | 0.0031 | 10 / 10 / 10 | PASS |
| 21.5 s | very low latency | 100 % (0 / 17,192) | 0.0025 | 10 / 10 / 10 | PASS |
| 21.5 s | ultra low latency | 100 % (0 / 17,192) | 0.0027 | 10 / 10 / 10 | PASS |
| 21.5 s | offline | 99.9941 % (1 / 17,072) | 0.0037 | 9 / 9 / 9 | PASS |
| 97.6 s | low latency, fp32 graph, CPU only | 100 % (0 / 78,072) | 5.5e-6 | 37 / 37 / 37 | PASS |
| 97.6 s | low latency, fp16 graph, Mac Neural Engine | 99.7746 % (176 / 78,072) | 0.296 | 37 / 42 / 36 | below bar |
| 21.5 s | low latency, fp16 graph, Mac Neural Engine | 100 % (0 / 17,192) | 0.0076 | 10 / 10 / 10 | PASS |
| 97.6 s | low latency, iPhone 17 Pro GPU (h18p) | 99.9987 % (1 / 78,072) | 0.026 | 37 / 37 / 37 | PASS |
| 21.5 s | low latency, iPhone 17 Pro GPU (h18p) | 100 % (0 / 17,192) | 0.0017 | 10 / 10 / 10 | PASS |
| 97.6 s | offline, iPhone 17 Pro GPU (h18p) | 99.9987 % (1 / 77,952) | 0.026 | 29 / 29 / 29 | PASS |
| 21.5 s | offline, iPhone 17 Pro GPU (h18p) | 99.9941 % (1 / 17,072) | 0.0030 | 9 / 9 / 9 | PASS |
| 97.6 s | low latency, iPhone 17 Pro Neural Engine (h18p) | 99.7758 % (175 / 78,072) | 0.296 | 37 / 42 / 36 | below bar |
| 21.5 s | low latency, iPhone 17 Pro Neural Engine (h18p) | 100 % (0 / 17,192) | 0.0082 | 10 / 10 / 10 | PASS |

iPhone rows: iOS 27.0 (build 24A437), thermal state nominal before and after every step, measured by
[`apps/N3DGate`](../../apps/N3DGate) on 2026-09-24 (run `20260924-174829`). The iPhone GPU's decisions
equal the Mac GPU's on 78,070 of 78,072 elements of the 97.6 s low-latency run (max |Δlogit| 0.053); the
logits are not bit-identical across the two GPUs.

The Python host (`host_loop.py`) and the Swift host (`NemotronDiarizer`) produce these rows with
bit-identical logits in all 8 GPU runs. Each stage was gated before the next one:

- Re-authored graph in eager fp32 vs transformers, one chunk at a time: max |Δlogit| 4.96e-5 over
  166 chunks, with the padding rows zero or random. Three negative controls (no RoPE, no key mask, no
  `× valid`) all fail.
- Core AI fp32 on the CPU alone: max |Δlogit| 4.58e-5 over the same 166 chunks. fp16 on the GPU:
  max |Δlogit| 7.60e-2, and 99.9991 % agreement over 3,378,432 elements.
- Host log-mel vs the transformers feature extractor: max |Δ| 1.81e-4, bar 5e-4 (28 of 1,249,280
  cells above 1e-4). With it in every chunk, the eager graph's logits stay within 5.15e-5 of
  transformers'. The Swift mel is bit-identical to the NumPy mel.
- Poisoned loops fail: without compression 98.2298 %, without the FIFO pop 87.17 %.

The Neural Engine bundle compiles as one region, fully placed (`mps.fullyPlacedOnANE`). Its per-chunk
max |Δlogit| is 0.194, against 0.076 on the GPU. On the 97.6 s clip its cache departs from the reference
at the first compression, and 173 of its 175 differing frames come after that. No ANE bundle is shipped.

## Speed

Mac: M4 Max, macOS 27.0 (26A428). Another session held the GPU lock during every run, so all rows are
**contended**. One chunk is 0.72 s of audio in low latency.

| engine | host | ms per chunk, median / p90 | 97.6 s end to end | real-time factor |
|---|---|---|---|---|
| fp16, GPU | Python | 15.88 / 16.09 | 2.25 s | 0.023 |
| fp16, GPU | Swift (5 runs) | 15.37–15.54 / 15.55–15.77 | 2.13–2.15 s in 4 runs; 6.16 s in 1 (cause not found) | 0.022; 0.063 in the slow run |
| fp16, Neural Engine | Python | 33.05 / 33.47 | 4.61 s | 0.047 |

Warm loads took 0.12 s (GPU) and 0.01 s (Neural Engine). A first load took 1.6 s for the Neural Engine
from the `.aimodel`, and 23.7 s for its ahead-of-time compile (`.h16c.aimodelc`).

iPhone 17 Pro (iPhone18,1, iOS 27.0 build 24A437), measured by [`apps/N3DGate`](../../apps/N3DGate) on
2026-09-24, thermal state nominal before and after every step, low power mode off:

| bundle | ms per chunk, median / p90 | 97.6 s end to end | real-time factor | load, first / second | first call | footprint |
|---|---|---|---|---|---|---|
| `n3d_streaming_float16.h18p.aimodelc`, GPU | 30.05 / 30.52 | 4.16 s | 0.043 | 0.68 s / 0.12 s | 2.69 s | 111 MB |
| `n3d_offline_float16.h18p.aimodelc`, GPU (one chunk = 30.4 s of audio) | 34.55 / 35.76 | 0.17 s | 0.0018 | 0.55 s / 0.06 s | 0.78 s | 251 MB |
| streaming graph, Neural Engine compile (not shipped) | 31.47 / 32.50 | 4.36 s | 0.045 | 1.27 s / 0.01 s | 0.03 s | 314 MB |

The gate is one command (`_gate.sh <udid>`): install, md5-checked asset push, launch, poll, summary. Its
result files are under `conversion/nemotron3_diar/_work/device_runs/`.

## Reproduce

cwd `conversion/nemotron3_diar/`. `O` is the oracle venv (Python 3.12, torch 2.9.0, transformers git
`4b28d51`); `E` is the export venv (Python 3.11, torch 2.9.0, coreai-torch 0.4.1, coreai-core 1.0.0b2).

```bash
$O make_reference.py                                        # transformers references and per-chunk captures
$E gate_reauthor.py                                         # re-authored graph vs transformers + 3 negative controls
$E export_n3d.py --dtype float32                            # constants + fp32 bundle (T = 541)
$E export_n3d.py --dtype float16 --skip-assets              # fp16 bundle
$E gate_engine.py --runs float32:cpu,float16:gpu
$E export_n3d.py --assets-only                              # hann_window_400.f32le
$O gate_frontend.py --downstream                            # host mel vs transformers
$O make_reference.py --mode very_low_latency,ultra_low_latency --skip-expected
$E export_n3d.py --dtype float32 --profile offline --skip-assets
$E export_n3d.py --dtype float16 --profile offline --skip-assets
$E gate_engine.py --profile offline --runs float32:cpu,float16:gpu
$E gate_closed_loop.py --parts t,a,b,f,c,d,e                # closed loops, the table above
$E aot_compile.py                                           # h18p / h16c, GPU and Neural Engine
$E bench_mac.py
$E export_golden.py                                         # golden files for the Swift and device gates
$E export_n3d.py --metadata
cd swift && swift build -c release && ./gate_swift.sh && ./gate_swift.sh bench
```

Swift needs `DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer`. For shipping:
`aot_compile.py --bundle _work/artifacts/n3d_offline_float16.aimodel --tag offline --targets ios_gpu`,
then `export_n3d.py --metadata --ship` (the published `metadata.json`, four bundles) and `stage_ship.py`,
which lays out the repo in `_work/ship/` and checks it against that file.

## Lessons

- **Compression decisions can sit inside float32 rounding.** In 3 of the 16 compressions of these
  clips, transformers' own top-k choice is 2 ulp from its boundary. A float32 host flipped one of them,
  and the cache diverged from there on. The host scores in float64.
- **The Neural Engine error is not LayerNorm overflow.** The final LayerNorm's input reaches
  |x| = 1,216, so fp16 variance sums can overflow. LayerNorms rewritten so fp16 cannot overflow barely
  moved the per-chunk error (max |Δlogit| 0.194 → 0.163, agreement 99.9970 → 99.9965 %), the closed
  loop stayed below the bar (99.8271 %), and neither version produced NaN or inf.
- **The reference's own fp32 STFT sets the mel bar.** transformers' `torch.stft` is 1.82e-4 from a
  float64 STFT; NumPy with the same window bits is 1.0e-6. A bar of 1e-4 was below the reference's noise.
- **Sum the Swift DFT in Double.** A float32 matmul DFT drifted 1.18e-4 from NumPy's rFFT and changed
  compression decisions on the 97.6 s clip. Summing in Double and rounding to float32 made the mel bit-identical.
- **The last encoder frame offline is padding.** The feature extractor's extra centered frame becomes a
  masked row that transformers does not zero before its Conv1d, so its last 8 frames carry that row's
  garbage. The host drops the frame, and the gates skip the last 16 frames, as transformers' test does.
- **Do not name a wrapper attribute `graph`.** torch.export's unlift resolves `graph.*` against
  `GraphModule.graph` and fails (`references nonexistent attribute model of graph`).

## License

OpenMDW-1.1, the source model's license. The published repo carries `LICENSE` (the agreement) and
`NOTICE` (origin, revision, what was converted); keep both with any part of it you redistribute.
