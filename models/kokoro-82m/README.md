# Kokoro-82M — Core AI

The zoo's **first text-to-speech model**. [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0) is a tiny, high-quality **StyleTTS2 + iSTFTNet** synthesizer — 82M
parameters, 24 kHz output, non-autoregressive. Phonemes and a voice/style vector go
in; a waveform comes out in one pass (no token-by-token decode). It runs fully
on-device, English-first, with the grapheme→phoneme step on the host.

**Not an LLM.** There is no KV cache and no sampling. The acoustic graph is a fixed
feed-forward network whose only data-dependent length is the duration→alignment
expansion `L = Σ pred_dur`. The model is cut at that boundary into **three
`.aimodel` bundles** with two cheap host steps between them. Catalog id: **`kokoro-82m`**.

<!-- gen-cards:use-it begin id=kokoro-82m (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

⚡ **One line** — run the kit's task op on this model
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let audio = try await CoreAI.speak(text, options: .model("kokoro-82m"))
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Speak runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak)
(GUI + CLI, one app for every text-to-speech model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Speak/Speak.xcodeproj
# → Run, then pick "Kokoro 82M" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Speak
swift run speak-cli --model kokoro-82m --text "Hello from Core AI." --output hello.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let speaker = try await KitSpeaker(catalog: "kokoro-82m")
let audio = try await speaker.synthesize(text)
// audio.samples: 24 kHz mono PCM in [-1, 1] — play it or write a WAV
```

The take-home is [`Examples/Speak/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Speak/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitSpeaker(catalog:)` and plays the samples.
English-first: G2P is a dictionary over the bundled misaki lexicons (~180k words);
out-of-dictionary words are letter-spelled (no neural fallback). 28 voices ride the
download — `af_heart` is the default; the underlying `KokoroTTS` takes a `voice:`
label. Streaming? `synthesizeStreaming(_:onChunk:)` hands you a chunk per sentence.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 0.3 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Pipeline

```
text ──(misaki G2P, host)──▶ phoneme ids
  1. predictor.aimodel : ids[1,T] i32, ref_s[1,256], attn_mask[1,T]
                         ─▶ duration[1,T], d[1,T,640], t_en[1,512,T]
  host: pred_dur = round(duration); alignment one-hot aln[1,T,L]; frame_mask[1,L]
  2. prosody.aimodel   : d, t_en, aln, ref_s, frame_mask
                         ─▶ asr[1,512,L], F0[1,2L], N[1,2L]
  host: har = STFT( SineGen( f0_upsamp(F0) ) )   ── the hn-nsf source, a windowed FFT
  3. vocoder.aimodel   : asr, F0, N, har, ref_s, frame_mask ─▶ audio[1, L·600]
  host: trim to L·600 samples
```

The bundles are **voice-independent** — the voice *is* the `ref_s` input, one of the
shipped `voices/*.pt` packs (use `pack[len(tokens)−1]`). Token length **T** and frame
length **L** are fixed **buckets** (default **128 / 512**); the host left-pads to the
bucket and trims the output. Longer input is split into sentences host-side (as
Kokoro's own `KPipeline` does), each ≤ the token bucket.

### Graph contracts

```
predictor  in  input_ids[1,128] i32 · ref_s[1,256] f32 · attn_mask[1,128] f32 (1=real,0=pad)
           out duration[1,128] · d[1,128,640] · t_en[1,512,128]
prosody    in  d · t_en · aln[1,128,512] · ref_s · frame_mask[1,512] (1=real frame)
           out asr[1,512,512] · F0[1,1024] · N[1,1024]
vocoder    in  asr · F0 · N · har[1,22,frames] · ref_s · frame_mask
           out audio[1, 512·600]   (host trims to the real L·600)
```

`har` is the hn-nsf excitation's STFT, computed on the **host** (`compute_har` — an
`f0_upsamp` + `SineGen` + windowed FFT, ~10 lines of DSP / an Accelerate FFT on
device). It is the one piece that must stay off the engine: the source STFT's
`atan2` phase flips by 2π at the F0→0 pad boundary under the engine's fp32, so the
established CoreML ports compute it on the host too.

## Voices

All **28 English voices are Apache-2.0** (no attribution) — 11 `af_*`, 9 `am_*`
(American), 4 `bf_*`, 4 `bm_*` (British). Quality leaders: `af_heart` (A),
`af_bella` (A−), `af_nicole` / `bf_emma` (B−). (The Japanese/French voice packs in
the upstream repo carry a CC-BY attribution clause and are not shipped here.) The
host loads a voice pack and indexes it by utterance length: `ref_s = pack[len(ids)−1]`.

## Measured (macOS 27 beta, M4 Max, Core AI **CPU** compute unit)

| stage | latency | bundle (fp32) |
|---|---|---|
| predictor | ~70 ms | 83 MB |
| prosody | (in synth) | 38 MB |
| vocoder | (in synth) | 214 MB |
| **full utterance** (3 bundles + host) | **~0.75 s** | **~335 MB** |

Run on the **CPU** compute unit: the masked LSTMs are unrolled (StyleTTS2 has six
bidirectional LSTMs that torch.export cannot keep dynamic), and an unrolled LSTM is
~8 ms on the Core AI CPU vs dispatch-bound on the GPU. The fixed L bucket computes a
constant frame count regardless of the real length, so short utterances finish in
the same time — pick the smallest bucket that covers your text (re-export with
`--frame-bucket`).

## Numerics gate

The hn-nsf source phase is arbitrary (stock Kokoro randomizes it every call), so the
gate is **spectral**, not raw-waveform: per utterance the engine output's
magnitude-spectrogram correlation vs the patched torch reference is **0.999** (two
test sentences, `af_heart`). Raw waveform correlation is **~0.98** — the bounded
effect of the bucket pad boundary (the masked InstanceNorm + the convs at the
real/pad seam), perceptually inaudible. The torch reference itself is bit-exact
against stock Kokoro with the source noise removed (the deterministic export).

## ⬇️ Bundle

**[mlboydaisuke/Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI)**
— `kokoro_predictor.aimodel` + `kokoro_prosody.aimodel` + `kokoro_vocoder.aimodel`
(~335 MB, token bucket 128 / frame bucket 512) + the English `voices/*.pt` packs.
Apache-2.0. `kokoro_host_glue/` is the Swift-ready sidecar the kit path downloads:
`vocab.json`, `l_linear.bin`, all 28 voice packs as raw f32 `.bin`, and the misaki US
gold/silver lexicons — `KitSpeaker(catalog: "kokoro-82m")` phonemizes with a
**dictionary-first G2P** (~180k words; out-of-dictionary words are letter-spelled, no
neural fallback). Want misaki's full OOV model on-device instead? Use
[MisakiSwift](https://github.com/mlalma/MisakiSwift) in your app (it carries an MLX
dependency, which is why the kit doesn't bundle it).

Convert / re-bucket yourself:
[`conversion/export_kokoro.py`](../../conversion/export_kokoro.py)
(`python export_kokoro.py --out-dir out [--token-bucket 128 --frame-bucket 512]`;
`--verify --voice af_heart --text "…"` runs the engine-vs-torch spectral gate).
Host G2P is [misaki](https://github.com/hexgrad/misaki) (`misaki[en]`, no espeak for
English); on-device [MisakiSwift](https://github.com/mlalma/MisakiSwift) gives the
same English phonemes Python-free.

## The port in five lessons

StyleTTS2's vocoder is hostile to a fixed-shape, numerically-different engine — every
workaround below is load-bearing.

1. **Fold weight_norm, or the convs are random.** kokoro uses the *old hook-based*
   `torch.nn.utils.weight_norm`, where `module.weight` keeps its random init value
   until a forward fires the hook. The manual conv stand-ins read `module.weight`
   directly, so without `remove_weight_norm` the model is non-deterministic and
   explodes (output range to millions). This one bug masquerades as every other
   symptom — fix it first.
2. **Six bidirectional LSTMs → masked unrolls.** torch.export specializes nn.LSTM's
   sequence length (dynamic export fails), so lengths are fixed buckets and the host
   pads. A fused nn.LSTM leaks pad tokens into the backward pass and destroys the
   prosody (audio corr 0.02); the fix is a **masked** unrolled bi-LSTM that carries
   state through the right-padding — bit-identical to nn.LSTM at full length, exact
   under padding.
3. **58 AdaIN InstanceNorms normalize over L.** Bucket pad frames poison the mean/var;
   each decoder AdaIN takes the frame mask, normalizes over real frames only, and
   zeros its pad output so the convs see exact-like zeros (raw padding: 0.13 → 0.98).
   Resize the mask with `arange < real·N/Lb`, **not** `F.interpolate(nearest)` — the
   engine rounds the nearest boundary differently and misaligns the mask.
4. **The hn-nsf source goes on the host.** Its STFT phase (atan2) flips 2π at the
   F0→0 pad boundary under fp32 on the engine; computing that one windowed FFT on the
   host (`compute_har`) makes the engine match torch.
5. **Two coreai op traps.** `ConvTranspose1d` with `output_padding` gives a *symbolic*
   output length that poisons later concats, and `conv_transpose1d` returns **all
   zeros** for the iSTFT — both replaced by a bit-exact **zero-insertion + conv1d**.
   `input_ids` must be int32; `atan2(y,x)` → `2·atan(y/(|z|+x))`; SineGen's `%1` is a
   no-op for speech (f0/sr<1) and dropped.
