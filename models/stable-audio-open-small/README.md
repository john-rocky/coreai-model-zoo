# Stable Audio Open Small — Core AI (on-device music generation)

**The model zoo's first MUSIC / AUDIO generation model for Apple Core AI.** Type a prompt, get ~11s
of 44.1 kHz stereo audio — generated entirely **on-device** on Apple Silicon. A community port of
[`stabilityai/stable-audio-open-small`](https://huggingface.co/stabilityai/stable-audio-open-small)
(Stability AI + Arm) to Core AI.

A latent **diffusion** text-to-audio model: a T5 text encoder conditions a DiT (diffusion transformer)
that denoises a latent over **8 rectified-flow steps**, then an Oobleck VAE decodes the latent to a
waveform. Distilled (ARC) for few-step generation, so it's fast.

<!-- gen-cards:use-it begin id=stable-audio-open-small (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
![Stable Audio Open Small demo](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI/resolve/main/demo.gif)
*Stable Audio Open Small on iPhone 17 Pro — the zoo's coreai-audio app, 12 s of audio in ~1 s.*

## Use it

⚡ **One line** — this model is the default behind the kit's task op
(`import CoreAIOps`; no session, no model plumbing, downloads on first use):

```swift
let audio = try await CoreAI.compose(prompt)
```

Every op, one shape — [Cookbook](https://github.com/john-rocky/coreai-kit/blob/main/docs/COOKBOOK.md).

▶️ **Run it (source)** — the [Music runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Music)
(GUI + CLI, one app for every text-to-music model in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/Music/Music.xcodeproj
# → Run, then pick "Stable Audio Open Small" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/Music
swift run music-cli --model stable-audio-open-small --prompt "128 BPM tech house drum loop" --output loop.wav
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let musician = try await KitMusician(catalog: "stable-audio-open-small")
let audio = try await musician.generate(prompt)
// audio.samples: 44.1 kHz stereo (planar L/R) — play it or write a WAV
```

The take-home is [`Examples/Music/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/Music/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI drives the same `KitMusician(catalog:)` and plays the result.
Length? `generate(_:seconds:)` up to the model's ~11 s window. The WAV container is your
app's territory (the runner ships a 30-line writer with planar-stereo support).

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed (macOS)
- First run downloads the model — 1.1 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## What's in the bundle (`macos/`)

Three Core AI `.aimodel` bundles + a tiny host sampler loop:

| bundle | role | I/O |
|---|---|---|
| `sa_cond_fp16b` | T5-base encoder + number conditioner | `input_ids[1,64], attention_mask[1,64], seconds_norm[1] → cross_attn_cond[1,65,768], global_embed[1,768], cond_mask[1,65]` |
| `sa_dit_fp16` | diffusion transformer (run 8×) | `x[1,64,256], t[1], cross_attn_cond, global_embed, cross_attn_cond_mask → v[1,64,256]` |
| `sa_vae_fp16` | Oobleck VAE decoder | `latent[1,64,256] → audio[1,2,524288]` |

**Host loop** (`StableAudioRunner`): tokenize (T5, `t5_tokenizer/`) → conditioner → start from Gaussian
noise → 8-step rectified-flow euler `x = x + (t_next − t)·v` over the fixed schedule
`[1.0, .9944, .9845, .9579, .8909, .7455, .5125, .2739] → 0` → VAE decode → 44.1 kHz stereo wav.
No KV cache, no CFG (cfg_scale 1.0 — the model is ARC-distilled).

## Performance (M4 Max, GPU)

| metric | value |
|---|---|
| 8-step DiT | ~200 ms (25 ms/step) |
| VAE decode | ~185 ms |
| **total** | **~0.4 s for ~11.9 s of audio (~30× real-time)** |
| size | fp16, ~1.0 GB (DiT 651M + cond 210M + VAE 149M) |

Numerics: each bundle engine-gated vs the reference at cos ≥ 0.9999; full pipeline reproduces the
reference audio exactly.

## Roadmap

- iPhone (h18p) build — bundles AOT-compile; device RTF pending
- int8 (further size cut)
- a music-generation tab in the zoo app

## Credits & license

A community **Core AI conversion** — all credit to **Stability AI** (and Arm) for
[Stable Audio Open Small](https://huggingface.co/stabilityai/stable-audio-open-small); T5 text encoder
by Google. This bundle is governed by the **[Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE.md)**
(free for non-commercial use and for commercial use under \$1M annual revenue; review the license
before use). No retraining — conversion only.

Part of the [Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo).

---

**⬇️ Download:** [🤗 mlboydaisuke/Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI) — this card and the
model page are the same document; `scripts/gen-cards` keeps the *Use it* block in sync.
Reproduce it: `python3 conversion/zoo_convert.py show stable-audio-open-small` prints the command and its
prerequisites; [`recipe.toml`](recipe.toml) is the record.
