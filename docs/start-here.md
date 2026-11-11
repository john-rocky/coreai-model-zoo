# Where to start

- **You're a coding agent** (or you're pointing one here) → [**`AGENTS.md`**](../AGENTS.md) — the
  porting contract in one file: why conversion isn't conversion, the two gates, the traps agents
  hit, and what isn't an agent's call. No install needed. One fetch for everything here:
  [**`llms.txt`**](https://john-rocky.github.io/coreai-model-zoo/llms.txt).
- **Looking something up about the runtime itself** →
  [**john-rocky.github.io/coreai-model-zoo**](https://john-rocky.github.io/coreai-model-zoo/) —
  the knowledge base as a page per topic. Apple documents the API surface well; what these notes
  add is what the runtime *does* when you run it — the thresholds, the failure modes, the
  measured numbers — which is not written down anywhere. Plain HTML, so a reader without JS, a
  search crawler and a coding agent all get the same words.
- **Reading start to finish rather than looking one thing up** →
  [**The Art of Core AI**](https://john-rocky.github.io/the-art-of-core-ai/) — a free book built
  from these same measurements ([Japanese edition](https://zenn.dev/mlboydaisuke/books/coreai-textbook)).
- **Surveying the whole Core AI ecosystem**, not just this catalog →
  [**awesome-core-ai**](https://github.com/john-rocky/awesome-core-ai) — Apple's own tooling,
  other people's converters and runtimes, sample apps, benchmarks, and learning material.
- **Try the app** (iOS 27 / macOS 27; the model downloads in-app):
  - **Demo app, no build** → Mac: [**2.0.3 (build 12) .dmg**](https://github.com/john-rocky/coreai-model-zoo/releases/download/mac-2.0.3.12/CoreAIZoo-2.0.3-12.dmg) (signed and notarized; Apple silicon, macOS 27) · iPhone: [**CoreAI Zoo on the App Store**](https://apps.apple.com/us/app/coreai-zoo/id6780135339)
    Mac quick start: drag the app into Applications, open **Chat → Qwen3 0.6B → Download & Load**, wait for **Ready**, then send a message. [Release notes and tested environment](https://github.com/john-rocky/coreai-model-zoo/releases/tag/mac-2.0.3.12).
  - **Build it** → [`apps/`](../apps/) — Xcode 27 + xcodegen, the `coreai-models` patch stack + `tokenizer.json`
- **Use a model in your own app** → add [**CoreAIKit**](https://github.com/john-rocky/coreai-kit)
  (SPM) and load the catalog id; the model's card has the complete snippet + a 5-line
  integration checklist (golden example: [`models/whisper-large-v3-turbo/README.md`](../models/whisper-large-v3-turbo/README.md)).
  Engine-level deep-dive: [`knowledge/swift-runtime.md`](../knowledge/swift-runtime.md)
- **Port a model, end to end** → [**`PORTING.md`**](../PORTING.md) — the complete walk from HF
  checkpoint to a verified `.aimodel` on iPhone (oracle → export → gates → device → publish),
  with a vision and an LLM worked example. Start here to contribute a port.
- **Rebuild a published bundle** → [`models/<model>/recipe.toml`](../models/) via
  `python3 conversion/zoo_convert.py run <name>` — see [Rebuild a bundle](rebuild.md)
- **Convert a model** (export API + gotchas) → [`knowledge/conversion-guide.md`](../knowledge/conversion-guide.md)
- **Compress** → [`knowledge/compression.md`](../knowledge/compression.md)
- **Make it fast** → [`knowledge/custom-metal-kernels.md`](../knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](../knowledge/performance-ceiling.md)
- **Known issue, still present on 26A428** (in-graph KV-write crash; workarounds + the input-mask escape) → [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](../knowledge/coreai-beta-mpsgraph-kvwrite-bug.md) — FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)
- **Hit an error string** (`LLVM ERROR: …`, `failed assertion …`, `failedToSpecialize`, `NSPOSIXErrorDomain Code=2`, …) → [`knowledge/coreai-error-index.md`](../knowledge/coreai-error-index.md) — every exact string this project has observed, verbatim as a heading, with the verified cause, the fix, and the log or Apple issue behind it
- **Still stuck** → [conversion clinic](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=conversion-clinic.yml) — paste the error, get an answer

## Repository layout

| Dir | What |
|---|---|
| [**coreai-kit** ↗](https://github.com/john-rocky/coreai-kit) | (sibling repo) The Swift package that runs this zoo: 1-line `catalog:` APIs (`ChatSession`, `KitTranscriber`, …), model download + cache, and per-kind example apps in [`Examples/`](https://github.com/john-rocky/coreai-kit/tree/main/Examples) — the cards' ▶️ / 💻 doors point there. |
| [`models/`](../models/) | One directory per model — the card, the `recipe.toml` that reproduces its published bundles, and the generated [`_INVENTORY.md`](../models/_INVENTORY.md) / [`index.json`](../models/index.json). Laid out like [`apple/coreai-models`](https://github.com/apple/coreai-models). |
| [`skills/`](../skills/) | Agent skills — install them and your coding agent can pick a model, reproduce its bundle, and verify the result. |
| [`knowledge/`](../knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](../conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`cli/`](../cli/) | `coreai export` / `doctor` / `verify` — find a model's export route, lint an artifact against the failure patterns in [`knowledge/`](../knowledge/), gate it against an HF oracle. [`DOCTOR_RULES.md`](../cli/DOCTOR_RULES.md) is the table those checks come from: 64 ways a conversion succeeds and the result is quietly wrong. |
| [`swift/`](../swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](../apps/) | **Engine showcases** — apps for models that need a hand-tuned backend (custom Metal kernels, patch stack: BitCPM, RWKV-7, LLaDA, …) and the device-verification bench behind the published numbers. Want to *just run a model*? Use the [kit examples ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples) instead. |

## Contributing

Every model in the table is a conversion recipe anyone could have written — and some are, by
[people who are not me](../README.md#community-ports). [Nanbeige4.2-3B](../models/nanbeige4.2-3b/README.md)
([PR #6](https://github.com/john-rocky/coreai-model-zoo/pull/6), by
[@ukint-vs](https://github.com/ukint-vs)) is the reference for what a contributed port looks
like, and [pocket-tts](../models/pocket-tts/README.md)
([PR #12](https://github.com/john-rocky/coreai-model-zoo/pull/12), by
[@RahulRachuri](https://github.com/RahulRachuri)) is the reference for how far the gates can be
taken — an ASR round trip and a 302-sentence sweep, because tensor cosine can pass while the
audio is unintelligible. The shape of both: pinned checkpoint + immutable bundle revision, the
overlay/recipe/gates in the PR, the bundle published under the **contributor's own HF
namespace** (you keep ownership and credit), and the maintainer runs the iPhone hardware gate
for you if you don't have the device.

- **Port a model** — [`PORTING.md`](../PORTING.md) walks the whole path (oracle → export → gates →
  publish); PRs welcome.
- **Conversion requests** — a model you'd like to see here? [Open an issue](https://github.com/john-rocky/coreai-model-zoo/issues/new) with the Hugging Face link and what you'd use it for.
- **Stuck converting?** Open a [conversion clinic](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=conversion-clinic.yml) issue. The error text alone is enough to start; we ask for the rest. On the clinic day after the OS 27 release every ticket gets a same-day answer — a fix, or the reason it cannot be done yet. Search your error string first in [`knowledge/coreai-error-index.md`](../knowledge/coreai-error-index.md); if it is there, so is the fix.
- **No code needed** — run the Bench tab in [CoreAI Zoo (App Store)](https://apps.apple.com/us/app/coreai-zoo/id6780135339) and submit the result: your device becomes a row in [`BENCHMARKS.md`](../BENCHMARKS.md).

## Built with the zoo

Projects outside this repo that carry code ported from the zoo, read its model index, or follow
one of its conversion recipes:

- **[RF-DETR](https://github.com/roboflow/rf-detr)** — Roboflow's real-time object detection and
  segmentation model. Its `RFDETR.export(format="coreai")` (since 1.11.0) was ported from the
  zoo's [RF-DETR conversion](../models/rf-detr/README.md)
  ([pull request](https://github.com/roboflow/rf-detr/pull/1521)).
- **[Archon](https://github.com/VM451/archon-swift)** — a local-first Swift SDK for building AI
  features on Apple platforms. Its `CoreAIModelZooCatalog` fetches the zoo's
  [`models/index.json`](../models/index.json) and turns the recipes marked `verified` into entries
  in its model catalog
  ([source](https://github.com/VM451/archon-swift/blob/86af1356a9f2029635e6a676385da251c40163a6/Sources/ArchonModels/CoreAIModelZooCatalog.swift)).
- **[Slurper](https://github.com/TrevorS/slurper)** — a macOS CLI that transcribes YouTube or
  audio files and splits them into stems. Its Mel-Band RoFormer conversion script follows the
  zoo's [recipe](../models/melband-roformer/README.md): the STFT and inverse STFT become constant
  DFT matmuls inside the graph. The script targets Core ML (`coremltools`), not Core AI
  ([source](https://github.com/TrevorS/slurper/blob/eab5f8b064338f1bf9cd63fc6683852929a5146d/scripts/convert_melband_roformer.py)).

Built something? Open a
[showcase issue](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=showcase.yml)
— a name, a link, and one line is all it takes.
