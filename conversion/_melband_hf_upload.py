"""Stage + upload the Mel-Band RoFormer (Kim Vocal) Core AI bundles to HF.

Stages the macOS fp16 `.aimodel`, the iOS AOT `.h18p.aimodelc`, the shared metadata and the
self-test goldens into /tmp/melband_hf, then uploads to mlboydaisuke/MelBandRoformer-Vocal-CoreAI.

    coreai-models/.venv/bin/python conversion/_melband_hf_upload.py
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"          # xet stalls/panics on these bundles
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/MelBandRoformer-Vocal-CoreAI"
HERE = Path(__file__).resolve().parent
CONV = HERE / "melband_roformer"
MACOS = CONV / "ship_macos"                      # export_full2_ship.py
IOS = CONV / "ship_ios"                          # xcrun coreai-build compile ... --architecture h18p
STAGE = Path("/tmp/melband_hf")

CARD = """---
license: mit
library_name: coreai
pipeline_tag: audio-to-audio
base_model: KimberleyJSN/melbandroformer
tags: [core-ai, coreaikit, source-separation, stem-separation, vocals, karaoke, on-device, apple]
---

# Mel-Band RoFormer (Kim Vocal) — Core AI

[`KimberleyJSN/melbandroformer`](https://huggingface.co/KimberleyJSN/melbandroformer) (MIT, ~228 M)
converted to **Apple Core AI** — the [zoo](https://github.com/john-rocky/coreai-model-zoo)'s **first
source-separation model**. Split any song into a **vocals (acapella)** stem and an **instrumental
(karaoke)** stem, entirely on device: iPhone (AOT) and Mac.

Mel-Band RoFormer is `STFT -> band-split (mel, overlapping) -> axial rotary transformer x6 -> mask
estimator -> band-average -> complex mask multiply -> iSTFT`. The neural core lowers to Core AI
directly (no rewrite), and two moves keep the on-device host trivial:

- **Real-arithmetic core** — the band-average scatter becomes a constant matmul and the complex mask
  multiply becomes real ops, so the graph carries no complex tensors and no `scatter_add`.
- **STFT/iSTFT folded into the graph** as constant DFT matmuls (window baked in). The shipped graph is
  `frames[1,2,801,2048] -> recon[1,2,801,2048]`; the host only does reflect-pad, framing and
  overlap-add — **no FFT, no vDSP packing**.

Fixed shapes throughout (8 s chunk = 352 800 samples @ 44.1 kHz, 801 STFT frames), so there are no
dynamic-shape recompiles. The instrumental stem is `mix - vocals`.

## Contents

- `mbr_full_fp16.aimodel` — macOS bundle (fp16, ~470 MB).
- `mbr_full_fp16.h18p.aimodelc` — iOS AOT specialization (A19 / h18p, GPU).
- `metadata.json` — sample rate, chunk size, STFT parameters, graph I/O, host recipe.
- `golden_raw.f32` / `golden_vocals.f32` — an 8 s demo chunk and its expected vocals stem
  (stereo, channel-major, float32) for a host-side self-test.

## Host recipe

Reflect-pad the chunk by `n_fft // 2`, frame with `hop_length`, feed `frames[1,2,801,2048]`, then
overlap-add the output, divide by the summed squared window and trim the pad. Chunks are 8 s with
`num_overlap` crossfade; `instrumental = mix - vocals`.

## Gates

| gate | result |
|---|---|
| re-authored real-arithmetic core vs the reference model | cos **1.0000000** |
| in-graph STFT/iSTFT vs `torch.stft` reference | cos **0.9999984** |
| Core AI fp16, Mac GPU, framing + overlap-add round trip | cos **0.9999453** |
| **iPhone 17 Pro** (A19 Pro, AOT h18p, GPU) vs the Mac golden | cos **1.000000**, rms ratio **1.0000** |

On device (iPhone 17 Pro, GPU): **8 s chunk in 1.23 s ~ 6.5x real-time** warm (load 0.57 s); the
cold first run is 3.82 s ~ 2.1x real-time (load 1.24 s), before the GPU clocks ramp.

## Use it

```swift
import CoreAIKit

let separator = try await KitSeparator(catalog: "melband-roformer-vocal")
let stems = try await separator.separate(contentsOf: songURL)
// stems.vocals / stems.instrumental — [channel][sample] at 44.1 kHz
```

Ships in the zoo's **[coreai-audio](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/coreai-audio)**
app (**Separate** tab), which pairs it with the Music tab (Stable Audio Open Small): generate a
track, then rip its stems. Conversion recipe and the Swift host reference:
[`conversion/melband_roformer`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/melband_roformer).

## Attribution

Mel-Band RoFormer (Ju-Chiang Wang, Wei-Tsung Lu, Minz Won — ByteDance AI Labs); checkpoint by
KimberleyJensen; lucidrains BS-RoFormer implementation; ZFTurbo training code. MIT.

*Community port — not an Apple model.*
"""


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copytree(MACOS / "mbr_full_fp16.aimodel", STAGE / "mbr_full_fp16.aimodel")
    shutil.copytree(IOS / "mbr_full_fp16.h18p.aimodelc", STAGE / "mbr_full_fp16.h18p.aimodelc")
    for f in ["metadata.json", "golden_raw.f32", "golden_vocals.f32"]:
        shutil.copy2(MACOS / f, STAGE / f)
    (STAGE / "README.md").write_text(CARD)
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file()) / 1e6
    print(f"staged {total:.0f} MB in {STAGE}")

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE), commit_message="Mel-Band RoFormer (Kim Vocal) Core AI: macOS .aimodel + iOS h18p AOT + goldens")
    print(f"uploaded -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
