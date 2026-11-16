"""Stage + upload the Streaming Sortformer (speaker diarization) Core AI bundle to HF.

Stages the macOS fp16 graph + the iOS h18p AOT graph + the librosa-slaney mel filterbank + metadata
+ a cc-by-4.0 model card into /tmp/sortformer_hf, then uploads to
mlboydaisuke/Streaming-Sortformer-Diar-CoreAI. Bundles come from conversion/sortformer_diar/ship_*.

    coreai-models/.venv/bin/python conversion/_sortformer_hf_upload.py
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/Streaming-Sortformer-Diar-CoreAI"
HERE = Path(__file__).resolve().parent
SD = HERE / "sortformer_diar"
MAC = SD / "ship_macos"
IOS = SD / "ship_ios"
STAGE = Path("/tmp/sortformer_hf")

CARD = """---
license: cc-by-4.0
library_name: coreai
pipeline_tag: voice-activity-detection
base_model: nvidia/diar_streaming_sortformer_4spk-v2
tags: [core-ai, coreaikit, sortformer, speaker-diarization, diarization, streaming, on-device, apple]
---

# Streaming Sortformer 4-spk v2 — Core AI

[`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
(cc-by-4.0, 117M) converted to **Apple Core AI** — streaming **speaker diarization** ("who spoke
when", up to 4 speakers) running fully on-device via the
[zoo](https://github.com/john-rocky/coreai-model-zoo). Only the neural core is a graph; the NeMo
128-mel frontend, the streaming chunk loop, and the AOSC speaker-cache compression run in the Swift
host — a 1:1 port of NeMo `sortformer_modules.py` (inference path).

⚠️ Use the **streaming v2** checkpoint (cc-by-4.0). The offline `diar_sortformer_4spk-v1` is CC-BY-**NC**.

## Files

- `sortformer_float16.aimodel` — the static `forward_for_export` core, fp16 (~237 MB, macOS GPU).
- `sortformer_float16.h18p.aimodelc` — the same graph AOT-compiled for iPhone (h18p, ~450 MB).
- `sortformer_mel_filters_128x257.f32` — librosa-slaney mel filterbank (host log-mel frontend).
- `metadata.json` — streaming params + the fixed-buffer graph contract.

## Fixed-buffer graph contract

```
inputs:  chunk_mel [1,1520,128]   host zero-pads each mel chunk
         spkcache  [1,188,512]     host-maintained speaker cache
         valid     [1,378]         1 = real frame / 0 = pad (spkcache block [0:len], chunk block [188:188+pe_len])
outputs: preds     [1,378,4]       sigmoid speaker activity
         chunk_pe  [1,190,512]     pre-encode embeddings (host appends them to the speaker cache)
```
Host: NeMo 128-mel (preemph 0.97 → STFT n_fft=512/win=400/hop=160 → slaney mel → log, normalize=NA)
→ chunk the mel (188·8 frames, ±1 subsample ctx) → run the graph → slice chunk preds →
`streaming_update` + `compress_spkcache` (AOSC) → threshold 0.5/frame/speaker (frame = 80 ms) → turns.

## Verification

Byte-gated vs NeMo `forward_streaming` at **100.00 % speaker-activity agreement** (@0.5) on a 21.5 s
and a 64.5 s clip (the latter exercises the AOSC cache compression ~4×), in Python, in Swift on
**Mac GPU**, and on **iPhone 17 Pro** (A19 Pro, AOT h18p) — all driving this exported fp16 graph.

## Use

Ships in the **coreai-audio** app (Transcribe tab, "Diarize — who said what"): the diarizer segments
each speaker turn, then the on-device ASR (Whisper / Qwen3-ASR / Parakeet / Nemotron) transcribes it
into a **diarized transcript** — `Speaker 1 [0.3–4.1s]: …`. Speaker diarization already ships
on-device elsewhere (e.g. CoreML/ANE); this is speed parity, offered as a diarized transcript wired to
the zoo's own ASR. Conversion + Swift host loop: see
[`conversion/sortformer_diar`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/sortformer_diar).

Derived from NVIDIA's `diar_streaming_sortformer_4spk-v2` (CC-BY-4.0); this Core AI conversion is
released under the same license.
"""


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copytree(MAC / "sortformer_float16.aimodel", STAGE / "sortformer_float16.aimodel")
    shutil.copytree(IOS / "sortformer_float16.h18p.aimodelc", STAGE / "sortformer_float16.h18p.aimodelc")
    shutil.copy2(MAC / "sortformer_mel_filters_128x257.f32", STAGE / "sortformer_mel_filters_128x257.f32")
    shutil.copy2(MAC / "metadata.json", STAGE / "metadata.json")
    (STAGE / "README.md").write_text(CARD)

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Streaming Sortformer 4-spk v2 -> Core AI: macOS + iOS(h18p) graphs + mel filterbank")
    print("uploaded", REPO)


if __name__ == "__main__":
    main()
