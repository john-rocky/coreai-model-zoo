"""Upload the Nemotron-3-Nano-4B Core AI bundles to HF.

Stages two variants of the int8hu ship config, because a 4B graph cannot specialize
on-device and the AOT bundle is iOS-only:

  gpu-pipelined/nemotron_3_nano_4b_decode_int8hu/   .aimodel        (Mac, JIT)
  ios-h18p/nemotron_3_nano_4b_decode_int8hu/        .h18p.aimodelc  (iPhone, AOT)

Each carries its own metadata.json (assets.main points at that variant's asset —
CoreAIShared.ModelBundle reads metadata.json at the model dir) plus the tokenizer.

Build them first:
  python export_nemotron_h_decode_pipelined.py int8hu --head-sym
  xcrun coreai-build compile exports/<name>/<name>.aimodel \
      --platform iOS --preferred-compute gpu --architecture h18p --output exports/<name>
"""
import json
import os
import shutil
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"   # Xet stalls silently; plain HTTP is faster
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/Nemotron-3-Nano-4B-CoreAI"
HF_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NAME = "nemotron_3_nano_4b_decode_int8hu"
SRC = Path.home() / "code/coreai/coreai-models/exports" / NAME
STAGE = Path("/tmp/nemotron_h_hf")

VARIANTS = {
    "gpu-pipelined": f"{NAME}.aimodel",
    "ios-h18p": f"{NAME}.h18p.aimodelc",
}

README = f"""---
license: other
license_name: nvidia-open-model-license
base_model: {HF_ID}
tags: [core-ai, coreml, mamba2, ssm, on-device, apple-silicon]
---

# Nemotron-3-Nano-4B — Core AI

Decode-only (S=1) Core AI bundles for NVIDIA's Mamba2 + attention + MLP hybrid
(42 blocks: 21 Mamba2 / 17 MLP / 4 GQA NoPE attention), int8 weights with an
absmax int8 head. **No custom Metal kernel** — at S=1 the selective scan is a
single recurrence step, so the graph is loop-free.

| variant | asset | for |
|---|---|---|
| `gpu-pipelined/` | `{NAME}.aimodel` | Mac (JIT specialization) |
| `ios-h18p/` | `{NAME}.h18p.aimodelc` | iPhone (AOT — a 4B graph cannot specialize on-device) |

<!-- gen-cards:use-it begin id=nemotron-3-nano-4b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
<!-- gen-cards:use-it end -->

Measured: **16.0 tok/s decode on an iPhone 17 Pro** (cooled, AOT h18p, bandwidth-saturated)
and 85.2 tok/s on an M4 Max GPU. Greedy output is token-identical to the fp32
`transformers` rollout on the probe prompts.

Requires `COREAI_CHUNK_THRESHOLD=1` (S=1 bundle) and an engine that carries two extra
fixed-shape states (the Mamba conv columns + SSM state) alongside the KV cache.

Port + recipe: [coreai-model-zoo / nemotron-3-nano](https://github.com/john-rocky/coreai-model-zoo/blob/main/models/nemotron-3-nano/README.md)
"""


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    meta = json.loads((SRC / "metadata.json").read_text())
    for variant, asset in VARIANTS.items():
        src = SRC / asset
        if not src.exists():
            raise FileNotFoundError(f"{src} — build it first (see this file's docstring)")
        dst = STAGE / variant / NAME
        dst.mkdir(parents=True)
        shutil.copytree(src, dst / asset)
        shutil.copytree(SRC / "tokenizer", dst / "tokenizer")
        m = dict(meta)
        m["assets"] = {"main": asset}
        m["compilation"] = {**meta["compilation"],
                            "targets": ["h18p"] if variant == "ios-h18p" else []}
        (dst / "metadata.json").write_text(json.dumps(m, indent=2))
        print(f"staged {variant}/{NAME} ({asset})")

    for f in ("LICENSE", "LICENSE.md", "NOTICE"):
        try:
            shutil.copy(hf_hub_download(HF_ID, f), STAGE / f)
            print(f"included upstream {f}")
            break
        except Exception:  # noqa: BLE001 — the upstream repo may name it differently
            continue
    (STAGE / "README.md").write_text(README)


if __name__ == "__main__":
    stage()
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Nemotron-3-Nano-4B -> Core AI: int8hu decode bundles (Mac JIT + iOS AOT h18p)")
    print("uploaded", REPO)
