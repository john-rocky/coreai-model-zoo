"""Stage + upload the Nemotron 3.5 ASR Streaming bundle to the Hub.
Whisper-style platform subtrees so `ModelID.nemotronASRStreaming` (path nil)
resolves per platform:

  macos/  six JIT .aimodel graphs + tokenizer.json/tokenizer_config.json
  ios/    conformer_a/b AOT h18p .aimodelc + the four small JIT graphs + tokenizer

Also writes README.md (model card) + LICENSE (OpenMDW-1.1) at the repo root.

    coreai-models/.venv/bin/python _nemotron_hf_upload.py [--stage-only]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
REPO = "mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI"

JIT_GRAPHS = [
    "nemotron_asr_stream_pre_first_float16.aimodel",
    "nemotron_asr_stream_pre_float16.aimodel",
    "nemotron_asr_stream_conformer_a_float16.aimodel",
    "nemotron_asr_stream_conformer_b_float16.aimodel",
    "nemotron_asr_predict_float32.aimodel",
    "nemotron_asr_joint_float32.aimodel",
]
AOT_CONFORMERS = ["ios/nemotron_asr_stream_conformer_a_float16.h18p.aimodelc",
                  "ios/nemotron_asr_stream_conformer_b_float16.h18p.aimodelc"]
TOKENIZER = ["tokenizer.json", "tokenizer_config.json"]


def stage() -> Path:
    root = ART / "hf_stage"
    shutil.rmtree(root, ignore_errors=True)
    (root / "macos").mkdir(parents=True)
    (root / "ios").mkdir(parents=True)

    for g in JIT_GRAPHS:
        shutil.copytree(ART / g, root / "macos" / g)
    # iOS: AOT conformer halves + the small graphs as JIT (they specialize fine on-device).
    for aot in AOT_CONFORMERS:
        shutil.copytree(ART / aot, root / "ios" / Path(aot).name)
    for g in JIT_GRAPHS:
        if "conformer" in g:
            continue
        shutil.copytree(ART / g, root / "ios" / g)
    for t in TOKENIZER:
        for sub in ("macos", "ios"):
            shutil.copy(ART / "bundle_assets" / t, root / sub / t)

    for name, src in (("README.md", HERE / "_hf_README.md"),
                      ("LICENSE", HERE / "_hf_LICENSE")):
        if src.exists():
            shutil.copy(src, root / name)
        else:
            print(f"[warn] {src.name} missing — write it before upload")
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
    print(f"[stage] {root} ({total:.2f} GB)")
    return root


def upload(root: Path) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_large_folder(repo_id=REPO, folder_path=str(root), repo_type="model")
    print(f"[upload] https://huggingface.co/{REPO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()
    root = stage()
    if not args.stage_only:
        upload(root)
