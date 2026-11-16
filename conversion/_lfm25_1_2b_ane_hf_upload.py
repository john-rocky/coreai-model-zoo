"""Add the LFM2.5-1.2B Neural Engine bundles to HF (S6, 2026-09-17).

mlboydaisuke/LFM2.5-1.2B-CoreAI:
  ios-ane-h18p/   metadata.json + lfm2_5_1_2b_instruct_lfm25_pal8_g32_static.h18p.aimodelc + tokenizer/   (AOT h18p, ANE)
  ios-static/     metadata.json + lfm2_5_1_2b_instruct_lfm25_pal8_g32_static.aimodel      + tokenizer/   (the IR before AOT)
plus the "Neural Engine bundle" section of the card (README.md) and the upstream LICENSE (LFM Open License v1.0)
which the repo already carries. Staging = coreai-models/exports/hf_stage/LFM2.5-1.2B-CoreAI/ (built by
conversion/export_lfm25_ane_static.py --stage LFM2.5-1.2B-CoreAI, gated on the phone: apps/AneGate).

    HF_HUB_DISABLE_XET=1 coreai-models/.venv/bin/python conversion/_lfm25_1_2b_ane_hf_upload.py [--stage-only] [--card-only]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/LFM2.5-1.2B-CoreAI"
STAGE = Path.home() / "code/coreai/coreai-models/exports/hf_stage/LFM2.5-1.2B-CoreAI"
SUBTREES = ["ios-ane-h18p", "ios-static"]
MARK = "## Neural Engine bundle"
CARD_SECTION = Path(__file__).resolve().parent.parent / "models/lfm2.5/_hf_card_ane_section.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true", help="print what would be uploaded and exit")
    ap.add_argument("--card-only", action="store_true", help="only rewrite the card section")
    a = ap.parse_args()
    for sub in SUBTREES:
        d = STAGE / sub
        if not (d / "metadata.json").exists():
            sys.exit(f"missing staged subtree {d} (run conversion/export_lfm25_ane_static.py --skip-export --stage LFM2.5-1.2B-CoreAI)")
        meta = json.loads((d / "metadata.json").read_text())
        size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9
        print(f"{sub}: main={meta['assets']['main']} {size:.2f} GB")
    section = CARD_SECTION.read_text()
    if a.stage_only:
        print(section[:400]); return
    api = HfApi()
    if not a.card_only:
        for sub in SUBTREES:
            print(f"uploading {sub}/ ...", flush=True)
            api.upload_folder(repo_id=REPO, folder_path=str(STAGE / sub), path_in_repo=sub,
                              commit_message=f"{sub}: Apple static iOS export for the Neural Engine (8-bit k-means g32, "
                                             "LFM2 conv history in the KV rows; gated on an iPhone 17 Pro)")
    readme = Path(hf_hub_download(REPO, "README.md"))
    text = readme.read_text()
    if MARK in text:
        text = re.sub(rf"{re.escape(MARK)}.*?(?=\n## |\Z)", section.rstrip() + "\n", text, count=1, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
    out = Path("/tmp/_lfm25_1_2b_README.md"); out.write_text(text)
    api.upload_file(path_or_fileobj=str(out), path_in_repo="README.md", repo_id=REPO,
                    commit_message="card: Neural Engine bundle section (ios-ane-h18p/, ios-static/)")
    print("done")


if __name__ == "__main__":
    main()
