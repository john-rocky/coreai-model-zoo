"""Stage + upload the LFM2.5-2.6B Core AI bundles (int8hu + int4lin) to HF.

Publishes the two bundles that are worth running: `int8hu --head-sym` (the quality
ship) and `int4lin` (2.0 GB, no quality cliff observed). `int8lin` is deliberately
not published -- it is slower than int8hu and only 0.2 GB smaller, so it has no
"use this" case of its own.

Layout matches the rest of the catalog (`conversion/_hf_catalog.py` reads it):

    gpu-pipelined/<bundle>/<bundle>.aimodel/  + metadata.json + tokenizer/
    config.json      -- the source config, so zoo_verify can compare against it
    LICENSE          -- carried from the source repo
    README.md        -- the card
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402

REPO = "mlboydaisuke/LFM2.5-2.6B-CoreAI"
SOURCE = "LiquidAI/LFM2.5-2.6B"
BUNDLES = ["lfm2_5_2_6b_decode_int8hu_block32_sym", "lfm2_5_2_6b_decode_int4lin"]
CARD = Path(__file__).parents[1] / "models" / "lfm2.5-2.6b" / "HF_README.md"
STAGE = Path(os.environ.get("ZOO_STAGE", "/tmp")) / "lfm25_26b_hf"


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "gpu-pipelined").mkdir(parents=True)

    for name in BUNDLES:
        src = exports_dir() / name
        if not src.is_dir():
            raise SystemExit(f"missing bundle {src} -- export it first")
        shutil.copytree(src, STAGE / "gpu-pipelined" / name)
        print(f"  staged {name}")

    snap = Path(hf_snapshot(SOURCE))
    shutil.copy2(snap / "config.json", STAGE / "config.json")
    for license_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if (snap / license_name).exists():
            shutil.copy2(snap / license_name, STAGE / "LICENSE")
            break
    else:
        print("  WARNING: no LICENSE in the source snapshot")
    shutil.copy2(CARD, STAGE / "README.md")

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({total / 1e9:.2f} GB)")
    return STAGE


def main() -> None:
    folder = stage()
    if "--dry-run" in sys.argv:
        print("dry run -- not uploading")
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(folder)}  {p.stat().st_size / 1e6:.1f} MB")
        return

    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=REPO, repo_type="model",
        commit_message="LFM2.5-2.6B -> Core AI: decode int8hu (--head-sym) + int4lin, "
                       "oracle gate 16/16 both")
    print("uploaded", REPO)
    print(json.dumps({"repo": REPO, "bundles": BUNDLES}, indent=1))


if __name__ == "__main__":
    main()
