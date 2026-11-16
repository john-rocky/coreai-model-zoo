"""Stage + upload the North-Micro-Vision Core AI bundles to HF.

Publishes what was gated: the fp16 tower and the int8lin decoder (the pair that
IS the model), the text core (same weights, no image inputs — the only shape
`llm-benchmark`/`llm-runner` can drive), and `ios-h18p/` AOT variants of the
shipped pair, both device-gated on an iPhone 17 Pro.

Deliberately NOT published: `int4lin`. It is 0/9 on the suite with repetition and
instruction-boilerplate leaks — see the card.

    gpu-pipelined/<bundle>/<bundle>.aimodel/   + metadata.json + tokenizer/
    ios-h18p/<bundle>/<bundle>.h18p.aimodelc/  + metadata.json + tokenizer/
    config.json      -- the source config, so zoo_verify can compare against it
    README.md        -- the card

Build the iOS variants after the Mac export with:

    xcrun coreai-build compile exports/<name>/<name>.aimodel \
        --platform iOS --preferred-compute gpu --architecture h18p \
        --output exports/<name>_ios

No `--expect-frequent-reshapes` on iOS: it makes the runtime discard the AOT
specialization and compile on device, which SIGSEGVs with no log.

Run: `python _northmv_hf_upload.py [--dry-run]`.
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

REPO = "mlboydaisuke/North-Micro-Vision-CoreAI"
SOURCE = "CohereLabs/North-Micro-Vision-Instruct"
BUNDLES = [
    "north_micro_vision_instruct_vision_fp16",
    "north_micro_vision_instruct_decode_int8lin",
    "north_micro_vision_instruct_decode_int8lin_textcore",
]
IOS_BUNDLES = [
    "north_micro_vision_instruct_decode_int8lin",
    "north_micro_vision_instruct_vision_fp16",
]
CARD = Path(__file__).parents[1] / "models" / "north-micro-vision" / "HF_README.md"
STAGE = Path(os.environ.get("ZOO_STAGE", "/tmp")) / "northmv_hf"


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

    for name in IOS_BUNDLES:
        aotc = exports_dir() / f"{name}_ios" / f"{name}.h18p.aimodelc"
        if not aotc.is_dir():
            print(f"  SKIP ios-h18p/{name} (no {aotc.name})")
            continue
        dst = STAGE / "ios-h18p" / name
        dst.mkdir(parents=True)
        shutil.copytree(aotc, dst / aotc.name)
        if "vision" not in name:  # the tower is a bare .aimodel, no bundle metadata
            meta = json.loads((exports_dir() / name / "metadata.json").read_text())
            meta["assets"]["main"] = aotc.name
            (dst / "metadata.json").write_text(json.dumps(meta, indent=1))
            shutil.copytree(exports_dir() / name / "tokenizer", dst / "tokenizer")
        print(f"  staged ios-h18p/{name}")

    snap = Path(hf_snapshot(SOURCE))
    shutil.copy2(snap / "config.json", STAGE / "config.json")
    shutil.copy2(CARD, STAGE / "README.md")
    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({total / 1e9:.2f} GB)")
    return STAGE


def main() -> None:
    folder = stage()
    if "--dry-run" in sys.argv:
        print("dry run -- not uploading")
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=REPO, repo_type="model",
        commit_message="North-Micro-Vision -> Core AI: fp16 Qwen3-VL-shaped tower + int8lin "
                       "Cohere decoder; suite 9/9, iPhone 17 Pro 24/24")
    print("uploaded", REPO)


if __name__ == "__main__":
    main()
