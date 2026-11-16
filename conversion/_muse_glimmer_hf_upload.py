"""Stage + upload the Muse-Glimmer-30B Core AI text-decoder bundle to HF.

Publishes the dynamic-ids `int4hu --head-sym` bundle — the ship shape for a Mac-only
model. The `--static-ids` variant is deliberately not published: it buys 1.5% decode
and costs 9.3x prefill, which only pays off on an iPhone-class deployment this model
cannot reach anyway.

Layout matches the rest of the catalog (`conversion/_hf_catalog.py` reads it):

    gpu-pipelined/<bundle>/<bundle>.aimodel/  + metadata.json + tokenizer/
    config.json      -- the source config, so zoo_verify can compare against it
    LICENSE          -- carried from the source repo (plain Apache-2.0)
    README.md        -- the card
    gate-*.json      -- the token-exact gate transcript the card cites
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

REPO = "mlboydaisuke/Muse-Glimmer-30B-CoreAI"
SOURCE = "meta-models/Muse-Glimmer-30B"
BUNDLES = ["muse_glimmer_30b_decode_int4hu_block32_sym"]
MODEL_DIR = Path(__file__).parents[1] / "models" / "muse-glimmer-30b"
CARD = MODEL_DIR / "HF_README.md"
GATES = ["gate-muse-glimmer-30b-int4hu.json"]
STAGE = Path(os.environ.get("ZOO_STAGE", "/tmp")) / "muse_glimmer_hf"


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
        # The source repo carries plain unmodified Apache-2.0 but the allow-patterns
        # used to fetch the weights skip extensionless files; pull it on demand
        # rather than shipping a bundle with no licence text.
        from huggingface_hub import hf_hub_download

        shutil.copy2(hf_hub_download(SOURCE, "LICENSE"), STAGE / "LICENSE")

    shutil.copy2(CARD, STAGE / "README.md")
    for gate in GATES:
        path = MODEL_DIR / gate
        if not path.exists():
            raise SystemExit(f"missing gate transcript {path} -- run coreai_gate.py --transcript")
        shutil.copy2(path, STAGE / gate)

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
        folder_path=str(folder),
        repo_id=REPO,
        repo_type="model",
        commit_message="Muse-Glimmer-30B text decoder -> Core AI: decode int4hu "
                       "(--head-sym), token-exact vs fp16 oracle, 26.69 tok/s on M4 Max",
    )
    print("uploaded", REPO)
    print(json.dumps({"repo": REPO, "bundles": BUNDLES}, indent=1))


if __name__ == "__main__":
    main()
