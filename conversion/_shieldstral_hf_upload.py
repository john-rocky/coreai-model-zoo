"""Stage + upload the Shieldstral Core AI classifier bundles to HF.

Publishes what was gated: the int4lin classifier at two grids (S=512 general, S=256
for short messages — same 2.53 GB of weights, 232.5 ms vs 123.6 ms per verdict).

Both grids also ship as `ios-h18p/` AOT builds, device-gated on an iPhone 17 Pro:
9/9 verdicts vs fp32 at both, 624.7 ms (S=512) and 371.9 ms (S=256) per verdict,
resources.bin 2.336 GiB either way.

Deliberately NOT published: fp16 (6.88 GB) and int8lin (4.04 GB). Both are 9/9 like
int4lin and neither is faster — at this graph shape quantization buys size, not
speed — so there is nothing for a 4-GB-larger download to be better at. Both are one
`export_shieldstral.py` run away if you want them.

    gpu-classify/<bundle>/<bundle>.aimodel/  + reference.json + tokenizer/
    ios-h18p/<bundle>/<bundle>.h18p.aimodelc/
    config.json   -- the source config, so zoo_verify can compare against it
    README.md     -- the card

Build the iOS variant after the Mac export with:

    xcrun coreai-build compile exports/<name>/<name>.aimodel \
        --platform iOS --preferred-compute gpu --architecture h18p \
        --output exports/<name>_ios/<name>.h18p.aimodelc

No `--expect-frequent-reshapes` on iOS: it makes the runtime discard the AOT
specialization and compile on device, which SIGSEGVs with no log.

Run: `python _shieldstral_hf_upload.py [--dry-run]`.
"""
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402

REPO = "mlboydaisuke/Shieldstral-CoreAI"
SOURCE = "mistralai/Shieldstral-1.0-3B"
BUNDLES = [
    "shieldstral_1_0_3b_classify_int4lin_s512",
    "shieldstral_1_0_3b_classify_int4lin_s256",
]
IOS_BUNDLES = [
    "shieldstral_1_0_3b_classify_int4lin_s512",
    "shieldstral_1_0_3b_classify_int4lin_s256",
]
CARD = Path(__file__).parents[1] / "models" / "shieldstral" / "HF_README.md"
STAGE = Path(os.environ.get("ZOO_STAGE", "/tmp")) / "shieldstral_hf"


STORE = (Path.home() / "Library/Application Support/CoreAIKit/Models"
         / REPO.split("/")[0] / REPO.split("/")[1])


def bundle_dir(name: str) -> Path:
    """Where a bundle actually is: fresh out of the exporter, or seeded into the kit's
    ModelStore. Published bundles get `mv`d there rather than deleted, so re-running an
    upload after a cleanup should not require re-exporting 2.5 GB."""
    direct = exports_dir() / name
    if direct.is_dir():
        return direct
    hits = sorted(STORE.glob(f"*/gpu-classify/{name}"))
    if hits:
        return hits[-1]
    raise SystemExit(f"missing bundle {name} -- export it first (looked in {exports_dir()} "
                     f"and {STORE})")


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "gpu-classify").mkdir(parents=True)

    for name in BUNDLES:
        shutil.copytree(bundle_dir(name), STAGE / "gpu-classify" / name)
        print(f"  staged {name}")

    for name in IOS_BUNDLES:
        aotc = exports_dir() / f"{name}_ios" / f"{name}.h18p.aimodelc"
        if not aotc.is_dir():
            print(f"  SKIP ios-h18p/{name} (no {aotc.name})")
            continue
        dst = STAGE / "ios-h18p" / name
        dst.mkdir(parents=True)
        shutil.copytree(aotc, dst / aotc.name)
        # The classifier has no bundle metadata.json: the host contract is
        # reference.json, and it is identical for both grids.
        shutil.copy2(bundle_dir(name) / "reference.json", dst / "reference.json")
        shutil.copytree(bundle_dir(name) / "tokenizer", dst / "tokenizer")
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
        commit_message="Shieldstral-1.0-3B -> Core AI: policy-conditioned safety classifier, "
                       "one forward = one verdict; 9/9 vs fp32 at int4lin")
    print("uploaded", REPO)


if __name__ == "__main__":
    main()
