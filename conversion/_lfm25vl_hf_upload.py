"""Stage + upload the LFM2.5-VL Core AI bundles to HF.

Run: `python _lfm25vl_hf_upload.py [450m|3b] [--ios-only] [--dry-run]`.

Publishes three bundles: the fp16 vision tower, the int8lin VLM decoder (the pair
that IS the model), and the text core -- the same decoder weights with no image
input, which is both a usable 350M LFM2 text bundle and the only configuration
`llm-benchmark` / `llm-runner` can drive, since neither can bind an image buffer.

Deliberately NOT published: `int4lin` (0/9 gate cases, fluent drift -- see the card)
and the int8 vision tower (smaller but slower AND less accurate on Mac; it may still
be the right call on a phone, which is unmeasured).

Layout matches the rest of the catalog (`conversion/_hf_catalog.py` reads it):

    gpu-pipelined/<bundle>/<bundle>.aimodel/         + metadata.json + tokenizer/
    ios-h18p/<bundle>/<bundle>.h18p.aimodelc/        + metadata.json + tokenizer/
    config.json      -- the source config, so zoo_verify can compare against it
    LICENSE          -- carried from the source repo
    README.md        -- the card

The iOS subtree carries its own `metadata.json` (assets.main points at the
`.aimodelc`, which `AIModel(contentsOf:)` auto-detects as AOT and does not
re-JIT). Build it after the Mac export with:

    xcrun coreai-build compile exports/<name>/<name>.aimodel \
        --platform iOS --preferred-compute gpu --architecture h18p \
        --output exports/<name>_ios

No `--expect-frequent-reshapes` on iOS: it makes the runtime discard the AOT
specialization and compile on device, which SIGSEGVs with no log.
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

SIZES = {
    "450m": {
        "repo": "mlboydaisuke/LFM2.5-VL-450M-CoreAI",
        "source": "LiquidAI/LFM2.5-VL-450M",
        "card": "HF_README.md",
        # The pair that IS the model, plus the text core (a usable 350M LFM2 text bundle and
        # the only configuration llm-benchmark/llm-runner can drive).
        "bundles": [
            "lfm2_5_vl_450m_vision_fp16",
            "lfm2_5_vl_450m_decode_int8lin",
            "lfm2_5_vl_450m_decode_int8lin_textcore",
        ],
        # Device-gated on an iPhone 17 Pro.
        "ios": [
            "lfm2_5_vl_450m_decode_int8lin",
            "lfm2_5_vl_450m_decode_int8lin_textcore",
            "lfm2_5_vl_450m_vision_fp16",
        ],
        "message": ("LFM2.5-VL-450M -> Core AI: fp16 SigLIP2-NaFlex tower + int8lin "
                    "decoder (658 MB pair) + text core; oracle gate 16/16"),
    },
    "3b": {
        "repo": "mlboydaisuke/LFM2.5-VL-3B-CoreAI",
        "source": "LiquidAI/LFM2.5-VL-3B",
        "card": "HF_README_3B.md",
        # int8lin is the Mac ship; int4lin is published beside it because it costs this
        # model nothing on the suite (7/9, same as fp16) and is the only variant whose AOT
        # fits under the iOS 2 GiB load wall.
        "bundles": [
            "lfm2_5_vl_3b_vision_fp16",
            "lfm2_5_vl_3b_decode_int8lin",
            "lfm2_5_vl_3b_decode_int4lin",
            "lfm2_5_vl_3b_decode_int8lin_textcore",
        ],
        # int4lin + the tower: device-gated on an iPhone 17 Pro (the int8 AOT is 3.13 GiB
        # and does not load; int4's 2.03 GiB does).
        "ios": ["lfm2_5_vl_3b_decode_int4lin", "lfm2_5_vl_3b_vision_fp16"],
        "message": ("LFM2.5-VL-3B -> Core AI: fp16 SigLIP2-NaFlex tower + int8lin/int4lin "
                    "decoders + text core; fp32 ladder cos 1.000000, suite 7/9 = fp16 baseline"),
    },
}

def _bundle_metadata(cfg: dict, name: str, asset: str) -> dict | None:
    """metadata.json for an iOS variant, from the local export or the published repo.

    The local Mac export is the normal source, but it is also the first thing a
    session deletes once the bundles are on HF — and an iOS bundle without
    metadata.json fails at load with `BundleError` rather than anything that
    names the cause, so fall back to the repo instead of shipping it missing.
    """
    local = exports_dir() / name / "metadata.json"
    if local.exists():
        meta = json.loads(local.read_text())
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(cfg["repo"], f"gpu-pipelined/{name}/metadata.json")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING no metadata for {name}: {exc}")
            return None
        meta = json.loads(Path(path).read_text())
    meta["assets"]["main"] = asset
    return meta


def _tokenizer_dir(cfg: dict, name: str, dst: Path) -> None:
    local = exports_dir() / name / "tokenizer"
    if local.is_dir():
        shutil.copytree(local, dst)
        return
    from huggingface_hub import hf_hub_download

    dst.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "chat_template.jinja"):
        try:
            path = hf_hub_download(cfg["repo"], f"gpu-pipelined/{name}/tokenizer/{f}")
        except Exception:  # noqa: BLE001, S112
            continue
        (dst / f).write_bytes(Path(path).read_bytes())


def stage(cfg: dict, stage_dir: Path, ios_only: bool = False) -> Path:
    STAGE = stage_dir
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "gpu-pipelined").mkdir(parents=True)

    for name in [] if ios_only else cfg["bundles"]:
        src = exports_dir() / name
        if not src.is_dir():
            raise SystemExit(f"missing bundle {src} -- export it first")
        shutil.copytree(src, STAGE / "gpu-pipelined" / name)
        print(f"  staged {name}")

    # iOS AOT variants, for the bundles that were device-gated. Same directory
    # shape, metadata pointed at the .aimodelc.
    for name in cfg["ios"]:
        aotc = exports_dir() / f"{name}_ios" / f"{name}.h18p.aimodelc"
        if not aotc.is_dir():
            print(f"  SKIP ios-h18p/{name} (no {aotc.name}; run coreai-build compile)")
            continue
        dst = STAGE / "ios-h18p" / name
        dst.mkdir(parents=True)
        shutil.copytree(aotc, dst / aotc.name)
        # The vision tower is a bare .aimodel with no bundle metadata; decoders need it.
        if "vision" not in name:
            meta = _bundle_metadata(cfg, name, aotc.name)
            if meta is not None:
                (dst / "metadata.json").write_text(json.dumps(meta, indent=1))
            _tokenizer_dir(cfg, name, dst / "tokenizer")
        print(f"  staged ios-h18p/{name}")

    snap = Path(hf_snapshot(cfg["source"]))
    shutil.copy2(snap / "config.json", STAGE / "config.json")
    for license_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if (snap / license_name).exists():
            shutil.copy2(snap / license_name, STAGE / "LICENSE")
            break
    else:
        print("  WARNING: no LICENSE in the source snapshot")
    shutil.copy2(Path(__file__).parents[1] / "models" / "lfm2.5-vl" / cfg["card"],
                 STAGE / "README.md")

    total = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"staged {STAGE} ({total / 1e9:.2f} GB)")
    return STAGE


def main() -> None:
    size = next((a for a in sys.argv[1:] if a in SIZES), "450m")
    ios_only = "--ios-only" in sys.argv
    cfg = SIZES[size]
    folder = stage(
        cfg, Path(os.environ.get("ZOO_STAGE", "/tmp")) / f"lfm25vl_hf_{size}", ios_only
    )
    if "--dry-run" in sys.argv:
        print("dry run -- not uploading")
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(folder)}  {p.stat().st_size / 1e6:.1f} MB")
        return

    api = HfApi()
    api.create_repo(cfg["repo"], repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=cfg["repo"], repo_type="model",
        commit_message=cfg["message"])
    print("uploaded", cfg["repo"])
    print(json.dumps({"repo": cfg["repo"], "bundles": cfg["bundles"],
                      "ios": cfg["ios"]}, indent=1))


if __name__ == "__main__":
    main()
