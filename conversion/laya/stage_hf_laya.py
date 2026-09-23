#!/usr/bin/env python3
"""Stage the Hugging Face repo mlboydaisuke/Laya-Multilingual-CoreAI in a local folder. Uploads nothing.

    python3 conversion/laya/stage_hf_laya.py [--overwrite]

Builds <exports>/laya-multilingual/hf_stage/ from the gated variant folders and the pinned source:

  README.md             the zoo card adapted for the Hub: front matter, the one-line Core AI introduction, the
                        DeviceMark block (managed by scripts/gen-cards), repo-relative links pointed at GitHub
  LICENSE               the Apache-2.0 text (sha256-pinned; the upstream revision ships no LICENSE file)
  LICENSE-NOTE.md       what this repository changes relative to the upstream checkpoint
  UPSTREAM_README.md    the publisher's README.md at the pinned revision, unmodified
  config.json           the publisher's multilingual/encoder/config.json, unmodified
  rl_agent_config.json  the publisher's multilingual/rl_agent_config.json, unmodified
  coreai-kit.json       kind "decision", format "encoder": variants, host recipe, graph contract
  macos/ ios/ ios-h18p/ the gated variant folders, hard-linked (no symlinks); the fp16 recipe is not staged
  SHA256SUMS            every staged file
  STAGING.md            what is staged, sizes, hashes and the upload command (not run)
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODEL_ID, MODEL_SHA, SUBFOLDER, export_root, sha256_of, verify_source  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

HF_REPO = "mlboydaisuke/Laya-Multilingual-CoreAI"
GITHUB = "https://github.com/john-rocky/coreai-model-zoo/blob/main"
APACHE_2_0_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
VARIANTS = ("macos/wfp16-s256", "macos/wfp16-s512", "macos/fp32-s256", "macos/fp32-s512",
            "ios/wfp16-s256", "ios/wfp16-s512", "ios-h18p/wfp16-s256", "ios-h18p/wfp16-s512")
FRONT_MATTER = """---
library_name: coreai
license: apache-2.0
base_model: convaiinnovations/laya
tags:
  - coreai
  - apple-silicon
  - on-device
  - modernbert
  - laya
  - calibrated-decisions
  - text-classification
language:
  - multilingual
  - en
  - ja
pipeline_tag: text-classification
---
"""
INTRODUCTION = ("Core AI is Apple's on-device ML runtime in iOS 27 / macOS 27 and the successor to Core ML: PyTorch "
                "models are exported with Apple's `coreai-torch` (LLMs: `coreai.llm.export`) into `.aimodel` bundles "
                "that run on the GPU or the Neural Engine.")
DEVICEMARK = ("<!-- gen-cards:devicemark begin (managed by scripts/gen-cards + tools/devicemark_row.py — edit cards.json, "
              "not this block) -->\nThis model has no row on [DeviceMark](https://devicemark.github.io/), the on-device "
              "LLM leaderboard.\n<!-- gen-cards:devicemark end -->")


def apache_text(path: Path | None) -> Path:
    candidates = [path] if path else sorted(Path(sysconfig.get_paths()["purelib"]).glob("*.dist-info/**/LICENSE*"))
    for candidate in candidates:
        if candidate and candidate.is_file() and sha256_of(candidate) == APACHE_2_0_SHA256:
            return candidate
    raise SystemExit("no Apache-2.0 LICENSE text with the pinned sha256 found; pass --license-text")


def check_variant(folder: Path) -> dict:
    manifest_path = folder / "provenance" / ("aot-manifest.json" if folder.parent.name == "ios-h18p" else "export-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if folder.parent.name == "ios-h18p":
        assert manifest["status"] == "COMPILED" and manifest["architecture"] == "h18p", folder
    else:
        assert manifest["status"] == "PASS" and not manifest.get("measure_only"), folder
    if folder.parent.name == "macos":
        runtime = json.loads((folder / "provenance" / "runtime-gate.json").read_text())
        assert runtime["cpu_only"]["status"] == "PASS" and runtime["gpu"]["status"] == "PASS", folder
    assert folder.name.split("-")[0] != "fp16", "the fp16 recipe is not staged"
    return manifest


def hub_readme(card: str) -> str:
    body = card.split("\n", 1)[1].lstrip("\n")  # drop the zoo card's title line
    body = body.replace("](../../conversion/laya/README.md)", f"]({GITHUB}/conversion/laya/README.md)")
    body = body.replace("`gate-laya-multilingual.json` beside this card is the transcript",
                        f"[`gate-laya-multilingual.json`]({GITHUB}/models/laya-multilingual/gate-laya-multilingual.json) "
                        "in the zoo repository is the transcript")
    body = body.replace("`measurements-coreai-kit.json` beside this card",
                        f"[`measurements-coreai-kit.json`]({GITHUB}/models/laya-multilingual/measurements-coreai-kit.json)")
    body = body.replace("`recipe.toml` here names the commands",
                        f"[`recipe.toml`]({GITHUB}/models/laya-multilingual/recipe.toml) names the commands")
    body = body.replace("Hugging Face repo **mlboydaisuke/Laya-Multilingual-CoreAI** — upload pending. One folder per variant, each",
                        "This repository holds one folder per variant, each")
    leftovers = re.findall(r"\]\((?!https?://)[^)]*\)", body)
    assert not leftovers, f"relative links left in the Hub README: {leftovers}"
    return (FRONT_MATTER + "\n" + INTRODUCTION + "\n\n" + DEVICEMARK + "\n\n# laya multilingual — Core AI export\n\n"
            f"Zoo card, recipe and gate transcript: [coreai-model-zoo/models/laya-multilingual]({GITHUB}/models/laya-multilingual/README.md).\n\n"
            + body)


def license_note() -> str:
    return f"""# License and changes

Convai Innovations declares Apache-2.0 for `{MODEL_ID}` in the upstream model card at the pinned revision
`{MODEL_SHA}`; no standalone LICENSE or NOTICE file is listed at that revision. The unmodified upstream card
(`UPSTREAM_README.md`) and the standard Apache-2.0 text (`LICENSE`) are included.

Changes relative to the upstream `{SUBFOLDER}/` checkpoint:
- re-authored as a static Core AI graph from the raw weights (no transformers or laya code in the graph),
  windows 256 and 512, batch 1;
- one bundle with two functions: `main` (encoder, type embedding, the two head layers and the scorer at every
  position, plus the first position's hidden state) and `act` (the act head, fp32);
- `wfp16` folders store the weights in fp16 (exact: the checkpoint is stored in F16) and compute in fp32;
  `fp32` folders are the reference; `ios-h18p/` holds an iPhone 17 Pro (h18p) GPU compile of the wfp16 bundle;
- `metadata.json` carries the checkpoint's temperatures (T = 1) and, beside them, the fitted calibration
  published with litert-community/Laya-Multilingual-LiteRT (`laya_ml_calibration.json`);
- `tokenizer/` holds the checkpoint's `tokenizer.json` and `tokenizer_config.json`, unmodified;
  `config.json` and `rl_agent_config.json` are the checkpoint's own files, unmodified.
Source: `{MODEL_ID}@{MODEL_SHA}`, subfolder `{SUBFOLDER}`.
"""


def kit_json(out: Path, manifests: dict) -> dict:
    def size_mb(path: str) -> int:
        return round(sum(p.stat().st_size for p in (out / path).rglob("*") if p.is_file()) / 1e6)

    def variant(path: str, **extra) -> dict:
        fmt = "aimodelc" if path.startswith("ios-h18p/") else "aimodel"
        entry = {"path": path, "sizeMB": size_mb(path), "format": fmt}
        if path.startswith("ios"):
            entry["minOS"] = "27.0"
        if fmt == "aimodelc":
            entry["architecture"] = "h18p"
        return entry | extra

    metadata = json.loads((out / "macos/wfp16-s256/metadata.json").read_text())["decision"]
    return {
        "kind": "decision", "format": "encoder", "model": MODEL_ID, "subfolder": SUBFOLDER, "revision": MODEL_SHA,
        "variants": {
            "macos": variant("macos/wfp16-s256"), "macos-s512": variant("macos/wfp16-s512"),
            "ios": variant("ios/wfp16-s256"), "ios-s512": variant("ios/wfp16-s512"),
            "ios-h18p": variant("ios-h18p/wfp16-s256"), "ios-h18p-s512": variant("ios-h18p/wfp16-s512"),
            "macos-fp32": variant("macos/fp32-s256", reference=True),
            "macos-fp32-s512": variant("macos/fp32-s512", reference=True),
        },
        "host": {
            "layout": metadata["layout"], "tokenizer": "tokenizer/tokenizer.json",
            "cls_id": metadata["cls_token_id"], "sep_id": metadata["sep_token_id"], "pad_id": metadata["pad_token_id"],
            "mask_id": metadata["mask_token_id"], "head_max_len": metadata["head_max_len"],
            "option_text_tokens": metadata["option_text_tokens"], "source_max_len": metadata["source_max_len"],
            "sequence": "[CLS] <type> question: <instructions> [SEP] ([MASK] <option text, <=48 tokens>)... [SEP] <state, right-truncated> [SEP]",
            "readout": ("token_logits at the marker positions; softmax of the raw logits -> the four act features -> act; "
                        "answer = softmax(logits / T), T by bucket (type:K) first, then by type"),
            "temperatures": "metadata.json decision block: source_temperature [1,1,1] and the fitted calibration",
            "compute_units": "gpu (a Neural Engine preference is refused: its answers fall outside the bar and vary run to run)",
        },
        "graph": {"functions": {"main": {"inputs": metadata["inputs"], "outputs": metadata["outputs"]},
                                "act": metadata["act"]},
                  "windows": [256, 512]},
    }


def link_tree(src: Path, dst: Path) -> None:
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_symlink():
            raise SystemExit(f"symlink in a variant folder: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def df() -> str:
    return subprocess.run(["df", "-h", "/System/Volumes/Data"], capture_output=True, text=True).stdout.splitlines()[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=export_root() / "hf_stage")
    parser.add_argument("--license-text", type=Path, help="an Apache-2.0 LICENSE file (sha256-checked)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    before = df()
    source = verify_source()
    card = (repo_root() / "models" / "laya-multilingual" / "README.md").read_text()
    manifests = {v: check_variant(export_root() / v) for v in VARIANTS}
    out = args.out
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for v in VARIANTS:
        link_tree(export_root() / v, out / v)
    (out / "README.md").write_text(hub_readme(card))
    shutil.copy2(apache_text(args.license_text), out / "LICENSE")
    (out / "LICENSE-NOTE.md").write_text(license_note())
    shutil.copy2(source.parent / "README.md", out / "UPSTREAM_README.md")
    shutil.copy2(source / "encoder" / "config.json", out / "config.json")
    shutil.copy2(source / "rl_agent_config.json", out / "rl_agent_config.json")
    (out / "coreai-kit.json").write_text(json.dumps(kit_json(out, manifests), indent=2, ensure_ascii=False) + "\n")

    files = sorted(p for p in out.rglob("*") if p.is_file())
    sums = [(sha256_of(p), str(p.relative_to(out))) for p in files]
    (out / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for h, name in sums))
    total = sum(p.stat().st_size for p in files)
    folders = {v: sum(p.stat().st_size for p in (out / v).rglob("*") if p.is_file()) for v in VARIANTS}
    after = df()
    staging = [
        "# STAGING — mlboydaisuke/Laya-Multilingual-CoreAI (local; nothing uploaded)", "",
        f"Staged {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} by conversion/laya/stage_hf_laya.py "
        f"from {export_root()}; source {MODEL_ID}@{MODEL_SHA}/{SUBFOLDER}.", "",
        f"- {len(files)} files listed in SHA256SUMS ({total:,} bytes), plus SHA256SUMS and this file; "
        f"{len(VARIANTS)} variant folders",
        "- variant files are hard links to the gated export folders (no symlinks); the fp16 recipe is not staged",
        f"- disk before: `{before}`", f"- disk after:  `{after}`",
        "- UPSTREAM_README.md is the publisher's README, unmodified; it names and compares against another product "
        "by name, so whether to include it is the user's decision", "",
        "| Folder | Bytes |", "|---|---:|", *[f"| `{v}/` | {b:,} |" for v, b in folders.items()], "",
        "Upload (not run; needs the user's GO):", "",
        "```", f"HF_HUB_DISABLE_XET=1 hf upload-large-folder {HF_REPO} {out} --repo-type model --exclude STAGING.md", "```", "",
        "Afterwards: compare the Hub's lfs.sha256 + size (model_info(files_metadata=True)) with SHA256SUMS, then put the",
        "revision into models/laya-multilingual/README.md and recipe.toml.", "",
        "## sha256", "", "```", *[f"{h}  {name}" for h, name in sums], "```", ""]
    (out / "STAGING.md").write_text("\n".join(staging))
    print(json.dumps({"out": str(out), "files_in_sha256sums": len(files), "bytes": total, "folders": folders,
                      "disk_before": before, "disk_after": after}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
