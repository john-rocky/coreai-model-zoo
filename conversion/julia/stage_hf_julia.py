#!/usr/bin/env python3
"""Stage the Hugging Face repo mlboydaisuke/Julia-1-CoreAI in a local folder. Uploads nothing.

    python3 conversion/julia/stage_hf_julia.py --variants macos/fp32-s1024 macos/wfp16-s1024 [--overwrite]

Builds <exports>/julia-1/hf_stage/ from the gated variant folders and the pinned source:

  README.md          the zoo card adapted for the Hub: front matter, the one-line Core AI introduction
                     (tools/card_first_line.txt), the DeviceMark block (managed by scripts/gen-cards),
                     repo-relative links pointed at GitHub
  LICENSE            the Apache-2.0 text (sha256-pinned; the upstream revision ships no LICENSE file)
  LICENSE-NOTE.md    what this repository changes relative to the upstream checkpoint
  config.json        the publisher's encoder/config.json, unmodified (the file the Hub counts downloads by)
  julia_config.json  the publisher's julia_config.json, unmodified
  macos/…            the gated variant folders, hard-linked (no symlinks)
  SHA256SUMS         every staged file
  STAGING.md         what is staged, sizes, hashes and the upload command (not run)
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
from _common import HF_REPO, MODEL_ID, MODEL_SHA, export_root, sha256_of, verify_source  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

GITHUB = "https://github.com/john-rocky/coreai-model-zoo/blob/main"
APACHE_2_0_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
FRONT_MATTER = """---
library_name: coreai
license: apache-2.0
base_model: SupersonicLabs/Julia-1
tags:
  - coreai
  - apple-silicon
  - on-device
  - modernbert
  - decision-model
  - text-classification
  - multilingual
language:
  - multilingual
pipeline_tag: text-classification
---
"""
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
    manifest = json.loads((folder / "provenance" / "export-manifest.json").read_text())
    assert manifest["status"] == "PASS" and not manifest.get("measure_only"), folder
    runtime = json.loads((folder / "provenance" / "runtime-gate.json").read_text())
    assert runtime["cpu_only"]["status"] == "PASS" and runtime["gpu"]["status"] == "PASS", folder
    return manifest


def hub_readme(card: str) -> str:
    body = card.split("\n", 1)[1].lstrip("\n")  # drop the zoo card's title line
    body = re.sub(r"\]\(\.\./\.\./([^)]+)\)", lambda m: f"]({GITHUB}/{m.group(1)})", body)
    body = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"]({GITHUB}/models/{m.group(1)})", body)
    body = re.sub(r"\]\((?!https?://|#)([^)]+)\)", lambda m: f"]({GITHUB}/models/julia-1/{m.group(1)})", body)
    body = body.replace("`fixtures-julia-1.json` beside this card", f"[`fixtures-julia-1.json`]({GITHUB}/models/julia-1/fixtures-julia-1.json) in the zoo repository")
    body = body.replace("`recipe.toml` beside this card names the commands", f"[`recipe.toml`]({GITHUB}/models/julia-1/recipe.toml) names the commands")
    body = body.replace(f"`{HF_REPO}` (upload pending). One folder per window", "This repository holds one folder per window")
    leftovers = re.findall(r"\]\((?!https?://|#)[^)]*\)", body)
    assert not leftovers, f"relative links left in the Hub README: {leftovers}"
    first_line = (repo_root() / "tools" / "card_first_line.txt").read_text().strip()
    return (FRONT_MATTER + "\n" + first_line + "\n\n" + DEVICEMARK + "\n\n# Julia-1 — Core AI export\n\n"
            f"Zoo card, recipe and gate transcript: [coreai-model-zoo/models/julia-1]({GITHUB}/models/julia-1/README.md).\n\n"
            + body)


def license_note() -> str:
    return f"""# License and changes

Supersonic Labs declares Apache-2.0 for `{MODEL_ID}` in the upstream model card at the pinned revision
`{MODEL_SHA}` ("The model artifacts are licensed under Apache 2.0"); no standalone LICENSE or NOTICE file is
listed at that revision. The standard Apache-2.0 text is included as `LICENSE`. The upstream base encoder is
`jhu-clsp/mmBERT-small` (MIT).

Changes relative to the upstream checkpoint:
- re-authored as a static Core AI graph from the raw weights (no transformers or `julia` package code in the
  graph), batch 1, one function `main` (encoder, type embedding, the two head layers and the scorer at every
  position); the checkpoint's `act_head` and `temperature` buffer, which the upstream inference API does not
  read, are not exported;
- `fp32` folders keep the weights as published (F32); `wfp16` folders store them in fp16 (rounded) and
  compute in fp32;
- `tokenizer/` holds the checkpoint's `tokenizer.json` and `tokenizer_config.json`, unmodified;
  `config.json` (the upstream `encoder/config.json`) and `julia_config.json` are the checkpoint's own files,
  unmodified.
Source: `{MODEL_ID}@{MODEL_SHA}`.
"""


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
    parser.add_argument("--variants", nargs="+", required=True, help="e.g. macos/fp32-s1024 macos/wfp16-s1024")
    parser.add_argument("--out", type=Path, default=export_root() / "hf_stage")
    parser.add_argument("--license-text", type=Path, help="an Apache-2.0 LICENSE file (sha256-checked)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    before = df()
    source = verify_source()
    card = (repo_root() / "models" / "julia-1" / "README.md").read_text()
    for v in args.variants:
        check_variant(export_root() / v)
    out = args.out
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for v in args.variants:
        link_tree(export_root() / v, out / v)
    (out / "README.md").write_text(hub_readme(card))
    shutil.copy2(apache_text(args.license_text), out / "LICENSE")
    (out / "LICENSE-NOTE.md").write_text(license_note())
    shutil.copy2(source / "encoder" / "config.json", out / "config.json")
    shutil.copy2(source / "julia_config.json", out / "julia_config.json")

    files = sorted(p for p in out.rglob("*") if p.is_file())
    sums = [(sha256_of(p), str(p.relative_to(out))) for p in files]
    (out / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for h, name in sums))
    total = sum(p.stat().st_size for p in files)
    folders = {v: sum(p.stat().st_size for p in (out / v).rglob("*") if p.is_file()) for v in args.variants}
    staging = [
        f"# STAGING — {HF_REPO} (local; nothing uploaded)", "",
        f"Staged {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} by conversion/julia/stage_hf_julia.py "
        f"from {export_root()}; source {MODEL_ID}@{MODEL_SHA}.", "",
        f"- {len(files)} files listed in SHA256SUMS ({total:,} bytes), plus SHA256SUMS and this file",
        "- variant files are hard links to the gated export folders (no symlinks)",
        f"- disk before: `{before}`", f"- disk after:  `{df()}`", "",
        "| Folder | Bytes |", "|---|---:|", *[f"| `{v}/` | {b:,} |" for v, b in folders.items()], "",
        "Upload (not run; needs the user's GO):", "",
        "```", f"HF_HUB_DISABLE_XET=1 hf upload-large-folder {HF_REPO} {out} --repo-type model --exclude STAGING.md", "```", "",
        "Afterwards: compare the Hub's lfs.sha256 + size (model_info(files_metadata=True)) with SHA256SUMS, delete the",
        "staging dir's .cache/huggingface/, then put the revision into models/julia-1/README.md and recipe.toml.", "",
        "## sha256", "", "```", *[f"{h}  {name}" for h, name in sums], "```", ""]
    (out / "STAGING.md").write_text("\n".join(staging))
    print(json.dumps({"out": str(out), "files_in_sha256sums": len(files), "bytes": total, "folders": folders}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
