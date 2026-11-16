#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Lay out the Hugging Face repo mlboydaisuke/Fun-ASR-Nano-2512-CoreAI in <work>/ship/ exactly as it is
uploaded (APFS clones of the gated bundles), check sizes and metadata, write SHA256SUMS and print the size
table. Nothing is uploaded here; the upload is a separate step:

    HF_HOME=~/code/coreai/_funasr_nano/hf ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py            # stage + checks
    HF_HOME=~/code/coreai/_funasr_nano/hf ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py --upload
(HF_HOME points hf_snapshot at the cache that holds the source repos' config files.)

Layout (the paths CoreAIKit's FunASRModelID names; one subtree serves macOS and iOS because both bundles are
JIT `.aimodel`s the device specializes itself):

    gpu-pipelined/funasr_nano_2512_decode_int8lin_n63_s1/     <name>.aimodel + metadata.json + tokenizer/
    gpu-pipelined/funasr_nano_audio_encoder_fp16w32_l500/     funasr_nano_audio_encoder_fp16w32_l500.aimodel
    config.json, config.yaml, preprocessor_config.json          the source model's (vLLM repo / official repo)
    LICENSE, NOTICE, README.md                                  Apache-2.0 text, attribution, the card
    SHA256SUMS                                                  every file except itself

The card is the one file this script does not write: <work>/ship/README.md comes from _hf_README.md and is
copied only if the staged copy is missing or older.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

WORK = work_path("_funasr_nano")
EXPORTS = WORK / "exports"
SHIP = WORK / "ship"
REPO = "mlboydaisuke/Fun-ASR-Nano-2512-CoreAI"
DECODER = "funasr_nano_2512_decode_int8lin_n63_s1"
ENCODER = "funasr_nano_audio_encoder_fp16w32_l500.aimodel"
VLLM_REPO, VLLM_REV = "FunAudioLLM/Fun-ASR-Nano-2512-vllm", "a4362c943d48951f98ca2a62181cc028970270c5"
OFFICIAL_REPO, OFFICIAL_REV = "FunAudioLLM/Fun-ASR-Nano-2512", "272c57b82523ada6fd87095e955f8e29100979ab"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clone(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    subprocess.run(["cp", "-cR", str(src), str(dst)], check=True)   # APFS clone: no second copy on disk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upload", action="store_true", help="upload the staged tree")
    args = ap.parse_args()

    dec_src, enc_src = EXPORTS / DECODER, EXPORTS / ENCODER
    for p in (dec_src / f"{DECODER}.aimodel", dec_src / "metadata.json", dec_src / "tokenizer" / "tokenizer.json", enc_src):
        assert p.exists(), f"missing export: {p}"
    SHIP.mkdir(parents=True, exist_ok=True)
    clone(dec_src, SHIP / "gpu-pipelined" / DECODER)
    clone(enc_src, SHIP / "gpu-pipelined" / ENCODER.removesuffix(".aimodel") / ENCODER)
    for name, repo, rev in (("config.json", VLLM_REPO, VLLM_REV), ("preprocessor_config.json", VLLM_REPO, VLLM_REV),
                            ("config.yaml", OFFICIAL_REPO, OFFICIAL_REV)):
        shutil.copy(hf_snapshot(repo, name, revision=rev), SHIP / name)
    shutil.copy(HERE / "_hf_LICENSE", SHIP / "LICENSE")
    shutil.copy(HERE / "_hf_NOTICE", SHIP / "NOTICE")
    card_src, card_dst = HERE / "_hf_README.md", SHIP / "README.md"
    if not card_dst.exists() or card_src.stat().st_mtime > card_dst.stat().st_mtime:
        shutil.copy(card_src, card_dst)

    # metadata says what the graph is (residual scale + eps are part of the shipped decoder)
    import json
    meta = json.loads((SHIP / "gpu-pipelined" / DECODER / "metadata.json").read_text())
    assert meta.get("residual_scale") == 0.25 and meta.get("rmsnorm_eps_residual") == 6.25e-8, meta
    assert (SHIP / "gpu-pipelined" / DECODER / "tokenizer" / "tokenizer.json").exists()

    lines, total = [], 0
    for f in sorted(p for p in SHIP.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rel = f.relative_to(SHIP).as_posix()
        if "/.cache/" in f"/{rel}":
            continue
        total += f.stat().st_size
        lines.append(f"{sha256(f)}  {rel}")
    (SHIP / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for top in sorted(SHIP.iterdir()):
        n = sum(p.stat().st_size for p in ([top] if top.is_file() else top.rglob("*")) if p.is_file())
        print(f"{n / 1e6:9.1f} MB  {top.name}")
    print(f"{total / 1e6:9.1f} MB  total ({len(lines)} files) -> {SHIP}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(REPO, repo_type="model", exist_ok=True)
        # upload_large_folder leaves a .cache/huggingface/ inside the staging dir: remove it before re-hashing
        api.upload_large_folder(repo_id=REPO, folder_path=str(SHIP), repo_type="model")
        print(f"uploaded -> https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
