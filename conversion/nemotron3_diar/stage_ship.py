"""Lay out the Hugging Face repo mlboydaisuke/Nemotron-3-Diarization-CoreAI in _work/ship/ file for file, as it
would be uploaded (APFS clones of the gated artifacts), check the clones against metadata.json, write SHA256SUMS
over every other file and print the size table. Nothing is uploaded here.

    ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py

The card is the one file this script does not make: _work/ship/README.md is the draft and survives re-runs.
Left out on purpose: the float32 bundles (parity only), the safe-LN bundles (not used) and the ANE AOT bundle
(below the agreement bar on the Mac; the device gate decides).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "_work"
SHIP = WORK / "ship"
sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

REPO_ID = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
LEGAL = Path(os.environ.get("N3D_SHIP_LEGAL", work_path("_n3d", "ship")))   # LICENSE + NOTICE

A, AOT = WORK / "artifacts", WORK / "aot"
FILES = {  # repo path -> source
    "n3d_streaming_float16.aimodel": A / "n3d_streaming_float16.aimodel",
    "n3d_offline_float16.aimodel": A / "n3d_offline_float16.aimodel",
    "n3d_streaming_float16.h18p.aimodelc": AOT / "ios_gpu" / "n3d_streaming_float16.h18p.aimodelc",
    "n3d_offline_float16.h18p.aimodelc": AOT / "offline_ios_gpu" / "n3d_offline_float16.h18p.aimodelc",
    "embedder_projection.f32le": A / "embedder_projection.f32le",
    "silence_embeds.f32le": A / "silence_embeds.f32le",
    "mel_filters_128x257.f32le": A / "mel_filters_128x257.f32le",
    "hann_window_400.f32le": A / "hann_window_400.f32le",
    "metadata.json": A / "metadata.ship.json",          # export_n3d.py --metadata --ship: bundles = the 4 shipped
    "config.json": Path(hf_snapshot(REPO_ID, "config.json", revision=REVISION)),
    "processor_config.json": Path(hf_snapshot(REPO_ID, "processor_config.json", revision=REVISION)),
    "LICENSE": LEGAL / "LICENSE",
    "NOTICE": LEGAL / "NOTICE",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """export_n3d.tree_sha256: sha256 over the sorted '<relative path>\\0<file sha256>\\n' lines of a bundle."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(root).as_posix()}\0{sha256(f)}\n".encode())
    return h.hexdigest()


def size(p: Path) -> tuple[int, int]:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files), len(files)


def main() -> None:
    missing = [f"{k} <- {v}" for k, v in FILES.items() if not v.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(missing))
    card = SHIP / "README.md"
    if not card.is_file():
        raise SystemExit(f"no card draft at {card}")
    SHIP.mkdir(parents=True, exist_ok=True)
    keep = set(FILES) | {"README.md"}
    for item in SHIP.iterdir():  # anything else in the staging dir would be uploaded by mistake
        if item.name not in keep and item.name != "SHA256SUMS":
            raise SystemExit(f"unexpected item in {SHIP}: {item.name} (remove it or add it to FILES)")
    for name, src in FILES.items():
        dst = SHIP / name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        subprocess.run(["cp", "-cR" if src.is_dir() else "-c", str(src), str(dst)], check=True)

    # metadata.json lists exactly the staged bundles, and they and the constants are the ones it describes
    meta = json.loads((SHIP / "metadata.json").read_text())
    shipped = sorted(k for k in FILES if k.endswith((".aimodel", ".aimodelc")))
    assert sorted(meta["bundles"]) == shipped, f"metadata.json bundles {sorted(meta['bundles'])} != staged {shipped}"
    for name in shipped:
        want = meta["bundles"][name]["tree_sha256"]
        got = tree_sha256(SHIP / name)
        assert got == want, f"{name}: tree sha256 {got} != metadata {want}"
    for name, rec in meta["assets"].items():
        got = sha256(SHIP / name)
        assert got == rec["sha256"], f"{name}: sha256 {got} != metadata {rec['sha256']}"
    print(f"metadata.json lists exactly the {len(shipped)} staged bundles; their tree hashes and the "
          f"{len(meta['assets'])} asset sha256 match")

    lines = []
    for f in sorted((p for p in SHIP.rglob("*") if p.is_file() and p.name != "SHA256SUMS"),
                    key=lambda p: p.relative_to(SHIP).as_posix()):
        lines.append(f"{sha256(f)}  {f.relative_to(SHIP).as_posix()}")
    (SHIP / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    print(f"\n{SHIP}\n")
    print("| path | MB | files |\n|---|---|---|")
    total = 0
    for name in sorted(keep | {"SHA256SUMS"}):
        b, n = size(SHIP / name)
        total += b
        print(f"| `{name}{'/' if (SHIP / name).is_dir() else ''}` | {b / 1e6:.3f} | {n} |")
    print(f"| total | {total / 1e6:.1f} | {len(lines) + 1} |")


if __name__ == "__main__":
    main()
