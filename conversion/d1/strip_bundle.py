#!/usr/bin/env python3
"""Copy a d1 bundle to a new name with the export-time debug locations stripped from its `.aimodel`.

coreai-torch writes every op's Python source location into the bytecode (`sources`, `call_stack`), so an exported
`main.mlirb` carries the export machine's absolute paths (the source tree, the venv's torch files). This writes
`<out>/` = the bundle with its `.aimodel` re-saved after coreai-torch's `strip_debug_info` (every location replaced
by an unknown one; the ops and the weights are not touched), the other files copied as they are, and the bundle's
`metadata.json` given the new `name` / `assets.main` and a `strip` record (the source and the stripped `main.mlirb`
sha256 and bytes, the tool and its versions). A stripped `.aimodel` is a new asset (a new `main.hash`, a new runtime
cache entry, other AOT bytes), so it is gated again; the d1 release ships the stripped bundles.

    python strip_bundle.py <bundle> <out bundle> [--aot]

    python strip_bundle.py $K/exports/bundles/d1_3b_decode_fp16_pf64 $K/exports/bundles_ship/d1_3b_decode_fp16_pf64_s --aot
    python strip_bundle.py $K/exports/vision/d1_3b_vision_fp16w32 $K/exports/bundles_ship/d1_3b_vision_fp16w32_s --aot

`--aot` compiles the stripped `.aimodel` for the Mac GPU into `<out parent>_aotc/<name>.h16c.aimodelc` with the
exporters' flags (`--platform macOS --preferred-compute gpu --architecture h16c`, plus `--expect-frequent-reshapes`
for a decoder), where the gates and `decide.py` look for it.

Keep one program object: `AIModelAsset.program` builds a new object on every access, so
`strip_debug_info(asset.program)` followed by `asset.program.save_asset(...)` saves the unstripped program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

AOT_BASE = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c"]
EFR = "--expect-frequent-reshapes"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def count(p: Path, needle: bytes) -> int:
    n, tail = 0, b""
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            n += (tail + b).count(needle) - tail.count(needle)
            tail = b[-(len(needle) - 1):]
    return n


def aot_compile(aimodel: Path, out_dir: Path, flags: list[str]) -> dict:
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if target.exists():
        sys.exit(f"{target} exists: an AOT compile never overwrites an asset (remove it on purpose, then run again)")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *flags]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"aimodelc": str(target), "flags": flags, "seconds": round(time.monotonic() - t0, 1),
           "returncode": proc.returncode, "stderr_tail": proc.stderr.splitlines()[-20:]}
    if proc.returncode != 0 or not target.exists():
        sys.exit(f"coreai-build failed: {json.dumps(rec)}")
    rec["main_hash_hex"] = (target / "main.hash").read_bytes().hex()
    rec["bytes"] = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the exported bundle directory (metadata.json + <name>.aimodel + its files)")
    ap.add_argument("out", help="the new bundle directory; its name becomes the bundle's and the asset's name")
    ap.add_argument("--aot", action="store_true", help="also compile the stripped .aimodel (h16c) into <out parent>_aotc/")
    a = ap.parse_args()
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info
    import coreai.runtime as rt

    src, out = Path(a.bundle).resolve(), Path(a.out).resolve()
    meta = json.loads((src / "metadata.json").read_text())
    src_asset = src / meta["assets"]["main"]
    name = out.name
    if out.exists():
        sys.exit(f"{out} exists: a stripped bundle is written once (remove it on purpose, then run again)")
    out.mkdir(parents=True)
    for p in sorted(src.iterdir()):                       # every file but the .aimodel and metadata.json, as it is
        if p == src_asset or p.name == "metadata.json":
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, out / p.name)
    dst_asset = out / f"{name}.aimodel"
    home = str(Path.home()).encode()
    before = {"bytes": (src_asset / "main.mlirb").stat().st_size, "sha256": sha256_file(src_asset / "main.mlirb"),
              "home_paths": count(src_asset / "main.mlirb", home)}
    t0 = time.monotonic()
    program = AIModelAsset.load(src_asset).program      # one object: strip and save the same program
    strip_debug_info(program)
    program.save_asset(dst_asset, rt.AIModelAssetMetadata())
    secs = round(time.monotonic() - t0, 1)
    after = {"bytes": (dst_asset / "main.mlirb").stat().st_size, "sha256": sha256_file(dst_asset / "main.mlirb"),
             "home_paths": count(dst_asset / "main.mlirb", home)}
    if after["sha256"] == before["sha256"] or after["home_paths"]:
        shutil.rmtree(out)
        sys.exit(f"the strip did not take: before {before}, after {after}")
    meta["name"] = name
    meta["assets"]["main"] = dst_asset.name
    meta["strip"] = {
        "what": "the .aimodel re-saved after coreai-torch's strip_debug_info: every op's source location (file path, "
                "call stack) replaced by an unknown location; the ops and the weights are not touched",
        "from_bundle": src.name,
        "from_main_mlirb": {"bytes": before["bytes"], "sha256": before["sha256"]},
        "main_mlirb": {"bytes": after["bytes"], "sha256": after["sha256"]},
        "tool": "conversion/d1/strip_bundle.py (coreai_torch.debugging.debug_info.strip_debug_info)",
        "versions": {"coreai-torch": version("coreai-torch"), "coreai-core": version("coreai-core")},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    rec = {"bundle": str(out), "from": str(src), "seconds": secs, "before": before, "after": after,
           "metadata_sha256": sha256_file(out / "metadata.json")}
    if a.aot:
        flags = AOT_BASE + ([EFR] if meta.get("kind") == "decision-backbone" else [])
        rec["aot"] = aot_compile(dst_asset, out.parent.parent / f"{out.parent.name}_aotc", flags)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
