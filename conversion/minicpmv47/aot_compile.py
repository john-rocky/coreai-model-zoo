#!/usr/bin/env python3
"""AOT-compile a MiniCPM-V-4.7 bundle's `.aimodel` for a GPU (Mac h16c by default) and record the result.

    coreai-build compile <bundle>/<name>.aimodel --output <aot-dir> --platform macOS --preferred-compute gpu
        --architecture h16c [--expect-frequent-reshapes]   -> <aot-dir>/<name>.h16c.aimodelc

`--platform iOS --arch h19p` compiles for the iPhone 18 Pro (iPhone19,2; an h18p asset does not load
there: `incompatibleCompiledAssetArchitecture`) -> <aot-dir>/<name>.h19p.aimodelc.

`--reshapes` adds `--expect-frequent-reshapes` (the decoder: dynamic position / KV dims, as
`decider_vision/export_decoder.py --aot`); the fixed-shape vision tower compiles without it (decider
README §2). `coreai-build` is resolved with `xcrun -f` under DEVELOPER_DIR (the Xcode 27 RC).

Record: the flags, the compile wall, the `.aimodelc` bytes and per-file sha256, the ANE region count
(files whose name contains `ANE_region`, de-duplicated by name; 0 expected for a GPU compile), the
`main.hash` bytes as hex (`xxd -p`), the source `.aimodel`'s own `main.hash`, and the compiler log tail.

    DEVELOPER_DIR=… QW python conversion/minicpmv47/aot_compile.py <bundle-dir> [--reshapes] --record <json>
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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir  # noqa: E402

def base_flags(platform: str, arch: str) -> list[str]:
    return ["--platform", platform, "--preferred-compute", "gpu", "--architecture", arch]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree,
            "files": {k: {"bytes": (path / k).stat().st_size, "sha256": v} for k, v in per.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="<exports>/<name>/ holding <name>.aimodel")
    ap.add_argument("--aot-dir", default=str(exports_dir() / "minicpmv47_aotc"))
    ap.add_argument("--reshapes", action="store_true", help="add --expect-frequent-reshapes (dynamic dims)")
    ap.add_argument("--platform", default="macOS", choices=["macOS", "iOS"])
    ap.add_argument("--arch", default="h16c", help="h16c (Mac M4 Max), h19p (iPhone 18 Pro)")
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    name = bundle.name
    aimodel = bundle / f"{name}.aimodel"
    if not aimodel.is_dir():
        sys.exit(f"no {aimodel}")
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    cb = cb.stdout.strip()
    version = subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip()
    aot_dir = Path(args.aot_dir)
    aot_dir.mkdir(parents=True, exist_ok=True)
    target = aot_dir / f"{name}.{args.arch}.aimodelc"
    if target.exists():
        shutil.rmtree(target)
    flags = base_flags(args.platform, args.arch) + (["--expect-frequent-reshapes"] if args.reshapes else [])
    cmd = [cb, "compile", str(aimodel), "--output", str(aot_dir), *flags]
    print(f"[aot] {' '.join(cmd[1:])}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    log = (proc.stdout + proc.stderr).splitlines()
    for line in log[-20:]:
        print(f"[coreai-build] {line}", flush=True)
    if proc.returncode != 0 or not target.exists():
        Path(args.record).write_text(json.dumps({"bundle": str(bundle), "cmd": cmd, "returncode": proc.returncode,
                                                 "wall_seconds": wall, "log": log}, indent=1) + "\n")
        sys.exit(f"[aot] FAILED rc={proc.returncode}; target exists {target.exists()}")
    ane = sorted({p.name for p in target.rglob("*") if "ANE_region" in p.name})
    main_hash = target / "main.hash"
    src_hash = aimodel / "main.hash"
    rec = {
        "schema": "minicpmv47-aot/1", "bundle": str(bundle), "aimodel": str(aimodel), "aimodelc": str(target),
        "coreai_build": {"path_resolved_by": "xcrun -f coreai-build", "version": version,
                         "developer_dir": os.environ.get("DEVELOPER_DIR")},
        "flags": flags, "wall_seconds": wall, "returncode": proc.returncode,
        "aimodelc_digest": tree_digest(target),
        "ane_region_files": ane, "ane_region_count": len(ane),
        "main_hash_hex": main_hash.read_bytes().hex() if main_hash.exists() else None,
        "aimodel_main_hash_hex": src_hash.read_bytes().hex() if src_hash.exists() else None,
        "log_tail": log[-40:], "log_lines": len(log),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[aot] {target}: {rec['aimodelc_digest']['bytes']:,} B, {wall:.1f}s, ANE regions {len(ane)}, "
          f"main.hash {rec['main_hash_hex']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
