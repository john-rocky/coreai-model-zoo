#!/usr/bin/env python3
"""Compile a stripped static iOS bundle for one iPhone's Neural Engine, laid out as the subtree CoreAIKit downloads on
that phone (`ios-<arch>/`).

    python conversion/ios_device_bundle.py <stripped bundle> <out dir> --arch h19p --expect-regions 30

<stripped bundle> = metadata.json + <name>.aimodel + tokenizer/ (export_lfm3_ane_static.py's export after
d1/strip_bundle.py). <out dir> is written once and holds exactly:
- `<name>.<arch>.ane.aimodelc`: `xcrun coreai-build compile <name>.aimodel --platform iOS --preferred-compute
  neural-engine --architecture <arch>` (export_lfm3_ane_static.aot: the same command, Xcode and record);
- `metadata.json`: the stripped bundle's, with `assets.main` on the `.aimodelc` and `compilation.targets` = that
  command, and without `source.weights` (a local checkpoint path names the export machine, not the model);
- `tokenizer/`: the stripped bundle's, copied.
The compile log goes beside <out dir> (`<out dir>.compile.log`), never into it.

CoreAIKit's catalog names the iPhone variant `ios`; on a phone the kit takes `ios-<arch>/` for the architecture Core AI
reports (`AIModel.deviceArchitectureName`: h18p on the iPhone 17 Pro, h19p on the iPhone 18 Pro) and falls back to
`ios/`, so the subtree is named after the arch the graph was compiled for. Nothing here loads the compile: an iOS
bundle run on a Mac wedged the machine twice; it is checked on the phone. It exits 3 and leaves the output in place
for inspection when the compile does not put exactly `--expect-regions` transformer functions in Neural Engine
regions with no ANECCompileOffline failure, or when a file of the output holds the export machine's paths.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARCHS = ("h18p", "h19p")


def count(p: Path, needle: bytes) -> int:
    n, tail = 0, b""
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            n += (tail + b).count(needle) - tail.count(needle)
            tail = b[-(len(needle) - 1):]
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the stripped bundle: metadata.json + <name>.aimodel + tokenizer/")
    ap.add_argument("out", help="the device bundle directory to write (named ios-<arch> where it is published)")
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--expect-regions", type=int, required=True,
                    help="the transformer functions that must each get a Neural Engine region")
    a = ap.parse_args()
    from export_lfm3_ane_static import aot

    src, out = Path(a.bundle).resolve(), Path(a.out).resolve()
    meta = json.loads((src / "metadata.json").read_text())
    asset = src / meta["assets"]["main"]
    if asset.suffix != ".aimodel" or not asset.is_dir():
        sys.exit(f"{asset}: assets.main is not the bundle's .aimodel")
    if not (src / "tokenizer").is_dir():
        sys.exit(f"{src}: no tokenizer/")
    if out.exists():
        sys.exit(f"{out} exists: a device bundle is written once (remove it on purpose, then run again)")
    out.mkdir(parents=True)
    name = asset.stem
    dest = out / f"{name}.{a.arch}.ane.aimodelc"
    rec = aot(asset, dest, a.arch, out.parent / f"{out.name}.compile.log")
    rec.pop("ane_region_names", None)
    if rec["rc"] != 0 or "error" in rec:
        print(json.dumps(rec, indent=1, default=str))
        return 3
    meta["assets"]["main"] = dest.name
    meta.setdefault("compilation", {})["targets"] = [
        f"{a.arch} neural-engine (xcrun coreai-build compile --platform iOS --preferred-compute neural-engine "
        f"--architecture {a.arch})"]
    dropped = meta.get("source", {}).pop("weights", None)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    shutil.copytree(src / "tokenizer", out / "tokenizer")
    needles = [b"/Users/", b"/private/", b"/tmp/claude", str(Path.home()).encode()]
    hits = {}
    for p in sorted(q for q in out.rglob("*") if q.is_file()):
        n = sum(count(p, x) for x in needles)
        if n:
            hits[str(p.relative_to(out))] = n
    files = sorted(str(q.relative_to(out)) for q in out.rglob("*") if q.is_file())
    rec.update({"device_bundle": str(out), "files": len(files),
                "bytes": sum((out / f).stat().st_size for f in files),
                "dropped_source_weights": dropped is not None, "path_hits": hits})
    ok = (rec.get("ane_regions_unique") == a.expect_regions and rec.get("anec_failures") == 0 and not hits)
    rec["ok"] = ok
    print(json.dumps(rec, indent=1, default=str))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
