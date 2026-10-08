#!/usr/bin/env python3
"""AOT-compile the shipping bundles for an iPhone (iPhone 18 Pro = h19p). Never loads them on this Mac.

    python3 conversion/d1_omni/aot_ios.py                      # the 10 ship bundles, h19p
    python3 conversion/d1_omni/aot_ios.py --only decide-fp16-L256 vision-fp16

The ship form (round 7's speed ladder, decided by the supervisor on 2026-10-08): the fp16 decision graph at L = 256 /
512 / 1024 / 2048 / 4096, the fp16 vision graph and the fp16 audio graph at 5 / 10 / 20 / 30 s. Each bundle's files are
checked against its export manifest first, then

    $(xcrun -f coreai-build) compile <bundle>.aimodel --output <dir> --platform iOS --min-deployment-version 27.0
        --preferred-compute gpu --architecture h19p

(static graphs: no --expect-frequent-reshapes; DEVELOPER_DIR = the Xcode whose Metal Toolchain holds coreai-build) into
<work>/compiled/ios-<arch>/<name>/. The architecture is the device generation's: h19p = iPhone 18 Pro, h18p = iPhone 17
Pro; a .aimodelc of another architecture is refused at load (incompatibleCompiledAssetArchitecture).

Recorded per bundle in <dir>/provenance/aot-manifest.json: the command, seconds, exit code, stderr, the .aimodelc's
files (bytes, sha256), the bytes of every resources.bin, the hash files (main.hash names the device's compile-cache
entry), the Neural Engine regions (by unique name; a GPU compile should have none), disk free before and after; the
set in <work>/compiled/ios-<arch>/manifest.json. A compiled iPhone bundle is never loaded on a Mac (it can wedge the
GPU stack): this script only reads its files. An existing folder is never replaced.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

from aot_decide import ane_regions, coreai_build, disk_free, main_hash, tool_env  # noqa: E402
from export_decide import environment, file_inventory, sha256_file, write_json  # noqa: E402

WORK = work_path("_d1_omni")
MACOS = WORK / "bundles" / "d1-omni-600m" / "macos"
MIN_OS = "27.0"
# name -> the export folder (bundles/d1-omni-600m/macos/<folder>) of the ship form
SHIP = {**{f"decide-fp16-L{L}": f"fp16-L{L}" for L in (256, 512, 1024, 2048, 4096)},
        "vision-fp16": "vision-fp16",
        **{f"audio-fp16-{s}s": f"audio-fp16-{s}s" for s in (5, 10, 20, 30)}}


def compile_one(cb: str, name: str, arch: str, out_root: Path) -> dict:
    source_dir = MACOS / SHIP[name]
    manifest_path = source_dir / "provenance" / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle = source_dir / manifest["bundle"]
    for item in manifest["files"]:
        if sha256_file(bundle / item["path"]) != item["sha256"]:
            raise SystemExit(f"{bundle / item['path']} changed since its export")
    out_dir = out_root / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing folder is never replaced")
    out_dir.mkdir(parents=True)
    argv = [cb, "compile", str(bundle), "--output", str(out_dir), "--platform", "iOS", "--min-deployment-version",
            MIN_OS, "--preferred-compute", "gpu", "--architecture", arch]
    before = disk_free()
    print("$", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=tool_env())
    seconds = time.perf_counter() - t0
    after = disk_free()
    (out_dir / "provenance").mkdir()
    (out_dir / "provenance" / "coreai-build.log").write_text(f"$ {' '.join(argv)}\n{done.stdout}\n--- stderr\n{done.stderr}")
    compiled = sorted(out_dir.glob("*.aimodelc"))
    record = {"name": name, "platform": "iOS", "min_deployment_version": MIN_OS, "architecture": arch,
              "preferred_compute": "gpu", "command": argv, "developer_dir": tool_env()["DEVELOPER_DIR"],
              "coreai_build_version": subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip(),
              "seconds": seconds, "exit_code": done.returncode, "stderr_tail": done.stderr[-4000:],
              "disk_free_before": before, "disk_free_after": after,
              "source": {"folder": str(source_dir), "bundle": manifest["bundle"], "bytes": manifest["bytes"],
                         "export_manifest_sha256": sha256_file(manifest_path), "export_status": manifest["status"],
                         "precision": manifest["precision"],
                         "main_hash": (bundle / "main.hash").read_bytes().hex() if (bundle / "main.hash").exists() else None},
              "loaded_on_this_mac": False}
    if done.returncode != 0 or len(compiled) != 1:
        record["status"] = "FAILED"
        write_json(out_dir / "provenance" / "aot-manifest.json", record)
        print(f"FAILED {name}: exit {done.returncode}, {len(compiled)} .aimodelc\n{done.stderr[-2000:]}", flush=True)
        return record
    aimodelc = compiled[0]
    files = file_inventory(aimodelc)
    resources = [{"path": f["path"], "bytes": f["bytes"]} for f in files if Path(f["path"]).name == "resources.bin"]
    record.update(status="COMPILED", aimodelc=aimodelc.name, bytes=sum(f["bytes"] for f in files), files=files,
                  resources_bin=resources, resources_bin_bytes=sum(r["bytes"] for r in resources),
                  resources_bin_over_source_bytes=sum(r["bytes"] for r in resources) / manifest["bytes"],
                  hashes=main_hash(aimodelc), ane=ane_regions(aimodelc),
                  code_sha256={"aot_ios.py": sha256_file(HERE / "aot_ios.py")},
                  environment=environment(("coreai-core", "coreai-torch")))
    write_json(out_dir / "provenance" / "aot-manifest.json", record)
    print(json.dumps({"name": name, "aimodelc": aimodelc.name, "seconds": round(seconds, 1), "bytes": record["bytes"],
                      "resources_bin_bytes": record["resources_bin_bytes"], "source_bytes": manifest["bytes"],
                      "ane_regions": record["ane"]["regions"]}), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architecture", default="h19p", help="h19p = iPhone 18 Pro, h18p = iPhone 17 Pro")
    parser.add_argument("--only", nargs="*", choices=sorted(SHIP), help="a subset of the ship bundles")
    args = parser.parse_args()
    names = args.only or list(SHIP)
    out_root = WORK / "compiled" / f"ios-{args.architecture}"
    out_root.mkdir(parents=True, exist_ok=True)
    cb = coreai_build()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    records = {name: compile_one(cb, name, args.architecture, out_root) for name in names}
    summary_path = out_root / "manifest.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {"bundles": {}}
    bundles = previous["bundles"]
    for name, r in records.items():
        if name in bundles:
            raise SystemExit(f"{name} is already in {summary_path}")
        bundles[name] = {k: r.get(k) for k in ("status", "aimodelc", "bytes", "resources_bin_bytes", "seconds",
                                               "exit_code", "hashes")} | {
            "ane_regions": (r.get("ane") or {}).get("regions"), "source_bundle": r["source"]["bundle"],
            "source_bytes": r["source"]["bytes"], "source_main_hash": r["source"]["main_hash"],
            "manifest": str((out_root / name / "provenance" / "aot-manifest.json").relative_to(WORK))}
    summary = {"schema": "d1-omni-ios-aot/1", "architecture": args.architecture, "platform": "iOS",
               "min_deployment_version": MIN_OS, "preferred_compute": "gpu",
               "runs": previous.get("runs", []) + [{"started": started, "seconds": time.perf_counter() - t0,
                                                     "names": names}],
               "loaded_on_this_mac": False, "bundles": bundles,
               "total_bytes": sum(b["bytes"] or 0 for b in bundles.values())}
    write_json(summary_path, summary)
    failed = [n for n, r in records.items() if r["status"] != "COMPILED"]
    print(json.dumps({"out": str(out_root), "compiled": len(records) - len(failed), "failed": failed,
                      "total_bytes": summary["total_bytes"]}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
