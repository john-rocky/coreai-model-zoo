#!/usr/bin/env python3
"""AOT-compile the stripped shipping bundles (macos-ship/, strip_ship.py) for this Mac (h16c) or the iPhone 18 Pro (h19p).

    python3 conversion/d1_omni/aot_ship.py --target h16c          # -> <work>/compiled/ship-h16c/<name>/
    python3 conversion/d1_omni/aot_ship.py --target h19p          # -> <work>/compiled/ship-h19p/<name>/ (never loaded here)
    python3 conversion/d1_omni/aot_ship.py --target h16c --only fp16-L256
    python3 conversion/d1_omni/aot_ship.py --target h19p --compute neural-engine --only audio-fp16-10s fp16-L256
                                                                  # -> <work>/compiled/ship-h19p-ane/<name>/ (round 11)
    python3 conversion/d1_omni/aot_ship.py --target h16c --only fp16-L64 fp16-L128
                                  # round 12: macos-ship-small/ -> ship-h16c/<name>/, listed in ship-h16c/manifest.small.json

Each bundle's files are checked against its provenance/strip.json first (the stripped bytes), then

    $(xcrun -f coreai-build) compile <bundle>.aimodel --output <dir> --platform macOS --preferred-compute gpu
        --architecture h16c
    $(xcrun -f coreai-build) compile <bundle>.aimodel --output <dir> --platform iOS --min-deployment-version 27.0
        --preferred-compute gpu --architecture h19p

(aot_decide.py's and aot_ios.py's commands: static graphs, no --expect-frequent-reshapes; DEVELOPER_DIR = the Xcode
whose Metal Toolchain holds coreai-build). `--compute neural-engine` passes `--preferred-compute neural-engine` instead
and writes into `ship-<target>-ane/`: the Neural Engine regions the compile formed are counted by name (deduped; the
exit code says nothing about placement). Recorded per bundle in <dir>/provenance/aot-manifest.json, in the shape the
readers expect (runtime_check.py, vision_check.py, audio_check.py, gate_swift.py, host_dump_media.py, the Swift CLI's
--aot-root, stage_hf.py): the command, seconds, exit code, stderr, the .aimodelc's files (bytes, sha256), the bytes of
every resources.bin, the hash files (main.hash names the compile-cache entry a load creates), the Neural Engine regions
(a GPU compile has none), a byte scan of every compiled file for "/Users/", the home directory and the user name, disk
free before and after; the set in <work>/compiled/ship-<target>/manifest.json. An h19p .aimodelc is never loaded on
this Mac (it can wedge the GPU stack): this script only reads its files. An existing folder is never replaced.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

from aot_decide import ane_regions, coreai_build, disk_free, inspect, main_hash, tool_env  # noqa: E402
from export_decide import environment, file_inventory, sha256_file, write_json  # noqa: E402
from strip_ship import NEEDLES, SHIP, SMALL, scan  # noqa: E402
from strip_ship import out_root as source_root  # noqa: E402

WORK = work_path("_d1_omni")
MIN_OS = "27.0"
TARGETS = {"h16c": ["--platform", "macOS"],
           "h19p": ["--platform", "iOS", "--min-deployment-version", MIN_OS]}


def compile_one(cb: str, name: str, target: str, out_root: Path, compute: str = "gpu") -> dict:
    source_dir = source_root(name) / name
    strip_path = source_dir / "provenance" / "strip.json"
    strip = json.loads(strip_path.read_text())
    if strip["status"] != "PASS":
        raise SystemExit(f"{source_dir}: strip {strip['status']}")
    manifest_path = source_dir / "provenance" / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle = source_dir / strip["bundle"]
    for item in strip["after"]["files"]:
        f = bundle / item["path"]
        if f.stat().st_size != item["bytes"] or sha256_file(f) != item["sha256"]:
            raise SystemExit(f"{f} changed since it was stripped")
    out_dir = out_root / name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing folder is never replaced")
    out_dir.mkdir(parents=True)
    argv = [cb, "compile", str(bundle), "--output", str(out_dir), *TARGETS[target], "--preferred-compute", compute,
            "--architecture", target]
    before = disk_free()
    print("$", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=tool_env())
    seconds = time.perf_counter() - t0
    after = disk_free()
    (out_dir / "provenance").mkdir()
    (out_dir / "provenance" / "coreai-build.log").write_text(f"$ {' '.join(argv)}\n{done.stdout}\n--- stderr\n{done.stderr}")
    compiled = sorted(out_dir.glob("*.aimodelc"))
    record = {"name": name, "precision": manifest["precision"], "seq_len": manifest.get("seq_len"),
              "bucket_s": manifest.get("bucket_s"), "platform": "macOS" if target == "h16c" else "iOS",
              "min_deployment_version": None if target == "h16c" else MIN_OS, "architecture": target,
              "preferred_compute": compute, "command": argv, "developer_dir": tool_env()["DEVELOPER_DIR"],
              "coreai_build_version": subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip(),
              "seconds": seconds, "exit_code": done.returncode, "stderr_tail": done.stderr[-4000:],
              "disk_free_before": before, "disk_free_after": after,
              "source": {"folder": str(source_dir), "bundle": strip["bundle"], "bytes": strip["after"]["bytes"],
                         "main_hash": strip["after"]["main_hash"], "unstripped_main_hash": strip["before"]["main_hash"],
                         "strip_json_sha256": sha256_file(strip_path),
                         "export_manifest_sha256": sha256_file(manifest_path), "export_status": manifest["status"],
                         "precision": manifest["precision"]}}
    if target == "h19p":
        record["loaded_on_this_mac"] = False
    if done.returncode != 0 or len(compiled) != 1:
        record["status"] = "FAILED"
        write_json(out_dir / "provenance" / "aot-manifest.json", record)
        print(f"FAILED {name}: exit {done.returncode}, {len(compiled)} .aimodelc\n{done.stderr[-2000:]}", flush=True)
        return record
    aimodelc = compiled[0]
    files = file_inventory(aimodelc)
    resources = [{"path": f["path"], "bytes": f["bytes"]} for f in files if Path(f["path"]).name == "resources.bin"]
    local = {str(p.relative_to(aimodelc)): scan(p, NEEDLES) for p in sorted(aimodelc.rglob("*")) if p.is_file()}
    record.update(status="COMPILED", aimodelc=aimodelc.name, aimodelc_kept=True, bytes=sum(f["bytes"] for f in files),
                  files=files, resources_bin=resources, resources_bin_bytes=sum(r["bytes"] for r in resources),
                  resources_bin_over_source_bytes=sum(r["bytes"] for r in resources) / strip["after"]["bytes"],
                  hashes=main_hash(aimodelc), ane=ane_regions(aimodelc),
                  local_paths={f: c for f, c in local.items() if any(c.values())},
                  local_path_files_scanned=len(local),
                  code_sha256={"aot_ship.py": sha256_file(HERE / "aot_ship.py")},
                  environment=environment(("coreai-core", "coreai-torch")))
    if target == "h16c":
        record["compiled_inspect"] = inspect(cb, aimodelc)
    write_json(out_dir / "provenance" / "aot-manifest.json", record)
    print(json.dumps({"name": name, "aimodelc": aimodelc.name, "seconds": round(seconds, 1), "bytes": record["bytes"],
                      "resources_bin_bytes": record["resources_bin_bytes"], "source_bytes": strip["after"]["bytes"],
                      "main_hash": record["hashes"].get("main.hash", "")[:16], "ane_regions": record["ane"]["regions"],
                      "files_with_local_paths": len(record["local_paths"])}), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=sorted(TARGETS), required=True, help="h16c = this Mac, h19p = iPhone 18 Pro")
    parser.add_argument("--only", nargs="*", choices=SHIP + SMALL,
                        help="a subset of the ship bundles, or round 12's small buckets (macos-ship-small/)")
    parser.add_argument("--compute", choices=["gpu", "neural-engine"], default="gpu",
                        help="--preferred-compute of the compile (neural-engine -> ship-<target>-ane/)")
    args = parser.parse_args()
    names = args.only or list(SHIP)
    sources = {source_root(name) for name in names}
    if len(sources) != 1:
        parser.error("the ship bundles and round 12's small buckets come from different folders: one set per run")
    source = sources.pop()
    out_root = WORK / "compiled" / (f"ship-{args.target}" + ("-ane" if args.compute == "neural-engine" else ""))
    out_root.mkdir(parents=True, exist_ok=True)
    cb = coreai_build()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    records = {name: compile_one(cb, name, args.target, out_root, args.compute) for name in names}
    # round 12's small buckets compile into the same folder (each AOT names its source folder) but are listed in a
    # summary of their own: the ship set's manifest.json is not rewritten
    summary_path = out_root / ("manifest.small.json" if names[0] in SMALL else "manifest.json")
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {"bundles": {}}
    bundles = previous["bundles"]
    for name, r in records.items():
        if name in bundles:
            raise SystemExit(f"{name} is already in {summary_path}")
        bundles[name] = {k: r.get(k) for k in ("status", "aimodelc", "bytes", "resources_bin_bytes", "seconds",
                                               "exit_code", "hashes")} | {
            "ane_regions": (r.get("ane") or {}).get("regions"), "source_bundle": r["source"]["bundle"],
            "source_bytes": r["source"]["bytes"], "source_main_hash": r["source"]["main_hash"],
            "files_with_local_paths": len(r.get("local_paths") or {}),
            "manifest": str((out_root / name / "provenance" / "aot-manifest.json").relative_to(WORK))}
    summary = {"schema": "d1-omni-ship-aot/1", "architecture": args.target,
               "platform": "macOS" if args.target == "h16c" else "iOS",
               "min_deployment_version": None if args.target == "h16c" else MIN_OS, "preferred_compute": args.compute,
               "source": str(source.relative_to(WORK)),
               "runs": previous.get("runs", []) + [{"started": started, "seconds": time.perf_counter() - t0,
                                                     "names": names}],
               "bundles": bundles, "total_bytes": sum(b["bytes"] or 0 for b in bundles.values())}
    if args.target == "h19p":
        summary["loaded_on_this_mac"] = False
    write_json(summary_path, summary) if not summary_path.exists() else summary_path.write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    failed = [n for n, r in records.items() if r["status"] != "COMPILED"]
    print(json.dumps({"out": str(out_root), "compiled": len(records) - len(failed), "failed": failed,
                      "total_bytes": summary["total_bytes"]}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
