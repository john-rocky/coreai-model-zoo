#!/usr/bin/env python3
"""AOT-compile an exported decision bundle for this Mac (h16c), or probe it for the Neural Engine.

    python3 conversion/d1_omni/aot_decide.py --precision wfp16 --seq-len 256                                # GPU
    python3 conversion/d1_omni/aot_decide.py --precision wfp16 --seq-len 256 --preferred-compute neural-engine
    python3 conversion/d1_omni/aot_decide.py --precision int8 --seq-len 256                                 # GPU
    python3 conversion/d1_omni/aot_decide.py --precision wfp16 --seq-len 2048 --tag r5      # again, beside the first

Compiles <work>/bundles/d1-omni-600m/macos/<precision>-L<L>/ (export_decide.py; its files are checked against the
export manifest first) with

    $(xcrun -f coreai-build) compile <bundle>.aimodel --output <dir> --platform macOS
        --preferred-compute gpu|neural-engine --architecture h16c

(DEVELOPER_DIR = the Xcode whose Metal Toolchain holds coreai-build; the graph is static, so no
--expect-frequent-reshapes) into <work>/compiled/<precision>-L<L>-h16c/ (gpu), -ane/ (neural-engine) or -none/ (no
preference: the form tried for a cpu_only load, which the gpu compile refuses with failedToSpecialize). coreai-build
exits 0 whether or not the Neural Engine takes any of the graph, so the evidence is the `*_ANE_region_*` entries in
the .aimodelc, counted by unique name (a find counts each region twice: the .bc directory and its .mlir.bc).

Recorded in <dir>/provenance/aot-manifest.json and a copy in <work>/results/aot_<precision>_L<L>_<compute>.json:
the command, seconds, exit code, the .aimodelc's files (bytes, sha256), the bytes of every resources.bin (an AOT
can hold a weight in a wider dtype than the bundle when a cast is folded at specialization), main.hash (the name
of the compile-cache entry a load creates), the ANE regions (names, IR bytes), `coreai-build inspect` of the
source and the compiled asset (storage, compute, ops), disk free before and after. A neural-engine probe with no
region keeps only the manifest (the .aimodelc is removed); an existing folder is never replaced. --tag <t> compiles
again into <dir>-<t>/ and results/aot_<precision>_L<L>_<compute>_<t>.json (a round that needs an AOT a previous round
compiled and removed keeps the first one's provenance where it is).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

from export_decide import SEQ_LENS, bundle_dir, environment, file_inventory, sha256_file, write_json  # noqa: E402

WORK = work_path("_d1_omni")
DEVELOPER_DIR = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"
ARCHITECTURE = "h16c"
SUFFIX = {"gpu": "h16c", "neural-engine": "ane", "none": "none"}  # none: tried for a cpu_only load of the fp32 bundle


def tool_env() -> dict:
    return dict(os.environ, DEVELOPER_DIR=os.environ.get("DEVELOPER_DIR", DEVELOPER_DIR))


def coreai_build() -> str:
    path = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, env=tool_env(),
                          check=True).stdout.strip()
    if not path:
        raise SystemExit("xcrun -f coreai-build resolved nothing")
    return path


def disk_free() -> str:
    return subprocess.run(["df", "-h", "/System/Volumes/Data"], capture_output=True, text=True).stdout.splitlines()[-1]


def inspect(cb: str, path: Path) -> dict:
    """coreai-build inspect --storage --compute --ops --json (the asset's own account of dtypes and placement)."""
    done = subprocess.run([cb, "inspect", str(path), "--storage", "--compute", "--ops", "--json"], capture_output=True,
                          text=True, env=tool_env())
    try:
        return {"exit_code": done.returncode, "json": json.loads(done.stdout)}
    except json.JSONDecodeError:
        return {"exit_code": done.returncode, "stdout": done.stdout[-4000:], "stderr": done.stderr[-4000:]}


def ane_regions(aimodelc: Path) -> dict:
    names = sorted({p.name.split(".")[0] for p in aimodelc.rglob("*") if "_ANE_region_" in p.name})
    regions = [{"name": n, "ir_bytes": sum(p.stat().st_size for p in aimodelc.rglob(f"{n}*") if p.is_file())}
               for n in names]
    return {"regions": len(regions), "ir_bytes": sum(r["ir_bytes"] for r in regions), "detail": regions,
            "find_entries": sum(1 for p in aimodelc.rglob("*") if "_ANE_region_" in p.name)}


def main_hash(aimodelc: Path) -> dict:
    """The hash files at the top of the .aimodelc (hex), and the sha256 of each *.mlirb next to them."""
    out = {}
    for p in sorted(aimodelc.glob("*.hash")):
        out[p.name] = p.read_bytes().hex()
    for p in sorted(aimodelc.glob("*.mlirb")):
        out[f"sha256({p.name})"] = sha256_file(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=["fp32", "wfp16", "fp16", "int8"], required=True)
    parser.add_argument("--seq-len", type=int, choices=SEQ_LENS, required=True)
    parser.add_argument("--preferred-compute", choices=sorted(SUFFIX), default="gpu")
    parser.add_argument("--tag", default=None, help="compile again into <dir>-<tag>/ (a new folder and results file)")
    args = parser.parse_args()
    source_dir = bundle_dir(args.precision, args.seq_len)
    manifest = json.loads((source_dir / "provenance" / "export-manifest.json").read_text())
    bundle = source_dir / manifest["bundle"]
    for item in manifest["files"]:
        if sha256_file(bundle / item["path"]) != item["sha256"]:
            raise SystemExit(f"{bundle / item['path']} changed since its export")
    tag = f"{args.precision}-L{args.seq_len}-{SUFFIX[args.preferred_compute]}" + (f"-{args.tag}" if args.tag else "")
    result_name = f"aot_{args.precision}_L{args.seq_len}_{SUFFIX[args.preferred_compute]}" + (
        f"_{args.tag}" if args.tag else "") + ".json"
    out_dir = WORK / "compiled" / tag
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing folder is never replaced")
    out_dir.mkdir(parents=True)
    cb = coreai_build()
    argv = [cb, "compile", str(bundle), "--output", str(out_dir), "--platform", "macOS", "--preferred-compute",
            args.preferred_compute, "--architecture", ARCHITECTURE]
    before = disk_free()
    print("$", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=tool_env())
    seconds = time.perf_counter() - t0
    after = disk_free()
    (out_dir / "provenance").mkdir()
    (out_dir / "provenance" / "coreai-build.log").write_text(f"$ {' '.join(argv)}\n{done.stdout}\n--- stderr\n{done.stderr}")
    compiled = sorted(out_dir.glob("*.aimodelc"))
    record = {"precision": args.precision, "seq_len": args.seq_len, "preferred_compute": args.preferred_compute,
              "architecture": ARCHITECTURE, "platform": "macOS", "command": argv, "developer_dir": tool_env()["DEVELOPER_DIR"],
              "coreai_build_version": subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip(),
              "seconds": seconds, "exit_code": done.returncode, "stderr_tail": done.stderr[-4000:],
              "disk_free_before": before, "disk_free_after": after,
              "source": {"folder": str(source_dir), "bundle": manifest["bundle"], "bytes": manifest["bytes"],
                         "export_manifest_sha256": sha256_file(source_dir / "provenance" / "export-manifest.json"),
                         "export_status": manifest["status"], "precision": manifest["precision"]},
              "source_inspect": inspect(cb, bundle)}
    if done.returncode != 0 or len(compiled) != 1:
        record["status"] = "FAILED"
        write_json(out_dir / "provenance" / "aot-manifest.json", record)
        write_json(WORK / "results" / result_name, record)
        raise SystemExit(f"coreai-build exit {done.returncode}, {len(compiled)} .aimodelc:\n{done.stderr[-2000:]}")
    aimodelc = compiled[0]
    files = file_inventory(aimodelc)
    resources = [{"path": f["path"], "bytes": f["bytes"]} for f in files if Path(f["path"]).name == "resources.bin"]
    regions = ane_regions(aimodelc)
    record.update(status="COMPILED", aimodelc=aimodelc.name, bytes=sum(f["bytes"] for f in files), files=files,
                  resources_bin=resources, resources_bin_bytes=sum(r["bytes"] for r in resources),
                  resources_bin_over_source_bytes=sum(r["bytes"] for r in resources) / manifest["bytes"],
                  hashes=main_hash(aimodelc), ane=regions, compiled_inspect=inspect(cb, aimodelc),
                  code_sha256={"aot_decide.py": sha256_file(HERE / "aot_decide.py")},
                  environment=environment(("coreai-core", "coreai-torch")))
    kept = True
    if args.preferred_compute == "neural-engine" and regions["regions"] == 0:
        shutil.rmtree(aimodelc)  # no region: the evidence is this manifest, not the bundle
        kept = False
    record["aimodelc_kept"] = kept
    write_json(out_dir / "provenance" / "aot-manifest.json", record)
    write_json(WORK / "results" / result_name, record)
    print(json.dumps({"folder": str(out_dir), "aimodelc": aimodelc.name, "kept": kept, "seconds": round(seconds, 1),
                      "bytes": record["bytes"], "resources_bin_bytes": record["resources_bin_bytes"],
                      "source_bytes": manifest["bytes"], "ane_regions": regions["regions"],
                      "ane_ir_bytes": regions["ir_bytes"], "hashes": record["hashes"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
