#!/usr/bin/env python3
"""Stage 3b: AOT-compile a gated macos/ bundle for the iPhone 17 Pro (h18p), with a Neural Engine probe.

    python3 conversion/laya/aot_laya.py --window 256 --dtype wfp16

Compiles the portable JIT bundle whose export manifest says PASS with `xcrun coreai-build compile <bundle>
--output <dir> --platform iOS --min-deployment-version 27.0 --preferred-compute gpu --architecture h18p`
into <exports>/laya-multilingual/ios-h18p/<dtype>-s<S>/ — named for what it is: the network runs on the
GPU. Before that it compiles the same bundle once more with `--preferred-compute neural-engine` into a
temporary folder, counts the Neural Engine regions (unique `*_ANE_region_*` names, each with its IR bytes),
records them in the manifest as the evidence, and deletes that bundle: coreai-build exits 0 whether or not
the Neural Engine takes the graph, so the regions are the evidence, not the exit code. (2026-09-23: wfp16
gets one region of 5.7 KB IR, the fp16 recipe 47 regions of 1.1–1.3 MB. The llir Manifest lists no weights
for either, so that field says nothing about where the weights run.)

`--preferred-compute neural-engine` instead writes the neural-engine bundle itself, to
<work>/laya-multilingual/aot-probes/ — a probe, never a published folder.

The output folder holds the .aimodelc, tokenizer/, metadata.json (assets.main = the .aimodelc),
reference.json and provenance/aot-manifest.json (commands, seconds, bytes, regions, disk before/after,
source hashes). A compiled iPhone bundle is never loaded on a Mac.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import environment, export_root, file_inventory, sha256_of, work_dir, write_json  # noqa: E402


def disk_free() -> str:
    return subprocess.run(["df", "-h", "/System/Volumes/Data"], capture_output=True, text=True).stdout.splitlines()[-1]


def compile_bundle(bundle: Path, out_dir: Path, preferred: str, architecture: str) -> dict:
    argv = ["xcrun", "coreai-build", "compile", str(bundle), "--output", str(out_dir), "--platform", "iOS",
            "--min-deployment-version", "27.0", "--preferred-compute", preferred, "--architecture", architecture]
    print("$", " ".join(argv), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    seconds = time.perf_counter() - started
    compiled = sorted(out_dir.glob("*.aimodelc"))
    if completed.returncode != 0 or len(compiled) != 1:
        raise SystemExit(f"coreai-build exit {completed.returncode}, {len(compiled)} .aimodelc:\n{completed.stderr[-2000:]}")
    return {"command": argv, "seconds": seconds, "exit_code": completed.returncode, "aimodelc": compiled[0],
            "log": completed.stdout + completed.stderr}


def ane_regions(aimodelc: Path) -> dict:
    """One region shows up as a <name>_ANE_region_<i>_<j>.bc directory holding <arch>/<same name>.mlir.bc:
    count unique region names; record each one's IR bytes and whether its llir bundle lists weights."""
    names = sorted({p.name.split(".")[0] for p in aimodelc.rglob("*") if "_ANE_region_" in p.name})
    regions = []
    for name in names:
        ir = [p for p in aimodelc.rglob(f"{name}*") if p.is_file()]
        bundles = {region.parent for region in aimodelc.rglob(f"{name}.bc")}  # the *.llir.bundle holding the region
        manifests = [json.loads((b / "Manifest.json").read_text()) for b in bundles if (b / "Manifest.json").exists()]
        regions.append({"name": name, "ir_bytes": sum(p.stat().st_size for p in ir),
                        "llir_bundle_weights": [w for m in manifests for w in m.get("Weights", [])]})
    weighted = [r for r in regions if r["llir_bundle_weights"]]
    # The llir Manifest lists no weights for any region here (the fp16 recipe's 47 regions included), so that
    # field does not tell where the weights run: the verdict reports the region count and IR size only.
    verdict = ("0 ANE regions: nothing is compiled for the Neural Engine (coreai-build still exits 0)" if not regions else
               f"{len(regions)} ANE region(s), {sum(r['ir_bytes'] for r in regions)} bytes of region IR; "
               "everything else stays in the MPSGraph (GPU) package")
    return {"regions": len(regions), "regions_listing_weights": len(weighted),
            "ir_bytes": sum(r["ir_bytes"] for r in regions), "detail": regions, "verdict": verdict,
            "gpu_package_resources_bytes": sum(p.stat().st_size for p in aimodelc.rglob("resources.bin"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, required=True, choices=[256, 512])
    parser.add_argument("--dtype", choices=["wfp16", "fp32", "fp16"], default="wfp16")
    parser.add_argument("--preferred-compute", choices=["gpu", "neural-engine"], default="gpu")
    parser.add_argument("--architecture", default="h18p")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--measure-only-source", action="store_true",
                        help="neural-engine probe of a bundle built with --measure-only (the fp16 recipe): evidence only")
    args = parser.parse_args()
    source_dir = export_root() / "macos" / f"{args.dtype}-s{args.window}"
    manifest = json.loads((source_dir / "provenance" / "export-manifest.json").read_text())
    if args.measure_only_source:
        assert manifest.get("measure_only") and args.preferred_compute == "neural-engine", \
            "--measure-only-source is for a neural-engine probe of a measure-only bundle"
    else:
        assert manifest["status"] == "PASS", "compile a gated macos bundle"
    assert manifest["intended_runtime"] == "macos"
    bundle = source_dir / manifest["bundle"]
    for item in manifest["files"]:
        assert sha256_of(bundle / item["path"]) == item["sha256"], f"source changed since its gate: {item['path']}"
    if args.preferred_compute == "gpu":
        out_dir = export_root() / f"ios-{args.architecture}" / f"{args.dtype}-s{args.window}"
    else:
        out_dir = work_dir() / "aot-probes" / f"{args.dtype}-s{args.window}-neural-engine"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "provenance").mkdir()
    before = disk_free()
    probe = None
    if args.preferred_compute == "gpu":
        with tempfile.TemporaryDirectory(dir=work_dir()) as scratch:
            ne = compile_bundle(bundle, Path(scratch), "neural-engine", args.architecture)
            probe = {"command": ne["command"], "seconds": ne["seconds"], **ane_regions(ne["aimodelc"]),
                     "bundle_kept": False}
            (out_dir / "provenance" / "coreai-build-neural-engine-probe.log").write_text(ne["log"])
    built = compile_bundle(bundle, out_dir, args.preferred_compute, args.architecture)
    after = disk_free()
    (out_dir / "provenance" / "coreai-build.log").write_text(built["log"])
    aimodelc = built["aimodelc"]
    for name in ("tokenizer", "reference.json"):
        src = source_dir / name
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, out_dir / name)
    metadata = json.loads((source_dir / "metadata.json").read_text())
    metadata["assets"] = {"main": aimodelc.name}
    write_json(out_dir / "metadata.json", metadata)
    files = file_inventory(aimodelc)
    record = {
        "status": "COMPILED", "intended_runtime": "ios", "aot": True, "architecture": args.architecture,
        "preferred_compute": args.preferred_compute, "command": built["command"], "seconds": built["seconds"],
        "exit_code": built["exit_code"], "bundle": aimodelc.name, "bytes": sum(f["bytes"] for f in files), "files": files,
        "ane": ane_regions(aimodelc), "neural_engine_probe": probe,
        "source": {"variant": f"macos/{source_dir.name}", "bundle": manifest["bundle"],
                   "export_manifest_sha256": sha256_of(source_dir / "provenance" / "export-manifest.json"),
                   "files": manifest["files"], "precision": manifest["precision"]},
        "disk_free_before": before, "disk_free_after": after,
        "runtime_gate": "NOT RUN — never on a Mac; the device gate on the iPhone 17 Pro runs this folder",
        "environment": environment(()),
    }
    write_json(out_dir / "provenance" / "aot-manifest.json", record)
    print(json.dumps({"folder": str(out_dir), "bundle": aimodelc.name, "seconds": built["seconds"], "bytes": record["bytes"],
                      "ane": record["ane"]["verdict"], "neural_engine_probe": probe["verdict"] if probe else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
