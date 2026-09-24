"""AOT-compile a Nemotron-3-Diarization bundle with `xcrun coreai-build compile` for iPhone (iOS h18p)
and this Mac (macOS h16c), neural-engine and gpu preferred, one after the other, and record per target:
wall time, disk free before/after, the .aimodelc size, and the ANE regions (entries named
*_ANE_region_*, counted once per name: a plain find lists each region more than once).

    ~/code/coreai/coreai-models/.venv/bin/python aot_compile.py                       # fp16 streaming, 4 targets
    ~/code/coreai/coreai-models/.venv/bin/python aot_compile.py --bundle _work/artifacts/n3d_offline_float16.aimodel \
        --tag offline --targets ios_gpu,mac_gpu                                       # offline (T=684): iPhone + Mac GPU

Outputs: _work/aot/<target>/ (tag "" = the fp16 streaming bundle, else _work/aot/<tag>_<target>/),
_work/aot/aot_compile<_tag>.json. Compile only: nothing here loads or runs an iOS bundle.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "_work"
AOT = WORK / "aot"

TARGETS = {
    "ios_ane": ("iOS", "h18p", "neural-engine"),
    "ios_gpu": ("iOS", "h18p", "gpu"),
    "mac_ane": ("macOS", "h16c", "neural-engine"),
    "mac_gpu": ("macOS", "h16c", "gpu"),
}


def df_free_gb() -> float:
    out = subprocess.run(["df", "-k", "/System/Volumes/Data"], capture_output=True, text=True).stdout.splitlines()[-1]
    return int(out.split()[3]) / 1e6


def tree_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def ane_regions(p: Path) -> dict:
    """ANE regions = distinct `ANE_region_<i>_<j>` ids in the entry names (one region shows up as a
    .bc directory, its .bc.weights and its .mlir.bc, so paths and even file names overcount), plus the
    placement flags the MPSGraph package manifest records (mps.fullyPlacedOnANE, mps.noGPUActivity)."""
    ids, paths = set(), 0
    for f in p.rglob("*"):
        m = re.search(r"ANE_region_\d+_\d+", f.name)
        if m:
            ids.add(m.group(0))
            paths += 1
    flags = set()
    for man in p.rglob("manifest.plist"):
        text = man.read_text(errors="ignore")
        flags.update(re.findall(r"<string>(mps\.(?:fullyPlacedOnANE|noGPUActivity|aneAlignedIO))</string>", text))
    return {"n_regions": len(ids), "n_paths": paths, "ids": sorted(ids), "manifest_flags": sorted(flags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(WORK / "artifacts" / "n3d_streaming_float16.aimodel"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()
    bundle = Path(args.bundle).resolve()
    AOT.mkdir(parents=True, exist_ok=True)
    report = {"bundle": str(bundle), "bundle_mb": round(tree_bytes(bundle) / 1e6, 1), "targets": {}}
    for t in args.targets.split(","):
        platform, arch, compute = TARGETS[t]
        out = AOT / (f"{args.tag}_{t}" if args.tag else t)
        shutil.rmtree(out, ignore_errors=True)
        cmd = ["xcrun", "coreai-build", "compile", str(bundle), "--output", str(out), "--platform", platform,
               "--architecture", arch, "--preferred-compute", compute, "--min-deployment-version", "27.0"]
        free0 = df_free_gb()
        print(f"[{t}] {' '.join(cmd)} (disk free {free0:.1f} GB)", flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        free1 = df_free_gb()
        (AOT / f"{args.tag + '_' if args.tag else ''}{t}.log").write_text(
            f"$ {' '.join(cmd)}\nexit {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
        compiled = sorted(str(x.relative_to(out)) for x in out.rglob("*.aimodelc")) if out.exists() else []
        rec = {"cmd": " ".join(cmd), "exit": p.returncode, "wall_s": round(wall, 1), "disk_free_gb_before": round(free0, 1),
               "disk_free_gb_after": round(free1, 1), "out": str(out), "aimodelc": compiled,
               "out_mb": round(tree_bytes(out) / 1e6, 1), "ane_regions": ane_regions(out),
               "stderr_tail": p.stderr.strip().splitlines()[-5:] if p.stderr.strip() else []}
        report["targets"][t] = rec
        ar = rec["ane_regions"]
        print(f"[{t}] exit {p.returncode} in {wall:.1f} s, {rec['out_mb']} MB, aimodelc {compiled}, ANE regions "
              f"{ar['n_regions']} {ar['ids']} ({ar['n_paths']} paths), manifest {ar['manifest_flags']}, "
              f"disk free {free1:.1f} GB", flush=True)
        if p.returncode != 0:
            print("\n".join(rec["stderr_tail"]), flush=True)
        name = f"aot_compile{'_' + args.tag if args.tag else ''}.json"
        (AOT / name).write_text(json.dumps(report, indent=1))
    sys.exit(0)


if __name__ == "__main__":
    main()
