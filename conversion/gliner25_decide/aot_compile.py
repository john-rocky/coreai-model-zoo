"""AOT-compile a GLiNER2.5-Decide bundle (../export_gliner25_decide.py) with `coreai-build compile` for iPhone
(iOS h18p and h19p) and this Mac (macOS h16c), gpu preferred, one target after the other, and record per target:
wall time, disk free before/after, the .aimodelc size, and the ANE regions (entries named *_ANE_region_*, counted
once per name: a plain find lists each region more than once). The graph has fixed shapes, so no
--expect-frequent-reshapes (on a fixed-shape iOS graph it throws the AOT specialization away and crashes).
A compiled bundle loads only on its own architecture: the iPhone 18 Pro (iPhone19,2) reports h19p and refuses an
h18p bundle with incompatibleCompiledAssetArchitecture(device: "h19p", asset: ["h18p"]); h18p is the iPhone 17 Pro.

    ~/code/coreai/coreai-models/.venv/bin/python aot_compile.py --tag s256                  # every target
    ~/code/coreai/coreai-models/.venv/bin/python aot_compile.py --tag s512 --targets ios_gpu_h19p \
        --bundle ~/code/coreai/_gliner25_decide/exports/gliner25-decide_float16_s512_m32.aimodel

--bundle defaults to $ZOO_EXPORTS/gliner25-decide_float16_<tag>_m32.aimodel (conversion/_paths.py).
Outputs: _work/aot/<tag>_<target>/<bundle name>.<arch>.aimodelc and _work/aot/aot_compile_<tag>.json (a run of
some targets keeps the records of the others when the source bundle is the same).
Compile only: nothing here loads or runs a bundle (an iOS bundle must never run on a Mac).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE / "_work"
AOT = WORK / "aot"
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir  # noqa: E402

XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"
TARGETS = {
    "ios_gpu": ("iOS", "h18p", "gpu"),               # iPhone 17 Pro
    "ios_gpu_h19p": ("iOS", "h19p", "gpu"),          # iPhone 18 Pro (iPhone19,2)
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
    ap.add_argument("--tag", required=True, choices=["s256", "s512"])
    ap.add_argument("--bundle", default=None, help="default: $ZOO_EXPORTS/gliner25-decide_float16_<tag>_m32.aimodel")
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()
    bundle = Path(args.bundle).expanduser().resolve() if args.bundle else \
        exports_dir() / f"gliner25-decide_float16_{args.tag}_m32.aimodel"
    if not bundle.is_dir():
        raise SystemExit(f"no bundle at {bundle}")
    env = dict(os.environ)
    env.setdefault("DEVELOPER_DIR", XCODE)
    tool = subprocess.run(["xcrun", "-f", "coreai-build"], env=env, capture_output=True, text=True, check=True).stdout.strip()
    version = subprocess.run([tool, "--version"], env=env, capture_output=True, text=True).stdout.strip()
    AOT.mkdir(parents=True, exist_ok=True)
    name = f"aot_compile_{args.tag}.json"
    report = {"bundle": str(bundle), "bundle_mb": round(tree_bytes(bundle) / 1e6, 1), "tool": tool,
              "tool_version": version, "developer_dir": env["DEVELOPER_DIR"], "targets": {}}
    if (AOT / name).exists():                                   # keep the other targets' records (same source)
        old = json.loads((AOT / name).read_text())
        if old.get("bundle") == str(bundle):
            report["targets"] = old.get("targets", {})
    for t in args.targets.split(","):
        platform, arch, compute = TARGETS[t]
        out = AOT / f"{args.tag}_{t}"
        shutil.rmtree(out, ignore_errors=True)
        cmd = [tool, "compile", str(bundle), "--output", str(out), "--platform", platform,
               "--architecture", arch, "--preferred-compute", compute, "--min-deployment-version", "27.0"]
        free0 = df_free_gb()
        print(f"[{args.tag} {t}] {' '.join(cmd)} (disk free {free0:.1f} GB)", flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        wall = time.time() - t0
        free1 = df_free_gb()
        (AOT / f"{args.tag}_{t}.log").write_text(
            f"$ {' '.join(cmd)}\nexit {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
        compiled = sorted(str(x.relative_to(out)) for x in out.rglob("*.aimodelc")) if out.exists() else []
        rec = {"cmd": " ".join(cmd), "exit": p.returncode, "wall_s": round(wall, 1), "disk_free_gb_before": round(free0, 1),
               "disk_free_gb_after": round(free1, 1), "out": str(out), "aimodelc": compiled,
               "out_bytes": tree_bytes(out), "out_mb": round(tree_bytes(out) / 1e6, 1), "ane_regions": ane_regions(out),
               "stderr_tail": p.stderr.strip().splitlines()[-5:] if p.stderr.strip() else []}
        report["targets"][t] = rec
        ar = rec["ane_regions"]
        print(f"[{args.tag} {t}] exit {p.returncode} in {wall:.1f} s, {rec['out_mb']} MB, aimodelc {compiled}, ANE regions "
              f"{ar['n_regions']} {ar['ids']} ({ar['n_paths']} paths), manifest {ar['manifest_flags']}, "
              f"disk free {free1:.1f} GB", flush=True)
        if p.returncode != 0:
            print("\n".join(rec["stderr_tail"]), flush=True)
        (AOT / name).write_text(json.dumps(report, indent=1))
    print(f"wrote {AOT / name}")
    ran = args.targets.split(",")
    sys.exit(0 if all(report["targets"][t]["exit"] == 0 and report["targets"][t]["aimodelc"] for t in ran) else 1)


if __name__ == "__main__":
    main()
