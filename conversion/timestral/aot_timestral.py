"""AOT-compile an exported Timestral bundle for this Mac (macOS h16c) or the iPhone 18 Pro (iOS h19p), for the GPU
or the Neural Engine.

    $(xcrun -f coreai-build) compile <name>.aimodel --output <dir> --platform macOS|iOS
        --preferred-compute gpu|neural-engine --architecture h16c|h19p

An iOS asset is compiled and counted here, never loaded: an iOS bundle run on a Mac wedges the GPU stack.

DEVELOPER_DIR = the Xcode whose Metal Toolchain holds coreai-build. The graph is static (fixed shapes), so
no --expect-frequent-reshapes. coreai-build exits 0 whether or not the Neural Engine takes any of the graph:
the evidence is the `*_ANE_region_*` entries in the .aimodelc, counted by unique name (a find counts each
region twice, its .bc directory and the .mlir.bc inside).

Writes <out-root>/<name>-<suffix>/<name>.aimodelc and <out-root>/<name>-<suffix>.aot.json (the command, exit
code, seconds, files and bytes, resources.bin bytes, main.hash, the ANE regions, `coreai-build inspect`).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import exports_dir  # noqa: E402

DEVELOPER_DIR = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"
PLATFORM = {"h16c": "macOS", "h19p": "iOS"}


def suffix(arch, compute):
    return arch + ("-ane" if compute == "neural-engine" else "")


def env():
    """DEVELOPER_DIR from the environment, else the Xcode above when it is installed, else the active one."""
    dev = os.environ.get("DEVELOPER_DIR") or (DEVELOPER_DIR if Path(DEVELOPER_DIR).is_dir() else None)
    return dict(os.environ, DEVELOPER_DIR=dev) if dev else dict(os.environ)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ane_regions(aimodelc):
    names = sorted({p.name.split(".")[0] for p in aimodelc.rglob("*") if "_ANE_region_" in p.name})
    return dict(regions=len(names), names=names,
                find_entries=sum(1 for p in aimodelc.rglob("*") if "_ANE_region_" in p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aimodel", required=True, type=Path,
                    help="the .aimodel, or a bundle name in <ZOO_EXPORTS>/timestral (export_timestral.py's default)")
    ap.add_argument("--preferred-compute", choices=["gpu", "neural-engine"], default="gpu")
    ap.add_argument("--arch", choices=sorted(PLATFORM), default="h16c")
    ap.add_argument("--out-root", type=Path, default=None, help="default: <ZOO_EXPORTS>/timestral/aot")
    a = ap.parse_args()
    if a.aimodel.suffix != ".aimodel":
        a.aimodel = exports_dir() / "timestral" / f"{a.aimodel.name}.aimodel"
    a.out_root = a.out_root or exports_dir() / "timestral" / "aot"
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, env=env(), check=True).stdout.strip()
    name = a.aimodel.name[: -len(".aimodel")]
    sfx = suffix(a.arch, a.preferred_compute)
    out_dir = a.out_root / f"{name}-{sfx}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    argv = [cb, "compile", str(a.aimodel), "--output", str(out_dir), "--platform", PLATFORM[a.arch],
            "--preferred-compute", a.preferred_compute, "--architecture", a.arch]
    print("$", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=env())
    secs = time.perf_counter() - t0
    (out_dir / "coreai-build.log").write_text(f"$ {' '.join(argv)}\n{done.stdout}\n--- stderr\n{done.stderr}")
    compiled = sorted(out_dir.glob("*.aimodelc"))
    rec = dict(source=str(a.aimodel), source_main_hash=(a.aimodel / "main.hash").read_bytes().hex(),
               preferred_compute=a.preferred_compute, arch=a.arch, platform=PLATFORM[a.arch], command=argv,
               developer_dir=env().get("DEVELOPER_DIR"),
               coreai_build_version=subprocess.run([cb, "--version"], capture_output=True, text=True, env=env()).stdout.strip(),
               exit_code=done.returncode, seconds=round(secs, 1), stderr_tail=done.stderr[-3000:])
    if done.returncode != 0 or len(compiled) != 1:
        rec["status"] = "FAILED"
    else:
        c = compiled[0]
        files = sorted(p for p in c.rglob("*") if p.is_file())
        res = [p for p in files if p.name == "resources.bin"]
        rec.update(status="COMPILED", aimodelc=str(c), bytes=sum(p.stat().st_size for p in files),
                   resources_bin_bytes=sum(p.stat().st_size for p in res),
                   hash_files={p.name: p.read_bytes().hex() for p in sorted(c.glob("*.hash"))},
                   mlirb_sha256={p.name: sha256_file(p) for p in sorted(c.glob("*.mlirb"))},
                   ane=ane_regions(c))
        insp = subprocess.run([cb, "inspect", str(c), "--compute", "--json"], capture_output=True, text=True, env=env())
        try:
            rec["inspect_compute"] = json.loads(insp.stdout)
        except json.JSONDecodeError:
            rec["inspect_compute"] = dict(exit_code=insp.returncode, stdout=insp.stdout[-2000:], stderr=insp.stderr[-2000:])
    (a.out_root / f"{name}-{sfx}.aot.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("status", "exit_code", "seconds", "bytes", "resources_bin_bytes",
                                                "hash_files", "ane")}))
    print("AOT_DONE", rec["status"], name, a.preferred_compute)


if __name__ == "__main__":
    main()
