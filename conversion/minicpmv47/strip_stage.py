#!/usr/bin/env python3
"""Stage the three MiniCPM-V-4.7-1B ship bundles for Hugging Face, MLIR debug locations stripped; uploads nothing.
Per bundle: AIModelAsset.load -> strip_debug_info -> save_asset; the export's metadata.json and tokenizer/ are cloned.
Fatal unless op counts and `coreai-build inspect` equal the export's and no staged byte holds a local path.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import exports_dir, work_path  # noqa: E402

HOME = Path.home()
K = work_path("_minicpmv47")
EX = exports_dir()
ST = K / "stage_hf" / "gpu-pipelined"
R6 = K / "r6"
NAMES = ("minicpmv47_vlm_decode_int8hu", "minicpmv47_vision_int8lin", "minicpmv47_vision_fp16")
# <lane>/r3/FILES.md: main.mlirb sha256 of the r3 exports
R3_MLIRB = {
    "minicpmv47_vlm_decode_int8hu": "3b6ee67657cc1e9c886ec43f8001b12bae0e8c33e04d9d26b78237ad6c07d2ef",
    "minicpmv47_vision_int8lin": "94118d314de98445fcc3d28da563d3e4cea873f9c6af46982b8854b2a12b7a95",
    "minicpmv47_vision_fp16": "0950ecb567d4753d6db2a25033bd7511b004b51cba98d3e54226b14f2c2b69af",
}
NEEDLES = {"users_dir": b"/Users/", "private_dir": b"/private/", "tmp_claude": b"/tmp/claude",
           "home_dir": str(HOME).encode(), "user_name": HOME.name.encode()}
RECORDED = {".py": b".py", "site-packages": b"site-packages", "conversion/minicpmv47": b"conversion/minicpmv47"}


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(d: Path) -> list[dict]:
    return [{"path": str(p.relative_to(d)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(d.rglob("*")) if p.is_file()]


def scan(p: Path, needles: dict[str, bytes]) -> dict[str, int]:
    data = p.read_bytes()
    return {k: data.count(v) for k, v in needles.items()}


def tool_env() -> dict:
    return dict(os.environ)


def coreai_build() -> str:
    if not os.environ.get("DEVELOPER_DIR"):
        raise SystemExit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    out = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, env=tool_env(), check=True)
    if not out.stdout.strip():
        raise SystemExit("xcrun -f coreai-build resolved nothing")
    return out.stdout.strip()


def inspect(cb: str, p: Path) -> dict:
    done = subprocess.run([cb, "inspect", str(p), "--storage", "--compute", "--ops", "--json"], capture_output=True,
                          text=True, env=tool_env())
    if done.returncode != 0:
        raise SystemExit(f"coreai-build inspect {p} exit {done.returncode}: {done.stderr[-2000:]}")
    return json.loads(done.stdout)


def normalized(summary: dict) -> dict:
    def by_json(items):
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))

    out = dict(summary)
    for key in ("storageTypes", "operationDistribution"):
        if key in out:
            out[key] = by_json(out[key])
    if "computeTypes" in out:
        out["computeTypes"] = sorted(out["computeTypes"])
    if "functions" in out:
        out["functions"] = by_json([{**f, "inputs": by_json(f.get("inputs", [])),
                                     "outputs": by_json(f.get("outputs", []))} for f in out["functions"]])
    return out


def op_counts(program) -> Counter:
    ops = Counter()

    def walk(operation):
        ops[operation.name] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    walk(program._mlir_module.operation)
    return ops


def versions() -> dict:
    from importlib.metadata import version
    out = {"python": sys.version.split()[0]}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt", "torch", "numpy"):
        try:
            out[pkg] = version(pkg)
        except Exception as e:  # noqa: BLE001
            out[pkg] = f"? {e}"
    return out


def strip_one(name: str, cb: str) -> dict:
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    src = EX / name
    bundle_name = f"{name}.aimodel"
    bundle = src / bundle_name
    got = sha256_file(bundle / "main.mlirb")
    if got != R3_MLIRB[name]:
        raise SystemExit(f"{bundle}/main.mlirb sha256 {got} != r3 FILES.md {R3_MLIRB[name]}")
    out = ST / name
    if out.exists():
        raise SystemExit(f"{out} exists: an existing folder is never replaced")
    stage = ST / f".staging-{name}-{os.getpid()}"
    stage.mkdir(parents=True)
    started, t_all = now(), time.perf_counter()
    cloned = []
    for entry in sorted(src.iterdir()):
        if entry.name != bundle_name:
            subprocess.run(["cp", "-c", "-R", str(entry), str(stage / entry.name)], check=True)
            cloned.append(entry.name)

    t0 = time.perf_counter()
    asset = AIModelAsset.load(bundle)
    metadata = asset.metadata
    program = asset.program
    load_s = time.perf_counter() - t0
    before = op_counts(program)
    t0 = time.perf_counter()
    strip_debug_info(program)
    strip_s = time.perf_counter() - t0
    after = op_counts(program)
    target = stage / bundle_name
    if target.exists():
        raise SystemExit(f"{target} exists")
    t0 = time.perf_counter()
    program.save_asset(target, metadata)
    save_s = time.perf_counter() - t0
    del program, asset
    t0 = time.perf_counter()
    reloaded = op_counts(AIModelAsset.load(target).program)
    reload_s = time.perf_counter() - t0
    meta_src = json.loads((bundle / "metadata.json").read_text())
    meta_out = json.loads((target / "metadata.json").read_text())

    failures = []
    if not (before == after == reloaded):
        failures.append(f"ops: source {sum(before.values())}, stripped {sum(after.values())}, "
                        f"saved {sum(reloaded.values())}")
    insp_src, insp_out = inspect(cb, bundle), inspect(cb, target)
    insp_equal = normalized(insp_src["summary"]) == normalized(insp_out["summary"])
    if not insp_equal:
        failures.append("coreai-build inspect: the summaries differ")
    for key in ("author", "license", "description"):
        if meta_src.get(key) != meta_out.get(key):
            failures.append(f"metadata {key}: {meta_src.get(key)!r} -> {meta_out.get(key)!r}")
    hash_before = (bundle / "main.hash").read_bytes().hex()
    hash_after = (target / "main.hash").read_bytes().hex()
    mlirb_after = sha256_file(target / "main.mlirb")
    if hash_after != mlirb_after:
        failures.append("main.hash is not sha256(main.mlirb)")
    paths_after = {str(p.relative_to(stage)): scan(p, NEEDLES) for p in sorted(stage.rglob("*")) if p.is_file()}
    hits = {f: c for f, c in paths_after.items() if any(c.values())}
    if hits:
        failures.append(f"local paths left in the staged folder: {hits}")
    paths_before = {str(p.relative_to(src)): scan(p, NEEDLES) for p in sorted(src.rglob("*")) if p.is_file()}
    rec_before = {str(p.relative_to(src)): scan(p, RECORDED) for p in sorted(bundle.rglob("*")) if p.is_file()}
    rec_after = {str(p.relative_to(stage)): scan(p, RECORDED) for p in sorted(target.rglob("*")) if p.is_file()}
    record = {
        "schema": "minicpmv47-r6-strip/1", "status": "FAIL" if failures else "PASS", "failures": failures,
        "name": name, "started": started,
        "how": "AIModelAsset.load -> strip_debug_info(asset.program) (coreai_torch.debugging.debug_info) -> "
               "program.save_asset(out, the source's AIModelAssetMetadata); metadata.json and tokenizer/ cloned (cp -c)",
        "source": {"folder": str(src), "r3_files_md_main_mlirb_sha256": R3_MLIRB[name], "files": inventory(src),
                   "main_hash": hash_before, "local_paths": paths_before, "recorded_strings": rec_before,
                   "asset_metadata": meta_src},
        "staged": {"folder": str(out), "files": inventory(stage), "main_hash": hash_after,
                   "main_mlirb_sha256": mlirb_after, "local_paths": paths_after, "recorded_strings": rec_after,
                   "asset_metadata": meta_out},
        "ops": {"equal": before == after == reloaded, "source": sum(before.values()),
                "stripped": sum(after.values()), "saved_reloaded": sum(reloaded.values()),
                "op_counts": dict(sorted(before.items()))},
        "inspect": {"equal": insp_equal, "source_summary": insp_src.get("summary"),
                    "stripped_summary": insp_out.get("summary")},
        "needles": {k: v.decode() for k, v in NEEDLES.items()},
        "cloned": cloned,
        "seconds": {"load": load_s, "strip": strip_s, "save": save_s, "reload_and_count": reload_s,
                    "total": time.perf_counter() - t_all},
        "script_sha256": sha256_file(Path(__file__)),
        "versions": versions(),
        "coreai_build": cb,
    }
    R6.mkdir(parents=True, exist_ok=True)
    (R6 / f"strip_{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    if failures:
        raise SystemExit(f"{name}: {failures} (staging folder left at {stage})")
    os.rename(stage, out)
    print(json.dumps({"name": name, "status": "PASS", "main_hash": [hash_before[:16], hash_after[:16]],
                      "ops": record["ops"]["stripped"],
                      "paths_before": sum(sum(c.values()) for c in paths_before.values()),
                      "paths_after": sum(sum(c.values()) for c in paths_after.values()),
                      "bytes_after": sum(f["bytes"] for f in record["staged"]["files"]),
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}), flush=True)
    return record


def main() -> int:
    names = sys.argv[1:] or list(NAMES)
    for n in names:
        if n not in NAMES:
            raise SystemExit(f"unknown bundle {n}")
    cb = coreai_build()
    ST.mkdir(parents=True, exist_ok=True)
    for n in names:
        strip_one(n, cb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
