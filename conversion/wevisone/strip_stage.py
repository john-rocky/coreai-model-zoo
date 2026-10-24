#!/usr/bin/env python3
"""Stage the four WeVisOne-VL-2B graphs for Hugging Face (`ios/`), MLIR debug locations stripped; uploads nothing.
coreai-torch 0.4.1 records each op's PyTorch source location in the converted graph, so every exported main.mlirb
names the exporting machine's files (under the user's home directory, `…/torch/nn/modules/…`), and an AOT compiled
from it carries them too. Per graph: AIModelAsset.load -> strip_debug_info -> save_asset with the source's metadata (the method of
conversion/d1_omni/strip_ship.py); the decoder bundle's metadata.json and tokenizer/ are cloned (cp -c). Fatal unless
op counts (source, stripped, saved and loaded again) and `coreai-build inspect` equal the export's, main.hash is
sha256(main.mlirb), and no staged byte holds a local path. The record says whether each staged main.mlirb is the
published one (PUBLISHED_MLIRB) — a check of the export, not a condition of the staging.
Sources, as the recipes write them (models/wevisone-vl-2b/recipe.toml):
  <exports>/hidden/int8hu-pf64/wevisone_vl_2b_decode_int8hu_pf64/   mask/mask_hidden.py int8hu-pf64
  <exports>/wevisone_vl_2b_vision_taps_g14/                         mask/mask_vision_taps.py build --grid 14
  <exports>/wv_mask_a_g28_n16_fp16_hf/                              mask/mask_build.py a --grid 28 --host-fourier
  <exports>/wv_mask_b_fp16/                                         mask/mask_build.py b --precision fp16
Output = the repository's `ios/` folder; the macOS folder is compiled from it (README.md, "Stage").
    DEVELOPER_DIR=… <coreai venv>/bin/python -I -B strip_stage.py --out <stage>/ios --record <json> \
        [--exports <dir>] [name ...]
"""
from __future__ import annotations

import argparse
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
from _paths import exports_dir  # noqa: E402

HOME = Path.home()
DECODER = "wevisone_vl_2b_decode_int8hu_pf64"
# staged name -> (folder under <exports>, the asset inside it, whether the folder is a bundle with metadata/tokenizer)
GRAPHS = {
    DECODER: (Path("hidden/int8hu-pf64") / DECODER, f"{DECODER}.aimodel", True),
    "wevisone_vl_2b_vision_taps_g14": (Path("wevisone_vl_2b_vision_taps_g14"), "wevisone_vl_2b_vision_taps_g14.aimodel", False),
    "wv_mask_a_g28_n16_fp16_hf": (Path("wv_mask_a_g28_n16_fp16_hf"), "wv_mask_a_g28_n16_fp16_hf.aimodel", False),
    "wv_mask_b_fp16": (Path("wv_mask_b_fp16"), "wv_mask_b_fp16.aimodel", False),
}
# main.mlirb sha256 of the staged graphs that go to mlboydaisuke/WeVisOne-VL-2B-CoreAI `ios/` (2026-10-25)
PUBLISHED_MLIRB = {
    DECODER: "f62a162875ab9411a8d3c24d5173b91e2a76f7f244adf4d8c287026ea75bc94e",
    "wevisone_vl_2b_vision_taps_g14": "d8dd7a3f69c47cb95829f7f268d061704214548283eebbfb57acb5d914f56264",
    "wv_mask_a_g28_n16_fp16_hf": "ffcf2e67979851773849aa9d7d28cf2c1b006c45333375422e59ba7249288ac8",
    "wv_mask_b_fp16": "13bd5dfef16b7967fb3378727731d7a3930dc058b429e504c607f467a96ac4b2",
}
NEEDLES = {"homes_dir": (str(HOME.parent) + "/").encode(), "private_dir": b"/private/", "volumes_dir": b"/Volumes/",
           "var_folders": b"/var/folders/", "tmp_claude": b"/tmp/claude",
           "home_dir": str(HOME).encode(), "user_name": HOME.name.encode()}


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


def scan(p: Path) -> dict[str, int]:
    data = p.read_bytes()
    return {k: data.count(v) for k, v in NEEDLES.items()}


def coreai_build() -> str:
    if not os.environ.get("DEVELOPER_DIR"):
        raise SystemExit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    out = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, check=True)
    if not out.stdout.strip():
        raise SystemExit("xcrun -f coreai-build resolved nothing")
    return out.stdout.strip()


def inspect(cb: str, p: Path) -> dict:
    done = subprocess.run([cb, "inspect", str(p), "--storage", "--compute", "--ops", "--json"], capture_output=True,
                          text=True)
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


def strip_one(name: str, exports: Path, out_root: Path, cb: str) -> dict:
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    folder, asset_name, is_bundle = GRAPHS[name]
    src = exports / folder
    bundle = src / asset_name
    if not (bundle / "main.mlirb").is_file():
        raise SystemExit(f"{bundle}/main.mlirb not found: export it first (the recipe's script, README.md)")
    out = out_root / (name if is_bundle else asset_name)
    if out.exists():
        raise SystemExit(f"{out} exists: an existing folder is never replaced")
    stage = out_root / f".staging-{name}-{os.getpid()}"
    stage.mkdir(parents=True)
    started, t_all = now(), time.perf_counter()
    cloned = []
    if is_bundle:
        for entry in sorted(src.iterdir()):
            if entry.name != asset_name:
                subprocess.run(["cp", "-c", "-R", str(entry), str(stage / entry.name)], check=True)
                cloned.append(entry.name)
    target = stage / asset_name

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
    staged_root = stage if is_bundle else target
    paths_after = {str(p.relative_to(staged_root)): scan(p) for p in sorted(staged_root.rglob("*")) if p.is_file()}
    hits = {f: c for f, c in paths_after.items() if any(c.values())}
    if hits:
        failures.append(f"local paths left in the staged folder: {hits}")
    paths_before = {str(p.relative_to(src)): scan(p) for p in sorted(src.rglob("*")) if p.is_file()}
    published = PUBLISHED_MLIRB[name]
    record = {
        "schema": "wevisone-strip-stage/1", "status": "FAIL" if failures else "PASS", "failures": failures,
        "name": name, "started": started,
        "how": "AIModelAsset.load -> strip_debug_info(asset.program) (coreai_torch.debugging.debug_info) -> "
               "program.save_asset(out, the source's AIModelAssetMetadata); metadata.json and tokenizer/ cloned (cp -c)",
        "source": {"folder": str(src), "files": inventory(src), "main_hash": hash_before, "local_paths": paths_before,
                   "asset_metadata": meta_src},
        "staged": {"folder": str(out), "files": inventory(staged_root), "main_hash": hash_after,
                   "main_mlirb_sha256": mlirb_after, "local_paths": paths_after, "asset_metadata": meta_out,
                   "published_main_mlirb_sha256": published,
                   "same_as_published": mlirb_after == published},
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
    if failures:
        return record
    os.rename(staged_root, out)
    if not is_bundle:
        stage.rmdir()
    print(json.dumps({"name": name, "status": "PASS", "main_hash": [hash_before[:16], hash_after[:16]],
                      "same_as_published": record["staged"]["same_as_published"],
                      "ops": record["ops"]["stripped"],
                      "paths_before": sum(sum(c.values()) for c in paths_before.values()),
                      "paths_after": sum(sum(c.values()) for c in paths_after.values()),
                      "bytes_after": sum(f["bytes"] for f in record["staged"]["files"]),
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}), flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help=f"graphs to stage (default: all four: {', '.join(GRAPHS)})")
    ap.add_argument("--exports", default=None, help="the exports root (default: conversion/_paths.py exports_dir())")
    ap.add_argument("--out", required=True, help="the staged ios/ folder")
    ap.add_argument("--record", required=True, help="the record (json); never inside --out")
    args = ap.parse_args()
    names = args.names or list(GRAPHS)
    for n in names:
        if n not in GRAPHS:
            raise SystemExit(f"unknown graph {n} (one of {', '.join(GRAPHS)})")
    exports = Path(args.exports).expanduser().resolve() if args.exports else exports_dir()
    out_root = Path(args.out).expanduser().resolve()
    record_path = Path(args.record).expanduser().resolve()
    if record_path.is_relative_to(out_root):
        raise SystemExit(f"--record {record_path} is inside --out: the record names local paths")
    cb = coreai_build()
    out_root.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    records, failed = [], []
    for n in names:  # the record is rewritten after each graph, so a later stop keeps the earlier ones
        records.append(strip_one(n, exports, out_root, cb))
        failed = [r["name"] for r in records if r["status"] != "PASS"]
        summary = {"schema": "wevisone-strip-stage/1", "status": "FAIL" if failed else "PASS", "failures": failed,
                   "exports": str(exports), "out": str(out_root), "graphs": records}
        record_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print("FAIL" if failed else "PASS", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
