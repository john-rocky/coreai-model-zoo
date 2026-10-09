#!/usr/bin/env python3
"""Strip the MLIR debug locations from the ten shipping bundles into bundles/d1-omni-600m/macos-ship/<name>/.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/strip_ship.py                                   # the 10 ship bundles
    $PY conversion/d1_omni/strip_ship.py --only fp16-L256 vision-fp16
    $PY conversion/d1_omni/strip_ship.py --only fp16-L64 fp16-L128           # round 12 -> macos-ship-small/<name>/

Why. coreai-torch 0.4.1 records each op's PyTorch source location in the converted graph, so every exported main.mlirb
names the exporting machine's files (/Users/<user>/.../conversion/d1_omni/d1_omni_model.py and its folder) in its
bytecode string table, and so does every AOT .aimodelc compiled from it. `grep -rIl` skips binary files: a text-only
check reports 0.

How (knowledge/coreai-torch-041-ir-incident.md, "the in-place fix: strip_debug_info"): AIModelAsset.load(<bundle>)
-> .program -> coreai_torch.debugging.debug_info.strip_debug_info(program) (each op's location becomes an unknown
location carrying a fresh sequential operation id) -> program.save_asset(<out>, the source's AIModelAssetMetadata):
author, license and description are kept; save_asset stamps creationDate and producer again and writes main.hash =
sha256(main.mlirb). These bundles were converted with coreai-torch 0.4.1 / coreai-core 1.0.0b2, so the b2 reader
parses them (the incident note's coreai-core 1.0.0b1 detour is for 0.4.0-era assets).

Checks per bundle, into <out>/provenance/strip.json (all fatal):
  source   the bundle's files equal its export manifest (bytes, sha256)
  ops      the op counts by name of the source program, the stripped program and the saved bundle loaded again (walked
           as export_decide.py counts them) are equal (recorded beside them: equal to the export manifest's op_counts,
           which were counted on the converted program before it was saved)
  inspect  `coreai-build inspect --storage --compute --ops --json` of the source and of the stripped bundle: the same
           functions, storage types (element counts), compute types and op distribution
  meta     the stripped asset's author, license and description equal the source's; main.hash = sha256(main.mlirb)
  paths    no file of the stripped .aimodel holds "/Users/", the home directory or the user name (a byte scan: binary
           files included). The folder's other files are the lane's own records (they hold paths; the HF staging
           scrubs them): their counts are recorded, not checked
Recorded: every file of the bundle before / after (bytes, sha256), main.hash before / after, the seconds.

The rest of the source folder (metadata.json, tokenizer/, reference.json, position_table.f32 or
mel_filters_128x257_f32.bin, provenance/) is cloned unchanged (cp -c). The source folder is only read. Each output is
built in a staging folder beside it and renamed at the end; an existing output folder is never replaced. The set is
summarized in macos-ship/manifest.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

from aot_decide import coreai_build, disk_free, tool_env  # noqa: E402
from export_decide import environment, file_inventory, sha256_file, write_json  # noqa: E402

WORK = work_path("_d1_omni")
MACOS = WORK / "bundles" / "d1-omni-600m" / "macos"
SHIP_DIR = WORK / "bundles" / "d1-omni-600m" / "macos-ship"
SHIP = (*(f"fp16-L{L}" for L in (256, 512, 1024, 2048, 4096)), "vision-fp16",
        *(f"audio-fp16-{s}s" for s in (5, 10, 20, 30)))
# Round 12 (the chunk lever): the small decision buckets are stripped beside the ship set, not into it. The Swift host's
# D1Omni.folders(macos:) takes every fp16-L<L> folder of a directory, so an L64 / L128 folder in macos-ship/ would move
# the short rows of every Swift run over macos-ship/ (round 10's gate_swift.py --ship, round 11's Mac runs) to L64 / L128.
SMALL = tuple(f"fp16-L{L}" for L in (64, 128))
SMALL_DIR = WORK / "bundles" / "d1-omni-600m" / "macos-ship-small"


def out_root(name: str) -> Path:
    """The folder a stripped bundle goes to: macos-ship-small/ for round 12's small buckets, macos-ship/ otherwise."""
    return SMALL_DIR if name in SMALL else SHIP_DIR
NEEDLES = {"users_dir": b"/Users/", "home_dir": str(Path.home()).encode(), "user_name": Path.home().name.encode()}
RECORDED_NEEDLES = {".py": b".py", "site-packages": b"site-packages", "conversion/d1_omni": b"conversion/d1_omni"}


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def op_counts(program) -> Counter:
    """Every operation under the module (the module included), by name: export_decide.convert's walk."""
    ops = Counter()

    def walk(operation):
        ops[operation.name] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    walk(program._mlir_module.operation)
    return ops


def scan(path: Path, needles: dict[str, bytes]) -> dict[str, int]:
    data = path.read_bytes()
    return {k: data.count(v) for k, v in needles.items()}


def inspect(cb: str, path: Path) -> dict:
    done = subprocess.run([cb, "inspect", str(path), "--storage", "--compute", "--ops", "--json"], capture_output=True,
                          text=True, env=tool_env())
    if done.returncode != 0:
        raise SystemExit(f"coreai-build inspect {path} exit {done.returncode}: {done.stderr[-2000:]}")
    return json.loads(done.stdout)


def normalized(summary: dict) -> dict:
    """inspect's summary with every list in a fixed order (the tool does not keep one)."""
    def by_name(items):
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))

    out = {k: v for k, v in summary.items()}
    for key in ("storageTypes", "operationDistribution"):
        if key in out:
            out[key] = by_name(out[key])
    if "computeTypes" in out:
        out["computeTypes"] = sorted(out["computeTypes"])
    if "functions" in out:
        out["functions"] = by_name([{**f, "inputs": by_name(f.get("inputs", [])), "outputs": by_name(f.get("outputs", []))}
                                    for f in out["functions"]])
    return out


def strip_one(name: str, cb: str) -> dict:
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    src = MACOS / name
    manifest_path = src / "provenance" / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle_name = manifest["bundle"]
    bundle = src / bundle_name
    for item in manifest["files"]:
        f = bundle / item["path"]
        if f.stat().st_size != item["bytes"] or sha256_file(f) != item["sha256"]:
            raise SystemExit(f"{f} changed since its export")
    out = out_root(name) / name
    if out.exists():
        raise SystemExit(f"{out} exists: an existing folder is never replaced")
    stage = out_root(name) / f".staging-{name}-{os.getpid()}"
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
    if target.exists():  # save_asset removes an existing path first: never hand it one
        raise SystemExit(f"{target} exists")
    t0 = time.perf_counter()
    program.save_asset(target, metadata)
    save_s = time.perf_counter() - t0
    del program
    t0 = time.perf_counter()
    saved = AIModelAsset.load(target)
    reloaded = op_counts(saved.program)
    reload_s = time.perf_counter() - t0
    meta_src = json.loads((bundle / "metadata.json").read_text())
    meta_out = json.loads((target / "metadata.json").read_text())

    failures = []
    if not (before == after == reloaded):
        failures.append(f"ops: source {sum(before.values())}, stripped {sum(after.values())}, saved {sum(reloaded.values())}")
    inspect_src, inspect_out = inspect(cb, bundle), inspect(cb, target)
    inspect_equal = normalized(inspect_src["summary"]) == normalized(inspect_out["summary"])
    if not inspect_equal:
        failures.append("coreai-build inspect: the summaries differ")
    for key in ("author", "license", "description"):
        if meta_src.get(key) != meta_out.get(key):
            failures.append(f"metadata {key}: {meta_src.get(key)!r} -> {meta_out.get(key)!r}")
    hash_before = (bundle / "main.hash").read_bytes().hex()
    hash_after = (target / "main.hash").read_bytes().hex()
    if hash_after != sha256_file(target / "main.mlirb"):
        failures.append("main.hash is not sha256(main.mlirb)")
    paths_bundle = {str(p.relative_to(stage)): scan(p, NEEDLES) for p in sorted(target.rglob("*")) if p.is_file()}
    hits = {f: c for f, c in paths_bundle.items() if any(c.values())}
    if hits:
        failures.append(f"local paths left in the stripped bundle: {hits}")
    paths_source = {str(p.relative_to(src)): scan(p, NEEDLES) for p in sorted(bundle.rglob("*")) if p.is_file()}
    recorded_source = {str(p.relative_to(src)): scan(p, RECORDED_NEEDLES) for p in sorted(bundle.rglob("*")) if p.is_file()}
    recorded_out = {str(p.relative_to(stage)): scan(p, RECORDED_NEEDLES) for p in sorted(target.rglob("*")) if p.is_file()}
    others = {str(p.relative_to(stage)): scan(p, NEEDLES) for p in sorted(stage.rglob("*"))
              if p.is_file() and target not in p.parents}

    files_before, files_after = file_inventory(bundle), file_inventory(target)
    record = {
        "schema": "d1-omni-strip/1", "status": "FAIL" if failures else "PASS", "failures": failures,
        "name": name, "bundle": bundle_name, "started": started,
        "source": {"folder": str(src), "export_manifest_sha256": sha256_file(manifest_path),
                   "export_status": manifest["status"], "precision": manifest["precision"]},
        "how": "AIModelAsset.load -> strip_debug_info(asset.program) (coreai_torch.debugging.debug_info, coreai-torch "
               "0.4.1) -> program.save_asset(out, the source's AIModelAssetMetadata)",
        "before": {"bytes": sum(f["bytes"] for f in files_before), "main_hash": hash_before, "files": files_before,
                   "ops_total": sum(before.values()), "local_paths": paths_source, "recorded_strings": recorded_source,
                   "metadata": meta_src},
        "after": {"bytes": sum(f["bytes"] for f in files_after), "main_hash": hash_after, "files": files_after,
                  "ops_total": sum(after.values()), "local_paths": paths_bundle, "recorded_strings": recorded_out,
                  "metadata": meta_out},
        "ops": {"equal": before == after == reloaded, "equal_export_manifest": dict(before) == manifest["op_counts"],
                "source": sum(before.values()), "stripped": sum(after.values()), "saved_reloaded": sum(reloaded.values()),
                "export_manifest": manifest["ops_total"], "op_counts": dict(sorted(before.items()))},
        "inspect": {"equal": inspect_equal, "source": inspect_src, "stripped": inspect_out},
        "local_path_needles": {k: v.decode() for k, v in NEEDLES.items()},
        "other_files_local_paths": others,
        "other_files_note": "cloned unchanged from the source folder (cp -c): the lane's records, which hold the "
                            "lane's paths; the HF staging scrubs what it stages",
        "cloned": cloned,
        "seconds": {"load": load_s, "strip": strip_s, "save": save_s, "reload_and_count": reload_s,
                    "total": time.perf_counter() - t_all},
        "code_sha256": {f: sha256_file(HERE / f) for f in ("strip_ship.py",)},
        "environment": environment(("coreai-torch", "coreai-core", "torch")),
    }
    write_json(stage / "provenance" / "strip.json", record)
    if failures:
        raise SystemExit(f"{name}: {failures} (staging folder left at {stage})")
    os.rename(stage, out)
    print(json.dumps({"name": name, "bytes": [record["before"]["bytes"], record["after"]["bytes"]],
                      "main_hash": [hash_before[:16], hash_after[:16]], "ops": record["ops"]["stripped"],
                      "local_paths_after": sum(sum(c.values()) for c in paths_bundle.values()),
                      "local_paths_before": sum(sum(c.values()) for c in paths_source.values()),
                      "seconds": {k: round(v, 1) for k, v in record["seconds"].items()}}), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=SHIP + SMALL,
                        help="a subset of the ship bundles, or round 12's small buckets (-> macos-ship-small/)")
    args = parser.parse_args()
    names = args.only or list(SHIP)
    roots = {out_root(name) for name in names}
    if len(roots) != 1:
        parser.error("the ship bundles and round 12's small buckets go to different folders: one set per run")
    root = roots.pop()
    root.mkdir(parents=True, exist_ok=True)
    cb = coreai_build()
    started, before = now(), disk_free()
    t0 = time.perf_counter()
    records = {name: strip_one(name, cb) for name in names}
    summary_path = root / "manifest.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {"bundles": {}}
    bundles = previous["bundles"]
    for name, r in records.items():
        if name in bundles:
            raise SystemExit(f"{name} is already in {summary_path}")
        bundles[name] = {"status": r["status"], "bundle": r["bundle"], "source": str((MACOS / name).relative_to(WORK)),
                         "bytes_before": r["before"]["bytes"], "bytes_after": r["after"]["bytes"],
                         "main_hash_before": r["before"]["main_hash"], "main_hash_after": r["after"]["main_hash"],
                         "ops": r["ops"]["stripped"], "ops_equal": r["ops"]["equal"], "inspect_equal": r["inspect"]["equal"],
                         "local_paths_before": sum(sum(c.values()) for c in r["before"]["local_paths"].values()),
                         "local_paths_after": sum(sum(c.values()) for c in r["after"]["local_paths"].values()),
                         "record": str((root / name / "provenance" / "strip.json").relative_to(WORK))}
    what = ("the ten shipping bundles of macos/ with their MLIR debug locations stripped (strip_ship.py)" if root == SHIP_DIR
            else "round 12's small decision buckets (fp16 L64 / L128) of macos/ with their MLIR debug locations stripped "
                 "(strip_ship.py), kept apart from macos-ship/ (the Swift host's folder scan)")
    summary = {"schema": "d1-omni-macos-ship/1", "what": what,
               "runs": previous.get("runs", []) + [{"started": started, "seconds": time.perf_counter() - t0,
                                                     "names": names, "disk_free_before": before,
                                                     "disk_free_after": disk_free()}],
               "bundles": bundles}
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(root), "stripped": len(records),
                      "bytes_after": sum(r["after"]["bytes"] for r in records.values())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
