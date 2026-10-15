#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "coreai-torch==0.4.1",
#     "coreai-core==1.0.0b2",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Strip the MLIR debug locations from the shipping bundles, compile the stripped ones for the iPhone, compare.

    PY=<venv>/bin/python
    $PY conversion/eg2_transcribe/strip_ship.py strip     # gated .aimodel -> <exports>/ship/macos/<name>.aimodel
    $PY conversion/eg2_transcribe/strip_ship.py aot       # <exports>/ship/macos -> <exports>/ship/ios-h19p/<name>.aimodelc
    $PY conversion/eg2_transcribe/strip_ship.py compare   # Mac runtime: each gated bundle vs its stripped copy

    # reader v2 alone: strip, the three AOT compiles of the stripped copy, compare JIT and AOT on the Mac
    $PY conversion/eg2_transcribe/strip_ship.py strip --only eg2_reader_v2_fp16_T750
    $PY conversion/eg2_transcribe/strip_ship.py aot --only eg2_reader_v2_fp16_T750 --target h19p h16c-gpu h16c-ane
    $PY conversion/eg2_transcribe/strip_ship.py compare --only eg2_reader_v2_fp16_T750 --aot --record compare_mac_reader_v2.json

The shipping bundles are the gated ones of the speed ladder: the fp16 encoder `fp16/eg2_audio_fp16_T3000.aimodel`
(rows 1, 27, 27a), the fp16 reader + word table of reader v1 `reader/fp16/eg2_reader_fp16_T750.aimodel` (row 53), and
the same graph with reader v2's six-layer body and weights `reader_v2/fp16/eg2_reader_v2_fp16_T750.aimodel` (the reader
v2 rows); `--only` picks some of them.

Why. coreai-torch 0.4.1 records each op's PyTorch source location in the converted graph, so an exported main.mlirb
names the exporting machine's files (eg2_audio_coreai.py / reader_coreai.py and torch's nn/modules, as absolute
paths) in its bytecode string table, and an AOT .aimodelc compiled from it does too (main-<arch>.mlirb and the MPSGraph
package). `grep -rIl` skips binary files and reports 0; `grep -rl -a` finds them.

strip (no inference): AIModelAsset.load(bundle).program, one program object, -> coreai_torch's strip_debug_info (every
op's location becomes an unknown location) -> the same object's save_asset(out, the source's metadata). Calling
`asset.program` again returns another object, which would save the unstripped graph. Checks, all fatal, recorded in
<ship>/provenance/strip_<name>.json: the op counts by name of the source, the stripped and the saved-then-reloaded
program are equal; the `coreai-build inspect --storage --compute --ops --json` summaries of source and copy are equal;
author, license and description are kept; main.hash = sha256(main.mlirb) and differs from the source's (a new Core
AI cache entry, so a load never returns the source's specialization); no file of the copy holds "/Users/", the home
directory or the user name (a byte scan, binary files included).

aot (--target, default h19p): `coreai-build compile <copy>.aimodel --platform iOS --preferred-compute gpu
--architecture h19p`, the flags of the phone-gated bundles (ladder rows 27 and 80), into ios-h19p/<name>.aimodelc;
h16c-gpu / h16c-ane compile the copy for this Mac (`--platform macOS --architecture h16c`, GPU or Neural Engine
preferred, the flags of export_*.py --aot) into macos-h16c/<name>_<gpu|ane>_h16c.aimodelc, so the stripped bytes can be
compared with the gated AOT bundles the gates and timings ran. Checks, all fatal: exit 0; the compiled metadata's
sourceHash is the copy's sha256(main.mlirb); a GPU compile has no Neural Engine region, a Neural Engine compile has
as many as the gated compile of the unstripped source; the same byte scan over every compiled file. Recorded beside
them: which files equal those of the gated compile of the unstripped source for the same target (the weights,
resources.bin, are expected to). An h19p bundle is never loaded on a Mac (it can wedge the GPU stack): this step only
reads its files; the phone runs it.

compare (the Core AI runtime; run it alone on the machine): each gated bundle and its copy as JIT .aimodel on this
Mac, the 20 fixture clips: the encoder with the GPU preferred (soft tokens of features + half_mask), the reader with
the GPU preferred and with the Neural Engine preferred (top1_index, top1_logit, blank_logit of the clips' reference
soft tokens). With --aot, a reader also compares its stripped h16c compiles with the gated ones (GPU, Neural Engine)
and its stripped JIT copy with the gated AOT bundles (the bundles the gates and timings ran). PASS = every output array
equal bit for bit on every clip, and two controls that must differ: the comparison of each clip with the next clip's
output, and the encoder's output after one input value moved by one float16 step. Recorded with each bundle's
main.hash, its Core AI cache entry and the load seconds.

Nothing is replaced: an output bundle or record that exists stops the run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import exports_dir, fixtures_dir, load_clips, sha256_of, window_input, write_json  # noqa: E402

SOURCES = {"eg2_audio_fp16_T3000": Path("fp16") / "eg2_audio_fp16_T3000.aimodel",
           "eg2_reader_fp16_T750": Path("reader") / "fp16" / "eg2_reader_fp16_T750.aimodel",
           "eg2_reader_v2_fp16_T750": Path("reader_v2") / "fp16" / "eg2_reader_v2_fp16_T750.aimodel"}
SHIPPING = ["eg2_audio_fp16_T3000", "eg2_reader_v2_fp16_T750"]     # the bundles the app downloads (reader v2)
# The h19p compiles of the unstripped sources: the encoder's ran on the phone (ladder row 27); the readers' compiles
# are the references for their stripped twins (the phone's reader row 80 ran v1's graph with interim weights).
GATED_AOT = {"eg2_audio_fp16_T3000": Path("fp16") / "eg2_audio_fp16_T3000_gpu_h19p.aimodelc",
             "eg2_reader_fp16_T750": Path("reader") / "fp16" / "eg2_reader_fp16_T750_gpu_h19p.aimodelc",
             "eg2_reader_v2_fp16_T750": Path("reader_v2") / "fp16" / "eg2_reader_v2_fp16_T750_gpu_h19p.aimodelc"}
AOT_ARGS = ["--platform", "iOS", "--preferred-compute", "gpu", "--architecture", "h19p"]
# target -> (folder under <ship>, name suffix, coreai-build flags, Neural Engine compile)
AOT_TARGETS = {
    "h19p": ("ios-h19p", "", AOT_ARGS, False),
    "h16c-gpu": ("macos-h16c", "_gpu_h16c", ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c"], False),
    "h16c-ane": ("macos-h16c", "_ane_h16c",
                 ["--platform", "macOS", "--preferred-compute", "neural-engine", "--architecture", "h16c"], True),
}


def gated_compile(name: str, target: str) -> Path:
    """The compile of the unstripped (gated) source for a target, relative to the exports directory."""
    if target == "h19p":
        return GATED_AOT[name]
    src = SOURCES[name]
    return src.with_name(src.name.replace(".aimodel", AOT_TARGETS[target][1] + ".aimodelc"))
NEEDLES = {"users_dir": b"/Users/", "home_dir": str(Path.home()).encode(), "user_name": Path.home().name.encode()}
ENCODER_FRAMES, READER_TOKENS = 3000, 750


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def ship_dir(args) -> Path:
    return Path(args.out).expanduser() if args.out else exports_dir(args.exports) / "ship"


def file_inventory(bundle: Path) -> list[dict]:
    return [{"path": str(p.relative_to(bundle)), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
            for p in sorted(bundle.rglob("*")) if p.is_file()]


def scan(bundle: Path) -> dict[str, dict[str, int]]:
    """Per file of the bundle: how often each local-path needle occurs in its bytes (binary files included)."""
    out = {}
    for p in sorted(bundle.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            out[str(p.relative_to(bundle))] = {k: data.count(v) for k, v in NEEDLES.items()}
    return out


def hits(scanned: dict[str, dict[str, int]]) -> int:
    return sum(sum(c.values()) for c in scanned.values())


def op_counts(program) -> Counter:
    counts: Counter = Counter()

    def walk(op):
        counts[op.name] += 1
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)

    walk(program._mlir_module.operation)
    return counts


def tool_env() -> dict:
    env = dict(os.environ)
    if "DEVELOPER_DIR" not in env:
        xcodes = sorted(Path("/Applications").glob("Xcode-27*.app"))
        if xcodes:
            env["DEVELOPER_DIR"] = str(xcodes[-1] / "Contents" / "Developer")
    return env


def coreai_build() -> tuple[str, str]:
    env = tool_env()
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True, check=True,
                        env=env).stdout.strip()
    ver = subprocess.run([cb, "--version"], capture_output=True, text=True, env=env)
    return cb, (ver.stdout + ver.stderr).strip().splitlines()[0] if (ver.stdout + ver.stderr).strip() else ""


def inspect_summary(cb: str, bundle: Path) -> dict:
    done = subprocess.run([cb, "inspect", str(bundle), "--storage", "--compute", "--ops", "--json"],
                          capture_output=True, text=True, env=tool_env())
    if done.returncode != 0:
        raise SystemExit(f"coreai-build inspect {bundle.name} exit {done.returncode}: {done.stderr[-2000:]}")
    return json.loads(done.stdout)["summary"]


def normalized(summary: dict) -> dict:
    """inspect's summary with every list in a fixed order (the tool does not keep one)."""
    def ordered(items):
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))

    out = dict(summary)
    for key in ("storageTypes", "operationDistribution"):
        if key in out:
            out[key] = ordered(out[key])
    if "computeTypes" in out:
        out["computeTypes"] = sorted(out["computeTypes"])
    if "functions" in out:
        out["functions"] = ordered([{**f, "inputs": ordered(f.get("inputs", [])), "outputs": ordered(f.get("outputs", []))}
                                    for f in out["functions"]])
    return out


def environment(*packages: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    env = {"python": platform.python_version(),
           "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()}
    for name in packages:
        try:
            env[name] = version(name)
        except PackageNotFoundError:
            env[name] = None
    return env


def refuse_existing(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"{p} exists: an existing output is never replaced")


# ---------------------------------------------------------------- strip

def strip_one(name: str, exports: Path, ship: Path, cb: str) -> dict:
    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    src = exports / SOURCES[name]
    target = ship / "macos" / f"{name}.aimodel"
    record_path = ship / "provenance" / f"strip_{name}.json"
    refuse_existing(target, record_path)
    stage_dir = target.parent / f".staging-{name}-{os.getpid()}"
    staged = stage_dir / target.name
    stage_dir.mkdir(parents=True)
    started, t_all = now(), time.perf_counter()

    t0 = time.perf_counter()
    asset = AIModelAsset.load(src)
    metadata = asset.metadata
    program = asset.program
    load_s = time.perf_counter() - t0
    before = op_counts(program)
    t0 = time.perf_counter()
    strip_debug_info(program)
    strip_s = time.perf_counter() - t0
    after = op_counts(program)
    t0 = time.perf_counter()
    program.save_asset(staged, metadata)
    save_s = time.perf_counter() - t0
    del program, asset
    reloaded = op_counts(AIModelAsset.load(staged).program)

    failures = []
    if not before == after == reloaded:
        failures.append(f"ops: source {sum(before.values())}, stripped {sum(after.values())}, "
                        f"saved {sum(reloaded.values())}")
    inspect_src, inspect_out = inspect_summary(cb, src), inspect_summary(cb, staged)
    inspect_equal = normalized(inspect_src) == normalized(inspect_out)
    if not inspect_equal:
        failures.append("coreai-build inspect: the summaries differ")
    meta_src = json.loads((src / "metadata.json").read_text())
    meta_out = json.loads((staged / "metadata.json").read_text())
    for key in ("author", "license", "description"):
        if meta_src.get(key) != meta_out.get(key):
            failures.append(f"metadata {key}: {meta_src.get(key)!r} -> {meta_out.get(key)!r}")
    hash_src = (src / "main.hash").read_bytes().hex()
    hash_out = (staged / "main.hash").read_bytes().hex()
    if hash_out != sha256_of(staged / "main.mlirb"):
        failures.append("main.hash is not sha256(main.mlirb)")
    if hash_out == hash_src:
        failures.append("main.hash equals the source's: a load would reuse the source's cache entry")
    scanned_src, scanned_out = scan(src), scan(staged)
    if hits(scanned_out):
        failures.append(f"local paths left in the stripped bundle: {hits(scanned_out)}")

    record = {
        "schema": "eg2-transcribe-strip/1", "status": "FAIL" if failures else "PASS", "failures": failures,
        "name": name, "started": started,
        "source": {"bundle": str(SOURCES[name]), "main_hash": hash_src, "files": file_inventory(src),
                   "local_paths": scanned_src, "op_total": sum(before.values()), "metadata": meta_src},
        "how": "AIModelAsset.load(source).program -> coreai_torch.debugging.debug_info.strip_debug_info(program) -> "
               "program.save_asset(out, the source's AIModelAssetMetadata); one program object throughout",
        "after": {"bundle": f"macos/{target.name}", "main_hash": hash_out, "files": file_inventory(staged),
                  "local_paths": scanned_out, "op_total": sum(after.values()), "metadata": meta_out},
        "ops": {"equal": before == after == reloaded, "source": sum(before.values()), "stripped": sum(after.values()),
                "saved_reloaded": sum(reloaded.values()), "op_counts": dict(sorted(before.items()))},
        "inspect": {"equal": inspect_equal, "source": inspect_src, "stripped": inspect_out},
        "local_path_needles": sorted(NEEDLES),
        "seconds": {"load": round(load_s, 2), "strip": round(strip_s, 2), "save": round(save_s, 2),
                    "total": round(time.perf_counter() - t_all, 2)},
        "code_sha256": {"strip_ship.py": sha256_of(HERE / "strip_ship.py")},
        "environment": {**environment("coreai-core", "coreai-torch", "torch", "numpy"), "coreai_build": coreai_build()[1]},
    }
    write_json(record_path, record)
    if failures:
        raise SystemExit(f"{name}: {failures} (the copy is left in {stage_dir})")
    os.rename(staged, target)
    stage_dir.rmdir()
    print(json.dumps({"strip": name, "status": record["status"], "main_hash": [hash_src[:16], hash_out[:16]],
                      "ops": record["ops"]["stripped"], "local_paths": [hits(scanned_src), hits(scanned_out)],
                      "seconds": record["seconds"]}), flush=True)
    return record


def cmd_strip(args) -> int:
    exports, ship = exports_dir(args.exports), ship_dir(args)
    cb, _ = coreai_build()
    for name in args.only or SHIPPING:
        strip_one(name, exports, ship, cb)
    return 0


# ---------------------------------------------------------------- aot

def ane_region_names(bundle: Path) -> list[str]:
    """Unique Neural Engine region names in a compiled bundle (a region shows up as a directory and nested files)."""
    import re
    return sorted({re.sub(r"(_ANE_region_\d+).*$", r"\1", p.name) for p in bundle.rglob("*") if "_ANE_region_" in p.name})


def aot_one(name: str, exports: Path, ship: Path, cb: str, cb_version: str, target_kind: str = "h19p") -> dict:
    folder, suffix, flags, neural_engine = AOT_TARGETS[target_kind]
    src = ship / "macos" / f"{name}.aimodel"
    strip_rec = json.loads((ship / "provenance" / f"strip_{name}.json").read_text())
    if strip_rec["status"] != "PASS" or file_inventory(src) != strip_rec["after"]["files"]:
        raise SystemExit(f"{src} is not the copy its strip record describes")
    target = ship / folder / f"{name}{suffix}.aimodelc"
    record_path = ship / "provenance" / f"aot_{target_kind}_{name}.json"
    refuse_existing(target, record_path)
    stage_dir = target.parent / f".staging-{name}-{os.getpid()}"
    staged = stage_dir / target.name
    stage_dir.mkdir(parents=True)
    argv = [cb, "compile", str(src), "--output", str(staged), *flags]
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=tool_env())
    seconds = time.perf_counter() - t0
    log = (done.stdout + done.stderr).strip().splitlines()

    failures = []
    if done.returncode != 0 or not staged.exists():
        failures.append(f"coreai-build exit {done.returncode}")
    files, scanned, regions, source_hash, gated = [], {}, [], None, None
    ref = exports / gated_compile(name, target_kind)
    ref_regions = ane_region_names(ref) if ref.exists() else None
    if staged.exists():
        files, scanned = file_inventory(staged), scan(staged)
        regions = ane_region_names(staged)
        source_hash = json.loads((staged / "metadata.json").read_text()).get("sourceHash", "")
        if source_hash.lower() != sha256_of(src / "main.mlirb"):
            failures.append(f"sourceHash {source_hash[:16]} is not the stripped main.mlirb's sha256")
        if not neural_engine and regions:
            failures.append(f"{len(regions)} Neural Engine region names in a GPU compile")
        if neural_engine and ref_regions is not None and len(regions) != len(ref_regions):
            failures.append(f"{len(regions)} Neural Engine regions; the gated compile has {len(ref_regions)}")
        if hits(scanned):
            failures.append(f"local paths in the compiled bundle: {hits(scanned)}")
        if ref.exists():
            ref_files = {f["path"]: f["sha256"] for f in file_inventory(ref)}
            gated = {"bundle": str(gated_compile(name, target_kind)), "ane_regions": len(ref_regions),
                     "files": {f["path"]: ("equal" if ref_files.get(f["path"]) == f["sha256"] else
                                           "differs" if f["path"] in ref_files else "absent there")
                               for f in files}}
    record = {
        "schema": "eg2-transcribe-aot/1", "status": "FAIL" if failures else "PASS", "failures": failures,
        "name": name, "target": target_kind, "started": now(),
        "source": {"bundle": f"macos/{src.name}", "main_mlirb_sha256": sha256_of(src / "main.mlirb")},
        "command": ["coreai-build", "compile", f"macos/{src.name}", "--output", f"{folder}/{target.name}", *flags],
        "coreai_build": cb_version, "returncode": done.returncode, "seconds": round(seconds, 1), "log_tail": log[-15:],
        "after": {"bundle": f"{folder}/{target.name}", "files": files, "bytes": sum(f["bytes"] for f in files),
                  "source_hash": source_hash, "ane_region_names": regions, "local_paths": scanned},
        f"vs_unstripped_{target_kind}_compile": gated,
        "note": ("never loaded on this Mac; the phone runs this bundle" if target_kind == "h19p" else
                 "this Mac's compile of the stripped copy, compared with the gated one by `compare --aot`"),
        "code_sha256": {"strip_ship.py": sha256_of(HERE / "strip_ship.py")},
        "environment": environment(),
    }
    write_json(record_path, record)
    if failures:
        raise SystemExit(f"{name} {target_kind}: {failures} (the compile is left in {stage_dir})")
    os.rename(staged, target)
    stage_dir.rmdir()
    print(json.dumps({"aot": name, "target": target_kind, "status": record["status"], "seconds": record["seconds"],
                      "bytes": record["after"]["bytes"], "ane_regions": len(regions), "local_paths": hits(scanned),
                      "vs_unstripped": gated and Counter(gated["files"].values())}), flush=True)
    return record


def cmd_aot(args) -> int:
    exports, ship = exports_dir(args.exports), ship_dir(args)
    cb, ver = coreai_build()
    for name in args.only or SHIPPING:
        for target_kind in args.target:
            aot_one(name, exports, ship, cb, ver, target_kind)
    return 0


# ---------------------------------------------------------------- compare

def half_mask(n_valid: int, frames: int) -> np.ndarray:
    m = np.zeros((1, (frames + 1) // 2), dtype=np.float16)
    m[0, : (n_valid + 1) // 2] = 1
    return m


def encoder_feeds(clips: list[dict]) -> list[dict]:
    feeds = []
    for c in clips:
        x, nv = window_input(c["features"], ENCODER_FRAMES)
        feeds.append({"features": np.ascontiguousarray(x[None].astype(np.float16)),
                      "half_mask": np.ascontiguousarray(half_mask(nv, ENCODER_FRAMES))})
    return feeds


def reader_feeds(clips: list[dict]) -> list[dict]:
    from reader_coreai import pieces, window
    feeds = []
    for c in clips:
        for a, b in pieces(c["ref"].shape[0], READER_TOKENS):
            x, m = window(c["ref"], a, b, READER_TOKENS, np.float16)
            feeds.append({"soft_tokens": np.ascontiguousarray(x), "token_mask": np.ascontiguousarray(m)})
    return feeds


def cache_entry(bundle: Path) -> str:
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return f"coreai-cache/{build}/{Path(sys.executable).name}/{(bundle / 'main.hash').read_bytes().hex()}"


async def run_bundle(rt, bundle: Path, backend: str, feeds: list[dict]) -> tuple[list[dict], dict]:
    kind = rt.ComputeUnitKind.gpu() if backend == "gpu" else rt.ComputeUnitKind.neural_engine()
    t0 = time.perf_counter()
    model = await rt.AIModel.load(bundle, rt.SpecializationOptions.from_preferred_compute_unit_kind(kind))
    fn = model.load_function("main")
    load_s = time.perf_counter() - t0
    outs, secs = [], []
    for feed in feeds:                       # one call at a time: two calls of one function at once corrupt outputs
        t0 = time.perf_counter()
        res = await asyncio.wait_for(fn({k: rt.NDArray(v) for k, v in feed.items()}), timeout=900)
        secs.append(time.perf_counter() - t0)
        outs.append({k: res[k].numpy().copy() for k in fn.desc.output_names})
    entry = cache_entry(bundle)
    info = {"main_hash": (bundle / "main.hash").read_bytes().hex(), "cache_entry": entry,
            "cache_entry_exists": (Path.home() / "Library" / "Caches" / entry).exists(),
            "load_s": round(load_s, 2), "first_call_s": round(secs[0], 4),
            "median_call_ms": round(float(np.median(secs)) * 1000, 3)}
    del fn, model
    return outs, info


def diff(a: dict, b: dict) -> dict:
    return {k: {"equal": bool(np.array_equal(a[k], b[k])),
                "max_abs": float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())} for k in a}


async def compare_main(args) -> int:
    import coreai.runtime as rt

    exports, ship = exports_dir(args.exports), ship_dir(args)
    record_path = ship / "provenance" / args.record
    refuse_existing(record_path)
    clips = load_clips(fixtures_dir(args.fixtures))
    names = args.only or SHIPPING
    feeds = {n: encoder_feeds(clips) if n.startswith("eg2_audio") else reader_feeds(clips) for n in names}
    # (name, backend, label, gated bundle, stripped bundle): the JIT pairs; with --aot a reader also pairs its h16c
    # compiles (gated vs stripped) and its stripped JIT copy with the gated AOT bundles the gates and timings ran
    plan = []
    for name in names:
        jit_g, jit_s = exports / SOURCES[name], ship / "macos" / f"{name}.aimodel"
        for backend in (("gpu",) if name.startswith("eg2_audio") else ("gpu", "ane")):
            plan.append((name, backend, f"JIT, {backend} preferred", jit_g, jit_s))
        if args.aot and not name.startswith("eg2_audio"):
            for backend in ("gpu", "ane"):
                kind = f"h16c-{backend}"
                aot_g = exports / gated_compile(name, kind)
                aot_s = ship / AOT_TARGETS[kind][0] / f"{name}{AOT_TARGETS[kind][1]}.aimodelc"
                plan.append((name, backend, f"AOT h16c {backend} (gated) vs AOT h16c {backend} (stripped)", aot_g, aot_s))
                plan.append((name, backend, f"AOT h16c {backend} (gated) vs JIT {backend} preferred (stripped)", aot_g, jit_s))
    runs: dict = {}

    async def outputs(bundle: Path, backend: str, name: str):
        key = (str(bundle), backend)
        if key not in runs:                # each bundle is loaded once per compute preference
            runs[key] = await run_bundle(rt, bundle, backend, feeds[name])
        return runs[key]

    pairs, started = [], now()
    for name, backend, label, gated_b, stripped_b in plan:
        out_g, info_g = await outputs(gated_b, backend, name)
        out_s, info_s = await outputs(stripped_b, backend, name)
        per_call = [diff(g, s) for g, s in zip(out_g, out_s)]
        shifted = [diff(g, s) for g, s in zip(out_g[:-1], out_s[1:])]
        row = {"bundle": name, "backend": label, "calls": len(per_call),
               "gated_bundle": str(gated_b.relative_to(exports)), "stripped_bundle": str(stripped_b.relative_to(ship)),
               "equal_calls": sum(all(v["equal"] for v in d.values()) for d in per_call),
               "max_abs": {k: max(d[k]["max_abs"] for d in per_call) for k in per_call[0]},
               "control_next_clip_differs": sum(not all(v["equal"] for v in d.values()) for d in shifted),
               "control_calls": len(shifted), "gated": info_g, "stripped": info_s}
        if name == "eg2_audio_fp16_T3000":
            # one input value moved by one float16 step must move the output: bit equality above is not vacuous
            nudged = {k: v.copy() for k, v in feeds[name][0].items()}
            f = nudged["features"]
            f[0, 100, 5] = np.nextafter(f[0, 100, 5], np.float16(np.inf))
            out_n, _ = await run_bundle(rt, stripped_b, backend, [nudged])
            row["control_one_step_differs"] = not all(v["equal"] for v in diff(out_s[0], out_n[0]).values())
        row["status"] = ("SAME" if row["equal_calls"] == row["calls"]
                         and row["control_next_clip_differs"] == row["control_calls"]
                         and row.get("control_one_step_differs", True) else "DIFFERS")
        pairs.append(row)
        print(json.dumps({k: row[k] for k in ("bundle", "backend", "calls", "equal_calls", "max_abs", "status")}),
              flush=True)
    status = "SAME" if all(p["status"] == "SAME" for p in pairs) else "DIFFERS"
    record = {"schema": "eg2-transcribe-strip-compare/1", "status": status, "started": started, "finished": now(),
              "what": "each gated bundle and its stripped copy, JIT .aimodel on this Mac, the 20 fixture clips "
                      "(encoder: features + half_mask; reader: the clips' reference soft tokens in 750-token windows)"
                      + ("; with --aot the readers' h16c AOT compiles too" if args.aot else ""),
              "bundles": names, "label": args.label, "pairs": pairs,
              "code_sha256": {"strip_ship.py": sha256_of(HERE / "strip_ship.py")},
              "environment": environment("coreai-core", "numpy")}
    write_json(record_path, record)
    print(status, "->", record_path.relative_to(ship), flush=True)
    return 0 if status == "SAME" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--exports", help="the exports directory (default: <work>/exports, _common.py)")
    ap.add_argument("--out", help="the ship directory (default: <exports>/ship)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, text in (("strip", "strip the gated .aimodel bundles into <ship>/macos/"),
                      ("aot", "compile the stripped bundles: for the iPhone 18 Pro (h19p, GPU) into <ship>/ios-h19p/, "
                              "for this Mac (h16c) into <ship>/macos-h16c/")):
        p = sub.add_parser(cmd, help=text)
        p.add_argument("--only", nargs="*", choices=list(SOURCES), help="default: the shipping pair, " + " + ".join(SHIPPING))
        if cmd == "aot":
            p.add_argument("--target", nargs="+", choices=list(AOT_TARGETS), default=["h19p"])
    p = sub.add_parser("compare", help="Core AI runtime on this Mac: each gated bundle vs its stripped copy")
    p.add_argument("--only", nargs="*", choices=list(SOURCES), help="default: the shipping pair, " + " + ".join(SHIPPING))
    p.add_argument("--aot", action="store_true", help="readers: also the h16c AOT compiles (strip_ship.py aot "
                                                     "--target h16c-gpu h16c-ane)")
    p.add_argument("--record", default="compare_mac.json", help="file name under <ship>/provenance/")
    p.add_argument("--fixtures")
    p.add_argument("--label", default="", help="free text recorded in the json (e.g. the measurement window)")
    args = ap.parse_args()
    if args.cmd == "strip":
        return cmd_strip(args)
    if args.cmd == "aot":
        return cmd_aot(args)
    return asyncio.run(compare_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
