#!/usr/bin/env python3
"""Tower gate: the exact vision tower's AOT asset on the Mac GPU against transformers' image path, crop by crop.

For every crop of the oracle (`vision_toy_oracle.py` on the toy: the 12 fixture pictures and the 6 random ones, 64
crops; the model's oracle has the same layout), the graph — the AOT h16c `.aimodelc` under
`SpecializationOptions.default()`, never the JIT — gets the crop's four host inputs from the oracle's npz (patches,
pos_table, key_bias, unshuffle_idx: `vision_host.tower_inputs`; the float ones cast to the bundle's input dtype), and
the first h w / 4 rows of its image_embeds are compared with the oracle's: the cosine over the crop and the lowest row
cosine (float64), max |d|.

Bar, fixed before any result (written at the top of the transcript before the first GPU process):
  fp32     every crop cos >= 0.99999 and max |d| <= 1e-4; a second process reproduces every output bit for bit
  fp16w32  every crop cos >= 0.9999 and every row cos >= 0.999
  fp16     recorded only (judged on the model)
Negative controls, each run on the same asset and judged against that asset's bar (fp16: the fp16w32 bar, recorded):
  no_mask     key_bias all zero: the padded patches become keys. A crop with no padding (1024 patches: a tile, a
              full single crop) cannot move and is listed as immune.
  transposed  unshuffle_idx with the merged grid read column-major (output row k = the token at (k mod h/2, k div
              h/2)) against the oracle's row-major rows. A crop whose oracle rows are unchanged by that reordering
              (a flat picture) is immune.
A control is red when every crop it can move misses the bar.

Processes: the Python runtime leaks an IOSurface per call, so a worker takes at most MAX_CALLS calls; `main` runs every
crop and the controls, `rerun` runs every crop again in a fresh process (bit-equality). The GPU is shared with other
sessions: the per-call ms are contended reference values for the transcript only.

    cd conversion/d1
    PY=<coreai-models venv>/bin/python
    ~/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 -- $PY gate_tower.py run \\
        $ZOO_WORK_ROOT/_d1_3b/exports/toy_vision/d1_toy_vision_fp32 --transcript $ZOO_WORK_ROOT/_d1_3b/results/<json>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402

LANE = work_path("_d1_3b")
ORACLE = LANE / "oracle_toy_vision" / "oracle.json"
BARS = {"fp32": {"cos": 0.99999, "max_abs": 1e-4, "rerun": "bit-equal"},
        "fp16w32": {"cos": 0.9999, "min_row": 0.999},
        "fp16": None}
CONTROL_BAR = {"fp32": "fp32", "fp16w32": "fp16w32", "fp16": "fp16w32"}
CONTROLS = ("no_mask", "transposed")
MAX_CALLS = 240
INPUTS = ("patches", "pos_table", "key_bias", "unshuffle_idx")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def transposed_index(idx: np.ndarray, grid: tuple[int, int], n_tok: int) -> np.ndarray:
    """unshuffle_idx with the merged grid read column-major: row k = the token at (k mod h/2, k div h/2)."""
    h2, w2 = grid[0] // 2, grid[1] // 2
    out = idx.copy()
    out[:n_tok] = idx[:n_tok].reshape(h2, w2, 4).transpose(1, 0, 2).reshape(-1, 4)
    return out


def transposed_rows(rows: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    h2, w2 = grid[0] // 2, grid[1] // 2
    return rows.reshape(h2, w2, -1).transpose(1, 0, 2).reshape(h2 * w2, -1)


def compare(got: np.ndarray, want: np.ndarray) -> dict:
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    rows = (g * w).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    d = np.abs(g - w)
    i = int(rows.argmin())
    return {"cos": float(g.ravel() @ w.ravel() / (np.linalg.norm(g) * np.linalg.norm(w))), "min_row": float(rows[i]),
            "min_row_index": i, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "finite": bool(np.isfinite(g).all())}


def passes(s: dict, bar: dict | None) -> bool | None:
    if bar is None:
        return None
    ok = s["finite"] and s["cos"] >= bar["cos"]
    if "max_abs" in bar:
        ok = ok and s["max_abs"] <= bar["max_abs"]
    if "min_row" in bar:
        ok = ok and s["min_row"] >= bar["min_row"]
    return bool(ok)


# --------------------------------------------------------------------------- worker
async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def worker(spec_path: Path) -> int:
    import coreai.runtime as rt

    spec = json.loads(spec_path.read_text())
    dt = {"float16": np.float16, "float32": np.float32, "int32": np.int32}
    in_dtype = {k: dt[v[1]] for k, v in spec["graph"]["inputs"].items()}
    out = {"pid": os.getpid(), "calls": [], "started": time.time()}
    store = {}

    async def go() -> None:
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(spec["aimodelc"], rt.SpecializationOptions.default()))
        fn = await maybe(model.load_function("main"))
        out["load_seconds"] = time.perf_counter() - t0
        d = fn.desc
        out["descriptor"] = {"inputs": {n: [[int(x) for x in d.input_descriptor(n).shape],
                                            str(d.input_descriptor(n).dtype).split(".")[-1]] for n in d.input_names},
                             "outputs": {n: [[int(x) for x in d.output_descriptor(n).shape],
                                             str(d.output_descriptor(n).dtype).split(".")[-1]] for n in d.output_names}}
        cache: dict[str, dict] = {}
        for i, call in enumerate(spec["calls"]):
            z = cache.get(call["npz"])
            if z is None:
                cache.clear()
                with np.load(call["npz"]) as f:
                    z = cache[call["npz"]] = {k: f[k] for k in INPUTS + ("grid", "n_tokens")}
            feeds = {k: np.ascontiguousarray(z[k].astype(in_dtype[k])) for k in INPUTS}
            if call["variant"] == "no_mask":
                feeds["key_bias"] = np.zeros_like(feeds["key_bias"])
            elif call["variant"] == "transposed":
                feeds["unshuffle_idx"] = transposed_index(feeds["unshuffle_idx"], tuple(z["grid"]), int(z["n_tokens"]))
            nd = {k: rt.NDArray(v) for k, v in feeds.items()}
            t1 = time.perf_counter()
            res = await maybe(fn(inputs=nd))
            o = np.asarray(res["image_embeds"].numpy()).copy()
            ms = (time.perf_counter() - t1) * 1e3
            store[f"{i:03d}"] = o
            out["calls"].append({"index": i, "key": call["key"], "variant": call["variant"], "ms": ms,
                                 "shape": list(o.shape), "dtype": str(o.dtype),
                                 "sha256": hashlib.sha256(o.tobytes()).hexdigest()})

    asyncio.run(go())
    out["finished"] = time.time()
    prefix = Path(spec["out"])
    np.savez(prefix.with_suffix(".npz"), **store)
    prefix.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- driver
def gpu_lock_state() -> dict:
    from _paths import gpu_lock

    p = gpu_lock()
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "bytes": st.st_size, "content": p.read_text()[:300],
            "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def env_record() -> dict:
    import importlib.metadata as md
    import platform

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-models", "torch"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    return {"versions": v, "platform": platform.platform(),
            "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc, SpecializationOptions.default(), no JIT",
            "gpu": "shared with other sessions, _GPU_LOCK read only (the ms are contended reference values)",
            "loadavg": os.getloadavg()}


def run_worker(tag: str, spec: dict) -> dict:
    spec_path = Path(spec["out"]).with_suffix(".spec.json")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec) + "\n")
    print(f"[{tag}] {Path(spec['out']).name}: {len(spec['calls'])} calls", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)])
    js = Path(spec["out"]).with_suffix(".json")
    if proc.returncode != 0 or not js.exists():
        raise SystemExit(f"{Path(spec['out']).name}: worker failed (exit {proc.returncode})")
    rec = json.loads(js.read_text())
    rec["process_wall_seconds"] = time.monotonic() - t0
    rec["npz"] = str(Path(spec["out"]).with_suffix(".npz"))
    return rec


def chunks(calls: list, n: int) -> list[list]:
    k = max(1, -(-len(calls) // n))
    size = -(-len(calls) // k)
    return [calls[i:i + size] for i in range(0, len(calls), size)]


def gate(args) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    name, dtype = meta["name"], meta["dtype"]["name"]
    aimodelc = (Path(args.aimodelc) if args.aimodelc else
                bundle.parent.parent / f"{bundle.parent.name}_aotc" / f"{name}.h16c.aimodelc").resolve()
    if not aimodelc.exists():
        raise SystemExit(f"no AOT asset {aimodelc} (export_vision.py --aot)")
    out = Path(args.transcript)
    if out.exists():
        raise SystemExit(f"{out} exists: transcripts are never overwritten")
    oracle_path = Path(args.oracle).resolve()
    oracle = json.loads(oracle_path.read_text())
    if meta.get("toy") and oracle["snapshot"]["model_safetensors_sha256"] != meta["toy"]["snapshot_sha256"]["model.safetensors"]:
        raise SystemExit("the oracle was made on another toy snapshot than the bundle's")
    bar, cbar = BARS[dtype], BARS[CONTROL_BAR[dtype]]
    crops = [(p["id"], c) for p in oracle["pictures"] for c in p["crops"]]
    work = LANE / ("gate_tower_toy" if meta.get("toy") else "gate_tower") / (args.tag or name)
    work.mkdir(parents=True, exist_ok=True)
    mh = aimodelc / "main.hash"
    asset = {"bundle": str(bundle), "name": name, "dtype": dtype, "attention": meta["graph"]["attention"],
             "metadata_sha256": sha256_file(bundle / "metadata.json"), "aimodelc": tree_digest(aimodelc),
             "aimodelc_main_hash": mh.read_bytes().hex() if mh.exists() else None, "graph": meta["graph"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "d1-vision-tower-gate/1", "status": "running", "started": now(), "bar": bar,
                               "control_bar": cbar, "controls": CONTROLS, "asset": name,
                               "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
                               "crops": len(crops)}, indent=1) + "\n")
    # which control calls can move an output (the rest are immune)
    calls_main, immune = [], {c: [] for c in CONTROLS}
    for pid, c in crops:
        key = f"{pid}/{c['crop']}"
        calls_main.append({"key": key, "variant": "base", "npz": c["npz"]})
        if c["n_patches"] < 1024:
            calls_main.append({"key": key, "variant": "no_mask", "npz": c["npz"]})
        else:
            immune["no_mask"].append(key)
        with np.load(c["npz"]) as z:
            want = z["image_embeds"]
        if np.array_equal(transposed_rows(want, tuple(c["grid"])), want):
            immune["transposed"].append(key)
        else:
            calls_main.append({"key": key, "variant": "transposed", "npz": c["npz"]})
    base_spec = {"aimodelc": str(aimodelc), "graph": meta["graph"]}
    t0 = time.monotonic()
    lock_start, env = gpu_lock_state(), env_record()
    procs = []
    for n, part in enumerate(chunks(calls_main, MAX_CALLS)):
        procs.append(("main", run_worker(f"gate {name}", {**base_spec, "calls": part, "out": str(work / f"main_{n:02d}")})))
    rerun_calls = [x for x in calls_main if x["variant"] == "base"]
    for n, part in enumerate(chunks(rerun_calls, MAX_CALLS)):
        procs.append(("rerun", run_worker(f"rerun {name}", {**base_spec, "calls": part, "out": str(work / f"rerun_{n:02d}")})))
    gpu_seconds = time.monotonic() - t0
    lock_end = gpu_lock_state()
    # collect
    outs: dict[tuple[str, str, str], tuple[np.ndarray, dict]] = {}
    for phase, rec in procs:
        z = np.load(rec["npz"])
        for cl in rec["calls"]:
            outs[(phase, cl["key"], cl["variant"])] = (z[f"{cl['index']:03d}"], cl)
    rows = []
    for pid, c in crops:
        key = f"{pid}/{c['crop']}"
        with np.load(c["npz"]) as z:
            want = z["image_embeds"].astype(np.float64)
        n = c["n_tokens"]
        got, cl = outs[("main", key, "base")]
        s = compare(got[:n], want)
        row = {"key": key, "picture": pid, "crop": c["crop"], "kind": c["kind"], "grid": c["grid"],
               "n_patches": c["n_patches"], "n_tokens": n, **s, "pass": passes(s, bar), "ms": cl["ms"],
               "out_shape": cl["shape"], "out_dtype": cl["dtype"],
               "padding_rows_finite": bool(np.isfinite(got[n:].astype(np.float64)).all())}
        if bar is None:
            row["pass_at_control_bar"] = passes(s, cbar)
        r2 = outs.get(("rerun", key, "base"))
        row["rerun_bit_equal"] = None if r2 is None else bool(r2[1]["sha256"] == cl["sha256"])
        for ctl in CONTROLS:
            g = outs.get(("main", key, ctl))
            if g is None:
                row[ctl] = {"immune": True}
                continue
            cs = compare(g[0][:n], want)
            row[ctl] = {"immune": False, "cos": cs["cos"], "min_row": cs["min_row"], "max_abs": cs["max_abs"],
                        "red": not passes(cs, cbar)}
        rows.append(row)

    def worst(field, fn=min):
        r = fn(rows, key=lambda x: x[field])
        return {"value": r[field], "key": r["key"]}

    ctl_sum = {}
    for ctl in CONTROLS:
        moved = [r for r in rows if not r[ctl]["immune"]]
        ctl_sum[ctl] = {"crops": len(moved), "immune": len(immune[ctl]), "immune_keys": immune[ctl],
                        "all_red": bool(moved) and all(r[ctl]["red"] for r in moved),
                        "max_cos": max((r[ctl]["cos"] for r in moved), default=None),
                        "min_max_abs": min((r[ctl]["max_abs"] for r in moved), default=None)}
    ms = [r["ms"] for r in rows]
    summary = {"crops": len(rows), "pictures": len(oracle["pictures"]), "min_cos": worst("cos"),
               "min_row": worst("min_row"), "max_abs": worst("max_abs", max),
               "n_pass": sum(bool(r["pass"]) for r in rows) if bar else None,
               "finite_all": all(r["finite"] for r in rows),
               "rerun_bit_equal": f"{sum(bool(r['rerun_bit_equal']) for r in rows)}/{len(rows)}",
               "by_kind": {k: {"crops": sum(r["kind"] == k for r in rows),
                               "min_cos": min((r["cos"] for r in rows if r["kind"] == k), default=None),
                               "min_row": min((r["min_row"] for r in rows if r["kind"] == k), default=None),
                               "max_abs": max((r["max_abs"] for r in rows if r["kind"] == k), default=None)}
                           for k in ("single", "tile", "thumbnail")},
               "controls": ctl_sum}
    checks = {"controls_red": all(v["all_red"] for v in ctl_sum.values()), "finite": summary["finite_all"]}
    if bar:
        checks["all_crops_pass"] = summary["n_pass"] == len(rows)
    if dtype == "fp32":
        checks["rerun_bit_equal"] = all(r["rerun_bit_equal"] for r in rows)
    result = ("PASS" if all(checks.values()) else "FAIL") if bar else ("RECORDED" if checks["finite"] else "FAIL")
    timing = {"contended": True, "load_seconds": [p["load_seconds"] for _, p in procs],
              "first_call_ms": [p["calls"][0]["ms"] for _, p in procs],
              "ms_per_crop_base": {"n": len(ms), "median": float(np.median(ms)), "p10": float(np.quantile(ms, 0.1)),
                                   "p90": float(np.quantile(ms, 0.9))}}
    cache_entry = None
    if asset["aimodelc_main_hash"]:
        build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
        p = Path.home() / "Library/Caches/coreai-cache" / build / "python" / asset["aimodelc_main_hash"]
        cache_entry = {"path": str(p), "exists": p.exists()}
    record = {"schema": "d1-vision-tower-gate/1",
              "gate": "the exact tower's AOT asset on the Mac GPU, every oracle crop, against transformers 5.19's "
                      "get_image_features rows", "bar": bar, "control_bar": cbar, "result": result, "checks": checks,
              "asset": asset, "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path),
                                         "schema": oracle.get("schema"), "n_crops": oracle.get("n_crops")},
              "script": {"path": "conversion/d1/gate_tower.py", "sha256": sha256_file(Path(__file__).resolve())},
              "environment": env, "gpu_lock": {"start": lock_start, "end": lock_end},
              "processes": [{"phase": ph, "pid": p["pid"], "calls": len(p["calls"]), "load_seconds": p["load_seconds"],
                             "process_wall_seconds": p["process_wall_seconds"], "descriptor": p["descriptor"],
                             "npz": p["npz"]} for ph, p in procs],
              "runtime_cache_entry": cache_entry, "summary": summary, "timing": timing, "rows": rows,
              "seconds": {"gpu_processes": gpu_seconds}, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{result}: {name} crops {len(rows)} min cos {summary['min_cos']['value']:.9f} min row "
          f"{summary['min_row']['value']:.9f} max|d| {summary['max_abs']['value']:.3e} rerun {summary['rerun_bit_equal']}")
    for ctl, v in ctl_sum.items():
        print(f"  control {ctl}: {v['crops']} crops, immune {v['immune']}, all red {v['all_red']}, max cos {v['max_cos']}")
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAIL'}")
    print(f"  ms/crop (contended) median {timing['ms_per_crop_base']['median']:.2f}")
    return 0 if result in ("PASS", "RECORDED") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the gate: workers on the GPU, then the comparison with the oracle")
    r.add_argument("bundle")
    r.add_argument("--transcript", required=True)
    r.add_argument("--oracle", default=str(ORACLE))
    r.add_argument("--aimodelc", help="the compiled asset (default <bundles>_aotc/<name>.h16c.aimodelc)")
    r.add_argument("--tag", help="worker directory name (default: the bundle name)")
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(Path(args.spec))
    return gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
