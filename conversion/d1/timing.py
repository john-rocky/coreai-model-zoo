#!/usr/bin/env python3
"""d1 decision latency on the Mac GPU (the bundles' AOT `.aimodelc`, decide.py's path), for the card's columns, inside a quiet window.

The window is the caller's: `apps/D1/_time_mac.sh` (or `~/code/standup/tools/quiet/quiet_hold.py d1b-<label> --`)
holds `~/code/coreai/_GPU_LOCK` for the whole command. This script never writes the lock; it records it (and whether
its holder is one of this process's ancestors), the other GPU-capable processes, the one-minute load and
`sysctl vm.swapusage` before, between and after the worker processes. A run outside a window is marked contended.

The card's columns (one decoder bundle per invocation, the tower bundle with `--tower`; requests from
`fixtures/records.json` and `fixtures/image_records.json`):

    one_question     card_refund's `refund` question alone (the model card's example state)
    three_shared     card_refund's three questions as one request, the shared state run once (decide.py's shared path:
                     the state's first k = floor(Ls / S) * S stable tokens from zero states, the three states copied,
                     each question's remaining tokens from position k on its copy = the direct run's calls past k, so
                     the hidden rows equal the direct run's bit for bit on a static-S graph)
    three_direct     the same request, every row from zero states (the reference for three_shared)
    state_3_4k       long_34k's first question alone (a 3.4k-token state, one question)
    image_384px      img01_shapes_384x384's first question: one 384 x 384 picture (one crop, 144 image rows) and one
                     question; the tower bundle's AOT asset per crop, then the decoder (decide.py's picture path)

The decisions are decide.py's (`D1.build`, `Tower.crop`, `D1.image_nd`, `D1.hidden`, `D1.answer`), the Python reference
the Swift host is gated against. A decision's `latency_ms` = the tower calls and the image rows (a picture item) + the
graph calls (state allocations and copies included) + the readout and the response (host.readout over
head/option_rows, host.response); the row build (tokenizing) is not in it, and neither is the picture's decoding and
cutting (vision_host, once per process and item, `pixels_ms`), as in `d1 decide` (images + graph + readout of its
trace). `e2e_ms` = request in -> response out. Each worker process loads the assets twice (cold, then warm), and per
item runs `--warmup` (5) decisions, then `--reps` (>= 20) timed ones; the probabilities of every timed rep must equal
the first rep's bit for bit (and three_shared's three_direct's). Medians with p10 / p90 over the reps of all processes,
and per process. With `--gate-transcript` (the readout gate of the same decoder asset) every text item's p and hidden
rows are compared with the gate's rows bit for bit; with `--e2e-transcript` (readout_gate_vision.py's e2e gate of the
same decoder and tower assets) the picture item's. The ladder rows (`ladder_rows`) are text for SPEED_LADDER.md.

    cd conversion/d1
    $PY timing.py run --dry-run [--bundle <bundle>]      # the plan (rows, calls, the shared prefix); S = the bundle's
    <window holder> $PY timing.py run --bundle $ZOO_WORK_ROOT/_d1_3b/exports/bundles/d1_3b_decode_fp16_pf16 \\
        --tower $ZOO_WORK_ROOT/_d1_3b/exports/vision/d1_3b_vision_fp16w32 --gate-transcript <readout gate json> \\
        [--e2e-transcript <e2e gate json>] --tag <tag>    # -> results/timing_<tag>.json + timing/<tag>/ (raw)

The Python runtime names its cache entry of an AOT asset by the asset's `main.hash`, under a directory named after
the interpreter's file name (`.venv/bin/python` -> `python/`, `.venv/bin/python3.11` -> `python3-11/`): two assets
with one `main.hash` (int8mlp at block 32 and 16) need different entries, and the record says which entry ran.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import resource
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
import host  # noqa: E402
import vision_host as vh  # noqa: E402
from _paths import gpu_lock, hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
FIXTURES = LANE / "fixtures" / "records.json"
IMAGE_FIXTURES = LANE / "fixtures" / "image_records.json"
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
# (item, fixture, record, questions (a list, None = all, "first"), shared)
ITEMS = (("one_question", "text", "card_refund", ["refund"], False),
         ("three_shared", "text", "card_refund", None, True),
         ("three_direct", "text", "card_refund", None, False),
         ("state_3_4k", "text", "long_34k", "first", False),
         ("image_384px", "image", "img01_shapes_384x384", "first", False))
WARMUP, REPS, MIN_REPS = 5, 20, 20
MANIFEST = "MPSGraph/mpsExecutable.mpsgraphpackage/manifest.plist"


def stats(a) -> dict:
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)),
            "p90": float(np.quantile(a, 0.9)), "min": float(a.min()), "max": float(a.max())}


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sub_request(request: dict, names: list[str] | str | None) -> dict:
    qs = list(request["questions"].items())
    if names is None:
        return request
    if names == "first":
        names = [qs[0][0]]
    return {**request, "questions": {n: q for n, q in qs if n in names}}


def item_requests(names: list[str] | None = None) -> dict:
    """{item: {record, request, shared, pictures}}; a picture item's files are relative to its fixture file."""
    recs = {"text": {r["id"]: r for r in json.loads(FIXTURES.read_text())["records"]},
            "image": {r["id"]: r for r in json.loads(IMAGE_FIXTURES.read_text())["records"]}}
    out = {}
    for name, kind, rid, qs, shared in ITEMS:
        if names and name not in names:
            continue
        r = recs[kind][rid]
        pics = [IMAGE_FIXTURES.parent / x for x in r.get("images", [])] if kind == "image" else []
        out[name] = {"record": rid, "request": sub_request(r["request"], qs), "shared": shared, "pictures": pics}
    return out


# --------------------------------------------------------------------------- the engine: decide.py's decoder and tower
class Engine:
    """One decoder bundle (decide.D1) and, for picture items, one tower bundle (decide.Tower), loaded together."""

    def __init__(self, bundle: Path, aimodelc: str | None = None, tower: str | None = None):
        import decide

        self.d1 = decide.D1(Path(bundle), Path(aimodelc) if aimodelc else None)
        if self.d1.toy:
            raise SystemExit(f"{self.d1.name} is a toy bundle: timing measures the model's bundles only")
        self.tower = decide.Tower(Path(tower)) if tower else None
        if self.tower is not None and self.tower.d != self.d1.d:
            raise SystemExit(f"the tower's width {self.tower.d} != the decoder's {self.d1.d}")
        self.S = self.d1.S

    async def load(self) -> dict:
        d = await self.d1.load()
        out = {k: d[k] for k in ("model_seconds", "main_seconds", "seconds")}
        if self.tower is not None:
            out["tower_seconds"] = (await self.tower.load())["seconds"]
        return out

    def pictures(self, files: list[Path]) -> dict | None:
        """vision_host on the request's pictures (decode, cap, plan, crops, the tower's four inputs per crop)."""
        if not files:
            return None
        if self.tower is None:
            raise SystemExit("a picture item needs --tower")
        t0 = time.perf_counter()
        plans, inputs = [], []
        for f in files:
            p, tis = self.tower.inputs(f)
            plans.append(p)
            inputs.extend(tis)
        return {"plans": plans, "inputs": inputs, "ms": (time.perf_counter() - t0) * 1e3}

    async def decide(self, request: dict, shared: bool, pictures: dict | None, trace: dict) -> dict:
        t0 = time.perf_counter()
        b = self.d1.build(request, None, pictures["plans"] if pictures else None)
        k = (b["state_tokens"] // self.S) * self.S if shared else 0
        t1 = time.perf_counter()
        tower_ms = []
        if pictures:
            rows = []
            for ti in pictures["inputs"]:
                tc = time.perf_counter()
                out = await self.tower.crop(ti)
                rows.append(out[:ti["n_tokens"]].astype(np.float16))
                tower_ms.append((time.perf_counter() - tc) * 1e3)
            img = self.d1.image_nd(np.concatenate(rows))
        else:
            img = self.d1.image_nd(None)
        t2 = time.perf_counter()
        call_ms: list = []
        hs = await self.d1.hidden(b["rows"], img, "shared" if shared else "direct", k, call_ms)
        t3 = time.perf_counter()
        tr: dict = {}
        body = self.d1.answer(b, hs, tr)
        t4 = time.perf_counter()
        trace.update({"mode": "shared" if k else "direct", "shared_tokens": k, "rows": [r["row_len"] for r in b["rows"]],
                      "names": [r["name"] for r in b["rows"]], "input_tokens": b["input_tokens"], "calls": len(call_ms),
                      "call_ms": call_ms, "tower_ms": tower_ms, "plan_ms": (t1 - t0) * 1e3, "images_ms": (t2 - t1) * 1e3,
                      "graph_ms": (t3 - t2) * 1e3, "readout_ms": (t4 - t3) * 1e3, "latency_ms": (t4 - t1) * 1e3,
                      "_probs": tr["_probs"], "_hidden": hs})
        return body


# --------------------------------------------------------------------------- worker: one process
def worker(args) -> int:
    reqs = item_requests(args.items.split(",") if args.items else None)
    e = Engine(Path(args.bundle), args.aimodelc, args.tower)
    doc: dict = {"pid": os.getpid(), "slot": args.slot, "bundle": e.d1.name, "aimodelc": str(e.d1.aimodelc),
                 "tower": str(e.tower.aimodelc) if e.tower else None, "executable": sys.executable,
                 "started": time.time(), "items": []}

    async def one(item: dict, pics: dict | None) -> tuple[dict, dict]:
        tr: dict = {}
        t0 = time.perf_counter()
        await e.decide(item["request"], item["shared"], pics, tr)
        e2e = (time.perf_counter() - t0) * 1e3
        rec = {k: tr[k] for k in ("latency_ms", "plan_ms", "images_ms", "graph_ms", "readout_ms", "calls", "shared_tokens")}
        rec.update({"e2e_ms": e2e, "tower_ms": float(sum(tr["tower_ms"])), "tower_crop_ms": tr["tower_ms"]})
        return rec, tr

    async def go():
        doc["load_cold"] = await e.load()
        doc["load_warm"] = await e.load()
        ref_probs = {}
        for name, item in reqs.items():
            pics = e.pictures(item["pictures"])
            warm = [(await one(item, pics))[0] for _ in range(args.warmup)]
            reps, same = [], True
            first = None
            for _ in range(args.reps):
                r, tr = await one(item, pics)
                reps.append(r)
                if first is None:
                    first = tr
                    continue
                same = same and all(np.array_equal(a, b) for a, b in zip(first["_probs"], tr["_probs"]))
            ref_probs[name] = first["_probs"]
            lat = [r["latency_ms"] for r in reps]
            doc["items"].append({
                "item": name, "record": item["record"], "questions": first["names"], "shared": item["shared"],
                "pictures": [str(p) for p in item["pictures"]], "pixels_ms": pics["ms"] if pics else None,
                "crops": len(pics["inputs"]) if pics else 0, "rows": first["rows"], "input_tokens": first["input_tokens"],
                "calls": reps[0]["calls"], "shared_tokens": reps[0]["shared_tokens"],
                "warmup_latency_ms": [w["latency_ms"] for w in warm], "latency_ms": stats(lat),
                "graph_ms": stats([r["graph_ms"] for r in reps]), "images_ms": stats([r["images_ms"] for r in reps]),
                "tower_ms": stats([r["tower_ms"] for r in reps]), "readout_ms": stats([r["readout_ms"] for r in reps]),
                "plan_ms": stats([r["plan_ms"] for r in reps]), "e2e_ms": stats([r["e2e_ms"] for r in reps]),
                "probs_bit_equal_every_rep": bool(same), "probs": [[float(x) for x in p] for p in first["_probs"]],
                "hidden_sha256": [sha256_array(h) for h in first["_hidden"]],
                "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "reps": reps})
            print(f"  [{os.getpid()}] {name}: {reps[0]['calls']} calls, median {np.median(lat):.2f} ms "
                  f"(p10 {np.quantile(lat, 0.1):.2f}, p90 {np.quantile(lat, 0.9):.2f})", flush=True)
        if "three_shared" in ref_probs and "three_direct" in ref_probs:
            doc["shared_equals_direct_bit"] = all(np.array_equal(a, b) for a, b in zip(ref_probs["three_shared"],
                                                                                     ref_probs["three_direct"]))

    asyncio.run(go())
    doc["finished"] = time.time()
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------------- driver
def ancestors() -> list[int]:
    out, pid = [], os.getpid()
    for _ in range(32):
        ppid = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not ppid or int(ppid) <= 1:
            break
        pid = int(ppid)
        out.append(pid)
    return out


def window_state() -> dict:
    """The measurement lock as the caller wrote it (read only): open, holder pid, ours."""
    p = gpu_lock()
    content = p.read_text().strip() if p.exists() else ""
    m = re.search(r"\bpid\s+(\d+)", content)
    holder = int(m.group(1)) if m else None
    return {"path": str(p), "content": content[:300], "open": "timing" in content.lower(), "holder_pid": holder,
            "ours": bool(holder and holder in ancestors())}


def machine_state() -> dict:
    from readout_gate import other_gpu_processes

    mine = {os.getpid(), *ancestors()}
    top = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    return {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "window": window_state(),
            "loadavg": subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip(),
            "swapusage": subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip(),
            "top": [ln for ln in top if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))],
            "other_gpu_processes": [ln for ln in other_gpu_processes() if int(ln.split()[0]) not in mine]}


def cache_entry(aimodelc: Path) -> dict:
    """This interpreter's runtime cache entry for an AOT asset: the entry named by its main.hash in the directory named
    after the interpreter's file name, and whether the entry's manifest.plist is the asset's own."""
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    d = Path.home() / "Library/Caches/coreai-cache" / build / re.sub(r"[^A-Za-z0-9-]", "-", Path(sys.executable).name)
    h = (aimodelc / "main.hash").read_bytes().hex()
    am = hashlib.sha256((aimodelc / "main-h16c-delegates" / MANIFEST).read_bytes()).hexdigest()
    e = d / h
    out = {"dir": str(d), "main_hash": h, "aot_manifest_sha256": am, "entry_exists": e.exists()}
    if e.exists():
        ms = sorted(e.rglob("manifest.plist"))
        out["entry_manifest_sha256"] = [hashlib.sha256(m.read_bytes()).hexdigest() for m in ms]
        out["entry_holds_this_asset"] = len(ms) == 1 and out["entry_manifest_sha256"][0] == am
    return out


def plan(items: dict, d1=None, tok=None) -> dict:
    """Rows, calls and the shared prefix of every item: decide.D1.build with the bundle (pictures planned by
    vision_host), else host.build_request on the checkpoint's tokenizer (text items only)."""
    out = {}
    for name, item in items.items():
        if d1 is not None:
            plans = [vh.plan(*vh.cap_pixels(vh.to_rgb(p)).shape[:2]) for p in item["pictures"]] or None
            b = d1.build(item["request"], None, plans)
            S = d1.S
            rows = [r["row_len"] for r in b["rows"]]
            k = (b["state_tokens"] // S) * S if item["shared"] else 0
            extra = {"image_tokens": vh.n_image_tokens(plans) if plans else 0,
                     "crops": sum(len(p.crops) for p in plans) if plans else 0}
        elif item["pictures"]:
            out[name] = {"record": item["record"], "needs": "--bundle (the picture row is built by decide.D1)"}
            continue
        else:
            S = 16
            b = host.build_request(item["request"], tok)
            rows = [q["row_len"] for q in b["questions"]]
            k = (host.shared_prefix([q["row_ids"] for q in b["questions"]]) // S * S) if item["shared"] and len(rows) > 1 else 0
            extra = {}
        calls = (k // S + sum(-(-(t - k) // S) for t in rows)) if k else sum(-(-t // S) for t in rows)
        out[name] = {"record": item["record"], "questions": list(item["request"]["questions"]), "rows": rows,
                     "input_tokens": b["input_tokens"], "shared": item["shared"], "shared_tokens": k, "calls": calls,
                     "padded_tokens": calls * S, **extra}
    return out


def gate_rows(path: str | None, kind: str) -> tuple[dict, dict | None]:
    """(record, question) -> {probs, hidden_sha256} of a gate transcript: the readout gate's base runs, or the e2e
    gate's picture rows (arm e2e)."""
    if not path:
        return {}, None
    doc = json.loads(Path(path).read_text())
    rows = {}
    for r in doc["runs"]:
        if kind == "text" and r.get("variant") == "base":
            rows[(r["id"], r["name"])] = {"probs": r["probs"], "hidden_sha256": r["hidden_sha256"]}
        elif kind == "image" and r.get("arm") == "e2e":
            rows[(r["record"], r["name"])] = {"probs": r["row"]["probs"], "hidden_sha256": r["hidden_sha256"]}
    return rows, doc


def parity(item: dict, ref: dict, against: str | None) -> dict | None:
    """The first process's p and hidden rows of an item against the gate's rows of the same (record, question)."""
    if not ref:
        return None
    qs = []
    for name, p, h in zip(item["questions"], item["probs"], item["hidden_sha256"]):
        g = ref.get((item["record"], name))
        qs.append({"name": name, "in_gate": g is not None, "p_bit_equal": bool(g and g["probs"] == p),
                   "hidden_sha256_equal": bool(g and g["hidden_sha256"] == h)})
    return {"against": against, "questions": qs, "p_bit_equal_all": all(q["p_bit_equal"] for q in qs),
            "hidden_equal_all": all(q["hidden_sha256_equal"] for q in qs)}


def ladder_rows(summary: dict, b, gate: dict | None, timing_json: str) -> list[str]:
    par = "gate not given"
    if gate:
        s = gate["summary"]
        par = (f"{gate['result']} argmax {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']} non-near-tie, "
               f"max|Δp| {s['max_abs_dp']:.2g}")
    form = f"{b.name} (AOT h16c efr, static S {b.S}" + (f", {b.meta['compression']['scheme']}" if b.meta.get("compression") else ", fp16")
    rows = []
    for it in summary["items"]:
        mode = "shared state" if it["shared"] else "direct"
        rows.append(f"| ? | {form}, {mode}, Python reference host (decide.py)) | gpu | Mac {summary['chip']} macOS "
                    f"{summary['macos_build']} | {it['item']} ({it['calls']} calls) | {it['latency_ms']['median']:.1f} ms | "
                    f"{par} | measured | {timing_json} | |")
    return rows


def cmd_run(args) -> int:
    if args.reps < MIN_REPS:
        raise SystemExit(f"--reps {args.reps}: a median needs at least {MIN_REPS} reps")
    items = item_requests(args.items.split(",") if args.items else None)
    if args.dry_run:
        import decide

        d1 = tok = None
        if args.bundle:
            d1 = decide.D1(Path(args.bundle), Path(args.aimodelc) if args.aimodelc else None)
        else:
            tok = host.load_tokenizer(Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"])) / "tokenizer.json")
        doc = {"dry_run": True, "bundle": str(d1.dir) if d1 else None, "S": d1.S if d1 else 16, "warmup": args.warmup,
               "reps": args.reps, "processes": args.processes, "items": plan(items, d1, tok), "machine_now": machine_state(),
               "would_write": [str(LANE / "results" / "timing_<tag>.json"), str(LANE / "timing" / "<tag>/")],
               "note": "no graph is loaded; run it inside a measurement window with --bundle and --tower to measure"}
        print(json.dumps(doc, indent=1, ensure_ascii=False))
        return 0
    if not args.bundle:
        raise SystemExit("--bundle is required (or --dry-run)")
    if any(it["pictures"] for it in items.values()) and not args.tower:
        raise SystemExit("image_384px needs --tower (or leave it out with --items)")
    from readout_gate import Bundle, sha256_file, tree_digest

    b = Bundle(args.bundle, args.aimodelc)
    tag = args.tag or f"{b.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out or LANE / "results" / f"timing_{tag}.json")
    if out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    rec = b.record()
    gate_ref, gate = gate_rows(args.gate_transcript, "text")
    if gate and gate["bundle"]["aimodelc"]["tree_sha256"] != rec["aimodelc"]["tree_sha256"]:
        raise SystemExit("the gate transcript was taken on another asset")
    e2e_ref, e2e = gate_rows(args.e2e_transcript, "image")
    tower_rec = None
    if args.tower:
        import decide

        tw = decide.Tower(Path(args.tower))
        tower_rec = {"bundle": str(tw.dir), "name": tw.name, "aimodelc": str(tw.aimodelc),
                     "tree_sha256": tree_digest(tw.aimodelc)["tree_sha256"],
                     "main_hash": (tw.aimodelc / "main.hash").read_bytes().hex()}
    if e2e and (e2e["decoder"]["aimodelc"]["tree_sha256"] != rec["aimodelc"]["tree_sha256"] or not tower_rec
                or e2e["tower"]["aimodelc"]["tree_sha256"] != tower_rec["tree_sha256"]):
        raise SystemExit("the e2e transcript was taken on another decoder or tower asset")
    raw = LANE / "timing" / tag
    raw.mkdir(parents=True, exist_ok=True)
    window = {"before": machine_state(), "between": []}
    procs = []
    for slot in range(args.processes):
        path = raw / f"p{slot}.json"
        log = raw / f"p{slot}.log"
        t0 = time.monotonic()
        with open(log, "w") as fh:
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--bundle", str(b.dir),
                                   "--slot", str(slot), "--out", str(path), "--warmup", str(args.warmup), "--reps",
                                   str(args.reps)] + (["--aimodelc", args.aimodelc] if args.aimodelc else [])
                                  + (["--tower", args.tower] if args.tower else [])
                                  + (["--items", args.items] if args.items else []),
                                  stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not path.exists():
            raise SystemExit(f"process {slot} failed; see {log}")
        d = json.loads(path.read_text())
        d["process_wall_seconds"] = time.monotonic() - t0
        procs.append(d)
        window["between"].append(machine_state())
    window["after"] = machine_state()
    entries = {"decoder": cache_entry(b.aimodelc)}
    if args.tower:
        entries["tower"] = cache_entry(Path(tower_rec["aimodelc"]))
    items_out = []
    for i, first in enumerate(procs[0]["items"]):
        reps = [r for p in procs for r in p["items"][i]["reps"]]
        lat = [r["latency_ms"] for r in reps]
        it = {"item": first["item"], "record": first["record"], "questions": first["questions"],
              "shared": first["shared"], "pictures": first["pictures"], "crops": first["crops"], "rows": first["rows"],
              "input_tokens": first["input_tokens"], "calls": first["calls"], "shared_tokens": first["shared_tokens"],
              "latency_ms": stats(lat), "latency_ms_median_per_process": [p["items"][i]["latency_ms"]["median"] for p in procs],
              "graph_ms": stats([r["graph_ms"] for r in reps]), "images_ms": stats([r["images_ms"] for r in reps]),
              "tower_ms": stats([r["tower_ms"] for r in reps]), "readout_ms": stats([r["readout_ms"] for r in reps]),
              "plan_ms": stats([r["plan_ms"] for r in reps]), "e2e_ms": stats([r["e2e_ms"] for r in reps]),
              "pixels_ms_per_process": [p["items"][i]["pixels_ms"] for p in procs],
              "probs_bit_equal_every_rep": all(p["items"][i]["probs_bit_equal_every_rep"] for p in procs),
              "probs_equal_across_processes": all(p["items"][i]["probs"] == first["probs"] for p in procs),
              "hidden_equal_across_processes": all(p["items"][i]["hidden_sha256"] == first["hidden_sha256"] for p in procs),
              "probs": first["probs"], "hidden_sha256": first["hidden_sha256"]}
        it["parity"] = parity(it, e2e_ref if it["pictures"] else gate_ref,
                              args.e2e_transcript if it["pictures"] else args.gate_transcript)
        items_out.append(it)
    contended = not all(s["window"]["open"] and s["window"]["ours"]
                        for s in [window["before"], *window["between"], window["after"]])
    summary = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
               "items": items_out,
               "shared_equals_direct_bit": all(p.get("shared_equals_direct_bit", False) for p in procs),
               "parity_all": (all(it["parity"]["p_bit_equal_all"] and it["parity"]["hidden_equal_all"]
                                  for it in items_out if it["parity"]) if any(it["parity"] for it in items_out) else None),
               "items_without_a_gate_row": [it["item"] for it in items_out if not it["parity"]],
               "cache_entries_hold_the_assets": all(e.get("entry_holds_this_asset", False) for e in entries.values())}
    doc = {"schema": "d1-timing/2", "what": __doc__.splitlines()[0], "bundle": rec, "tower": tower_rec, "tag": tag,
           "contended": contended, "window": window, "warmup": args.warmup, "reps": args.reps,
           "processes": [{k: v for k, v in p.items() if k != "items"} for p in procs],
           "gate_transcript": ({"path": str(Path(args.gate_transcript).resolve()), "result": gate["result"]} if gate else None),
           "e2e_transcript": ({"path": str(Path(args.e2e_transcript).resolve()), "result": e2e["result"]} if e2e else None),
           "runtime_cache_entries": entries,
           "machine": {"python": platform.python_version(), "executable": sys.executable, "memory_bytes": int(subprocess.run(
               ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)},
           "scripts": {n: sha256_file(HERE / n) for n in ("timing.py", "decide.py", "host.py", "vision_host.py", "readout_gate.py")},
           "summary": summary, "raw": str(raw),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    doc["ladder_rows"] = ladder_rows(summary, b, gate, str(out))
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for it in items_out:
        par = it["parity"]
        print(f"{it['item']}: {it['calls']} calls, median {it['latency_ms']['median']:.2f} ms "
              f"(p10 {it['latency_ms']['p10']:.2f}, p90 {it['latency_ms']['p90']:.2f}), per process "
              f"{[round(x, 2) for x in it['latency_ms_median_per_process']]}, p = gate "
              f"{par['p_bit_equal_all'] if par else 'no gate row'}")
    print("\n".join(doc["ladder_rows"]))
    print(f"contended {contended}; cache entries hold the assets {summary['cache_entries_hold_the_assets']}; wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure (inside a measurement window), or --dry-run for the plan")
    r.add_argument("--bundle", help="the decoder bundle directory (its AOT asset beside it in <bundles>_aotc/)")
    r.add_argument("--aimodelc", help="the compiled asset (default <bundles>_aotc/<name>.h16c.aimodelc)")
    r.add_argument("--tower", help="the tower bundle (export_vision.py) for image_384px")
    r.add_argument("--gate-transcript", help="the readout gate transcript of the same decoder asset (p, hidden rows)")
    r.add_argument("--e2e-transcript", help="the e2e gate transcript of the same decoder and tower assets (the picture item)")
    r.add_argument("--items", help="comma list of items (default: all five)")
    r.add_argument("--processes", type=int, default=2)
    r.add_argument("--warmup", type=int, default=WARMUP)
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--tag")
    r.add_argument("--out")
    r.add_argument("--prefill-chunk", type=int, default=16, help="unused (kept for old command lines); S is the bundle's")
    r.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("--bundle", required=True)
    w.add_argument("--aimodelc")
    w.add_argument("--tower")
    w.add_argument("--items")
    w.add_argument("--slot", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--warmup", type=int, default=WARMUP)
    w.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    if args.cmd == "worker":
        return worker(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
