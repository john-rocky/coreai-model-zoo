#!/usr/bin/env python3
"""d1 decision latency on the Mac GPU (the bundle's AOT `.aimodelc`), for the card's columns, inside a quiet window.

The window is the caller's: the supervisor runs this under `~/code/standup/tools/quiet/quiet_hold.py d1b-<label> --`,
which holds `~/code/coreai/_GPU_LOCK` for the whole command. This script never writes the lock; it records it (and
whether its holder is one of this process's ancestors), the other GPU-capable processes, the one-minute load and
`sysctl vm.swapusage` before, between and after the worker processes. A run outside a window is marked contended.

The card's columns (one bundle per invocation; requests from `fixtures/records.json`):

    one_question     card_refund's `refund` question alone (the model card's example state)
    three_shared     card_refund's three questions as one request, the shared state run once: the rows' first
                     k = floor(L / S) * S tokens (L = the rows' common prefix, host.shared_prefix) from zero states, the
                     three states copied, each question's remaining tokens from position k on its copy (the same calls
                     as a direct run past k: the hidden rows equal the direct run's bit for bit on a static-S graph)
    three_direct     the same request, every row from zero states (the reference for three_shared)
    state_3_4k       long_34k's first question alone (a 3.4k-token state, one question)
    image_384px      one 384 px image and one question: pending (round 5, the vision encoder)

A decision's `latency_ms` = the graph calls + the readout (host.readout over head/option_rows, the answers): the state
allocations and copies included, the tokenizing not; `e2e_ms` = request in -> response out. Each worker process loads
the asset twice (cold, then warm), and per item runs `--warmup` (5) decisions, then `--reps` (>= 20) timed ones; the
probabilities of every timed rep must equal the first rep's bit for bit (and three_shared's three_direct's). Medians
with p10 / p90 over the reps of all processes, and per process. The ladder rows (`ladder_rows`) are text for
SPEED_LADDER.md; the parity column names the gate transcript (`--gate-transcript`) when one is given.

    cd conversion/d1
    $PY timing.py run --dry-run                       # no bundle: the plan (rows, calls, the shared prefix) at S = 16
    ~/code/standup/tools/quiet/quiet_hold.py d1b-<label> -- \\
        $PY timing.py run --bundle $ZOO_WORK_ROOT/_d1_3b/exports/bundles/d1_3b_decode_fp16_pf16 \\
        --gate-transcript <readout gate json> --tag <tag>     # -> results/timing_<tag>.json + timing/<tag>/ (raw)
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import platform
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
from _paths import gpu_lock, hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
FIXTURES = LANE / "fixtures" / "records.json"
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
ITEMS = (("one_question", "card_refund", ["refund"], False),
         ("three_shared", "card_refund", None, True),
         ("three_direct", "card_refund", None, False),
         ("state_3_4k", "long_34k", "first", False))
PENDING = {"image_384px": "round 5: the vision encoder and the image rows (round 2b writes the grid rule)"}
WARMUP, REPS, MIN_REPS = 5, 20, 20


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def stats(a) -> dict:
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)),
            "p90": float(np.quantile(a, 0.9)), "min": float(a.min()), "max": float(a.max())}


def sub_request(request: dict, names: list[str] | str | None) -> dict:
    qs = list(request["questions"].items())
    if names is None:
        return request
    if names == "first":
        names = [qs[0][0]]
    return {**request, "questions": {n: q for n, q in qs if n in names}}


def item_requests() -> dict:
    recs = {r["id"]: r for r in json.loads(FIXTURES.read_text())["records"]}
    return {name: {"record": rid, "request": sub_request(recs[rid]["request"], names), "shared": shared}
            for name, rid, names, shared in ITEMS}


# --------------------------------------------------------------------------- the engine
class Engine:
    """One bundle's graph, option rows and tokenizer, loaded once; `decide()` answers one request."""

    def __init__(self, bundle: Path, aimodelc: str | None = None):
        from readout_gate import Bundle

        self.b = Bundle(str(bundle), aimodelc)
        if self.b.toy:
            raise SystemExit(f"{self.b.name} is a toy bundle: timing measures the model's bundles only")
        if not self.b.aimodelc.exists():
            raise SystemExit(f"no AOT asset {self.b.aimodelc} (the JIT is never used)")
        self.S, self.max_ctx, self.H = self.b.S, self.b.max_ctx, self.b.hidden
        self.tok = host.load_tokenizer(self.b.dir / "tokenizer" / "tokenizer.json")
        self.table = self.b.table()
        self.fn = None

    async def load(self) -> dict:
        import coreai.runtime as rt
        from readout_gate import check_contract, fn_desc

        self.rt = rt
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(str(self.b.aimodelc), rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await maybe(model.load_function("main"))
        t2 = time.perf_counter()
        bad = check_contract(fn_desc(fn), self.b.contract)
        if bad:
            raise SystemExit(f"the graph's contract differs from the bundle's: {bad}")
        self._model, self.fn = model, fn
        shape, dt = self.b.image_embeds
        self.img = rt.NDArray(np.zeros(shape, np.dtype(dt)))
        return {"model_seconds": t1 - t0, "main_seconds": t2 - t1, "seconds": t2 - t0}

    def zero_arrays(self) -> dict[str, np.ndarray]:
        return {n: np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt))
                for n, (shape, dt) in self.b.contract["states"].items()}

    def to_state(self, arrays: dict) -> dict:
        return {n: self.rt.NDArray(np.ascontiguousarray(a)) for n, a in arrays.items()}

    async def run_ids(self, ids: list[int], p0: int, state: dict, call_ms: list) -> np.ndarray:
        """ids from position p0 (a multiple of S) on, in S-token calls, the last padded -> hidden fp16 [len, H]."""
        S = self.S
        n = -(-len(ids) // S)
        x = np.full(n * S, self.b.pad_id, np.int32)
        x[:len(ids)] = ids
        out = np.empty((n * S, self.H), np.float16)
        for c in range(n):
            t = time.perf_counter()
            res = await maybe(self.fn(inputs={"input_ids": self.rt.NDArray(np.ascontiguousarray(x[c * S:(c + 1) * S].reshape(1, S))),
                                              "position_ids": self.rt.NDArray(np.arange(p0 + (c + 1) * S, dtype=np.int32)[None]),
                                              "image_embeds": self.img}, state=state))
            out[c * S:(c + 1) * S] = np.asarray(res["hidden"].numpy())[0]
            call_ms.append((time.perf_counter() - t) * 1e3)
        return out[:len(ids)]

    def probs(self, q: dict, h: np.ndarray) -> list[float]:
        import export_option_rows as eor

        ids = host.group_ids(q["groups"])
        return host.readout(h[q["slot"]].astype(np.float32), eor.rows_for(self.table, ids), ids, q["groups"])

    async def decide(self, request: dict, shared: bool, trace: dict) -> dict:
        t0 = time.perf_counter()
        b = host.build_request(request, self.tok, table_ids=self.table)
        for q in b["questions"]:
            host.graph_context_check(q["row_len"], self.S, self.max_ctx)
        t1 = time.perf_counter()
        call_ms: list = []
        k = 0
        if shared and len(b["questions"]) > 1:
            k = host.shared_prefix([q["row_ids"] for q in b["questions"]]) // self.S * self.S
        hs = []
        if k:
            st = self.to_state(self.zero_arrays())
            h_pre = await self.run_ids(b["questions"][0]["row_ids"][:k], 0, st, call_ms)
            snap = {n: np.array(v.numpy(), copy=True) for n, v in st.items()}
            for q in b["questions"]:
                h = await self.run_ids(q["row_ids"][k:], k, self.to_state({n: a.copy() for n, a in snap.items()}), call_ms)
                hs.append(np.concatenate([h_pre, h]))
        else:
            for q in b["questions"]:
                hs.append(await self.run_ids(q["row_ids"], 0, self.to_state(self.zero_arrays()), call_ms))
        t2 = time.perf_counter()
        ps = [self.probs(q, h) for q, h in zip(b["questions"], hs)]
        body = host.response(b["validated"], ps, b["input_tokens"])
        t3 = time.perf_counter()
        trace.update({"mode": "shared" if k else "direct", "shared_tokens": k, "rows": [q["row_len"] for q in b["questions"]],
                      "input_tokens": b["input_tokens"], "calls": len(call_ms), "graph_ms": float(sum(call_ms)),
                      "call_ms": call_ms, "host_ms": (t1 - t0) * 1e3, "readout_ms": (t3 - t2) * 1e3,
                      "latency_ms": (t3 - t1) * 1e3, "_probs": ps})
        return body


# --------------------------------------------------------------------------- worker: one process
def worker(args) -> int:
    reqs = item_requests()
    e = Engine(Path(args.bundle), args.aimodelc)
    doc: dict = {"pid": os.getpid(), "slot": args.slot, "bundle": e.b.name, "aimodelc": str(e.b.aimodelc),
                 "started": time.time(), "items": []}

    async def one(item: dict) -> tuple[dict, list]:
        tr: dict = {}
        t0 = time.perf_counter()
        await e.decide(item["request"], item["shared"], tr)
        e2e = (time.perf_counter() - t0) * 1e3
        return {k: tr[k] for k in ("latency_ms", "graph_ms", "readout_ms", "host_ms", "calls", "shared_tokens")} | \
            {"e2e_ms": e2e}, tr["_probs"]

    async def go():
        doc["load_cold"] = await e.load()
        doc["load_warm"] = await e.load()
        ref_probs = {}
        for name, item in reqs.items():
            warm = [(await one(item))[0] for _ in range(args.warmup)]
            reps, same = [], True
            first = None
            for _ in range(args.reps):
                r, p = await one(item)
                reps.append(r)
                first = p if first is None else first
                same = same and all(np.array_equal(a, b) for a, b in zip(first, p))
            ref_probs[name] = first
            lat = [r["latency_ms"] for r in reps]
            doc["items"].append({"item": name, "record": item["record"], "questions": list(item["request"]["questions"]),
                                 "shared": item["shared"], "calls": reps[0]["calls"], "shared_tokens": reps[0]["shared_tokens"],
                                 "warmup_latency_ms": [w["latency_ms"] for w in warm], "latency_ms": stats(lat),
                                 "graph_ms": stats([r["graph_ms"] for r in reps]), "e2e_ms": stats([r["e2e_ms"] for r in reps]),
                                 "probs_bit_equal_every_rep": bool(same), "probs": [[float(x) for x in p] for p in first],
                                 "reps": reps})
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
    """The measurement lock as the caller's quiet_hold.py wrote it (read only): open, holder pid, ours."""
    import re

    p = gpu_lock()
    content = p.read_text().strip() if p.exists() else ""
    m = re.search(r"\bpid\s+(\d+)", content)
    holder = int(m.group(1)) if m else None
    return {"path": str(p), "content": content[:300], "open": "timing" in content.lower(), "holder_pid": holder,
            "ours": bool(holder and holder in ancestors())}


def machine_state() -> dict:
    from readout_gate import other_gpu_processes

    top = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    return {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "window": window_state(),
            "loadavg": subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip(),
            "swapusage": subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip(),
            "top": [ln for ln in top if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))],
            "other_gpu_processes": other_gpu_processes()}


def plan(S: int, tok) -> dict:
    out = {}
    for name, item in item_requests().items():
        b = host.build_request(item["request"], tok)
        rows = [q["row_len"] for q in b["questions"]]
        k = (host.shared_prefix([q["row_ids"] for q in b["questions"]]) // S * S) if item["shared"] and len(rows) > 1 else 0
        calls = (k // S + sum(-(-(t - k) // S) for t in rows)) if k else sum(-(-t // S) for t in rows)
        out[name] = {"record": item["record"], "questions": list(item["request"]["questions"]), "rows": rows,
                     "input_tokens": b["input_tokens"], "shared": item["shared"], "shared_tokens": k, "calls": calls}
    out.update({k: {"pending": v} for k, v in PENDING.items()})
    return out


def ladder_rows(summary: dict, b, gate: dict | None, timing_json: str) -> list[str]:
    parity = "gate not given"
    if gate:
        s = gate["summary"]
        parity = (f"{gate['result']} argmax {s['argmax_equal_non_near_tie']}/{s['questions_non_near_tie']} non-near-tie, "
                  f"max|Δp| {s['max_abs_dp']:.2g}")
    form = f"{b.name} (AOT h16c efr, static S {b.S}" + (f", {b.meta['compression']['scheme']}" if b.meta.get("compression") else ", fp16")
    rows = []
    for it in summary["items"]:
        mode = "shared state" if it["shared"] else "direct"
        rows.append(f"| ? | {form}, {mode}) | gpu | Mac {summary['chip']} macOS {summary['macos_build']} | {it['item']} "
                    f"({it['calls']} calls) | {it['latency_ms']['median']:.1f} ms | {parity} | measured | {timing_json} | |")
    for k, v in PENDING.items():
        rows.append(f"| ? | {b.name} | gpu | Mac | {k} | — | — | pending: {v} | — | |")
    return rows


def cmd_run(args) -> int:
    if args.reps < MIN_REPS:
        raise SystemExit(f"--reps {args.reps}: a median needs at least {MIN_REPS} reps")
    if args.dry_run:
        snap = Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
        tok = host.load_tokenizer((Path(args.bundle) / "tokenizer" / "tokenizer.json") if args.bundle
                                  else snap / "tokenizer.json")
        S = args.prefill_chunk
        if args.bundle:
            S = int(json.loads((Path(args.bundle) / "metadata.json").read_text())["language"]["prefill_chunk"])
        doc = {"dry_run": True, "S": S, "warmup": args.warmup, "reps": args.reps, "processes": args.processes,
               "items": plan(S, tok), "machine_now": machine_state(),
               "would_write": [str(LANE / "results" / "timing_<tag>.json"), str(LANE / "timing" / "<tag>/")],
               "note": "no bundle is loaded; run it under quiet_hold.py with --bundle to measure"}
        print(json.dumps(doc, indent=1, ensure_ascii=False))
        return 0
    if not args.bundle:
        raise SystemExit("--bundle is required (or --dry-run)")
    from readout_gate import Bundle, sha256_file

    b = Bundle(args.bundle, args.aimodelc)
    tag = args.tag or f"{b.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out or LANE / "results" / f"timing_{tag}.json")
    if out.exists():
        raise SystemExit(f"{out} exists: records are never overwritten")
    raw = LANE / "timing" / tag
    raw.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(args.gate_transcript).read_text()) if args.gate_transcript else None
    if gate and gate["bundle"]["aimodelc"]["tree_sha256"] != b.record()["aimodelc"]["tree_sha256"]:
        raise SystemExit("the gate transcript was taken on another asset")
    window = {"before": machine_state(), "between": []}
    procs = []
    for slot in range(args.processes):
        path = raw / f"p{slot}.json"
        log = raw / f"p{slot}.log"
        t0 = time.monotonic()
        with open(log, "w") as fh:
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--bundle", str(b.dir),
                                   "--slot", str(slot), "--out", str(path), "--warmup", str(args.warmup), "--reps",
                                   str(args.reps)] + (["--aimodelc", args.aimodelc] if args.aimodelc else []),
                                  stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not path.exists():
            raise SystemExit(f"process {slot} failed; see {log}")
        d = json.loads(path.read_text())
        d["process_wall_seconds"] = time.monotonic() - t0
        procs.append(d)
        window["between"].append(machine_state())
    window["after"] = machine_state()
    items = []
    for i, first in enumerate(procs[0]["items"]):
        reps = [r for p in procs for r in p["items"][i]["reps"]]
        lat = [r["latency_ms"] for r in reps]
        items.append({"item": first["item"], "record": first["record"], "questions": first["questions"],
                      "shared": first["shared"], "calls": first["calls"], "shared_tokens": first["shared_tokens"],
                      "latency_ms": stats(lat), "latency_ms_median_per_process": [p["items"][i]["latency_ms"]["median"] for p in procs],
                      "graph_ms": stats([r["graph_ms"] for r in reps]), "e2e_ms": stats([r["e2e_ms"] for r in reps]),
                      "probs_bit_equal_every_rep": all(p["items"][i]["probs_bit_equal_every_rep"] for p in procs),
                      "probs_equal_across_processes": all(p["items"][i]["probs"] == first["probs"] for p in procs)})
    contended = not all(s["window"]["open"] and s["window"]["ours"]
                        for s in [window["before"], *window["between"], window["after"]])
    summary = {"chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
               "items": items, "pending": PENDING,
               "shared_equals_direct_bit": all(p.get("shared_equals_direct_bit", False) for p in procs)}
    doc = {"schema": "d1-timing/1", "what": __doc__.splitlines()[0], "bundle": b.record(), "tag": tag,
           "contended": contended, "window": window, "warmup": args.warmup, "reps": args.reps,
           "processes": [{k: v for k, v in p.items() if k != "items"} for p in procs],
           "gate_transcript": ({"path": str(Path(args.gate_transcript).resolve()), "result": gate["result"]} if gate else None),
           "machine": {"python": platform.python_version(), "memory_bytes": int(subprocess.run(
               ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)},
           "scripts": {n: sha256_file(HERE / n) for n in ("timing.py", "host.py", "readout_gate.py")},
           "summary": summary, "raw": str(raw),
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    doc["ladder_rows"] = ladder_rows(summary, b, gate, str(out))
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for it in items:
        print(f"{it['item']}: {it['calls']} calls, median {it['latency_ms']['median']:.2f} ms "
              f"(p10 {it['latency_ms']['p10']:.2f}, p90 {it['latency_ms']['p90']:.2f}), per process "
              f"{[round(x, 2) for x in it['latency_ms_median_per_process']]}")
    print("\n".join(doc["ladder_rows"]))
    print(f"contended {contended}; wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure (under quiet_hold.py), or --dry-run for the plan without a bundle")
    r.add_argument("--bundle", help="the bundle directory (its AOT asset beside it in <bundles>_aotc/)")
    r.add_argument("--aimodelc", help="the compiled asset (default <bundles>_aotc/<name>.h16c.aimodelc)")
    r.add_argument("--gate-transcript", help="the readout gate transcript of the same asset (the ladder's parity column)")
    r.add_argument("--processes", type=int, default=2)
    r.add_argument("--warmup", type=int, default=WARMUP)
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--tag")
    r.add_argument("--out")
    r.add_argument("--prefill-chunk", type=int, default=16, help="--dry-run without a bundle: S")
    r.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("worker")
    w.add_argument("--bundle", required=True)
    w.add_argument("--aimodelc")
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
