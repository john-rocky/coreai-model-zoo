"""Mac latency of Timestral graph forms (Core AI Python runtime), one process, the forms alternated.

    python3 timing_mac.py --forms forms.json --fixtures <dir> --out t.json

Run it alone on the machine (no other GPU or Neural Engine job): the snapshot before and after records the load,
the swap, the busiest processes and the advisory GPU lock (conversion/_paths.py gpu_lock()).

forms.json = [{"name", "bundle" (.aimodelc or .aimodel), "unit" (gpu | neural_engine | cpu), "dtype", "rows",
"bucket": [L, h]}, ...]. One forecast's graph work is R = 2 rows in one call (rows 2), or the two rows in two
calls (rows 1). The input is one fixed fixture window (--group, --window), prepared by the host outside the
timed region. Timed per forecast: the graph only (NDArray feeds + await + output copy, all calls of the
forecast), and, apart, the whole forecast (host + graph).

Protocol: each bundle loaded once in this process (its load seconds and its first forecast = cold, the
compile-cache entries of the bundles dropped by the caller before the run); then --rounds rounds over the
forms, forward order on even rounds and reversed on odd ones, the first --warmup rounds not kept; per form the
median and min / max of the kept rounds. Every timed forecast's quantiles are compared with the form's first
forecast (count of non-identical) and with the oracle (max |d| / max |q|), so a wrong graph is not timed as right.
"""
import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import gpu_lock  # noqa: E402

import engines  # noqa: E402
import host_forecast as H  # noqa: E402


def snapshot():
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    lock = gpu_lock()
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), lock=lock.read_text().strip() if lock.exists() else None,
                uptime=run(["uptime"]), swap=run(["sysctl", "-n", "vm.swapusage"]),
                top=run(["ps", "-Ao", "pcpu,pid,comm", "-r"]).splitlines()[:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--group", default="airq_h168_c2048")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    forms = json.loads(a.forms.read_text())
    man = {g["name"]: g for g in json.loads((a.fixtures / "manifest.json").read_text())["groups"]}
    g = man[a.group]
    ctx = np.load(a.fixtures / f"{a.group}.context.npy")[a.window]
    orc = np.load(a.fixtures / f"{a.group}.oracle_q.npy")[a.window, 0]
    h_len = g["horizon"]
    doc = dict(python=platform.python_version(), group=a.group, window=a.window, horizon=h_len,
               context_steps=int(ctx.size), rounds=a.rounds, warmup=a.warmup, before=snapshot(), forms={})

    graphs, prepared, failed = {}, {}, {}
    for f in forms:
        bucket = H.Bucket(*f["bucket"])
        x, mask, _ = H.prepare_context(ctx)
        patches, cos, sin, bias = H.bucket_inputs(x, mask, bucket.n_ctx, bucket.n_hor)
        mirror = patches.copy()
        mirror[..., : H.PATCH] = -mirror[..., : H.PATCH]
        both = np.concatenate([patches, mirror])
        try:
            t0 = time.perf_counter()
            eng = engines.CoreAIGraph(f["bundle"], f["unit"], f["dtype"], f["rows"])
            load_s = time.perf_counter() - t0
            feeds = ([eng.feeds(both, cos, sin, bias)] if f["rows"] == 2 else
                     [eng.feeds(patches, cos, sin, bias), eng.feeds(mirror, cos, sin, bias)])
            t0 = time.perf_counter()
            first = [asyncio.run(eng.run(fd)) for fd in feeds]
            cold_ms = (time.perf_counter() - t0) * 1e3
        except Exception as e:  # a form that cannot load or run is recorded, not timed
            failed[f["name"]] = dict(spec=f, error=f"{type(e).__name__}: {e}"[:2000])
            print(f"FAILED {f['name']}: {failed[f['name']]['error'][:300]}", flush=True)
            continue
        # A form whose numbers are wrong is still timed (its ladder row is `measured` with parity FAIL); the
        # forecast column is timed only when the host gets a finite forecast out of it.
        try:
            q0 = H.forecast(eng, ctx, h_len, bucket, mode="batched" if f["rows"] == 2 else "split")
            rel0 = float(np.abs(q0.astype(np.float64) - orc).max() / np.abs(orc).max())
            forecast_ok = bool(np.isfinite(rel0))
        except ValueError as e:
            rel0, forecast_ok = None, False
            print(f"{f['name']}: forecast not timed ({e})", flush=True)
        graphs[f["name"]] = eng
        prepared[f["name"]] = dict(feeds=feeds, first=first, bucket=bucket, forecast_ok=forecast_ok)
        doc["forms"][f["name"]] = dict(spec=f, spec_used=eng.spec_used, load_error=eng.load_error, load_s=round(load_s, 3), first_forecast_graph_ms=round(cold_ms, 2),
                                       first_forecast_max_rel_vs_oracle=rel0, graph_ms=[], forecast_ms=[], drift=0,
                                       max_rel_vs_oracle=0.0)
        print(f"loaded {f['name']}: load {load_s:.2f} s, first forecast (graph) {cold_ms:.1f} ms, rel {rel0}",
              flush=True)
    doc["failed"] = failed

    async def graph_only(eng, feeds):
        return [await eng.run(fd) for fd in feeds]

    names = [f["name"] for f in forms if f["name"] in graphs]
    loop = asyncio.new_event_loop()  # one loop for the graph-only timing (asyncio.run builds a loop per call)
    for rnd in range(a.rounds):
        order = names if rnd % 2 == 0 else names[::-1]
        for n in order:
            eng, p, rec = graphs[n], prepared[n], doc["forms"][n]
            t0 = time.perf_counter()
            outs = loop.run_until_complete(graph_only(eng, p["feeds"]))
            g_ms = (time.perf_counter() - t0) * 1e3
            if rnd >= a.warmup:
                rec["graph_ms"].append(g_ms)
            rec["drift"] += int(any(not np.array_equal(o, f0, equal_nan=True) for o, f0 in zip(outs, p["first"])))
            if p["forecast_ok"]:
                t0 = time.perf_counter()
                q = H.forecast(eng, ctx, h_len, p["bucket"], mode="batched" if eng.rows == 2 else "split")
                f_ms = (time.perf_counter() - t0) * 1e3
                if rnd >= a.warmup:
                    rec["forecast_ms"].append(f_ms)
                rec["max_rel_vs_oracle"] = max(rec["max_rel_vs_oracle"],
                                               float(np.abs(q.astype(np.float64) - orc).max() / np.abs(orc).max()))
    for n, rec in doc["forms"].items():
        for k in ("graph_ms", "forecast_ms"):
            v = np.array(rec[k])
            rec[k + "_summary"] = (dict(n=int(v.size), median=round(float(np.median(v)), 3),
                                        min=round(float(v.min()), 3), max=round(float(v.max()), 3))
                                   if v.size else dict(n=0, median=None, min=None, max=None))
        g, fc = rec["graph_ms_summary"], rec["forecast_ms_summary"]
        print(f"{n:52s} graph {g['median']} ms [{g['min']}-{g['max']}]  forecast {fc['median']} ms  "
              f"drift {rec['drift']}  rel {rec['max_rel_vs_oracle']:.2e}", flush=True)
    doc["after"] = snapshot()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print("TIMING_DONE")


if __name__ == "__main__":
    main()
