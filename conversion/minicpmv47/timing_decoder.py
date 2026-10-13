#!/usr/bin/env python3
"""First Mac timing of the MiniCPM-V-4.7 decoder bundles (AOT h16c, python runtime), taken inside a quiet window.

Run the driver under `quiet_hold.py mv47-r3-timing` (QW QH ...): it records the window (the lock line, the
1-minute load, `sysctl vm.swapusage`, the other processes that could hold the GPU), then runs one worker
process per decoder bundle, one after another. A worker loads the `.aimodelc` (`SpecializationOptions.default()`),
feeds P1 (the oracle's ids with the 64 image rows `<P1>__image_features` as fp16, start / amount from
`host_static_inputs`) one token per call from fresh states, then decodes greedily: 8 warm `main` calls, then
64 timed calls (the stop ids do not end it; the tokens are recorded). ms/call = wall around one `fn(...)` call
including the logits readback; median, p10, p90, and tok/s = 1000 / median.

The tower's encode ms are G3's (`r3/gate_vision_<mode>.json`, measured in its own quiet window) and are copied
into the record with their source.

    QW QH mv47-r3-timing -- coreai-models/.venv/bin/python conversion/minicpmv47/timing_decoder.py \
        --bundles <exports>/minicpmv47_vlm_decode_int8hu,<exports>/minicpmv47_vlm_decode_fp16 --out <lane>/r3/timing.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as G  # noqa: E402

N_WARM, N_TIMED = 8, 64
async def worker_async(bundle: Path, out: Path) -> int:
    import coreai.runtime as rt
    from minicpmv47_vl_pipelined import host_static_inputs

    meta = json.loads((bundle / "metadata.json").read_text())
    aimodelc = G.aimodelc_for(bundle)
    z = np.load(G.ORACLE / "oracle.npz")
    dec = await G.Decoder.load(rt, aimodelc, int(meta["language"]["max_context_length"]))
    dec.check_contract()
    mapped, start, amount = host_static_inputs(z["P1__input_ids"].astype(np.int64))
    mapped = mapped.numpy().astype(np.int64)
    static = dec.static(z["P1__image_features"].astype(np.float16), int(start[0]), int(amount[0]))
    state = dec.fresh_state()
    prompt_ms = []
    lg = None
    for t, tk in enumerate(mapped):
        t0 = time.perf_counter()
        lg = await dec.step(int(tk), t, static, state)
        prompt_ms.append((time.perf_counter() - t0) * 1e3)
    t = len(mapped)
    nxt = int(np.argmax(lg.astype(np.float32)))
    gen, ms = [nxt], []
    for _ in range(N_WARM + N_TIMED):
        t0 = time.perf_counter()
        lg = await dec.step(nxt, t, static, state)
        ms.append((time.perf_counter() - t0) * 1e3)
        t += 1
        nxt = int(np.argmax(lg.astype(np.float32)))
        gen.append(nxt)
    timed = np.asarray(ms[N_WARM:])
    med = float(np.median(timed))
    rec = {"bundle": str(bundle), "name": meta["name"], "aimodelc": G.tree_digest(aimodelc), "pid": os.getpid(),
           "load_seconds": dec.load_seconds, "prompt": "P1", "prompt_tokens": len(mapped),
           "prompt_ms_per_call_median": float(np.median(prompt_ms)), "prompt_ms_total": float(sum(prompt_ms)),
           "decode": {"warm_calls": N_WARM, "timed_calls": N_TIMED, "positions": [len(mapped) + N_WARM, t - 1],
                      "ms_median": med, "ms_p10": float(np.quantile(timed, 0.1)), "ms_p90": float(np.quantile(timed, 0.9)),
                      "ms_min": float(timed.min()), "ms_max": float(timed.max()), "tok_per_s": 1000.0 / med,
                      "warm_ms": ms[:N_WARM], "timed_ms": timed.tolist()},
           "generated_ids": gen, "stop_seen_at": next((i for i, x in enumerate(gen) if x in G.STOP_IDS), None)}
    G.write_json(out, rec)
    print(f"[timing] {meta['name']}: main {med:.3f} ms/call (p10 {rec['decode']['ms_p10']:.3f}, p90 "
          f"{rec['decode']['ms_p90']:.3f}) = {rec['decode']['tok_per_s']:.1f} tok/s; prompt {len(mapped)} calls "
          f"median {rec['prompt_ms_per_call_median']:.3f} ms", flush=True)
    return 0


async def decoders_interleaved_async(bundles: list[Path], out: Path) -> int:
    """All decoder bundles in one process, interleaved: each gets P1's prompt on its own fresh states and 8 warm
    decode calls, then R rounds in which every decoder makes N timed greedy calls (order rotated per round).
    Cancels the drift between separate processes (run 1 vs run 2 of round 3 moved one decoder by 20 %)."""
    import coreai.runtime as rt
    from minicpmv47_vl_pipelined import host_static_inputs

    z = np.load(G.ORACLE / "oracle.npz")
    mapped, start, amount = host_static_inputs(z["P1__input_ids"].astype(np.int64))
    mapped = mapped.numpy().astype(np.int64)
    decs = []
    for b in bundles:
        meta = json.loads((b / "metadata.json").read_text())
        dec = await G.Decoder.load(rt, G.aimodelc_for(b), int(meta["language"]["max_context_length"]))
        dec.check_contract()
        static = dec.static(z["P1__image_features"].astype(np.float16), int(start[0]), int(amount[0]))
        state = dec.fresh_state()
        lg = None
        for t, tk in enumerate(mapped):
            lg = await dec.step(int(tk), t, static, state)
        decs.append({"name": b.name, "dec": dec, "static": static, "state": state, "t": len(mapped),
                     "nxt": int(np.argmax(lg.astype(np.float32))), "ms": [], "gen": []})

    async def calls(st, n, keep):
        for _ in range(n):
            t0 = time.perf_counter()
            lg = await st["dec"].step(st["nxt"], st["t"], st["static"], st["state"])
            dt = (time.perf_counter() - t0) * 1e3
            st["t"] += 1
            st["nxt"] = int(np.argmax(lg.astype(np.float32)))
            st["gen"].append(st["nxt"])
            if keep:
                st["ms"].append(dt)

    for st in decs:
        await calls(st, N_WARM, False)
    R, N = 4, N_TIMED // 4
    rounds = []
    for r in range(R):
        order = decs[r % len(decs):] + decs[:r % len(decs)]
        row = {}
        for st in order:
            k = len(st["ms"])
            await calls(st, N, True)
            row[st["name"]] = float(np.median(st["ms"][k:]))
        rounds.append({"order": [o["name"] for o in order], "median_ms": row})
    rec = {"protocol": f"one process, {len(decs)} decoders, P1 prompt each on its own states, {N_WARM} warm calls each, "
                       f"then {R} rounds x {N} timed greedy main calls per decoder (order rotated per round); ms = wall "
                       "around one call incl. the logits readback",
           "decoders": {st["name"]: {"ms_median": float(np.median(st["ms"])), "ms_p10": float(np.quantile(st["ms"], 0.1)),
                                     "ms_p90": float(np.quantile(st["ms"], 0.9)), "n": len(st["ms"]),
                                     "tok_per_s": 1000.0 / float(np.median(st["ms"])), "positions_end": st["t"] - 1,
                                     "generated_ids": st["gen"]}
                        for st in decs},
           "rounds": rounds, "pid": os.getpid()}
    G.write_json(out, rec)
    for k, v in rec["decoders"].items():
        print(f"[decoders interleaved] {k}: median {v['ms_median']:.3f} ms (p10 {v['ms_p10']:.3f}, p90 {v['ms_p90']:.3f}) "
              f"= {v['tok_per_s']:.1f} tok/s; rounds {[round(r['median_ms'][k], 3) for r in rounds]}", flush=True)
    return 0


async def towers_worker_async(bundles: list[Path], out: Path) -> int:
    """All tower forms in one process, interleaved: 3 warm calls each, then R rounds; in each round every tower
    gets N calls on P1 (the order rotates by one per round). One process so that the comparison shares the GPU
    state, the page cache and the clock; the per-form ms of G3 (one process each) stay the gate's record."""
    import coreai.runtime as rt
    import host as H

    px = H.preprocess(G.ORACLE / "fixtures" / "earth.448.png")
    towers = []
    for b in bundles:
        a = G.aimodelc_for(b)
        m = await G.maybe(rt.AIModel.load(str(a), rt.SpecializationOptions.default()))
        names = list(getattr(m, "function_names", []) or [])
        fn = await G.maybe(m.load_function(names[0] if names else "main"))
        dt = np.dtype(G.fn_desc(fn)["inputs"]["pixel_values"][1])
        towers.append((b.name, fn, G.nd(rt, px.astype(dt)), m))

    async def call(fn, x) -> float:
        t0 = time.perf_counter()
        res = await G.maybe(fn(inputs={"pixel_values": x}))
        np.asarray(res["image_features"].numpy())
        return (time.perf_counter() - t0) * 1e3

    for name, fn, x, _ in towers:
        for _ in range(3):
            await call(fn, x)
    R, N = 3, 10
    ms = {name: [] for name, *_ in towers}
    rounds = []
    for r in range(R):
        order = towers[r % len(towers):] + towers[:r % len(towers)]
        row = {}
        for name, fn, x, _ in order:
            v = [await call(fn, x) for _ in range(N)]
            ms[name] += v
            row[name] = float(np.median(v))
        rounds.append({"order": [o[0] for o in order], "median_ms": row})
    rec = {"protocol": f"one process, {len(towers)} towers, 3 warm calls each, {R} rounds x {N} calls per tower on "
                       "P1 (order rotated per round); ms = wall around one call incl. the output readback",
           "towers": {k: {"median_ms": float(np.median(v)), "p10_ms": float(np.quantile(v, 0.1)),
                          "p90_ms": float(np.quantile(v, 0.9)), "n": len(v)} for k, v in ms.items()},
           "rounds": rounds, "pid": os.getpid()}
    G.write_json(out, rec)
    for k, v in rec["towers"].items():
        print(f"[towers] {k}: median {v['median_ms']:.2f} ms (p10 {v['p10_ms']:.2f}, p90 {v['p90_ms']:.2f}) "
              f"rounds {[round(r['median_ms'][k], 2) for r in rounds]}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundles", help="comma list of decoder bundle dirs (driver)")
    ap.add_argument("--out", help="the timing record (driver)")
    ap.add_argument("--label", default="mv47-r3-timing")
    ap.add_argument("--worker", help="one bundle dir (worker)")
    ap.add_argument("--worker-out")
    ap.add_argument("--towers", help="comma list of tower bundle dirs, timed interleaved in one process (driver)")
    ap.add_argument("--towers-worker", help="comma list (worker)")
    ap.add_argument("--interleave", action="store_true", help="time the --bundles decoders interleaved in one process")
    ap.add_argument("--decoders-worker", help="comma list (worker)")
    args = ap.parse_args()
    if args.worker:
        return asyncio.run(worker_async(Path(args.worker), Path(args.worker_out)))
    if args.decoders_worker:
        return asyncio.run(decoders_interleaved_async([Path(b) for b in args.decoders_worker.split(",")],
                                                      Path(args.worker_out)))
    if args.towers_worker:
        return asyncio.run(towers_worker_async([Path(b) for b in args.towers_worker.split(",")], Path(args.worker_out)))

    out = Path(args.out)
    rec = {"schema": "minicpmv47-timing/1", "environment": G.env_record(),
           "window_before": G.window_record(args.label), "decoder": {}, "vision": {},
           "protocol": {"decoder": f"P1 prompt fed S=1 from fresh states, then {N_WARM} warm + {N_TIMED} timed greedy "
                                   "main calls; ms = wall around one call incl. logits readback; one process per bundle",
                        "vision": "G3: cold = first call after load; warm = median of 5 calls on P1, same process, "
                                  "each tower in its own quiet_hold window",
                        "vision_interleaved": "this window: the towers in one process, interleaved (towers_worker_async)",
                        "decoder_interleaved": "with --interleave: the decoders in one process, interleaved "
                                               "(decoders_interleaved_async) instead of one process each"}}
    t_all = time.perf_counter()
    if args.interleave:
        dout = out.parent / "timing_decoders_interleaved.json"
        db = ",".join(str(Path(b).expanduser().resolve()) for b in args.bundles.split(","))
        proc = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--decoders-worker", db,
                               "--worker-out", str(dout)])
        if proc.returncode == 0 and dout.exists():
            rec["decoder_interleaved"] = json.loads(dout.read_text())
            dout.unlink()
        else:
            rec["decoder_interleaved"] = {"failed": True, "returncode": proc.returncode}
    for b in ([] if args.interleave else args.bundles.split(",")):
        b = Path(b).expanduser().resolve()
        wout = out.parent / f"timing_{b.name}.json"
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", str(b),
                               "--worker-out", str(wout)])
        if proc.returncode != 0 or not wout.exists():
            rec["decoder"][b.name] = {"failed": True, "returncode": proc.returncode}
            continue
        w = json.loads(wout.read_text())
        w["process_wall_seconds"] = time.perf_counter() - t0
        rec["decoder"][b.name] = w
        wout.unlink()
    if args.towers:
        tout = out.parent / "timing_towers_interleaved.json"
        tb = ",".join(str(Path(b).expanduser().resolve()) for b in args.towers.split(","))
        proc = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--towers-worker", tb,
                               "--worker-out", str(tout)])
        if proc.returncode == 0 and tout.exists():
            rec["vision_interleaved"] = json.loads(tout.read_text())
            tout.unlink()
        else:
            rec["vision_interleaved"] = {"failed": True, "returncode": proc.returncode}
    for mode in ("fp16w32", "int8lin", "fp16"):
        g = out.parent / f"gate_vision_{mode}.json"
        if g.exists():
            gv = json.loads(g.read_text())
            rec["vision"][f"minicpmv47_vision_{mode}"] = {
                "source": str(g), "cold_first_call_ms": gv["timing"]["cold_first_call_ms"],
                "warm_median_ms": gv["timing"]["warm_median_ms"], "warm_ms": gv["timing"]["warm_ms"],
                "per_prompt_ms": gv["timing"]["per_prompt_ms"], "window": gv["environment"].get("gpu_lock_at_start"),
                "g3_result": gv["result"]}
    rec["window_after"] = G.window_record(args.label)
    rec["wall_seconds"] = time.perf_counter() - t_all
    rec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["script"] = {"path": "conversion/minicpmv47/timing_decoder.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(out, rec)
    print(f"[timing] -> {out}", flush=True)
    return 0 if all(not v.get("failed") for v in rec["decoder"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
