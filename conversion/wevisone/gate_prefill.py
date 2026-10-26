#!/usr/bin/env python3
"""The `_pf64` bundle's "prefill" function (static S=64 chunks) against its "main" (S=1) and the fp32 oracle.

The Swift engine feeds a prompt to a bundle that carries a "prefill" function as: full 64-token chunks through
"prefill" while at least 65 tokens remain, the rest one token per call through "main" (CoreAIPipelinedEngine.swift
`processChunkedInput`). This replays that split in the python runtime (AOT h16c, one load, both functions from it,
one fresh KV state per sequence), then teacher-forces the oracle's first 15 generated ids through "main", and puts
it next to the all-"main" run of the same sequence in the same process:
  argmax at the last four prompt positions (chunked / main-only / oracle; oracle knife edges flagged),
  D = the 16 logits rows' cosine to the oracle (chunked and main-only), and chunked vs main-only max |d| / cos,
  the argmax of every chunk position vs the main-only run's.

    QH python3 watch_scratch.py ... -- <coreai venv python> gate_prefill.py <bundle> --aot <x.aimodelc> \
        --oracle <dir> --fixtures <json> --grid 14 --out <json> [--rows a,b]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as G  # noqa: E402

N_STEPS = 16
COS_BAR = 0.9999
FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


async def main_async(args) -> int:
    import coreai.runtime as rt

    g = args.grid
    bundle = Path(args.bundle).resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    odir = Path(args.oracle) / FORM_FOR_GRID[g]
    rows = [r["id"] for r in json.loads(Path(args.fixtures).read_text())["rows"]]
    if args.rows:
        rows = [r for r in rows if r in args.rows.split(",")]
    dec = await G.Decoder.load(rt, Path(args.aot).resolve(), int(meta["language"]["max_context_length"]), g * g, 0)
    dec.check_contract()
    pf = await G.maybe(dec.model.load_function("prefill"))
    pdesc = G.fn_desc(pf)
    C = pdesc["inputs"]["input_ids"][0][1]
    out = {"schema": "wevisone-gate-prefill/1", "bundle": str(bundle), "grid": g, "chunk": C, "prefill_descriptor": pdesc,
           "environment": G.env_record(), "rows": {}}

    async def prefill_call(ids64, offset, static, state):
        res = await G.maybe(pf(inputs={"input_ids": G.nd(rt, np.asarray(ids64, np.int32)[None]),
                                       "position_ids": G.nd(rt, np.arange(offset + len(ids64), dtype=np.int32)[None]),
                                       **static}, state=state))
        lg = np.asarray(res["logits"].numpy())
        assert lg.shape == (1, C, G.V), lg.shape
        return lg[0]

    for rid in rows:
        z = np.load(odir / f"{rid}.npz")
        mapped, start, amount = G.host_inputs(z["input_ids"].tolist(), g)
        S = len(mapped)
        gen = z["gen_ids"].tolist()
        steps = z["step_logits"]
        n_d = min(N_STEPS, len(gen), steps.shape[0])
        seq = np.concatenate([mapped, np.asarray(gen[:n_d - 1], np.int64)])
        static = dec.static(z["image_embeds"], z["deepstack_embeds"], start, amount)
        keep = set(range(S - 4, S)) | set(range(S - 1, S - 1 + n_d))
        t0 = time.perf_counter()
        main_only = await dec.run(seq, static, keep=keep | set(range(S)))
        t_main = time.perf_counter() - t0
        # chunked prompt as the engine splits it, then S=1 on "main" for the tail and the teacher-forced steps
        t0 = time.perf_counter()
        state = dec.fresh_state()
        argmax, kept = {}, {}
        p = 0
        n_chunks = 0
        while S - p - 1 >= C:
            lg = await prefill_call(mapped[p:p + C], p, static, state)
            for j in range(C):
                argmax[p + j] = int(np.argmax(lg[j].astype(np.float32)))
                if p + j in keep:
                    kept[p + j] = lg[j].copy()
            p += C
            n_chunks += 1
        for t in range(p, len(seq)):
            lg = await dec.step(int(seq[t]), t, static, state)
            argmax[t] = int(np.argmax(lg.astype(np.float32)))
            if t in keep:
                kept[t] = lg.copy()
        t_chunk = time.perf_counter() - t0
        o_top1 = z["prompt_top5_ids"][:, 0]
        o_gap = z["prompt_gap_softmax"]
        last4 = [{"p": q, "chunked": argmax[q], "main_only": main_only["argmax"][q], "oracle": int(o_top1[q]),
                  "oracle_gap": float(o_gap[q]), "knife_edge": bool(o_gap[q] < G.KNIFE_EDGE)} for q in range(S - 4, S)]
        d_ch, d_mo, ch_vs_mo = [], [], []
        for k in range(n_d):
            q = S - 1 + k
            a = kept[q].astype(np.float32)
            b = main_only["kept"][q].astype(np.float32)
            d_ch.append(G.cos(a, steps[k]))
            d_mo.append(G.cos(b, steps[k]))
            ch_vs_mo.append({"cos": G.cos(a, b), "max_abs_d": float(np.abs(a - b).max())})
        chunk_pos = [q for q in range(n_chunks * C)]
        rec = {"S": S, "chunks": n_chunks, "tail_s1_calls": S - n_chunks * C, "last4": last4,
               "last4_chunked_eq_oracle_non_knife": sum(r["chunked"] == r["oracle"] for r in last4 if not r["knife_edge"]),
               "last4_non_knife": sum(not r["knife_edge"] for r in last4),
               "last4_chunked_eq_main_only": sum(r["chunked"] == r["main_only"] for r in last4),
               "D_cos_min_chunked": min(d_ch), "D_cos_min_main_only": min(d_mo),
               "D_pass_chunked": min(d_ch) >= COS_BAR, "D_pass_main_only": min(d_mo) >= COS_BAR,
               "chunked_vs_main_only_cos_min": min(x["cos"] for x in ch_vs_mo),
               "chunked_vs_main_only_max_abs_d": max(x["max_abs_d"] for x in ch_vs_mo),
               "chunk_positions_argmax_eq_main_only": f"{sum(argmax[q] == main_only['argmax'][q] for q in chunk_pos)}/{len(chunk_pos)}",
               "seconds": {"main_only": t_main, "chunked": t_chunk}}
        out["rows"][rid] = rec
        print(f"[prefill g{g}] {rid}: chunks {n_chunks} | last4 chunked=oracle {rec['last4_chunked_eq_oracle_non_knife']}/"
              f"{rec['last4_non_knife']} chunked=main {rec['last4_chunked_eq_main_only']}/4 | D min chunked "
              f"{rec['D_cos_min_chunked']:.6f} main {rec['D_cos_min_main_only']:.6f} | chunked vs main cos "
              f"{rec['chunked_vs_main_only_cos_min']:.6f} max|d| {rec['chunked_vs_main_only_max_abs_d']:.3f} | chunk argmax "
              f"{rec['chunk_positions_argmax_eq_main_only']} | {t_main:.1f}s / {t_chunk:.1f}s", flush=True)
    done = out["rows"].values()
    out["summary"] = {"rows": len(done),
                      "last4_chunked_eq_oracle_non_knife": f"{sum(r['last4_chunked_eq_oracle_non_knife'] for r in done)}/{sum(r['last4_non_knife'] for r in done)}",
                      "last4_chunked_eq_main_only": f"{sum(r['last4_chunked_eq_main_only'] for r in done)}/{4 * len(done)}",
                      "D_pass_chunked": sum(r["D_pass_chunked"] for r in done), "D_pass_main_only": sum(r["D_pass_main_only"] for r in done),
                      "D_cos_min_chunked": min(r["D_cos_min_chunked"] for r in done),
                      "D_cos_min_main_only": min(r["D_cos_min_main_only"] for r in done),
                      "chunked_vs_main_only_cos_min": min(r["chunked_vs_main_only_cos_min"] for r in done),
                      "prefill_ms_per_chunk_median": None}
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script_sha256"] = G.sha256_file(Path(__file__))
    G.write_json(Path(args.out), out)
    print(f"[prefill g{g}] {json.dumps(out['summary'])}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--aot", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--rows", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
