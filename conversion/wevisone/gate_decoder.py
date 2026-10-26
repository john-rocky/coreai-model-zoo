#!/usr/bin/env python3
"""Engine gate: one WeVisOne-VL-2B decoder bundle on the Mac GPU (AOT h16c, python runtime) vs the fp32 oracle (O2).

One process, one bundle (`SpecializationOptions.default()`), fresh zero states per sequence, S=1 calls of the
"main" function with the full position ramp (gate_common.Decoder). Static inputs per row: image_embeds and the three
deepstack rows = the oracle's own vision outputs cast to fp16; ids / rope_shift_start / rope_shift_amount = the
exporter's host rule (`<|image_pad|>` -> V + slot, start = first pad + N, amount = N - g).

  A  the prompt one token per call; at the last four prompt positions the engine's argmax vs the oracle's
     (`prompt_top5_ids[:, 0]`); a position whose oracle softmax top-2 gap is < 0.1 is a knife edge, excluded and
     listed. Bar: every other position equal.
  B  from the prompt, free-run greedy for 16 tokens (a stop id ends it) vs the oracle's generation (O2, rp 1.0).
     Bar: no divergence, or the first divergence at an oracle gap < 0.1 (coreai_gate.py).
  D  the oracle's generated ids teacher-forced after the prompt: the engine's logits at the prompt-last position
     and after each of the next 15 ids vs the oracle's (`step_logits`). Bar: cosine >= 0.9999 at every step;
     the ordered top-5 equality is recorded.
  red  on `--red-ids` rows, values only, same process: rope_shift_amount + 1 / planes_1d (start = 1 << 30,
     amount = 0, image rows in place) / image_embeds = 0. RED when the free-run differs from the green run within
     the first 8 tokens, or the worst D step's (1 - cos) exceeds 2x the green run's.
  reset  the process ends by re-running its first teacher-forced sequence; its kept logits must be bit-equal.

    QH <coreai venv>/bin/python gate_decoder.py <bundle-dir> --aot <x.aimodelc> --oracle <dir> --fixtures <json> \
        --grid 14 --out <json> [--rows a,b] [--red-ids a,b] [--reload-every 0]
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
RED_WINDOW = 8
COS_BAR = 0.9999
FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


async def gate_row(dec, z, g: int, tok, static_override=None) -> dict:
    ids = z["input_ids"].tolist()
    mapped, start, amount = G.host_inputs(ids, g)
    S = len(mapped)
    gen = z["gen_ids"].tolist()
    steps = z["step_logits"]
    n_d = min(N_STEPS, len(gen), steps.shape[0])
    emb, ds = z["image_embeds"], z["deepstack_embeds"]
    st, am = start, amount
    if static_override:
        emb = static_override.get("image_embeds", emb)
        st = static_override.get("start", st)
        am = static_override.get("amount", am)
    static = dec.static(emb, ds, st, am)
    # teacher-forced: prompt + the oracle's first n_d - 1 generated ids
    seq = np.concatenate([mapped, np.asarray(gen[:n_d - 1], np.int64)])
    keep = set(range(S - 4, S)) | set(range(S - 1, S - 1 + n_d))
    t0 = time.perf_counter()
    tf = await dec.run(seq, static, keep=keep)
    tf_s = time.perf_counter() - t0
    o_top1 = z["prompt_top5_ids"][:, 0]
    o_gap = z["prompt_gap_softmax"]
    a_rows, excluded, a_bad = [], [], []
    for p in range(S - 4, S):
        e = tf["argmax"][p]
        knife = bool(o_gap[p] < G.KNIFE_EDGE)
        row = {"p": p, "engine": e, "oracle": int(o_top1[p]), "equal": e == int(o_top1[p]), "oracle_gap": float(o_gap[p]),
               "knife_edge": knife}
        a_rows.append(row)
        if knife:
            excluded.append(row)
        elif not row["equal"]:
            te, ve = G.topk(tf["kept"][p], 5)
            a_bad.append({**row, "engine_top5": te, "engine_top5_logits": ve})
    d_rows = []
    for k in range(n_d):
        lg = tf["kept"][S - 1 + k].astype(np.float32)
        ref = steps[k]
        te, _ = G.topk(lg, 5)
        to, _ = G.topk(ref, 5)
        d_rows.append({"step": k, "cos": G.cos(lg, ref), "max_abs_d": float(np.abs(lg.astype(np.float64) - ref).max()),
                       "top5_ordered_equal": te == to, "argmax_equal": te[0] == to[0]})
    cos_min = min(r["cos"] for r in d_rows)
    t0 = time.perf_counter()
    fr = await dec.run(mapped, static, free=N_STEPS)
    fr_s = time.perf_counter() - t0
    div = G.first_divergence(fr["gen"], gen, z["gap_softmax"])
    rec = {"S": S, "rope_shift_start": st, "rope_shift_amount": am,
           "A": {"positions": [S - 4, S - 1], "rows": a_rows, "excluded_knife_edge": excluded,
                 "non_knife_edge": 4 - len(excluded), "non_knife_edge_equal": 4 - len(excluded) - len(a_bad),
                 "mismatches": a_bad, "pass": not a_bad},
           "B": {**div, "engine_gen": fr["gen"], "oracle_gen": gen[:len(fr["gen"]) + 2],
                 "engine_text": tok.decode(fr["gen"], skip_special_tokens=False),
                 "oracle_text": tok.decode(gen[:len(fr["gen"])], skip_special_tokens=False),
                 "pass": div["verdict"] != "FAIL"},
           "D": {"steps": n_d, "cos_min": cos_min, "one_minus_cos_max": 1.0 - cos_min,
                 "max_abs_d_max": max(r["max_abs_d"] for r in d_rows),
                 "top5_ordered_equal": sum(r["top5_ordered_equal"] for r in d_rows),
                 "argmax_equal": sum(r["argmax_equal"] for r in d_rows), "rows": d_rows, "pass": cos_min >= COS_BAR},
           "prompt_argmax_all_equal": int(sum(int(tf["argmax"][p] == int(o_top1[p])) for p in range(S))),
           "step_ms_median": float(np.median(tf["step_ms"] + fr["gen_ms"])),
           "decode_ms_median": float(np.median(fr["gen_ms"])) if fr["gen_ms"] else None,
           "seconds": {"teacher_forced": tf_s, "free_run": fr_s}}
    rec["pass"] = bool(rec["A"]["pass"] and rec["B"]["pass"] and rec["D"]["pass"])
    return rec, (seq, static, keep, tf["kept"]), fr["gen"]


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers

    g = args.grid
    form = FORM_FOR_GRID[g]
    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodelc = Path(args.aot).resolve()
    odir = Path(args.oracle) / form
    rows = [r["id"] for r in json.loads(Path(args.fixtures).read_text())["rows"]]
    if args.rows:
        want = args.rows.split(",")
        rows = [r for r in rows if r in want]
    red_ids = [r for r in args.red_ids.split(",") if r]
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    win = G.window_record(args.label)
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    dec = await G.Decoder.load(rt, aimodelc, int(meta["language"]["max_context_length"]), g * g, args.reload_every)
    dec.check_contract()
    out = {"schema": "wevisone-gate-decoder/1", "bundle": str(bundle), "name": meta["name"], "grid": g, "form": form,
           "aimodelc": G.tree_digest(aimodelc), "metadata_sha256": G.sha256_file(bundle / "metadata.json"),
           "environment": G.env_record(), "window": win, "load_seconds": dec.load_seconds, "descriptor": dec.desc,
           "reload_every": args.reload_every,
           "bar": {"A": "last 4 prompt positions: every non-knife-edge argmax = the oracle's",
                   "B": "16-token free-run: no divergence, or the first at an oracle gap < 0.1",
                   "D": f"teacher-forced logits cos >= {COS_BAR} at each of 16 steps",
                   "knife_edge": f"oracle softmax top-2 gap < {G.KNIFE_EDGE}",
                   "red": f"free-run differs from green within {RED_WINDOW} tokens, or worst (1 - cos) > 2x green"},
           "rows": {}}
    first = None
    greens = {}
    dumps = {}
    for rid in rows:
        npz = odir / f"{rid}.npz"
        if not npz.exists():
            out["rows"][rid] = {"skipped": "no oracle npz"}
            continue
        z = np.load(npz)
        rec, seq_info, green_gen = await gate_row(dec, z, g, tok)
        out["rows"][rid] = rec
        greens[rid] = (rec, green_gen)
        if args.dump_logits:
            kept = seq_info[3]
            dumps[rid] = np.stack([kept[p] for p in sorted(kept)])
        if first is None:
            first = (rid, *seq_info)
        b = rec["B"]
        print(f"[gate g{g}] {rid}: A {rec['A']['non_knife_edge_equal']}/{rec['A']['non_knife_edge']}"
              f"{' (knife ' + str([r['p'] for r in rec['A']['excluded_knife_edge']]) + ')' if rec['A']['excluded_knife_edge'] else ''}"
              f" | B {b['verdict']} {b['exact_prefix']}/{b['compared']}"
              f"{' gap ' + format(b['oracle_gap_at_divergence'], '.3f') if b.get('oracle_gap_at_divergence') is not None else ''}"
              f" | D cos {rec['D']['cos_min']:.6f} top5 {rec['D']['top5_ordered_equal']}/{rec['D']['steps']}"
              f" | {rec['step_ms_median']:.1f} ms/step | {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    red = {}
    for rid in red_ids:
        if rid not in greens:
            continue
        z = np.load(odir / f"{rid}.npz")
        grec, ggen = greens[rid]
        mapped, start, amount = G.host_inputs(z["input_ids"].tolist(), g)
        arms = {"amount_plus_1": {"amount": amount + 1},
                "planes_1d": {"start": G.TEXT_ONLY_START, "amount": 0},
                "image_embeds_zero": {"image_embeds": np.zeros_like(z["image_embeds"])}}
        red[rid] = {"green": {"gen_first": ggen[:RED_WINDOW], "one_minus_cos_max": grec["D"]["one_minus_cos_max"]}}
        for arm, ov in arms.items():
            rrec, _, rgen = await gate_row(dec, z, g, tok, static_override=ov)
            moved = rgen[:RED_WINDOW] != ggen[:RED_WINDOW]
            big = rrec["D"]["one_minus_cos_max"] > 2 * grec["D"]["one_minus_cos_max"]
            red[rid][arm] = {"gen_first": rgen[:RED_WINDOW], "free_run_moved": moved,
                             "one_minus_cos_max": rrec["D"]["one_minus_cos_max"], "over_2x_green": big,
                             "A_pass": rrec["A"]["pass"], "B_verdict": rrec["B"]["verdict"],
                             "text": tok.decode(rgen, skip_special_tokens=False),
                             "result": "RED" if (moved or big) else "NOT RED"}
            print(f"[red g{g}] {rid} {arm}: {red[rid][arm]['result']} moved {moved} 1-cos "
                  f"{rrec['D']['one_minus_cos_max']:.2e} vs green {grec['D']['one_minus_cos_max']:.2e}", flush=True)
    out["red_arms"] = red
    if args.dump_logits:
        np.savez(args.dump_logits, **dumps)
    if first is not None:
        rid, seq, static, keep, kept0 = first
        r = await dec.run(seq, static, keep=keep)
        same = all(np.array_equal(kept0[p], r["kept"][p]) for p in keep)
        out["reset_check"] = {"row": rid, "positions": len(keep), "bit_equal": bool(same),
                              "max_abs_diff": max(float(np.abs(kept0[p].astype(np.float32) - r["kept"][p].astype(np.float32)).max()) for p in keep)}
        print(f"[reset g{g}] {rid}: bit-equal {same}", flush=True)
    done = {k: v for k, v in out["rows"].items() if "skipped" not in v}
    out["summary"] = {"rows": len(done),
                      "A": f"{sum(v['A']['pass'] for v in done.values())}/{len(done)}",
                      "A_knife_edge_excluded": {k: [r['p'] for r in v['A']['excluded_knife_edge']] for k, v in done.items() if v['A']['excluded_knife_edge']},
                      "B": f"{sum(v['B']['pass'] for v in done.values())}/{len(done)}",
                      "B_exact16": sum(v['B']['verdict'] == 'PASS' for v in done.values()),
                      "B_knife_edge": {k: v['B']['first_divergence'] for k, v in done.items() if v['B']['verdict'] == 'PASS_TIE'},
                      "D": f"{sum(v['D']['pass'] for v in done.values())}/{len(done)}",
                      "D_cos_min": min(v['D']['cos_min'] for v in done.values()) if done else None,
                      "red": {rid: {a: x["result"] for a, x in arms.items() if a != "green"} for rid, arms in red.items()},
                      "reset_bit_equal": out.get("reset_check", {}).get("bit_equal"),
                      "decoder_reloads": dec.reloads, "reload_seconds": dec.reload_seconds}
    out["result"] = "PASS" if done and all(v["pass"] for v in done.values()) and out.get("reset_check", {}).get("bit_equal") else "FAIL"
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/wevisone/gate_decoder.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(Path(args.out), out)
    print(f"[gate g{g}] {out['name']}: {out['result']} {json.dumps(out['summary'])} wall {out['wall_seconds']:.0f}s -> {args.out}",
          flush=True)
    return 0 if out["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--aot", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--red-ids", default="")
    ap.add_argument("--reload-every", type=int, default=16)
    ap.add_argument("--label", default="wv-r1-coreai-gate")
    ap.add_argument("--dump-logits", default="", help="save each row's kept teacher-forced logits (npz)")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
