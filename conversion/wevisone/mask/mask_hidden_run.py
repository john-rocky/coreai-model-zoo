#!/usr/bin/env python3
"""`<OBJ>` hidden states from the two-output decoder (logits + last_hidden) on Core AI, against the oracle's.

Python runtime, AOT h16c, default specialization, one load per asset. Per dense row of the grid's O2 form, the
static inputs are the oracle's own vision outputs (fp16) and the prompt ids follow the exporter's host rule:
  tf      teacher-forced, S=1 "main": prompt + the oracle's generated ids; at each fed `<OBJ>` the call's
          last_hidden vs the oracle's `dense_t_seg` (teacher-forced capture) and `dense_obj_hidden_generation`
          (cos, max|d|); red arm = the position before the `<OBJ>`. The same calls through the one-output
          asset (`--r1-aot`): logits bit-equal? ms per step of each asset.
  gen     free-run greedy (<= --max-new): at each fed `<OBJ>` keep last_hidden (= what the engine has in hand when
          it feeds `<OBJ>` back); compared with the oracle's where the generated prefix equals the oracle's.
  second  the state lever's other arm: prompt + response once more through the decoder (the "prefill" function in
          64-token chunks while >= 65 tokens remain, then S=1, as the engine splits; S=1 only without "prefill"),
          hidden at every `<OBJ>`: wall ms, and cos vs the generation-time hidden.

    WATCH -- <coreai venv>/bin/python mask_hidden_run.py <bundle> --aot <aimodelc> --r1-aot <aimodelc> \
        --answers <dir> --grid 14 --out <json> [--rows a,b] [--max-new 768]
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
sys.path.insert(0, str(HERE.parent))
import gate_common as G  # noqa: E402

OBJ = 152678
FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


def cos(a, b) -> float:
    return G.cos(a, b)


class HDecoder(G.Decoder):
    """S=1 "main" of the two-output asset (logits, last_hidden); the one-output decoder's inputs and states."""

    def check_contract(self) -> None:
        self.io_dtype = self.desc["inputs"]["image_embeds"][1]
        outs = list(self.desc["outputs"].items())
        if [n for n, _ in outs] != ["logits", "last_hidden"] or tuple(outs[0][1][0]) != (1, 1, G.V) \
                or tuple(outs[1][1][0]) != (1, 1, G.HIDDEN):
            raise SystemExit(f"outputs {self.desc['outputs']} != logits [1,1,V], last_hidden [1,1,{G.HIDDEN}]")

    async def step2(self, token: int, t: int, static: dict, state: dict):
        rt = self.rt
        res = await G.maybe(self.fn(inputs={"input_ids": G.nd(rt, np.array([[token]], np.int32)),
                                            "position_ids": G.nd(rt, np.arange(t + 1, dtype=np.int32)[None]),
                                            **static}, state=state))
        return np.asarray(res["logits"].numpy())[0, 0], np.asarray(res["last_hidden"].numpy())[0, 0]


async def main_async(args) -> int:
    import coreai.runtime as rt

    g = args.grid
    form = FORM_FOR_GRID[g]
    bundle = Path(args.bundle).resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    max_ctx = int(meta["language"]["max_context_length"])
    t_all = time.perf_counter()
    win = G.window_record(args.label)
    dec = await HDecoder.load(rt, Path(args.aot).resolve(), max_ctx, g * g, 0)
    dec.check_contract()
    r1 = None
    if args.r1_aot:
        r1 = await G.Decoder.load(rt, Path(args.r1_aot).resolve(), max_ctx, g * g, 0)
        r1.check_contract()
    pf = pdesc = None
    if "prefill" in (meta["language"].get("function_map") or {}).get("main", []):
        pf = await G.maybe(dec.model.load_function("prefill"))
        pdesc = G.fn_desc(pf)
    C = pdesc["inputs"]["input_ids"][0][1] if pdesc else None
    out = {"schema": "wevisone-r2-hidden-run/1", "bundle": str(bundle), "grid": g, "form": form, "window": win,
           "environment": G.env_record(), "descriptor": dec.desc, "prefill_descriptor": pdesc,
           "assets": {"two_output": G.tree_digest(Path(args.aot).resolve()),
                      "r1_one_output": G.tree_digest(Path(args.r1_aot).resolve()) if args.r1_aot else None},
           "rows": {}}
    want = set(filter(None, args.rows.split(",")))
    ms = {"two": [], "r1": []}
    for jf in sorted((Path(args.answers) / form).glob("*.json")):
        mj = json.loads(jf.read_text())
        if "dense" not in mj or not mj["dense"].get("count_decoded"):
            continue
        rid = mj["id"]
        if want and rid not in want:
            continue
        z = np.load(jf.with_suffix(".npz"))
        mapped, start, amount = G.host_inputs(z["input_ids"].tolist(), g)
        S = len(mapped)
        gen_o = [int(t) for t in z["gen_ids"]]
        seq = np.concatenate([mapped, np.asarray(gen_o, np.int64)])
        obj_pos = [S + j for j, t in enumerate(gen_o) if t == OBJ]
        t_seg = z["dense_t_seg"].astype(np.float32)
        h_gen_o = z["dense_obj_hidden_generation"].astype(np.float32) if "dense_obj_hidden_generation" in z.files else None
        static = dec.static(z["image_embeds"], z["deepstack_embeds"], start, amount)
        rec = {"S": S, "gen_len": len(gen_o), "n_obj": len(obj_pos), "obj_positions": obj_pos}
        # tf: teacher-forced through the two-output asset (and the one-output asset on the same calls)
        state = dec.fresh_state()
        st1 = r1.fresh_state() if r1 else None
        hid, prev, bit_eq, maxd = {}, {}, 0, 0.0
        for t, tk in enumerate(seq):
            t0 = time.perf_counter()
            lg, hd = await dec.step2(int(tk), t, static, state)
            ms["two"].append((time.perf_counter() - t0) * 1e3)
            if r1:
                t0 = time.perf_counter()
                lg1 = await r1.step(int(tk), t, static, st1)
                ms["r1"].append((time.perf_counter() - t0) * 1e3)
                bit_eq += int(np.array_equal(lg, lg1))
                maxd = max(maxd, float(np.abs(lg.astype(np.float32) - lg1.astype(np.float32)).max()))
            if t + 1 in obj_pos:
                prev[t + 1] = hd.astype(np.float32).copy()
            if t in obj_pos:
                hid[t] = hd.astype(np.float32).copy()
        tf = []
        for k, p in enumerate(obj_pos):
            row = {"pos": p, "cos_vs_tf": cos(hid[p], t_seg[k]), "max_abs_d_vs_tf": float(np.abs(hid[p] - t_seg[k]).max()),
                   "red_prev_pos_cos_vs_tf": cos(prev[p], t_seg[k]), "ref_absmax": float(np.abs(t_seg[k]).max())}
            if h_gen_o is not None and k < len(h_gen_o):
                row["cos_vs_generation"] = cos(hid[p], h_gen_o[k])
            tf.append(row)
        rec["tf"] = tf
        if r1:
            rec["logits_vs_r1_one_output"] = {"steps": len(seq), "bit_equal_steps": bit_eq, "max_abs_d": maxd}
        # gen: free-run greedy, the hidden of each fed <OBJ>
        state = dec.fresh_state()
        lg = None
        for t, tk in enumerate(mapped):
            lg, hd = await dec.step2(int(tk), t, static, state)
        gen, gen_h, t = [], {}, S
        nxt = int(np.argmax(lg.astype(np.float32)))
        t_gen0 = time.perf_counter()
        while True:
            gen.append(nxt)
            if nxt in G.STOP_IDS or len(gen) >= args.max_new:
                break
            lg, hd = await dec.step2(nxt, t, static, state)
            if nxt == OBJ:
                gen_h[t] = hd.astype(np.float32).copy()
            t += 1
            nxt = int(np.argmax(lg.astype(np.float32)))
        gen_s = time.perf_counter() - t_gen0
        same_prefix = next((i for i in range(min(len(gen), len(gen_o))) if gen[i] != gen_o[i]), None)
        rec["gen"] = {"ids_equal_oracle": gen == gen_o, "first_difference": same_prefix, "len": len(gen),
                      "n_obj": len(gen_h), "seconds": gen_s}
        gen_cmp = []
        for k, p in enumerate(sorted(gen_h)):
            j = p - S
            ok = same_prefix is None or same_prefix > j
            row = {"pos": p, "prefix_equals_oracle": ok}
            if ok and k < len(t_seg):
                row["cos_vs_tf"] = cos(gen_h[p], t_seg[k])
                row["cos_vs_coreai_tf"] = cos(gen_h[p], hid[p]) if p in hid else None
            gen_cmp.append(row)
        rec["gen"]["obj"] = gen_cmp
        rec["gen"]["ids"] = gen
        # second forward of prompt + response (the state lever's other arm), split as the engine splits
        full = np.concatenate([mapped, np.asarray(gen, np.int64)])
        objs2 = [S + j for j, tk in enumerate(gen) if tk == OBJ]
        t0 = time.perf_counter()
        state = dec.fresh_state()
        h2 = {}
        p = 0
        n_chunks = 0
        if pf is not None:
            while len(full) - p - 1 >= C:
                res = await G.maybe(pf(inputs={"input_ids": G.nd(rt, np.asarray(full[p:p + C], np.int32)[None]),
                                               "position_ids": G.nd(rt, np.arange(p + C, dtype=np.int32)[None]),
                                               **static}, state=state))
                hh = np.asarray(res["last_hidden"].numpy())[0]
                for q in objs2:
                    if p <= q < p + C:
                        h2[q] = hh[q - p].astype(np.float32).copy()
                p += C
                n_chunks += 1
        for q in range(p, len(full)):
            _, hd = await dec.step2(int(full[q]), q, static, state)
            if q in objs2:
                h2[q] = hd.astype(np.float32).copy()
        second_ms = (time.perf_counter() - t0) * 1e3
        rec["second_forward"] = {"tokens": int(len(full)), "chunks": n_chunks, "s1_calls": int(len(full) - n_chunks * (C or 0)),
                                 "ms": second_ms,
                                 "cos_vs_generation_time": [cos(h2[q], gen_h[q]) for q in objs2 if q in gen_h and q in h2],
                                 "max_abs_d_vs_generation_time": [float(np.abs(h2[q] - gen_h[q]).max())
                                                                  for q in objs2 if q in gen_h and q in h2]}
        out["rows"][rid] = rec
        tf_s = ", ".join("%.6f" % r["cos_vs_tf"] for r in tf)
        red_s = ", ".join("%.3f" % r["red_prev_pos_cos_vs_tf"] for r in tf)
        bit_s = rec.get("logits_vs_r1_one_output", {}).get("bit_equal_steps")
        print(f"[hidden g{g}] {rid}: n_obj {len(obj_pos)} | tf cos {tf_s} (red prev {red_s}) | logits vs r1 bit "
              f"{bit_s}/{len(seq)} | gen ids==oracle {rec['gen']['ids_equal_oracle']} | second fwd {second_ms:.0f} ms "
              f"cos {rec['second_forward']['cos_vs_generation_time']}", flush=True)
    rows = out["rows"].values()
    tf_all = [r for x in rows for r in x["tf"]]
    out["summary"] = {
        "rows": len(out["rows"]), "obj": len(tf_all),
        "tf_cos_min": min((r["cos_vs_tf"] for r in tf_all), default=None),
        "tf_cos_vs_generation_min": min((r["cos_vs_generation"] for r in tf_all if "cos_vs_generation" in r), default=None),
        "red_prev_pos_cos_max": max((r["red_prev_pos_cos_vs_tf"] for r in tf_all), default=None),
        "logits_bit_equal_rows": (f"{sum(1 for x in rows if x['logits_vs_r1_one_output']['bit_equal_steps'] == x['S'] + x['gen_len'])}"
                                  f"/{len(out['rows'])}") if r1 else None,
        "logits_max_abs_d_vs_r1": max((x["logits_vs_r1_one_output"]["max_abs_d"] for x in rows), default=None) if r1 else None,
        "gen_ids_equal_oracle": f"{sum(x['gen']['ids_equal_oracle'] for x in rows)}/{len(out['rows'])}",
        "gen_obj_cos_vs_tf_min": min((o["cos_vs_tf"] for x in rows for o in x["gen"]["obj"] if "cos_vs_tf" in o), default=None),
        "second_forward_ms_median": float(np.median([x["second_forward"]["ms"] for x in rows])) if rows else None,
        "second_forward_cos_min": min((c for x in rows for c in x["second_forward"]["cos_vs_generation_time"]), default=None),
        "ms_per_step_median": {k: (float(np.median(v)) if v else None) for k, v in ms.items()},
    }
    if ms["r1"]:
        out["summary"]["second_output_step_cost_pct_python"] = 100.0 * (np.median(ms["two"]) / np.median(ms["r1"]) - 1.0)
    out["mpsgraph_scratch_cleanup"] = G.clean_own_mpsgraph_scratch()
    out["wall_seconds"] = time.perf_counter() - t_all
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script_sha256"] = G.sha256_file(Path(__file__))
    G.write_json(Path(args.out), out)
    print(json.dumps(out["summary"], indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--aot", required=True)
    ap.add_argument("--r1-aot", default="")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--grid", type=int, default=14, choices=[14, 16])
    ap.add_argument("--rows", default="")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="wv-c-r2")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
