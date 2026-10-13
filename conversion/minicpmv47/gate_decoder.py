#!/usr/bin/env python3
"""G1 / G2: one MiniCPM-V-4.7 decoder bundle on the Mac GPU (AOT h16c, python runtime) vs the fp32 oracle.

One process, one bundle (`<exports>/minicpmv47_aotc/<name>.h16c.aimodelc`, `SpecializationOptions.default()`),
fresh zero states per sequence, S=1 calls with the full position ramp (gate_common.Decoder).

  G1 --text   P0 (text only; image_embeds 0, start 1 << 30, amount 0).
              teacher-forced: the oracle's 67-token sequence (prompt 19 + its greedy 48) fed one token per
              call; at positions 19..66 the engine's argmax against the argmax of `P0__logits` (the oracle's
              full-vocab logits at those positions). A position whose oracle softmax top-2 gap is < 0.1 is a
              knife edge and is excluded (listed with its gap); every other position must agree.
              free-run: from the 19-token prompt, the engine's own greedy (<= 32, the stop ids end it), first
              divergence from `P0__gen_ids` judged by `margins.npz` (coreai_gate.py's rule).
  G2 --vl     P1..P4: ids / start / amount = `host_static_inputs(<P>__input_ids)`, image_embeds = the oracle's
              tower rows `<P>__image_features` cast to fp16. The prompt one token per call, then free-run
              greedy <= 32 from the same states, judged as above; both answers decoded with the bundle's
              tokenizer. Recorded too (no verdict): at every prompt position the engine's argmax against the
              oracle's (`<P>__top5_ids[:, 0]`) and |engine logit - oracle logit| at the oracle's top-5 ids, and
              the last prompt position against `<P>__logits` (ordered top-5, cos, max |d|).
  --red       three arms on P1 in the same process, values only: rope_shift_amount + 1 (canvas origin and the
              text after the frame move by one), image_embeds zeroed, and planes_1d (start = 1 << 30, amount = 0
              with the image rows in place: every token at (p, p, p), MiniCPM-V-4.6's 1-D scheme). Fixed before running: an arm is RED
              when its free-run differs from the green P1 run within the first 8 tokens, or its max |d| at the
              oracle's top-5 ids over the positions from the frame on exceeds 2x the green run's.
  reset       the process ends by re-running its first sequence; its kept logits must be bit-equal.

Bar (the supervisor's, fixed): G1 non-knife-edge TF 48/48-excluded and free-run divergence none or knife
edge; G2 P1..P4 4/4 by the same rule. Prompts per process: <= 10 (the runtime leaks an IOSurface per call).

    QW coreai-models/.venv/bin/python conversion/minicpmv47/gate_decoder.py <bundle-dir> --text --vl --red \
        --out <lane>/r3/gate_decoder_<mode>.json
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

TEXT_ONLY_START = 1 << 30
RED_WINDOW = 8


def per_position(eng_argmax: list[int], kept: dict, top5_ids: np.ndarray, top5_vals: np.ndarray,
                 positions: range) -> dict:
    """Engine vs oracle at the given prompt positions: argmax agreement and |d| at the oracle's top-5 ids."""
    agree, worst, worst_pos = 0, 0.0, None
    rows = []
    for p in positions:
        lg = kept[p].astype(np.float32)
        d = float(np.abs(lg[top5_ids[p]] - top5_vals[p]).max())
        a = eng_argmax[p] == int(top5_ids[p][0])
        agree += a
        if d > worst:
            worst, worst_pos = d, p
        rows.append({"p": p, "argmax_equal": bool(a), "max_abs_d_top5": d})
    return {"positions": [positions.start, positions.stop - 1], "n": len(rows), "argmax_equal": agree,
            "max_abs_d_top5": worst, "worst_position": worst_pos,
            "argmax_mismatch_positions": [r["p"] for r in rows if not r["argmax_equal"]]}


def last_vs_oracle(lg: np.ndarray, ref: np.ndarray) -> dict:
    a, b = lg.astype(np.float64), ref.astype(np.float64)
    te, _ = G.topk(lg, 5)
    to, _ = G.topk(ref, 5)
    return {"top5_engine": te, "top5_oracle": to, "top5_ordered_equal": te == to, "argmax_equal": te[0] == to[0],
            "cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), "max_abs_d": float(np.abs(a - b).max()),
            "ref_absmax": float(np.abs(b).max())}


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers
    from minicpmv47_vl_pipelined import host_static_inputs

    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodelc = G.aimodelc_for(bundle)
    z = np.load(G.ORACLE / "oracle.npz")
    mz = np.load(G.ORACLE / "margins.npz")
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    dec = await G.Decoder.load(rt, aimodelc, int(meta["language"]["max_context_length"]))
    dec.check_contract()
    out: dict = {"schema": "minicpmv47-gate-decoder/1", "bundle": str(bundle), "name": meta["name"],
                 "aimodelc": G.tree_digest(aimodelc), "metadata_sha256": G.sha256_file(bundle / "metadata.json"),
                 "oracle_npz_sha256": G.sha256_file(G.ORACLE / "oracle.npz"),
                 "margins_npz_sha256": G.sha256_file(G.ORACLE / "margins.npz"),
                 "environment": G.env_record(), "load_seconds": dec.load_seconds, "descriptor": dec.desc,
                 "bar": {"G1": "teacher-forced positions 19..66: every non-knife-edge position's argmax = the oracle's; "
                               "free-run: no divergence, or the first one at an oracle gap < 0.1",
                         "G2": "P1..P4 free-run 4/4 by the same rule", "knife_edge": f"oracle softmax top-2 gap < {G.KNIFE_EDGE}"},
                 "process_prompts": 0}
    zero = np.zeros((64, 1024), np.float16)
    first_seq = None   # (callable, kept logits) for the reset re-run

    def dec_str(ids, skip=False):
        return tok.decode([int(i) for i in ids], skip_special_tokens=skip)

    if args.text:
        ids = z["P0__input_ids"].astype(np.int64)
        S0 = int(z["P0__prompt_len"])
        L0 = z["P0__logits"]
        pos = [int(p) for p in z["P0__logits_positions"]]
        o_top1, o_top2, o_gap = G.softmax_gap(L0)
        static = dec.static(zero, TEXT_ONLY_START, 0)
        t0 = time.perf_counter()
        tf = await dec.run(ids, static, keep=set(pos))
        out["process_prompts"] += 1
        rows, excluded, mism = [], [], []
        for i, p in enumerate(pos):
            e = tf["argmax"][p]
            knife = bool(o_gap[i] < G.KNIFE_EDGE)
            eq = e == int(o_top1[i])
            row = {"p": p, "engine": e, "oracle": int(o_top1[i]), "equal": eq, "oracle_gap": float(o_gap[i]),
                   "knife_edge": knife}
            rows.append(row)
            if knife:
                excluded.append(row)
            elif not eq:
                te, ve = G.topk(tf["kept"][p], 5)
                to, vo = G.topk(L0[i], 5)
                mism.append({**row, "engine_top5": te, "engine_top5_logits": ve, "oracle_top5": to,
                             "oracle_top5_logits": vo, "ids_through_p": ids[:p + 1].tolist()[-8:]})
        lg_all = np.stack([tf["kept"][p].astype(np.float64) for p in pos])
        cos = (lg_all * L0).sum(-1) / (np.linalg.norm(lg_all, axis=-1) * np.linalg.norm(L0, axis=-1))
        all67 = [int(a == b) for a, b in zip(tf["argmax"], z["P0__argmax_all"])]
        gap_all = mz["P0__gap_all"]
        all67_bad = [p for p in range(len(all67)) if not all67[p] and gap_all[p] >= G.KNIFE_EDGE]
        tf_rec = {"sequence": len(ids), "prompt_tokens": S0, "positions": [pos[0], pos[-1]], "n": len(pos),
                  "excluded_knife_edge": excluded, "non_knife_edge": len(pos) - len(excluded),
                  "non_knife_edge_equal": len(pos) - len(excluded) - len(mism), "mismatches": mism,
                  "all_positions_equal": sum(r["equal"] for r in rows),
                  "logits_cos_min": float(cos.min()), "logits_max_abs_d": float(np.abs(lg_all - L0).max()),
                  "all_67_positions_argmax_equal_oracle": int(sum(all67)),
                  "all_67_positions_non_knife_mismatch": all67_bad,
                  "step_ms_median": float(np.median(tf["step_ms"])), "seconds": time.perf_counter() - t0}
        tf_rec["verdict"] = "PASS" if not mism else "FAIL"
        first_seq = ("P0 teacher-forced", ids, static, set(pos), tf["kept"])
        print(f"[G1 TF] {tf_rec['non_knife_edge_equal']}/{tf_rec['non_knife_edge']} non-knife-edge equal "
              f"(excluded {[(r['p'], round(r['oracle_gap'], 4)) for r in excluded]}), all {tf_rec['all_positions_equal']}/48, "
              f"cos min {tf_rec['logits_cos_min']:.6f}, max|d| {tf_rec['logits_max_abs_d']:.4f} -> {tf_rec['verdict']}",
              flush=True)
        t0 = time.perf_counter()
        fr = await dec.run(ids[:S0], static, free=G.N_FREE)
        out["process_prompts"] += 1
        og = z["P0__gen_ids"].astype(np.int64).tolist()
        div = G.first_divergence(fr["gen"], og, mz["P0__margins"])
        fr_rec = {"engine_gen": fr["gen"], "oracle_gen": og[:len(fr["gen"]) + 2], **div,
                  "engine_text": dec_str(fr["gen"]), "oracle_text": dec_str(og[:len(fr["gen"])]),
                  "decode_ms_median": float(np.median(fr["gen_ms"])) if fr["gen_ms"] else None,
                  "seconds": time.perf_counter() - t0}
        if div["first_divergence"] is not None:
            d = div["first_divergence"]
            fr_rec["engine_ids_through_divergence"] = fr["gen"][:d + 1]
        print(f"[G1 free] {div['verdict']} first divergence {div['first_divergence']} "
              f"(margin {div.get('oracle_margin_at_divergence')}) engine {fr_rec['engine_text']!r}", flush=True)
        out["G1"] = {"teacher_forced": tf_rec, "free_run": fr_rec,
                     "verdict": "PASS" if tf_rec["verdict"] == "PASS" and div["verdict"] != "FAIL" else "FAIL"}

    green_p1 = None
    if args.vl:
        g2 = {}
        for P in ("P1", "P2", "P3", "P4"):
            ids_o = z[f"{P}__input_ids"].astype(np.int64)
            mapped, start, amount = host_static_inputs(ids_o)
            mapped = mapped.numpy().astype(np.int64)
            S = len(mapped)
            static = dec.static(z[f"{P}__image_features"].astype(np.float16), int(start[0]), int(amount[0]))
            t0 = time.perf_counter()
            r = await dec.run(mapped, static, keep=set(range(S)), free=G.N_FREE)
            out["process_prompts"] += 1
            og = z[f"{P}__gen_ids"].astype(np.int64).tolist()
            div = G.first_divergence(r["gen"], og, mz[f"{P}__margins"])
            o5i, o5v = z[f"{P}__top5_ids"], z[f"{P}__top5_vals"]
            O = int(start[0]) - int(amount[0]) - 9
            rec = {"prompt_tokens": S, "rope_shift_start": int(start[0]), "rope_shift_amount": int(amount[0]),
                   "canvas_origin": O, **div, "engine_gen": r["gen"], "oracle_gen": og,
                   "engine_text": dec_str(r["gen"], skip=True), "oracle_text": dec_str(og, skip=True),
                   "engine_text_raw": dec_str(r["gen"]),
                   "prompt_positions_all": per_position(r["argmax"], r["kept"], o5i, o5v, range(0, S)),
                   "prompt_positions_from_frame": per_position(r["argmax"], r["kept"], o5i, o5v, range(O, S)),
                   "last_position_vs_oracle": last_vs_oracle(r["kept"][S - 1], z[f"{P}__logits"]),
                   "prompt_step_ms_median": float(np.median(r["step_ms"])),
                   "decode_ms_median": float(np.median(r["gen_ms"])) if r["gen_ms"] else None,
                   "seconds": time.perf_counter() - t0}
            if div["first_divergence"] is not None:
                d = div["first_divergence"]
                rec["divergence_context_ids"] = (mapped.tolist() + r["gen"][:d])[-8:]
            g2[P] = rec
            if P == "P1":
                green_p1 = (mapped, start, amount, r)
                if first_seq is None:
                    first_seq = ("P1 prompt + free-run", mapped, static, set(range(S)), r["kept"])
            print(f"[G2 {P}] {div['verdict']} first divergence {div['first_divergence']} (margin "
                  f"{div.get('oracle_margin_at_divergence')}) | prompt argmax {rec['prompt_positions_all']['argmax_equal']}/{S} "
                  f"max|d| top5 {rec['prompt_positions_all']['max_abs_d_top5']:.4f} | last top5 "
                  f"{rec['last_position_vs_oracle']['top5_ordered_equal']} cos {rec['last_position_vs_oracle']['cos']:.6f} "
                  f"| {rec['engine_text']!r}", flush=True)
        n_pass = sum(v["verdict"] != "FAIL" for v in g2.values())
        out["G2"] = {"prompts": g2, "pass": f"{n_pass}/4", "verdict": "PASS" if n_pass == 4 else "FAIL"}

    if args.red and green_p1 is not None:
        mapped, start, amount, gr = green_p1
        S = len(mapped)
        o5i, o5v = z["P1__top5_ids"], z["P1__top5_vals"]
        O = int(start[0]) - int(amount[0]) - 9
        green_from = per_position(gr["argmax"], gr["kept"], o5i, o5v, range(O, S))
        arms = {"amount_plus_1": (z["P1__image_features"].astype(np.float16), int(start[0]), int(amount[0]) + 1),
                "image_embeds_zero": (zero, int(start[0]), int(amount[0])),
                "planes_1d": (z["P1__image_features"].astype(np.float16), TEXT_ONLY_START, 0)}
        red = {"rule": f"RED when the free-run differs from the green P1 run within the first {RED_WINDOW} tokens, or "
                       "the max |d| at the oracle's top-5 ids over positions >= canvas origin exceeds 2x the green run's",
               "green": {"gen_first": gr["gen"][:RED_WINDOW], "from_frame_max_abs_d_top5": green_from["max_abs_d_top5"]}}
        for name, (emb, st, am) in arms.items():
            r = await dec.run(mapped, dec.static(emb, st, am), keep=set(range(S)), free=G.N_FREE)
            out["process_prompts"] += 1
            pp = per_position(r["argmax"], r["kept"], o5i, o5v, range(O, S))
            moved = r["gen"][:RED_WINDOW] != gr["gen"][:RED_WINDOW]
            big = pp["max_abs_d_top5"] > 2 * green_from["max_abs_d_top5"]
            red[name] = {"start": st, "amount": am, "gen_first": r["gen"][:RED_WINDOW],
                         "text": dec_str(r["gen"], skip=True), "from_frame": pp,
                         "free_run_moved": moved, "top5_d_over_2x_green": big,
                         "result": "RED" if (moved or big) else "NOT RED"}
            print(f"[red {name}] {red[name]['result']}: moved {moved}, max|d| top5 {pp['max_abs_d_top5']:.4f} vs green "
                  f"{green_from['max_abs_d_top5']:.4f}, text {red[name]['text']!r}", flush=True)
        out["red_arms"] = red

    if first_seq is not None:
        label, ids, static, keep, kept0 = first_seq
        r = await dec.run(ids, static, keep=keep, free=0)
        out["process_prompts"] += 1
        same = all(np.array_equal(kept0[p], r["kept"][p]) for p in keep)
        diff = max(float(np.abs(kept0[p].astype(np.float32) - r["kept"][p].astype(np.float32)).max()) for p in keep)
        out["reset_check"] = {"sequence": label, "positions": len(keep), "bit_equal": bool(same), "max_abs_diff": diff}
        print(f"[reset] {label}: bit-equal {same} (max|d| {diff})", flush=True)

    verdicts = [out[k]["verdict"] for k in ("G1", "G2") if k in out]
    out["result"] = "PASS" if verdicts and all(v == "PASS" for v in verdicts) and out.get("reset_check", {}).get(
        "bit_equal", False) else "FAIL"
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/minicpmv47/gate_decoder.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(Path(args.out), out)
    print(f"[gate] {out['name']}: {out['result']} (G1 {out.get('G1', {}).get('verdict')}, G2 {out.get('G2', {}).get('pass')}, "
          f"reset {out.get('reset_check', {}).get('bit_equal')}) prompts {out['process_prompts']} "
          f"wall {out['wall_seconds']:.1f}s -> {args.out}", flush=True)
    return 0 if out["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--text", action="store_true", help="G1 (P0 teacher-forced + free-run)")
    ap.add_argument("--vl", action="store_true", help="G2 (P1..P4 free-run)")
    ap.add_argument("--red", action="store_true", help="the two P1 red arms (needs --vl)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not (args.text or args.vl):
        ap.error("pass --text and/or --vl")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
