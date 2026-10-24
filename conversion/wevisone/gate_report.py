#!/usr/bin/env python3
"""Merge engine-gate records (gate_decoder.py, run in chunks by chunked.py) and write the GATE table.

Per record: A (last four prompt positions, knife edges excluded and listed), B (16-token free-run, coreai_gate.py
rule), D (teacher-forced logits cos >= 0.9999 at every step), the red arms, the reset check. The verdict of a bundle
is recomputed over every row of every chunk (the chunk records each judge only their own rows).

    python3 gate_report.py --gate int8hu_g14=<json> --gate int8hu_g16=<json> --gate fp16_g14=<json> \
        --ladder g14=<json> --ladder g16=<json> --out <json> --md <GATE.md>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(rec: dict) -> dict:
    rows = {k: v for k, v in rec["rows"].items() if "skipped" not in v}
    a_pass = sum(v["A"]["pass"] for v in rows.values())
    b_pass = sum(v["B"]["pass"] for v in rows.values())
    d_pass = sum(v["D"]["pass"] for v in rows.values())
    resets = [c.get("reset_check", {}).get("bit_equal") for c in rec.get("chunks", [])] or [rec.get("reset_check", {}).get("bit_equal")]
    red = {}
    for c in rec.get("chunks", []) or [rec]:
        for rid, arms in (c.get("red_arms") or {}).items():
            red[rid] = {a: x["result"] for a, x in arms.items() if a != "green"}
    s = {"rows": len(rows), "A": f"{a_pass}/{len(rows)}", "B": f"{b_pass}/{len(rows)}", "D": f"{d_pass}/{len(rows)}",
         "B_exact16": sum(v["B"]["verdict"] == "PASS" for v in rows.values()),
         "B_knife_edge": {k: {"step": v["B"]["first_divergence"], "gap": v["B"]["oracle_gap_at_divergence"]}
                          for k, v in rows.items() if v["B"]["verdict"] == "PASS_TIE"},
         "B_fail": {k: {"step": v["B"]["first_divergence"], "gap": v["B"]["oracle_gap_at_divergence"]}
                    for k, v in rows.items() if v["B"]["verdict"] == "FAIL"},
         "A_knife_edge_excluded": {k: [(r["p"], round(r["oracle_gap"], 4)) for r in v["A"]["excluded_knife_edge"]]
                                   for k, v in rows.items() if v["A"]["excluded_knife_edge"]},
         "A_fail": {k: v["A"]["mismatches"] for k, v in rows.items() if not v["A"]["pass"]},
         "D_cos_min": min(v["D"]["cos_min"] for v in rows.values()) if rows else None,
         "D_fail": {k: v["D"]["cos_min"] for k, v in rows.items() if not v["D"]["pass"]},
         "D_top5_ordered_equal": f"{sum(v['D']['top5_ordered_equal'] for v in rows.values())}/{sum(v['D']['steps'] for v in rows.values())}",
         "red": red, "reset_bit_equal": all(r is True for r in resets),
         "step_ms_median": sorted(v["step_ms_median"] for v in rows.values())[len(rows) // 2] if rows else None}
    s["pass"] = bool(rows) and a_pass == len(rows) and b_pass == len(rows) and d_pass == len(rows) and s["reset_bit_equal"]
    s["red_all_red"] = bool(red) and all(x == "RED" for arms in red.values() for x in arms.values())
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--ladder", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()
    out = {"gates": {}, "ladder": {}}
    for item in args.gate:
        name, path = item.split("=", 1)
        rec = json.loads(Path(path).read_text())
        out["gates"][name] = {"path": path, "name": rec.get("name"), "grid": rec.get("grid"),
                              "reload_every": rec.get("reload_every"), **summarize(rec)}
    for item in args.ladder:
        g, path = item.split("=", 1)
        out["ladder"][g] = json.loads(Path(path).read_text())["summary"]
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    L = ["# GATE — WeVisOne-VL-2B on Core AI (Mac M4 Max, AOT h16c GPU, python runtime) vs the fp32 oracle (O2)", "",
         "Bars (ROUND1 pre-registration): A = last 4 prompt positions argmax 4/4 (oracle softmax top-2 gap < 0.1 "
         "excluded and listed); B = 16-token free-run equal or first divergence at a knife edge; D = teacher-forced "
         "logits cos >= 0.9999 at each of 16 steps; red arms must fall; reset re-run bit-equal.", "",
         "## Torch ladder (fp32 overlay vs oracle)", "",
         "| grid | rows | positions | vision cos min | prefill top-1 | decode | e2e top-1 | red (amount+1 / planes_1d) | layers cos min | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in out["ladder"].items():
        lay = min(s["layers"].values()) if s.get("layers") else None
        L.append(f"| {g} | {s['rows']} | {'all equal' if s['positions_all_equal'] else 'MISMATCH'} | {s['vision_cos_min']:.6f} | "
                 f"{s['prefill_top1']} | {s['decode']} | {s['e2e_prefill_top1']} | {s['red']['amount_plus_1']} / {s['red']['planes_1d']} | "
                 f"{'—' if lay is None else format(lay, '.6f')} | {s['verdict']} |")
    L += ["", "## Engine gate (table 1)", "",
          "| bundle | grid | rows | A | B (exact 16) | B knife-edge divergences | D | D cos min | D top-5 ordered | reset | red | ms/step | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in out["gates"].items():
        red = "; ".join(f"{rid}: " + ", ".join(f"{a} {r}" for a, r in arms.items()) for rid, arms in s["red"].items()) or "—"
        L.append(f"| {name} | {s['grid']} | {s['rows']} | {s['A']} | {s['B']} ({s['B_exact16']}) | "
                 f"{', '.join(f'{k}@{v['step']} gap {v['gap']:.3f}' for k, v in s['B_knife_edge'].items()) or '—'} | "
                 f"{s['D']} | {s['D_cos_min']:.6f} | {s['D_top5_ordered_equal']} | {s['reset_bit_equal']} | {red} | "
                 f"{s['step_ms_median']:.1f} | {'PASS' if s['pass'] else 'FAIL'} |")
    L += ["", "A knife-edge exclusions: " + json.dumps({n: s["A_knife_edge_excluded"] for n, s in out["gates"].items()}), "",
          "Failures: " + json.dumps({n: {"A": s["A_fail"], "B": s["B_fail"], "D": s["D_fail"]} for n, s in out["gates"].items()}), ""]
    Path(args.md).write_text("\n".join(L) + "\n")
    print(json.dumps({n: {k: s[k] for k in ("A", "B", "D", "D_cos_min", "reset_bit_equal", "pass", "red_all_red")}
                      for n, s in out["gates"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
