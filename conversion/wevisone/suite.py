#!/usr/bin/env python3
"""Structured-output suite for WeVisOne-VL-2B: score one set of answers against another, task by task.

Answers are the package's cut response strings (`<|im_end|>` / `<|endoftext|>` removed, stripped). The model writes
`<ref>label</ref>` then a `<box>` block of `<x1><y1><x2><y2>` groups separated by commas (a `<OBJ>` after a group on
segmentation / matting rows), or a `<point>` block of `<x><y>` groups; counting rows end with `Total number: N.`;
an absent target is `<box>none</box>` (or a bare `None`). Coordinates are 0..1000 of the image side.

Metrics (fixed in ROUND1 before any Core AI answer existed):
  box     per label, reference and candidate boxes paired greedily by IoU (largest first, one to one); a reference
          box counts as found when its pair has IoU >= 0.75. A row whose reference has no box is one "empty" unit,
          found when the candidate has no box either. Extra boxes = candidate boxes without such a pair.
          Rows: detection, referring, ocr_grounding, segmentation, matting, count_box.
  count   count_box / count_point rows: the number of boxes / points equal (exact). The `Total number` values are
          compared too (recorded).
  point   per label, reference and candidate points paired greedily by distance; found when |dx| <= 20 and
          |dy| <= 20 (2 % of the side). Empty reference rows as for boxes. Rows: point, count_point.
  ocr     ocr rows: the labels in output order joined by spaces; CER = Levenshtein / reference characters, summed over
          rows (micro); the word boxes paired by IoU regardless of text, found at IoU >= 0.75.
  obj     segmentation / matting rows: the number of `<OBJ>` equal (recorded).
Bars: box >= 0.90, count >= 0.90, point >= 0.90, CER <= 0.05, or twice the fp32-vs-bf16 wobble of the oracle
(O2-448), whichever is looser (`bar`).

    python3 suite.py pair --ref <answers> --cand <answers> --fixtures <json>
    python3 suite.py bar --oracle <dir> --fixtures <json>
    python3 suite.py tables --oracle <dir> --fixtures <json> --coreai g14=<json>,g16=<json> --out <json> --md <md>
<answers> = an oracle form dir (`<id>.json` with `response`) or a suite_coreai json (`rows.<id>.response`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IOU_BAR = 0.75
POINT_TOL = 20
LINE = {"box": 0.90, "count": 0.90, "point": 0.90, "cer": 0.05}
BOX_TASKS = {"detection", "referring", "ocr_grounding", "segmentation", "matting", "count_box"}
POINT_TASKS = {"point", "count_point"}
COUNT_TASKS = {"count_box", "count_point"}
OBJ_TASKS = {"segmentation", "matting"}

REF_RE = re.compile(r"<ref>(.*?)</ref>", re.S)
BLOCK_RE = re.compile(r"<(box|point|polygon)>(.*?)(?:</\1>|$)", re.S)
NUM_RE = re.compile(r"<(\d{1,4})>")
TOTAL_RE = re.compile(r"Total number:\s*(\d+)", re.I)
NONE_RE = re.compile(r"<(?:box|point)>\s*none\s*</(?:box|point)>", re.I)


def parse(text: str) -> dict:
    """-> {"boxes": [(label, [x1,y1,x2,y2], has_obj)], "points": [(label, [x,y])], "words": [labels in order],
    "total": int | None, "none": bool, "n_obj": int}."""
    text = text or ""
    events = sorted([(m.start(), "ref", m.group(1).strip()) for m in REF_RE.finditer(text)]
                    + [(m.start(), m.group(1), m.group(2)) for m in BLOCK_RE.finditer(text)])
    label, boxes, points, words = "", [], [], []
    for _, kind, body in events:
        if kind == "ref":
            label = body
            words.append(body)
            continue
        for part in body.split(","):
            nums = [int(x) for x in NUM_RE.findall(part)]
            if kind == "box":
                for i in range(0, len(nums) - 3, 4):
                    boxes.append((label, nums[i:i + 4], "<OBJ>" in part))
            elif kind == "point":
                for i in range(0, len(nums) - 1, 2):
                    points.append((label, nums[i:i + 2]))
    m = TOTAL_RE.search(text)
    return {"boxes": boxes, "points": points, "words": words, "total": int(m.group(1)) if m else None,
            "none": (text.strip().lower().startswith("none") or bool(NONE_RE.search(text))) and not boxes and not points,
            "n_obj": text.count("<OBJ>")}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else (1.0 if a == b else 0.0)


def greedy(ref: list, cand: list, score, better_is_larger: bool) -> list[tuple[int, int, float]]:
    pairs = [(score(r, c), i, j) for i, r in enumerate(ref) for j, c in enumerate(cand)]
    pairs.sort(key=lambda t: -t[0] if better_is_larger else t[0])
    used_r, used_c, out = set(), set(), []
    for s, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j, s))
    return out


def by_label(items):
    d: dict[str, list] = {}
    for it in items:
        d.setdefault(it[0], []).append(it[1])
    return d


def score_boxes(ref_boxes, cand_boxes, label_aware: bool = True) -> dict:
    if not ref_boxes:
        return {"units": 1, "found": int(not cand_boxes), "extra": len(cand_boxes), "empty_ref": True}
    if label_aware:
        rl, cl = by_label(ref_boxes), by_label(cand_boxes)
    else:
        rl, cl = {"": [b[1] for b in ref_boxes]}, {"": [b[1] for b in cand_boxes]}
    found, extra, ious = 0, 0, []
    for lab, rs in rl.items():
        cs = cl.get(lab, [])
        m = greedy(rs, cs, iou, True)
        ok = [s for _, _, s in m if s >= IOU_BAR]
        found += len(ok)
        extra += len(cs) - len(ok)
        ious.extend(s for _, _, s in m)
    extra += sum(len(cs) for lab, cs in cl.items() if lab not in rl)
    return {"units": len(ref_boxes), "found": found, "extra": extra, "empty_ref": False,
            "iou_min_paired": min(ious) if ious else None}


def score_points(ref_pts, cand_pts) -> dict:
    if not ref_pts:
        return {"units": 1, "found": int(not cand_pts), "extra": len(cand_pts), "empty_ref": True}
    rl, cl = by_label(ref_pts), by_label(cand_pts)
    found, extra = 0, 0
    for lab, rs in rl.items():
        cs = cl.get(lab, [])
        m = greedy(rs, cs, lambda r, c: ((r[0] - c[0]) ** 2 + (r[1] - c[1]) ** 2) ** 0.5, False)
        ok = [(i, j) for i, j, _ in m if abs(rs[i][0] - cs[j][0]) <= POINT_TOL and abs(rs[i][1] - cs[j][1]) <= POINT_TOL]
        found += len(ok)
        extra += len(cs) - len(ok)
    extra += sum(len(cs) for lab, cs in cl.items() if lab not in rl)
    return {"units": len(ref_pts), "found": found, "extra": extra, "empty_ref": False}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_answers(path: str) -> dict[str, dict]:
    p = Path(path)
    if p.is_dir():
        out = {}
        for f in sorted(p.glob("*.json")):
            o = json.loads(f.read_text())
            out[o["id"]] = {"response": o["response"], "hit_max": o.get("hit_max", False), "gen_ids": o.get("gen_ids")}
        return out
    d = json.loads(p.read_text())
    return {k: {"response": v["response"], "hit_max": v.get("hit_max", False), "gen_ids": v.get("gen_ids")}
            for k, v in d["rows"].items() if "response" in v}


def score_pair(ref: dict, cand: dict, rows: list[dict]) -> dict:
    tot = {"box": [0, 0], "point": [0, 0], "count": [0, 0], "total_number": [0, 0], "obj": [0, 0],
           "ocr_edits": 0, "ocr_chars": 0, "ocr_box": [0, 0], "extra_boxes": 0, "extra_points": 0}
    per_task: dict[str, dict] = {}
    per_row = {}
    missing = []
    for row in rows:
        rid, task = row["id"], row["task"]
        if rid not in ref or rid not in cand:
            missing.append(rid)
            continue
        a, b = parse(ref[rid]["response"]), parse(cand[rid]["response"])
        rec = {"task": task, "identical_text": ref[rid]["response"] == cand[rid]["response"],
               "identical_ids": (ref[rid].get("gen_ids") is not None and ref[rid].get("gen_ids") == cand[rid].get("gen_ids")),
               "ref_hit_max": ref[rid]["hit_max"], "cand_hit_max": cand[rid]["hit_max"]}
        t = per_task.setdefault(task, {"rows": 0, "identical_text": 0, "units": 0, "found": 0, "extra": 0,
                                       "count_equal": 0, "cer_edits": 0, "cer_chars": 0})
        t["rows"] += 1
        t["identical_text"] += rec["identical_text"]
        if task in BOX_TASKS:
            s = score_boxes(a["boxes"], b["boxes"])
            rec["box"] = s
            tot["box"][0] += s["found"]
            tot["box"][1] += s["units"]
            tot["extra_boxes"] += s["extra"]
            t["units"] += s["units"]
            t["found"] += s["found"]
            t["extra"] += s["extra"]
        if task in POINT_TASKS:
            s = score_points(a["points"], b["points"])
            rec["point"] = s
            tot["point"][0] += s["found"]
            tot["point"][1] += s["units"]
            tot["extra_points"] += s["extra"]
            t["units"] += s["units"]
            t["found"] += s["found"]
            t["extra"] += s["extra"]
        if task in COUNT_TASKS:
            na = len(a["boxes"]) if task == "count_box" else len(a["points"])
            nb = len(b["boxes"]) if task == "count_box" else len(b["points"])
            rec["count"] = {"ref": na, "cand": nb, "equal": na == nb,
                            "total_ref": a["total"], "total_cand": b["total"]}
            tot["count"][0] += na == nb
            tot["count"][1] += 1
            t["count_equal"] += na == nb
            if a["total"] is not None:
                tot["total_number"][0] += a["total"] == b["total"]
                tot["total_number"][1] += 1
        if task in OBJ_TASKS:
            rec["obj"] = {"ref": a["n_obj"], "cand": b["n_obj"]}
            tot["obj"][0] += a["n_obj"] == b["n_obj"]
            tot["obj"][1] += 1
        if task == "ocr":
            ra, ca = " ".join(a["words"]), " ".join(b["words"])
            e = levenshtein(ca, ra)
            s = score_boxes(a["boxes"], b["boxes"], label_aware=False)
            rec["ocr"] = {"ref_text": ra, "cand_text": ca, "edits": e, "chars": len(ra), "box": s}
            tot["ocr_edits"] += e
            tot["ocr_chars"] += len(ra)
            tot["ocr_box"][0] += s["found"]
            tot["ocr_box"][1] += s["units"]
            t["cer_edits"] += e
            t["cer_chars"] += len(ra)
        per_row[rid] = rec
    frac = lambda x: (x[0] / x[1]) if x[1] else None  # noqa: E731
    summary = {"rows": len(per_row), "missing": missing,
               "identical_text_rows": sum(r["identical_text"] for r in per_row.values()),
               "box_found": frac(tot["box"]), "box_units": tot["box"][1], "extra_boxes": tot["extra_boxes"],
               "count_exact": frac(tot["count"]), "count_rows": tot["count"][1],
               "total_number_equal": frac(tot["total_number"]),
               "point_found": frac(tot["point"]), "point_units": tot["point"][1], "extra_points": tot["extra_points"],
               "cer": (tot["ocr_edits"] / tot["ocr_chars"]) if tot["ocr_chars"] else None,
               "ocr_chars": tot["ocr_chars"], "ocr_box_found": frac(tot["ocr_box"]),
               "obj_equal": frac(tot["obj"]),
               "ref_hit_max": [k for k, r in per_row.items() if r["ref_hit_max"]],
               "cand_hit_max": [k for k, r in per_row.items() if r["cand_hit_max"]]}
    return {"summary": summary, "per_task": per_task, "per_row": per_row}


def bar_from_wobble(wobble: dict) -> dict:
    """The looser of the launch line and twice the fp32-vs-bf16 wobble."""
    out = {}
    for key, metric in (("box", "box_found"), ("count", "count_exact"), ("point", "point_found")):
        agree = wobble.get(metric)
        two_x = None if agree is None else 1.0 - 2.0 * (1.0 - agree)
        out[key] = {"line": LINE[key], "twice_wobble": two_x,
                    "bar": LINE[key] if two_x is None else min(LINE[key], two_x)}
    cer = wobble.get("cer")
    two_x = None if cer is None else 2.0 * cer
    out["cer"] = {"line": LINE["cer"], "twice_wobble": two_x, "bar": LINE["cer"] if two_x is None else max(LINE["cer"], two_x)}
    return out


def verdict(summary: dict, bar: dict) -> dict:
    checks = {"box": (summary["box_found"], ">=", bar["box"]["bar"]),
              "count": (summary["count_exact"], ">=", bar["count"]["bar"]),
              "point": (summary["point_found"], ">=", bar["point"]["bar"]),
              "cer": (summary["cer"], "<=", bar["cer"]["bar"])}
    res = {}
    for k, (v, op, b) in checks.items():
        res[k] = {"value": v, "bar": b, "op": op,
                  "pass": None if v is None else (v >= b if op == ">=" else v <= b)}
    res["pass"] = all(r["pass"] is not False for r in res.values() if isinstance(r, dict))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["pair", "bar", "tables"])
    ap.add_argument("--ref")
    ap.add_argument("--cand")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--oracle")
    ap.add_argument("--coreai", default="", help="g14=<json>,g16=<json>")
    ap.add_argument("--out")
    ap.add_argument("--md")
    args = ap.parse_args()
    rows = json.loads(Path(args.fixtures).read_text())["rows"]
    if args.cmd == "pair":
        r = score_pair(load_answers(args.ref), load_answers(args.cand), rows)
        print(json.dumps(r["summary"], indent=1))
        return 0
    o = Path(args.oracle)
    wob = score_pair(load_answers(str(o / "O2-448")), load_answers(str(o / "O2-448-bf16")), rows)
    bar = bar_from_wobble(wob["summary"])
    if args.cmd == "bar":
        print(json.dumps({"wobble_O2-448_fp32_vs_bf16": wob["summary"], "bar": bar}, indent=1))
        return 0
    tables = {"iv_fp32_vs_bf16": {"ref": "O2-448", "cand": "O2-448-bf16", **wob},
              "iii_rp105_vs_rp10": {"ref": "O2-448-rp105", "cand": "O2-448",
                                    **score_pair(load_answers(str(o / "O2-448-rp105")), load_answers(str(o / "O2-448")), rows)},
              "ii_O1_vs_O2-448": {"ref": "O1", "cand": "O2-448",
                                  **score_pair(load_answers(str(o / "O1")), load_answers(str(o / "O2-448")), rows)},
              "ii_O1_vs_O2-512": {"ref": "O1", "cand": "O2-512",
                                  **score_pair(load_answers(str(o / "O1")), load_answers(str(o / "O2-512")), rows)}}
    for item in [s for s in args.coreai.split(",") if s]:
        g, path = item.split("=", 1)
        form = {"g14": "O2-448", "g16": "O2-512"}[g]
        t = score_pair(load_answers(str(o / form)), load_answers(path), rows)
        t["verdict"] = verdict(t["summary"], bar)
        tables[f"i_coreai_{g}"] = {"ref": form, "cand": path, **t}
    tables["iii_rp105_vs_rp10"]["verdict_vs_bar"] = verdict(tables["iii_rp105_vs_rp10"]["summary"], bar)
    loops = {"O2-448_rp1.0_hit_max": tables["iii_rp105_vs_rp10"]["summary"]["cand_hit_max"],
             "O2-448_rp1.05_hit_max": tables["iii_rp105_vs_rp10"]["summary"]["ref_hit_max"]}
    loops["rp1.0_only"] = sorted(set(loops["O2-448_rp1.0_hit_max"]) - set(loops["O2-448_rp1.05_hit_max"]))
    out = {"schema": "wevisone-suite/1", "bar": bar, "tables": tables, "loops": loops,
           "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    if args.md:
        Path(args.md).write_text(render_md(out))
    print(json.dumps({k: v["summary"] | ({"verdict": v["verdict"]["pass"]} if "verdict" in v else {})
                      for k, v in tables.items()}, indent=1, default=str)[:4000])
    return 0


def render_md(out: dict) -> str:
    def f(x, pct=True):
        if x is None:
            return "—"
        return f"{100 * x:.1f} %" if pct else f"{x:.4f}"

    lines = ["# SUITE — WeVisOne-VL-2B structured outputs (Core AI r1)", "",
             f"Scoring code: `conversion/wevisone/suite.py` sha256 `{out['script_sha256']}`.", "",
             "Bars (the looser of the launch line and twice the O2-448 fp32-vs-bf16 wobble):", "",
             "| metric | launch line | 2x wobble | bar |", "|---|---|---|---|"]
    for k in ("box", "count", "point"):
        b = out["bar"][k]
        lines.append(f"| {k} | ≥ {f(b['line'])} | {f(b['twice_wobble'])} | ≥ {f(b['bar'])} |")
    b = out["bar"]["cer"]
    lines.append(f"| OCR CER | ≤ {b['line']:.3f} | {'—' if b['twice_wobble'] is None else format(b['twice_wobble'], '.4f')} | ≤ {b['bar']:.3f} |")
    lines += ["", "| table | ref | cand | rows | identical text | box IoU≥0.75 | extra boxes | count exact | point ≤2 % | OCR CER | OCR box | ref / cand hit max | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, t in out["tables"].items():
        s = t["summary"]
        v = t.get("verdict", {}).get("pass")
        lines.append(f"| {name} | {t['ref']} | {Path(str(t['cand'])).name} | {s['rows']} | {s['identical_text_rows']} | "
                     f"{f(s['box_found'])} ({s['box_units']}) | {s['extra_boxes']} | {f(s['count_exact'])} ({s['count_rows']}) | "
                     f"{f(s['point_found'])} ({s['point_units']}) | {'—' if s['cer'] is None else format(s['cer'], '.4f')} | "
                     f"{f(s['ocr_box_found'])} | {len(s['ref_hit_max'])} / {len(s['cand_hit_max'])} | "
                     f"{'—' if v is None else ('PASS' if v else 'FAIL')} |")
    lines += ["", f"Loops (max tokens reached): {json.dumps(out['loops'])}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
