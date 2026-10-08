#!/usr/bin/env python3
"""The Python runtime gates of the stripped bundles against the same gates of the unstripped ones (round 10).

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/compare_ship.py         # -> <work>/results/ship_runtime_compare.json (never replaced)
    $PY conversion/d1_omni/compare_ship.py --small # round 12: L64 / L128 -> <work>/results/small_runtime_compare.json

strip_debug_info (strip_ship.py) changes only the graphs' debug locations, so the runtime should return the same
numbers: every gate row's marker logits and probabilities equal (bit for bit: the JSON holds float32 values as exact
Python floats) and every summary number the same. Pairs (the unstripped run is the one that made the form shippable,
rounds 3 / 4 / 7; the stripped run used the same command on the ship AOT, compiled/ship-h16c/):
  decision L  results/runtime_fp16_L<L>_gpu.json                  vs results/ship_runtime_decide_fp16_L<L>_gpu.json
  vision      results/vision_fp16_gpu.run2.json (per image)       vs results/ship_runtime_vision_fp16_gpu.json
              results/vision_e2e_fp16.run4.json (46 rows, 3 arms) vs results/ship_runtime_vision_e2e_fp16.json
  audio <s>   results/audio_fp16_<s>s_gpu.run2.json (per clip)    vs results/ship_runtime_audio_fp16_<s>s_gpu.json
              results/audio_e2e_fp16.run4.json (46 rows, 4 arms)  vs results/ship_runtime_audio_e2e_fp16.json
Per pair: the rows paired by id / qid / mode (and set, when both runs record it: round 3's L256 run does not), the rows
paired, the rows whose logits and p are equal, max |d logit|, max |d p| (None when no row paired),
argmax changes; the summaries (argmax counts, max |dp|, the mean of the rows' max |dp|, max |dlogit|, the controls,
the drift) before and after. A prefix file carries statistics per item (cos, max |d|, min row cos), compared as numbers.
Status per pair: SAME (everything equal), WITHIN (every row paired, argmax counts equal and max |dp| within 1e-6: the
launch's stop rule), DIFFERS.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

WORK = work_path("_d1_omni")
RESULTS = WORK / "results"
STOP = 1e-6
SUMMARY_KEYS = ("rows", "argmax_equal", "max_abs_dp", "mean_row_max_abs_dp", "max_abs_dlogit")
PAIRS = {**{f"decide-fp16-L{L}": ("decide", f"runtime_fp16_L{L}_gpu.json", f"ship_runtime_decide_fp16_L{L}_gpu.json")
            for L in (256, 512, 1024, 2048, 4096)},
         "vision-fp16": ("prefix", "vision_fp16_gpu.run2.json", "ship_runtime_vision_fp16_gpu.json"),
         "vision-e2e": ("e2e", "vision_e2e_fp16.run4.json", "ship_runtime_vision_e2e_fp16.json"),
         **{f"audio-fp16-{s}s": ("prefix", f"audio_fp16_{s}s_gpu.run2.json", f"ship_runtime_audio_fp16_{s}s_gpu.json")
            for s in (5, 10, 20, 30)},
         "audio-e2e": ("e2e", "audio_e2e_fp16.run4.json", "ship_runtime_audio_e2e_fp16.json")}
# round 12: the small decision buckets, unstripped (macos/fp16-L<L>, AOT compiled/fp16-L<L>-h16c) against stripped
# (macos-ship-small/fp16-L<L>, AOT compiled/ship-h16c/fp16-L<L>), the same runtime_check.py command on each
PAIRS_SMALL = {f"decide-fp16-L{L}": ("decide", f"runtime_fp16_L{L}_gpu.json", f"ship_runtime_decide_fp16_L{L}_gpu.json")
               for L in (64, 128)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_compare(new: list[dict], old: list[dict]) -> dict:
    # Round 3's L256 run predates the "set" field (every row there is native): pair on the set only when both runs
    # record it, and only on keys that are unique in both.
    with_set = all("set" in r for r in (*new, *old))
    key = lambda r: (r["id"], r["qid"], r["mode"], r["set"] if with_set else None)  # noqa: E731
    before = {key(r): r for r in old}
    assert len(before) == len(old) and len({key(r) for r in new}) == len(new), "row keys are not unique"
    paired, equal, dl, dp, flips, missing, first = 0, 0, 0.0, 0.0, [], [], []
    for r in new:
        o = before.get(key(r))
        if o is None:
            missing.append(str(key(r)))
            continue
        paired += 1
        same = r["logits"] == o["logits"] and r["probs"] == o["probs"]
        equal += same
        if not same and len(first) < 10:
            first.append("/".join(str(k) for k in key(r) if k is not None))
        dl = max(dl, max(abs(a - b) for a, b in zip(r["logits"], o["logits"])))
        if r["probs"] is not None and o["probs"] is not None:
            dp = max(dp, max(abs(a - b) for a, b in zip(r["probs"], o["probs"])))
        if r["argmax_equal"] != o["argmax_equal"]:
            flips.append("/".join(str(k) for k in key(r) if k is not None))
    # No paired row measures nothing: report None, not a 0 that reads as "equal".
    return {"rows": len(new), "rows_before": len(old), "paired_on": "id/qid/mode" + ("/set" if with_set else ""),
            "paired": paired, "missing_before": missing[:10], "logits_and_p_equal": equal,
            "max_abs_dlogit": dl if paired else None, "max_abs_dp": dp if paired else None,
            "argmax_changed": flips, "first_different": first}


def summary_compare(new: dict, old: dict) -> dict:
    out = {k: {"before": old[k], "after": new[k], "diff": new[k] - old[k]} for k in SUMMARY_KEYS}
    for k in ("non_near_tie", "near_tie"):
        out[f"{k}_argmax_equal"] = {"before": old[k]["argmax_equal"], "after": new[k]["argmax_equal"],
                                    "diff": new[k]["argmax_equal"] - old[k]["argmax_equal"]}
    return out


def verdict(rows: list[dict], summaries: list[dict], extra_equal: bool = True) -> str:
    same = (extra_equal and all(r["logits_and_p_equal"] == r["rows"] == r["rows_before"] for r in rows)
            and all(v["diff"] == 0 for s in summaries for v in s.values()))
    if same:
        return "SAME"
    within = all(s[k]["diff"] == 0 for s in summaries for k in ("argmax_equal",)) and all(
        abs(s["max_abs_dp"]["diff"]) <= STOP for s in summaries) and not any(r["argmax_changed"] for r in rows) and all(
        r["paired"] == r["rows"] == r["rows_before"] for r in rows)
    return "WITHIN" if within else "DIFFERS"


def compare_decide(new: dict, old: dict) -> dict:
    rows = rows_compare(new["rows"], old["rows"])
    summary = summary_compare(new["summary"], old["summary"])
    extra = {"status": [old["status"], new["status"]],
             "wrong_pairing_control": [old["wrong_pairing_control"]["status"], new["wrong_pairing_control"]["status"]],
             "drift": [old["repeat_drift"]["max_abs_scores"], new["repeat_drift"]["max_abs_scores"]],
             "repeats": [old["repeat_drift"]["repeats_per_row"], new["repeat_drift"]["repeats_per_row"]],
             "main_hash": [old["compiled"]["hashes"]["main.hash"], new["compiled"]["hashes"]["main.hash"]]}
    same_extra = all(a == b for k, (a, b) in extra.items() if k != "main_hash")
    return {"rows": rows, "summary": summary, "other": extra,
            "status": verdict([rows], [summary], same_extra)}


def compare_prefix(new: dict, old: dict) -> dict:
    items = "images" if "images" in new else "clips"
    before = {x["id"]: x for x in old[items]}
    stats = ("prefix", "prefix_numpy_mel")
    equal, first = 0, []
    for x in new[items]:
        o = before.get(x["id"])
        same = o is not None and all(x.get(k) == o.get(k) for k in (*stats, "drift", "finite"))
        equal += same
        if not same and len(first) < 10:
            first.append(x["id"])
    status = "SAME" if equal == len(new[items]) == len(old[items]) and new["status"] == old["status"] else "DIFFERS"
    return {"items": len(new[items]), "items_before": len(old[items]), "stats_equal": equal, "first_different": first,
            "compared": [*stats, "drift", "finite"], "status_before_after": [old["status"], new["status"]],
            "status": status}


def compare_e2e(new: dict, old: dict) -> dict:
    arms = {}
    for arm in new["arms"]:
        rows = rows_compare(new["arms"][arm], old["arms"][arm])
        summary = summary_compare(new["summaries"][arm], old["summaries"][arm])
        arms[arm] = {"rows": rows, "summary": summary, "status": verdict([rows], [summary])}
    extra = {"status": [old["status"], new["status"]], "control_must_fail": [old["control_must_fail"], new["control_must_fail"]],
             "wrong_pairing_oracle_swap": [old["wrong_pairing_oracle_swap"]["status"], new["wrong_pairing_oracle_swap"]["status"]],
             "repeat_drift_max": [old["repeat_drift_max"], new["repeat_drift_max"]]}
    statuses = {a["status"] for a in arms.values()}
    status = ("SAME" if statuses == {"SAME"} and all(a == b for a, b in extra.values()) else
              "WITHIN" if statuses <= {"SAME", "WITHIN"} else "DIFFERS")
    return {"arms": arms, "other": extra, "status": status}


def main() -> int:
    small = "--small" in sys.argv[1:]
    out = RESULTS / ("small_runtime_compare.json" if small else "ship_runtime_compare.json")
    if out.exists():
        raise SystemExit(f"{out} exists: never replaced")
    pairs = {}
    for name, (kind, before, after) in (PAIRS_SMALL if small else PAIRS).items():
        old, new = json.loads((RESULTS / before).read_text()), json.loads((RESULTS / after).read_text())
        result = {"decide": compare_decide, "prefix": compare_prefix, "e2e": compare_e2e}[kind](new, old)
        pairs[name] = {"before": before, "after": after, "after_status": new["status"], **result}
        print(name, result["status"], flush=True)
    statuses = {p["status"] for p in pairs.values()}
    doc = {"schema": "d1-omni-ship-runtime-compare/1",
           "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "status": "SAME" if statuses == {"SAME"} else "WITHIN" if statuses <= {"SAME", "WITHIN"} else "DIFFERS",
           "stop_rule": f"argmax or max |dp| differing by more than {STOP} stops the round (launch r10)",
           "set": "round 12 small buckets (L64 / L128)" if small else "round 10 ship bundles",
           "after_all_pass": all(p["after_status"] == "PASS" for p in pairs.values()),
           "pairs": pairs, "code_sha256": {"compare_ship.py": sha256_file(HERE / "compare_ship.py")}}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(doc["status"], "after all PASS:", doc["after_all_pass"], "->", out)
    return 0 if doc["status"] in ("SAME", "WITHIN") and doc["after_all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
