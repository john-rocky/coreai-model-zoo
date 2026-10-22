"""Score the kit sample's Swift host against the Python host (sample_ref.py), set by set.

Per window: the forecasts' max |Swift - Python| over the window's max |Python quantile| (the line: <= 1e-5), and
whether the graph inputs (patches, cos, sin, attn_mask) and the graph's raw output are bit-identical. Per group and
for the aggregate: MASE and CRPS of both hosts against the oracle with e2e_gate's own formulas, and each host's
verdict at the set's level (l2 for an fp32 graph: |ratio - 1| <= 0.001; l3 for fp16 / Neural Engine graphs:
ratio <= 1.010, no NaN / inf, monotone). A set passes when every window is within the line and both hosts get the
same verdict.

    python sample_score.py --fixtures <fixtures> --ref <dir> --set fp32:l2 --set ane:l3 [--swift swift] --out <json>

The Swift side writes <ref>/<tag>/<swift>/{forecast.f32.bin, inputs.f32.bin, raw.f32.bin, manifest.json} with the
same layouts as sample_ref.py and its own offsets.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import e2e_gate as E

LINE = 1e-5


def floats(path):
    return np.fromfile(path, dtype=np.float32)


def bit_diff(a, b):
    """Elements whose float32 bit patterns differ, and the largest |a - b| among the finite pairs."""
    if a.size != b.size:
        return dict(size_a=int(a.size), size_b=int(b.size), differ=None, max_abs=None)
    neq = a.view(np.uint32) != b.view(np.uint32)
    fin = np.isfinite(a) & np.isfinite(b) & neq
    return dict(differ=int(neq.sum()), max_abs=float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0)


def verdict(ratios, level, nan_inf, nonmono):
    rs = [r for r in ratios if r is not None]
    if level == "l2":
        return all(abs(r - 1.0) <= 1e-3 for r in rs)
    return all(r <= 1.010 for r in rs) and nan_inf == 0 and nonmono == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--set", action="append", default=[], help="tag:l2|l3")
    ap.add_argument("--swift", default="swift", help="the Swift output folder inside each set")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    wj = json.loads((a.ref / "windows.json").read_text())
    by_id = {w["id"]: w for w in wj["windows"]}
    contexts = floats(a.ref / wj["contexts_file"])
    targets, oracle = {}, {}

    def fixture(group):
        if group not in targets:
            targets[group] = np.load(a.fixtures / f"{group}.target.npy")
            oracle[group] = np.load(a.fixtures / f"{group}.oracle_q.npy")[:, 0]
        return targets[group], oracle[group]

    results, lines = {}, []
    for item in a.set:
        tag, level = item.split(":")
        d = a.ref / tag
        py_m = json.loads((d / "manifest.json").read_text())
        sw_dir = d / a.swift
        if not (sw_dir / "manifest.json").exists():
            results[tag] = dict(error=f"no Swift output at {sw_dir}")
            lines.append(f"SCORE {tag} {a.swift}: MISSING ({sw_dir})")
            continue
        sw_m = json.loads((sw_dir / "manifest.json").read_text())
        py_fc, py_in, py_raw = (floats(d / n) for n in ("forecast.f32.bin", "inputs.f32.bin", "raw.f32.bin"))
        sw_fc = floats(sw_dir / "forecast.f32.bin")
        sw_in = floats(sw_dir / "inputs.f32.bin") if (sw_dir / "inputs.f32.bin").exists() else None
        sw_raw = floats(sw_dir / "raw.f32.bin") if (sw_dir / "raw.f32.bin").exists() else None
        sw_by_id = {e["id"]: e for e in sw_m["windows"]}
        n_ctx, n_hor = py_m["bucket"]
        s = n_ctx + n_hor
        seg = [("patches", 2 * n_ctx * 64), ("cos", s * 64), ("sin", s * 64), ("attn_mask", s * s)]
        per, groups = [], {}
        worst, nan_inf, nonmono, missing = 0.0, 0, 0, []
        in_diff = {k: 0 for k, _ in seg}
        in_max = {k: 0.0 for k, _ in seg}
        raw_differ, raw_max, raw_windows_differ = 0, 0.0, 0
        for e in py_m["windows"]:
            se = sw_by_id.get(e["id"])
            if se is None or se.get("error"):
                missing.append(dict(id=e["id"], error=(se or {}).get("error", "no Swift forecast")))
                continue
            hh = e["horizon"]
            qp = py_fc[e["forecast_offset"]:e["forecast_offset"] + e["forecast_floats"]].reshape(hh, 21)
            qs = sw_fc[se["forecast_offset"]:se["forecast_offset"] + se["forecast_floats"]].reshape(hh, 21)
            diff = np.abs(qs.astype(np.float64) - qp.astype(np.float64))
            rel = float(diff.max() / max(float(np.abs(qp).max()), 1e-30))
            worst = max(worst, rel)
            bad = int((~np.isfinite(qs)).sum())
            mono = bool((np.diff(qs, axis=-1) >= 0).all())
            nan_inf += bad
            nonmono += 0 if mono else 1
            row = dict(id=e["id"], max_rel=rel, max_abs=float(diff.max()), within_line=rel <= LINE, nan_inf=bad,
                       monotone=mono, swift_ms=se.get("ms"), python_ms=e.get("ms"))
            if sw_in is not None and "inputs_offset" in se:
                po, so = e["inputs_offset"], se["inputs_offset"]
                ins = {}
                for k, n in seg:
                    bd = bit_diff(py_in[po:po + n], sw_in[so:so + n])
                    ins[k] = bd
                    if bd["differ"] is not None:
                        in_diff[k] += bd["differ"]
                        in_max[k] = max(in_max[k], bd["max_abs"])
                    po += n
                    so += n
                row["inputs"] = ins
            if sw_raw is not None and "raw_offset" in se:
                n = e["raw_floats"]
                bd = bit_diff(py_raw[e["raw_offset"]:e["raw_offset"] + n], sw_raw[se["raw_offset"]:se["raw_offset"] + n])
                row["raw"] = bd
                if bd["differ"]:
                    raw_differ += bd["differ"]
                    raw_windows_differ += 1
                    raw_max = max(raw_max, bd["max_abs"])
            per.append(row)
            w = by_id[e["id"]]
            ctx = contexts[w["context_offset"]:w["context_offset"] + w["context_length"]]
            ys, orc = fixture(e["group"])
            g = groups.setdefault(e["group"], dict(ctx=[], y=[], sw=[], py=[], orc=[], m=w["seasonality"]))
            g["ctx"].append(ctx)
            g["y"].append(ys[e["window"]])
            g["sw"].append(qs)
            g["py"].append(qp)
            g["orc"].append(orc[e["window"]])
        gres = []
        for name, g in groups.items():
            m_sw = E.group_metrics(g["ctx"], g["y"], g["sw"], g["m"])
            m_py = E.group_metrics(g["ctx"], g["y"], g["py"], g["m"])
            m_or = E.group_metrics(g["ctx"], g["y"], g["orc"], g["m"])
            gres.append(dict(group=name, windows=len(g["y"]), swift=m_sw, python=m_py, oracle=m_or,
                             swift_mase_ratio=E.ratio(m_sw["mase"], m_or["mase"]),
                             swift_crps_ratio=E.ratio(m_sw["crps"], m_or["crps"]),
                             python_mase_ratio=E.ratio(m_py["mase"], m_or["mase"]),
                             python_crps_ratio=E.ratio(m_py["crps"], m_or["crps"])))

        def agg(who, key):
            vals = [(gr[who][key], gr["oracle"][key]) for gr in gres
                    if np.isfinite(gr[who][key]) and np.isfinite(gr["oracle"][key])]
            if not vals:
                return None
            return E.ratio(float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))

        ratios = {}
        for who in ("swift", "python"):
            rs = [gr[f"{who}_{k}_ratio"] for gr in gres for k in ("mase", "crps")]
            rs += [agg(who, "mase"), agg(who, "crps")]
            ratios[who] = rs
        py_nan = 0
        py_nonmono = 0
        for gname, g in groups.items():
            for q in g["py"]:
                py_nan += int((~np.isfinite(q)).sum())
                py_nonmono += 0 if bool((np.diff(q, axis=-1) >= 0).all()) else 1
        v_sw = verdict(ratios["swift"], level, nan_inf, nonmono) and not missing
        v_py = verdict(ratios["python"], level, py_nan, py_nonmono) and not py_m.get("errors")
        within = sum(1 for r in per if r["within_line"])
        passed = within == len(per) and not missing and v_sw == v_py
        worst_win = max(per, key=lambda r: r["max_rel"])["id"] if per else None
        res = dict(tag=tag, level=level, line=LINE, passed=passed, windows=len(per), within_line=within,
                   missing=missing, worst_max_rel=worst, worst_window=worst_win,
                   swift_verdict=v_sw, python_verdict=v_py,
                   swift_aggregate=dict(mase=agg("swift", "mase"), crps=agg("swift", "crps")),
                   python_aggregate=dict(mase=agg("python", "mase"), crps=agg("python", "crps")),
                   swift_worst_group_ratio=max([r for r in ratios["swift"] if r is not None], default=None),
                   python_worst_group_ratio=max([r for r in ratios["python"] if r is not None], default=None),
                   nan_inf=nan_inf, non_monotone=nonmono,
                   inputs_elements_differing=in_diff if sw_in is not None else None,
                   inputs_max_abs=in_max if sw_in is not None else None,
                   raw_elements_differing=raw_differ if sw_raw is not None else None,
                   raw_windows_differing=raw_windows_differ if sw_raw is not None else None,
                   raw_max_abs=raw_max if sw_raw is not None else None,
                   swift_median_ms=float(np.median([r["swift_ms"] for r in per if r["swift_ms"] is not None]))
                   if per else None,
                   python_median_ms=py_m.get("median_ms"), swift_bucket=sw_m.get("bucket"),
                   swift_rows=sw_m.get("rows"), swift_input_type=sw_m.get("input_type"),
                   python_set=py_m["set"], groups=gres, per_window=per)
        results[tag] = res
        lines.append(
            f"SCORE {tag} {a.swift} [{level}]: {'PASS' if passed else 'FAIL'} windows {within}/{len(per)} within "
            f"{LINE:g} (worst {worst:.3e} at {worst_win}), verdict swift {'PASS' if v_sw else 'FAIL'} / python "
            f"{'PASS' if v_py else 'FAIL'}, worst group ratio swift {res['swift_worst_group_ratio']} python "
            f"{res['python_worst_group_ratio']}, inputs differing {in_diff if sw_in is not None else '-'}, raw differing "
            f"{raw_differ if sw_raw is not None else '-'} in {raw_windows_differ} windows, missing {len(missing)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o)
                     + "\n")
    for line in lines:
        print(line, flush=True)
    print("SAMPLE_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
