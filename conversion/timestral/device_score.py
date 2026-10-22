"""Score one TimestralGate run (the iPhone app or the Mac tool) with the port's own measures (engine_l1.py, e2e_gate.py). NumPy only.

    PY device_score.py --run <run dir: run.json + pass1/quantiles.f32.bin> --inputs <inputs dir> --fixtures <dir>
        --refs <dir of ref_*.npz> [--vs <label>=<run dir> ...] [--mac-l1 <json>] [--mac-e2e <json>] --out <json>

The run's output is [N windows, 2, h*32, 21] float32 in the inputs pack's window order (row 0 = the series, row 1 =
its mirror; an R = 1 graph's two calls side by side). Measured:
  L1   vs the released module's fp32 output on the same inputs (ref_<bucket>.npz `ref32`, the inputs bit-equal to the
       pack): max |d|, the median channel's max |d| and correlation, NaN / inf, monotone over the 21 levels; PASS =
       finite, monotone and max |d| <= the npz's bar (engine_l1.py's rule). No reference npz (L64_h4) = no L1.
  L3   the forecasts the host makes from these outputs (host_forecast.forecast with a graph that replays the run,
       batched for an R = 2 graph, split for R = 1, the inputs checked equal to the pack on every call), MASE and CRPS
       per group and on the aggregate by e2e_gate.group_metrics, ratio = ours / oracle: L3 PASS = every ratio <=
       1.010, no NaN / inf, every forecast monotone, no window the host refused; L2 (recorded) = every
       |ratio - 1| <= 0.001 and no refused window.
  pass 1 vs pass 2 (the same process, bit for bit), and each --vs run vs this one (bit-different elements, max |d|).
  L4   when the Mac's verdict files are given (the Mac's engine_l1.py / e2e_gate.py records, or a scored Mac-tool run with
       --mac-swift): the same PASS / FAIL for L1 and L3.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import e2e_gate as E
import host_forecast as H

L3_MAX = 1.010
L2_TOL = 1e-3


def load_run(d):
    d = Path(d)
    rec = json.loads((d / "run.json").read_text())
    if rec.get("error"):
        raise SystemExit(f"{d}/run.json carries an error: {rec['error']}")
    out = rec["output"]
    shape = [out["windows"], 2, out["horizon_steps"], out["quantiles"]]
    passes = []
    for p in out["passes"]:
        a = np.fromfile(d / p["file"], dtype="<f4")
        if a.size != np.prod(shape):
            raise SystemExit(f"{d / p['file']}: {a.size} floats, expected {shape}")
        passes.append(a.reshape(shape))
    return rec, passes


def bits_compare(a, b):
    if a.shape != b.shape:
        return dict(comparable=False, shapes=[list(a.shape), list(b.shape)])
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    diff = ua != ub
    fin = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(comparable=True, elements=int(a.size), bit_identical=bool(not diff.any()),
                bit_different_elements=int(diff.sum()),
                max_abs=float(d[fin & diff].max()) if (fin & diff).any() else 0.0,
                nonfinite_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                windows_bit_identical=int((~diff.reshape(a.shape[0], -1).any(-1)).sum()))


class Replay:
    """A graph for host_forecast that returns the run's output for the current window, after checking that the host
    asks with exactly the packed inputs (a wrong window or a wrong row order stops the scoring)."""

    def __init__(self, inputs):
        self.inp = inputs
        self.n = None
        self.calls = 0

    def __call__(self, patches, cos, sin, attn_mask):
        n, inp = self.n, self.inp
        assert np.array_equal(cos, inp["cos"][n]) and np.array_equal(sin, inp["sin"][n]), "cos / sin differ from the pack"
        assert np.array_equal(attn_mask, inp["mask"][n]), "attn_mask differs from the pack"
        self.calls += 1
        if patches.shape[0] == 2:
            assert np.array_equal(patches, inp["patches"][n]), "patches differ from the pack"
            return self.out[n].copy()
        for r in (0, 1):
            if np.array_equal(patches[0], inp["patches"][n, r]):
                return self.out[n, r:r + 1].copy()
        raise AssertionError("an R = 1 call with patches that are neither packed row")


def load_inputs(d):
    man = json.loads((d / "manifest.json").read_text())
    n, s, l = len(man["windows"]), man["n_ctx"] + man["n_hor"], man["n_ctx"]
    shapes = dict(patches=(n, 2, l, 64), cos=(n, s, 64), sin=(n, s, 64), mask=(n, 1, 1, s, s))
    return man, {k: np.fromfile(d / f"{k}.f32.bin", dtype="<f4").reshape(v) for k, v in shapes.items()}


def l1(out, z, windows):
    assert [str(g) for g in z["group"]] == [w["group"] for w in windows]
    rows, nan_inf, nonmono, corr_min = [], 0, 0, 1.0
    worst = dict(max_abs=0.0, max_abs_vs_true64=0.0, median_max=0.0)
    for n, w in enumerate(windows):
        o, r = out[n].astype(np.float64), z["ref32"][n].astype(np.float64)
        bad = int((~np.isfinite(o)).sum())
        d = np.abs(o - r)
        mo, mr = o[..., H.MEDIAN].ravel(), r[..., H.MEDIAN].ravel()
        c = float(np.corrcoef(mo, mr)[0, 1]) if bad == 0 and mr.std() > 0 else None
        mono = bool((np.diff(o, axis=-1) >= 0).all()) if bad == 0 else False
        row = dict(id=w["id"], nan_inf=bad, monotone=mono, median_corr=c,
                   max_abs=float(np.nanmax(d)) if bad < d.size else None,
                   max_abs_vs_true64=float(np.nanmax(np.abs(o - z["ref64t"][n]))) if bad < d.size else None,
                   median_max=float(np.nanmax(np.abs(mo - mr))) if bad < d.size else None)
        rows.append(row)
        nan_inf += bad
        nonmono += 0 if mono else 1
        for k in worst:
            if row[k] is not None:
                worst[k] = max(worst[k], row[k])
        if c is not None:
            corr_min = min(corr_min, c)
    bar = float(z["bar"])
    return dict(windows=len(rows), bar=bar, **worst, median_corr_min=corr_min, nan_inf=nan_inf, non_monotone=nonmono,
                l1_pass=bool(nan_inf == 0 and nonmono == 0 and worst["max_abs"] <= bar), rows=rows)


def l3(out, inputs, man, fixtures, rows_graph):
    fx = {g["name"]: g for g in json.loads((fixtures / "manifest.json").read_text())["groups"]}
    replay = Replay(inputs)
    replay.out = out
    bucket = H.Bucket(man["n_ctx"], man["n_hor"])
    mode = "batched" if rows_graph == 2 else "split"
    by_group = {}
    for n, w in enumerate(man["windows"]):
        by_group.setdefault(w["group"], []).append((n, w))
    groups, failures, n_fc, nan_inf, nonmono, worst_q = [], [], 0, 0, 0, 0.0
    for name, items in by_group.items():
        g = fx[name]
        ctxs = np.load(fixtures / f"{name}.context.npy")
        ys = np.load(fixtures / f"{name}.target.npy")
        orc = np.load(fixtures / f"{name}.oracle_q.npy")[:, 0]
        ours, orcs, keep_c, keep_y, per_window = [], [], [], [], []
        for n, w in items:
            b = w["index"]
            if not np.isfinite(orc[b]).all():
                failures.append(dict(id=w["id"], error="the oracle has no finite forecast here"))
                continue
            replay.n = n
            try:
                q = H.forecast(replay, ctxs[b], g["horizon"], bucket, mode=mode)
            except ValueError as e:
                failures.append(dict(id=w["id"], error=str(e)))
                continue
            n_fc += 1
            bad = int((~np.isfinite(q)).sum())
            nan_inf += bad
            mono = bool((np.diff(q, axis=-1) >= 0).all())
            nonmono += 0 if mono else 1
            d = np.abs(q.astype(np.float64) - orc[b])
            rel = float(d.max() / max(np.abs(orc[b]).max(), 1e-30))
            worst_q = max(worst_q, rel)
            per_window.append(dict(id=w["id"], max_abs_diff=float(d.max()), max_rel_to_window=rel, nan_inf=bad,
                                   monotone=mono))
            ours.append(q)
            orcs.append(orc[b])
            keep_c.append(ctxs[b])
            keep_y.append(ys[b])
        if not ours:
            continue
        m_ours = E.group_metrics(keep_c, keep_y, ours, g["seasonality"])
        m_orc = E.group_metrics(keep_c, keep_y, orcs, g["seasonality"])
        groups.append(dict(group=name, windows=len(ours), horizon=g["horizon"], ours=m_ours, oracle=m_orc,
                           mase_ratio=E.ratio(m_ours["mase"], m_orc["mase"]),
                           crps_ratio=E.ratio(m_ours["crps"], m_orc["crps"]),
                           max_rel_to_window=max(x["max_rel_to_window"] for x in per_window), per_window=per_window))

    def agg(key):
        vals = [(gr["ours"][key], gr["oracle"][key]) for gr in groups
                if np.isfinite(gr["ours"][key]) and np.isfinite(gr["oracle"][key])]
        o, r = float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
        return dict(ours=o, oracle=r, ratio=E.ratio(o, r), groups=len(vals))

    aggregate = dict(mase=agg("mase"), crps=agg("crps"))
    ratios = [gr[k] for gr in groups for k in ("mase_ratio", "crps_ratio") if gr[k] is not None]
    ratios += [r for r in (aggregate["mase"]["ratio"], aggregate["crps"]["ratio"]) if r is not None]
    # a group can have no ratio at all (syn_constant: no seasonal error, a CRPS of 0 on both sides)
    worst_group = max(groups, key=lambda gr: max([r for r in (gr["mase_ratio"], gr["crps_ratio"]) if r is not None],
                                                 default=-math.inf))
    return dict(mode=mode, forecasts=n_fc, windows=len(man["windows"]), replay_calls=replay.calls, nan_inf=nan_inf,
                non_monotone=nonmono, failures=failures, aggregate=aggregate,
                max_ratio=max(ratios) if ratios else None,
                worst_group=dict(group=worst_group["group"], mase_ratio=worst_group["mase_ratio"],
                                 crps_ratio=worst_group["crps_ratio"]),
                max_abs_log_ratio=max(abs(math.log(r)) for r in ratios) if ratios else None,
                worst_max_rel_to_window=worst_q,
                l3_pass=bool(ratios and all(r <= L3_MAX for r in ratios) and nan_inf == 0 and nonmono == 0
                             and not failures),
                l2_pass=bool(ratios and all(abs(r - 1.0) <= L2_TOL for r in ratios) and not failures),
                groups=groups)


def mac_verdicts(a):
    v = {}
    if a.mac_l1:
        j = json.loads(Path(a.mac_l1).read_text())
        v["l1_pass"], v["l1_max_abs"], v["l1_source"] = j["l1_pass"], j["max_abs"], str(a.mac_l1)
    if a.mac_e2e:
        j = json.loads(Path(a.mac_e2e).read_text())
        ratios = [g[k] for g in j["groups"] for k in ("mase_ratio", "crps_ratio") if g[k] is not None]
        ratios += [r for r in (j["aggregate"]["mase"]["ratio"], j["aggregate"]["crps"]["ratio"]) if r is not None]
        v["l3_pass"] = bool(ratios and all(r <= L3_MAX for r in ratios) and j["nan_inf"] == 0
                            and j["non_monotone"] == 0 and not j["failures"])
        v["l3_max_ratio"] = max(ratios) if ratios else None
        v["l3_source"] = str(a.mac_e2e)
    if a.mac_swift:
        j = json.loads(Path(a.mac_swift).read_text())
        if "l1_pass" not in v and j.get("l1"):
            v["l1_pass"], v["l1_max_abs"], v["l1_source"] = j["l1"]["l1_pass"], j["l1"]["max_abs"], str(a.mac_swift)
        if "l3_pass" not in v and j.get("l3"):
            v["l3_pass"], v["l3_max_ratio"], v["l3_source"] = j["l3"]["l3_pass"], j["l3"]["max_ratio"], str(a.mac_swift)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path, help="the inputs dir holding <bucket>/")
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--vs", action="append", default=[], help="<label>=<run dir>, compared bit for bit")
    ap.add_argument("--mac-l1", type=Path)
    ap.add_argument("--mac-e2e", type=Path)
    ap.add_argument("--mac-swift", type=Path, help="a scored Mac-tool run of the same form (used where the Python records have none)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rec, passes = load_run(a.run)
    b = rec["bundle"]
    bucket = rec["output"]["bucket"]
    man, inputs = load_inputs(a.inputs / bucket)
    if rec["output"].get("inputs_manifest_md5") and rec["output"]["inputs_manifest_md5"] != E_md5(a.inputs / bucket / "manifest.json"):
        raise SystemExit("the run used another inputs manifest than this one")
    ids = [w["id"] for w in man["windows"]]
    if rec["output"]["window_ids"] != ids:
        raise SystemExit("the run's window order differs from the pack")
    out = passes[0]
    res = dict(run=str(a.run), tag=rec["tag"], device=rec["device"], bundle=b["bundle"], route=b["route"],
               compute=b["compute"], io_dtype=b["io_dtype"], rows_graph=b["rows"], bucket=bucket,
               load_ms=b["load_ms"], cold_requested=b["cold_requested"],
               passes=len(passes),
               passes_bit_identical=[bits_compare(passes[0], p)["bit_identical"] for p in passes[1:]],
               nan_inf_output=int((~np.isfinite(out)).sum()))
    ref = a.refs / f"ref_{bucket}.npz"
    res["l1"] = l1(out, np.load(ref), man["windows"]) if ref.exists() else None
    res["l3"] = l3(out, inputs, man, a.fixtures, b["rows"])
    res["vs"] = {}
    for spec in a.vs:
        label, d = spec.split("=", 1)
        _, other = load_run(d)
        res["vs"][label] = dict(run=d, **bits_compare(out, other[0]))
    mac = mac_verdicts(a)
    res["mac"] = mac
    l4 = {}
    if res["l1"] is not None and "l1_pass" in mac:
        l4["l1_same"] = res["l1"]["l1_pass"] == mac["l1_pass"]
    if "l3_pass" in mac:
        l4["l3_same"] = res["l3"]["l3_pass"] == mac["l3_pass"]
    res["l4"] = dict(**l4, l4_pass=bool(l4) and all(l4.values())) if l4 else dict(l4_pass=None, reason="no Mac verdict")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    show = dict(bundle=res["bundle"], route=res["route"], bucket=bucket, passes_bit_identical=res["passes_bit_identical"],
                l1=None if res["l1"] is None else {k: res["l1"][k] for k in ("max_abs", "bar", "nan_inf", "non_monotone",
                                                                              "median_corr_min", "l1_pass")},
                l3={k: res["l3"][k] for k in ("forecasts", "nan_inf", "non_monotone", "max_ratio", "worst_group",
                                              "l3_pass", "l2_pass")} | dict(failures=len(res["l3"]["failures"]),
                                                                           mase=res["l3"]["aggregate"]["mase"]["ratio"],
                                                                           crps=res["l3"]["aggregate"]["crps"]["ratio"]),
                vs={k: {kk: v[kk] for kk in ("bit_identical", "bit_different_elements", "max_abs") if kk in v}
                    for k, v in res["vs"].items()}, mac=mac, l4=res["l4"])
    print(json.dumps(show, indent=1))
    print("SCORE_DONE")


def E_md5(p):
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
