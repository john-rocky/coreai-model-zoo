#!/usr/bin/env python3
"""Assemble spec-decode A/B runs into tables, EXCLUDING rows the machine spoiled.

Run it in the directory holding the `--json` summaries `spec-decode` wrote
(`_muse_spec_*.json`). Prints one row per (workload, config): acceptance, the mean
forward cost the run implies, and the speedup over that run's own baseline.

The reason this exists rather than a `cat` of the JSONs: on a machine that throttles
under sustained load, a row where only ONE half ran at full clock is worse than a
missing row, because its ratio looks like a result. A collapsed baseline inflates it
(one discarded row here read a flattering 2.10x off a 15.55 tok/s baseline); a
throttled spec-on half deflates it. So every row is validated and the bad ones are
named and dropped — see `valid()`. Put the check in the script, not in your judgement.
"""
import glob
import json
import re

DFLASH = 37.8  # Meta's published DFlash decode figure, M4 Max

# The machine's cool-state no-draft baseline, established over 30+ runs: 26.7-27.6 tok/s.
# A run whose spec-off half falls outside this band was throttled or contended, and its
# ratio is meaningless in EITHER direction (a slow baseline inflates it, a slow spec-on
# half deflates it). Such rows are flagged and excluded rather than quietly averaged in.
BASELINE_LO, BASELINE_HI = 26.0, 28.0
# A spec-on half whose mean forward cost sits far above what its draft length can reach on
# the staircase was throttled even if its own baseline looked fine. The ceiling has to be
# K-aware: K=8 feeds S=9, which legitimately costs ~85 ms, while K=2 can never leave ~38 ms.
def forward_ceiling(k):
    if k and k <= 2:
        return 46.0  # plateau 1 (S<=3, ~38 ms) + headroom
    if k and k <= 7:
        return 64.0  # plateau 2 (S=4-8, ~54 ms) + headroom
    return 95.0  # plateau 3 (S>=9, ~86 ms) + headroom


def rows():
    for path in sorted(glob.glob("_muse_spec_*.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "ab" not in d:
            continue
        ab = d["ab"]
        on = ab["on"][-1]
        off = ab["off"][-1]
        label = d.get("label", path)
        workload, _, config = label.partition(":")
        yield {
            "workload": workload,
            "config": config or "k8",
            "gen": on["generated"],
            "alpha": on["alpha"],
            "tok_fwd": on["tokens_per_forward"],
            "mean_k": on.get("mean_draft_budget", 0),
            "ms_fwd": 1000 * on.get("forward_seconds", 0) / max(on["target_forwards"], 1),
            "off": ab["off_best_tps"],
            "on": ab["on_best_tps"],
            "x": ab["speedup"],
            "lossless": ab["lossless"],
            "host_pct": 100 * on.get("host_seconds", 0) / max(on["decode_seconds"], 1e-9),
            "off_fwd_ms": 1000 * off.get("forward_seconds", 0) / max(off["target_forwards"], 1),
            "k": on.get("k", 0),
        }


def valid(r):
    """A row is usable only if BOTH halves ran at full clock — see BASELINE_LO."""
    if not (BASELINE_LO <= r["off"] <= BASELINE_HI):
        return False, "baseline"
    if r["ms_fwd"] and r["ms_fwd"] > forward_ceiling(r["k"]):
        return False, "fwd-cost"
    return True, ""


def main():
    data = list(rows())
    order = {"chat": 0, "code": 1, "tools": 2}
    data.sort(key=lambda r: (order.get(r["workload"], 9), r["config"]))

    print(
        f"{'workload':6} {'config':10} {'gen':>4} {'a-bar':>6} {'tok/fwd':>7} {'meanK':>6} "
        f"{'ms/fwd':>7} {'off':>6} {'on':>6} {'x':>5} {'vs DFlash':>9} {'host%':>6} loss"
    )
    for r in data:
        ok, why = valid(r)
        print(
            f"{r['workload']:6} {r['config']:10} {r['gen']:4d} {r['alpha']:6.2f} {r['tok_fwd']:7.2f} "
            f"{r['mean_k']:6.1f} {r['ms_fwd']:7.1f} {r['off']:6.2f} {r['on']:6.2f} {r['x']:5.2f} "
            f"{r['on'] / DFLASH:8.2f}x {r['host_pct']:5.1f}% {'ok' if r['lossless'] else 'FAIL'}"
            f"{'' if ok else '  <-- THROTTLED (' + why + '), excluded'}"
        )

    clean = [r for r in data if valid(r)[0]]
    dirty = [r for r in data if not valid(r)[0]]
    fails = [r for r in data if not r["lossless"]]
    print()
    print(f"{len(data)} A/B runs, lossless {len(data) - len(fails)}/{len(data)}")
    print(f"usable {len(clean)} · excluded {len(dirty)} (throttled/contended baseline)")
    offs = [r["off"] for r in clean]
    if offs:
        print(
            f"baseline (spec off) over usable runs: {min(offs):.2f}–{max(offs):.2f} tok/s, "
            f"mean {sum(offs)/len(offs):.2f} · shipped pipelined engine reference 26.69"
        )
    # Best usable config per workload.
    print("\nbest usable config per workload:")
    for w in ("chat", "code", "tools"):
        candidates = [r for r in clean if r["workload"] == w and r["gen"] >= 256]
        if candidates:
            b = max(candidates, key=lambda r: r["on"])
            print(
                f"  {w:6} {b['config']:14} {b['on']:6.2f} tok/s  {b['x']:.2f}x  "
                f"{b['on']/DFLASH:.2f}x DFlash"
            )

    # Verify-cost sweeps
    for path in sorted(glob.glob("_muse_spec_verifycost*.json")):
        d = json.load(open(path))
        print(f"\n{path}:")
        print(f"  {'S':>3} {'ms':>7} {'c_v':>6}  break-even accept")
        for row in d["verify_cost"]:
            print(
                f"  {row['s']:3d} {row['median_ms']:7.1f} {row['vs_decode_step']:6.2f}"
                f"  > {row['vs_decode_step']:.2f} tok/round"
            )


if __name__ == "__main__":
    main()
