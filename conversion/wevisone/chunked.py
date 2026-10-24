#!/usr/bin/env python3
"""Run a per-row python-runtime script over fixture rows in sequential chunks, then merge the chunk records.

The Core AI python runtime leaks an IOSurface per call and dies near 25k S=1 calls in one process, so a gate over
every row runs as several processes, one after another (never two GPU processes at once). Each chunk gets
`--rows <ids>` and `--out <out>.cK.json`; the merged record has every chunk's `rows`, the first chunk's other keys,
and `chunks` = each chunk's summary keys.

    python3 chunked.py --python <venv python> --script gate_decoder.py --fixtures <json> --size 10 \
        --out <json> [--red-first a,b] -- <script args except --rows / --out>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="", help="subset of row ids (default: all)")
    args = ap.parse_args(own)

    rows = [r["id"] for r in json.loads(Path(args.fixtures).read_text())["rows"]]
    if args.rows:
        want = set(args.rows.split(","))
        rows = [r for r in rows if r in want]
    chunks = [rows[i:i + args.size] for i in range(0, len(rows), args.size)]
    out = Path(args.out)
    merged: dict = {}
    rcs = []
    for k, chunk in enumerate(chunks):
        part = out.with_suffix(f".c{k}.json")
        cmd = [args.python, str(HERE / args.script), *rest, "--fixtures", args.fixtures, "--rows", ",".join(chunk),
               "--out", str(part)]
        if k > 0 and "--red-ids" in cmd:            # red arms run once, in the first chunk
            i = cmd.index("--red-ids")
            del cmd[i:i + 2]
        t0 = time.monotonic()
        print(f"[chunked] {time.strftime('%H:%M:%S')} chunk {k + 1}/{len(chunks)}: {len(chunk)} rows", flush=True)
        rc = subprocess.run(cmd).returncode
        rcs.append(rc)
        if not part.exists():
            print(f"[chunked] chunk {k} wrote no record (rc {rc})", flush=True)
            continue
        rec = json.loads(part.read_text())
        if not merged:
            merged = {kk: vv for kk, vv in rec.items() if kk != "rows"}
            merged["rows"] = {}
            merged["chunks"] = []
        merged["rows"].update(rec.get("rows", {}))
        merged["chunks"].append({"rows": chunk, "rc": rc, "seconds": time.monotonic() - t0,
                                 **{kk: rec[kk] for kk in ("summary", "result", "reset_check", "red_arms",
                                                           "decoder_reloads", "wall_seconds", "gpu_utilization")
                                    if kk in rec}})
    merged["chunk_returncodes"] = rcs
    out.write_text(json.dumps(merged, indent=1) + "\n")
    print(f"[chunked] merged {len(merged.get('rows', {}))} rows -> {out}; rc {rcs}", flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
