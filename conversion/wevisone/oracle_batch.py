#!/usr/bin/env python3
"""Run the WeVisOne-VL-2B oracle over every fixture row and form, a few one-thread processes at a time.

Each form is one task (its rows without a finished json); up to `--workers` tasks run at once, each as an
`oracle_wevisone.py` process with one CPU thread. A task started while time remains gets the rest of the
batch's `--deadline-s` as its own deadline (it finishes the row it is on and stops), so one invocation fits
in one measurement window and the next invocation resumes where this one stopped.

    python3 oracle_batch.py --python <oracle venv python> --pkg <dir> --model <snapshot> \
        --fixtures <fixtures.json> --out <oracle dir> --logs <dir> [--forms O1,O2-448,...] [--workers 4] \
        [--deadline-s 1500] [--hidden-ids a,b]
Exit 0 = every row of every form has a json; 3 = rows remain (run it again); 1 = a task failed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["O1", "O2-448", "O2-512", "O2-448-rp105", "O2-448-bf16"]
DTYPE = {"O2-448-bf16": "bfloat16"}


def remaining(out: Path, form: str, ids: list[str]) -> list[str]:
    return [i for i in ids if not (out / form / f"{i}.json").exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--forms", default=",".join(ORDER))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline-s", type=float, default=1500.0)
    ap.add_argument("--hidden-ids", default="")
    args = ap.parse_args()

    t0 = time.monotonic()
    out, logs = Path(args.out), Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in json.loads(Path(args.fixtures).read_text())["rows"]]
    forms = [f for f in args.forms.split(",") if f]
    queue = [(f, remaining(out, f, ids)) for f in forms]
    queue = [(f, r) for f, r in queue if r]
    print(f"[batch] {sum(len(r) for _, r in queue)} rows to run over {len(queue)} forms", flush=True)
    running: list[tuple[str, subprocess.Popen, object]] = []
    failed = []
    while queue or running:
        left = args.deadline_s - (time.monotonic() - t0)
        while queue and len(running) < args.workers and left > 60:
            form, rows = queue.pop(0)
            cmd = [args.python, "-I", str(HERE / "oracle_wevisone.py"), "--pkg", args.pkg, "--model", args.model,
                   "--fixtures", args.fixtures, "--out", args.out, "--dtype", DTYPE.get(form, "float32"),
                   "--jobs", ",".join(f"{form}:{r}" for r in rows), "--deadline-s", f"{left:.0f}"]
            if args.hidden_ids:
                cmd += ["--hidden-ids", args.hidden_ids]
            log = open(logs / f"oracle_{form}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start {len(rows)} rows, deadline {left:.0f}s\n")
            log.flush()
            running.append((form, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[batch] {time.strftime('%H:%M:%S')} start {form}: {len(rows)} rows (deadline {left:.0f}s)",
                  flush=True)
        if not running:
            break
        time.sleep(2)
        for item in list(running):
            form, proc, log = item
            rc = proc.poll()
            if rc is None:
                continue
            running.remove(item)
            log.close()
            print(f"[batch] {time.strftime('%H:%M:%S')} {form} exited {rc}", flush=True)
            if rc not in (0, 3):
                failed.append((form, rc))
    rest = {f: len(remaining(out, f, ids)) for f in forms}
    print(f"[batch] {time.strftime('%H:%M:%S')} done in {time.monotonic() - t0:.0f}s; rows left {rest}; "
          f"failed {failed}", flush=True)
    if failed:
        return 1
    return 3 if any(rest.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
