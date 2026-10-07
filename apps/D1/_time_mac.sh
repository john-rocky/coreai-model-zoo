#!/bin/zsh
# Time the Release `d1` CLI on this Mac in one _GPU_LOCK window — apps/Kev/_time_mac.sh's window (its round-7 core:
# the lock, the GPU-job check, the snapshots), with d1's items (conversion/d1/timing.py's protocol, so the rows line up
# with its results/timing_<tag>.json). Written in round 3c; not run yet.
#   * the lock (~/code/coreai/_GPU_LOCK, ZOO_GPU_LOCK overrides) is taken only when it is empty, no other process holds it
#     open (lsof; the kit's scripts hold a flock(2) on it) and no other GPU job runs; polled every 30 s for D1_LOCK_WAIT
#     seconds (default 1800), then this script gives up with exit 3 (no contended run). Held: a flock(2) on the file and
#     the file says "<D1_TAG> pid <pid> since <time>"; emptied (0 B) and unlocked at the end, a failed run included.
#   * the items (timing.py ITEMS): one_question (card_refund's `refund` alone), three_shared (card_refund's three
#     questions, --shared), three_direct (the same, direct), state_3_4k (long_34k's first question alone). One `d1 decide`
#     process per item and form: it loads the asset, runs one warm-up call (--warm), decides once, then D1_REPS (20) more
#     (`reps` in its trace: each decision's graph / readout / wall seconds).
#   * D1_FORMS="label=bundle ..." (a name under <lane>/exports/bundles or an absolute path; default
#     "fp16=d1_3b_decode_fp16_pf16"), D1_ROUNDS rounds (default 2) of the forms A B .. A B ..; then, when D1_PYTIME=1,
#     the Python reference in the same window (`timing.py run --bundle <the first form>`).
#   * before, between and after the processes: the load average, top's CPU / memory lines, swap, the five busiest
#     processes and any other GPU job.
#   ./_time_mac.sh   -> <lane>/swift/timing/<run id>/{p<slot>_<label>_<item>.json, resp_*.json, *.log, window.json}
set -u
L=${D1_LANE:-$HOME/code/coreai/_d1_3b}
BIN=${D1_BIN:-$L/swift/.build/release/d1}
PY=${D1_PY:-$HOME/code/coreai/coreai-models/.venv/bin/python}
TIMING=${0:A:h}/../../conversion/d1/timing.py
[ -x $BIN ] || { echo "no Release build at $BIN (swift build -c release --scratch-path $L/swift/.build)"; exit 1; }
RUN_ID=${D1_RUN_ID:-r_$(date +%Y%m%d_%H%M%S)}
OUT=$L/swift/timing/$RUN_ID
mkdir -p $OUT
LOCK=${ZOO_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
echo "[$(date '+%H:%M:%S')] run $RUN_ID: $BIN (lock $LOCK)" | tee -a $OUT/run.out

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "$L" "$PY" "${TIMING:A}" "${D1_LOCK_WAIT:-1800}" "${D1_REPS:-20}" <<'PY' 2>&1 | tee -a $OUT/run.out
import fcntl, json, os, re, subprocess, sys, time
from datetime import datetime
lock, out, binary, lane, py, timing, wait_cap, reps = sys.argv[1:9]
wait_cap = float(wait_cap)
GPU = re.compile(r"readout_gate\.py|timing\.py (run|worker)|decide\.py (check|worker|run|e2e)|--accel gpu|gate_tower\.py|"
                 r"gate_swift\.py|llm-bench|yardstick|coreai_verify|mlx_lm|litert-lm (run|benchmark)|--backend gpu|"
                 r"/d1 (decide|fixture|prepare-test) |/kev (fixture|time|run) "
                 + ("|" + os.environ["D1_GPU_EXTRA"] if os.environ.get("D1_GPU_EXTRA") else ""))
me = os.getpid()

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def gpu_jobs(skip=()):
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for ln in ps.splitlines():
        p = ln.split(None, 2)
        if len(p) < 3 or int(p[0]) in (me, *skip) or "claude" in p[2] or "grep" in p[2] or "quiet_wait.py" in p[2]:
            continue
        if "--backend cpu" in p[2] or re.search(r"xcodebuild|swift-frontend|swift-build|platform=iOS|devicectl", p[2]):
            continue   # a CPU-backend gate or a build is not Mac GPU work
        if GPU.search(p[2]):
            rows.append(ln.strip()[:200])
    return rows

def top():
    t = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout.splitlines()
    return [ln for ln in t if ln.startswith(("Load Avg", "CPU usage", "PhysMem"))]

def busiest():
    ps = subprocess.run(["ps", "-Ao", "%cpu,rss,command"], capture_output=True, text=True).stdout.splitlines()[1:]
    return [r.strip()[:160] for r in sorted(ps, key=lambda s: -float(s.split(None, 1)[0]))[:5]]

def snapshot(tag, skip=()):
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return {"at": tag, "time": now(), "load_avg": os.getloadavg(), "top": top(), "swap": sw, "busiest": busiest(),
            "gpu_jobs": gpu_jobs(skip), "lock": open(lock).read().strip() if os.path.exists(lock) else None}

def lock_text():
    return open(lock).read() if os.path.exists(lock) else ""

def holders():
    r = subprocess.run(["lsof", "-t", lock], capture_output=True, text=True).stdout.split()
    return [int(x) for x in r if int(x) != me]

# the items: timing.py's, written as request files (the fixture's records as they are)
recs = {r["id"]: r for r in json.load(open(f"{lane}/fixtures/records.json"))["records"]}
def sub(rid, names):
    q = recs[rid]["request"]["questions"]
    keep = names if names is not None else list(q)
    return {"state": recs[rid]["request"]["state"], "questions": {n: q[n] for n in keep}}
ITEMS = [("one_question", sub("card_refund", ["refund"]), False), ("three_shared", sub("card_refund", None), True),
         ("three_direct", sub("card_refund", None), False),
         ("state_3_4k", sub("long_34k", [next(iter(recs["long_34k"]["request"]["questions"]))]), False)]
for name, req, _ in ITEMS:
    json.dump(req, open(f"{out}/item_{name}.json", "w"), ensure_ascii=False)
FORMS = {}
for f in (os.environ.get("D1_FORMS") or "fp16=d1_3b_decode_fp16_pf16").split():
    lab, b = f.split("=", 1)
    FORMS[lab] = b if b.startswith("/") else f"{lane}/exports/bundles/{b}"
ROUNDS = int(os.environ.get("D1_ROUNDS", "2"))

win = {"run_id": os.path.basename(out), "forms": FORMS, "items": [i[0] for i in ITEMS], "order": [], "between": []}
t0 = time.time()
waits = []
lockf = None
while True:
    held, jobs, others = lock_text().strip(), gpu_jobs(), holders()
    if not held and not jobs and not others:
        lockf = open(lock, "a+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not lock_text().strip():
                break
            fcntl.flock(lockf, fcntl.LOCK_UN)
            others = ["a tag appeared before the flock"]
        except BlockingIOError:
            others = ["flock held"]
        lockf.close()
        lockf = None
    waits.append({"t_s": round(time.time() - t0), "lock": held[:200], "gpu_jobs": jobs[:4], "holders": others})
    if time.time() - t0 > wait_cap:
        win.update({"taken": False, "waited_s": round(time.time() - t0), "waits": waits[-5:],
                    "why": f"the lock or another GPU job for {wait_cap:.0f} s; not measuring contended"})
        json.dump(win, open(f"{out}/window.json", "w"), indent=1)
        print(f"GPU not free after {wait_cap:.0f} s: lock {held!r}, jobs {jobs[:2]}", flush=True)
        sys.exit(3)
    print(f"waiting: lock {held[:80]!r} open by {others}, {len(jobs)} GPU job(s) {jobs[:1]}", flush=True)
    time.sleep(30)
tag = f"{os.environ.get('D1_TAG', 'd1 timing')} pid {me} since {now()}"
lockf.seek(0)
lockf.truncate()
lockf.write(tag + "\n")
lockf.flush()
win.update({"taken": True, "lock_content": tag, "waited_s": round(time.time() - t0), "waits": waits[-5:]})
print(f"lock taken: {tag}", flush=True)
plan = [("swift", lab, r * len(FORMS) + i) for r in range(ROUNDS) for i, lab in enumerate(FORMS)]
if os.environ.get("D1_PYTIME") == "1":
    plan.append(("python", next(iter(FORMS)), None))
rc_all = 0
try:
    win["before"] = snapshot("before")
    for kind, lab, slot in plan:
        cmds = []
        if kind == "swift":
            for name, _, shared in ITEMS:
                cmds.append((f"p{slot}_{lab}_{name}", [binary, "decide", "--bundle", FORMS[lab], "--request",
                                                        f"{out}/item_{name}.json", "--reps", reps, "--warm",
                                                        "--out", f"{out}/resp_{slot}_{lab}_{name}.json",
                                                        "--trace", f"{out}/p{slot}_{lab}_{name}.json"]
                             + (["--shared"] if shared else [])))
        else:
            cmds.append((f"py_{lab}", [py, timing, "run", "--bundle", FORMS[lab], "--tag", f"{win['run_id']}_{lab}"]))
        for name, cmd in cmds:
            t1, load_start = time.time(), os.getloadavg()[0]
            with open(f"{out}/{name}.log", "w") as logf:
                p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                seen, lock_others = set(), set()
                while p.poll() is None:
                    for j in gpu_jobs(skip=(p.pid,)):
                        seen.add(j)
                    for h in holders():
                        lock_others.add(h)
                    if lock_text().strip() != tag:
                        lock_others.add("tag gone: " + lock_text().strip()[:120])
                    time.sleep(2)
            win["order"].append({"kind": kind, "form": lab, "slot": slot, "name": name, "exit": p.returncode,
                                 "wall_s": round(time.time() - t1, 1), "load_1min_start": load_start,
                                 "load_1min_end": os.getloadavg()[0], "other_gpu_jobs_during": sorted(seen),
                                 "other_lock_holders_during": sorted(map(str, lock_others))})
            print(f"[{kind} {lab} {slot} {name}] exit {p.returncode}, other GPU jobs during: {len(seen)}", flush=True)
            if p.returncode != 0:
                rc_all = p.returncode
        win["between"].append(snapshot(f"after {kind} {lab} {slot}"))
    win["after"] = snapshot("after")
finally:
    cur = lock_text().strip()
    if cur == tag:
        lockf.seek(0)
        lockf.truncate()
        lockf.flush()
        win["release"] = {"emptied": True, "bytes_after": os.path.getsize(lock), "at": now()}
    else:
        win["release"] = {"emptied": False, "why": f"the lock holds {cur[:200]!r}, not ours"}
    try:
        fcntl.flock(lockf, fcntl.LOCK_UN)
    finally:
        lockf.close()
    print(f"lock: {win['release']}", flush=True)
    json.dump(win, open(f"{out}/window.json", "w"), indent=1)
sys.exit(rc_all)
PY
rc=${pipestatus[1]}
echo "[$(date '+%H:%M:%S')] done (exit $rc): $OUT" | tee -a $OUT/run.out
exit $rc
