#!/bin/zsh
# Time the Release `kev` CLI on this Mac in one _GPU_LOCK window — conversion/kev/timing.py's protocol (round 5), so the
# rows line up with results/timing_r5.json:
#   * the lock (~/code/coreai/_GPU_LOCK, ZOO_GPU_LOCK overrides) is taken only when it is empty, no other process
#     holds it open (lsof; the clef-flash scripts hold a flock(2) on it) and no other GPU job runs (the lane's readout gates / timing / decide workers, the LiteRT lane's GPU parity, which does not read the
#     lock, the shared benches); polled every 30 s for KEV_LOCK_WAIT seconds (default 1800), then this script gives up
#     with exit 3 (no contended run). Held: a flock(2) on the file (the kit's convention) and the file says "kev timing
#     r7 pid <pid> since <time>" (the lane's); emptied (0 B) and unlocked at the end, a failed run included.
#   * in the window, one process each, in order: Swift Kev-0.8B, Swift Kev-4B, the Python reference's two same-window
#     items on Kev-0.8B and Kev-4B (`gate_swift.py pytime`: tv4_000:q0 and own_m01's first 5 questions shared), Swift
#     Kev-0.8B, Swift Kev-4B — the Swift processes A B A B. A Swift process (`kev time`) loads the AOT asset twice
#     (cold, warm) and runs timing.py's items, one warm-up each, then KEV_REPS (10) decisions.
#   * before, between and after the processes: the load average, top's CPU / memory lines, swap, the five busiest
#     processes and any other GPU job.
#   ./_time_mac.sh            -> ~/code/coreai/_kev/swift/timing/<run id>/{p<slot>_<model>.json, py_<model>.json, *.log, window.json}
#   then: python conversion/kev/gate_swift.py timing --run-dir <that dir>   -> results/swift_timing_r7.json
# Round 11 (all optional; unset = the round-7 window above, unchanged):
#   KEV_FORMS="U16=kev_0_8b_decode_fp16_pf16 K64=kev_0_8b_decode_fp16_metal_pf64 ..."  label=bundle (a name under
#       <lane>/exports/bundles or an absolute path): the Swift processes run the forms A B C ... A B C ... (KEV_ROUNDS
#       rounds, default 2), `kev time --model-label <label>`, files p<slot>_<label>.json
#   KEV_JIT="U16 @rank1"   after those rounds: one `kev time --asset jit` process per label (jit_<label>.json); @rank1 =
#       the first form of `gate_swift.py timing-r11 --print-rank1` on this run (KEV_CANDIDATES = the labels it may pick)
#   KEV_PYTIME="@rank1"    then `gate_swift.py pytime --model $KEV_MODEL --bundle <its bundle> --label <label>`
#   KEV_TAG="kev speed r11 timing"   the lock's tag before " pid <pid> since <time>" (default "kev timing r7")
#   KEV_GPU_EXTRA='export_decoder\.py|coreai-build'   more GPU job patterns to wait for (another lane's exports)
#   then: python conversion/kev/gate_swift.py timing-r11 --run-dir <that dir> --out <json>
# Round 15 (all optional; unset = the windows above, unchanged):
#   KEV_FORMS entries take options after the bundle, comma-separated: "D=kev_0_8b_decode_fp16_metal_dyn128,asset=jit,
#       call_max=128,multiple=16,warm" -> `kev time --asset jit --call-max 128 --multiple 16 --warm` (asset aot|jit; warm =
#       every call length of the plan once after the load; a dynamic-S bundle's L and q default to its metadata)
#   KEV_CLEAN=1   the clean-process rule (10-04): a process counts when no other GPU job ran during it and the
#       1-minute load average was <= KEV_LOAD_MAX (12) at its start and its end; the rounds A B C ... go on, only for the
#       forms with fewer than 2 clean processes, up to KEV_MAX_ROUNDS (4); every process is kept in window.json with its
#       verdict and reasons (`clean`), and after the lock is taken a load above KEV_LOAD_MAX is waited out for up to 10 min
#   KEV_TURN_KEY="r15 timing"   take the lock only while the lane's queue (<lane>/logs/gpu_turn.txt) starts with this
#       line, and remove the line after the window
set -u
L=${KEV_LANE:-$HOME/code/coreai/_kev}
BIN=${KEV_BIN:-$L/swift/.build/release/kev}
PY=${KEV_PY:-$HOME/code/coreai/coreai-models/.venv/bin/python}
GATE=${0:A:h}/../../conversion/kev/gate_swift.py
[ -x $BIN ] || { echo "no Release build at $BIN (swift build -c release --scratch-path $L/swift/.build)"; exit 1; }
RUN_ID=${KEV_RUN_ID:-r7_$(date +%Y%m%d_%H%M%S)}
OUT=$L/swift/timing/$RUN_ID
mkdir -p $OUT
LOCK=${ZOO_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
echo "[$(date '+%H:%M:%S')] run $RUN_ID: $BIN (lock $LOCK)" | tee -a $OUT/run.out

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "$L" "$PY" "${GATE:A}" "${KEV_LOCK_WAIT:-1800}" "${KEV_REPS:-10}" <<'PY' 2>&1 | tee -a $OUT/run.out
import fcntl, json, os, re, subprocess, sys, time
from datetime import datetime
lock, out, binary, lane, py, gate, wait_cap, reps = sys.argv[1:9]
wait_cap = float(wait_cap)
B = {"kev-0.8b": f"{lane}/exports/bundles/kev_0_8b_decode_fp16_pf16", "kev-4b": f"{lane}/exports/bundles/kev_4b_decode_fp16_pf16"}
GPU = re.compile(r"readout_gate\.py|timing\.py (run|worker)|decide\.py (check|worker|run)|--accel gpu|gpu_gate_subprocess|"
                 r"litert_parity\.py .*gpu|llm-bench|yardstick|coreai_verify|mlx_lm|/kev (fixture|time|run) |gate_swift\.py (pytime|longrow-python)|"
                 r"litert-lm (run|benchmark)|--backend gpu|gsm8k_cli_k\.py|gate8q_full\.py|conv_gate\.py|"
                 r"r13_probe\.py|/kev(_pregate)? (fixture|time|run|warm) "   # round 15: round 13's Mac probe, `kev warm`, the pregate copy
                 + ("|" + os.environ["KEV_GPU_EXTRA"] if os.environ.get("KEV_GPU_EXTRA") else ""))
TURN = os.path.join(lane, "logs", "gpu_turn.txt")
TURN_KEY = os.environ.get("KEV_TURN_KEY", "")
CLEAN = os.environ.get("KEV_CLEAN") == "1"
LOAD_MAX = float(os.environ.get("KEV_LOAD_MAX", "12"))
me = os.getpid()

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def gpu_jobs(skip=()):
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for ln in ps.splitlines():
        p = ln.split(None, 2)
        if len(p) < 3 or int(p[0]) in (me, *skip) or "claude" in p[2] or "grep" in p[2]:
            continue
        if any(w in p[2] for w in ("_window.py", "timing_when_quiet", "timing_retry", "timing_mac.py", "r11_locked.py",
                                   "/bin/zsh ./r13_run.sh ")):
            continue   # round 11: a driver waiting for its turn, not a GPU job (round 15: also the LiteRT lane's
                       # wrapper, which waits in wait_gpu_quiet before it starts its GPU child)
        if "--backend cpu" in p[2]:
            continue   # round 14: a CPU-backend gate is not GPU work
        if re.search(r"xcodebuild|swift-frontend|swift-build|platform=iOS|devicectl|[0-9A-F]{8}-[0-9A-F]{16}", p[2]):
            continue   # round 14/15: an iOS build or anything carrying an iPhone UDID is not Mac GPU work
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
    """Other processes with the lock file open (the clef-flash scripts hold a flock(2) on it while they measure)."""
    r = subprocess.run(["lsof", "-t", lock], capture_output=True, text=True).stdout.split()
    return [int(x) for x in r if int(x) != me]

def turn_head():
    """round 15: the first line of the lane's GPU queue ('' when there is none)."""
    try:
        lines = [x.strip() for x in open(TURN).read().splitlines() if x.strip()]
    except FileNotFoundError:
        return ""
    return lines[0] if lines else ""

def turn_pop():
    """round 15: remove the queue's first line when it is ours (temp file + rename)."""
    try:
        lines = open(TURN).read().splitlines()
    except FileNotFoundError:
        return ""
    i = next((k for k, x in enumerate(lines) if x.strip()), None)
    if i is None or not lines[i].strip().startswith(TURN_KEY):
        return ""
    tmp = f"{TURN}.tmp{me}"
    open(tmp, "w").write("\n".join(lines[:i] + lines[i + 1:]) + ("\n" if len(lines) > 1 else ""))
    os.replace(tmp, TURN)
    return lines[i]

win = {"run_id": os.path.basename(out), "order": [], "swift_processes": [], "python_processes": [], "between": []}
FORMS_MODE = bool(os.environ.get("KEV_FORMS", "").split())

def ps_top():
    """round 11: the 15 busiest processes, full command lines (to find a GPU job the patterns miss)"""
    r = subprocess.run(["ps", "-axo", "pid,pcpu,command", "-r"], capture_output=True, text=True).stdout.splitlines()
    return [ln[:220] for ln in r[:16]]
t0 = time.time()
waits = []
# Round 11: the file is opened only at the moment of taking it (held, with its flock, until the end of the window). A
# waiter that keeps it open shows up in lsof, and two such waiters wait for each other (round 12's r12_window.py too).
lockf = None
while True:
    held, jobs, others = lock_text().strip(), gpu_jobs(), holders()
    if FORMS_MODE:   # round 11: the lock first (its tag stops the other lanes' new GPU steps), the GPU jobs after
        jobs, others = [], []
    if TURN_KEY and not turn_head().startswith(TURN_KEY):   # round 15: not our turn in the lane's queue yet
        others = list(others) + [f"turn: {turn_head()!r}"]
    if not held and not jobs and not others:
        lockf = open(lock, "a+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the kit's / clef-flash's convention, beside the tag
            if not lock_text().strip():                         # nobody wrote a tag between the check and the flock
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
tag = f"{os.environ.get('KEV_TAG', 'kev timing r7')} pid {me} since {now()}"
lockf.seek(0)
lockf.truncate()
lockf.write(tag + "\n")
lockf.flush()
win.update({"taken": True, "lock_content": tag, "waited_s": round(time.time() - t0), "waits": waits[-5:]})
print(f"lock taken: {tag}", flush=True)
if FORMS_MODE:
    t_d, drain = time.time(), []
    while gpu_jobs() and time.time() - t_d < float(os.environ.get("KEV_DRAIN_CAP", "600")):
        drain.append({"time": now(), "gpu_jobs": gpu_jobs()[:6]})
        time.sleep(5)
    win["drain"] = {"seconds": round(time.time() - t_d), "clear": not gpu_jobs(), "waits": drain[-20:]}
    print(f"GPU drain: {win['drain']['seconds']} s, clear {win['drain']['clear']}", flush=True)
if CLEAN:   # round 15: a load above KEV_LOAD_MAX is waited out (10 min at most) before the first process
    t_l = time.time()
    while os.getloadavg()[0] > LOAD_MAX and time.time() - t_l < 600:
        time.sleep(15)
    win["load_wait"] = {"seconds": round(time.time() - t_l), "load_after": os.getloadavg()[0]}
plan = [("swift", "kev-0.8b", 0), ("swift", "kev-4b", 1), ("python", "kev-0.8b", None), ("python", "kev-4b", None),
        ("swift", "kev-0.8b", 2), ("swift", "kev-4b", 3)]
FORMS = os.environ.get("KEV_FORMS", "").split()
OPTS = {}   # round 15: per form, the extra `kev time` arguments
if FORMS:   # round 11: label=bundle pairs, A B C ... A B C ..., then the JIT and Python processes
    B = {}
    for f in FORMS:
        lab, b = f.split("=", 1)
        b, *opts = b.split(",")   # round 15: bundle,asset=jit,call_max=128,multiple=16,warm
        B[lab] = b if b.startswith("/") else f"{lane}/exports/bundles/{b}"
        extra = []
        for o in opts:
            key, _, val = o.partition("=")
            extra += {"asset": ["--asset", val], "call_max": ["--call-max", val], "multiple": ["--multiple", val],
                      "warm": ["--warm"]}[key]
        OPTS[lab] = extra
    rounds = int(os.environ.get("KEV_ROUNDS", "2"))
    plan = [("swift", lab, r * len(B) + i) for r in range(rounds) for i, lab in enumerate(B)]
    if CLEAN:   # round 15: one round at a time; more rounds only for the forms short of 2 clean processes
        plan = [("swift", lab, i) for i, lab in enumerate(B)] + [("clean-round", 1, None)]
    if os.environ.get("KEV_JIT") or os.environ.get("KEV_PYTIME"):
        plan.append(("resolve", None, None))
    win["forms"] = B
rc_all = 0

def resolve():
    """@rank1 in KEV_JIT / KEV_PYTIME -> the first form of this run's ranking (gate_swift.py timing-r11)."""
    labels = (os.environ.get("KEV_JIT", "") + " " + os.environ.get("KEV_PYTIME", "")).split()
    rank1 = None
    if "@rank1" in labels:
        r = subprocess.run([py, gate, "timing-r11", "--run-dir", out, "--candidates", os.environ.get("KEV_CANDIDATES", ""),
                            "--print-rank1"], capture_output=True, text=True)
        rank1 = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
        win["rank1"] = {"label": rank1, "exit": r.returncode, "stderr": r.stderr[-2000:]}
        print(f"rank 1 of this run: {rank1} (exit {r.returncode})", flush=True)
    def sub(x):
        return rank1 if x == "@rank1" else x
    jit = list(dict.fromkeys(sub(x) for x in os.environ.get("KEV_JIT", "").split() if sub(x)))
    pyt = list(dict.fromkeys(sub(x) for x in os.environ.get("KEV_PYTIME", "").split() if sub(x)))
    return [("swift-jit", lab, 90 + i) for i, lab in enumerate(jit)] + [("python", lab, None) for lab in pyt]

clean_count = {}   # round 15: label -> clean processes so far
next_slot = len(plan)

def process_clean(rec):
    """round 15: the clean-process rule -> (clean, reasons)."""
    reasons = []
    if rec.get("other_gpu_jobs_during"):
        reasons.append(f"other GPU jobs: {rec['other_gpu_jobs_during'][:3]}")
    if rec.get("load_1min_start", 0) > LOAD_MAX:
        reasons.append(f"load at start {rec['load_1min_start']:.2f} > {LOAD_MAX}")
    if rec.get("load_1min_end", 0) > LOAD_MAX:
        reasons.append(f"load at end {rec['load_1min_end']:.2f} > {LOAD_MAX}")
    if rec.get("exit"):
        reasons.append(f"exit {rec['exit']}")
    return not reasons, reasons

try:
    win["before"] = snapshot("before")
    k = 0
    while k < len(plan):
        kind, model, slot = plan[k]
        k += 1
        if kind == "resolve":
            plan += resolve()
            continue
        if kind == "clean-round":   # round 15: another round for the forms short of 2 clean processes
            short = [lab for lab in B if clean_count.get(lab, 0) < 2]
            more = []
            if short and model < int(os.environ.get("KEV_MAX_ROUNDS", "4")):
                for lab in short:
                    more.append(("swift", lab, next_slot))
                    next_slot += 1
                more.append(("clean-round", model + 1, None))
            plan[k:k] = more   # right after this marker (a trailing "resolve" stays last)
            win["clean_count"] = dict(clean_count)
            continue
        if kind in ("swift", "swift-jit"):
            name = f"p{slot}_{model}.json" if kind == "swift" else f"jit_{model}.json"
            cmd = [binary, "time", "--bundle", B[model], "--records", f"{lane}/fixtures/records.json", "--model-label", model,
                   "--slot", str(slot), "--reps", reps, "--out", f"{out}/{name}"] + OPTS.get(model, [])
            if kind == "swift-jit":
                cmd[4:4] = ["--asset", "jit"]   # after "--bundle <dir>"
        elif FORMS:
            name = f"py_{model}.json"
            cmd = [py, gate, "pytime", "--model", os.environ.get("KEV_MODEL", "kev-0.8b"), "--bundle", B[model], "--label", model,
                   "--reps", reps, "--out", f"{out}/{name}"]
        else:
            name = f"py_{model}.json"
            cmd = [py, gate, "pytime", "--model", model, "--reps", reps, "--out", f"{out}/{name}"]
        t1 = time.time()
        top_start, load_start = (ps_top() if FORMS_MODE else None), os.getloadavg()[0]
        with open(f"{out}/{name.replace('.json', '.log')}", "w") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            seen, lock_others = set(), set()
            while p.poll() is None:
                for j in gpu_jobs(skip=(p.pid,)):
                    seen.add(j)
                for h in holders():
                    lock_others.add(h)
                if lock_text().strip() != tag:
                    lock_others.add("tag gone: " + lock_text().strip()[:120])
                    if FORMS:   # round 11: put the tag back (r12 holds its heavy work while the lock says timing)
                        lockf.seek(0)
                        lockf.truncate()
                        lockf.write(tag + "\n")
                        lockf.flush()
                time.sleep(5)
        wall = time.time() - t1
        win["order"].append({"kind": kind, "model": model, "slot": slot, "file": name, "exit": p.returncode,
                             "wall_s": round(wall, 1), "other_gpu_jobs_during": sorted(seen),
                             "other_lock_holders_during": sorted(map(str, lock_others))})
        if FORMS_MODE:
            win["order"][-1].update({"start": t1, "load_1min_start": load_start, "load_1min_end": os.getloadavg()[0],
                                     "ps_top_start": top_start, "ps_top_end": ps_top()})
        if CLEAN:   # round 15
            ok, why = process_clean(win["order"][-1])
            win["order"][-1].update({"clean": ok, "reasons": why})
            if ok and kind == "swift":
                clean_count[model] = clean_count.get(model, 0) + 1
        (win["swift_processes"] if kind == "swift" else win.setdefault("jit_processes", []) if kind == "swift-jit"
         else win["python_processes"]).append(name)
        print(f"[{kind} {model} {slot}] exit {p.returncode}, {wall:.0f} s, other GPU jobs during: {len(seen)}, "
              f"other lock holders: {sorted(map(str, lock_others))}", flush=True)
        win["between"].append(snapshot(f"after {kind} {model} {slot}"))
        if p.returncode != 0:
            rc_all = p.returncode
            if not CLEAN:
                break
    win["after"] = snapshot("after")
    if CLEAN:
        win["clean_count"] = dict(clean_count)
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
    if TURN_KEY and win["release"].get("emptied"):   # round 15: our queue line goes
        win["turn_popped"] = turn_pop()
    json.dump(win, open(f"{out}/window.json", "w"), indent=1)
sys.exit(rc_all)
PY
rc=${pipestatus[1]}
echo "[$(date '+%H:%M:%S')] done (exit $rc): $OUT" | tee -a $OUT/run.out
exit $rc
