#!/bin/zsh
# Time the Release `d1` CLI on this Mac in one _GPU_LOCK window — apps/Kev/_time_mac.sh's window (its round-7 core:
# the lock, the GPU-job check, the snapshots), with d1's items (conversion/d1/timing.py's protocol, so the rows line up
# with its results/timing_<tag>.json). Written in round 3c, first run in round 7.
#   * the lock (~/code/coreai/_GPU_LOCK, ZOO_GPU_LOCK overrides) is taken only when it is empty, no other process holds it
#     open (lsof; the kit's scripts hold a flock(2) on it) and no other GPU job runs; polled every 30 s for D1_LOCK_WAIT
#     seconds (default 1800), then this script gives up with exit 3 (no contended run). Held: a flock(2) on the file and
#     the file says "<D1_TAG> pid <pid> since <time>" (D1_TAG gets the word "timing" when it lacks it: the guard hook and
#     quiet_wait.py see a window by that word); emptied (0 B) and unlocked at the end, a failed run included.
#   * the items (timing.py ITEMS): one_question (card_refund's `refund` alone), three_shared (card_refund's three
#     questions, --shared), three_direct (the same, direct), state_3_4k (long_34k's first question alone), image_384px
#     (img01_shapes_384x384's first question with its picture: --tower D1_TOWER --images D1_IMAGES). One `d1 decide`
#     process per item and form: it loads the asset (and the tower for the picture item), runs one warm-up call (--warm),
#     decides once, then D1_REPS (20) more (`reps` in its trace: each decision's plan / images / graph / readout / wall
#     seconds).
#   * D1_FORMS="label=bundle ..." (a name under <lane>/exports/bundles or an absolute path; `label=bundle@jit` runs the
#     bundle's .aimodel specialized by the d1 process, --asset jit (and --tower-asset jit), instead of the AOT asset;
#     default "fp16=d1_3b_decode_fp16_pf16"; D1_ITEMS="label=item,item .." runs only those items of a form, none when
#     empty: a re-measure). The order: D1_ROUNDS rounds (default 2) of the AOT forms A B .. A B ..; then the Python
#     reference in the same window, when D1_PYTIME=1 for the first AOT form, D1_PYTIME=all for every one,
#     D1_PYTIME=label,label for those (`timing.py run --bundle <form> --tower D1_TOWER`, --gate-transcript /
#     --e2e-transcript from D1_GATES / D1_E2E "label=path ..."; D1_PY_EXE "label=path" runs that form's Python with
#     another interpreter file name, which names the runtime's cache directory); then D1_ROUNDS rounds of the JIT forms.
#   * D1_FORM_HOOK (an executable): `<hook> pre|post <swift|python> <label> <bundle> <aot|jit> [<python>]` before the
#     first and after the last process of every form's group; a `pre` that exits non-zero skips the group (the runtime
#     names its cache entry of an AOT asset by the asset's main.hash: two forms with one main.hash need their entries
#     swapped between the groups). Its output goes to hook.log.
#   * before, between and after the processes: the load average, top's CPU / memory lines, swap, the five busiest
#     processes and any other GPU job; per process the other GPU jobs and lock holders seen every 2 s (the process's own
#     descendants and the waiters left out) and the one-minute load at its start, end and highest.
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
NOT_GPU = {"tee", "tail", "head", "cat", "less", "more", "grep", "sleep", "wc", "awk", "sed", "ps", "lsof", "jq"}
me = os.getpid()

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def descendants(root):
    kids = {}
    for ln in subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True).stdout.splitlines():
        p = ln.split()
        if len(p) == 2:
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    out, todo = set(), [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out

def gpu_jobs(skip=()):
    ps = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for ln in ps.splitlines():
        p = ln.split(None, 2)
        if len(p) < 3 or int(p[0]) in (me, *skip) or "claude" in p[2] or "grep" in p[2] or "quiet_wait.py" in p[2]:
            continue
        if "quiet_hold.py" in p[2]:
            continue   # a window's holder that is not running its command: it waits on the flock this script holds
        if os.path.basename(p[2].split()[0]) in NOT_GPU:
            continue   # a log writer or reader (a phone job's `tee -a .../edge-llm-bench-.../console_...`)
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

def pairs(var):
    return dict(f.split("=", 1) for f in (os.environ.get(var) or "").split())

# the items: timing.py's, written as request files (the fixture's records as they are)
recs = {r["id"]: r for r in json.load(open(f"{lane}/fixtures/records.json"))["records"]}
img_recs = {r["id"]: r for r in json.load(open(os.environ.get("D1_IMAGE_RECORDS") or f"{lane}/fixtures/image_records.json"))["records"]}
TOWER = os.environ.get("D1_TOWER") or f"{lane}/exports/vision/d1_3b_vision_fp16w32"
IMAGES = os.environ.get("D1_IMAGES") or f"{lane}/fixtures/images"
def sub(rid, names, table=recs):
    q = table[rid]["request"]["questions"]
    keep = names if names is not None else list(q)
    return {"state": table[rid]["request"]["state"], "questions": {n: q[n] for n in keep}}
img = sub("img01_shapes_384x384", [next(iter(img_recs["img01_shapes_384x384"]["request"]["questions"]))], img_recs)
img["images"] = [os.path.basename(x) for x in img_recs["img01_shapes_384x384"]["images"]]
ITEMS = [("one_question", sub("card_refund", ["refund"]), False), ("three_shared", sub("card_refund", None), True),
         ("three_direct", sub("card_refund", None), False),
         ("state_3_4k", sub("long_34k", [next(iter(recs["long_34k"]["request"]["questions"]))]), False),
         ("image_384px", img, False)]
for name, req, _ in ITEMS:
    json.dump(req, open(f"{out}/item_{name}.json", "w"), ensure_ascii=False)
FORMS, ASSET = {}, {}
for f in (os.environ.get("D1_FORMS") or "fp16=d1_3b_decode_fp16_pf16").split():
    lab, b = f.split("=", 1)
    b, _, asset = b.partition("@")
    FORMS[lab] = b if b.startswith("/") else f"{lane}/exports/bundles/{b}"
    ASSET[lab] = asset or "aot"
ROUNDS = int(os.environ.get("D1_ROUNDS", "2"))
GATES, E2E, PY_EXE = pairs("D1_GATES"), pairs("D1_E2E"), pairs("D1_PY_EXE")
HOOK = os.environ.get("D1_FORM_HOOK")
aot_forms = [lab for lab in FORMS if ASSET[lab] == "aot"]
jit_forms = [lab for lab in FORMS if ASSET[lab] == "jit"]

win = {"run_id": os.path.basename(out), "forms": FORMS, "assets": ASSET, "items": [i[0] for i in ITEMS],
       "tower": TOWER, "images": IMAGES, "gates": GATES, "e2e": E2E, "py_exe": PY_EXE, "hook": HOOK,
       "order": [], "between": [], "hooks": []}
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
label = os.environ.get("D1_TAG", "d1 timing")
if "timing" not in label.lower():
    label += " timing"
tag = f"{label} pid {me} since {now()}"
lockf.seek(0)
lockf.truncate()
lockf.write(tag + "\n")
lockf.flush()
win.update({"taken": True, "lock_content": tag, "waited_s": round(time.time() - t0), "waits": waits[-5:]})
print(f"lock taken: {tag}", flush=True)
ITEM_SEL = {lab: [x for x in v.split(",") if x] for lab, v in pairs("D1_ITEMS").items()}   # a form's subset of items
win["items_selected"] = ITEM_SEL
plan = [("swift", lab, r * len(aot_forms) + i) for r in range(ROUNDS) for i, lab in enumerate(aot_forms)
        if ITEM_SEL.get(lab, True)]
pyt = os.environ.get("D1_PYTIME") or ""
if pyt:
    plan += [("python", lab, None) for lab in (aot_forms[:1] if pyt == "1" else aot_forms if pyt == "all"
                                               else [x for x in pyt.split(",") if x in aot_forms])]
plan += [("swift", lab, ROUNDS * len(aot_forms) + r * len(jit_forms) + i) for r in range(ROUNDS)
         for i, lab in enumerate(jit_forms)]
win["plan"] = [list(p) for p in plan]

def hook(stage, kind, lab):
    if not HOOK:
        return 0
    cmd = [HOOK, stage, kind, lab, FORMS[lab], ASSET[lab]] + ([PY_EXE.get(lab, py)] if kind == "python" else [])
    with open(f"{out}/hook.log", "a") as h:
        h.write(f"[{now()}] {' '.join(cmd)}\n")
        h.flush()
        rc = subprocess.run(cmd, stdout=h, stderr=subprocess.STDOUT).returncode
    win["hooks"].append({"stage": stage, "kind": kind, "form": lab, "exit": rc, "time": now()})
    return rc

rc_all = 0
try:
    win["before"] = snapshot("before")
    for kind, lab, slot in plan:
        cmds = []
        if kind == "swift":
            jit = ASSET[lab] == "jit"
            for name, _, shared in ITEMS:
                if lab in ITEM_SEL and name not in ITEM_SEL[lab]:
                    continue
                pic = name == "image_384px"
                cmds.append((f"p{slot}_{lab}_{name}", [binary, "decide", "--bundle", FORMS[lab], "--request",
                                                        f"{out}/item_{name}.json", "--reps", reps, "--warm",
                                                        "--out", f"{out}/resp_{slot}_{lab}_{name}.json",
                                                        "--trace", f"{out}/p{slot}_{lab}_{name}.json"]
                             + (["--shared"] if shared else []) + (["--asset", "jit"] if jit else [])
                             + (["--tower", TOWER, "--images", IMAGES] if pic else [])
                             + (["--tower-asset", "jit"] if pic and jit else [])))
        else:
            cmds.append((f"py_{lab}", [PY_EXE.get(lab, py), timing, "run", "--bundle", FORMS[lab], "--tower", TOWER,
                                       "--tag", f"{win['run_id']}_{lab}"]
                         + (["--gate-transcript", GATES[lab]] if lab in GATES else [])
                         + (["--e2e-transcript", E2E[lab]] if lab in E2E else [])))
        hrc = hook("pre", kind, lab)
        if hrc != 0:
            win["order"].append({"kind": kind, "form": lab, "slot": slot, "skipped": f"pre hook exit {hrc}"})
            print(f"[{kind} {lab} {slot}] skipped: pre hook exit {hrc}", flush=True)
            rc_all = rc_all or 4
            continue
        for name, cmd in cmds:
            t1, load_start = time.time(), os.getloadavg()[0]
            load_max = load_start
            with open(f"{out}/{name}.log", "w") as logf:
                p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
                seen, lock_others = set(), set()
                while p.poll() is None:
                    mine = {p.pid, *descendants(p.pid)}
                    for j in gpu_jobs(skip=mine):
                        seen.add(j)
                    for h in holders():
                        if h not in mine:
                            lock_others.add(h)
                    if lock_text().strip() != tag:
                        lock_others.add("tag gone: " + lock_text().strip()[:120])
                    load_max = max(load_max, os.getloadavg()[0])
                    time.sleep(2)
            win["order"].append({"kind": kind, "form": lab, "slot": slot, "name": name, "exit": p.returncode,
                                 "wall_s": round(time.time() - t1, 1), "load_1min_start": load_start,
                                 "load_1min_end": os.getloadavg()[0], "load_1min_max": load_max,
                                 "other_gpu_jobs_during": sorted(seen),
                                 "other_lock_holders_during": sorted(map(str, lock_others))})
            print(f"[{kind} {lab} {slot} {name}] exit {p.returncode}, other GPU jobs during: {len(seen)}, "
                  f"load max {load_max:.1f}", flush=True)
            if p.returncode != 0:
                rc_all = p.returncode
        hook("post", kind, lab)
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
