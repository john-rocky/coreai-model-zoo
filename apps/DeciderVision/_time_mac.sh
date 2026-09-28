#!/bin/zsh
# Time the Release decider-vision CLI on this Mac under the machine-wide GPU lock: the round-1 fixture (76 decisions)
# through the AOT assets, DV_PASSES passes per bundle (default 2), each pass its own process with --reload (the first
# load in the process, then everything dropped and loaded again) and --no-dump.
#   ./_time_mac.sh                 waits for DV_WAIT_FILE first (default ~/code/coreai/_decider2bv/logs/r7_done)
#   DV_BUNDLES="decider_2b_vision_decode_int8mix_pf16" ./_time_mac.sh
# The lock protocol is apps/FunASRGate/_run_mac.sh's: ~/code/coreai/_GPU_LOCK (COREAI_GPU_LOCK overrides) is an
# advisory file two conventions share — the kit's scripts/with-gpu-lock.py holds flock(2) on it for the length of a
# command (and leaves the 0-byte file behind); the zoo's tools read its existence as "held", write a tag and remove it.
# This script takes the flock, writes a tag line while it runs, and leaves the file as it found it (a 0-byte file stays;
# a file that did not exist goes). Another session counts as using the GPU while it holds the flock, while the file
# carries a tag naming a live process, or while the GPU reads busy ("Device Utilization %" >= 50 in 3 of 5 samples 1 s
# apart); then this script waits in 30 s steps up to DV_LOCK_WAIT s (default 1800), and after that runs anyway and
# records "contended". The decision, the readings, a process snapshot before and after, and every 5 s the passes'
# scheduling priority with the load average and the GPU utilization go to <out>/gpu_lock.json.
# Output: ~/code/coreai/_decider2bv/swift/timing/<run id>/{<bundle>_pass<k>.json,<bundle>_pass<k>.log,gpu_lock.json}
set -u
L=${DV_LANE:-$HOME/code/coreai/_decider2bv}
BIN=${DV_BIN:-$L/swift/.build/release/decider-vision}
[ -x $BIN ] || { echo "no Release build at $BIN (swift build -c release --scratch-path $L/swift/.build)"; exit 1; }
WAIT_FILE=${DV_WAIT_FILE:-$L/logs/r7_done}
RUN_ID=${DV_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)}
OUT=$L/swift/timing/$RUN_ID
mkdir -p $OUT
LOCK=${COREAI_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

if [ -n "$WAIT_FILE" ] && [ ! -e "$WAIT_FILE" ]; then
  say "waiting for $WAIT_FILE (30 s steps, cap ${DV_WAIT_CAP:-10800} s)"
  t0=$(date +%s)
  while [ ! -e "$WAIT_FILE" ]; do
    (( $(date +%s) - t0 > ${DV_WAIT_CAP:-10800} )) && { say "gave up waiting for $WAIT_FILE"; exit 2; }
    sleep 30
  done
  say "$WAIT_FILE is there"
fi
say "run $RUN_ID: $BIN (GPU lock $LOCK)"

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "$L" "$RUN_ID" "${DV_LOCK_WAIT:-1800}" "${DV_PASSES:-2}" \
  "${DV_BUNDLES:-decider_2b_vision_decode_int8mix_pf16 decider_2b_vision_decode_fp16_pf16}" <<'PY'
import fcntl, json, os, re, subprocess, sys, time
lock_path, out, binary, lane, run_id, wait_cap, passes, bundles = sys.argv[1:9]
wait_cap, passes, bundles = float(wait_cap), int(passes), bundles.split()
rec = {"lock": lock_path, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_avg_start": os.getloadavg()}
WATCH = re.compile(r"python|decider|coreai|Gate|mlx|llama|ollama|Xcode|xcodebuild|swift-frontend|MTLCompiler|metal", re.I)

def gpu_util():
    o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
    m = re.findall(r'"Device Utilization %"=(\d+)', o)
    return max(map(int, m)) if m else -1

def gpu_busy():
    vals = []
    for i in range(5):
        vals.append(gpu_util())
        if i < 4: time.sleep(1)
    return sum(v >= 50 for v in vals) >= 3, vals

def snapshot(tag):
    o = subprocess.run(["ps", "-axo", "pid=,ppid=,pri=,%cpu=,rss=,etime=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in o.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7: continue
        pid, ppid, pri, cpu, rss, etime, cmd = parts
        if int(pid) == os.getpid(): continue
        if float(cpu) >= 5.0 or WATCH.search(cmd):
            rows.append({"pid": int(pid), "ppid": int(ppid), "pri": int(pri), "cpu_pct": float(cpu), "rss_mb": int(rss) // 1024,
                         "etime": etime, "command": cmd[:240]})
    busy, vals = gpu_busy()
    return {"at": tag, "time": time.strftime("%H:%M:%S"), "load_avg": os.getloadavg(), "gpu_util_samples": vals,
            "gpu_busy": busy, "processes": rows}

def tag_holder():
    try:
        t = open(lock_path).read().strip()
    except FileNotFoundError:
        return None, None
    m = re.search(r"pid (\d+)", t)
    if m:
        try:
            os.kill(int(m.group(1)), 0)
            return t, True
        except (ProcessLookupError, PermissionError):
            return t, False
    return t, None

rec["snapshot_before_lock"] = snapshot("before lock")
existed = os.path.exists(lock_path)
size_before = os.path.getsize(lock_path) if existed else None
mtime_before = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lock_path))) if existed else None
rec.update({"file_existed": existed, "file_bytes_before": size_before, "file_mtime_before": mtime_before})
f = open(lock_path, "a+")
t0 = time.time()
readings, state = [], None
while True:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        have = True
    except BlockingIOError:
        have = False
    tag, live = tag_holder()
    busy, vals = gpu_busy()
    readings.append({"t_s": round(time.time() - t0, 1), "flock_free": have, "tag": tag, "tag_pid_live": live,
                     "gpu_util_samples": vals, "gpu_busy": busy})
    if have and live is not True and not busy:
        state = ("taken by this run (flock; file " + ("absent before" if not existed else
                 f"{size_before} bytes, last written {mtime_before}") + f"; GPU utilization {vals})")
        break
    if have:
        fcntl.flock(f, fcntl.LOCK_UN)
    if time.time() - t0 >= wait_cap:
        state = (f"contended: after {time.time() - t0:.0f} s, flock {'free' if have else 'held'}, tag {tag!r} (live {live}), "
                 f"GPU utilization {vals}; running anyway")
        break
    print(f"GPU in use by another session (flock {'free' if have else 'held'}, tag {tag!r}, live {live}, GPU {vals}); "
          f"waiting 30 s ({time.time() - t0:.0f} s so far, cap {wait_cap:.0f} s)", flush=True)
    time.sleep(30)
rec.update({"state": state, "waited_s": round(time.time() - t0, 1), "readings": readings})
print(f"GPU lock: {state}", flush=True)
ours = state.startswith("taken")
if ours:
    f.seek(0); f.truncate(); f.write(f"decider-vision timing {run_id} pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); f.flush()
rec["snapshot_after_lock"] = snapshot("after lock, before the passes")

ex = f"{lane}/exports"
towers = {g: f"{ex}/decider_2b_vision_{g}_vision_fp16w32_aotc/decider_2b_vision_{g}_vision_fp16w32.h16c.aimodelc"
          for g in ("g256", "g448")}
rec["passes"] = []
for b in bundles:
    for k in range(1, passes + 1):
        js = f"{out}/{b.replace('decider_2b_vision_decode_', '')}_pass{k}.json"
        cmd = [binary, "fixture", "--bundle", f"{ex}/bundles/{b}", "--tower-g256", towers["g256"],
               "--tower-g448", towers["g448"], "--rows", f"{lane}/fixtures/rows.json", "--images", f"{lane}/fixtures/images",
               "--meta", f"{lane}/fixtures/meta.json", "--out", js, "--reload", "--no-dump",
               "--label", f"D timing {run_id} {b} pass {k} ({'lock taken' if ours else 'contended'})"]
        prio = []
        t1 = time.time()
        with open(js.replace(".json", ".log"), "w") as so:
            p = subprocess.Popen(cmd, stdout=so, stderr=subprocess.STDOUT)
            while p.poll() is None:
                pr = subprocess.run(["ps", "-o", "pri=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
                if pr:
                    prio.append([round(time.time() - t1, 1), int(pr), round(os.getloadavg()[0], 2), gpu_util()])
                time.sleep(5)
        low = [x for x in prio if x[1] < 20]
        rec["passes"].append({"bundle": b, "pass": k, "json": js, "exit": p.returncode, "s": round(time.time() - t1, 1),
                              "priority_every_5s": prio, "priority_min": min((x[1] for x in prio), default=None),
                              "demoted_samples": len(low)})
        print(f"{b} pass {k}: exit {p.returncode} after {time.time() - t1:.1f} s"
              + (f" (WARNING: background priority in {len(low)} samples)" if low else ""), flush=True)
rec["snapshot_after_passes"] = snapshot("after the passes, lock still held")
rec.update({"load_avg_end": os.getloadavg(), "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
if ours:
    f.seek(0); f.truncate(); f.flush()
    if not existed:
        os.remove(lock_path)
    fcntl.flock(f, fcntl.LOCK_UN)
f.close()
rec["file_after"] = ("absent" if not os.path.exists(lock_path) else f"{os.path.getsize(lock_path)} bytes")
json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
print(f"lock file after: {rec['file_after']}", flush=True)
sys.exit(0 if all(x["exit"] == 0 for x in rec["passes"]) else 1)
PY
rc=$?
say "done (exit $rc): $OUT"
exit $rc
