#!/bin/zsh
# Run the macOS build of DeciderVisionGate on this Mac, under the machine-wide GPU lock, and print its summary.
#   ./_run_mac.sh                              every stage on the stage directory (./_stage.sh)
#   DV_STAGES=load1,warmup DV_LIMIT=3 ./_run_mac.sh   any DV_* knob of GateRunner.swift passes through the environment
# The GPU lock is ~/code/coreai/_GPU_LOCK (COREAI_GPU_LOCK overrides), an advisory file two conventions share: the kit's
# scripts/with-gpu-lock.py holds flock(2) on it for the length of a command (and leaves the 0-byte file behind); the zoo's
# tools (conversion/zoo_smoke.py, nemotron3_diar/bench_mac.py) read its existence as "held", write a tag and remove it at
# the end. This script takes the flock, writes a tag line into the file while the gate runs, and at the end leaves the
# file as it found it (a 0-byte file stays; a file that did not exist goes). Another session counts as using the GPU
# while it holds the flock, while the file carries a tag naming a live process, or while the GPU reads busy (ioreg
# "Device Utilization %" >= 50 in 3 of 5 samples 1 s apart); then this script waits in 30 s steps up to DV_LOCK_WAIT
# s (default 1800), and after that runs anyway and records "contended". The decision and the readings go to
# <run>/gpu_lock.json and into result.json (config.env.DV_GPU_LOCK).
# The app quits when the gate is done; DV_MAC_CAP s (default 3600) is the most this script waits for it.
# Output: _work/mac_runs/<run id>/{result.json,result.log,app.stdout,gpu_lock.json,run.out}
set -u
DIR=${0:A:h}
W=${DV_WORK:-$DIR/_work}
V=${DV_MAC_VARIANT:-}
APP=$(cat $W/app_path_mac${V:+_$V}.txt 2>/dev/null)
[ -d "$APP" ] || { echo "no macOS build: run ./_build.sh --mac${V:+ --$V}"; exit 1; }
BIN=$APP/Contents/MacOS/DeciderVisionGate
S=${DV_ASSETS:-$W/device_stage/DeciderVisionAssets}
[ -f $S/MD5SUMS ] || { echo "nothing staged at $S: run ./_stage.sh"; exit 1; }
RUN_ID=${DV_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)${V:+-$V}}
OUT=$W/mac_runs/$RUN_ID
mkdir -p $OUT
LOCK=${COREAI_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }
say "run $RUN_ID: $BIN on $S (GPU lock $LOCK)"

# The lock is held by a python process for the length of the app's run: it takes the flock (or gives up after the
# wait), then runs the app with DV_GPU_LOCK set to its decision, and restores the lock file.
/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "$S" "$RUN_ID" "${DV_LOCK_WAIT:-1800}" "${DV_MAC_CAP:-3600}" <<'PY'
import fcntl, json, os, re, subprocess, sys, time
lock_path, out, binary, assets, run_id, wait_cap, run_cap = sys.argv[1:8]
wait_cap, run_cap = float(wait_cap), float(run_cap)
rec = {"lock": lock_path, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "load_avg_start": os.getloadavg()}

def gpu_busy():
    vals = []
    for i in range(5):
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.findall(r'"Device Utilization %"=(\d+)', o)
        vals.append(max(map(int, m)) if m else -1)
        if i < 4: time.sleep(1)
    return sum(v >= 50 for v in vals) >= 3, vals

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

existed = os.path.exists(lock_path)
size_before = os.path.getsize(lock_path) if existed else None
mtime_before = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lock_path))) if existed else None
rec.update({"file_existed": existed, "file_bytes_before": size_before, "file_mtime_before": mtime_before})
f = open(lock_path, "a+")
t0 = time.time()
waited_steps, readings, state = 0, [], None
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
    other = (not have) or live is True or busy
    if not other:
        state = ("taken by this run (flock; file " + ("absent before" if not existed else
                 f"{size_before} bytes, last written {mtime_before}") + f"; GPU utilization {vals})")
        break
    if have:
        fcntl.flock(f, fcntl.LOCK_UN)
    if time.time() - t0 >= wait_cap:
        state = (f"contended: after {time.time() - t0:.0f} s, flock {'free' if have else 'held'}, tag {tag!r} (live {live}), "
                 f"GPU utilization {vals}; running anyway")
        break
    waited_steps += 1
    print(f"GPU in use by another session (flock {'free' if have else 'held'}, tag {tag!r}, live {live}, GPU {vals}); "
          f"waiting 30 s ({time.time() - t0:.0f} s so far, cap {wait_cap:.0f} s)", flush=True)
    time.sleep(30)
rec.update({"state": state, "waited_s": round(time.time() - t0, 1), "readings": readings})
print(f"GPU lock: {state}", flush=True)
ours = state.startswith("taken")
if ours:
    f.seek(0); f.truncate(); f.write(f"DeciderVisionGate mac run {run_id} pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); f.flush()
env = dict(os.environ, DV_ASSETS=assets, DV_OUT=out, DV_RUN_ID=run_id, DV_GPU_LOCK=state)
rc = None
t1 = time.time()
# the app's scheduling priority every 5 s (ps PRI: 31 = a foreground user process; App Nap / background drops it to 4,
# and every time measured meanwhile is throttled), with the machine's load and the GPU's utilization
prio = []
with open(f"{out}/app.stdout", "w") as so:
    p = subprocess.Popen([binary], env=env, stdout=so, stderr=subprocess.STDOUT)
    while p.poll() is None:
        if time.time() - t1 > run_cap:
            print(f"the app ran past {run_cap:.0f} s: terminating pid {p.pid}", flush=True)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            break
        pr = subprocess.run(["ps", "-o", "pri=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip()
        if pr:
            o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
            m = re.findall(r'"Device Utilization %"=(\d+)', o)
            prio.append([round(time.time() - t1, 1), int(pr), round(os.getloadavg()[0], 2), max(map(int, m)) if m else -1])
        time.sleep(5)
    rc = p.returncode
low = [x for x in prio if x[1] < 20]
rec.update({"app_exit": rc, "app_s": round(time.time() - t1, 1), "load_avg_end": os.getloadavg(),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "app_priority_every_5s": prio, "app_priority_min": min((x[1] for x in prio), default=None),
            "app_demoted_samples": len(low)})
if low:
    print(f"WARNING the app ran at background priority in {len(low)} of {len(prio)} samples (first at {low[0][0]} s): "
          f"the times of that stretch are throttled", flush=True)
if ours:
    # leave the file as it was found: 0 bytes if it existed (with-gpu-lock.py's leftover), gone if it did not
    f.seek(0); f.truncate(); f.flush()
    if not existed:
        os.remove(lock_path)
    fcntl.flock(f, fcntl.LOCK_UN)
f.close()
rec["file_after"] = ("absent" if not os.path.exists(lock_path) else f"{os.path.getsize(lock_path)} bytes")
json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
print(f"app exit {rc} after {rec['app_s']} s; lock file {rec['file_after']}", flush=True)
sys.exit(0 if rc == 0 else 1)
PY
rc=$?
if [ -f $OUT/result.json ]; then
  echo "--- last lines of result.log"; tail -6 $OUT/result.log | cut -c1-230
  echo "--- summary"; $DIR/_run.sh --summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out
else
  say "no result.json (app exit $rc; see $OUT/app.stdout)"
fi
echo "files: $OUT"
[ -f $OUT/result.json ] || exit 1
/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="done" and r.get("pass") else 3)' $OUT/result.json
