#!/bin/zsh
# Run the macOS build of D1Gate on this Mac on the Mac's own AOT assets (h16c), short, and print its summary: the check
# that the harness scores what the Mac's gates scored (round 5c's readout gate, round 8's picture reference), before the
# phone. A correctness run, not a timing: it never takes the machine-wide GPU lock (~/code/coreai/_GPU_LOCK) and starts
# only while no measurement window is open (the lock names `timing` and its holder lives) and no process holds the
# lock file open (lsof: a kit flock); the other GPU jobs of the lanes and the GPU's utilization are recorded, not waited
# for. Polled every 30 s for D1_LOCK_WAIT s (default 1800), then it gives up (exit 4). While the app runs, a window that
# opens stops it (SIGSTOP within 0.2 s) until the window closes (SIGCONT): every stop is in gpu_lock.json.
# Adapted from apps/KevGate/_run_mac.sh (zoo d1-3b 4955a23).
#   ./_run_mac.sh                  D1_STAGES=load_aot,red,e2e_images,e2e_fixture,reset on the stage directory, the 20
#                                  text records of D1_IDS and the 3 picture records of D1_IMAGE_IDS (defaults below);
#                                  D1_AOT = the lane's h16c decoder asset, D1_TOWER = the lane's tower bundle with
#                                  D1_TOWER_ASSET=aot (its h16c asset beside it) — never the phone's h19p: the app
#                                  refuses it on macOS
#   ./_run_mac.sh --red            the same with an oracle_slim whose tv4_000 answer probabilities are reversed: the bar
#                                  must fail (D1_STAGES=load_aot,e2e_fixture)
#   any D1_* knob of GateRunner.swift passes through the environment
# Output: _work/mac_runs/<run id>/{result.json,result.log,memory.tsv,app.stdout,gpu_lock.json,run.out}
set -u
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
L=${D1_LANE:-$HOME/code/coreai/_d1_3b}
APP=$(cat $W/app_path_mac.txt 2>/dev/null)
[ -d "$APP" ] || { echo "no macOS build: run ./_build.sh --mac"; exit 1; }
BIN=$APP/Contents/MacOS/D1Gate
S=${D1_ASSETS:-$W/device_stage/D1Assets}
[ -f $S/MD5SUMS ] || { echo "nothing staged at $S: run ./_stage.sh"; exit 1; }
RED=0
[ "${1:-}" = "--red" ] && RED=1
RUN_ID=${D1_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)$( (( RED )) && echo -red)}
OUT=$W/mac_runs/$RUN_ID
mkdir -p $OUT
LOCK=${COREAI_GPU_LOCK:-$HOME/code/coreai/_GPU_LOCK}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }
export D1_ASSETS=$S D1_OUT=$OUT D1_RUN_ID=$RUN_ID
export D1_AOT=${D1_AOT:-$L/exports/bundles_aotc/d1_3b_decode_int8mlp_pf16.h16c.aimodelc}
export D1_TOWER=${D1_TOWER:-$L/exports/vision/d1_3b_vision_fp16w32} D1_TOWER_ASSET=${D1_TOWER_ASSET:-aot}
# load_tower_aot on the Mac: the tower's h16c AOT (the phone's h19p is refused here)
export D1_TOWER_AOT=${D1_TOWER_AOT:-$L/exports/vision_aotc/d1_3b_vision_fp16w32.h16c.aimodelc}
# 20 text records over every source, the red arms' bases, int8mlp's worst row (tv4x_composition_holdout_12), the refused
# request (own_email_03), multi-question requests (shared), a 1.4k-token state (long_15k); round 3b's 3 picture records
export D1_IDS=${D1_IDS:-tv4_000,tv4_001,tv4_002,tv4x_qnli_00,tv4x_paws_00,tv4x_emotion_00,tv4x_composition_holdout_12,tv4s_00,tv4s_01,semif_a3f18f3a63d45345942b,semif_0b43ea8e24e74d621f1b,semif_33d6e5da58f0fee2490d,semif_8e8c3804a3c15ebb31e3,own_ticket_01,own_email_03,own_order_06,own_fiveq_09,card_refund,card_ticket_00,long_15k}
export D1_IMAGE_IDS=${D1_IMAGE_IDS:-img01_shapes_384x384,img06_grid_1024x768,img12_small_300x300}
if (( RED )); then
  export D1_STAGES=${D1_STAGES:-load_aot,e2e_fixture}
else
  export D1_STAGES=${D1_STAGES:-load_aot,red,e2e_images,e2e_fixture,reset}
fi
[[ $D1_AOT == *.h16c.aimodelc ]] || { say "D1_AOT $D1_AOT: the Mac runs the h16c asset only"; exit 1; }
[[ $D1_TOWER_AOT == *.h16c.aimodelc ]] || { say "D1_TOWER_AOT $D1_TOWER_AOT: the Mac runs the h16c asset only"; exit 1; }
if (( RED )); then
  /usr/bin/python3 - $S/fixtures/oracle_slim.json $OUT/oracle_slim_red.json <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))
q = o["records"]["tv4_000"]["questions"][0]
before = list(q["probs"])
q["probs"] = before[::-1]          # reversed: |dp| > 0.02 on the row (the oracle's argmax key stays)
o["red_arm"] = {"row": "tv4_000:" + q["name"], "probs_before": before, "probs_after": q["probs"]}
json.dump(o, open(sys.argv[2], "w"))
print(f"red arm: tv4_000 {q['name']} probs {before} -> {q['probs']}")
PY
  export D1_ORACLE=$OUT/oracle_slim_red.json
fi
say "run $RUN_ID: $BIN on $S, D1_AOT $D1_AOT, tower $D1_TOWER ($D1_TOWER_ASSET), stages $D1_STAGES${D1_ORACLE:+, oracle $D1_ORACLE}"

/usr/bin/python3 - "$LOCK" "$OUT" "$BIN" "${D1_LOCK_WAIT:-1800}" "${D1_MAC_CAP:-2400}" <<'PY' 2>&1 | tee -a $OUT/run.out
import json, os, re, signal, subprocess, sys, time
lock, out, binary, wait_cap, run_cap = sys.argv[1:6]
wait_cap, run_cap = float(wait_cap), float(run_cap)
GPU = re.compile(r"readout_gate\.py|timing\.py (run|worker)|decide\.py (check|worker|run)|--accel gpu|gpu_gate_subprocess|"
                 r"litert_parity\.py .*gpu|llm-bench|yardstick|coreai_verify|mlx_lm|/d1 (fixture|decide|prepare-test) |"
                 r"gate_swift\.py|with-gpu-lock|parity_catalog")
me = os.getpid()

def window():
    """the lock's content when a live measurement window is open (quiet_wait.py's rule), else None"""
    try:
        st = os.stat(lock)
        c = open(lock, encoding="utf-8", errors="replace").read().strip()
    except OSError:
        return None
    if not c or "timing" not in c.lower():
        return None
    m = re.search(r"\bpid\s+(\d+)", c)
    if m:
        try:
            os.kill(int(m.group(1)), 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
    elif time.time() - st.st_mtime > 3 * 3600:
        return None
    return c

def reading():
    try:
        tag = open(lock).read().strip()
    except FileNotFoundError:
        tag = None
    lsof = subprocess.run(["lsof", lock], capture_output=True, text=True).stdout.strip().splitlines()[1:]
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
    jobs = [l.strip()[:160] for l in ps if GPU.search(l) and "claude" not in l and "grep" not in l
            and int(l.split(None, 1)[0]) != me and "D1Gate" not in l]
    vals = []
    for i in range(3):
        o = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        m = re.findall(r'"Device Utilization %"=(\d+)', o)
        vals.append(max(map(int, m)) if m else -1)
        if i < 2: time.sleep(1)
    load1 = os.getloadavg()[0]
    w = window()
    return {"t": time.strftime("%H:%M:%S"), "tag": tag, "window": w, "lsof": lsof, "gpu_jobs": jobs, "gpu_util": vals,
            "load1": round(load1, 2), "free": not w and not lsof}

rec = {"lock": lock, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "policy": "a correctness run: start only while no measurement window is open and no process holds the lock; never "
                 "take it; a window that opens while the app runs stops it (SIGSTOP) until it closes (SIGCONT)"}
t0 = time.time()
waits = []
while True:
    r = reading()
    waits.append(r)
    if r["free"]:
        break
    if time.time() - t0 >= wait_cap:
        rec.update({"state": f"not free after {time.time() - t0:.0f} s: not run", "waits": waits})
        json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
        print(f"not free after {time.time() - t0:.0f} s (window {r['window']!r}, lsof {len(r['lsof'])}): not running", flush=True)
        sys.exit(4)
    print(f"waiting 30 s: window {r['window']!r}, lsof {len(r['lsof'])} ({time.time() - t0:.0f} s so far, cap {wait_cap:.0f} s)", flush=True)
    time.sleep(30)
rec.update({"state": f"free after {time.time() - t0:.0f} s", "waited_s": round(time.time() - t0, 1), "waits": waits})
print(f"no window, lock not held ({rec['state']}; other GPU jobs {len(r['gpu_jobs'])}, util {r['gpu_util']}, load {r['load1']}): "
      f"running the app", flush=True)
during, stops = [], []
t1 = time.time()
stopped_at = None
last_reading = 0.0
with open(f"{out}/app.stdout", "w") as so:
    p = subprocess.Popen([binary], env=dict(os.environ), stdout=so, stderr=subprocess.STDOUT)
    while p.poll() is None:
        now = time.time()
        if now - t1 > run_cap:
            print(f"the app ran past {run_cap:.0f} s: terminating pid {p.pid}", flush=True)
            if stopped_at is not None:
                os.kill(p.pid, signal.SIGCONT)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            break
        w = window()
        if w and stopped_at is None:
            os.kill(p.pid, signal.SIGSTOP)
            stopped_at = time.time()
            stops.append({"window": w[:120], "stop": time.strftime("%H:%M:%S"), "t_s": round(stopped_at - t1, 1)})
            print(f"window opened ({w[:80]}): app stopped", flush=True)
        elif not w and stopped_at is not None:
            os.kill(p.pid, signal.SIGCONT)
            stops[-1].update({"cont": time.strftime("%H:%M:%S"), "stopped_s": round(time.time() - stopped_at, 1)})
            print(f"window closed: app continued after {time.time() - stopped_at:.0f} s", flush=True)
            stopped_at = None
        if now - last_reading >= 10:
            rr = reading()
            rr["t_s"] = round(now - t1, 1)
            rr["app_stopped"] = stopped_at is not None
            during.append(rr)
            last_reading = time.time()
        time.sleep(0.2)
rec.update({"app_exit": p.returncode, "app_s": round(time.time() - t1, 1), "during": during, "stops": stops,
            "stopped_s_total": round(sum(s.get("stopped_s", 0) for s in stops), 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
json.dump(rec, open(f"{out}/gpu_lock.json", "w"), indent=1)
print(f"app exit {p.returncode} after {rec['app_s']} s; stopped by windows {len(stops)} times, {rec['stopped_s_total']} s; "
      f"readings while it ran: {len(during)}", flush=True)
sys.exit(0 if p.returncode == 0 else 1)
PY
rc=${pipestatus[1]}
if [ -f $OUT/result.json ]; then
  echo "--- last lines of result.log"; tail -6 $OUT/result.log | cut -c1-230
  echo "--- summary"; $DIR/_run.sh --summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out
else
  say "no result.json (exit $rc; see $OUT/app.stdout)"
fi
echo "files: $OUT"
[ -f $OUT/result.json ] || exit ${rc:-1}
/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="done" and r.get("pass") else 3)' $OUT/result.json
