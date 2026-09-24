#!/bin/zsh
# Launch N3DGate on the phone and collect its result. No --console (after an install a console launch can be
# refused as Busy for minutes); the app writes Documents/n3d_gate/result.log line by line and result.json after
# every stage, and this script pulls both every 10 s until result.json carries this run's id with status
# done / failed, the app's process is gone, or the cap passes. Then prints the summary.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"N3D_STAGES":"gpu,offline"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled result again
# App knobs (GateRunner.swift): N3D_STAGES, N3D_MODES, N3D_BENCH, N3D_GPU_UNIT, N3D_ANE_UNIT.
# Poll cap: N3D_CAP polls of 10 s (default 180 = 30 min; a first ANE load can take minutes).
# Output: _work/device_runs/<run id>/{result.json,result.log,run.out,launch.log}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.n3dgate
DIR=${0:A:h}
W=${N3D_WORK:-${DIR:h:h}/conversion/nemotron3_diar/_work}

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
d = r.get("device", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {r.get('elapsed_s', 0):.0f} s | "
      f"{d.get('machine')} {d.get('os')} (build {d.get('os_build')}), thermal {d.get('thermal')} -> "
      f"{r.get('device_end', {}).get('thermal')}, low power {d.get('low_power_mode')}")
if r.get("fatal"): print("FATAL", r["fatal"])
a = r.get("assets", {})
print(f"assets: {a.get('md5sums_listed')} files listed, missing {len(a.get('missing', []))}")
for k in r.get("stage_order", []):
    s = r["stages"][k]
    if "skipped" in s: print(f"{k}: skipped ({s['skipped']})"); continue
    if "error" in s and "runs" not in s: print(f"{k}: ERROR {s['error']}"); continue
    print(f"{k}: {'PASS' if s.get('pass') else 'FAIL'} | {s.get('bundle')} {s.get('bundle_mb', 0):.1f} MB, unit {s.get('unit')} | "
          f"load {s.get('load_first_s', 0):.2f} s first / {s.get('load_second_s', 0):.2f} s second, first call "
          f"{s.get('first_call_ms', 0):.1f} ms, footprint {s.get('footprint_mb_after_load', 0):.0f} MB")
    for rk in s.get("run_order", []):
        x = s["runs"].get(rk, {}); c = x.get("comparison")
        line = f"  {rk}: {x.get('steps')} steps, {x.get('graph_ms_median', 0):.2f} ms/step median"
        if c:
            g = c["segments"]
            line += (f" | agreement {100 * c['agreement']:.4f} % ({c['disagree']}/{c['elements']}), max|dp| {c['max_abs_p']:.3e}"
                     + (f", max|dlogit| {c['max_abs_logit']:.3e}" if 'max_abs_logit' in c else "")
                     + f", segments {g['n_ref']}/{g['n_ours']}/{g['matched']} shift {g['max_start_shift']}/{g['max_end_shift']}"
                     f" structural {g['structural']} -> {'PASS' if x.get('pass') else 'FAIL'}")
        print(line)
        for key, label in (("vs_mac_gpu", "vs Mac GPU"), ("vs_gpu_stage", "vs gpu_streaming")):
            if key in x:
                v = x[key]
                print(f"    {label}: max|dlogit| {v['max_abs_logit']:.3e}, bit-equal {v['bit_equal']}/{v['elements']}, "
                      f"decisions equal {v['decisions_equal']}/{v['elements']}")
    b = s.get("bench")
    if b:
        print(f"  bench on {b['on']}: {b['ms_per_chunk_median']:.2f} ms/chunk median, p90 {b['ms_per_chunk_p90']:.2f} "
              f"({b['calls']} calls after {b['warmup']} warm-up) | {b['audio_s']:.1f} s audio: wall {b['loop_wall_s']:.3f} s, "
              f"RTF {b['loop_rtf']:.4f} | thermal {b['thermal_start']} -> {b['thermal_end']}")
    th = s.get("thermal", [])
    if th: print("  thermal: " + ", ".join(f"{t['at']} {t['state']}" for t in th))
print("summary:", " ".join(r.get("summary", [])))
PY
}

if [ "${1:-}" = "--summary" ]; then summary "${2:?usage: _run.sh --summary <result.json>}"; exit 0; fi
UDID=${1:?usage: _run.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT=$W/device_runs/$RUN_ID
mkdir -p $OUT
CAP=${N3D_CAP:-180}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

HOLD=${N3D_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if [ -e $HOLD ] && [ "${N3D_IGNORE_HOLD:-0}" != 1 ]; then
  say "device hold present ($HOLD: $(head -c 200 $HOLD)); N3D_IGNORE_HOLD=1 to go anyway"; exit 2
fi
busy() { ps -axo pid,command | grep -qE "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/N3DAssets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"N3D_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
launched=0
for t in $(seq 1 12); do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "$ENVJ" $BID 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-140)"; sleep 15
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched run $RUN_ID (env $ENVJ); polling every 10 s, cap $((CAP * 10)) s"

state=""; last=""; gone=0
for i in $(seq 1 $CAP); do
  sleep 10
  rm -f $OUT/result.log.pull $OUT/result.json.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source Documents/n3d_gate/result.log --destination $OUT/result.log.pull >/dev/null 2>&1 \
    && [ -f $OUT/result.log.pull ] && mv $OUT/result.log.pull $OUT/result.log
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source Documents/n3d_gate/result.json --destination $OUT/result.json.pull >/dev/null 2>&1 \
    && [ -f $OUT/result.json.pull ] && mv $OUT/result.json.pull $OUT/result.json
  if [ -f $OUT/result.log ]; then
    cur=$(tail -1 $OUT/result.log)
    [ "$cur" != "$last" ] && { echo "  ${cur[1,220]}"; last=$cur; }
  fi
  if [ -f $OUT/result.json ]; then
    state=$(/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("status") if r.get("run_id")==sys.argv[2] else "other-run")' \
      $OUT/result.json $RUN_ID 2>/dev/null)
    [[ $state == done || $state == failed ]] && break
  fi
  # every minute: is the app still running? (two misses in a row = gone; a listing that fails says nothing)
  if (( i % 6 == 0 )); then
    P=$(xcrun devicectl device info processes --device $UDID 2>/dev/null) || continue
    if echo "$P" | grep -q "N3DGate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? see result.log)"; break; }
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s"
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-220; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] && /usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
