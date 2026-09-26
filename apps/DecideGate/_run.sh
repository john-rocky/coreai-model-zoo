#!/bin/zsh
# Launch DecideGate on the phone and collect its result. No --console (after an install a console launch can be
# refused as Busy for minutes); the app writes Documents/decide_gate/result.log line by line and result.json after
# every stage, and this script pulls both every 10 s until result.json carries this run's id with status
# done / failed, the app's process is gone, or the cap passes. Then prints the summary.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"DECIDE_STAGES":"s256"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and (when DECIDE_ALLOWED_DEVICES
# is set) on a device it does not list.
# App knobs (GateRunner.swift): DECIDE_STAGES, DECIDE_BENCH, DECIDE_UNIT, DECIDE_ASSETS, DECIDE_WAIT_NOMINAL,
# DECIDE_BENCH_FIRST, DECIDE_BUNDLE_KIND, DECIDE_LOAD_ONLY.
# Poll cap: DECIDE_CAP polls of 10 s (default 180 = 30 min).
# A run that does not end "done" (the app gone, or the cap) also lists the phone's crash logs and copies the ones of
# today that name DecideGate or a jetsam event into crash/.
# Output: _work/device_runs/<run id>/{result.json,result.log,run.out,launch.log[,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.decidegate
DIR=${0:A:h}
W=${DECIDE_WORK:-${DIR:h:h}/conversion/gliner25_decide/_work}

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
d = r.get("device", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {r.get('elapsed_s', 0):.0f} s | "
      f"{d.get('machine')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')} -> {r.get('device_end', {}).get('thermal')}, low power {d.get('low_power_mode')}")
if r.get("fatal"): print("FATAL", r["fatal"])
a = r.get("assets", {})
print(f"assets: {a.get('md5sums_listed')} files listed, missing {len(a.get('missing', []))}")
num = lambda v, f: (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
mbv = lambda b: num(b / 1e6 if isinstance(b, (int, float)) else None, "%.1f")
for k in r.get("stage_order", []):
    s = r["stages"][k]
    if s.get("mode") == "load_only":
        # one line: bundle, load 1 (wall, peak footprint, least available), first call, load 2, container cache, error
        if s.get("partial"): state = f"STOPPED during {s.get('step', '?')} (partial record: the app did not finish the stage)"
        elif s.get("pass"): state = "OK"
        else: state = f"ERROR at {s.get('error_step', '?')}"
        call = (f"first call {num(s.get('first_call_ms'), '%.1f')} ms" if "first_call_ms" in s
                else f"first call skipped ({s['first_call_skipped']})" if "first_call_skipped" in s else "first call -")
        out = (f"{k}: {state} | {s.get('bundle')} {num(s.get('bundle_mb'), '%.1f')} MB | load 1 {num(s.get('load_first_wall_s'), '%.2f')} s "
               f"wall, peak footprint {num(s.get('load_first_peak_footprint_mb'), '%.0f')} MB, least available "
               f"{num(s.get('load_first_min_available_mb'), '%.0f')} MB (before {num(s.get('available_mb_before_load'), '%.0f')}) | "
               f"{call} | load 2 {num(s.get('load_second_s'), '%.2f')} s | coreai-cache MB {mbv(s.get('cache_bytes_before_load'))} -> "
               f"{mbv(s.get('cache_bytes_after_load'))} after load 1 -> {mbv(s.get('cache_bytes_after_load_2'))} after load 2")
        if "error_detail" in s:
            e = s["error_detail"]
            out += f" | error {e.get('type')} | NSError {e.get('ns_domain')} {e.get('ns_code')} | {e.get('reflecting')}"
        elif "error" in s: out += f" | error {s['error']}"
        print(out[:900])
        continue
    if "error" in s and "summary" not in s: print(f"{k}: ERROR {s['error']}"); continue
    tag = " (partial: cases done, bench running)" if s.get("partial") else ""
    print(f"{k}: {'PASS' if s.get('pass') else 'FAIL'}{tag} | {s.get('bundle')} {s.get('bundle_mb', 0):.1f} MB, unit {s.get('unit')} | "
          f"load {s.get('load_first_s', 0):.2f} s first / {s.get('load_second_s', 0):.2f} s second, first call "
          f"{s.get('first_call_ms', 0):.1f} ms, footprint {s.get('footprint_mb_after_load', 0):.0f} MB")
    if "load_first_memory" in s:
        mb = lambda b: f"{(b or 0) / 1e6:.1f}"
        print(f"  kind {s.get('bundle_kind')} | load 1 wall {s.get('load_first_wall_s', 0):.2f} s: peak footprint "
              f"{s.get('load_first_peak_footprint_mb', -1):.0f} MB, least available {s.get('load_first_min_available_mb', -1):.0f} MB "
              f"(before {s.get('available_mb_before_load', -1):.0f} MB) | peak first call "
              f"{s.get('first_call_peak_footprint_mb', -1):.0f} MB, load 2 {s.get('load_second_peak_footprint_mb', -1):.0f} MB")
        print(f"  coreai-cache MB: before load 1 {mb(s.get('cache_bytes_before_load'))}, after load 1 {mb(s.get('cache_bytes_after_load'))} "
              f"({s.get('cache_files_after_load', 0)} files), after first call {mb(s.get('cache_bytes_after_first_call'))}, after load 2 "
              f"{mb(s.get('cache_bytes_after_load_2'))}, end {mb(s.get('cache_bytes_end'))}")
    m = s.get("summary")
    if m:
        print(f"  cases {m['cases']}, tasks {m['tasks']}: decisions equal {m['decisions_equal_tasks']}/{m['tasks']} "
              f"({m['cases_with_changed_decision']} cases changed), max|dlogit| {m['max_abs_dlogit']:.4g}, "
              f"max|dprob| {m['max_abs_dprob']:.4g}, min cos {m['min_cos']:.7f}, non-finite {m['nonfinite_cases']}, "
              f"call median {m['warm_call_ms_median']:.2f} ms over the cases")
    v = s.get("vs_mac_gpu")
    if v:
        print(f"  vs Mac GPU (same bundle): {v['cases']} cases, max|dlogit| {v['max_abs_dlogit']:.4g}, bit-equal "
              f"{v['bit_equal']}/{v['elements']}, decisions equal {v['decisions_equal_tasks']}/{v['tasks']}")
    b = s.get("bench")
    if b:
        w = b.get("wait_nominal")
        wt = (f" | waited {w['waited_s']:.1f} s for nominal ({w['state_before']} -> {w['state_after']}, cap {w['cap_s']:.0f} s)"
              if w else "")
        print(f"  bench ({b.get('position', 'after the cases')}): {b['ms_median']:.2f} ms median, p90 {b['ms_p90']:.2f}, "
              f"min {b['ms_min']:.2f}, max {b['ms_max']:.2f} ({b['calls']} calls after {b['warmup']} warm-up) | thermal "
              f"{b['thermal_start']} -> {b['thermal_end']}{wt}")
    if "error" in s: print(f"  ERROR {s['error']}")
    if "error_detail" in s:
        e = s["error_detail"]
        print(f"  error {e.get('type')} | NSError {e.get('ns_domain')} {e.get('ns_code')} | {e.get('reflecting')}"[:400])
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
CAP=${DECIDE_CAP:-180}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

if [ -n "${DECIDE_ALLOWED_DEVICES:-}" ] && [[ ",$DECIDE_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in DECIDE_ALLOWED_DEVICES ($DECIDE_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${DECIDE_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^GLiNER2.5-Decide device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
say "hold: $(head -1 $HOLD)"
IDS=($UDID ${(s:,:)${DECIDE_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/DecideAssets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"DECIDE_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
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
    --source Documents/decide_gate/result.log --destination $OUT/result.log.pull >/dev/null 2>&1 \
    && [ -f $OUT/result.log.pull ] && mv $OUT/result.log.pull $OUT/result.log
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source Documents/decide_gate/result.json --destination $OUT/result.json.pull >/dev/null 2>&1 \
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
    if echo "$P" | grep -q "DecideGate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? see result.log)"; break; }
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s"
if [[ $state != done ]]; then
  # crash reports: the listing, then today's files that name the app or a jetsam event
  CL=$(xcrun devicectl device info files --device $UDID --domain-type systemCrashLogs 2>&1)
  echo "$CL" > $OUT/crashlogs_listing.txt
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(DecideGate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names DecideGate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-220; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
