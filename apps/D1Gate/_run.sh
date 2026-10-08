#!/bin/zsh
# Launch D1Gate on the phone and collect its result. No --console (after an install a console launch can be refused as
# Busy for minutes); the app writes Documents/d1_gate/result.log line by line, memory.tsv reading by reading (at least
# once a second while the app lives) and result.json when a stage starts, after every stage (and every 5 records), and
# this script pulls the three every 10 s until result.json carries this run's id with status done / failed, the app's
# process is gone, the app stops writing (memory.tsv still for D1_FREEZE_S s while the process lives: the app is
# terminated by pid and the run recorded), or the cap passes. Copied from apps/KevGate/_run.sh (zoo d1-3b 4955a23);
# the summary is d1's, the freeze check is new (a frozen app keeps its process and writes nothing: Kev round 13).
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"D1_STAGES":"assets,load_jit"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled (or a Mac) result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and on a device D1_ALLOWED_DEVICES
# does not list. App knobs: the D1_* list at the top of Sources/GateRunner.swift.
# Poll cap: D1_CAP polls of 10 s (default 150 = 25 min). When the phone stops answering (unplugged, locked up), the
# script says so and keeps polling until the cap: reconnecting is a person's job.
# A run that does not end "done" (the app gone, frozen, or the cap) also lists the phone's crash logs and copies the ones
# of today that name D1Gate or a jetsam event into crash/.
# Output: _work/device_runs/<run id>/{result.json,result.log,memory.tsv,run.out,launch.log[,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.d1gate
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
RDIR=Documents/d1_gate

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
def num(v, f="%.2f"):
    return (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
d = r.get("device", {})
de = r.get("device_end", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, complete {r.get('complete')}, "
      f"{num(r.get('elapsed_s'), '%.0f')} s | {d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, "
      f"Core AI arch {d.get('coreai_architecture')}), thermal {d.get('thermal')} -> {de.get('thermal')}, battery "
      f"{num(d.get('battery_level', -1) * 100, '%.0f')} % {d.get('battery_state')} ({d.get('power_source')}) -> "
      f"{num((de.get('battery_level') or -1) * 100, '%.0f')} %, low power {d.get('low_power_mode')}, free "
      f"{num(d.get('free_gb'), '%.1f')} GB, available {num(d.get('available_mb'), '%.0f')} MB at launch")
p = r.get("previous_launch")
if p:
    print(f"previous launch: run {p.get('run_id')} {p.get('status')}" + (f", died in {p['died_in_stage']}" if p.get("died_in_stage") else ""))
if r.get("fatal"): print("FATAL", r["fatal"])
S = r.get("stages", {})
def bar(s):
    m, a = s.get("mac", {}), s.get("api_path", {})
    return (f"{s.get('questions')} q | argmax {s.get('argmax_equal_non_near_tie')}/{s.get('questions_non_near_tie')} + near-tie "
            f"{s.get('argmax_equal_near_tie')}/{s.get('near_tie_questions')}, ids {s.get('ids_equal_oracle')}, max|dp| "
            f"{num(s.get('max_abs_dp'), '%.6f')} ({s.get('worst_row')}), mean {num(s.get('mean_of_run_mean_abs_dp'), '%.6f')}, bar "
            f"{'PASS' if s.get('bar_pass') else 'FAIL'} | API path max|dp| {num(a.get('max_abs_dp'), '%.6f')} ({a.get('rows')} rows) "
            f"| Mac: hidden {m.get('hidden_sha256_equal')}/{m.get('rows_compared')}, p bits {m.get('p_bit_equal')}, argmax "
            f"{m.get('argmax_equal')}, max|dp| {num(m.get('max_abs_dp'), '%.6f')}")
for k in r.get("stage_order", []):
    s = S.get(k, {})
    if s.get("deadline_skipped"):
        print(f"{k}: SKIPPED (deadline)"); continue
    tag = f"{k}: {'SKIPPED' if s.get('skipped') else ('PASS' if s.get('pass') else 'FAIL')}" + (" (partial)" if s.get("partial") else "") \
        + (" (cut by the deadline)" if s.get("deadline_stop") else "")
    err = f" | ERROR {s['error']}"[:300] if "error" in s else ""
    if k == "assets":
        print(f"{tag} | {s.get('md5sums_listed')} files, missing {len(s.get('missing', []))}, md5 checked {s.get('md5_checked')} "
              f"(different {len(s.get('md5_mismatch', []))}), {len(s.get('md5_deferred', []))} model files for md5, free "
              f"{num(s.get('free_gb'), '%.1f')} GB{err}")
    elif k == "md5":
        print(f"{tag} | {s.get('list')}: {len(s.get('files', []))} files, {num(s.get('bytes', 0) / 1e6, '%.0f')} MB in "
              f"{num(s.get('seconds'), '%.1f')} s, different {len(s.get('md5_mismatch', []))}{err}")
    elif k in ("load_jit", "load_aot"):
        t, tw, dc = s.get("text", {}), s.get("tower", {}), s.get("decoder", {})
        print(f"{tag} | text {num(t.get('wall_s'))} s | tower {num(tw.get('wall_s'))} s, peak "
              f"{num((tw.get('memory') or {}).get('peak_footprint_mb'), '%.0f')} MB, cache +{num((tw.get('cache_bytes_added') or 0) / 1e6, '%.0f')} MB "
              f"| decoder {s.get('asset', '').split('/')[-1]} {num(s.get('asset_bytes', 0) / 1e6, '%.0f')} MB: wall {num(dc.get('wall_s'))} s "
              f"(AIModel {num(dc.get('decoder_model_s'))} s, main {num(dc.get('decoder_function_s'))} s, tower {num(dc.get('tower_load_s'))} s), "
              f"peak {num((dc.get('memory') or {}).get('peak_footprint_mb'), '%.0f')} MB, least available "
              f"{num((dc.get('memory') or {}).get('min_available_mb'), '%.0f')} MB, cache +{num((dc.get('cache_bytes_added') or 0) / 1e6, '%.0f')} MB{err}")
    elif k == "warm":
        run = s.get("run", {}).get("direct", {})
        print(f"{tag} | {s.get('record')} ({s.get('kind')}): latency {num(run.get('latency_ms'), '%.1f')} ms, {run.get('calls')} calls{err}")
    elif k == "red":
        print(f"{tag} | {s.get('red_arms')}/{s.get('arm_count')} arms red: " + ", ".join(
            f"{a.get('id')} {num(a.get('max_abs_dp_vs_base'), '%.4f')}{' RED' if a.get('red') else ' not red'}" for a in s.get("arms", [])) + err)
    elif k.startswith("e2e_"):
        sm, sh = s.get("summary", {}), s.get("shared_summary", {})
        line = (f"{tag} | {s.get('records_done')} of {s.get('records_planned')} records ({len(s.get('skipped_done', []))} done earlier): "
                f"{bar(sm)} | shared {sh.get('p_bit_equal_direct')}/{sh.get('records')} = direct")
        if s.get("ref_summary"):
            rs = s["ref_summary"]
            line += f" | vs JIT: p bits {rs.get('p_bit_equal')}/{rs.get('rows_compared')}, hidden {rs.get('hidden_sha256_equal')}, max|dp| {num(rs.get('max_abs_dp'), '%.6f')}"
        print(line + f" | {s.get('calls_total')} calls, call ms median {num(s.get('call_ms_median'))}, {num(s.get('seconds'), '%.0f')} s{err}")
        if s.get("union_summary"):
            print(f"    union of the launches: {bar(s['union_summary'])} (runs {s['union_summary'].get('runs')})")
    elif k == "reset":
        print(f"{tag} | {s.get('record')} hidden bit-equal {s.get('hidden_bit_equal')}, p bit-equal {s.get('p_bit_equal')}{err}")
    elif k.startswith("bench"):
        for it in s.get("items", []):
            sm = it.get("summary", {})
            wn = it.get("wait_nominal", {})
            print(f"{tag} | {it.get('item')} ({it.get('mode')}) rows {it.get('row_tokens')}: median {num(sm.get('median'), '%.1f')} ms "
                  f"[{' '.join(num(x, '%.0f') for x in sm.get('latency_ms', []))}] | thermal {it.get('thermal_start')} -> "
                  f"{it.get('thermal_end')} (waited {num(wn.get('waited_s'), '%.0f')} s), battery "
                  f"{num(it.get('battery_start', {}).get('level', -1) * 100, '%.0f')} -> {num(it.get('battery_end', {}).get('level', -1) * 100, '%.0f')} % "
                  f"{it.get('battery_end', {}).get('power')}, timed runs {num(it.get('timed_runs_start_s'), '%.1f')}-{num(it.get('timed_runs_end_s'), '%.1f')} s"
                  + (f" | ERROR {it['error']}" if 'error' in it else ""))
    elif k == "delete":
        print(f"{tag} | {[x.get('path') for x in s.get('deleted', [])]} free {num(s.get('free_gb_before'), '%.1f')} -> "
              f"{num(s.get('free_gb_after'), '%.1f')} GB{err}")
    else:
        print(f"{tag}{err}")
print("summary:", " ".join(r.get("summary", [])))
PY
}

if [ "${1:-}" = "--summary" ]; then summary "${2:?usage: _run.sh --summary <result.json>}"; exit 0; fi
UDID=${1:?usage: _run.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
RUN_ID=${D1_RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT=$W/device_runs/$RUN_ID
mkdir -p $OUT
CAP=${D1_CAP:-150}
FREEZE=${D1_FREEZE_S:-180}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { say "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^d1-3b device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
say "hold: $(head -1 $HOLD)"
IDS=($UDID ${(s:,:)${D1_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/D1Assets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"D1_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
launched=0
for t in $(seq 1 12); do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "$ENVJ" $BID 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-160)"; sleep 15
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched run $RUN_ID (env $ENVJ); polling every 10 s, cap $((CAP * 10)) s, freeze after ${FREEZE} s of a still memory.tsv"

pull() {  # pull <name>: Documents/d1_gate/<name> -> $OUT/<name> (kept when the pull fails)
  rm -f $OUT/$1.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source $RDIR/$1 --destination $OUT/$1.pull >/dev/null 2>&1 && [ -f $OUT/$1.pull ] && mv $OUT/$1.pull $OUT/$1
}
app_pid() {  # the D1Gate process on the phone: its pid, or nothing (from the JSON listing: any object naming the app's
  # executable, its integer member whose name says pid / processIdentifier)
  local J=$OUT/processes.json
  rm -f $J
  xcrun devicectl device info processes --device $UDID --json-output $J >/dev/null 2>&1 || return 0
  /usr/bin/python3 - $J <<'PY'
import json, sys
def walk(v):
    if isinstance(v, dict):
        if any(isinstance(x, str) and "D1Gate.app/D1Gate" in x for x in v.values()):
            for k, x in v.items():
                if isinstance(x, int) and ("pid" in k.lower() or "processidentifier" in k.lower()):
                    print(x); sys.exit(0)
        for x in v.values(): walk(x)
    elif isinstance(v, list):
        for x in v: walk(x)
try:
    walk(json.load(open(sys.argv[1])))
except Exception:
    pass
PY
}
state=""; last=""; gone=0; silent=0; frozen=0
msize=-1; mchanged=$SECONDS
for i in $(seq 1 $CAP); do
  sleep 10
  pull result.log; pull result.json; pull memory.tsv
  if [ -f $OUT/result.log ]; then
    cur=$(tail -1 $OUT/result.log)
    [ "$cur" != "$last" ] && { echo "  ${cur[1,260]}"; last=$cur; }
  fi
  if [ -f $OUT/memory.tsv ]; then
    sz=$(stat -f %z $OUT/memory.tsv)
    if [ "$sz" != "$msize" ]; then msize=$sz; mchanged=$SECONDS; fi
  fi
  if [ -f $OUT/result.json ]; then
    state=$(/usr/bin/python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("status") if r.get("run_id")==sys.argv[2] else "other-run")' \
      $OUT/result.json $RUN_ID 2>/dev/null)
    [[ $state == done || $state == failed ]] && break
  fi
  # every minute: is the app still running? (two misses in a row = gone; a listing that fails says nothing about the app,
  # and three failures in a row say the phone stopped answering)
  if (( i % 6 == 0 )); then
    if ! P=$(xcrun devicectl device info processes --device $UDID 2>&1); then
      silent=$((silent + 1))
      (( silent == 3 )) && say "the phone has not answered for 3 min (unplugged? $(echo "$P" | grep -m1 -iE 'error' | cut -c1-120)); still polling until the cap"
      continue
    fi
    silent=0
    if echo "$P" | grep -q "D1Gate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? jetsam? see result.log)"; break; }
    # alive but writing nothing: a frozen app (the memory sampler writes every second while it lives)
    if (( msize >= 0 && SECONDS - mchanged >= FREEZE )); then
      pid=$(app_pid)
      say "FROZEN: memory.tsv has not grown for $((SECONDS - mchanged)) s while the app's process lives (pid ${pid:-?}); terminating it"
      frozen=1
      if [ -n "$pid" ]; then
        xcrun devicectl device process terminate --device $UDID --pid $pid >> $OUT/run.out 2>&1 && say "terminated pid $pid"
      fi
      sleep 5
      pull result.log; pull result.json; pull memory.tsv
      break
    fi
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s$( (( frozen )) && echo ' (frozen, terminated)')"
if [[ $state != done ]]; then
  # crash reports: the listing, then today's files that name the app or a jetsam event
  CL=$(xcrun devicectl device info files --device $UDID --domain-type systemCrashLogs 2>&1)
  echo "$CL" > $OUT/crashlogs_listing.txt
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(D1Gate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names D1Gate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-260; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
