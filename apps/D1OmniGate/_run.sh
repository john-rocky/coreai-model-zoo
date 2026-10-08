#!/bin/zsh
# Launch D1OmniGate on the phone and collect its result. No --console (after an install a console launch can be refused as
# Busy for minutes); the app writes Documents/d1omni_gate/result.log line by line, memory.tsv every 100 ms while it lives
# and result.json when a stage starts, after every stage (and every 10 rows, every bench series), and this script pulls
# the three every 10 s until result.json carries this run's id with status done / failed, the app's process is gone, the
# app stops writing (memory.tsv still for D1_FREEZE_S s while the process lives: the app is terminated by pid and the run
# recorded), or the cap passes. Copied from apps/KevGate/_run.sh with the freeze check of the d1-3b lane's
# apps/D1Gate/_run.sh (a frozen app keeps its process and writes nothing: Kev round 13); the summary is d1-omni's.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"D1_STAGES":"assets,parity_jit"'
#   ./_run.sh --summary <result.json> [run id]       print the summary of a pulled (or a Mac) result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and on a device D1_ALLOWED_DEVICES
# does not list. App knobs: the D1_* list at the top of Sources/GateRunner.swift.
# Poll cap: D1_CAP polls of 10 s (default 150 = 25 min). When the phone stops answering (unplugged, locked up), the
# script says so and keeps polling until the cap: reconnecting is a person's job.
# A run that does not end "done" (the app gone, frozen, or the cap) also lists the phone's crash logs and copies the ones
# of today that name D1OmniGate or a jetsam event into crash/.
# Output: $D1_WORK/runs/<run id>/{result.json,result.log,memory.tsv,run.out,launch.log[,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.d1omnigate
DIR=${0:A:h}
W=${D1_WORK:-$HOME/code/coreai/_d1_omni/device}
RDIR=Documents/d1omni_gate

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 -I - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
def num(v, f="%.2f"):
    return (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
def bar(s):
    m = s.get("mac", {})
    return (f"{s.get('rows')} rows | argmax {s.get('argmax_equal_non_near_tie')}/{s.get('non_near_tie')} + near-tie "
            f"{s.get('argmax_equal_near_tie')}/{s.get('near_tie')}, ids {s.get('ids_markers_bucket_equal')}, max|dp| "
            f"{num(s.get('max_abs_dp'), '%.6f')} ({s.get('max_abs_dp_row')}), mean {num(s.get('mean_row_max_abs_dp'), '%.6f')}, bar "
            f"{'PASS' if s.get('bar_pass') else 'FAIL'} | Mac: logits bits {m.get('logits_bit_equal')}, p bits "
            f"{m.get('probs_bit_equal')}/{m.get('rows')}, max|dp| {num(m.get('max_abs_dp'), '%.2e')}, max|dlogit| {num(m.get('max_abs_dlogit'), '%.2e')}")
d, de = r.get("device", {}), r.get("device_end", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {num(r.get('elapsed_s'), '%.0f')} s | "
      f"{d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')} -> {de.get('thermal')}, battery {num((d.get('battery_level') or -1) * 100, '%.0f')} % "
      f"{d.get('battery_state')} ({d.get('power_source')}) -> {num((de.get('battery_level') or -1) * 100, '%.0f')} %, low power "
      f"{d.get('low_power_mode')}, free {num(d.get('free_gb'), '%.1f')} GB, available {num(d.get('available_mb'), '%.0f')} MB at launch")
p = r.get("previous_launch")
if p:
    print(f"previous launch: run {p.get('run_id')} {p.get('status')}" + (f", died in {p['died_in_stage']}" if p.get("died_in_stage") else ""))
if r.get("fatal"): print("FATAL", r["fatal"])
def parity_lines(tag, s, memt, err):
    groups = s.get("groups", [])
    inits = ", ".join(f"{g.get('group')} {num(g.get('instance', {}).get('init_s'))} s" for g in groups)
    print(f"{tag} | init {inits}{memt}{err}")
    for ld in s.get("loads", []):
        print(f"   load {ld.get('group', '')} {ld.get('what')}: {ld.get('kind')} wall {num(ld.get('wall_s'))} s (AIModel "
              f"{num(ld.get('model_s'))}, main {num(ld.get('function_s'))}), cache +{num((ld.get('cache_bytes_added') or 0) / 1e6, '%.0f')} MB, "
              f"peak {num(ld.get('memory', {}).get('peak_footprint_mb'), '%.0f')} MB" + (f" | ERROR {ld['error']}"[:200] if "error" in ld else ""))
    if s.get("summary"): print(f"   all: {bar(s['summary'])}")
    for m, sm in (s.get("summary_by_mode") or {}).items():
        if sm.get("rows"): print(f"   {m}: {bar(sm)}")
    if s.get("rows_l64_planned"):
        print(f"   L64 ({s.get('rows_l64_done')}/{s.get('rows_l64_planned')} rows): {bar(s.get('summary_l64', {}))}"
              + (f" | vs JIT L64: logits {s['vs_jit_l64'].get('logits_bit_equal')}/{s['vs_jit_l64'].get('rows')}" if s.get("vs_jit_l64") else ""))
    for g in groups:
        print(f"   group {g.get('group')}: buckets {g.get('instance', {}).get('buckets')}, peak footprint "
              f"{num(g.get('memory', {}).get('peak_footprint_mb'), '%.0f')} MB, least available "
              f"{num(g.get('memory', {}).get('min_available_mb'), '%.0f')} MB, footprint at the end {num(g.get('footprint_mb_end'), '%.0f')} MB")
    c = s.get("control", {})
    ma = s.get("media_arrays_equal_mac", {})
    print(f"   control {c.get('status')} (must FAIL; {c.get('rows_over_dp_bar')} of {c.get('paired_rows')} rows over the bar), "
          f"media arrays = Mac {ma.get('equal')}/{ma.get('items')}, rows {s.get('rows_done')}/{s.get('rows_planned')} in "
          f"{num(s.get('seconds_rows'), '%.1f')} s" + (f", stopped: {s['stopped']}" if s.get("stopped") else "")
          + (f" | vs JIT: logits {s['vs_jit'].get('logits_bit_equal')}/{s['vs_jit'].get('rows')}, p {s['vs_jit'].get('probs_bit_equal')}" if s.get("vs_jit") else ""))
    if s.get("errors"): print(f"   errors: {s['errors'][:3]}")
def bench_lines(tag, s, memt, err):
    print(f"{tag} | rules {s.get('rules')}{memt}{err}")
    for it in s.get("items", []):
        if it.get("skipped"):
            print(f"   {it.get('workload')}: SKIPPED ({it.get('reason')})"); continue
        fo = it.get("forms", {})
        oc = it.get("output_check", {})
        ser = [x for b in it.get("blocks", []) for x in b.get("series", [])]
        print(f"   {it.get('workload')} (buckets {it.get('buckets')}, interleave {it.get('interleave')}): " + ", ".join(
            f"{n} median {num(v.get('median_ms'))} ms (p10 {num(v.get('p10_ms'))}, p90 {num(v.get('p90_ms'))}, n {v.get('n')}, "
            f"thermal {v.get('thermal_of_timed')})" for n, v in sorted(fo.items()))
            + f" | {len(ser)} series {[round(x.get('duration_s', 0), 1) for x in ser]} s, outputs {oc.get('status')}, "
            f"JIT = AOT {oc.get('forms_bit_equal_every_round')}, max|dp| {num(oc.get('max_abs_dp_vs_oracle'), '%.2e')}"
            + (f" | stopped: {it['stopped']}" if it.get("stopped") else "") + (f" | ERROR {it['error']}"[:200] if "error" in it else ""))
S = r.get("stages", {})
for k in r.get("stage_order", []):
    s = S.get(k, {})
    tag = f"{k}: {'SKIPPED' if s.get('skipped') else ('PASS' if s.get('pass') else 'FAIL')}" + (" (partial)" if s.get("partial") else "")
    err = f" | ERROR {s['error']}"[:300] if "error" in s else ""
    mem = s.get("memory", {})
    memt = f" | peak footprint {num(mem.get('peak_footprint_mb'), '%.0f')} MB, least available {num(mem.get('min_available_mb'), '%.0f')} MB"
    if k == "assets":
        print(f"{tag} | {s.get('md5sums_listed')} files ({num((s.get('listed_bytes') or 0) / 1e6, '%.0f')} MB), missing "
              f"{len(s.get('missing', []))}, md5 checked {s.get('md5_checked')} (different {len(s.get('md5_mismatch', []))}), "
              f"{len(s.get('md5_deferred', []))} model files for md5, free {num(s.get('free_gb'), '%.1f')} GB{err}")
    elif k == "md5":
        print(f"{tag} | {len(s.get('files', []))} files, {num(s.get('bytes', 0) / 1e6, '%.0f')} MB in {num(s.get('seconds'), '%.1f')} s, "
              f"different {len(s.get('md5_mismatch', []))}{err}")
    elif k.startswith("parity_"):
        parity_lines(tag, s, memt, err)
    elif k == "bench":
        bench_lines(tag, s, memt, err)
    elif k == "long":
        print(f"{tag}{memt}{err}" + (f" | running {s.get('running')}" if s.get("running") else ""))
        for sub in ("parity_jit", "parity_aot"):
            if s.get(sub): parity_lines(f"  long {sub}: {'PASS' if s[sub].get('pass') else 'FAIL'}", s[sub], "", "")
        if s.get("bench"): bench_lines(f"  long bench: {'PASS' if s['bench'].get('pass') else 'FAIL'}", s["bench"], "", "")
    elif k == "ane":
        print(f"{tag}{memt}{err}")
        for part in ("audio", "decision"):
            a = s.get(part)
            if not a: continue
            if a.get("skipped"):
                print(f"   {part}: SKIPPED ({a.get('reason')})"); continue
            ld = a.get("load", {})
            print(f"   {part}: {a.get('ane_regions')} ANE regions, {num((a.get('asset_bytes') or 0) / 1e6, '%.0f')} MB, load wall "
                  f"{num(ld.get('wall_s'))} s (AIModel {num(ld.get('model_s'))}, main {num(ld.get('function_s'))})"
                  + (f" | ERROR {a['error']}"[:200] if "error" in a else ""))
            if a.get("summary"): print(f"      rows: {bar(a['summary'])}")
            if part == "audio":
                for c in a.get("clips", []):
                    print(f"      {c.get('id')}: call {num(c.get('call_ms'), '%.1f')} ms, prefix vs GPU max|d| {num(c.get('prefix_max_abs_d_gpu'), '%.2e')}")
                benches = {"W5": a.get("w5")} if a.get("w5") else {}
            else:
                vs_gpu = ("not measured (no GPU parity rows in that launch)" if a.get("rows_compared_gpu") == 0
                          else f"max|dp| {num(a.get('rows_max_abs_dp_gpu'), '%.3e')}")
                print(f"      drift {num(a.get('max_drift_marker_logits'), '%.3e')}, vs GPU rows {vs_gpu}")
                benches = a.get("bench") or {}
            for w, b in benches.items():
                if not b or b.get("skipped"): continue
                fo = b.get("forms", {})
                print(f"      {w}: " + ", ".join(f"{n} median {num(v.get('median_ms'))} ms (media {num(v.get('parts_median_ms', {}).get('media_ms'))}, "
                      f"n {v.get('n')}, thermal {v.get('thermal_of_timed')})" for n, v in fo.items()))
    else:
        print(f"{tag}{err}")
print("summary:", " ".join(r.get("summary", [])))
PY
}

if [ "${1:-}" = "--summary" ]; then summary "${2:?usage: _run.sh --summary <result.json> [run id]}" ${3:-}; exit 0; fi
UDID=${1:?usage: _run.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
RUN_ID=${D1_RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT=$W/runs/$RUN_ID
mkdir -p $OUT
CAP=${D1_CAP:-150}
FREEZE=${D1_FREEZE_S:-120}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { say "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^d1-omni device gate"; }; then
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
  --subdirectory "Library/Application Support/D1OmniAssets" --no-recurse 2>&1)
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

pull() {  # pull <name>: Documents/d1omni_gate/<name> -> $OUT/<name> (kept when the pull fails)
  rm -f $OUT/$1.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source $RDIR/$1 --destination $OUT/$1.pull >/dev/null 2>&1 && [ -f $OUT/$1.pull ] && mv $OUT/$1.pull $OUT/$1
}
app_pid() {  # the D1OmniGate process on the phone: its pid, or nothing (from the JSON listing)
  local J=$OUT/processes.json
  rm -f $J
  xcrun devicectl device info processes --device $UDID --json-output $J >/dev/null 2>&1 || return 0
  /usr/bin/python3 -I - $J <<'PY'
import json, sys
def walk(v):
    if isinstance(v, dict):
        if any(isinstance(x, str) and "D1OmniGate.app/D1OmniGate" in x for x in v.values()):
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
    state=$(/usr/bin/python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("status") if r.get("run_id")==sys.argv[2] else "other-run")' \
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
    if echo "$P" | grep -q "D1OmniGate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? jetsam? see result.log)"; break; }
    # alive but writing nothing: a frozen app (the memory monitor writes every 100 ms while it lives)
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
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(D1OmniGate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names D1OmniGate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-260; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
