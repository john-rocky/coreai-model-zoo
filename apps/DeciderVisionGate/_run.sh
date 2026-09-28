#!/bin/zsh
# Launch DeciderVisionGate on the phone and collect its result. No --console (after an install a console launch can be
# refused as Busy for minutes); the app writes Documents/dv_gate/result.log line by line, memory.tsv reading by reading and
# result.json after every stage (and every 5 runs), and this script pulls the three every 10 s until result.json carries
# this run's id with status done / failed, the app's process is gone, or the cap passes. Then it pulls dump/ (the slot
# logits and tower outputs of every run) and prints the summary.
#   ./_run.sh <udid> [extra env as JSON members]     e.g. ./_run.sh <udid> '"DV_STAGES":"assets,load1"'
#   ./_run.sh --summary <result.json>                print the summary of a pulled (or a Mac) result again
# Normally from ./_gate.sh, which holds the phone; refuses without this lane's hold, and on a device DV_ALLOWED_DEVICES
# does not list.
# App knobs (GateRunner.swift): DV_STAGES, DV_LIMIT, DV_WAIT_NOMINAL, DV_BENCH_RUNS, DV_BENCH_REST, DV_BENCH_G256,
# DV_BENCH_G448, DV_BENCH_TEXT, DV_WARMUP, DV_DECODER, DV_TOWER_G256, DV_TOWER_G448, DV_DECODER_AOT, DV_LOAD_SPLIT, DV_DUMP.
# Poll cap: DV_CAP polls of 10 s (default 270 = 45 min). When the phone stops answering (unplugged, locked up), the
# script says so and keeps polling until the cap: reconnecting is a person's job.
# A run that does not end "done" (the app gone, or the cap) also lists the phone's crash logs and copies the ones of
# today that name DeciderVisionGate or a jetsam event into crash/.
# Output: _work/device_runs/<run id>/{result.json,result.log,memory.tsv,run.out,launch.log[,dump/][,crash/]}
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.decidervisiongate
DIR=${0:A:h}
W=${DV_WORK:-$DIR/_work}
RDIR=Documents/dv_gate

summary() {  # summary <result.json> [run id]
  /usr/bin/python3 - "$@" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if len(sys.argv) > 2 and r.get("run_id") != sys.argv[2]:
    print(f"result.json is from run {r.get('run_id')}, not {sys.argv[2]}")
def num(v, f="%.2f"):
    return (f % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-"
d = r.get("device", {})
print(f"run {r.get('run_id')}: status {r.get('status')}, pass {r.get('pass')}, {num(r.get('elapsed_s'), '%.0f')} s | "
      f"{d.get('machine')} {d.get('hw_model')} {d.get('os')} (build {d.get('os_build')}, Core AI arch {d.get('coreai_architecture')}), "
      f"thermal {d.get('thermal')} -> {r.get('device_end', {}).get('thermal')}, battery {num(d.get('battery_level', -1) * 100, '%.0f')} % "
      f"{d.get('battery_state')} ({d.get('power_source')}), low power {d.get('low_power_mode')}, free {num(d.get('free_gb'), '%.1f')} GB")
if r.get("fatal"): print("FATAL", r["fatal"])
S = r.get("stages", {})
def load(tag, rec):
    if not rec: return ""
    dec = rec.get("decoder_s", {})
    tw = " ".join(f"tower {k} {num(v)} s" for k, v in sorted(rec.get("tower_s", {}).items()))
    m = rec.get("memory", {})
    return (f"{tag} {num(rec.get('wall_s'))} s (decoder {num(dec.get('total'))} s{', ' + tw if tw else ''}; peak "
            f"{num(m.get('peak_footprint_mb'), '%.0f')} MB, least avail {num(m.get('min_available_mb'), '%.0f')} MB)")
for k in r.get("stage_order", []):
    s = S.get(k, {})
    tag = f"{k}: {'PASS' if s.get('pass') else 'FAIL'}" + (" (partial)" if s.get("partial") else "")
    err = f" | ERROR {s.get('error_step', '')} {s['error']}"[:300] if "error" in s else ""
    if k == "assets":
        print(f"{tag} | {s.get('md5sums_listed')} files, missing {len(s.get('missing', []))}, md5 checked {s.get('md5_checked')} "
              f"(different {len(s.get('md5_mismatch', []))}), {len(s.get('md5_deferred', []))} model files for the md5 stage, "
              f"free {num(s.get('free_gb'), '%.1f')} GB{err}")
    elif k == "load1":
        ta = s.get("tower_g256_alone", {})
        parts = []
        if ta:
            parts.append(f"tower g256 alone {num(ta.get('wall_s'))} s (peak {num(ta.get('memory', {}).get('peak_footprint_mb'), '%.0f')} MB)")
        parts.append(load("decoder alone", s.get("decoder_alone")))
        parts.append(load("decider g256", s.get("decider_g256")))
        print(f"{tag} | " + "; ".join(p for p in parts if p) + f" | cache MB {num(s.get('cache_bytes_after_decider', 0) / 1e6, '%.0f')}{err}")
    elif k in ("load_g448", "load2"):
        print(f"{tag} | {load('decider ' + str(s.get('grid')), s.get('decider'))}"
              + (f" | check bit-equal {s.get('check_bit_equal_e2e')}" if 'check_bit_equal_e2e' in s else "") + err)
    elif k == "warmup":
        run = s.get("run", {})
        print(f"{tag} | {s.get('row')}/g256 wall {num(run.get('wall_from_file_s', 0) * 1e3, '%.1f')} ms{err}")
    elif k.startswith("e2e_"):
        m = s.get("summary", {})
        mac = m.get("mac", {})
        w = m.get("wall_ms", {})
        print(f"{tag} | {m.get('runs')} runs {m.get('slots')} slots, argmax {m.get('argmax_equal')}, full-vocab {m.get('full_vocab_top1_is_oracle_letter')}, "
              f"max|dp| {num(m.get('max_abs_dp'), '%.6f')}, mean {num(m.get('mean_of_run_mean_abs_dp'), '%.6f')} | Mac: argmax "
              f"{mac.get('argmax_equal')}/{mac.get('slots_compared')}, |dp| max {num(mac.get('max_abs_dp'), '%.6f')}, letter bits "
              f"{mac.get('letter_logits_bit_equal')}, tower = {mac.get('tower_embeds_equal_runs')}/{mac.get('tower_runs')} | wall ms median "
              f"{num(w.get('median'), '%.1f')}{err}")
    elif k.startswith("reset_"):
        print(f"{tag} | {s.get('run')} bit-equal {s.get('bit_equal')}{err}")
    elif k.startswith("bench_"):
        for key, row in s.get("rows", {}).items():
            m = row.get("summary", {})
            wn = row.get("wait_nominal", {})
            b0, b1 = row.get("battery_start", {}), row.get("battery_end", {})
            print(f"{tag} | {key}: wall median {num(m.get('wall_ms_median'), '%.1f')} ms (min {num(m.get('wall_ms_min'), '%.1f')}, max "
                  f"{num(m.get('wall_ms_max'), '%.1f')}), tower {num(m.get('tower_ms_median'), '%.1f')} ms, decoder "
                  f"{num(m.get('decoder_ms_median'), '%.1f')} ms, end {num(m.get('timed_runs_end_s'), '%.1f')} s | thermal "
                  f"{row.get('thermal_start')} -> {row.get('thermal_end')} (waited {num(wn.get('waited_s'), '%.0f')} s), battery "
                  f"{num(b0.get('level', -1) * 100, '%.0f')} -> {num(b1.get('level', -1) * 100, '%.0f')} % {b1.get('power')}"
                  + (f" | ERROR {row['error']}" if 'error' in row else ""))
    elif k == "md5":
        print(f"{tag} | {len(s.get('files', []))} model files, {num(s.get('bytes', 0) / 1e6, '%.0f')} MB, different "
              f"{len(s.get('md5_mismatch', []))}{err}")
    else:
        print(f"{tag}{err}")
e = r.get("e2e_summary", {})
if e.get("runs"):
    mac = e.get("mac", {})
    print(f"all arms: {e.get('runs')} runs {e.get('slots')} slots, argmax {e.get('argmax_equal')}, full-vocab "
          f"{e.get('full_vocab_top1_is_oracle_letter')}, max|dp| {num(e.get('max_abs_dp'), '%.6f')}, mean {num(e.get('mean_of_run_mean_abs_dp'), '%.6f')} "
          f"(alt {num(e.get('mean_of_run_slot_mean_abs_dp'), '%.6f')}), bar {'PASS' if e.get('bar_pass') else 'FAIL'}, reset {e.get('reset_bit_equal')} | "
          f"Mac: argmax {mac.get('argmax_equal')}/{mac.get('slots_compared')}, |dp| max {num(mac.get('max_abs_dp'), '%.6f')}")
print("summary:", " ".join(r.get("summary", [])))
PY
}

if [ "${1:-}" = "--summary" ]; then summary "${2:?usage: _run.sh --summary <result.json>}"; exit 0; fi
UDID=${1:?usage: _run.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT=$W/device_runs/$RUN_ID
mkdir -p $OUT
CAP=${DV_CAP:-270}
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

[ -n "${DV_ALLOWED_DEVICES:-}" ] || { say "refusing: DV_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$DV_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in DV_ALLOWED_DEVICES ($DV_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${DV_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^decider-2b-vision device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
say "hold: $(head -1 $HOLD)"
IDS=($UDID ${(s:,:)${DV_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

# the phone answers for this app's container (the list State alone flips while the phone is fine)
L=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
  --subdirectory "Library/Application Support/DeciderVisionAssets" 2>&1)
if echo "$L" | grep -q ERROR; then
  say "container not reachable (installed? phone unlocked, on USB?): $(echo "$L" | grep -m1 ERROR | cut -c1-160)"; exit 2
fi

ENVJ="{\"DV_RUN_ID\":\"$RUN_ID\"${EXTRA:+,$EXTRA}}"
launched=0
for t in $(seq 1 12); do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing --environment-variables "$ENVJ" $BID 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-160)"; sleep 15
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched run $RUN_ID (env $ENVJ); polling every 10 s, cap $((CAP * 10)) s"

pull() {  # pull <name>: Documents/dv_gate/<name> -> $OUT/<name> (kept when the pull fails)
  rm -f $OUT/$1.pull
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source $RDIR/$1 --destination $OUT/$1.pull >/dev/null 2>&1 && [ -f $OUT/$1.pull ] && mv $OUT/$1.pull $OUT/$1
}
state=""; last=""; gone=0; silent=0
for i in $(seq 1 $CAP); do
  sleep 10
  pull result.log; pull result.json; pull memory.tsv
  if [ -f $OUT/result.log ]; then
    cur=$(tail -1 $OUT/result.log)
    [ "$cur" != "$last" ] && { echo "  ${cur[1,240]}"; last=$cur; }
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
    if echo "$P" | grep -q "DeciderVisionGate"; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -ge 2 ] && { say "the app's process is gone and result.json says '${state:-none}' (crash? see result.log)"; break; }
  fi
done
say "state: ${state:-no result.json} after $((i * 10)) s"
if [[ $state == done ]]; then
  # the per-run dumps (slot logits, tower outputs): one directory pull
  rm -rf $OUT/dump.pull
  if xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
      --source $RDIR/dump --destination $OUT/dump.pull >> $OUT/run.out 2>&1; then
    D=$(find $OUT/dump.pull -type d -name logits 2>/dev/null | head -1)
    if [ -n "$D" ]; then mv ${D:h} $OUT/dump; rm -rf $OUT/dump.pull; fi
    say "dump: $(find $OUT/dump -type f 2>/dev/null | wc -l | tr -d ' ') files, $(du -sh $OUT/dump 2>/dev/null | cut -f1)"
  else
    say "dump pull failed (see run.out)"
  fi
else
  # crash reports: the listing, then today's files that name the app or a jetsam event
  CL=$(xcrun devicectl device info files --device $UDID --domain-type systemCrashLogs 2>&1)
  echo "$CL" > $OUT/crashlogs_listing.txt
  names=(${(f)"$(echo "$CL" | grep -oE "[A-Za-z0-9._+-]*(DeciderVisionGate|JetsamEvent)[A-Za-z0-9._+-]*$(date +%Y-%m-%d)[A-Za-z0-9._+-]*" | sort -u)"})
  if (( ${#names} )); then
    mkdir -p $OUT/crash
    for n in $names; do
      xcrun devicectl device copy from --device $UDID --domain-type systemCrashLogs --source "$n" --destination "$OUT/crash/$n" \
        >> $OUT/crash/copy.log 2>&1 && say "crash log: $OUT/crash/$n" || say "crash log $n: copy failed (see crash/copy.log)"
    done
  else
    say "no crash log of today names DeciderVisionGate or a jetsam event (listing: crashlogs_listing.txt)"
  fi
fi
[ -f $OUT/result.log ] && { echo "--- last lines of result.log"; tail -8 $OUT/result.log | cut -c1-240; }
[ -f $OUT/result.json ] && { echo "--- summary"; summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out; }
echo "files: $OUT"
[[ $state == done ]] || exit 1
/usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 3)' $OUT/result.json
