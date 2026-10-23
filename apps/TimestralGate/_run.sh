#!/bin/zsh
# Launch TimestralGate on the phone with a tag and the app's arguments, follow Documents/out/<tag>/log.txt until its
# last line is `DONE …` or `ERROR: …`, then pull the run's files to $TIMESTRAL_PULL/<tag>/ (default _work/device_runs).
#   ./_run.sh <udid> <tag> <app arguments…>          e.g. ./_run.sh <udid> g1 --mode gate --bundle <path> --compute gpu
# No --console (a console launch can be refused as Busy for minutes): the app writes its log line by line and this
# script pulls it every POLL_S (5) s. A log that stops growing for STALL_S (180) s ends the follow with exit 3 and the
# app is left running for the caller to stop (`devicectl device process terminate`); the run is not retried here.
# Pulled: log.txt, run.json, and every pass<k>/quantiles.f32.bin run.json names. Exit 0 = DONE, 1 = ERROR or no run,
# 2 = refused (hold, device), 3 = stalled.
# Runs only while this lane holds the phone and only on a device in TIMESTRAL_ALLOWED_DEVICES (see _install.sh).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _run.sh <udid> <tag> <app arguments…>}
TAG=${2:?usage: _run.sh <udid> <tag> <app arguments…>}
shift 2
BID=com.coreai.timestralgate
DIR=${0:A:h}
W=${TIMESTRAL_WORK:-$DIR/_work}
PULL=${TIMESTRAL_PULL:-$W/device_runs}
ALLOWED=${TIMESTRAL_ALLOWED_DEVICES:-00008160-000038CA02C00036}
HOLD=${TIMESTRAL_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
SCRIPT=${TIMESTRAL_HOLD_SCRIPT:-timestral_c_r2}
POLL_S=${POLL_S:-5}
STALL_S=${STALL_S:-180}
CAP_S=${CAP_S:-900}
OUT=$PULL/$TAG
mkdir -p $OUT
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }

[[ ",$ALLOWED," == *",$UDID,"* ]] || { say "refusing device $UDID: not in TIMESTRAL_ALLOWED_DEVICES ($ALLOWED)"; exit 2; }
held() { /usr/bin/python3 -c 'import json,os,sys
try:
    h = json.load(open(sys.argv[1])); os.kill(int(h["pid"]), 0)
    sys.exit(0 if h.get("script") == sys.argv[2] else 1)
except Exception:
    sys.exit(1)' $HOLD $SCRIPT 2>/dev/null; }
held || { say "the phone is not held by $SCRIPT ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent))"; exit 2; }

pullf() {  # pullf <container path> <local file>: 0 when the file arrived
  rm -f "$2.part"
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2.part" >/dev/null 2>&1 && [ -f "$2.part" ] && mv "$2.part" "$2"
}

launched=0
for t in 1 2 3 4 5 6; do
  LO=$(xcrun devicectl device process launch --device $UDID --terminate-existing $BID -- --tag $TAG "$@" 2>&1)
  echo "$LO" >> $OUT/launch.log
  if echo "$LO" | grep -q "Launched application"; then launched=1; break; fi
  say "launch retry $t: $(echo "$LO" | grep -m1 -iE 'error|busy' | cut -c1-160)"; sleep 10
done
[ $launched = 1 ] || { say "ERROR launch never accepted (see $OUT/launch.log)"; exit 1; }
say "launched $TAG: $*"

last=""; changed=$SECONDS; t0=$SECONDS; state=""
while (( SECONDS - t0 < CAP_S )); do
  sleep $POLL_S
  if pullf "Documents/out/$TAG/log.txt" $OUT/log.txt; then
    cur=$(tail -1 $OUT/log.txt)
    if [ "$cur" != "$last" ]; then echo "  ${cur[1,240]}"; last=$cur; changed=$SECONDS; fi
    case $cur in
      *"| DONE "*) state=done; break ;;
      *"| ERROR: "*) state=error; break ;;
    esac
  fi
  if (( SECONDS - changed >= STALL_S )); then state=stalled; break; fi
done
[ -z "$state" ] && state=cap
pullf "Documents/out/$TAG/run.json" $OUT/run.json || say "run.json did not come"
if [ -f $OUT/run.json ]; then
  for f in $(/usr/bin/python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
print(" ".join(p["file"] for p in (r.get("output") or {}).get("passes", [])))' $OUT/run.json); do
    mkdir -p $OUT/${f:h}
    pullf "Documents/out/$TAG/$f" $OUT/$f || say "$f did not come"
  done
fi
say "state $state after $((SECONDS - t0)) s; files in $OUT"
tail -3 $OUT/log.txt 2>/dev/null | cut -c1-300
case $state in
  done) exit 0 ;;
  stalled) say "STALL: no new log line for $STALL_S s (the app is still running; stop it before the next launch)"; exit 3 ;;
  *) exit 1 ;;
esac
