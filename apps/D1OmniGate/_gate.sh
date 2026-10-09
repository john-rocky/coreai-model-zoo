#!/bin/zsh
# The device gate in one command, holding the phone for one window of at most D1_FRAME_S seconds (default 1800 = 30 min)
# from the hold's take to its release: take the device hold (~/code/coreai/ondevice/.device_hold), record the phone,
# install the app and the staged assets (sizes checked; the app md5-checks the bytes), run the gate with a deadline that
# ends the window in time, launch once more for the stages a killed launch left, release the hold (on any exit).
# Adapted from apps/KevGate/_gate.sh with the d1-3b lane's window budget (apps/D1Gate/_gate.sh, 2026-10-08).
# The hold file is shared by two conventions (lane memory reference_iphone_device_hold_file): the gate apps' text line
# ("d1-omni device gate since <time>, session <name>"), and a JSON form {"pid": <keeper pid>, "started": ...} whose keeper
# runs while its lane uses the phone. The take: every 5 s, the file must be absent, the last JSON holder's keeper pid
# gone (`kill -0`), and no other process may name the phone (its UDID or CoreDevice id: devicectl, xcodebuild, another
# lane's script — this script and its parents excluded, and a Claude session whose prompt quotes the id is not a device
# job) — all three for 60 s in a row; then this lane's line is written with noclobber. The processes and the holder
# last seen go to the hold log. A JSON hold is never removed while its pid is alive; it counts as stale (removed, with a
# line in the hold log) only when the pid is gone AND it was started more than 10 h ago. Any other hold is waited for,
# never removed; after D1_HOLD_CAP_MIN minutes (default 60) the script stops as "contended" (exit 2).
# The window: after the install, D1_DEADLINE_S (the app stops its loops in time) and D1_CAP (the poll cap) are set from
# what is left of D1_FRAME_S minus D1_FRAME_TAIL s (default 150: the last pulls, the crash listing, the release). A
# launch that does not end "done" (the app killed: a jetsam event in the L4096 stage, say) is followed by one more launch
# with the stages it did not finish when D1_FRAME_MIN_RUN s (default 180) are left; the app skips the stage the killed
# launch died in.
# Build and staging come first and happen on the Mac only: ./_build.sh && ./_stage.sh
#   D1_SESSION=<session name> D1_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [extra env JSON members]
#   D1_SKIP_INSTALL=1 ./_gate.sh <udid> ...      run again on what is already installed
# Exit codes as ./_run.sh (0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = busy / held / refused).
# Hold log: $D1_WORK/hold.log (every take, wait and release, with the time).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${D1_WORK:-$HOME/code/coreai/_d1_omni/device}
UDID=${1:?usage: _gate.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
MARK="d1-omni device gate"
FRAME=${D1_FRAME_S:-1800}
STAGES=${D1_STAGES:-assets,parity_jit,parity_aot,bench,long,ane,md5}
mkdir -p $W/runs
HLOG=$W/hold.log
hsay() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $HLOG; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { hsay "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  hsay "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi
IDS=($UDID ${(s:,:)${D1_ALLOWED_DEVICES:-}})

holdline() { print -r -- "$MARK since $(date '+%Y-%m-%d %H:%M:%S %Z'), session ${D1_SESSION:-$USER@$(hostname -s)}"; }
ours() { [ -f $HOLD ] && head -1 $HOLD | grep -q "^$MARK"; }
take() { ( setopt noclobber; holdline > $HOLD ) 2>/dev/null; }
json_pid() { /usr/bin/python3 -I -c 'import json,sys
try:
    h = json.load(open(sys.argv[1])); p = h.get("pid")
    print(p if isinstance(p, int) else "")
except Exception:
    print("")' $HOLD 2>/dev/null; }
# a JSON hold whose keeper pid is gone and which started more than 10 h ago: prints the reason and returns 0
stale_json() {
  [ -f $HOLD ] || return 1
  /usr/bin/python3 -I - $HOLD <<'PY'
import json, os, sys, time
from datetime import datetime
try:
    h = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)                                     # not the JSON form: never stale here
pid, started = h.get("pid"), h.get("started")
if not isinstance(pid, int):
    sys.exit(1)
try:
    os.kill(pid, 0)
    sys.exit(1)                                     # the keeper is alive: held
except ProcessLookupError:
    pass
except PermissionError:
    sys.exit(1)                                     # alive under another user
age = None
if isinstance(started, (int, float)):
    age = time.time() - started
elif isinstance(started, str):
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            age = time.time() - datetime.strptime(started, fmt).timestamp()
            break
        except ValueError:
            continue
if age is None or age < 36000:
    sys.exit(1)                                     # too young (or unreadable): wait
print(f"JSON hold pid {pid} gone, started {started} ({age / 3600:.1f} h ago)")
sys.exit(0)
PY
}
# this script and every process above it (their command lines carry the UDID too)
MINE=($$)
p=$$
while :; do
  p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  [ -z "$p" ] || [ "$p" -le 1 ] && break
  MINE+=($p)
done
# other processes that name the phone: "<pid> <command>" lines (Claude sessions and grep excluded; so are this script, its
# parents, and its own children — a command substitution forks a copy of this script with the same command line)
foreign() {
  ps -axo pid=,ppid=,command= | grep -E -- "(${(j:|:)IDS})" | grep -v -E "grep|/claude( |$)|\.local/bin/claude|node .*claude" \
    | while read -r pid ppid cmd; do
        (( ${MINE[(Ie)$pid]} || ${MINE[(Ie)$ppid]} )) && continue
        echo "$pid ${cmd[1,160]}"
      done
}

if ours; then
  hsay "hold was already this lane's (left by an earlier run: $(head -1 $HOLD)); taking it over"
  holdline > $HOLD
else
  t0=$SECONDS; lastpid=""; clear_since=-1; lastsay=-60; lastholder=""; lastforeign=""
  while true; do
    F=$(foreign)
    [ -n "$F" ] && lastforeign=$F
    if [ -f $HOLD ]; then
      clear_since=-1
      lastholder=$(head -c 200 $HOLD 2>/dev/null)
      p=$(json_pid); [ -n "$p" ] && lastpid=$p
      if why=$(stale_json); then
        hsay "stale hold removed: $why ($lastholder)"
        rm -f $HOLD
        continue
      fi
      if (( SECONDS - lastsay >= 60 )); then
        hsay "held by another session, waiting (cap ${D1_HOLD_CAP_MIN:-60} min): $lastholder"
        lastsay=$SECONDS
      fi
    elif [ -n "$lastpid" ] && kill -0 $lastpid 2>/dev/null; then
      clear_since=-1
      if (( SECONDS - lastsay >= 60 )); then hsay "the hold file is gone but its keeper pid $lastpid still runs: waiting"; lastsay=$SECONDS; fi
    elif [ -n "$F" ]; then
      clear_since=-1
      if (( SECONDS - lastsay >= 60 )); then
        hsay "no hold file, but other processes name the phone: waiting: $(echo "$F" | head -3 | tr '\n' ';')"
        lastsay=$SECONDS
      fi
    else
      (( clear_since < 0 )) && { clear_since=$SECONDS; hsay "the phone is free (no hold file, no other process names it$( [ -n "$lastpid" ] && echo ", keeper pid $lastpid gone")): taking it after 60 s free"; }
      if (( SECONDS - clear_since >= 60 )); then
        if take; then break; fi
        hsay "the noclobber take lost a race: $(head -c 200 $HOLD 2>/dev/null)"
        clear_since=-1
        continue
      fi
    fi
    if (( SECONDS - t0 >= ${D1_HOLD_CAP_MIN:-60} * 60 )); then
      hsay "contended: the phone is still not free after $(( (SECONDS - t0) / 60 )) min ($(head -c 200 $HOLD 2>/dev/null || echo 'no file')); stopping"
      exit 2
    fi
    sleep 5
  done
  hsay "hold taken after $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s of waiting; the holder last seen: "\
"${lastholder:-none seen by this script}; the processes last seen naming the phone: $(echo "${lastforeign:-none}" | head -3 | tr '\n' ';')"
fi
HELD_AT=$SECONDS
hsay "hold taken: $(head -1 $HOLD) (window ${FRAME} s)"
release() { if ours; then rm -f $HOLD; hsay "hold released (held $((SECONDS - HELD_AT)) s)"; else hsay "hold not ours at exit (left as is)"; fi }
trap release EXIT
trap 'exit 130' INT TERM HUP

# the phone as CoreDevice reports it (the OS build), once per window
D=$W/device_details_$(date +%Y%m%d-%H%M%S).json
xcrun devicectl device info details --device $UDID --json-output $D > /dev/null 2>&1 && hsay "device details: $D ($(/usr/bin/python3 -I -c '
import json,sys
r = json.load(open(sys.argv[1])).get("result", {})
dp, hp = r.get("deviceProperties", {}), r.get("hardwareProperties", {})
print(f"{hp.get(\"marketingName\")} {hp.get(\"productType\")}, {dp.get(\"osVersionNumber\")} build {dp.get(\"osBuildUpdate\")}")' $D 2>/dev/null))"

rc=0
if [ "${D1_SKIP_INSTALL:-0}" != 1 ]; then
  $DIR/_install.sh $UDID
  rc=$?
  [ $rc -eq 0 ] || { hsay "install failed (exit $rc); not running the gate"; exit $rc; }
fi
hsay "installed and pushed after $((SECONDS - HELD_AT)) s of the window"
launch() {  # launch <stages>: one launch with the deadline of what is left of the window
  local LEFT=$(( FRAME - (SECONDS - HELD_AT) - ${D1_FRAME_TAIL:-150} ))
  if (( LEFT < ${D1_FRAME_MIN_RUN:-180} )); then hsay "only $LEFT s left of the ${FRAME} s window: not launching ($1)"; return 2; fi
  local DL=$(( LEFT - 45 ))
  local E="\"D1_DEADLINE_S\":\"$DL\",\"D1_STAGES\":\"$1\"${EXTRA:+,$EXTRA}"
  hsay "launching ($1) with $LEFT s of the window left: D1_DEADLINE_S $DL, poll cap $(( (LEFT + 9) / 10 )) polls"
  D1_RUN_ID=${D1_RUN_PREFIX:-r11}-$(date +%H%M%S) D1_CAP=$(( (LEFT + 9) / 10 )) $DIR/_run.sh $UDID "$E"
}
launch $STAGES
rc=$?
if [ $rc -eq 1 ]; then
  # the launch did not end "done": the stages it did not finish, from its result.json (the app skips the one it died in)
  R=$(ls -td $W/runs/*/ 2>/dev/null | head -1)
  LEFTST=$(/usr/bin/python3 -I -c 'import json,sys
plan = sys.argv[2].split(",")
try:
    r = json.load(open(sys.argv[1]))
    st = r.get("stages", {})
    done = [k for k in r.get("stage_order", []) if not st.get(k, {}).get("partial")]
except Exception:
    done = []
print(",".join(s for s in plan if s not in done))' ${R}result.json $STAGES 2>/dev/null)
  hsay "the launch did not end done (run ${R:t}); stages left: ${LEFTST:-none}"
  if [ -n "$LEFTST" ]; then launch $LEFTST; rc=$?; fi
fi
hsay "gate finished (exit $rc) after $((SECONDS - HELD_AT)) s of the window"
exit $rc
