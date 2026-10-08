#!/bin/zsh
# The device gate in one command, holding the phone for one window of at most D1_FRAME_S seconds (default 1800 = 30 min)
# from the hold's take to its release: take the device hold (~/code/coreai/ondevice/.device_hold), install the app and the
# staged assets (md5-checked), run the gate with a deadline that ends the window in time, release the hold (on any exit).
# Copied from apps/KevGate/_gate.sh (zoo d1-3b 4955a23) with the lane's names; the hold's take and the window budget are
# new.
# The hold file is shared by two conventions (lane memory reference_iphone_device_hold_file): the gate apps' text line
# ("d1-3b device gate since <time>, session <name>"), and a JSON form {"pid": <keeper pid>, "started": ...} whose keeper
# runs while its lane uses the phone. The take: every 5 s, the file must be absent — and the last JSON holder's keeper pid
# gone (`kill -0`) — for 60 s in a row; then this lane's line is written with noclobber. A JSON hold is never removed
# while its pid is alive; it counts as stale (removed, with a line in the hold log) only when the pid is gone AND it was
# started more than 10 h ago. Any other hold is waited for (a line every 60 s), never removed; after D1_HOLD_CAP_MIN
# minutes (default 120) the script stops as "contended" (exit 2).
# The window: after the install, D1_DEADLINE_S (the app stops its loops in time) and D1_CAP (the poll cap) are set from
# what is left of D1_FRAME_S minus D1_FRAME_TAIL s (default 150: the last pulls, the crash listing, the release); a
# smaller D1_DEADLINE_S given in the JSON members wins. Less than D1_FRAME_MIN_RUN s (default 240) left: no launch.
# Build and staging come first and happen on the Mac only: ./_build.sh && ./_stage.sh
#   D1_SESSION=<session name> D1_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [extra env JSON members]
#   D1_SKIP_INSTALL=1 ./_gate.sh <udid> ...      run again on what is already installed
#   D1_INSTALL_ONLY=1 ./_gate.sh <udid>          (with D1_STAGE_DIR / D1_PUSH_ONLY / D1_SKIP_APP: ./_install.sh's
#                                                push of another stage directory) under the hold, no launch
#   D1_KEEP_UNTIL=<file> ./_gate.sh <udid>       the keeper: take the hold as above and keep it (no install, no launch)
#                                                until <file> exists or D1_KEEP_MAX_S s pass, then release it
#   D1_HOLD_KEPT=1 ./_gate.sh <udid> ...         one slot inside the keeper's hold (D1_FRAME_S from the slot's start):
#                                                the hold must already be this lane's; it is neither taken nor released
# Exit codes as ./_run.sh (0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = busy / held / refused).
# Hold log: _work/device_runs/hold.log (every take, wait and release, with the time).
set -u
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
UDID=${1:?usage: _gate.sh <udid> [extra env JSON members]}
EXTRA=${2:-}
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
MARK="d1-3b device gate"
FRAME=${D1_FRAME_S:-1800}
mkdir -p $W/device_runs
HLOG=$W/device_runs/hold.log
hsay() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $HLOG; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { hsay "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  hsay "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi

holdline() { print -r -- "$MARK since $(date '+%Y-%m-%d %H:%M:%S %Z'), session ${D1_SESSION:-$USER@$(hostname -s)}"; }
ours() { [ -f $HOLD ] && head -1 $HOLD | grep -q "^$MARK"; }
take() { ( setopt noclobber; holdline > $HOLD ) 2>/dev/null; }
# the keeper pid of a JSON hold (empty for the text form)
json_pid() { /usr/bin/python3 -c 'import json,sys
try:
    h = json.load(open(sys.argv[1])); p = h.get("pid")
    print(p if isinstance(p, int) else "")
except Exception:
    print("")' $HOLD 2>/dev/null; }
# a JSON hold whose keeper pid is gone and which started more than 10 h ago: prints the reason and returns 0
stale_json() {
  [ -f $HOLD ] || return 1
  /usr/bin/python3 - $HOLD <<'PY'
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

if [ "${D1_HOLD_KEPT:-0}" = 1 ]; then
  # a slot under a hold the keeper (D1_KEEP_UNTIL) keeps: run inside it, never take or release it
  ours || { hsay "D1_HOLD_KEPT=1 but the hold is not this lane's ($(head -c 200 $HOLD 2>/dev/null || echo absent)): stopping"; exit 2; }
  HELD_AT=$SECONDS
  hsay "slot under the kept hold: $(head -1 $HOLD) (window ${FRAME} s from now)"
elif ours; then
  hsay "hold was already this lane's (left by an earlier run: $(head -1 $HOLD)); taking it over"
  holdline > $HOLD
else
  t0=$SECONDS; lastpid=""; clear_since=-1; lastsay=-60
  while true; do
    if [ -f $HOLD ]; then
      clear_since=-1
      p=$(json_pid); [ -n "$p" ] && lastpid=$p
      if why=$(stale_json); then
        hsay "stale hold removed: $why ($(head -c 200 $HOLD 2>/dev/null))"
        rm -f $HOLD
        continue
      fi
      if (( SECONDS - lastsay >= 60 )); then
        hsay "held by another session, waiting (cap ${D1_HOLD_CAP_MIN:-120} min): $(head -c 200 $HOLD 2>/dev/null)"
        lastsay=$SECONDS
      fi
    elif [ -n "$lastpid" ] && kill -0 $lastpid 2>/dev/null; then
      clear_since=-1
      if (( SECONDS - lastsay >= 60 )); then hsay "the hold file is gone but its keeper pid $lastpid still runs: waiting"; lastsay=$SECONDS; fi
    else
      (( clear_since < 0 )) && { clear_since=$SECONDS; hsay "the hold is free (no file$( [ -n "$lastpid" ] && echo ", keeper pid $lastpid gone")): taking it after 60 s free"; }
      if (( SECONDS - clear_since >= 60 )); then
        if take; then break; fi
        hsay "the noclobber take lost a race: $(head -c 200 $HOLD 2>/dev/null)"
        clear_since=-1
        continue
      fi
    fi
    if (( SECONDS - t0 >= ${D1_HOLD_CAP_MIN:-120} * 60 )); then
      hsay "contended: the phone is still not free after $(( (SECONDS - t0) / 60 )) min ($(head -c 200 $HOLD 2>/dev/null || echo 'no file')); stopping"
      exit 2
    fi
    sleep 5
  done
  hsay "hold taken after $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s of waiting"
fi
if [ "${D1_HOLD_KEPT:-0}" != 1 ]; then
  HELD_AT=$SECONDS
  hsay "hold taken: $(head -1 $HOLD) (window ${FRAME} s)"
  release() { if ours; then rm -f $HOLD; hsay "hold released (held $((SECONDS - HELD_AT)) s)"; else hsay "hold not ours at exit (left as is)"; fi }
  trap release EXIT
  trap 'exit 130' INT TERM HUP
fi
# the keeper: hold the phone across several slots (D1_HOLD_KEPT=1 runs), until the file D1_KEEP_UNTIL exists or
# D1_KEEP_MAX_S (default 10800) pass, then release; it stops early if the hold stops being this lane's
if [ -n "${D1_KEEP_UNTIL:-}" ]; then
  hsay "keeper pid $$: holding until $D1_KEEP_UNTIL exists (cap ${D1_KEEP_MAX_S:-10800} s)"
  until [ -e "$D1_KEEP_UNTIL" ] || (( SECONDS - HELD_AT >= ${D1_KEEP_MAX_S:-10800} )); do
    ours || { hsay "keeper: the hold is no longer this lane's ($(head -c 200 $HOLD 2>/dev/null || echo absent)): stopping"; exit 2; }
    sleep 5
  done
  hsay "keeper: $( [ -e "$D1_KEEP_UNTIL" ] && echo "release asked ($D1_KEEP_UNTIL)" || echo "cap reached")"
  exit 0
fi

rc=0
if [ "${D1_SKIP_INSTALL:-0}" != 1 ]; then
  $DIR/_install.sh $UDID
  rc=$?
  [ $rc -eq 0 ] || { hsay "install failed (exit $rc); not running the gate"; exit $rc; }
fi
if [ "${D1_INSTALL_ONLY:-0}" = 1 ]; then hsay "install only (D1_INSTALL_ONLY=1): no launch"; exit 0; fi
LEFT=$(( FRAME - (SECONDS - HELD_AT) - ${D1_FRAME_TAIL:-150} ))
if (( LEFT < ${D1_FRAME_MIN_RUN:-240} )); then
  hsay "only $LEFT s left of the ${FRAME} s window after the install: not launching"; exit 2
fi
DL=$(( LEFT - 60 ))
GIVEN=$(echo "$EXTRA" | sed -nE 's/.*"D1_DEADLINE_S" *: *"?([0-9]+)"?.*/\1/p')
if [ -n "$GIVEN" ]; then
  (( GIVEN < DL )) && DL=$GIVEN
  EXTRA=$(echo "$EXTRA" | sed -E 's/"D1_DEADLINE_S" *: *"?[0-9]+"? *,? *//; s/, *$//')
fi
EXTRA="\"D1_DEADLINE_S\":\"$DL\"${EXTRA:+,$EXTRA}"
hsay "launching with $LEFT s of the window left: D1_DEADLINE_S $DL, poll cap $(( (LEFT + 9) / 10 )) polls"
D1_CAP=$(( (LEFT + 9) / 10 )) $DIR/_run.sh $UDID "$EXTRA"
rc=$?
hsay "gate finished (exit $rc)"
exit $rc
