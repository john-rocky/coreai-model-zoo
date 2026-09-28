#!/bin/zsh
# The device gate in one command, holding the phone for its whole length: take the device hold
# (~/code/coreai/ondevice/.device_hold: one line "decider-2b-vision device gate since <time>, session <name>"; while
# another session holds it, wait in 60 s steps, and after 60 min stop as "contended"), install the app and the staged
# assets (md5-checked), run the gate, release the hold (on any exit).
# The hold file is shared by two conventions (lane memory reference_iphone_device_hold_file): the gate apps' text line,
# and a JSON form {"pid": <keeper pid>, "started": ...} whose keeper sleeps 10 h. A JSON hold is never removed while its
# pid is alive; it counts as stale (removed, with a line in the hold log) only when the pid is gone AND it was started
# more than 10 h ago. Any other hold is waited for, never removed.
# Build and staging come first and happen on the Mac only: ./_build.sh && ./_stage.sh
#   DV_SESSION=<session name> DV_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [extra env JSON members]
#   DV_FRESH=1 ./_gate.sh <udid> ...          uninstall first: a cold container (./_install.sh)
#   DV_SKIP_INSTALL=1 ./_gate.sh <udid> ...   run again on what is already installed
# Exit codes as ./_run.sh (0 = the gate passed, 3 = a stage failed, 1 = no result, 2 = busy / held / refused).
# Hold log: _work/device_runs/hold.log (every take, wait and release, with the time).
set -u
DIR=${0:A:h}
W=${DV_WORK:-$DIR/_work}
UDID=${1:?usage: _gate.sh <udid> [extra env JSON members]}
HOLD=${DV_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
MARK="decider-2b-vision device gate"
mkdir -p $W/device_runs
HLOG=$W/device_runs/hold.log
hsay() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $HLOG; }

[ -n "${DV_ALLOWED_DEVICES:-}" ] || { hsay "refusing: DV_ALLOWED_DEVICES is not set"; exit 2; }
if [[ ",$DV_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  hsay "refusing device $UDID: not in DV_ALLOWED_DEVICES ($DV_ALLOWED_DEVICES)"; exit 2
fi

holdline() { print -r -- "$MARK since $(date '+%Y-%m-%d %H:%M:%S %Z'), session ${DV_SESSION:-$USER@$(hostname -s)}"; }
ours() { [ -f $HOLD ] && head -1 $HOLD | grep -q "^$MARK"; }
take() { ( setopt noclobber; holdline > $HOLD ) 2>/dev/null; }
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
if ours; then
  hsay "hold was already this lane's (left by an earlier run: $(head -1 $HOLD)); taking it over"
  holdline > $HOLD
  hsay "hold taken: $(head -1 $HOLD)"
else
  waited=0
  until take; do
    if why=$(stale_json); then
      hsay "stale hold removed: $why ($(head -c 200 $HOLD 2>/dev/null))"
      rm -f $HOLD
      continue
    fi
    (( waited == 0 )) && hsay "held by another session, waiting in 60 s steps (cap 60 min): $(head -c 200 $HOLD 2>/dev/null)"
    if (( waited >= 60 )); then
      hsay "contended: the phone is still held after 60 min ($(head -c 200 $HOLD 2>/dev/null)); stopping"; exit 2
    fi
    sleep 60
    waited=$((waited + 1))
  done
  hsay "hold taken$( (( waited > 0 )) && echo " after $waited min"): $(head -1 $HOLD)"
fi
release() { if ours; then rm -f $HOLD; hsay "hold released"; else hsay "hold not ours at exit (left as is)"; fi }
trap release EXIT
trap 'exit 130' INT TERM HUP

rc=0
if [ "${DV_SKIP_INSTALL:-0}" != 1 ]; then
  $DIR/_install.sh $UDID
  rc=$?
  [ $rc -eq 0 ] || { hsay "install failed (exit $rc); not running the gate"; exit $rc; }
fi
$DIR/_run.sh "$@"
rc=$?
hsay "gate finished (exit $rc)"
exit $rc
