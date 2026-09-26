#!/bin/zsh
# The device gate in one command, holding the phone for its whole length: take the device hold
# (~/code/coreai/ondevice/.device_hold: one line "GLiNER2.5-Decide device gate since <time>, session <name>"; while
# another session holds it, wait in 60 s steps, and after 60 min stop as "contended"), install the app and the
# staged assets (md5-checked), run the gate, release the hold (on any exit).
# Build and staging come first and happen on the Mac only: ./_build.sh && ./_stage.sh
#   DECIDE_SESSION=<session name> DECIDE_ALLOWED_DEVICES=<udid>,<coredevice id> ./_gate.sh <udid> [extra env JSON members]
#   DECIDE_SKIP_INSTALL=1 ./_gate.sh <udid> ...     run again on what is already installed
# Exit codes as ./_run.sh (0 = every stage passed, 3 = a stage failed, 1 = no result, 2 = busy / held / refused).
# Hold log: _work/device_runs/hold.log (every take, wait and release, with the time).
set -u
DIR=${0:A:h}
W=${DECIDE_WORK:-${DIR:h:h}/conversion/gliner25_decide/_work}
UDID=${1:?usage: _gate.sh <udid> [extra env JSON members]}
HOLD=${DECIDE_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
MARK="GLiNER2.5-Decide device gate"
mkdir -p $W/device_runs
HLOG=$W/device_runs/hold.log
hsay() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $HLOG; }

if [ -n "${DECIDE_ALLOWED_DEVICES:-}" ] && [[ ",$DECIDE_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  hsay "refusing device $UDID: not in DECIDE_ALLOWED_DEVICES ($DECIDE_ALLOWED_DEVICES)"; exit 2
fi
[ -n "${DECIDE_ALLOWED_DEVICES:-}" ] || hsay "note: DECIDE_ALLOWED_DEVICES is not set; nothing checks which phone $UDID is"

holdline() { print -r -- "$MARK since $(date '+%Y-%m-%d %H:%M:%S %Z'), session ${DECIDE_SESSION:-$USER@$(hostname -s)}"; }
ours() { [ -f $HOLD ] && head -1 $HOLD | grep -q "^$MARK"; }
take() { ( setopt noclobber; holdline > $HOLD ) 2>/dev/null; }
if ours; then
  hsay "hold was already this lane's (left by an earlier run: $(head -1 $HOLD)); taking it over"
  holdline > $HOLD
  hsay "hold taken: $(head -1 $HOLD)"
else
  waited=0
  until take; do
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
if [ "${DECIDE_SKIP_INSTALL:-0}" != 1 ]; then
  $DIR/_install.sh $UDID
  rc=$?
  [ $rc -eq 0 ] || { hsay "install failed (exit $rc); not running the gate"; exit $rc; }
fi
$DIR/_run.sh "$@"
rc=$?
hsay "gate finished (exit $rc)"
exit $rc
