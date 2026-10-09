#!/bin/zsh
# The phone's side of a D1Demo run, one step per call (a recording script strings them together):
#   ./_run.sh launch <udid> <run id> [app args]   launch with --terminate-existing; the app's arguments go after "--"
#                                                 (devicectl reads `-log` as its own option otherwise); -runid and -log 1
#                                                 are added
#   ./_run.sh front <udid>                        launch again without --terminate-existing: brings the running app to
#                                                 the front (a terminating launch can leave the home screen showing)
#   ./_run.sh wait <udid> <run id> <mark> <cap s> <out dir>
#                                                 poll Documents/d1demo/autoplay.log every 2 s until this run's line with
#                                                 <mark> (exit 0) or FAILED (exit 3), the cap (exit 1); the log lands in
#                                                 <out dir>/autoplay.log
#   ./_run.sh trigger <udid> <name>               put an empty file Documents/<name> (the app waits for it)
#   ./_run.sh pull <udid> <run id> <out dir>      the run's JSON and the autoplay log
#   ./_run.sh shot <udid> <png>                   a screenshot of the phone (devicectl capture screenshot)
#   ./_run.sh stop <udid>                         terminate the app by its pid
# Every call refuses a device D1_ALLOWED_DEVICES does not list, and runs only while the hold is this lane's JSON
# (D1_HOLD_SCRIPT, as ./_install.sh). Never --remove-existing-content, never --console.
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BID=com.daisukemajima.d1demo
CMD=${1:?usage: _run.sh launch|front|wait|trigger|pull|shot|stop <udid> ...}
UDID=${2:?udid}
shift 2
[ -n "${D1_ALLOWED_DEVICES:-}" ] || { echo "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
[[ ",$D1_ALLOWED_DEVICES," == *",$UDID,"* ]] || { echo "refusing device $UDID: not in D1_ALLOWED_DEVICES"; exit 2; }
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
/usr/bin/python3 -c 'import json, os, sys
try:
    h = json.load(open(sys.argv[1])); p = h.get("pid")
    if h.get("script") != sys.argv[2] or not isinstance(p, int):
        sys.exit(1)
    os.kill(p, 0)
except PermissionError:
    sys.exit(0)
except Exception:
    sys.exit(1)' $HOLD "${D1_HOLD_SCRIPT:-}" 2>/dev/null \
  || { echo "the phone is not held by this lane ($(head -c 200 $HOLD 2>/dev/null || echo 'no hold file'))"; exit 2; }
dc() { xcrun devicectl device "$@"; }
pullfile() {  # pullfile <container path> <local file>
  rm -f "$2"
  dc copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID --source "$1" --destination "$2" \
    >/dev/null 2>&1
  [ -s "$2" ]
}

case $CMD in
  launch|front)
    args=()
    [ $CMD = launch ] && args=(--terminate-existing)
    app=()
    if [ $CMD = launch ]; then RID=${1:?run id}; shift; app=(-- -runid $RID -log 1 "$@"); fi
    for t in $(seq 1 12); do
      out=$(dc process launch --device $UDID $args $BID $app 2>&1)
      echo "$out" | grep -q "Launched application" && { echo "launched ($CMD) ${app[*]}"; exit 0; }
      echo "launch retry $t: $(echo "$out" | grep -m1 -iE 'error|busy' | cut -c1-160)"; sleep 10
    done
    exit 1 ;;
  wait)
    RID=${1:?run id}; MARK=${2:?mark}; CAP=${3:?cap}; OUT=${4:?out dir}
    mkdir -p $OUT
    t0=$SECONDS; last=""
    while (( SECONDS - t0 < CAP )); do
      if pullfile Documents/d1demo/autoplay.log $OUT/autoplay.log.pull; then
        mv $OUT/autoplay.log.pull $OUT/autoplay.log
        cur=$(grep " $RID " $OUT/autoplay.log | tail -1)
        [ "$cur" != "$last" ] && { echo "  ${cur[1,240]}"; last=$cur; }
        grep " $RID " $OUT/autoplay.log | grep -q " FAILED" && exit 3
        grep " $RID " $OUT/autoplay.log | grep -q " $MARK" && exit 0
      fi
      sleep 2
    done
    echo "no '$MARK' for run $RID after $CAP s"; exit 1 ;;
  trigger)
    NAME=${1:?name}
    T=$(mktemp -d)/$NAME; : > $T
    dc copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID --source $T --destination Documents/$NAME \
      >/dev/null 2>&1 && echo "trigger placed: Documents/$NAME" ;;
  pull)
    RID=${1:?run id}; OUT=${2:?out dir}
    mkdir -p $OUT
    pullfile Documents/d1demo/run-$RID.json $OUT/run-$RID.json && echo "pulled run-$RID.json ($(stat -f %z $OUT/run-$RID.json) B)"
    pullfile Documents/d1demo/autoplay.log $OUT/autoplay.log && echo "pulled autoplay.log" ;;
  shot)
    PNG=${1:?png}
    dc capture screenshot --device $UDID --destination $PNG >/dev/null 2>&1 && echo "screenshot $PNG" ;;
  stop)
    J=$(mktemp)
    dc info processes --device $UDID --json-output $J >/dev/null 2>&1
    pid=$(/usr/bin/python3 - $J <<'PY'
import json, sys
def walk(v):
    if isinstance(v, dict):
        if any(isinstance(x, str) and "D1Demo.app/D1Demo" in x for x in v.values()):
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
)
    if [ -n "$pid" ]; then dc process terminate --device $UDID --pid $pid >/dev/null 2>&1 && echo "terminated pid $pid"
    else echo "no D1Demo process"; fi ;;
  *) echo "unknown command $CMD"; exit 1 ;;
esac
