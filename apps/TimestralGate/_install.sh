#!/bin/zsh
# Install TimestralGate on the phone and push the files of a push list into its data container, one file at a time,
# then pull every file back and compare its md5 with the local one; a file that is missing or differs is pushed again
# on its own (up to 3 more rounds). `devicectl device copy to` can exit 0 with a file cut short, and it skips a
# destination of the same size with a newer mtime, hence the pull.
#   ./_install.sh <udid> <push list>     push list: one "<local file><TAB><container path>" per line
#                                         (container paths under Documents/: models/…, inputs/…)
#   TIMESTRAL_SKIP_APP=1 ./_install.sh …  files only (the app is already installed)
#   TIMESTRAL_SKIP_PUSH=1 ./_install.sh … no first push: pull everything back and re-push only what differs
# Never passes --remove-existing-content (it wipes the whole app container, not the destination). Every push names the
# file itself as the destination (a push onto a parent directory flattens a bundle's tree).
# Runs only while this lane holds the phone (the JSON in ~/code/coreai/ondevice/.device_hold names TIMESTRAL_HOLD_SCRIPT
# and its pid is alive) and only on a device in TIMESTRAL_ALLOWED_DEVICES (default: the iPhone 18 Pro bench phone).
# Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid> <push list>}
LIST=${2:?usage: _install.sh <udid> <push list>}
BID=com.coreai.timestralgate
DIR=${0:A:h}
W=${TIMESTRAL_WORK:-$DIR/_work}
ALLOWED=${TIMESTRAL_ALLOWED_DEVICES:-00008160-000038CA02C00036}
HOLD=${TIMESTRAL_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
SCRIPT=${TIMESTRAL_HOLD_SCRIPT:-timestral_c_r2}
mkdir -p $W
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

[[ ",$ALLOWED," == *",$UDID,"* ]] || { say "refusing device $UDID: not in TIMESTRAL_ALLOWED_DEVICES ($ALLOWED)"; exit 2; }
held() { /usr/bin/python3 -c 'import json,os,sys
try:
    h = json.load(open(sys.argv[1])); os.kill(int(h["pid"]), 0)
    sys.exit(0 if h.get("script") == sys.argv[2] else 1)
except Exception:
    sys.exit(1)' $HOLD $SCRIPT 2>/dev/null; }
held || { say "the phone is not held by $SCRIPT ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent))"; exit 2; }
[ -f $LIST ] || { say "no push list at $LIST"; exit 1; }

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${TIMESTRAL_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/app_path.txt 2>/dev/null)
  [ -d "$APP" ] || { say "no built app: run ./_build.sh ios"; exit 1; }
  ok=0
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID from $APP"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-160)"; sleep 10
  done
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi

typeset -A SUM
LOCALS=(); REMOTES=()
while IFS=$'\t' read -r loc rem; do
  [ -z "$loc" ] && continue
  [ -f "$loc" ] || { say "ERROR push list names a missing local file: $loc"; exit 1; }
  LOCALS+=("$loc"); REMOTES+=("$rem")
done < $LIST
say "push list: ${#LOCALS} files, $(( $(stat -f %z "${LOCALS[@]}" | paste -sd+ - | bc) / 1000000 )) MB"
for i in {1..${#LOCALS}}; do SUM[$i]=$(md5 -q "${LOCALS[$i]}"); done

push() {  # push <index>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "${LOCALS[$1]}" --destination "${REMOTES[$1]}" >> $LOG 2>&1 || say "copy to exited non-zero for ${REMOTES[$1]} (checked below)"
}

# 2. every file once
if [ "${TIMESTRAL_SKIP_PUSH:-0}" != 1 ]; then
  t0=$SECONDS
  for i in {1..${#LOCALS}}; do push $i; done
  say "pushed ${#LOCALS} files in $((SECONDS - t0)) s"
fi

# 3. pull every file back and compare; re-push what is missing or differs, one file at a time
P=$W/device_pull_check
BAD=()
verify() {
  rm -rf $P; mkdir -p $P
  BAD=()
  for i in {1..${#LOCALS}}; do
    local dst=$P/$i.bin
    xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
      --source "${REMOTES[$i]}" --destination $dst >> $LOG 2>&1
    if [ ! -f $dst ] || [ "$(md5 -q $dst)" != "${SUM[$i]}" ]; then BAD+=($i); fi
    rm -f $dst
  done
}
for round in 1 2 3 4; do
  t0=$SECONDS
  verify
  say "check $round: pulled back and compared ${#LOCALS} files in $((SECONDS - t0)) s: ${#BAD} missing or different"
  [ ${#BAD} -eq 0 ] && break
  [ $round -eq 4 ] && break
  for i in $BAD; do say "re-push ${REMOTES[$i]}"; touch "${LOCALS[$i]}"; push $i; done
done
rm -rf $P
if [ ${#BAD} -ne 0 ]; then
  say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${REMOTES[${BAD[1]}]} …"; exit 1
fi
say "INSTALL_OK: ${#LOCALS} files md5-equal on the phone"
