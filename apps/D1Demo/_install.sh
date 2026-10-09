#!/bin/zsh
# Install D1Demo on the phone and push _work/device_stage/D1Assets/ (./_stage.sh) into its data container as one
# directory (Library/Application Support/D1Assets); then pull the directory back and check every file against the staged
# MD5SUMS, pushing again on its own any file that is missing or differs (up to 3 rounds): `devicectl device copy to` can
# exit 0 with a file cut short. Copied from apps/D1Gate/_install.sh with the demo's bundle id.
#   D1_ALLOWED_DEVICES=<udid>,<coredevice id> D1_HOLD_SCRIPT=<the hold's "script"> ./_install.sh <udid>
#   D1_SKIP_APP=1 ...        assets only (the app is installed)
#   D1_SKIP_PUSH=1 ...       the app only (the assets stayed on the phone; nothing is pulled back)
# Never passes --remove-existing-content (it wipes the whole app container), never uninstalls. Runs only on a device
# D1_ALLOWED_DEVICES lists, and only while the hold (~/code/coreai/ondevice/.device_hold) is this lane's JSON (its
# "script" = D1_HOLD_SCRIPT, its keeper pid alive). Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.d1demo
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
S=${D1_STAGE_DIR:-$W/device_stage/D1Assets}
DEST="Library/Application Support/D1Assets"
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { say "refusing: D1_ALLOWED_DEVICES is not set"; exit 2; }
[[ ",$D1_ALLOWED_DEVICES," == *",$UDID,"* ]] || { say "refusing device $UDID: not in D1_ALLOWED_DEVICES"; exit 2; }
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
hold_is_ours() {
  [ -n "${D1_HOLD_SCRIPT:-}" ] && [ -f $HOLD ] || return 1
  /usr/bin/python3 -c 'import json, os, sys
try:
    h = json.load(open(sys.argv[1])); p = h.get("pid")
    if h.get("script") != sys.argv[2] or not isinstance(p, int):
        sys.exit(1)
    os.kill(p, 0)
except PermissionError:
    sys.exit(0)
except Exception:
    sys.exit(1)' $HOLD "$D1_HOLD_SCRIPT" 2>/dev/null
}
hold_is_ours || { say "the phone is not held by this lane ($(head -c 200 $HOLD 2>/dev/null || echo 'no hold file'))"; exit 2; }
IDS=($UDID ${(s:,:)D1_ALLOWED_DEVICES})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }
push() {  # push <local path> <container path>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2" >> $LOG 2>&1 || say "copy to exited non-zero for ${1:t} (checked below)"
}
[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${D1_SKIP_APP:-0}" != 1 ]; then
  APP=${D1_APP:-$(cat $W/app_path.txt 2>/dev/null)}
  [ -d "$APP" ] || { say "no built app: run ./_build.sh"; exit 1; }
  say "app: $APP ($(codesign -d --entitlements - "$APP" 2>/dev/null | grep -cE 'increased-memory') increased-memory-limit keys, executable $(shasum -a 256 $APP/D1Demo | cut -c1-16))"
  ok=0
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-240)"; sleep 10
  done
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi
[ "${D1_SKIP_PUSH:-0}" = 1 ] && { say "no push (D1_SKIP_PUSH=1)"; exit 0; }

# 2. the assets, the whole directory in one push
say "pushing $S ($(du -sh $S | cut -f1), $(grep -c . $S/MD5SUMS) files) -> $DEST"
t0=$SECONDS
push $S "$DEST"
say "push returned after $((SECONDS - t0)) s"

# 3. pull back, compare with MD5SUMS; re-push what is missing or differs, one file at a time
BAD=()
verify() {
  local P=$W/device_pull
  rm -rf $P; mkdir -p $P
  BAD=()
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST" --destination $P/tree >> $LOG 2>&1
  local M=$(find $P/tree -name MD5SUMS 2>/dev/null | awk '{ print gsub("/", "/"), $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  local R=${M:h}
  [ -n "$M" ] && cmp -s $M $S/MD5SUMS || BAD+=(MD5SUMS)
  while read -r sum rel; do
    if [ -z "$M" ] || [ ! -f "$R/$rel" ] || [ "$(md5 -q "$R/$rel")" != "$sum" ]; then BAD+=("$rel"); fi
  done < $S/MD5SUMS
  [ -n "$M" ] && (cd $R && find . -type f ! -name 'MD5SUMS' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done) \
    > $W/device_pulled_md5_$(date +%Y%m%d-%H%M%S).txt
}
for round in 1 2 3 4; do
  t0=$SECONDS
  verify
  say "check $round: pulled back and compared in $((SECONDS - t0)) s: ${#BAD} files missing or different"
  [ ${#BAD} -eq 0 ] && break
  [ $round -eq 4 ] && break
  say "check $round: ${BAD[1,4]}; pushing them one by one"
  for rel in $BAD; do push "$S/$rel" "$DEST/$rel"; done
done
rm -rf $W/device_pull
if [ ${#BAD} -ne 0 ]; then say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${BAD[1,8]}"; exit 1; fi
say "assets verified on the phone: $(grep -c . $S/MD5SUMS) files md5-equal to the stage, MD5SUMS equal"
