#!/bin/zsh
# Install N3DGate on the phone and push _work/device_stage/N3DAssets/ (./_stage.sh) into its data container
# as one directory (Library/Application Support/N3DAssets). Then pull the directory back and check every file
# against the staged MD5SUMS; a file that is missing or differs is pushed again on its own (up to 3 rounds).
# `devicectl device copy to` can exit 0 with a file cut short, hence the pull.
#   ./_install.sh <udid>                    udid: `xcrun devicectl list devices`
#   N3D_SKIP_APP=1 ./_install.sh <udid>     assets only (the app is already installed)
# Never passes --remove-existing-content (it wipes the whole app container, not the destination).
# Refuses while another session holds the phone (~/code/coreai/ondevice/.device_hold) or runs devicectl.
# Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.n3dgate
DIR=${0:A:h}
W=${N3D_WORK:-${DIR:h:h}/conversion/nemotron3_diar/_work}
S=$W/device_stage/N3DAssets
DEST="Library/Application Support/N3DAssets"
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

HOLD=${N3D_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if [ -e $HOLD ] && [ "${N3D_IGNORE_HOLD:-0}" != 1 ]; then
  say "device hold present ($HOLD: $(head -c 200 $HOLD)); N3D_IGNORE_HOLD=1 to go anyway"; exit 2
fi
# a real devicectl process only (a session whose prompt quotes these words must not count)
busy() { ps -axo pid,command | grep -qE "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy is running"; exit 2; }
[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${N3D_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/n3dgate_app_path.txt 2>/dev/null)
  [ -d "$APP" ] || { say "no built app: run ./_build.sh"; exit 1; }
  ok=0
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-160)"; sleep 10
  done
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi

push() {  # push <local path> <container path>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2" >> $LOG 2>&1 || say "copy to exited non-zero for ${1:t} (checked below)"
}

# 2. the assets, the whole directory in one push
say "pushing $S ($(du -sh $S | cut -f1), $(grep -c . $S/MD5SUMS) files) -> $DEST"
push $S "$DEST"

# 3. pull back, compare with MD5SUMS; re-push what is missing or differs, one file at a time
BAD=()
verify() {
  local P=$W/device_pull
  rm -rf $P; mkdir -p $P
  BAD=()
  # the exact path the app reads (a single-file pull lands at the destination path itself)
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST/MD5SUMS" --destination $P/MD5SUMS.exact >> $LOG 2>&1
  cmp -s $P/MD5SUMS.exact $S/MD5SUMS || BAD+=(MD5SUMS)
  # every file, one directory pull; the tree may land at the destination or one level under it, so its root is
  # wherever the shallowest MD5SUMS is
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST" --destination $P/tree >> $LOG 2>&1
  local M=$(find $P/tree -name MD5SUMS 2>/dev/null | awk '{ print gsub("/", "/"), $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  local R=${M:h}
  while read -r sum rel; do
    if [ -z "$M" ] || [ ! -f "$R/$rel" ] || [ "$(md5 -q "$R/$rel")" != "$sum" ]; then BAD+=("$rel"); fi
  done < $S/MD5SUMS
}
for round in 1 2 3 4; do
  verify
  [ ${#BAD} -eq 0 ] && break
  [ $round -eq 4 ] && break
  say "check $round: ${#BAD} files missing or different (${BAD[1,4]}); pushing them one by one"
  for rel in $BAD; do push "$S/$rel" "$DEST/$rel"; done
done
if [ ${#BAD} -ne 0 ]; then
  say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${BAD[1,8]}"; exit 1
fi
say "assets verified on the phone: $(grep -c . $S/MD5SUMS) files md5-equal to the stage, MD5SUMS equal"
rm -rf $W/device_pull
