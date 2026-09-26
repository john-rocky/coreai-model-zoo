#!/bin/zsh
# Install DecideGate on the phone and push _work/device_stage/DecideAssets/ (./_stage.sh) into its data container
# as one directory (Library/Application Support/DecideAssets). Then pull the directory back and check every file
# against the staged MD5SUMS; a file that is missing or differs is pushed again on its own (up to 3 rounds).
# `devicectl device copy to` can exit 0 with a file cut short, hence the pull.
#   ./_install.sh <udid>                       normally from ./_gate.sh, which holds the phone
#   DECIDE_SKIP_APP=1 ./_install.sh <udid>     assets only (the app is already installed)
#   DECIDE_SKIP_PUSH=1 ./_install.sh <udid>    no whole-directory push (the assets are already there): the md5
#                                              check below still pulls everything back and re-pushes what differs
#   DECIDE_FRESH=1 ./_install.sh <udid>        uninstall first: the app and its whole data container go (assets,
#                                              results, launch count, and the container's Core AI cache,
#                                              Library/Caches/coreai-cache), then install and push everything, so
#                                              the next launch loads every bundle with no container cache (the
#                                              system-side cache is out of reach)
# Never passes --remove-existing-content (it wipes the whole app container, not the destination). The destination
# names the pushed directory itself (.../DecideAssets, and .../<file> on a re-push): a push onto a parent flattens a
# bundle's tree and the load fails (failedToSpecialize).
# Runs only while this lane holds the phone (the first line of ~/code/coreai/ondevice/.device_hold starts with
# "GLiNER2.5-Decide device gate"; ./_gate.sh takes and releases it), and, when DECIDE_ALLOWED_DEVICES (comma-separated
# ids) is set, only on a device it lists.
# Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.decidegate
DIR=${0:A:h}
W=${DECIDE_WORK:-${DIR:h:h}/conversion/gliner25_decide/_work}
S=$W/device_stage/DecideAssets
DEST="Library/Application Support/DecideAssets"
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

if [ -n "${DECIDE_ALLOWED_DEVICES:-}" ] && [[ ",$DECIDE_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in DECIDE_ALLOWED_DEVICES ($DECIDE_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${DECIDE_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^GLiNER2.5-Decide device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
# a real devicectl process on this phone only (a session whose prompt quotes these words must not count; other
# phones do not count): the id given plus DECIDE_ALLOWED_DEVICES (the same phone by UDID and by CoreDevice id)
IDS=($UDID ${(s:,:)${DECIDE_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }
[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

# 0. DECIDE_FRESH=1: uninstall (the data container goes with the app), confirmed by the app list, grepped for the id
if [ "${DECIDE_FRESH:-0}" = 1 ]; then
  [ "${DECIDE_SKIP_APP:-0}" = 1 ] && { say "DECIDE_FRESH=1 reinstalls the app: unset DECIDE_SKIP_APP"; exit 1; }
  [ "${DECIDE_SKIP_PUSH:-0}" = 1 ] && { say "DECIDE_FRESH=1 pushes the assets again: unset DECIDE_SKIP_PUSH"; exit 1; }
  installed() {  # 0 = listed, 1 = not listed, 2 = no answer
    local A; A=$(xcrun devicectl device info apps --device $UDID 2>&1) || { echo "$A" >> $LOG; return 2; }
    echo "$A" | grep -qF "$BID"
  }
  installed; st=$?
  [ $st -eq 2 ] && { say "ERROR the app list did not answer (see $LOG)"; exit 1; }
  if [ $st -eq 0 ]; then
    OUT=$(xcrun devicectl device uninstall app --device $UDID $BID 2>&1); echo "$OUT" >> $LOG
    installed; st=$?
    [ $st -eq 1 ] || { say "ERROR $BID still listed (or no answer) after uninstall: $(echo "$OUT" | grep -m1 -i error | cut -c1-160)"; exit 1; }
    say "uninstalled $BID with its data container (DECIDE_FRESH=1)"
  else
    say "$BID was not installed (DECIDE_FRESH=1): nothing to uninstall"
  fi
fi

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${DECIDE_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/decidegate_app_path.txt 2>/dev/null)
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
if [ "${DECIDE_SKIP_PUSH:-0}" != 1 ]; then
  say "pushing $S ($(du -sh $S | cut -f1), $(grep -c . $S/MD5SUMS) files) -> $DEST"
  push $S "$DEST"
else
  say "no whole-directory push (DECIDE_SKIP_PUSH=1): checking what is on the phone against $S/MD5SUMS"
fi

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
if [ "${DECIDE_FRESH:-0}" = 1 ]; then
  # what the new container's Library/Caches holds before the first launch (a fresh one: no coreai-cache)
  C=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --subdirectory Library/Caches 2>&1); echo "$C" >> $LOG
  say "fresh container, Library/Caches before the first launch: $(echo "$C" | grep -c coreai-cache) lines naming coreai-cache"
fi
