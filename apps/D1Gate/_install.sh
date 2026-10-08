#!/bin/zsh
# Install D1Gate on the phone and push _work/device_stage/D1Assets/ (./_stage.sh) into its data container as one
# directory (Library/Application Support/D1Assets). Then pull the directory back and check every file against the
# staged MD5SUMS; a file that is missing or differs is pushed again on its own (up to 3 rounds). `devicectl device copy
# to` can exit 0 with a file cut short, hence the pull. Copied from apps/KevGate/_install.sh (zoo d1-3b 4955a23).
#   ./_install.sh <udid>                   normally from ./_gate.sh, which holds the phone
#   D1_SKIP_APP=1 ./_install.sh <udid>     assets only (the app is already installed)
#   D1_SKIP_PUSH=1 ./_install.sh <udid>    no whole-directory push (the assets are already there): the md5 check below
#                                          still pulls everything back and re-pushes what differs
#   D1_STAGE_DIR=<dir> D1_PUSH_ONLY=<rel>,<rel> D1_SKIP_APP=1 ./_install.sh <udid>
#                                          push only these paths of another stage directory into D1Assets/<rel> (the
#                                          AOT asset: ./_stage.sh --aot); every pushed file's size is then checked from
#                                          the phone's listing instead of pulling it back (the app's md5 stage reads the
#                                          bytes: D1_MD5SUMS=MD5SUMS_AOT)
# Never passes --remove-existing-content (it wipes the whole app container, not the destination), never uninstalls. The
# destination names the pushed directory itself (.../D1Assets, and .../<file> on a re-push): a push onto a parent
# flattens a bundle's tree. Runs only while this lane holds the phone (the first line of
# ~/code/coreai/ondevice/.device_hold starts with "d1-3b device gate"; ./_gate.sh takes and releases it), and only on
# a device D1_ALLOWED_DEVICES lists.
# Log: _work/device_install_<time>.log
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.d1gate
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
S=${D1_STAGE_DIR:-$W/device_stage/D1Assets}
DEST="Library/Application Support/D1Assets"
LOG=$W/device_install_$(date +%Y%m%d-%H%M%S).log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { say "refusing: D1_ALLOWED_DEVICES is not set (the 18 Pro's UDID and CoreDevice id)"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^d1-3b device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
# a real devicectl process on this phone only (a session whose prompt quotes these words must not count; other phones do
# not count): the id given plus D1_ALLOWED_DEVICES (the same phone by UDID and by CoreDevice id)
IDS=($UDID ${(s:,:)${D1_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }

push() {  # push <local path> <container path>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2" >> $LOG 2>&1 || say "copy to exited non-zero for ${1:t} (checked below)"
}

# an extra asset (the AOT): only the listed paths, size check from the phone's listing
if [ -n "${D1_PUSH_ONLY:-}" ]; then
  [ "${D1_SKIP_APP:-0}" = 1 ] || { say "D1_PUSH_ONLY pushes into the installed app: set D1_SKIP_APP=1"; exit 1; }
  rels=(${(s:,:)D1_PUSH_ONLY})
  for rel in $rels; do [ -e "$S/$rel" ] || { say "nothing staged at $S/$rel"; exit 1; }; done
  t0=$SECONDS
  for rel in $rels; do
    say "pushing $S/$rel ($(du -sh "$S/$rel" | cut -f1)) -> $DEST/$rel"
    t1=$SECONDS
    push "$S/$rel" "$DEST/$rel"
    say "push of $rel returned after $((SECONDS - t1)) s"
  done
  check_sizes() {  # every staged file under the pushed paths, by size in bytes, from the phone's listing of each file's
    # directory (--json-output: the table rounds sizes to "2.33 GB")
    local bad=0 f rel dir name want got J=$W/device_files_listing.json
    for rel in $rels; do
      for f in ${(f)"$(cd $S && find $rel -type f | LC_ALL=C sort)"}; do
        dir=${f:h}; name=${f:t}
        want=$(stat -f %z "$S/$f")
        [ "$dir" = . ] && dir=""
        rm -f $J
        xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
          --subdirectory "$DEST${dir:+/$dir}" --json-output $J >> $LOG 2>&1
        got=$(/usr/bin/python3 -c 'import json,sys
try:
    fs = json.load(open(sys.argv[1]))["result"]["files"]
except Exception:
    sys.exit(0)
print(next((str(x["metadata"]["size"]) for x in fs if x.get("relativePath") == sys.argv[2] or x.get("name") == sys.argv[2]), ""))' $J "$name")
        if [ "$got" != "$want" ]; then
          bad=$((bad + 1)); say "size differs: $f staged $want, phone '${got:-absent}'"
        fi
      done
    done
    return $bad
  }
  for round in 1 2 3; do
    if check_sizes; then say "sizes equal on the phone for every file of ${D1_PUSH_ONLY} (check $round)"; break; fi
    [ $round -eq 3 ] && { say "ERROR sizes still differ after 2 re-pushes"; exit 1; }
    for rel in $rels; do push "$S/$rel" "$DEST/$rel"; done
  done
  say "pushed ${D1_PUSH_ONLY} in $((SECONDS - t0)) s (the bytes are md5-checked on the phone by the md5 stage)"
  exit 0
fi

[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${D1_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/app_path.txt 2>/dev/null)
  [ -d "$APP" ] || { say "no built app: run ./_build.sh"; exit 1; }
  ok=0
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-200)"; sleep 10
  done
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi

# 2. the assets, the whole directory in one push
if [ "${D1_SKIP_PUSH:-0}" != 1 ]; then
  say "pushing $S ($(du -sh $S | cut -f1), $(grep -c . $S/MD5SUMS) files) -> $DEST"
  t0=$SECONDS
  push $S "$DEST"
  say "push returned after $((SECONDS - t0)) s"
else
  say "no whole-directory push (D1_SKIP_PUSH=1): checking what is on the phone against $S/MD5SUMS"
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
  # every file, one directory pull; the tree may land at the destination or one level under it, so its root is wherever
  # the shallowest MD5SUMS is
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST" --destination $P/tree >> $LOG 2>&1
  local M=$(find $P/tree -name MD5SUMS 2>/dev/null | awk '{ print gsub("/", "/"), $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  local R=${M:h}
  while read -r sum rel; do
    if [ -z "$M" ] || [ ! -f "$R/$rel" ] || [ "$(md5 -q "$R/$rel")" != "$sum" ]; then BAD+=("$rel"); fi
  done < $S/MD5SUMS
  # the md5 list of what came back, for the record (every file, as the phone holds it)
  [ -n "$M" ] && (cd $R && find . -type f ! -name 'MD5SUMS*' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done) \
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
if [ ${#BAD} -ne 0 ]; then
  say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${BAD[1,8]}"; exit 1
fi
say "assets verified on the phone: $(grep -c . $S/MD5SUMS) files md5-equal to the stage, MD5SUMS equal"
rm -rf $W/device_pull
