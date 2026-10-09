#!/bin/zsh
# Install D1OmniGate on the phone and push $D1_WORK/assets/D1OmniAssets/ (./_stage.sh) into its data container as one
# directory (Library/Application Support/D1OmniAssets). Then check every staged file's size against one recursive
# listing of the phone's copy (`devicectl device info files --json-output`: the table rounds sizes); a file that is
# missing or differs is pushed again on its own (up to 3 rounds). `devicectl device copy to` can exit 0 with a file cut
# short (lane memory reference_devicectl_transfer_gotchas), hence the check; the bytes themselves are md5-checked on the
# phone by the app (the small files in its assets stage at launch, the model files in its md5 stage), which reads 6.5 GB
# in seconds where a pull back over USB takes minutes. D1_VERIFY=pull pulls the whole directory back and compares md5
# on the Mac instead (apps/KevGate/_install.sh's check). Adapted from apps/KevGate/_install.sh.
#   ./_install.sh <udid>                   normally from ./_gate.sh, which holds the phone
#   D1_SKIP_APP=1 ./_install.sh <udid>     assets only (the app is already installed)
#   D1_SKIP_PUSH=1 ./_install.sh <udid>    no whole-directory push (the assets are already there): the check below still
#                                          lists the phone's copy and re-pushes what differs
# Never passes --remove-existing-content (it wipes the whole app container, not the destination), never uninstalls. The
# destination names the pushed directory itself (.../D1OmniAssets, and .../<file> on a re-push): a push onto a parent
# flattens a bundle's tree. Runs only while this lane holds the phone (the first line of
# ~/code/coreai/ondevice/.device_hold starts with "d1-omni device gate"; ./_gate.sh takes and releases it), and only on a
# device D1_ALLOWED_DEVICES lists.
# Log: $D1_WORK/install_<time>.log; the transfer's seconds and bytes: $D1_WORK/install_<time>.json
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
UDID=${1:?usage: _install.sh <udid>}
BID=com.daisukemajima.d1omnigate
DIR=${0:A:h}
W=${D1_WORK:-$HOME/code/coreai/_d1_omni/device}
S=${D1_STAGE_DIR:-$W/assets/D1OmniAssets}
DEST="Library/Application Support/D1OmniAssets"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=$W/install_$STAMP.log
REC=$W/install_$STAMP.json
mkdir -p $W
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

[ -n "${D1_ALLOWED_DEVICES:-}" ] || { say "refusing: D1_ALLOWED_DEVICES is not set (the 18 Pro's UDID and CoreDevice id)"; exit 2; }
if [[ ",$D1_ALLOWED_DEVICES," != *",$UDID,"* ]]; then
  say "refusing device $UDID: not in D1_ALLOWED_DEVICES ($D1_ALLOWED_DEVICES)"; exit 2
fi
HOLD=${D1_HOLD_FILE:-$HOME/code/coreai/ondevice/.device_hold}
if ! { [ -f $HOLD ] && head -1 $HOLD | grep -q "^d1-omni device gate"; }; then
  say "the phone is not held by this lane ($HOLD: $(head -c 200 $HOLD 2>/dev/null || echo absent)); go through ./_gate.sh"; exit 2
fi
# a real devicectl process on this phone only (a session whose prompt quotes these words must not count; other phones do
# not count): the id given plus D1_ALLOWED_DEVICES (the same phone by UDID and by CoreDevice id)
IDS=($UDID ${(s:,:)${D1_ALLOWED_DEVICES:-}})
busy() { ps -axo pid,command | grep -E "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))" \
  | grep -qE -- "--device (${(j:|:)IDS})( |\$)"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy on $UDID is running"; exit 2; }
[ -f $S/MD5SUMS ] || { say "nothing staged at $S: run ./_stage.sh"; exit 1; }

push() {  # push <local path> <container path>
  xcrun devicectl device copy to --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$1" --destination "$2" >> $LOG 2>&1 || say "copy to exited non-zero for ${1:t} (checked below)"
}
STAGED_BYTES=$(find $S -type f -exec stat -f %z {} + | paste -sd+ - | bc)
STAGED_FILES=$(find $S -type f | wc -l | tr -d ' ')
APP_S=0; PUSH_S=0; CHECK_S=0; REPUSHED=0

# 1. the app: success = an installationURL line (a failed install leaves the previous app in place)
if [ "${D1_SKIP_APP:-0}" != 1 ]; then
  APP=$(cat $W/app_path.txt 2>/dev/null)
  [ -d "$APP" ] || { say "no built app: run ./_build.sh"; exit 1; }
  ok=0; t0=$SECONDS
  for attempt in 1 2 3; do
    OUT=$(xcrun devicectl device install app --device $UDID "$APP" 2>&1); echo "$OUT" >> $LOG
    if echo "$OUT" | grep -q installationURL; then ok=1; say "installed $BID ($(du -sh $APP | cut -f1))"; break; fi
    say "install attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-200)"; sleep 10
  done
  APP_S=$((SECONDS - t0))
  [ $ok = 1 ] || { say "ERROR install failed 3 times"; exit 1; }
fi

# 2. the assets, the whole directory in one push
if [ "${D1_SKIP_PUSH:-0}" != 1 ]; then
  say "pushing $S ($STAGED_BYTES bytes, $STAGED_FILES files) -> $DEST"
  t0=$SECONDS
  push $S "$DEST"
  PUSH_S=$((SECONDS - t0))
  say "push returned after $PUSH_S s ($(( PUSH_S > 0 ? STAGED_BYTES / PUSH_S / 1000000 : 0 )) MB/s)"
else
  say "no whole-directory push (D1_SKIP_PUSH=1): checking what is on the phone against $S"
fi

# 3. every staged file's size against one recursive listing of the phone's copy; re-push what differs
BAD=()
listing() {  # -> BAD (relative paths missing or of another size); returns 2 when the listing could not be read
  local J=$W/device_files_listing.json
  rm -f $J
  xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --subdirectory "$DEST" --json-output $J >> $LOG 2>&1
  local out
  out=$(/usr/bin/python3 -I - $J $S <<'PY'
import json, os, sys
J, S = sys.argv[1:3]
try:
    files = json.load(open(J))["result"]["files"]
except Exception as e:
    print(f"UNREADABLE {e}"); sys.exit(0)
got = {}
for x in files:
    rel = x.get("relativePath") or x.get("name") or ""
    meta = x.get("metadata") or {}
    if meta.get("type") in ("directory", "dir") or rel.endswith("/"):
        continue
    size = meta.get("size")
    if isinstance(size, int):
        got[rel.lstrip("./")] = size
staged = {}
for root, _, fs in os.walk(S):
    for f in fs:
        p = os.path.join(root, f)
        staged[os.path.relpath(p, S)] = os.path.getsize(p)
if not got:
    print(f"UNREADABLE no file entries ({len(files)} listed): {json.dumps(files[:3])[:300]}"); sys.exit(0)
bad = [r for r, n in sorted(staged.items()) if got.get(r) != n]
print(f"LISTED {len(got)} STAGED {len(staged)}")
for r in bad:
    print(f"BAD {r}")
PY
)
  echo "$out" | head -3 >> $LOG
  if echo "$out" | grep -q '^UNREADABLE'; then say "listing: $(echo "$out" | head -1 | cut -c1-300)"; return 2; fi
  BAD=(${(f)"$(echo "$out" | sed -n 's/^BAD //p')"})
  say "listing: $(echo "$out" | grep '^LISTED')"
  return 0
}
pull_check() {  # D1_VERIFY=pull: the whole directory back, md5 against MD5SUMS -> BAD
  local P=$W/device_pull
  rm -rf $P; mkdir -p $P
  BAD=()
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $BID \
    --source "$DEST" --destination $P/tree >> $LOG 2>&1
  local M=$(find $P/tree -name MD5SUMS 2>/dev/null | awk '{ print gsub("/", "/"), $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  local R=${M:h}
  while read -r sum rel; do
    if [ -z "$M" ] || [ ! -f "$R/$rel" ] || [ "$(md5 -q "$R/$rel")" != "$sum" ]; then BAD+=("$rel"); fi
  done < $S/MD5SUMS
  cmp -s $R/MD5SUMS $S/MD5SUMS || BAD+=(MD5SUMS)
  rm -rf $P
}
MODE=${D1_VERIFY:-sizes}
for round in 1 2 3 4; do
  t0=$SECONDS
  if [ $MODE = sizes ]; then
    listing; st=$?
    if [ $st -eq 2 ]; then say "the listing could not be read: falling back to the pull check"; MODE=pull; pull_check; fi
  else
    pull_check
  fi
  CHECK_S=$((CHECK_S + SECONDS - t0))
  say "check $round ($MODE) in $((SECONDS - t0)) s: ${#BAD} files missing or different"
  [ ${#BAD} -eq 0 ] && break
  [ $round -eq 4 ] && break
  say "check $round: ${BAD[1,4]}; pushing them one by one"
  for rel in $BAD; do push "$S/$rel" "$DEST/$rel"; REPUSHED=$((REPUSHED + 1)); done
done
/usr/bin/python3 -I -c 'import json,sys; a=sys.argv; json.dump({"udid": a[1], "stage": a[2], "staged_bytes": int(a[3]),
  "staged_files": int(a[4]), "app_install_s": int(a[5]), "push_s": int(a[6]), "push_mb_per_s": (int(a[3]) / int(a[6]) / 1e6) if int(a[6]) else None,
  "check": a[7], "check_s": int(a[8]), "repushed_files": int(a[9]), "bad_after_checks": int(a[10]), "log": a[11]},
  open(a[12], "w"), indent=1)' $UDID $S $STAGED_BYTES $STAGED_FILES $APP_S $PUSH_S $MODE $CHECK_S $REPUSHED ${#BAD} $LOG $REC
if [ ${#BAD} -ne 0 ]; then
  say "ERROR after 3 re-pushes ${#BAD} files still missing or different: ${BAD[1,8]}"; exit 1
fi
say "assets on the phone: $STAGED_FILES files, sizes ($MODE) equal to the stage; the app md5-checks the bytes (record: $REC)"
