#!/bin/zsh
# Build D1Gate: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ). Build only: no install, no
# launch. Copied from apps/KevGate/_build.sh (zoo d1-3b 4955a23) with d1's names.
#   ./_build.sh                      generic iOS, no entitlements   -> _work/app_path.txt (+ app_path_noiml.txt)
#   D1_ENTITLED=1 ./_build.sh        generic iOS with D1Gate.entitlements (com.apple.developer.kernel.increased-memory-
#                                    limit)                         -> _work/app_path.txt (+ app_path_iml.txt)
#   D1_ENTITLED=1 D1_PROVISIONING_UPDATES=1 ./_build.sh
#                                    the same with -allowProvisioningUpdates: Xcode's account may register the App ID's
#                                    capability and fetch its profile (a new bundle id has no profile with the
#                                    capability: the wildcard one cannot carry it, apps/DeciderVisionGate README)
#   ./_build.sh --mac                macOS (arm64)                  -> _work/app_path_mac.txt
# The two iOS builds keep their own derived data (dd_ios, dd_ios_iml); _work/app_path.txt names the last one built,
# which ./_install.sh installs (D1_APP overrides). A signing failure is printed with its error lines and leaves
# app_path.txt as it was. The app records os_proc_available_memory either way. The library comes by path (project.yml: ../D1). A
# first build that resolves packages can write the path dependency's Package.resolved (memory:
# reference_xcodebuild_local_package_resolved): this script copies it before the build and, when the build changed it,
# keeps the diff in _work/ and puts the copy back; the md5 of every other file of ../D1 (not .build / .swiftpm) is
# compared before and after.
# A Mac measurement window (~/code/coreai/_GPU_LOCK naming `timing`, its holder alive) is waited out first, in 30 s
# steps, for at most D1_BUILD_WAIT_CAP s (default 2400; then exit 4 without building): a full-CPU build would disturb
# the window's numbers. D1_BUILD_JOBS (default 4) caps xcodebuild's parallel jobs.
# Derived data, package checkouts and logs: _work/ (git-ignored).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
LIB=${DIR:h}/D1
MAC=0
for a in "$@"; do
  case $a in
    --mac) MAC=1 ;;
    *) echo "unknown option $a (--mac)"; exit 1 ;;
  esac
done
IML=0
if (( MAC )); then
  TAG=mac; DEST='generic/platform=macOS'; PRODUCTS=Release; EXTRA=(ARCHS=arm64)
else
  TAG=ios; DEST='generic/platform=iOS'; PRODUCTS=Release-iphoneos; EXTRA=()
  if [ "${D1_ENTITLED:-0}" = 1 ]; then
    IML=1; TAG=ios_iml; EXTRA+=(D1_ENTITLEMENTS_FILE=D1Gate.entitlements)
    [ "${D1_PROVISIONING_UPDATES:-0}" = 1 ] && EXTRA+=(-allowProvisioningUpdates)
  fi
fi
DD=$W/dd_$TAG
LOG=$W/xcodebuild_$TAG.log
mkdir -p $W
[ -f $LIB/Package.swift ] || { echo "no D1 package at $LIB"; exit 1; }

# a measurement window: the lock names `timing` and its holder pid lives (quiet_wait.py's rule)
LOCK=$HOME/code/coreai/_GPU_LOCK
window() {
  local c; c=$(cat $LOCK 2>/dev/null) || return 1
  [[ ${c:l} == *timing* ]] || return 1
  local pid; pid=$(echo "$c" | sed -nE 's/.*pid ([0-9]+).*/\1/p')
  [ -z "$pid" ] && return 0
  kill -0 $pid 2>/dev/null
}
t0=$SECONDS
while window; do
  if (( SECONDS - t0 >= ${D1_BUILD_WAIT_CAP:-2400} )); then
    echo "[$(date '+%H:%M:%S')] a measurement window is still open after $((SECONDS - t0)) s ($(head -c 120 $LOCK)): not building"; exit 4
  fi
  echo "[$(date '+%H:%M:%S')] waiting: a measurement window is open ($(head -c 120 $LOCK))"
  sleep 30
done
echo "[$(date '+%H:%M:%S')] no measurement window (waited $((SECONDS - t0)) s): building"

libsums() { (cd $LIB && find . -type f -not -path './.build/*' -not -path './.swiftpm/*' | LC_ALL=C sort | xargs md5 -r) }
RES=$LIB/Package.resolved
SAVED=$W/D1.Package.resolved.before_build
SUM_BEFORE=""
if [ -f $RES ]; then cp -p $RES $SAVED; SUM_BEFORE=$(md5 -q $RES); fi
libsums > $W/d1_lib_md5_before_$TAG.txt

cd $DIR
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 1; }
echo "[$(date '+%H:%M:%S')] xcodebuild $TAG: $DEST, Release ${EXTRA[*]} ($( (( IML )) && echo 'increased-memory-limit' || echo 'no entitlements'), -jobs ${D1_BUILD_JOBS:-4})"
t0=$SECONDS
xcodebuild -project D1Gate.xcodeproj -scheme D1Gate -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $W/spm -jobs ${D1_BUILD_JOBS:-4} "${EXTRA[@]}" build > $LOG 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] xcodebuild returned $rc after $((SECONDS - t0)) s"

# the library's Package.resolved: put the copy back when this build changed it
SUM_AFTER=$(md5 -q $RES 2>/dev/null)
if [ "$SUM_AFTER" != "$SUM_BEFORE" ]; then
  D=$W/d1_package_resolved_$(date +%Y%m%d-%H%M%S).diff
  diff -u $SAVED $RES > $D 2>&1
  if [ -n "$SUM_BEFORE" ]; then cp -p $SAVED $RES; else rm -f $RES; fi
  echo "D1/Package.resolved: this build changed it; diff kept in $D, file restored ($(md5 -q $RES 2>/dev/null || echo absent))"
fi
libsums > $W/d1_lib_md5_after_$TAG.txt
if cmp -s $W/d1_lib_md5_before_$TAG.txt $W/d1_lib_md5_after_$TAG.txt; then
  echo "../D1: every file md5-equal before and after the build ($(grep -c . $W/d1_lib_md5_after_$TAG.txt) files)"
else
  echo "../D1: CHANGED during the build (another session's edit, or the build — check the mtime and the content):"
  diff $W/d1_lib_md5_before_$TAG.txt $W/d1_lib_md5_after_$TAG.txt | head -10
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  # the signing lines verbatim (a capability the profile lacks, no account, no profile), for the record
  grep -iE "provision|signing|account|capabilit|entitlement" $LOG | grep -iE "error|warning|note" | sort -u | head -12 \
    > $W/xcodebuild_${TAG}_signing_errors.txt
  [ -s $W/xcodebuild_${TAG}_signing_errors.txt ] && { echo "signing lines (also in $W/xcodebuild_${TAG}_signing_errors.txt):"; cat $W/xcodebuild_${TAG}_signing_errors.txt; }
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/D1Gate.app
[ -d $APP ] || { echo "no app at $APP"; exit 1; }
case $TAG in
  ios) echo $APP > $W/app_path.txt; echo $APP > $W/app_path_noiml.txt ;;
  ios_iml) echo $APP > $W/app_path.txt; echo $APP > $W/app_path_iml.txt ;;
  mac) echo $APP > $W/app_path_mac.txt ;;
esac
echo "app: $APP ($(du -sh $APP | cut -f1))"
echo "compiler warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
echo "other compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | head -8
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
echo "entitlements: $(codesign -d --entitlements - $APP 2>/dev/null | grep -cE 'increased-memory') increased-memory-limit keys"
if [ -f $APP/embedded.mobileprovision ]; then
  P=$W/embedded_$TAG.plist
  security cms -D -i $APP/embedded.mobileprovision > $P 2>/dev/null
  echo "profile: $(/usr/libexec/PlistBuddy -c 'Print :Name' $P 2>/dev/null) (application-identifier" \
    "$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:application-identifier' $P 2>/dev/null), increased-memory-limit" \
    "$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:com.apple.developer.kernel.increased-memory-limit' $P 2>/dev/null || echo absent))"
fi
