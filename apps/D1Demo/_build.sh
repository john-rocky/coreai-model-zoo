#!/bin/zsh
# Build D1Demo: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ). Build only: no install, no
# launch. Copied from apps/D1Gate/_build.sh with the demo's names; the iOS build always carries D1Demo.entitlements
# (com.apple.developer.kernel.increased-memory-limit: the iPhone's decoder does not specialize without it).
#   ./_build.sh                      generic iOS                    -> _work/app_path.txt
#   D1_PROVISIONING_UPDATES=1 ./_build.sh
#                                    the same with -allowProvisioningUpdates: Xcode's account registers the new App ID
#                                    com.daisukemajima.d1demo with the capability and fetches its profile (the team's
#                                    wildcard profile cannot carry the key: apps/D1Gate README, "Signing")
#   ./_build.sh --mac                macOS (arm64)                  -> _work/app_path_mac.txt
# A Mac measurement window (~/code/coreai/_GPU_LOCK naming `timing`, its holder alive) is waited out first, in 30 s steps,
# for at most D1_BUILD_WAIT_CAP s (default 2400; then exit 4 without building). D1_BUILD_JOBS (default 4) caps
# xcodebuild's parallel jobs. The library's Package.resolved is put back if package resolution rewrote it, and the md5 of
# every other file of ../D1 is compared before and after (the library is read, never changed). Derived data, package
# checkouts and logs: _work/ (git-ignored).
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
if (( MAC )); then
  TAG=mac; DEST='generic/platform=macOS'; PRODUCTS=Release; EXTRA=(ARCHS=arm64)
else
  TAG=ios; DEST='generic/platform=iOS'; PRODUCTS=Release-iphoneos; EXTRA=()
  [ "${D1_PROVISIONING_UPDATES:-0}" = 1 ] && EXTRA+=(-allowProvisioningUpdates)
fi
DD=$W/dd_$TAG
LOG=$W/xcodebuild_$TAG.log
mkdir -p $W
[ -f $LIB/Package.swift ] || { echo "no D1 package at $LIB"; exit 1; }

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
    echo "[$(date '+%H:%M:%S')] a measurement window is still open after $((SECONDS - t0)) s: not building"; exit 4
  fi
  echo "[$(date '+%H:%M:%S')] waiting: a measurement window is open ($(head -c 120 $LOCK))"
  sleep 30
done

libsums() { (cd $LIB && find . -type f -not -path './.build/*' -not -path './.swiftpm/*' | LC_ALL=C sort | xargs md5 -r) }
RES=$LIB/Package.resolved
SAVED=$W/D1.Package.resolved.before_build
SUM_BEFORE=""
if [ -f $RES ]; then cp -p $RES $SAVED; SUM_BEFORE=$(md5 -q $RES); fi
libsums > $W/d1_lib_md5_before_$TAG.txt

cd $DIR
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 1; }
echo "[$(date '+%H:%M:%S')] xcodebuild $TAG: $DEST, Release ${EXTRA[*]} (-jobs ${D1_BUILD_JOBS:-4})"
t0=$SECONDS
xcodebuild -project D1Demo.xcodeproj -scheme D1Demo -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $W/spm -jobs ${D1_BUILD_JOBS:-4} "${EXTRA[@]}" build > $LOG 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] xcodebuild returned $rc after $((SECONDS - t0)) s"

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
  echo "../D1: CHANGED during the build:"; diff $W/d1_lib_md5_before_$TAG.txt $W/d1_lib_md5_after_$TAG.txt | head -10
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  grep -iE "provision|signing|account|capabilit|entitlement" $LOG | grep -iE "error|warning|note" | sort -u | head -12 \
    > $W/xcodebuild_${TAG}_signing_errors.txt
  [ -s $W/xcodebuild_${TAG}_signing_errors.txt ] && { echo "signing lines (also in $W/xcodebuild_${TAG}_signing_errors.txt):"; cat $W/xcodebuild_${TAG}_signing_errors.txt; }
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/D1Demo.app
[ -d $APP ] || { echo "no app at $APP"; exit 1; }
if (( MAC )); then echo $APP > $W/app_path_mac.txt; else echo $APP > $W/app_path.txt; fi
echo "app: $APP ($(du -sh $APP | cut -f1))"
echo "compiler warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
echo "entitlements: $(codesign -d --entitlements - $APP 2>/dev/null | grep -cE 'increased-memory') increased-memory-limit keys"
if [ -f $APP/embedded.mobileprovision ]; then
  P=$W/embedded_$TAG.plist
  security cms -D -i $APP/embedded.mobileprovision > $P 2>/dev/null
  echo "profile: $(/usr/libexec/PlistBuddy -c 'Print :Name' $P 2>/dev/null) (application-identifier" \
    "$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:application-identifier' $P 2>/dev/null), increased-memory-limit" \
    "$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:com.apple.developer.kernel.increased-memory-limit' $P 2>/dev/null || echo absent))"
  echo "profile devices include the 18 Pro: $(/usr/libexec/PlistBuddy -c 'Print :ProvisionedDevices' $P 2>/dev/null | grep -c 00008160-000038CA02C00036)"
fi
