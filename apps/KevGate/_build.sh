#!/bin/zsh
# Build KevGate: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ), no entitlements. Build only:
# no install, no launch. Copied from apps/DeciderVisionGate/_build.sh (zoo main 2d214b3).
#   ./_build.sh             generic iOS            -> _work/app_path.txt
#   ./_build.sh --mac       macOS (arm64)          -> _work/app_path_mac.txt
# The bundle id is new and the team's only profile that lists the iPhone 18 Pro and fits a new id is the wildcard one,
# which cannot carry com.apple.developer.kernel.increased-memory-limit: the app is built without entitlements and runs at
# the default memory limit (it records os_proc_available_memory). The library comes by path (project.yml: ../Kev). A
# first build that resolves packages can write the path dependency's Package.resolved:
# this script copies it before the build and, when the build changed it,
# keeps the diff in _work/ and puts the copy back; the md5 of every other file of ../Kev (not .build / .swiftpm) is
# compared before and after.
# KEV_BUILD_WAIT_TAG=<text>: first wait (30 s steps, KEV_BUILD_WAIT_CAP s, default 2400) while ~/code/coreai/_GPU_LOCK
# contains <text> (a Mac timing window another session holds: a full-CPU build would disturb its numbers).
# Derived data, package checkouts and logs: _work/ (git-ignored).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${KEV_WORK:-$DIR/_work}
LIB=${DIR:h}/Kev
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
fi
DD=$W/dd_$TAG
LOG=$W/xcodebuild_$TAG.log
mkdir -p $W
[ -f $LIB/Package.swift ] || { echo "no Kev package at $LIB"; exit 1; }

if [ -n "${KEV_BUILD_WAIT_TAG:-}" ]; then
  LOCK=$HOME/code/coreai/_GPU_LOCK
  t0=$SECONDS
  while grep -qF -- "$KEV_BUILD_WAIT_TAG" $LOCK 2>/dev/null; do
    if (( SECONDS - t0 >= ${KEV_BUILD_WAIT_CAP:-2400} )); then
      echo "[$(date '+%H:%M:%S')] the lock still names '$KEV_BUILD_WAIT_TAG' after $((SECONDS - t0)) s: not building"; exit 4
    fi
    echo "[$(date '+%H:%M:%S')] waiting: the lock names '$KEV_BUILD_WAIT_TAG' ($(head -c 120 $LOCK))"
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] the lock does not name '$KEV_BUILD_WAIT_TAG' (waited $((SECONDS - t0)) s): building"
fi

libsums() { (cd $LIB && find . -type f -not -path './.build/*' -not -path './.swiftpm/*' | LC_ALL=C sort | xargs md5 -r) }
RES=$LIB/Package.resolved
SAVED=$W/Kev.Package.resolved.before_build
SUM_BEFORE=""
if [ -f $RES ]; then cp -p $RES $SAVED; SUM_BEFORE=$(md5 -q $RES); fi
libsums > $W/kev_lib_md5_before_$TAG.txt

cd $DIR
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 1; }
echo "[$(date '+%H:%M:%S')] xcodebuild $TAG: $DEST, Release ${EXTRA[*]} (no entitlements)"
t0=$SECONDS
xcodebuild -project KevGate.xcodeproj -scheme KevGate -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $W/spm "${EXTRA[@]}" build > $LOG 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] xcodebuild returned $rc after $((SECONDS - t0)) s"

# the library's Package.resolved: put the copy back when this build changed it
SUM_AFTER=$(md5 -q $RES 2>/dev/null)
if [ "$SUM_AFTER" != "$SUM_BEFORE" ]; then
  D=$W/kev_package_resolved_$(date +%Y%m%d-%H%M%S).diff
  diff -u $SAVED $RES > $D 2>&1
  if [ -n "$SUM_BEFORE" ]; then cp -p $SAVED $RES; else rm -f $RES; fi
  echo "Kev/Package.resolved: this build changed it; diff kept in $D, file restored ($(md5 -q $RES 2>/dev/null || echo absent))"
fi
libsums > $W/kev_lib_md5_after_$TAG.txt
if cmp -s $W/kev_lib_md5_before_$TAG.txt $W/kev_lib_md5_after_$TAG.txt; then
  echo "../Kev: every file md5-equal before and after the build ($(grep -c . $W/kev_lib_md5_after_$TAG.txt) files)"
else
  echo "../Kev: CHANGED by the build:"; diff $W/kev_lib_md5_before_$TAG.txt $W/kev_lib_md5_after_$TAG.txt | head -10
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/KevGate.app
[ -d $APP ] || { echo "no app at $APP"; exit 1; }
case $TAG in
  ios) echo $APP > $W/app_path.txt ;;
  mac) echo $APP > $W/app_path_mac.txt ;;
esac
echo "app: $APP ($(du -sh $APP | cut -f1))"
echo "compiler warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
echo "other compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | head -8
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
echo "entitlements: $(codesign -d --entitlements - $APP 2>/dev/null | grep -cE 'increased-memory') increased-memory-limit keys"
