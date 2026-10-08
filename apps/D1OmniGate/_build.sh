#!/bin/zsh
# Build D1OmniGate: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ), no entitlements. Build
# only: no install, no launch. Copied from apps/KevGate/_build.sh with the lane's names.
#   ./_build.sh             generic iOS            -> $D1_WORK/app_path.txt
#   ./_build.sh --mac       macOS (arm64)          -> $D1_WORK/app_path_mac.txt
# D1_WORK (default ~/code/coreai/_d1_omni/device) holds derived data, package checkouts and logs (build/), outside the
# repository. The bundle id is new and the team's only profile that lists the iPhone 18 Pro and fits a new id is the
# wildcard one, which cannot carry com.apple.developer.kernel.increased-memory-limit: the app is built without
# entitlements and runs at the default memory limit (it records os_proc_available_memory). The library comes by path
# (project.yml: ../D1Omni). A first build that resolves packages can write the path dependency's Package.resolved
# (memory: reference_xcodebuild_local_package_resolved): this script copies it before the build and, when the build
# changed it, keeps the diff in $D1_WORK/build/ and puts the copy back; the md5 of every other file of ../D1Omni (not
# .build / .swiftpm) is compared before and after. A full-CPU build disturbs another lane's Mac timing window: run it
# through ~/code/standup/tools/quiet/quiet_wait.py --.
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${D1_WORK:-$HOME/code/coreai/_d1_omni/device}
B=$W/build
LIB=${DIR:h}/D1Omni
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
DD=$B/dd_$TAG
LOG=$B/xcodebuild_$TAG.log
mkdir -p $B
[ -f $LIB/Package.swift ] || { echo "no D1Omni package at $LIB"; exit 1; }

libsums() { (cd $LIB && find . -type f -not -path './.build/*' -not -path './.swiftpm/*' | LC_ALL=C sort | xargs md5 -r) }
RES=$LIB/Package.resolved
SAVED=$B/D1Omni.Package.resolved.before_build
SUM_BEFORE=""
if [ -f $RES ]; then cp -p $RES $SAVED; SUM_BEFORE=$(md5 -q $RES); fi
libsums > $B/d1omni_lib_md5_before_$TAG.txt

cd $DIR
xcodegen generate > $B/xcodegen.log 2>&1 || { cat $B/xcodegen.log; exit 1; }
echo "[$(date '+%H:%M:%S')] xcodebuild $TAG: $DEST, Release ${EXTRA[*]} (no entitlements)"
t0=$SECONDS
xcodebuild -project D1OmniGate.xcodeproj -scheme D1OmniGate -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $B/spm "${EXTRA[@]}" build > $LOG 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] xcodebuild returned $rc after $((SECONDS - t0)) s"

# the library's Package.resolved: put the copy back when this build changed it
SUM_AFTER=$(md5 -q $RES 2>/dev/null)
if [ "$SUM_AFTER" != "$SUM_BEFORE" ]; then
  D=$B/d1omni_package_resolved_$(date +%Y%m%d-%H%M%S).diff
  diff -u $SAVED $RES > $D 2>&1
  if [ -n "$SUM_BEFORE" ]; then cp -p $SAVED $RES; else rm -f $RES; fi
  echo "D1Omni/Package.resolved: this build changed it; diff kept in $D, file restored ($(md5 -q $RES 2>/dev/null || echo absent))"
fi
libsums > $B/d1omni_lib_md5_after_$TAG.txt
if cmp -s $B/d1omni_lib_md5_before_$TAG.txt $B/d1omni_lib_md5_after_$TAG.txt; then
  echo "../D1Omni: every file md5-equal before and after the build ($(grep -c . $B/d1omni_lib_md5_after_$TAG.txt) files)"
else
  echo "../D1Omni: CHANGED by the build:"; diff $B/d1omni_lib_md5_before_$TAG.txt $B/d1omni_lib_md5_after_$TAG.txt | head -10
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/D1OmniGate.app
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
