#!/bin/zsh
# Build DeciderVisionGate: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ). Build only: no
# install, no launch.
#   ./_build.sh             generic iOS            -> _work/app_path.txt
#   ./_build.sh --mac       macOS (arm64)          -> _work/app_path_mac.txt
#   ./_build.sh --no-iml    without DeciderVisionGate.entitlements (combines with --mac)
# The target asks for com.apple.developer.kernel.increased-memory-limit. The bundle id is new: until its App ID has that
# capability (Xcode, the team's account, one ⌘B on the generated project), the only profile of the team that lists the
# iPhone 18 Pro is the wildcard one, which cannot carry it, and the build fails with "No Accounts" / "doesn't include the
# Increased Memory Limit capability"; --no-iml builds without the entitlement (the default memory limit, where the
# decoder's cold JIT specialization aborts with std::bad_alloc). The library comes by path (project.yml:
# ../DeciderVision). A first build that resolves packages can write the path dependency's Package.resolved (memory:
# reference_xcodebuild_local_package_resolved): this script copies it before the build and, when the build changed it,
# keeps the diff in _work/ and puts the copy back.
# Derived data, package checkouts and logs: _work/ (git-ignored).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${DV_WORK:-$DIR/_work}
LIB=${DIR:h}/DeciderVision
MAC=0; NOIML=0
for a in "$@"; do
  case $a in
    --mac) MAC=1 ;;
    --no-iml) NOIML=1 ;;
    *) echo "unknown option $a (--mac, --no-iml)"; exit 1 ;;
  esac
done
if (( MAC )); then
  TAG=mac; DEST='generic/platform=macOS'; PRODUCTS=Release; EXTRA=(ARCHS=arm64)
else
  TAG=ios; DEST='generic/platform=iOS'; PRODUCTS=Release-iphoneos; EXTRA=()
fi
(( NOIML )) && EXTRA+=(CODE_SIGN_ENTITLEMENTS=)
DD=$W/dd_$TAG
LOG=$W/xcodebuild_$TAG.log
mkdir -p $W
[ -f $LIB/Package.swift ] || { echo "no DeciderVision package at $LIB"; exit 1; }
RES=$LIB/Package.resolved
SAVED=$W/DeciderVision.Package.resolved.before_build
SUM_BEFORE=""
if [ -f $RES ]; then cp -p $RES $SAVED; SUM_BEFORE=$(md5 -q $RES); fi

cd $DIR
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 1; }
echo "xcodebuild $TAG: $DEST, Release ${EXTRA[*]}$( (( NOIML )) && echo ' (no entitlements)')"
xcodebuild -project DeciderVisionGate.xcodeproj -scheme DeciderVisionGate -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $W/spm "${EXTRA[@]}" build > $LOG 2>&1
rc=$?

# the library's Package.resolved: put the copy back when this build changed it
SUM_AFTER=$(md5 -q $RES 2>/dev/null)
if [ "$SUM_AFTER" != "$SUM_BEFORE" ]; then
  D=$W/decidervision_package_resolved_$(date +%Y%m%d-%H%M%S).diff
  diff -u $SAVED $RES > $D 2>&1
  if [ -n "$SUM_BEFORE" ]; then cp -p $SAVED $RES; else rm -f $RES; fi
  echo "DeciderVision/Package.resolved: this build changed it; diff kept in $D, file restored ($(md5 -q $RES 2>/dev/null || echo absent))"
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/DeciderVisionGate.app
[ -d $APP ] || { echo "no app at $APP"; exit 1; }
case $TAG in
  ios) echo $APP > $W/app_path.txt ;;
  mac) echo $APP > $W/app_path_mac.txt ;;
esac
echo "app: $APP ($(du -sh $APP | cut -f1))"
echo "compiler warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
echo "other compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | head -5
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
codesign -d --entitlements - $APP 2>/dev/null | grep -E "increased-memory|application-identifier" | head -3
