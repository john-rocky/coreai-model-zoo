#!/bin/zsh
# Build TimestralGate: xcodegen generate + xcodebuild Release. Build only: no install, no launch.
#   ./_build.sh ios    the iPhone app, automatic signing (team MFN25KNUGJ; the team's wildcard profile)  -> _work/app_path.txt
#   ./_build.sh mac    the Mac command-line tool (ad-hoc signed)                                        -> _work/tool_path.txt
# Derived data and logs: _work/ (git-ignored). xcodebuild is a heavy job: run it inside a Mac window (quiet_hold).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${TIMESTRAL_WORK:-$DIR/_work}
mkdir -p $W
cd $DIR || exit 2
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 2; }
case ${1:-} in
  mac)
    LOG=$W/xcodebuild_mac.log
    xcodebuild -project TimestralGate.xcodeproj -scheme TimestralGateMac -configuration Release \
      -derivedDataPath $W/dd_mac build > $LOG 2>&1
    rc=$?
    grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
    [ $rc -eq 0 ] || { grep -E "error:" $LOG | sort -u | head -30; exit 1; }
    P=$W/dd_mac/Build/Products/Release/TimestralGateMac
    [ -x $P ] || { echo "no tool at $P"; exit 1; }
    echo $P > $W/tool_path.txt
    echo "tool: $P" ;;
  ios)
    LOG=$W/xcodebuild_ios.log
    xcodebuild -project TimestralGate.xcodeproj -scheme TimestralGate -configuration Release -destination 'generic/platform=iOS' \
      -derivedDataPath $W/dd_ios -allowProvisioningUpdates build > $LOG 2>&1
    rc=$?
    grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
    [ $rc -eq 0 ] || { grep -E "error:" $LOG | sort -u | head -30; exit 1; }
    A=$W/dd_ios/Build/Products/Release-iphoneos/TimestralGate.app
    [ -d $A ] || { echo "no app at $A"; exit 1; }
    echo $A > $W/app_path.txt
    echo "app: $A ($(du -sh $A | cut -f1))"
    codesign -dv $A 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
    grep -E 'Provisioning Profile:|Signing Identity:' $LOG | sort -u ;;
  *)
    echo "usage: _build.sh ios|mac"; exit 2 ;;
esac
echo "warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
exit 0
