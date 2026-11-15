#!/bin/zsh
# Build FunASRGate: xcodegen generate + xcodebuild Release, automatic signing (team MFN25KNUGJ, as DecideGate).
# Build only: no install, no launch.
#   ./_build.sh                  generic iOS                     -> _work/app_path.txt
#   ./_build.sh --mac            macOS (arm64)                   -> _work/app_path_mac.txt
#   ./_build.sh --mac --public   macOS, public kit API only      -> _work/app_path_mac_public.txt
# Every build but --public compiles all modules with ENABLE_TESTABILITY=YES: the gate reads the generated ids and the
# per-stage times through `@testable import CoreAIKit` (KitFunASRModel.transcribeWindows), as the kit's
# FunASRSmokeTests do. --public leaves testability off everywhere and builds the gate's text-and-wall-time path
# (FUNASR_PUBLIC_API): the control for what -enable-testing costs in the numbers.
# The kit comes by path (project.yml). A build that resolves packages can write the kit checkout's Package.resolved:
# this script checks it before and after, and when the build
# changed a file that was clean before, it keeps the diff in _work/ and restores the file (git checkout).
# Derived data, package checkouts and logs: _work/ (git-ignored).
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
W=${FUNASR_WORK:-$DIR/_work}
KIT=${FUNASR_KIT:-${DIR:h:h:h:h}/coreai-kit-funasr-wt/coreai-kit}
MAC=0; PUBLIC=0
for a in "$@"; do
  case $a in
    --mac) MAC=1 ;;
    --public) PUBLIC=1 ;;
    *) echo "unknown option $a (--mac, --public)"; exit 1 ;;
  esac
done
(( PUBLIC && !MAC )) && { echo "--public is a macOS build: ./_build.sh --mac --public"; exit 1; }
if (( MAC )); then
  if (( PUBLIC )); then TAG=mac_public; else TAG=mac; fi
  DEST='generic/platform=macOS'
  PRODUCTS=Release
  EXTRA=(ARCHS=arm64)
else
  TAG=ios
  DEST='generic/platform=iOS'
  PRODUCTS=Release-iphoneos
  EXTRA=()
fi
if (( PUBLIC )); then
  EXTRA+=('SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) FUNASR_PUBLIC_API')
else
  EXTRA+=(ENABLE_TESTABILITY=YES)
fi
DD=$W/dd_$TAG
LOG=$W/xcodebuild_$TAG.log
mkdir -p $W
[ -f $KIT/Package.swift ] || { echo "no kit at $KIT (set FUNASR_KIT, and the path in project.yml)"; exit 1; }
[ -d $KIT/Sources/CoreAIKit/FunASR ] || { echo "the kit at $KIT has no Sources/CoreAIKit/FunASR"; exit 1; }
RES_BEFORE=$(git -C $KIT status --porcelain -- Package.resolved 2>/dev/null)
SUM_BEFORE=$(md5 -q $KIT/Package.resolved 2>/dev/null)

cd $DIR
xcodegen generate > $W/xcodegen.log 2>&1 || { cat $W/xcodegen.log; exit 1; }
echo "xcodebuild $TAG: $DEST, Release, ${EXTRA[*]}"
xcodebuild -project FunASRGate.xcodeproj -scheme FunASRGate -configuration Release -destination "$DEST" \
  -derivedDataPath $DD -clonedSourcePackagesDirPath $W/spm "${EXTRA[@]}" build > $LOG 2>&1
rc=$?

# the kit's Package.resolved: restore it when this build changed a clean file
SUM_AFTER=$(md5 -q $KIT/Package.resolved 2>/dev/null)
if [ "$SUM_AFTER" != "$SUM_BEFORE" ]; then
  D=$W/kit_package_resolved_$(date +%Y%m%d-%H%M%S).diff
  git -C $KIT diff -- Package.resolved > $D
  if [ -z "$RES_BEFORE" ]; then
    git -C $KIT checkout -- Package.resolved && echo "kit Package.resolved: this build changed it; diff kept in $D, file restored"
  else
    echo "kit Package.resolved: changed by this build but it had local changes before ($RES_BEFORE): left as is, diff in $D"
  fi
fi

grep -E "BUILD (SUCCEEDED|FAILED)" $LOG | tail -1
if [ $rc -ne 0 ]; then
  grep -E "error:" $LOG | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/$PRODUCTS/FunASRGate.app
[ -d $APP ] || { echo "no app at $APP"; exit 1; }
case $TAG in
  ios) echo $APP > $W/app_path.txt ;;
  mac) echo $APP > $W/app_path_mac.txt ;;
  mac_public) echo $APP > $W/app_path_mac_public.txt ;;
esac
echo "app: $APP ($(du -sh $APP | cut -f1))"
# warnings in this app's sources (the packages' own are listed apart, first 5)
echo "compiler warnings in Sources/ (unique): $(grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | wc -l | tr -d ' ')"
grep -E "^$DIR/Sources/.*: warning: " $LOG | sort -u | head -20
echo "other compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $LOG | grep -v "^$DIR/Sources/" | sort -u | head -5
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
