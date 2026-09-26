#!/bin/zsh
# Build DecideGate for the iPhone: xcodegen generate + xcodebuild Release, generic iOS, automatic signing
# (team MFN25KNUGJ, as N3DGate). Build only: no install, no launch.
#   ./_build.sh
# Derived data, logs and the .app path go to the conversion's _work/ (git-ignored):
#   _work/decidegate_build/  _work/decidegate_xcodegen.log  _work/decidegate_xcodebuild.log  _work/decidegate_app_path.txt
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
WORK=${DECIDE_WORK:-${DIR:h:h}/conversion/gliner25_decide/_work}
DD=$WORK/decidegate_build
mkdir -p $WORK
cd $DIR
xcodegen generate > $WORK/decidegate_xcodegen.log 2>&1 || { cat $WORK/decidegate_xcodegen.log; exit 1; }
xcodebuild -project DecideGate.xcodeproj -scheme DecideGate -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath $DD build > $WORK/decidegate_xcodebuild.log 2>&1
rc=$?
grep -E "BUILD (SUCCEEDED|FAILED)" $WORK/decidegate_xcodebuild.log
if [ $rc -ne 0 ]; then
  grep -E "error:" $WORK/decidegate_xcodebuild.log | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/Release-iphoneos/DecideGate.app
echo $APP > $WORK/decidegate_app_path.txt
echo "app: $APP ($(du -sh $APP | cut -f1))"
# compiler / linker warnings, then any other tool line that says "warning:" (e.g. appintentsmetadataprocessor)
echo "compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $WORK/decidegate_xcodebuild.log | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $WORK/decidegate_xcodebuild.log | sort -u | head -20
grep -E 'warning:' $WORK/decidegate_xcodebuild.log | grep -vE ': warning: |^ld: warning' | sed -E 's/^[0-9-]+ [0-9:.]+ //' | sort -u \
  | sed 's/^/other tool line: /' | head -5
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
