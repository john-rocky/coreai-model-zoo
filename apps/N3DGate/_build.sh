#!/bin/zsh
# Build N3DGate for the iPhone: xcodegen generate + xcodebuild Release, generic iOS, automatic signing
# (team MFN25KNUGJ, as coreai-audio). Build only: no install, no launch.
#   ./_build.sh
# Derived data, logs and the .app path go to the conversion's _work/ (git-ignored):
#   _work/n3dgate_build/  _work/n3dgate_xcodegen.log  _work/n3dgate_xcodebuild.log  _work/n3dgate_app_path.txt
set -u
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
DIR=${0:A:h}
WORK=${N3D_WORK:-${DIR:h:h}/conversion/nemotron3_diar/_work}
DD=$WORK/n3dgate_build
mkdir -p $WORK
cd $DIR
xcodegen generate > $WORK/n3dgate_xcodegen.log 2>&1 || { cat $WORK/n3dgate_xcodegen.log; exit 1; }
xcodebuild -project N3DGate.xcodeproj -scheme N3DGate -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath $DD build > $WORK/n3dgate_xcodebuild.log 2>&1
rc=$?
grep -E "BUILD (SUCCEEDED|FAILED)" $WORK/n3dgate_xcodebuild.log
if [ $rc -ne 0 ]; then
  grep -E "error:" $WORK/n3dgate_xcodebuild.log | sort -u | head -20
  exit 1
fi
APP=$DD/Build/Products/Release-iphoneos/N3DGate.app
echo $APP > $WORK/n3dgate_app_path.txt
echo "app: $APP ($(du -sh $APP | cut -f1))"
# compiler / linker warnings, then any other tool line that says "warning:" (e.g. appintentsmetadataprocessor)
echo "compiler/linker warnings (unique): $(grep -E ': warning: |^ld: warning' $WORK/n3dgate_xcodebuild.log | sort -u | wc -l | tr -d ' ')"
grep -E ': warning: |^ld: warning' $WORK/n3dgate_xcodebuild.log | sort -u | head -20
grep -E 'warning:' $WORK/n3dgate_xcodebuild.log | grep -vE ': warning: |^ld: warning' | sed -E 's/^[0-9-]+ [0-9:.]+ //' | sort -u \
  | sed 's/^/other tool line: /' | head -5
codesign -dv $APP 2>&1 | grep -E "^(Identifier|TeamIdentifier)="
