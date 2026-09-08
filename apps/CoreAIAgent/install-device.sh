#!/bin/zsh
# Build (signed, Release) and install CoreAIAgent on the paired iPhone 17 Pro, then optionally run
# the headless self-test.   ./install-device.sh          # build + install
#                           ./install-device.sh selftest # + launch with AGENT_SELFTEST=1 and stream the log
set -e
cd "$(dirname "$0")"
UDID=${UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}
BID=com.daisukemajima.llmbench014   # borrowed provisioned app id — see README
xcodegen generate >/dev/null
xcodebuild -project CoreAIAgent.xcodeproj -scheme CoreAIAgent -configuration Release \
  -destination generic/platform=iOS -derivedDataPath build build 2>&1 | grep -E "error:|BUILD (SUCCEEDED|FAILED)"
xcrun devicectl device install app --device "$UDID" build/Build/Products/Release-iphoneos/CoreAIAgent.app 2>&1 | grep -iE "installed|error"
if [[ "$1" == "selftest" ]]; then
  LOG=_selftest_$(date +%Y%m%d_%H%M%S).log
  xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
    --environment-variables '{"AGENT_SELFTEST":"1"}' "$BID" > "$LOG" 2>&1 &
  LP=$!
  for i in {1..120}; do grep -qE "\[selftest\] (DONE|ERROR)|terminated" "$LOG" && break; sleep 5; done
  kill $LP 2>/dev/null
  grep -E "selftest|terminated" "$LOG"
fi
