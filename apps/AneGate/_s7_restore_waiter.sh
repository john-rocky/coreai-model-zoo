#!/bin/zsh
# S7 close-out (2026-09-18 18:4x): the phone left the USB bus while the cand gate app (C + A) was installed under the
# PipelinedBench bundle id. Wait for the phone (live container listing, poll 30 s, cap 12 h), restore PipelinedBench,
# verify Documents/models, release the device hold. No gate run.
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; PHONE_UDID=00008150-0018713A0207801C
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$DIR/../..
cd $DIR
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
live() {
  xcrun devicectl list devices 2>/dev/null | grep "$PHONE_UDID" | grep -qE '[[:space:]]available[[:space:]]' || return 1
  local out; out=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents 2>&1)
  echo "$out" | grep -q ERROR && return 1
  return 0
}
say "waiting for the phone (list State = available AND live container listing), cap 12 h"
T_END=$(( $(date +%s) + 12*3600 ))
until live; do
  [ $(date +%s) -ge $T_END ] && { say "ERROR phone never came back (12 h cap); hold LEFT in place, PipelinedBench NOT restored"; exit 1; }
  sleep 30
done
say "phone is back; 60 s settle, then restore PipelinedBench"
sleep 60
for attempt in 1 2 3 4 5; do
  OUT=$(xcrun devicectl device install app --device $UDID $ROOT/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
  echo "$OUT" | grep -q installationURL && { say "PipelinedBench restored"; break; }
  say "restore attempt $attempt failed: $(echo "$OUT" | grep -m1 -i error | cut -c1-120)"; sleep 30
done
N=$(xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier com.coreai.pipelinedbench --subdirectory Documents/models 2>/dev/null | grep -c .)
say "PipelinedBench Documents/models entries: $N (S6 close: 138)"
if echo "$OUT" | grep -q installationURL; then
  rm -f $ROOT/ondevice/.device_hold; say "device released (hold removed)"
else
  say "ERROR restore failed 5x; hold LEFT in place"; exit 2
fi
