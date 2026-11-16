#!/bin/zsh
# S5 pause (2026-09-17 12:5x): let the 6-bit remainder finish on the phone, pull it, restore PipelinedBench, score
# what exists (4-bit, 6-bit, Mac bf16, Mac 6-bit sim). The int8 arm is left for the next resume (task app must be reinstalled).
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
UDID=A6F3E849-1947-5202-9AD1-9C881CA58EEF; ID=com.coreai.pipelinedbench; T=task_6bit_rest_124305
DIR=$(cd "$(dirname "$0")" && pwd); cd $DIR; OUT=$DIR/_queue_s5_task.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a $OUT; }
say "S5 pause: waiting for the 6-bit remainder to finish on the phone"
for i in $(seq 1 90); do
  xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $ID --source Documents/sustain/$T.log --destination _device_task_6bit_rest.log >/dev/null 2>&1
  grep -qE "TASK_DONE|FATAL" _device_task_6bit_rest.log && break; sleep 10
done
grep -E "TASK_DONE|FATAL" _device_task_6bit_rest.log | cut -c1-160 | tee -a $OUT
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer --domain-identifier $ID --source Documents/sustain/${T}_answers.log --destination _device_task_6bit_rest_answers.log >/dev/null 2>&1
cat _device_task_6bit_rest_answers.log >> _device_task_6bit_answers.log
say "6-bit answers total: $(grep -c '^{' _device_task_6bit_answers.log) (remainder $(grep -c '^{' _device_task_6bit_rest_answers.log))"
say "restoring PipelinedBench"
for t in 1 2 3; do
  R=$(xcrun devicectl device install app --device $UDID ~/code/coreai/ondevice/PipelinedBench/build/Build/Products/Release-iphoneos/PipelinedBench.app 2>&1)
  echo "$R" | grep -q installationURL && { echo "$R" | grep installationURL | tee -a $OUT; break; }
  say "install attempt $t failed: $(echo "$R" | grep -m1 -i error | cut -c1-120)"; sleep 30
done
xcrun devicectl device info files --device $UDID --domain-type appDataContainer --domain-identifier $ID --subdirectory Documents/models 2>&1 | head -1 | tee -a $OUT
python3 score_gsm8k.py --arm 4bit_ane=_device_task_4bit_answers.log --arm 6bit_ane=_device_task_6bit_answers.log \
  --arm hf_bf16=_ref_hf_bf16_mps_answers.log --arm sim6bit_mac=_sim6bit_bf16_mps_answers.log --json _gsm8k_s5_scores_partial.json 2>&1 | tee -a $OUT
say "S5 pause end — phone free (int8 arm pending)"
