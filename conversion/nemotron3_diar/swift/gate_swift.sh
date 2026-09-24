#!/bin/zsh
# Round-3 gates of the Swift host (n3d-selftest) on this Mac, against the golden files of
# ../export_golden.py. Logs + JSON go to ../_work/r3/<part>.{log,json} (N3D_GATE_OUT=<dir> to write elsewhere).
#
#   ./gate_swift.sh                     # units,mel,s1,s2,s2chunks,s3,s4
#   ./gate_swift.sh bench               # --bench 100 on the fp16 GPU bundle (takes ~/code/coreai/_GPU_LOCK if free)
#
#   units     teacher-forced N3DSpeakerCache.update vs the NumPy host, bit for bit (35 units, 17 compressions)
#   mel       Swift log-mel vs mel_frontend.py (streaming chunks 0 / 1 / last and the offline mel), both fixtures
#   s1        packed parity: rows fed to the graph at steps 0, 1, 54, 78, 128 (+ last) of 97.6 s low_latency vs
#             host_loop.py --engine eager --mel host (fp32 bundle on cpuOnly = the eager trajectory), and the fp16
#             GPU run vs the same steps of host_loop.py on the GPU
#   s2        closed loop, fp16 GPU, Swift mel: 2 fixtures x ll / vll / ull / offline vs transformers fp32, and
#             each run's logits vs host_loop.py's on the same bundle (golden/pygpu)
#   s2chunks  s2 with the Python host's chunk rows (--chunks): the loop alone, mel taken out
#   s3        --poison no-compress / no-pop on 97.6 s low_latency: must FAIL the 99.9 % bar
#   s4        fp32 bundle on cpuOnly, 97.6 s low_latency (the Swift side of round 2's (a))
set -u
cd "${0:A:h}"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
BIN=.build/release/n3d-selftest
A=../_work/artifacts
G=../_work/golden
FX=${N3D_FIXTURES:-$HOME/code/coreai/_n3d/fixtures}
OUT=${N3D_GATE_OUT:-../_work/r3}
mkdir -p $OUT
S16=$A/n3d_streaming_float16.aimodel
S32=$A/n3d_streaming_float32.aimodel
O16=$A/n3d_offline_float16.aimodel
FIXTURES=(diarization_example test_multispk)
TAGS=(ll vll ull offline)

run() {
  local name=$1; shift
  $BIN "$@" --json $OUT/$name.json > $OUT/$name.log 2>&1
  local rc=$?
  echo "== $name (exit $rc)"
  grep -E "^\[n3d\] (vs |packed|cache units|mel |bench|chunk rows|poison|ERROR)" $OUT/$name.log
}

parts=${1:-units,mel,s1,s2,s2chunks,s3,s4}
for part in ${(s:,:)parts}; do
  case $part in
    units)
      run units --assets $A --cache-golden $G ;;
    mel)
      for f in $FIXTURES; do run mel_$f --assets $A --wav $FX/${f}_16k.wav --mel-golden $G; done ;;
    s1)
      run s1_cpu_fp32 --assets $A --bundle $S32 --unit cpuOnly --wav $FX/diarization_example_16k.wav --mode ll \
        --dump-packed $OUT/packed_cpu_fp32 --packed-golden $G/diarization_example_ll_packed_step
      run s1_gpu_fp16_vs_eager --assets $A --bundle $S16 --unit gpu --wav $FX/diarization_example_16k.wav --mode ll \
        --dump-packed $OUT/packed_gpu_fp16 --packed-golden $G/diarization_example_ll_packed_step
      run s1_gpu_fp16_vs_pygpu --assets $A --bundle $S16 --unit gpu --wav $FX/diarization_example_16k.wav --mode ll \
        --dump-packed $OUT/packed_gpu_fp16 --packed-golden $G/pygpu/diarization_example_ll_packed_step ;;
    s2)
      for f in $FIXTURES; do for t in $TAGS; do
        b=$S16; [[ $t == offline ]] && b=$O16
        run s2_${f}_$t --assets $A --bundle $b --unit gpu --wav $FX/${f}_16k.wav --mode $t \
          --golden $G/${f}_${t}_probs.f32le --compare-logits $G/pygpu/${f}_${t}_logits.f32le
      done; done ;;
    s2chunks)
      for f in $FIXTURES; do for t in $TAGS; do
        b=$S16; [[ $t == offline ]] && b=$O16
        run s2chunks_${f}_$t --assets $A --bundle $b --unit gpu --wav $FX/${f}_16k.wav --mode $t --chunks $G \
          --golden $G/${f}_${t}_probs.f32le --compare-logits $G/pygpu/${f}_${t}_logits.f32le
      done; done ;;
    s3)
      for p in no-compress no-pop; do
        run s3_$p --assets $A --bundle $S16 --unit gpu --wav $FX/diarization_example_16k.wav --mode ll --poison $p \
          --golden $G/diarization_example_ll_probs.f32le
      done ;;
    s4)
      run s4_cpu_fp32 --assets $A --bundle $S32 --unit cpuOnly --wav $FX/diarization_example_16k.wav --mode ll \
        --golden $G/diarization_example_ll_probs.f32le ;;
    bench)
      run bench_gpu_fp16 --assets $A --bundle $S16 --unit gpu --wav $FX/diarization_example_16k.wav --mode ll \
        --golden $G/diarization_example_ll_probs.f32le --bench 100 --lock $HOME/code/coreai/_GPU_LOCK ;;
    *) echo "unknown part $part"; exit 2 ;;
  esac
done
