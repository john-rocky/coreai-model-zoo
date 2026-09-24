#!/bin/zsh
# Gather what N3DGate reads on the phone into _work/device_stage/N3DAssets/ (APFS clones: no extra disk), then
# write N3DAssets/MD5SUMS (every file but itself). ../_install.sh pushes the directory as it is.
#   ./_stage.sh
# Needs, from conversion/nemotron3_diar (see its README): the h18p AOT bundles
#   _work/aot/ios_gpu/n3d_streaming_float16.h18p.aimodelc           aot_compile.py
#   _work/aot/offline_ios_gpu/n3d_offline_float16.h18p.aimodelc     aot_compile.py --bundle ..offline.. --tag offline --targets ios_gpu
#   _work/aot/ios_ane/n3d_streaming_float16.h18p.aimodelc           aot_compile.py (optional: the ANE stage is skipped without it)
# the four f32le constants + metadata.ship.json (_work/artifacts, export_n3d.py [--metadata --ship]), the golden files
# (_work/golden, export_golden.py)
# and the two fixture wavs (N3D_FIXTURES, default ~/code/coreai/_n3d/fixtures).
set -euo pipefail
DIR=${0:A:h}
C=${DIR:h:h}/conversion/nemotron3_diar
W=${N3D_WORK:-$C/_work}
FX=${N3D_FIXTURES:-$HOME/code/coreai/_n3d/fixtures}
S=$W/device_stage/N3DAssets
FIXTURES=(diarization_example test_multispk)
TAGS=(ll vll ull offline)

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
need $W/aot/ios_gpu/n3d_streaming_float16.h18p.aimodelc
need $W/aot/offline_ios_gpu/n3d_offline_float16.h18p.aimodelc
for f in embedder_projection silence_embeds mel_filters_128x257 hann_window_400; do need $W/artifacts/$f.f32le; done
need $W/artifacts/metadata.ship.json
for f in $FIXTURES; do need $FX/${f}_16k.wav; done

[[ $S == */_work/device_stage/N3DAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/fixtures $S/golden/pygpu
cp -cR $W/aot/ios_gpu/n3d_streaming_float16.h18p.aimodelc $S/
cp -cR $W/aot/offline_ios_gpu/n3d_offline_float16.h18p.aimodelc $S/
if [ -d $W/aot/ios_ane/n3d_streaming_float16.h18p.aimodelc ]; then
  mkdir -p $S/ane
  cp -cR $W/aot/ios_ane/n3d_streaming_float16.h18p.aimodelc $S/ane/
else
  echo "note: no ANE bundle (_work/aot/ios_ane): the ane stage will be skipped"
fi
for f in embedder_projection silence_embeds mel_filters_128x257 hann_window_400; do cp -c $W/artifacts/$f.f32le $S/; done
cp -c $W/artifacts/metadata.ship.json $S/metadata.json      # the published one (export_n3d.py --metadata --ship)
for f in $FIXTURES; do cp -c $FX/${f}_16k.wav $S/fixtures/; done
# golden: transformers fp32 probs + logits (the gate) and host_loop.py's logits on the Mac GPU (same fp16 bundle)
for f in $FIXTURES; do for t in $TAGS; do
  need $W/golden/${f}_${t}_probs.f32le
  cp -c $W/golden/${f}_${t}_probs.f32le $W/golden/${f}_${t}_logits.f32le $S/golden/
  [ -f $W/golden/pygpu/${f}_${t}_logits.f32le ] && cp -c $W/golden/pygpu/${f}_${t}_logits.f32le $S/golden/pygpu/
done; done

(cd $S && find . -type f ! -name MD5SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in $S/n3d_streaming_float16.h18p.aimodelc $S/n3d_offline_float16.h18p.aimodelc $S/ane/n3d_streaming_float16.h18p.aimodelc; do
  [ -d $d ] && printf "  %-48s %6.1f MB\n" ${d#$S/} $(( $(find $d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 ))
done
