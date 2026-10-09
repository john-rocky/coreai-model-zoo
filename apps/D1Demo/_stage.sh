#!/bin/zsh
# Gather what D1Demo reads into _work/device_stage/D1Assets/ (APFS clones: no extra disk) and write D1Assets/MD5SUMS
# (every file but the list). ./_install.sh pushes the directory as it is into the app's container.
#   ./_stage.sh <decoder bundle> <tower bundle> <samples dir>
#       decoder/  <- the iPhone's decoder bundle (d1_3b_decode_int8mlp_pf64_s: metadata.json, <name>.aimodel/,
#                    tokenizer/, head/, LICENSE), as the Hugging Face repository holds it
#       tower/    <- the tower bundle (d1_3b_vision_fp16w32_s: metadata.json, <name>.aimodel/, host/, LICENSE)
#       demo/     <- make_samples.py's output: samples.json, requests/, pictures/
# Every source file is checked against the Hugging Face repository's SHA256SUMS when the bundles sit in a download of it
# (gpu-pipelined/<bundle> beside ../SHA256SUMS).
set -euo pipefail
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
DEC=${1:?usage: _stage.sh <decoder bundle> <tower bundle> <samples dir>}
TOW=${2:?}
DEMO=${3:?}
need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
DN=${DEC:t}; TN=${TOW:t}
for f in metadata.json $DN.aimodel/main.mlirb $DN.aimodel/main.hash tokenizer/tokenizer.json head/option_rows.json \
  head/option_rows.safetensors; do need $DEC/$f; done
for f in metadata.json $TN.aimodel/main.mlirb $TN.aimodel/main.hash host/position_embedding.safetensors; do need $TOW/$f; done
for f in samples.json requests pictures; do need $DEMO/$f; done
junk=$(find $DEC $TOW $DEMO \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file"; exit 1; }

# the bundles against the repository's SHA256SUMS (a download: <root>/gpu-pipelined/<bundle>, <root>/SHA256SUMS)
for B in $DEC $TOW; do
  SUMS=${B:h:h}/SHA256SUMS
  [ -f $SUMS ] || { echo "no SHA256SUMS beside ${B:h} (not a download of the repository): bundle bytes not checked"; continue; }
  (cd ${B:h:h} && grep " gpu-pipelined/${B:t}/" SHA256SUMS | nice -n 10 shasum -a 256 -c --quiet -) \
    && echo "${B:t}: every file equal to SHA256SUMS ($(grep -c " gpu-pipelined/${B:t}/" $SUMS) files)" \
    || { echo "${B:t}: differs from SHA256SUMS"; exit 1; }
done

S=$W/device_stage/D1Assets
[[ $S == */_work/device_stage/D1Assets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/decoder $S/tower $S/demo
for e in metadata.json $DN.aimodel tokenizer head LICENSE; do [ -e $DEC/$e ] && cp -cR $DEC/$e $S/decoder/$e; done
for e in metadata.json $TN.aimodel host LICENSE; do [ -e $TOW/$e ] && cp -cR $TOW/$e $S/tower/$e; done
cp -c $DEMO/samples.json $S/demo/
cp -cR $DEMO/requests $DEMO/pictures $S/demo/
(cd $S && find . -type f ! -name 'MD5SUMS' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in decoder tower demo; do
  printf "  %-8s %10.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
