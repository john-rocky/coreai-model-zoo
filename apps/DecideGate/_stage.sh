#!/bin/zsh
# Gather what DecideGate reads on the phone into _work/device_stage/DecideAssets/ (APFS clones: no extra disk), then
# write DecideAssets/MD5SUMS (every file but itself). ./_install.sh pushes the directory as it is.
#   ./_stage.sh                        the phone's Core AI architecture: DECIDE_ARCHS (default h19p = iPhone 18 Pro;
#   DECIDE_ARCHS=h18p ./_stage.sh      h18p = iPhone 17 Pro; comma-separated for several). A bundle loads only on
#                                      its own architecture, and the app picks the one named after its phone's.
# Needs, from conversion/gliner25_decide (see aot_compile.py): the AOT bundles of each architecture
#   _work/aot/s256_<target>/gliner25-decide_float16_s256_m32.<arch>.aimodelc   aot_compile.py --tag s256
#   _work/aot/s512_<target>/gliner25-decide_float16_s512_m32.<arch>.aimodelc   aot_compile.py --tag s512
# and, from the data directory (DECIDE_DATA, default ~/code/coreai/_gliner25_decide), the oracle fixtures
# (fixtures/<set>.json, conversion/gliner25_decide_oracle.py) and the Python Mac GPU gate of the same bundles
# (results/gate_s<S>_gpu.json, conversion/export_gliner25_decide.py), of which golden/pygpu_s<S>.json keeps
# case id -> logits only.
set -euo pipefail
DIR=${0:A:h}
C=${DIR:h:h}/conversion/gliner25_decide
W=${DECIDE_WORK:-$C/_work}
DATA=${DECIDE_DATA:-$HOME/code/coreai/_gliner25_decide}
S=$W/device_stage/DecideAssets
TAGS=(s256 s512)
ARCHS=(${(s:,:)${DECIDE_ARCHS:-h19p}})
FIXTURES=(readme21 fast_decisions_s256 fast_decisions_long)

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
typeset -A SRC
for t in $TAGS; do
  for a in $ARCHS; do
    hits=($W/aot/${t}_*/gliner25-decide_float16_${t}_m32.${a}.aimodelc(N/))
    [ ${#hits} -eq 1 ] || { echo "need exactly one ${t} ${a} bundle under $W/aot (found ${#hits}: $hits)"; exit 1; }
    SRC[$t.$a]=$hits[1]
  done
  need $DATA/results/gate_${t}_gpu.json
done
for f in $FIXTURES; do need $DATA/fixtures/$f.json; done

[[ $S == */_work/device_stage/DecideAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/fixtures $S/golden
for k in ${(ok)SRC}; do cp -cR $SRC[$k] $S/; done
for f in $FIXTURES; do cp -c $DATA/fixtures/$f.json $S/fixtures/; done
# golden: the Mac GPU logits of the same fp16 bundle (engine.clean.cases[*].logits of the Python gate)
for t in $TAGS; do
  /usr/bin/python3 - $DATA/results/gate_${t}_gpu.json $S/golden/pygpu_${t}.json <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
h = g["header"]
cases = g["engine"]["clean"]["cases"]
out = {"source": sys.argv[1].split("/")[-1], "bundle": h["bundle"].split("/")[-1], "path": h["path"], "S": h["S"],
       "MMAX": h["MMAX"], "created": h["created"], "mac_os_build": h["env"]["os_build"],
       "logits": {c["id"]: c["logits"] for c in cases}}
json.dump(out, open(sys.argv[2], "w"))
print(f"golden {sys.argv[2].split('/')[-1]}: {len(cases)} cases from {out['source']} ({out['bundle']}, {out['path']})")
PY
done

(cd $S && find . -type f ! -name MD5SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for k in ${(ok)SRC}; do
  d=$S/${SRC[$k]:t}
  printf "  %-50s %6.1f MB  (from %s)\n" ${d#$S/} $(( $(find $d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) ${SRC[$k]:h:t}
done
