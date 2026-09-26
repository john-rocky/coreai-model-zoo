#!/bin/zsh
# Gather what DecideGate reads on the phone into _work/device_stage/DecideAssets/ (APFS clones: no extra disk), then
# write DecideAssets/MD5SUMS (every file but itself). ./_install.sh pushes the directory as it is.
#   ./_stage.sh                        the phone's Core AI architecture: DECIDE_ARCHS (default h19p = iPhone 18 Pro;
#   DECIDE_ARCHS=h18p ./_stage.sh      h18p = iPhone 17 Pro; comma-separated for several). A bundle loads only on
#                                      its own architecture, and the app picks the one named after its phone's.
#   DECIDE_KINDS=aot,jit,mixed ./_stage.sh   which bundles (the app's DECIDE_BUNDLE_KIND picks one per run; default aot):
#     aot    <name>.<arch>.aimodelc per DECIDE_ARCHS
#     jit    <name>.aimodel: the JIT bundle (main.mlirb, main.hash, metadata.json), as macos/ ships it
#     mixed  <name>.mixed.aimodel, for DECIDE_MIXED_TAGS (default s256): the JIT bundle's three files plus the AOT
#            files of DECIDE_MIXED_ARCH (default h18p: not the phone's): main-<arch>.mlirb, main-<arch>-delegates/,
#            stats.json. That is the shape of GLiNER2-PII-CoreAI's ios/ bundle (JIT metadata.json and main.hash).
#     none   no GLiNER2.5-Decide bundle at all (alone; for the load-only mode below)
#   DECIDE_EXTRA="<label>=<path>,..." ./_stage.sh   any other bundle, for the app's load-only mode
#            (DECIDE_LOAD_ONLY=<label>,...): <path> is a .aimodel / .aimodelc directory as shipped, cloned to
#            extra/<label>/<basename> (only read). Labels: letters, digits, _ and -.
#   DECIDE_EXTRA_JIT="<label>=<path>,..."   the JIT IR alone of a .aimodel that also carries AOT files (the reverse
#            of mixed; e.g. GLiNER2-PII-CoreAI's ios/ bundle): only its main.mlirb, main.hash and metadata.json,
#            cloned to extra/<label>/<basename> (the source is only read).
# Needs, from conversion/gliner25_decide (see aot_compile.py): the AOT bundles of each architecture
#   _work/aot/s256_<target>/gliner25-decide_float16_s256_m32.<arch>.aimodelc   aot_compile.py --tag s256
#   _work/aot/s512_<target>/gliner25-decide_float16_s512_m32.<arch>.aimodelc   aot_compile.py --tag s512
# the JIT bundles for jit / mixed (DECIDE_DATA/exports/gliner25-decide_float16_s<S>_m32.aimodel,
# conversion/export_gliner25_decide.py), and, from the data directory (DECIDE_DATA, default
# ~/code/coreai/_gliner25_decide), the oracle fixtures
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
KINDS=(${(s:,:)${DECIDE_KINDS:-aot}})
MIXED_ARCH=${DECIDE_MIXED_ARCH:-h18p}
MIXED_TAGS=(${(s:,:)${DECIDE_MIXED_TAGS:-s256}})
FIXTURES=(readme21 fast_decisions_s256 fast_decisions_long)

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
has() { (( ${KINDS[(Ie)$1]} )); }
for k in $KINDS; do [[ $k == (aot|jit|mixed|none) ]] || { echo "unknown kind $k (DECIDE_KINDS takes aot, jit, mixed, or none alone)"; exit 1; }; done
if has none && (( ${#KINDS} > 1 )); then echo "DECIDE_KINDS=none stands alone (got ${DECIDE_KINDS})"; exit 1; fi
aot_of() {  # aot_of <tag> <arch>: the one AOT bundle of that shape and architecture under _work/aot
  local hits=($W/aot/${1}_*/gliner25-decide_float16_${1}_m32.${2}.aimodelc(N/))
  [ ${#hits} -eq 1 ] || { echo "need exactly one ${1} ${2} bundle under $W/aot (found ${#hits}: $hits)" >&2; return 1; }
  print -r -- $hits[1]
}
typeset -A SRC JIT MIX
for t in $TAGS; do
  if has aot; then
    for a in $ARCHS; do SRC[$t.$a]=$(aot_of $t $a) || exit 1; done
  fi
  if has jit || { has mixed && (( ${MIXED_TAGS[(Ie)$t]} )); }; then
    JIT[$t]=$DATA/exports/gliner25-decide_float16_${t}_m32.aimodel
    for f in main.mlirb main.hash metadata.json; do need $JIT[$t]/$f; done
  fi
  need $DATA/results/gate_${t}_gpu.json
done
if has mixed; then
  for t in $MIXED_TAGS; do
    [ -n "${JIT[$t]:-}" ] || { echo "no JIT bundle for mixed tag $t"; exit 1; }
    MIX[$t]=$(aot_of $t $MIXED_ARCH) || exit 1
    for f in main-$MIXED_ARCH.mlirb main-$MIXED_ARCH-delegates stats.json; do need $MIX[$t]/$f; done
  done
fi
for f in $FIXTURES; do need $DATA/fixtures/$f.json; done
typeset -A EXTRA
for e in ${(s:,:)${DECIDE_EXTRA:-}}; do
  label=${e%%=*}; p=${e#*=}; p=${p%/}
  [[ $e == *=* && $label =~ ^[A-Za-z0-9_-]+$ ]] || { echo "DECIDE_EXTRA entry '$e': want <label>=<path> (label: letters, digits, _, -)"; exit 1; }
  [ -z "${EXTRA[$label]:-}" ] || { echo "DECIDE_EXTRA label $label given twice"; exit 1; }
  [[ $p == *.aimodel || $p == *.aimodelc ]] || { echo "DECIDE_EXTRA $label: $p is not a .aimodel / .aimodelc directory"; exit 1; }
  [ -d "$p" ] || { echo "missing: $p"; exit 1; }
  junk=$(find "$p" \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
  [ -z "$junk" ] || { echo "DECIDE_EXTRA $label: $junk (a hidden or partial file: a download still running?)"; exit 1; }
  EXTRA[$label]=$p
done
JIT_FILES=(main.mlirb main.hash metadata.json)
typeset -A EXTRA_JIT
for e in ${(s:,:)${DECIDE_EXTRA_JIT:-}}; do
  label=${e%%=*}; p=${e#*=}; p=${p%/}
  [[ $e == *=* && $label =~ ^[A-Za-z0-9_-]+$ ]] || { echo "DECIDE_EXTRA_JIT entry '$e': want <label>=<path> (label: letters, digits, _, -)"; exit 1; }
  [ -z "${EXTRA[$label]:-}${EXTRA_JIT[$label]:-}" ] || { echo "label $label given twice (DECIDE_EXTRA / DECIDE_EXTRA_JIT)"; exit 1; }
  [[ $p == *.aimodel ]] || { echo "DECIDE_EXTRA_JIT $label: $p is not a .aimodel directory"; exit 1; }
  for f in $JIT_FILES; do
    need $p/$f
    [ ! -e $p/$f.aria2 ] || { echo "DECIDE_EXTRA_JIT $label: $p/$f.aria2 (a download still running?)"; exit 1; }
  done
  EXTRA_JIT[$label]=$p
done

[[ $S == */_work/device_stage/DecideAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/fixtures $S/golden
for k in ${(ok)SRC}; do cp -cR $SRC[$k] $S/; done
if has jit; then for t in ${(ok)JIT}; do cp -cR $JIT[$t] $S/; done; fi
for t in ${(ok)MIX}; do
  d=$S/gliner25-decide_float16_${t}_m32.mixed.aimodel
  mkdir -p $d
  for f in main.mlirb main.hash metadata.json; do cp -c $JIT[$t]/$f $d/; done
  for f in main-$MIXED_ARCH.mlirb main-$MIXED_ARCH-delegates stats.json; do cp -cR $MIX[$t]/$f $d/; done
done
for f in $FIXTURES; do cp -c $DATA/fixtures/$f.json $S/fixtures/; done
for l in ${(ok)EXTRA}; do mkdir -p $S/extra/$l; cp -cR $EXTRA[$l] $S/extra/$l/; done
for l in ${(ok)EXTRA_JIT}; do
  d=$S/extra/$l/${EXTRA_JIT[$l]:t}
  mkdir -p $d
  for f in $JIT_FILES; do cp -c $EXTRA_JIT[$l]/$f $d/; done
done
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
for d in $S/gliner25-decide_*(N/); do
  printf "  %-55s %7.1f MB  %2d files\n" ${d:t} $(( $(find $d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $d -type f | wc -l)
done
for d in $S/extra/*/*(N/); do
  l=${d:h:t}
  printf "  %-72s %7.1f MB  %2d files  <- %s\n" extra/$l/${d:t} \
    $(( $(find $d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) $(find $d -type f | wc -l) \
    "${EXTRA[$l]:-${EXTRA_JIT[$l]:-?} (${(j:, :)JIT_FILES} only)}"
done
