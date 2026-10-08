#!/bin/zsh
# Gather what D1OmniGate reads into $D1_WORK/assets/D1OmniAssets/ (APFS clones: no extra disk), write the fixtures
# (./_fixtures.py), then D1OmniAssets/MD5SUMS (every file but MD5SUMS). ./_install.sh pushes the directory as it is; the
# Mac run reads it in place (D1_ASSETS, ./_run_mac.sh). Adapted from apps/KevGate/_stage.sh.
#   ./_stage.sh             the phone's stage
#   ./_stage.sh --mac       also the Mac's own AOT for ./_run_mac.sh, apart: $D1_WORK/mac/aot/<name>/<stem>.h16c.aimodelc
#                           (compiled/ship-h16c, the same stripped bundles) and $D1_WORK/mac/ane/ (the Mac's Neural Engine
#                           AOT of the same graphs from round 7, compiled before the strip: the same ops)
#   D1_LANE=<dir>           the lane's working directory (default ~/code/coreai/_d1_omni); D1_WORK (default $D1_LANE/device)
# Layout (GateRunner.swift / Fixtures.swift read it):
#   jit/<name>/   <- $L/bundles/d1-omni-600m/macos-ship/<name>/ (fp16-L64: macos-ship-small/, round 12)
#                    {metadata.json, <stem>.aimodel/} (+ tokenizer/ for
#                    fp16-L256, position_table.f32 for vision-fp16, mel_filters_128x257_f32.bin for audio-fp16-10s);
#                    the bytes of the iOS `ios/` folder of the staging; reference.json and provenance/ stay behind
#   aot/<name>/   <- $L/compiled/ship-h19p/<name>/<stem>.h19p.aimodelc (coreai-build --platform iOS --architecture h19p
#                    --preferred-compute gpu, round 10; fp16-L64 round 12)
#   ane/<name>/   <- $L/compiled/ship-h19p-ane/<name>/<stem>.h19p.aimodelc (--preferred-compute neural-engine, round 11)
#                    for fp16-L256 and audio-fp16-10s
#   fixtures/     <- ./_fixtures.py $L $D1_WORK/fixtures (subset.json, oracle_slim.json, mac_ref.json, bench.json, media/)
# <name> = fp16-L64 (the rows of at most 64 positions, W1 / W2), fp16-L256, fp16-L2048 (img_03's rows: 7 crops,
# P = 1,770), fp16-L4096, vision-fp16, audio-fp16-10s.
# Every graph is checked first: its .aimodel's main.hash against provenance/strip.json (the stripped bytes), every AOT's
# provenance/aot-manifest.json (COMPILED, compiled from that stripped bundle). The texts of the fixture go to the phone
# (a private device run, not a publication).
set -euo pipefail
DIR=${0:A:h}
L=${D1_LANE:-$HOME/code/coreai/_d1_omni}
W=${D1_WORK:-$L/device}
SHIP=$L/bundles/d1-omni-600m/macos-ship
SMALL=$L/bundles/d1-omni-600m/macos-ship-small
H19=$L/compiled/ship-h19p
ANE=$L/compiled/ship-h19p-ane
NAMES=(fp16-L64 fp16-L256 fp16-L2048 fp16-L4096 vision-fp16 audio-fp16-10s)
srcdir() { [ $1 = fp16-L64 ] && echo $SMALL/$1 || echo $SHIP/$1; }
ANE_NAMES=(fp16-L256 audio-fp16-10s)
need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
hexhash() { xxd -p -c 256 "$1" | tr -d '\n'; }

# the stripped bundles and their AOT, checked
check() {  # check <name> <aot dir> <compute>
  local name=$1 adir=$2 compute=$3
  local src=$(srcdir $name)
  local strip=$src/provenance/strip.json
  need $strip
  local bundle after
  bundle=$(/usr/bin/python3 -I -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"]=="PASS", s["status"]; print(s["bundle"])' $strip)
  after=$(/usr/bin/python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["after"]["main_hash"])' $strip)
  need $src/$bundle/main.hash
  [ "$(hexhash $src/$bundle/main.hash)" = "$after" ] || { echo "$name: main.hash differs from strip.json"; exit 1; }
  local m=$adir/$name/provenance/aot-manifest.json
  need $m
  /usr/bin/python3 -I - $m $after $compute <<'PY'
import json, sys
m, after, compute = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
assert m["status"] == "COMPILED", m["status"]
assert m["source"]["main_hash"] == after, (m["source"]["main_hash"], after)
assert m["preferred_compute"] == compute, m["preferred_compute"]
PY
}

if [ "${1:-}" = "--mac" ]; then
  M=$W/mac
  [[ $M == */_d1_omni/device/mac ]] || [ -n "${D1_WORK:-}" ] || { echo "refusing to clear $M"; exit 1; }
  rm -rf $M
  for name in $NAMES; do
    src=($L/compiled/ship-h16c/$name/*.h16c.aimodelc)
    mkdir -p $M/aot/$name
    cp -cR $src[1] $M/aot/$name/
  done
  mkdir -p $M/ane/fp16-L256 $M/ane/audio-fp16-10s
  cp -cR $L/compiled/fp16-L256-ane/*.h16c.aimodelc $M/ane/fp16-L256/
  cp -cR $L/compiled/audio-fp16-10s-ane-r7/*.h16c.aimodelc $M/ane/audio-fp16-10s/
  echo "staged the Mac's AOT: $(find $M -maxdepth 3 -name '*.aimodelc' | sed "s|$M/||" | tr '\n' ' ')($(du -sh $M | cut -f1))"
  exit 0
fi

for name in $NAMES; do check $name $H19 gpu; done
for name in $ANE_NAMES; do check $name $ANE neural-engine; done
junk=$(find $SHIP $SMALL $H19 $ANE \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file"; exit 1; }

/usr/bin/python3 -I $DIR/_fixtures.py $L $W/fixtures

S=$W/assets/D1OmniAssets
[[ $S == */assets/D1OmniAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/jit $S/aot $S/ane $S/fixtures
for name in $NAMES; do
  src=$(srcdir $name)
  mkdir -p $S/jit/$name $S/aot/$name
  cp -c $src/metadata.json $S/jit/$name/
  for b in $src/*.aimodel; do cp -cR $b $S/jit/$name/; done
  [ $name = fp16-L256 ] && cp -cR $src/tokenizer $S/jit/$name/
  [ -f $src/position_table.f32 ] && cp -c $src/position_table.f32 $S/jit/$name/
  [ -f $src/mel_filters_128x257_f32.bin ] && cp -c $src/mel_filters_128x257_f32.bin $S/jit/$name/
  for c in $H19/$name/*.aimodelc; do cp -cR $c $S/aot/$name/; done
done
for name in $ANE_NAMES; do
  mkdir -p $S/ane/$name
  for c in $ANE/$name/*.aimodelc; do cp -cR $c $S/ane/$name/; done
done
cp -cR $W/fixtures/. $S/fixtures/

(cd $S && find . -type f ! -name 'MD5SUMS*' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS)
total=$(find $S -type f -exec stat -f %z {} + | paste -sd+ - | bc)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $total bytes ($(du -sh $S | cut -f1) as clones)"
for d in jit aot ane fixtures; do
  for n in $(ls $S/$d); do
    printf "  %-28s %12d B  %4d files\n" $d/$n $(find $S/$d/$n -type f -exec stat -f %z {} + | paste -sd+ - | bc) $(find $S/$d/$n -type f | wc -l)
  done
done
/usr/bin/python3 -I - $S $W/assets/stage_manifest.json $total <<'PY'
import hashlib, json, os, sys, time
s, out, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
md5sums = open(os.path.join(s, "MD5SUMS"), "rb").read()
dirs = {}
for top in ("jit", "aot", "ane", "fixtures"):
    for n in sorted(os.listdir(os.path.join(s, top))):
        b = f = 0
        for root, _, files in os.walk(os.path.join(s, top, n)):
            for x in files:
                b += os.path.getsize(os.path.join(root, x)); f += 1
        dirs[f"{top}/{n}"] = {"bytes": b, "files": f}
hashes = {}
for top in ("jit", "aot", "ane"):
    for root, ds, files in os.walk(os.path.join(s, top)):
        if "main.hash" in files and (root.endswith(".aimodel") or root.endswith(".aimodelc")):
            hashes[os.path.relpath(root, s)] = open(os.path.join(root, "main.hash"), "rb").read().hex()
json.dump({"schema": "d1omni-gate-stage/1", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dir": s,
           "files": md5sums.count(b"\n"), "bytes": total, "md5sums_sha256": hashlib.sha256(md5sums).hexdigest(),
           "dirs": dirs, "main_hashes": hashes}, open(out, "w"), indent=1)
print(f"stage manifest: {out}")
PY
