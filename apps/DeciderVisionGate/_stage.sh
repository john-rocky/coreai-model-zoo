#!/bin/zsh
# Gather what DeciderVisionGate reads into _work/device_stage/DeciderVisionAssets/ (APFS clones: no extra disk), then
# write DeciderVisionAssets/MD5SUMS (every file but itself). ./_install.sh pushes the directory as it is; the Mac run reads
# it in place (DV_ASSETS, ./_run_mac.sh).
#   ./_stage.sh
#   DV_LANE=<dir> ./_stage.sh          the port's working directory (default ~/code/coreai/_decider2bv)
#   DV_EXTRA="<rel path>=<src path>,..." ./_stage.sh     more assets cloned in beside them (e.g. the decoder's AOT
#                                      aot/<name>.h19p.aimodelc for DV_DECODER_AOT to name at launch)
# Layout (GateRunner.swift reads it):
#   decoder/          <- $DV_LANE/exports/bundles/decider_2b_vision_decode_int8mix_pf16/ (metadata.json, <name>.aimodel/,
#                        tokenizer/): the shipped decoder, main S=1 + prefill S=16
#   towers/<name>.aimodel/ <- $DV_LANE/exports/decider_2b_vision_{g256,g448}_vision_fp16w32/<name>.aimodel (fp16w32)
#   fixtures/         <- $DV_LANE/fixtures/{rows.json,meta.json,images/}
#                        oracle_slim.json: the g256 / g448 / text runs of $DV_LANE/oracle/fixture_oracle.json (ids,
#                        slot_idx, nopts, probs, argmax)
#                        mac_ref.json: the Mac's Swift JIT read-out of the same assets ($DV_LANE/swift/jit/fixture_jit.json,
#                        round 8) per run, with the sha256 of each run's dumped fp16 slot logits
set -euo pipefail
DIR=${0:A:h}
W=${DV_WORK:-$DIR/_work}
L=${DV_LANE:-$HOME/code/coreai/_decider2bv}
DEC=${DV_DECODER_SRC:-$L/exports/bundles/decider_2b_vision_decode_int8mix_pf16}
T256=${DV_TOWER_G256_SRC:-$L/exports/decider_2b_vision_g256_vision_fp16w32/decider_2b_vision_g256_vision_fp16w32.aimodel}
T448=${DV_TOWER_G448_SRC:-$L/exports/decider_2b_vision_g448_vision_fp16w32/decider_2b_vision_g448_vision_fp16w32.aimodel}
MAC=${DV_MAC_REF:-$L/swift/jit/fixture_jit.json}
ORACLE=$L/oracle/fixture_oracle.json
S=$W/device_stage/DeciderVisionAssets

need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }
for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do need $DEC/$f; done
DEC_MODEL=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["assets"]["main"])' $DEC/metadata.json)
for f in main.mlirb main.hash metadata.json; do need $DEC/$DEC_MODEL/$f; need $T256/$f; need $T448/$f; done
for f in $L/fixtures/rows.json $L/fixtures/meta.json $L/fixtures/images $ORACLE $MAC; do need $f; done
junk=$(find $DEC $T256 $T448 \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file (a download or export still running?)"; exit 1; }
typeset -A EXTRA
for e in ${(s:,:)${DV_EXTRA:-}}; do
  rel=${e%%=*}; p=${e#*=}; p=${p%/}
  [[ $e == *=* && -n $rel && $rel != /* && $rel != *..* ]] || { echo "DV_EXTRA entry '$e': want <rel path>=<src path>"; exit 1; }
  [[ $p == *.h[0-9]*p.* || $rel == *.h[0-9]*p.* ]] && [[ $rel != aot/* ]] && { echo "DV_EXTRA: put an iPhone AOT asset under aot/"; exit 1; }
  need $p
  EXTRA[$rel]=$p
done

[[ $S == */_work/device_stage/DeciderVisionAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/towers $S/fixtures
cp -cR $DEC $S/decoder
cp -cR $T256 $S/towers/${T256:t}
cp -cR $T448 $S/towers/${T448:t}
cp -c $L/fixtures/rows.json $L/fixtures/meta.json $S/fixtures/
cp -cR $L/fixtures/images $S/fixtures/images
for rel in ${(ok)EXTRA}; do mkdir -p $S/${rel:h}; cp -cR $EXTRA[$rel] $S/$rel; echo "extra: $rel <- $EXTRA[$rel]"; done

/usr/bin/python3 - $ORACLE $MAC $S/fixtures <<'PY'
import hashlib, json, os, sys
oracle_path, mac_path, out = sys.argv[1:4]
V, PAD = 248320, 248056

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

o = json.load(open(oracle_path))
runs = {}
for r in o["rows"]:
    if r["arm"] not in ("g256", "g448", "text"):
        continue
    runs[f"{r['id']}/{r['arm']}"] = {k: r[k] for k in ("ids", "slot_idx", "nopts", "probs", "argmax")}
json.dump({"source": {"path": oracle_path, "sha256": sha(oracle_path)}, "runs": runs},
          open(f"{out}/oracle_slim.json", "w"))
slots = sum(len(v["slot_idx"]) for v in runs.values())

m = json.load(open(mac_path))
dump = m["dump_dir"]
mruns = {}
for r in m["runs"]:
    rec = {k: r.get(k) for k in ("ids", "slots", "tower_embeds_sha256", "resized_rgb_sha256", "patches_sha256",
                                  "decoded_rgb_sha256", "wall_from_file_s")}
    rec["answers"] = [{k: a[k] for k in ("letter_logits", "probs", "argmax", "full_vocab_top1_id", "read_from")}
                      for a in r["answers"]]
    lf = os.path.join(dump, r["logits_file"]) if r.get("logits_file") else None
    rec["slot_logits_sha256"] = sha(lf) if lf and os.path.exists(lf) else None
    mruns[f"{r['id']}/{r['arm']}"] = rec
missing_logits = [k for k, v in mruns.items() if v["slot_logits_sha256"] is None]
assert set(mruns) == set(runs), f"Mac runs {len(mruns)} != oracle runs {len(runs)}"
json.dump({"source": {"path": mac_path, "sha256": sha(mac_path), "label": m.get("label"), "assets": m.get("assets"),
                      "environment": m.get("environment"), "reset_check": m.get("reset_check")},
           "runs": mruns}, open(f"{out}/mac_ref.json", "w"))
print(f"oracle_slim.json: {len(runs)} runs, {slots} slots; mac_ref.json: {len(mruns)} runs "
      f"(slot logits sha256 for {len(mruns) - len(missing_logits)})")
PY

(cd $S && find . -type f ! -name MD5SUMS | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in decoder towers fixtures ${(k)EXTRA}; do
  printf "  %-48s %8.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
