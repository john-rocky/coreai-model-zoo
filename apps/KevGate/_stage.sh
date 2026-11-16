#!/bin/zsh
# Gather what KevGate reads into _work/device_stage/KevAssets/ (APFS clones: no extra disk), write the fixture files,
# then KevAssets/MD5SUMS (every file but the MD5SUMS lists). ./_install.sh pushes the directory as it is; the Mac run
# reads it in place (KEV_ASSETS, ./_run_mac.sh). Copied from apps/DeciderVisionGate/_stage.sh (zoo main 2d214b3).
#   ./_stage.sh
#   KEV_LANE=<dir> ./_stage.sh        the lane's working directory (default ~/code/coreai/_kev)
#   ./_stage.sh --4b [efr|noefr]      the Kev-4B iPhone AOT for load_4b, apart: _work/device_stage_4b/KevAssets/
#                                     aot/kev_4b_decode_fp16_pf16.h19p.aimodelc + MD5SUMS_4B (pushed with
#                                     KEV_STAGE_DIR=… KEV_PUSH_ONLY=… ./_install.sh; checked on the phone by the md5 stage
#                                     with KEV_MD5SUMS=MD5SUMS_4B)
# Layout (GateRunner.swift / Fixtures.swift read it):
#   decoder/      <- $L/exports/bundles/kev_0_8b_decode_fp16_pf16/{metadata.json, <name>.aimodel/, tokenizer/, head/}
#   aot/          <- $L/exports/bundles_aotc_ios/kev_0_8b_decode_fp16_pf16.h19p.aimodelc (--expect-frequent-reshapes)
#   decoder_4b/   <- $L/exports/bundles/kev_4b_decode_fp16_pf16/{metadata.json, tokenizer/, head/} (no .aimodel)
#   fixtures/     requests.json: $L/fixtures/records.json (384 records) + heldout.json (130), {id, set, source, request}
#                 oracle_slim.json: $L/oracle/records_oracle.json + oracle/heldout/records_oracle.json per question
#                 (qid, type, keys, row_ids, decide, opts, probs, argmax, top2_margin, near_tie)
#                 mac_ref.json: the Mac's Swift run of the same bundle (round 7, AOT h16c), $L/swift/gate/
#                 {fixture,heldout}_kev-0.8b.json per question (hidden_sha256, p_bits, shared_p_bits) + answers_json
#                 bench.json: the timed items (conversion/kev/timing.py's)
#                 oracle_slim_4b.json / mac_ref_4b.json: tv4_000 of $L/oracle_4b and $L/swift/gate/fixture_kev-4b.json
# The tv4 / tv4x / tv4s texts go to the phone (a private device run, not a publication).
set -euo pipefail
DIR=${0:A:h}
W=${KEV_WORK:-$DIR/_work}
L=${KEV_LANE:-$HOME/code/coreai/_kev}
NAME=kev_0_8b_decode_fp16_pf16
NAME4=kev_4b_decode_fp16_pf16
need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }

if [ "${1:-}" = "--4b" ]; then
  V=${2:-efr}
  case $V in
    efr) SRC=$L/exports/bundles_aotc_ios/$NAME4.h19p.aimodelc ;;
    noefr) SRC=$L/exports/bundles_aotc_ios/noefr/$NAME4.h19p.aimodelc ;;
    *) echo "--4b efr|noefr"; exit 1 ;;
  esac
  need $SRC/main.hash
  S4=$W/device_stage_4b/KevAssets
  [[ $S4 == */_work/device_stage_4b/KevAssets ]] || { echo "refusing to clear $S4"; exit 1; }
  rm -rf $S4
  mkdir -p $S4/aot
  cp -cR $SRC $S4/aot/$NAME4.h19p.aimodelc
  (cd $S4 && find aot -type f | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS_4B)
  echo "$V" > $S4/aot/VARIANT_4B.txt
  echo "staged $S4 ($V): $(grep -c . $S4/MD5SUMS_4B) files, $(du -sh $S4 | cut -f1)"
  cat $S4/MD5SUMS_4B
  exit 0
fi

DEC=$L/exports/bundles/$NAME
DEC4=$L/exports/bundles/$NAME4
AOT=$L/exports/bundles_aotc_ios/$NAME.h19p.aimodelc
for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json head/head.safetensors head/kev_head.json \
  $NAME.aimodel/main.mlirb $NAME.aimodel/main.hash; do need $DEC/$f; done
for f in metadata.json tokenizer/tokenizer.json head/head.safetensors; do need $DEC4/$f; done
need $AOT/main.hash
for f in $L/fixtures/records.json $L/fixtures/heldout.json $L/oracle/records_oracle.json \
  $L/oracle/heldout/records_oracle.json $L/swift/gate/fixture_kev-0.8b.json $L/swift/gate/heldout_kev-0.8b.json \
  $L/oracle_4b/records_oracle.json $L/swift/gate/fixture_kev-4b.json; do need $f; done
junk=$(find $DEC $AOT \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file (an export still running?)"; exit 1; }

S=$W/device_stage/KevAssets
[[ $S == */_work/device_stage/KevAssets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/decoder $S/aot $S/decoder_4b $S/fixtures
for e in metadata.json $NAME.aimodel tokenizer head; do cp -cR $DEC/$e $S/decoder/$e; done
cp -cR $AOT $S/aot/$NAME.h19p.aimodelc
for e in metadata.json tokenizer head; do cp -cR $DEC4/$e $S/decoder_4b/$e; done

/usr/bin/python3 - $L $S/fixtures <<'PY'
import hashlib, json, sys
L, out = sys.argv[1:3]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

src = {k: f"{L}/{v}" for k, v in {
    "fixture": "fixtures/records.json", "heldout": "fixtures/heldout.json",
    "oracle_fixture": "oracle/records_oracle.json", "oracle_heldout": "oracle/heldout/records_oracle.json",
    "mac_fixture": "swift/gate/fixture_kev-0.8b.json", "mac_heldout": "swift/gate/heldout_kev-0.8b.json",
    "oracle_4b": "oracle_4b/records_oracle.json", "mac_4b": "swift/gate/fixture_kev-4b.json"}.items()}
sources = {k: {"path": p, "sha256": sha(p)} for k, p in src.items()}

recs = []
for set_name in ("fixture", "heldout"):
    for r in json.load(open(src[set_name]))["records"]:
        recs.append({"id": r["id"], "set": set_name, "source": r["source"], "request": r["request"]})
ids = [r["id"] for r in recs]
assert len(ids) == len(set(ids)) == 514, len(ids)
json.dump({"schema": "kev-gate-requests/1", "sources": {k: sources[k] for k in ("fixture", "heldout")}, "records": recs},
          open(f"{out}/requests.json", "w"))

KEYS = ("qid", "type", "keys", "row_ids", "decide", "opts", "probs", "argmax", "top2_margin", "near_tie")
def slim(path, set_name, only=None):
    o = json.load(open(path))
    return {r["id"]: {"set": set_name, "questions": [{k: q[k] for k in KEYS} for q in r["questions"]]}
            for r in o["records"] if only is None or r["id"] in only}

def macref(path, set_name, only=None):
    m = json.load(open(path))
    assert m["assets"]["kind"] == "aot", m["assets"]
    recs = {}
    for r in m["records"]:
        if only is not None and r["id"] not in only:
            continue
        rows = []
        for x in r["rows"]:
            row = {"hidden_sha256": x["hidden_sha256"], "p_bits": x["p_bits"]}
            if "shared_p_bits" in x:
                row["shared_p_bits"] = x["shared_p_bits"]
            rows.append(row)
        recs[r["id"]] = {"set": set_name, "answers_json": r["direct"]["answers_json"], "rows": rows}
    meta = {k: m.get(k) for k in ("label", "assets", "shared", "reset_check", "load", "environment", "started", "finished")}
    return recs, meta

oracle = slim(src["oracle_fixture"], "fixture")
oracle.update(slim(src["oracle_heldout"], "heldout"))
assert set(oracle) == set(ids), (len(oracle), len(ids))
qs = sum(len(v["questions"]) for v in oracle.values())
json.dump({"schema": "kev-gate-oracle-slim/1", "sources": {k: sources[k] for k in ("oracle_fixture", "oracle_heldout")},
           "records": oracle}, open(f"{out}/oracle_slim.json", "w"))

mac, mmeta = macref(src["mac_fixture"], "fixture")
mh, hmeta = macref(src["mac_heldout"], "heldout")
mac.update(mh)
assert set(mac) == set(ids), len(mac)
for rid, v in mac.items():
    assert len(v["rows"]) == len(oracle[rid]["questions"]), rid
json.dump({"schema": "kev-gate-mac-ref/1", "sources": {k: sources[k] for k in ("mac_fixture", "mac_heldout")},
           "runs": {"fixture": mmeta, "heldout": hmeta}, "records": mac}, open(f"{out}/mac_ref.json", "w"))

def item(name, record, keep, modes, reps):
    return {"name": name, "record": record, "keep": keep, "modes": modes, "reps": reps}
bench = {"schema": "kev-gate-bench/1", "source": "conversion/kev/timing.py items (round 5) and round 8",
         "bench": [item("tv4_000:q0", "tv4_000", [0], ["direct"], 5), item("own_j03:q0", "own_j03", [0], ["direct"], 5),
                   item("own_L02:q0", "own_L02", [0], ["direct"], 5), item("own_L01:q2", "own_L01", [2], ["direct"], 5),
                   item("own_m01:first5", "own_m01", [0, 1, 2, 3, 4], ["direct", "shared"], 5),
                   item("own_m01:all8", "own_m01", list(range(8)), ["direct", "shared"], 3),
                   item("own_L02:all4", "own_L02", [0, 1, 2, 3], ["direct", "shared"], 2)],
         "bench_aot": [item("tv4_000:q0", "tv4_000", [0], ["direct"], 5), item("own_j03:q0", "own_j03", [0], ["direct"], 5),
                       item("own_m01:first5", "own_m01", [0, 1, 2, 3, 4], ["shared"], 5)]}
for it in bench["bench"] + bench["bench_aot"]:
    assert it["record"] in oracle and max(it["keep"]) < len(oracle[it["record"]]["questions"]), it
json.dump(bench, open(f"{out}/bench.json", "w"), indent=1)

o4 = slim(src["oracle_4b"], "fixture", only={"tv4_000"})
m4, m4meta = macref(src["mac_4b"], "fixture", only={"tv4_000"})
assert set(o4) == set(m4) == {"tv4_000"}
json.dump({"schema": "kev-gate-oracle-slim/1", "model": "kev-4b", "sources": {"oracle_4b": sources["oracle_4b"]},
           "records": o4}, open(f"{out}/oracle_slim_4b.json", "w"))
json.dump({"schema": "kev-gate-mac-ref/1", "model": "kev-4b", "sources": {"mac_4b": sources["mac_4b"]},
           "runs": {"fixture": m4meta}, "records": m4}, open(f"{out}/mac_ref_4b.json", "w"))
print(f"requests.json: {len(recs)} records; oracle_slim.json: {len(oracle)} records, {qs} questions "
      f"(near-ties {sum(q['near_tie'] for v in oracle.values() for q in v['questions'])}); mac_ref.json: {len(mac)} records; "
      f"bench.json: {len(bench['bench'])} + {len(bench['bench_aot'])} items; 4B: tv4_000")
PY

(cd $S && find . -type f ! -name 'MD5SUMS*' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in decoder aot decoder_4b fixtures; do
  printf "  %-12s %10.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
