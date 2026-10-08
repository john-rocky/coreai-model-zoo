#!/bin/zsh
# Gather what D1Gate reads into _work/device_stage/D1Assets/ (APFS clones: no extra disk), write the fixture files,
# then D1Assets/MD5SUMS (every file but the MD5SUMS lists). ./_install.sh pushes the directory as it is; the Mac run
# reads it in place (D1_ASSETS, ./_run_mac.sh). Copied from apps/KevGate/_stage.sh (zoo d1-3b 4955a23), made d1's.
#   ./_stage.sh
#   D1_LANE=<dir> ./_stage.sh        the lane's working directory (default ~/code/coreai/_d1_3b)
#   ./_stage.sh --aot                the decoder's iPhone AOT for load_aot, apart: _work/device_stage_aot/D1Assets/
#                                    aot/d1_3b_decode_int8mlp_pf16.h19p.aimodelc + MD5SUMS_AOT (pushed with
#                                    D1_STAGE_DIR=… D1_PUSH_ONLY=… ./_install.sh; checked on the phone by the md5 stage
#                                    with D1_MD5SUMS=MD5SUMS_AOT)
# Layout (GateRunner.swift / Fixtures.swift read it):
#   decoder/    <- $L/exports/bundles/d1_3b_decode_int8mlp_pf16/{metadata.json, <name>.aimodel/, tokenizer/, head/, LICENSE}
#   tower/      <- $L/exports/vision/d1_3b_vision_fp16w32/{metadata.json, <name>.aimodel/, host/, LICENSE}
#   fixtures/   requests.json: $L/fixtures/records.json (361 records) + image_records.json (12 picture records; the URL
#               record card_cats listed as skipped), {id, set, source, request, images}
#               images/: the 12 PNG files (drawn for the fixture, CC0)
#               oracle_slim.json: $L/oracle/records_oracle.json + records_oracle_images.json per question (name, type,
#               keys, row_ids, groups, probs, api_probs, argmax, top2_margin, near_tie) and the provider's refusals
#               mac_ref.json: the Mac's read-out of the same decoder asset per question (hidden_sha256, p_bits = float64
#               bit patterns in hex): $L/results/r5c_readout_int8mlp_pf16.json (text) + $L/results/
#               r8_mac_ref_images_int8mlp.json (the pictures, decide.py run with the tower fp16w32)
#               red_arms.json: $L/fixtures/red_arms_r4.json with the provider's probabilities of every perturbed row
#               ($L/oracle/red_r4/records_oracle.json)
#               bench.json: round 7's timed items (conversion/d1/timing.py ITEMS) and the AOT subset
# The fixture texts go to the phone (a private device run, not a publication).
set -euo pipefail
DIR=${0:A:h}
W=${D1_WORK:-$DIR/_work}
L=${D1_LANE:-$HOME/code/coreai/_d1_3b}
NAME=d1_3b_decode_int8mlp_pf16
TNAME=d1_3b_vision_fp16w32
need() { [ -e "$1" ] || { echo "missing: $1"; exit 1; } }

if [ "${1:-}" = "--aot" ]; then
  SRC=$L/exports/bundles_aotc_ios/$NAME.h19p.aimodelc
  need $SRC/main.hash
  SA=$W/device_stage_aot/D1Assets
  [[ $SA == */_work/device_stage_aot/D1Assets ]] || { echo "refusing to clear $SA"; exit 1; }
  rm -rf $SA
  mkdir -p $SA/aot
  cp -cR $SRC $SA/aot/$NAME.h19p.aimodelc
  (cd $SA && find aot -type f | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS_AOT)
  echo "staged $SA: $(grep -c . $SA/MD5SUMS_AOT) files, $(du -sh $SA | cut -f1)"
  cat $SA/MD5SUMS_AOT
  exit 0
fi

DEC=$L/exports/bundles/$NAME
TOW=$L/exports/vision/$TNAME
for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json head/option_rows.json \
  head/option_rows.safetensors $NAME.aimodel/main.mlirb $NAME.aimodel/main.hash; do need $DEC/$f; done
for f in metadata.json host/position_embedding.safetensors $TNAME.aimodel/main.mlirb $TNAME.aimodel/main.hash; do need $TOW/$f; done
for f in $L/fixtures/records.json $L/fixtures/image_records.json $L/fixtures/red_arms_r4.json $L/oracle/records_oracle.json \
  $L/oracle/records_oracle_images.json $L/oracle/red_r4/records_oracle.json $L/results/r5c_readout_int8mlp_pf16.json \
  $L/results/r8_mac_ref_images_int8mlp.json; do need $f; done
junk=$(find $DEC $TOW \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
[ -z "$junk" ] || { echo "$junk: a hidden or partial file (an export still running?)"; exit 1; }

S=$W/device_stage/D1Assets
[[ $S == */_work/device_stage/D1Assets ]] || { echo "refusing to clear $S"; exit 1; }
rm -rf $S
mkdir -p $S/decoder $S/tower $S/fixtures/images
for e in metadata.json $NAME.aimodel tokenizer head LICENSE; do [ -e $DEC/$e ] && cp -cR $DEC/$e $S/decoder/$e; done
for e in metadata.json $TNAME.aimodel host LICENSE; do [ -e $TOW/$e ] && cp -cR $TOW/$e $S/tower/$e; done

/usr/bin/python3 - $L $S/fixtures <<'PY'
import hashlib, json, shutil, struct, sys
from pathlib import Path
L, out = Path(sys.argv[1]), Path(sys.argv[2])

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def bits(x):
    return format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "x")

src = {k: L / v for k, v in {
    "fixture": "fixtures/records.json", "images": "fixtures/image_records.json", "red_arms": "fixtures/red_arms_r4.json",
    "oracle_text": "oracle/records_oracle.json", "oracle_images": "oracle/records_oracle_images.json",
    "oracle_red": "oracle/red_r4/records_oracle.json", "mac_text": "results/r5c_readout_int8mlp_pf16.json",
    "mac_images": "results/r8_mac_ref_images_int8mlp.json"}.items()}
sources = {k: {"path": str(p), "sha256": sha(p)} for k, p in src.items()}

# requests: the text fixture, then the picture records (a URL picture is not fetched: listed as skipped)
recs, skipped = [], []
for r in json.load(open(src["fixture"]))["records"]:
    recs.append({"id": r["id"], "set": "fixture", "source": r["source"], "request": r["request"]})
for r in json.load(open(src["images"]))["records"]:
    imgs = r.get("images", [])
    if any("://" in x for x in imgs):
        skipped.append({"id": r["id"], "images": imgs, "why": "a URL picture is not fetched (the oracle skipped it too)"})
        continue
    names = []
    for x in imgs:
        p = src["images"].parent / x
        shutil.copyfile(p, out / "images" / p.name)
        names.append(f"images/{p.name}")
    recs.append({"id": r["id"], "set": "image", "source": r["source"], "request": r["request"], "images": names})
ids = [r["id"] for r in recs]
assert len(ids) == len(set(ids)) == 361 + 12, (len(ids), len(set(ids)))
json.dump({"schema": "d1-gate-requests/1", "sources": {k: sources[k] for k in ("fixture", "images")}, "records": recs,
           "skipped": skipped}, open(out / "requests.json", "w"), ensure_ascii=False)

# the provider's oracle per question (row form) + the API path (Tree) when the record has it
def slim(path, set_name):
    o = json.load(open(path))
    res = {}
    for r in o["records"]:
        api = (r.get("api") or {}).get("probs")
        qs = []
        for k, q in enumerate(r["questions"]):
            qs.append({"name": q["name"], "type": q["type"], "keys": q["keys"], "row_ids": q["row_ids"], "groups": q["groups"],
                       "probs": q["probs"], "api_probs": api[k] if api and len(api) == len(r["questions"]) else None,
                       "argmax": q["argmax"], "top2_margin": q["top2_margin"], "near_tie": q["near_tie"]})
        res[r["id"]] = {"set": set_name, "questions": qs, "refused": [x["name"] for x in r.get("refused") or []]}
    return res
oracle = slim(src["oracle_text"], "fixture")
oracle.update(slim(src["oracle_images"], "image"))
assert set(ids) <= set(oracle), sorted(set(ids) - set(oracle))[:5]
oracle = {k: v for k, v in oracle.items() if k in set(ids)}
nq = sum(len(v["questions"]) for v in oracle.values())
json.dump({"schema": "d1-gate-oracle-slim/1", "sources": {k: sources[k] for k in ("oracle_text", "oracle_images")},
           "records": oracle}, open(out / "oracle_slim.json", "w"))

# the Mac's read-out of the same decoder asset: round 5c's readout gate (text) and round 8's decide.py run (pictures)
mac = {}
g = json.load(open(src["mac_text"]))
assert g.get("result") == "PASS", g.get("result")
for r in g["runs"]:
    if r.get("variant", "base") != "base":
        continue
    mac.setdefault(r["id"], {"rows": {}})["rows"][r["name"]] = {"hidden_sha256": r["hidden_sha256"],
                                                               "p_bits": [bits(x) for x in r["probs"]]}
mi = json.load(open(src["mac_images"]))
for rid, rec in mi["records"].items():
    mac.setdefault(rid, {"rows": {}})
    for name, row in rec["rows"].items():
        mac[rid]["rows"][name] = {"hidden_sha256": row["hidden_sha256"], "p_bits": row["p_bits"]}
    if rec.get("tower_outputs_sha256"):
        mac[rid]["tower_outputs_sha256"] = rec["tower_outputs_sha256"]
for rid, v in oracle.items():
    have = set(mac.get(rid, {}).get("rows", {}))
    want = {q["name"] for q in v["questions"]}
    assert have == want, (rid, sorted(want - have), sorted(have - want))
json.dump({"schema": "d1-gate-mac-ref/1", "sources": {k: sources[k] for k in ("mac_text", "mac_images")},
           "assets": {"text": g.get("bundle"), "pictures": mi.get("assets")}, "records": mac}, open(out / "mac_ref.json", "w"))

# round 4's red arms, with the provider's probabilities of every perturbed row
arms_doc = json.load(open(src["red_arms"]))
ored = {r["id"]: {q["name"]: q["probs"] for q in r["questions"]} for r in json.load(open(src["oracle_red"]))["records"]}
arms = []
for a in arms_doc["arms"]:
    if a["kind"] in ("word", "not"):
        reqs = [(a["id"], a["request"], a["base"], [a["question"]])]
    elif a["kind"] == "state_swap":
        reqs = [(n, q, n.split("_with_state_of_")[0], None) for n, q in a["requests"].items()]
    else:
        raise SystemExit(f"unknown arm kind {a['kind']}")
    rq = []
    for name, req, base, only in reqs:
        req = json.loads(req) if isinstance(req, str) else req
        assert base in oracle, base
        rq.append({"name": name, "request": req, "base": base, "only": only, "oracle": ored[name]})
    arms.append({"id": a["id"], "kind": a["kind"], "requests": rq})
json.dump({"schema": "d1-gate-red-arms/1", "rule": arms_doc.get("rule"), "sources": {k: sources[k] for k in ("red_arms", "oracle_red")},
           "arms": arms}, open(out / "red_arms.json", "w"), ensure_ascii=False)

# round 7's timed items (timing.py ITEMS): one_question, three_shared, three_direct, state_3_4k, image_384px
by = {r["id"]: r for r in recs}
def item(name, kind, record, questions, shared, reps=5, rest=0):
    r = by[record]
    names = list(r["request"]["questions"])
    qs = None if questions is None else ([names[0]] if questions == "first" else questions)
    assert qs is None or all(q in names for q in qs), (name, qs)
    return {"name": name, "kind": kind, "record": record, "questions": qs, "shared": shared, "reps": reps, "rep_rest_s": rest}
bench = {"schema": "d1-gate-bench/1", "source": "conversion/d1/timing.py ITEMS (round 7) and the r8 launch: warm-up 1 + 5 reps, "
         "60 s before every item, nominal start; state_3_4k (one decision past 20 s of back-to-back GPU work) rests 30 s "
         "between its decisions",
         "bench": [item("one_question", "text", "card_refund", ["refund"], False),
                   item("three_shared", "text", "card_refund", None, True),
                   item("three_direct", "text", "card_refund", None, False),
                   item("state_3_4k", "text", "long_34k", "first", False, rest=30),
                   item("image_384px", "image", "img01_shapes_384x384", "first", False)],
         "bench_aot": [item("one_question", "text", "card_refund", ["refund"], False),
                       item("three_shared", "text", "card_refund", None, True)]}
json.dump(bench, open(out / "bench.json", "w"), indent=1)
print(f"requests.json: {len(recs)} records (+{len(skipped)} skipped); oracle_slim.json: {len(oracle)} records, {nq} questions "
      f"(near-ties {sum(q['near_tie'] for v in oracle.values() for q in v['questions'])}); mac_ref.json: {len(mac)} records, "
      f"{sum(len(v['rows']) for v in mac.values())} rows; red_arms.json: {len(arms)} arms; bench.json: {len(bench['bench'])} + "
      f"{len(bench['bench_aot'])} items")
PY

(cd $S && find . -type f ! -name 'MD5SUMS*' | sed 's|^\./||' | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS)
echo "staged $S: $(grep -c . $S/MD5SUMS) files + MD5SUMS, $(du -sh $S | cut -f1)"
for d in decoder tower fixtures; do
  printf "  %-12s %10.1f MB  %4d files\n" $d $(( $(find $S/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
    $(find $S/$d -type f | wc -l)
done
