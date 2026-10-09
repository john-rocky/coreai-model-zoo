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
#   ./_stage.sh --aot <x.aimodelc>... [--file <src> <aot/rel>]...
#                                    the same with these iPhone AOT assets under aot/ by their own names (e.g. the
#                                    decoder without efr, the tower's AOT for load_tower_aot) and any small file at
#                                    aot/<rel> (a Mac reference the app compares with); MD5SUMS_AOT lists every file
#   ./_stage.sh --decoder <bundle> --into decoder_pf<S> --mac-text <readout gate json> --mac-images <images json>
#                                    the same decoder at another chunk width S, apart: _work/device_stage_pf<S>/D1Assets/
#                                    with decoder_pf<S>/ (the bundle), fixtures/mac_ref_pf<S>.json (the Mac's read-out of
#                                    this bundle: its readout gate for the text, its decide.py run for the pictures),
#                                    fixtures/bench_pf<S>.json (the staged bench.json's items, with the calls each item
#                                    makes at this S from $L/results/r7_timing_plan.json) and MD5SUMS_PF<S> (every file
#                                    of the three). Pushed into the installed app's D1Assets with D1_STAGE_DIR=…
#                                    D1_PUSH_ONLY=decoder_pf<S>,fixtures/mac_ref_pf<S>.json,fixtures/bench_pf<S>.json,
#                                    MD5SUMS_PF<S> D1_SKIP_APP=1 ./_install.sh (sizes from the phone's listing), read by
#                                    the app with D1_DECODER=decoder_pf<S> D1_MAC_REF=mac_ref_pf<S>.json
#                                    D1_BENCH=bench_pf<S>.json, its bytes checked on the phone by the md5 stage
#                                    (D1_MD5SUMS=MD5SUMS,MD5SUMS_PF<S>). A bundle of the static form
#                                    (metadata language.contract.static) goes --into decoder_pf<S>_static (the files
#                                    then end in pf<S>_static: mac_ref_pf<S>_static.json, MD5SUMS_PF<S>_STATIC, ...)
#   ./_stage.sh --stripped <bundle> --like decoder_pf<S> [--tower <tower bundle>]
#                                    a bundle re-saved with its debug locations stripped (conversion/d1/strip_bundle.py:
#                                    the `strip` record in metadata.json), staged apart beside the bundle it was stripped
#                                    from: _work/device_stage_pf<S>_s/D1Assets/ with decoder_pf<S>_s/ (the bundle),
#                                    tower_s/ (--tower: the tower stripped the same way), the decoder_pf<S> stage's
#                                    fixtures/mac_ref_pf<S>.json and fixtures/bench_pf<S>.json unchanged (the same ops and
#                                    weights: the Mac's read-out of the unstripped bundle stays the reference) and
#                                    MD5SUMS_PF<S>_S (every file of the three). The strip record must name the bundle
#                                    staged as decoder_pf<S> and the sha256 of its main.mlirb (the tower's: tower/'s), and
#                                    every other file and field must equal that bundle's. Pushed with D1_PUSH_ONLY=
#                                    decoder_pf<S>_s,tower_s,fixtures/mac_ref_pf<S>.json,fixtures/bench_pf<S>.json,
#                                    MD5SUMS_PF<S>_S, read with D1_DECODER=decoder_pf<S>_s D1_TOWER=tower_s
#                                    D1_MAC_REF=mac_ref_pf<S>.json D1_BENCH=bench_pf<S>.json D1_MD5SUMS=MD5SUMS,
#                                    MD5SUMS_PF<S>_S
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
  shift
  (( $# )) || set -- $L/exports/bundles_aotc_ios/$NAME.h19p.aimodelc
  # D1_AOT_STAGE: another stage directory (_work/device_stage_aot<suffix>/D1Assets) to keep two AOT stages apart
  SA=${D1_AOT_STAGE:-$W/device_stage_aot/D1Assets}
  [[ $SA == */_work/device_stage_aot*/D1Assets ]] || { echo "refusing to clear $SA"; exit 1; }
  rm -rf $SA
  mkdir -p $SA/aot
  while (( $# )); do
    case $1 in
      --file)
        (( $# >= 3 )) || { echo "--file <src> <aot/rel>"; exit 1; }
        need $2
        [[ $3 == aot/* && $3 != *..* ]] || { echo "--file: the destination $3 is not under aot/"; exit 1; }
        mkdir -p $SA/${3:h}
        cp -c $2 $SA/$3
        shift 3 ;;
      *)
        [[ $1 == *.h1[0-9]p.*aimodelc || $1 == *.h1[0-9]p.aimodelc ]] || { echo "$1: not an iPhone (.h1Np.) AOT asset"; exit 1; }
        need $1/main.hash
        cp -cR $1 $SA/aot/${1:t}
        shift ;;
    esac
  done
  (cd $SA && find aot -type f | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > MD5SUMS_AOT)
  echo "staged $SA: $(grep -c . $SA/MD5SUMS_AOT) files, $(du -sh $SA | cut -f1)"
  cat $SA/MD5SUMS_AOT
  exit 0
fi

if [ "${1:-}" = "--stripped" ]; then
  B=${2:-} LIKE="" TB=""
  shift; (( $# )) && shift
  while (( $# >= 2 )); do
    case $1 in
      --like) LIKE=$2 ;;
      --tower) TB=$2 ;;
      *) echo "unknown argument $1"; exit 1 ;;
    esac
    shift 2
  done
  (( $# == 0 )) || { echo "a lone argument: $1"; exit 1; }
  [ -n "$B" ] && [ -n "$LIKE" ] || { echo "--stripped <bundle> --like decoder_pf<S> [--tower <tower bundle>]"; exit 1; }
  [[ $LIKE =~ '^decoder_pf[0-9]+$' ]] || { echo "--like $LIKE: want decoder_pf<S>"; exit 1; }
  SUF=${LIKE#decoder_}
  LS=$W/device_stage_$SUF/D1Assets
  M=$W/device_stage/D1Assets
  INTO=${LIKE}_s
  SD=$W/device_stage_${SUF}_s/D1Assets
  [[ $SD == */_work/device_stage_pf[0-9]*_s/D1Assets ]] || { echo "refusing to clear $SD"; exit 1; }
  BN=${B:t}
  for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json head/option_rows.json \
    head/option_rows.safetensors $BN.aimodel/main.mlirb $BN.aimodel/main.hash; do need $B/$f; done
  for f in $LIKE/metadata.json fixtures/mac_ref_$SUF.json fixtures/bench_$SUF.json MD5SUMS_${(U)SUF}; do need $LS/$f; done
  if [ -n "$TB" ]; then
    TBN=${TB:t}
    for f in metadata.json host/position_embedding.safetensors $TBN.aimodel/main.mlirb $TBN.aimodel/main.hash; do need $TB/$f; done
    need $M/tower/metadata.json
  fi
  junk=$(find $B ${TB:+$TB} \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
  [ -z "$junk" ] || { echo "$junk: a hidden or partial file (a copy still running?)"; exit 1; }
  # the strip record against the bundles staged before (decoder_pf<S>/, tower/): the source's main.mlirb by size and
  # sha256, this bundle's main.mlirb and main.hash, every other file and every other metadata field equal
  /usr/bin/python3 - $B $LS/$LIKE "${TB:-}" $M/tower <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def check(b: Path, like: Path, what: str) -> str:
    m, lm = json.load(open(b / "metadata.json")), json.load(open(like / "metadata.json"))
    st = m.get("strip")
    assert st, f"{what}: {b}/metadata.json has no strip record"
    assert st["from_bundle"] == lm["name"], f"{what}: stripped from {st['from_bundle']}, the staged bundle is {lm['name']}"
    src = like / lm["assets"]["main"] / "main.mlirb"
    out = b / m["assets"]["main"] / "main.mlirb"
    assert (src.stat().st_size, sha(src)) == (st["from_main_mlirb"]["bytes"], st["from_main_mlirb"]["sha256"]), \
        f"{what}: the strip record's source is not {src}"
    assert (out.stat().st_size, sha(out)) == (st["main_mlirb"]["bytes"], st["main_mlirb"]["sha256"]), \
        f"{what}: {out} is not the strip record's output"
    assert (out.parent / "main.hash").read_bytes().hex() == st["main_mlirb"]["sha256"], f"{what}: main.hash is not main.mlirb's sha256"
    rest = lambda d: {k: v for k, v in d.items() if k not in ("name", "assets", "strip")}
    assert rest(m) == rest(lm), f"{what}: metadata.json differs from {lm['name']}'s beyond name / assets / strip"
    # every file outside the .aimodel directory and metadata.json, by sha256
    mine ={str(p.relative_to(b)) for p in b.rglob("*") if p.is_file() and not str(p.relative_to(b)).startswith(m["assets"]["main"])}
    theirs = {str(p.relative_to(like)) for p in like.rglob("*") if p.is_file() and not str(p.relative_to(like)).startswith(lm["assets"]["main"])}
    assert mine == theirs, f"{what}: files {sorted(mine ^ theirs)} are in one bundle only"
    differ = [r for r in sorted(mine - {"metadata.json"}) if sha(b / r) != sha(like / r)]
    assert not differ, f"{what}: {differ} differ from {lm['name']}'s"
    return (f"{what} {m['name']}: stripped from the staged {lm['name']} (main.mlirb {st['from_main_mlirb']['bytes']} B "
            f"{st['from_main_mlirb']['sha256'][:12]}… -> {st['main_mlirb']['bytes']} B {st['main_mlirb']['sha256'][:12]}…), "
            f"{len(mine) - 1} other files equal")

print(check(Path(sys.argv[1]), Path(sys.argv[2]), "decoder"))
if sys.argv[3]:
    print(check(Path(sys.argv[3]), Path(sys.argv[4]), "tower"))
PY
  rm -rf $SD
  mkdir -p $SD/$INTO $SD/fixtures
  for e in metadata.json $BN.aimodel tokenizer head LICENSE; do [ -e $B/$e ] && cp -cR $B/$e $SD/$INTO/$e; done
  if [ -n "$TB" ]; then
    mkdir -p $SD/tower_s
    for e in metadata.json $TBN.aimodel host LICENSE; do [ -e $TB/$e ] && cp -cR $TB/$e $SD/tower_s/$e; done
  fi
  cp -c $LS/fixtures/mac_ref_$SUF.json $LS/fixtures/bench_$SUF.json $SD/fixtures/
  LST=MD5SUMS_${(U)SUF}_S
  (cd $SD && find $INTO ${TB:+tower_s} fixtures -type f | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > $LST)
  # the two fixture files are the ones the decoder_pf<S> stage put on the phone
  for f in fixtures/mac_ref_$SUF.json fixtures/bench_$SUF.json; do
    [ "$(grep " $f\$" $SD/$LST)" = "$(grep " $f\$" $LS/MD5SUMS_${(U)SUF})" ] || { echo "$f: md5 differs from the $LIKE stage's"; exit 1; }
  done
  echo "staged $SD: $(grep -c . $SD/$LST) files in $LST, $(du -sh $SD | cut -f1) (push: D1_PUSH_ONLY=$INTO${TB:+,tower_s},fixtures/mac_ref_$SUF.json,fixtures/bench_$SUF.json,$LST)"
  for d in $INTO ${TB:+tower_s} fixtures; do
    printf "  %-16s %10.1f MB  %4d files\n" $d $(( $(find $SD/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
      $(find $SD/$d -type f | wc -l)
  done
  exit 0
fi

if [ "${1:-}" = "--decoder" ]; then
  B=${2:-} INTO="" MT="" MI=""
  shift; (( $# )) && shift
  while (( $# >= 2 )); do
    case $1 in
      --into) INTO=$2 ;;
      --mac-text) MT=$2 ;;
      --mac-images) MI=$2 ;;
      *) echo "unknown argument $1"; exit 1 ;;
    esac
    shift 2
  done
  (( $# == 0 )) || { echo "a lone argument: $1"; exit 1; }
  [ -n "$B" ] && [ -n "$INTO" ] && [ -n "$MT" ] && [ -n "$MI" ] \
    || { echo "--decoder <bundle> --into decoder_pf<S> --mac-text <json> --mac-images <json>"; exit 1; }
  [[ $INTO == decoder_pf[0-9]* && $INTO != */* ]] || { echo "--into $INTO: want decoder_pf<S>"; exit 1; }
  BN=${B:t}
  for f in metadata.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json head/option_rows.json \
    head/option_rows.safetensors $BN.aimodel/main.mlirb $BN.aimodel/main.hash; do need $B/$f; done
  need $MT; need $MI; need $L/results/r7_timing_plan.json
  junk=$(find $B \( -name '.*' -o -name '*.incomplete' -o -name '*.lock' -o -name '*.aria2' \) -print -quit)
  [ -z "$junk" ] || { echo "$junk: a hidden or partial file (an export still running?)"; exit 1; }
  # the fixtures it shares with decoder/ (requests, oracle, red arms, the bench items) are the main stage's, as pushed
  M=$W/device_stage/D1Assets
  need $M/MD5SUMS; need $M/fixtures/bench.json
  SUF=${INTO#decoder_}
  SD=$W/device_stage_$SUF/D1Assets
  [[ $SD == */_work/device_stage_pf[0-9]*/D1Assets ]] || { echo "refusing to clear $SD"; exit 1; }
  rm -rf $SD
  mkdir -p $SD/$INTO $SD/fixtures
  for e in metadata.json $BN.aimodel tokenizer head LICENSE; do [ -e $B/$e ] && cp -cR $B/$e $SD/$INTO/$e; done
  /usr/bin/python3 - $L $B $MT $MI $M $SD $SUF <<'PY'
import hashlib, json, struct, sys
from pathlib import Path
L, B, MT, MI, M, SD = (Path(x) for x in sys.argv[1:7])
SUF = sys.argv[7]
meta = json.load(open(B / "metadata.json"))
S = meta["language"]["prefill_chunk"]
static = bool(meta["language"]["contract"].get("static"))
# the static form's bundle stages apart from the dynamic one of the same S: decoder_pf<S>_static
assert SUF == f"pf{S}" + ("_static" if static else ""), (SUF, S, static)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def bits(x):
    return format(struct.unpack("<Q", struct.pack("<d", float(x)))[0], "x")

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

# the main stage's shared fixtures, as staged (= the phone's D1Assets/fixtures)
listed = dict(reversed(l.split(" ", 1)) for l in (M / "MD5SUMS").read_text().splitlines() if " " in l)
for rel in ("fixtures/oracle_slim.json", "fixtures/bench.json"):
    assert md5(M / rel) == listed[rel], f"{rel} differs from the main stage's MD5SUMS"
oracle = json.load(open(M / "fixtures/oracle_slim.json"))["records"]

# the Mac's read-out of this bundle: its readout gate (text) and its decide.py run (pictures)
g = json.load(open(MT))
assert g.get("result") == "PASS", (MT, g.get("result"))
gb = (g.get("bundle") or {}).get("name")
assert gb == B.name, f"{MT}: the readout gate ran {gb}, not {B.name}"
mi = json.load(open(MI))
assert mi.get("pass") is True, (MI, mi.get("pass"))
assert all(Path(a[0]).name == B.name for a in mi["assets"]), f"{MI}: assets {mi['assets']} are not {B.name}"
mac = {}
for r in g["runs"]:
    if r.get("variant", "base") != "base":
        continue
    mac.setdefault(r["id"], {"rows": {}})["rows"][r["name"]] = {"hidden_sha256": r["hidden_sha256"],
                                                               "p_bits": [bits(x) for x in r["probs"]]}
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
ref_name = f"mac_ref_{SUF}.json"
json.dump({"schema": "d1-gate-mac-ref/1", "chunk": S, "static": static,
           "sources": {"mac_text": {"path": str(MT), "sha256": sha(MT)}, "mac_images": {"path": str(MI), "sha256": sha(MI)}},
           "assets": {"text": g.get("bundle"), "pictures": mi.get("assets")}, "records": mac},
          open(SD / "fixtures" / ref_name, "w"))

# the bench items of the main stage (round 9a's, unchanged) with the calls each makes at this S
plan = json.load(open(L / "results/r7_timing_plan.json"))
bench = json.load(open(M / "fixtures/bench.json"))
calls = {k: v[str(S)] for k, v in plan["calls"].items()}
for it in bench["bench"] + bench["bench_aot"]:
    assert it["name"] in calls, it["name"]
    it["calls_expected"] = calls[it["name"]]
bench["chunk"] = S
bench["static"] = static
# the static form's calls are the dynamic form's at the same S (round 10b's plan of both forms)
bench["source"] += f"; the same items at S = {S} (calls_expected from results/r7_timing_plan.json, sha256 {sha(L / 'results/r7_timing_plan.json')[:16]})"
json.dump(bench, open(SD / "fixtures" / f"bench_{SUF}.json", "w"), indent=1)
nrows = sum(len(v["rows"]) for v in mac.values())
print(f"S = {S}: {ref_name} {len(mac)} records, {nrows} rows (text {MT.name}, pictures {MI.name}); bench_{SUF}.json "
      f"{[(it['name'], it['calls_expected']) for it in bench['bench']]}")
PY
  LST=MD5SUMS_${(U)SUF}
  (cd $SD && find $INTO fixtures -type f | LC_ALL=C sort | while read -r f; do nice -n 19 md5 -r "$f"; done > $LST)
  echo "staged $SD: $(grep -c . $SD/$LST) files in $LST, $(du -sh $SD | cut -f1) (push: D1_PUSH_ONLY=$INTO,fixtures/mac_ref_$SUF.json,fixtures/bench_$SUF.json,$LST)"
  for d in $INTO fixtures; do
    printf "  %-14s %10.1f MB  %4d files\n" $d $(( $(find $SD/$d -type f -exec stat -f %z {} + | paste -sd+ - | bc) / 1e6 )) \
      $(find $SD/$d -type f | wc -l)
  done
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
