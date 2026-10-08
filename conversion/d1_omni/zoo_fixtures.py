#!/usr/bin/env python3
"""Write models/d1-omni-600m/fixtures-d1-omni-600m.json: the public fixture records and the publisher's fp32 numbers.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/zoo_fixtures.py            # write it (refuses to replace a different file without --force)
    $PY conversion/d1_omni/zoo_fixtures.py --check    # exit 1 unless the file = a fresh build = the HF staging's reference/

What it holds (schema `coreai-d1-omni-fixtures/1`): the 237 public records of the lane's fixture v3 (fixtures/records.json,
sha256 pinned) exactly as written there (request, media, gold, note, provenance), the licence notices of the two MIT
sources whose text the records carry (SemIf authored144, MMLU via cais/mmlu), the 18 media files by name, bytes and
sha256 (the files themselves are in the Hugging Face repository's reference/), and one row per question of every public
record in text mode and, when it has media, in its media mode (360 rows): the token ids and marker positions, the media
prefix length P, the positions P + len(ids), the decision bucket (host.ALL_BUCKETS, the shipped set; and host.BUCKETS,
the set before round 12), the temperature, and the publisher's fp32 CPU numbers (`D1OmniModel.probabilities`, the
oracle files ref/records_ref*.json: raw marker logits, probabilities before and after the temperature, the argmax, the
top-2 margin, the near-tie flag, the response). No model runs here.

The schema is its own, not `coreai-encoder-fixtures/1` (laya's): coreai-kit's `decide-cli parity` sends any schema
that starts with "coreai-encoder-fixtures" through laya's prompt builder (Examples/Decide/CLI/main.swift), which is not
this model's. The row fields keep laya's names where they mean the same thing (row_id, fixture_id, window, question_id,
question, sequence_ids, marker_positions, qtype, K, sequence_length, raw_logits, probabilities, official_answer).

--check rebuilds the document and asserts: the file equals the build; every record equals the staged
reference/records.json record of the same id (sha256 of its canonical JSON) and the two record sets are the same; every
media file's sha256 equals the staged reference/ file's; every row's ids, markers, raw logits and probabilities equal
the staged reference/oracle.json question. It prints a JSON verdict (zoo_transcripts.py --check writes it into
results/zoo_fixtures_check.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import repo_root, work_path  # noqa: E402
import host  # noqa: E402

MODEL_ID = "LiquidAI/d1-omni-600M"
MODEL_SHA = "414f8d6438174f5b2133a9c21a478fc42625e308"
SCHEMA = "coreai-d1-omni-fixtures/1"
WORK = work_path("_d1_omni")
OUT = repo_root() / "models" / "d1-omni-600m" / "fixtures-d1-omni-600m.json"
STAGING = WORK / "hf_staging" / "d1-omni-600M-CoreAI"
FIXTURES = ("fixtures/records.json", "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b")  # v3, round 6
ORACLES = (  # latest first: a (record, mode) is taken from the first file that has it (stage_hf.py's rule)
    ("ref/records_ref_audio.json", "c483f8c511cfcdb05a8211e495a7476b11f9318899a1125c387c9e722a6d8fbc"),
    ("ref/records_ref_images.json", "681d2d10eef35ecb8980a3caf93d922a77a17e88ee58795468e78b915ac91b08"),
    ("ref/records_ref.json", "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"),
)
ENCODE_V3 = "results/encode_rows.v3.json"  # the publisher's prompt.encode on every v3 row (round 6)
LICENSES = {  # the MIT notices of the sources whose text the public records carry
    "semif_authored144": ("fixtures/LICENSE-SemIf", "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be"),
    "mmlu": ("fixtures/LICENSE-MMLU-MIT.txt", "5d841b9eee8bf0d721359b79fbae956a8b426d3139bfefee32149c7d4f2a1a9c"),
}
QTYPE = {"choice": 0, "score": 1, "noul": 2}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def canon(value) -> str:
    """The canonical JSON a record's sha256 is taken over (sorted keys, no spaces, UTF-8)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_of(value) -> str:
    return hashlib.sha256(canon(value).encode()).hexdigest()


def pinned(rel: str, sha: str) -> Path:
    path = WORK / rel
    if sha256_file(path) != sha:
        raise SystemExit(f"{rel}: sha256 differs from its pin {sha[:12]}")
    return path


def media_of(record: dict) -> list[tuple[str, str]]:
    """(file relative to the lane, sha256 from the record's provenance) for each image / clip of a record."""
    out = []
    media = record.get("media") or {}
    for rel in media.get("images", []):
        out.append((rel, record["provenance"]["image"]["sha256"]))
    if media.get("audio"):
        prov = record["provenance"]
        out.append((media["audio"], (prov.get("clip") or prov.get("audio") or {})["sha256"]))
    return out


def oracle_rows(public_ids: set[str]) -> tuple[dict, list[dict]]:
    """(record, mode) -> the oracle record, each from the latest oracle file that has it; and the files' metadata."""
    enc = {(r["id"], r["mode"], r["qid"]): r["ids_sha256"] for r in json.loads((WORK / ENCODE_V3).read_text())["rows"]}
    rows, metas, bad = {}, [], []
    for rel, sha in ORACLES:
        doc = json.loads(pinned(rel, sha).read_text())
        metas.append({"file": Path(rel).name, "sha256": sha, "written": doc["written"],
                      "fixtures_sha256": doc["fixtures"]["sha256"]})
        for rec in doc["records"]:
            key = (rec["id"], rec["mode"])
            if rec["id"] not in public_ids or key in rows:
                continue
            for q in rec["questions"]:
                if enc.get((rec["id"], rec["mode"], q["qid"])) != hashlib.sha256(json.dumps(q["ids"]).encode()).hexdigest():
                    bad.append(f"{rec['id']}/{rec['mode']}/{q['qid']}")
            rows[key] = rec | {"oracle_file": Path(rel).name}
    if bad:
        raise SystemExit(f"oracle rows whose ids differ from the v3 host rows (encode_rows.v3.json): {bad[:10]}")
    return rows, metas


def build() -> dict:
    fixtures = json.loads(pinned(*FIXTURES).read_text())
    public = [r for r in fixtures["records"] if r["public"] is True]
    ids = {r["id"] for r in public}
    oracle, oracle_files = oracle_rows(ids)
    want = {(r["id"], "text") for r in public} | {(r["id"], "image" if "images" in r["media"] else "audio")
                                                    for r in public if r.get("media")}
    if set(oracle) != want:
        raise SystemExit(f"oracle rows missing {sorted(want - set(oracle))[:10]}, extra {sorted(set(oracle) - want)[:10]}")
    oracle_model = json.loads((WORK / ORACLES[-1][0]).read_text())
    by_id = {r["id"]: r for r in public}

    rows = []
    for key in sorted(oracle):
        rec = oracle[key]
        record = by_id[rec["id"]]
        questions = record["request"]["questions"]
        answers = rec["response"]["answers"]
        for q in rec["questions"]:
            positions = q["prefix"] + q["n_ids"]
            if positions != q["positions"] or q["bucket"] != host.bucket_for(positions):
                raise SystemExit(f"{rec['id']}/{rec['mode']}/{q['qid']}: positions / bucket disagree with host.py")
            rows.append({
                "row_id": f"{rec['id']}/{rec['mode']}/{q['qid']}", "fixture_id": rec["id"], "mode": rec["mode"],
                "question_id": q["qid"], "question": questions[q["qid"]], "type": q["type"], "qtype": QTYPE[q["type"]],
                "K": q["K"], "media": [rel for rel, _ in media_of(record)] if rec["mode"] != "text" else [],
                "prefix_rows": q["prefix"], "sequence_ids": q["ids"], "sequence_length": q["n_ids"],
                "marker_positions": q["markers"], "positions": positions,
                "window": host.bucket_for(positions, host.ALL_BUCKETS), "window_r10": q["bucket"],
                "temperature_key": q["temperature_key"], "temperature": q["T"],
                "raw_logits": q["logits_raw"], "probabilities_raw": q["probs_raw"], "probabilities": q["probs"],
                "argmax_index": q["argmax_index"], "argmax": q["argmax"], "top2_margin": q["top2_margin"],
                "near_tie": q["near_tie"], "gold": q["gold"], "official_answer": answers[q["qid"]],
                "usage_input_tokens": rec["usage_input_tokens"], "oracle_file": rec["oracle_file"],
            })

    media = {}
    for r in public:
        for rel, sha in media_of(r):
            path = WORK / rel
            if sha256_file(path) != sha:
                raise SystemExit(f"{rel}: differs from its record's sha256")
            media[rel] = {"file": rel, "bytes": path.stat().st_size, "sha256": sha}

    src = fixtures["sources"]
    licenses = {}
    for name, (rel, sha) in LICENSES.items():
        licenses[name] = {"file": Path(rel).name, "sha256": sha, "text": pinned(rel, sha).read_text()}
    mmlu_prov = json.loads((WORK / "fixtures" / "LICENSE-MMLU-MIT.json").read_text())
    tv4 = src["transfer_v4"]
    sources = {
        "card": {k: src["card"][k] for k in ("file", "sha256", "section", "repo", "revision")},
        "semif_authored144": {k: v for k, v in src["semif_authored144"].items() if not k.startswith("license")}
                             | {"license": "MIT, Copyright (c) 2026 TheoLeeCJ", "license_notice": licenses["semif_authored144"]},
        "transfer_v4": {k: tv4[k] for k in ("path_in_repo", "sha256", "lines", "author_repo")}
                       | {"records_here": "tv4_000 ... tv4_059: the file's first 60 lines (all mmlu)",
                          "mmlu": {"hf_id": mmlu_prov["dataset"]["hf_id"], "revision": mmlu_prov["dataset"]["revision"],
                                   "split": mmlu_prov["dataset"]["split"],
                                   "license": "MIT, Copyright (c) 2020 Dan Hendrycks",
                                   "license_from": mmlu_prov["url"], "license_notice": licenses["mmlu"]}},
        "own": {"license": "written for this port (Apache-2.0, as the conversion code)",
                "records": sorted(r["id"] for r in public if r["source"] in ("own_text", "own_json", "own_long",
                                                                            "own_long_extended"))},
        "images": {"what": "img_01: own model output (FLUX.2 klein 4B, Apache-2.0); img_02, img_03: CC0 1.0 photographs "
                           "(Wikimedia Commons), downscaled; origin and resize in the HF repository's "
                           "reference/images/manifest.json"},
        "audio": {"what": "aud_01 ... aud_15: speech synthesized with hexgrad/Kokoro-82M (Apache-2.0) from scripts written "
                          "for these ports; the scripts in the HF repository's reference/audio/script.json"},
    }

    summary = {"records": len(public), "request_modes": len(oracle), "rows": len(rows),
               "rows_by_mode": {m: sum(r["mode"] == m for r in rows) for m in ("text", "image", "audio")},
               "rows_by_window": {str(L): sum(r["window"] == L for r in rows) for L in host.ALL_BUCKETS},
               "rows_by_window_r10": {str(L): sum(r["window_r10"] == L for r in rows) for L in host.BUCKETS},
               "near_ties": [r["row_id"] for r in rows if r["near_tie"]],
               "by_source": {}}
    for r in public:
        s = summary["by_source"].setdefault(r["source"], {"records": 0, "questions": 0})
        s["records"] += 1
        s["questions"] += len(r["request"]["questions"])

    return {
        "schema": SCHEMA,
        "what": ("The public records of the d1-omni-600M Core AI port's fixture (v3) and the publisher's fp32 CPU numbers "
                 "on every question of them, in text mode and, for a record with an image or a clip, in that media mode. "
                 "The records are the HF repository's reference/records.json records, the rows its reference/oracle.json "
                 "questions; the images and clips are named here and stored there (reference/images/, reference/audio/)."),
        "model": {"hf_id": MODEL_ID, "revision": MODEL_SHA, "license": "LFM Open License v1.0",
                  "oracle": {"reference": oracle_model["reference"], "source_sha256": oracle_model["model"]["source_sha256"],
                             "weights_sha256": oracle_model["model"]["weights_sha256"],
                             "versions": oracle_model["versions"]},
                  "oracle_files": oracle_files},
        "contract": {
            "row": "ids = [1, 17] + enc(state)[:room] + [18] + enc(instructions)[:max(16, budget)] + per option ([19, 16] "
                   "+ enc(' ' + text)[:per] + [20]) + [21]; enc(s) = the tokenizer's ids of escape(s) without special "
                   "tokens; marker_positions = the positions of the 16s in ids (host.py, the publisher's prompt.encode)",
            "graph": "positions 0 .. P-1 = the media prefix (P = prefix_rows), P .. P+len(ids)-1 = ids; the decision graph "
                     "of the smallest window (L) that holds P + len(ids) returns a score per position; the host reads "
                     "scores[P + marker] (host.py graph_inputs, probabilities_from_logits)",
            "window": "host.bucket_for(positions, host.ALL_BUCKETS) = the shipped decision buckets (64 ... 4096); "
                      "window_r10 = host.bucket_for(positions) over (256 ... 4096), the set before round 12",
            "raw_logits": "the head's marker logits in the model's option order (noul: [false, true])",
            "probabilities_raw": "softmax(raw_logits), the model's option order, no temperature",
            "probabilities": "the reported probabilities: text rows softmax(raw_logits / temperature) with the temperature of "
                             "temperature_key from config.json (type default 1.0); image and audio rows no temperature; "
                             "noul reported as [yes, no] = [true, false] (the model's order reversed)",
            "argmax": "the option key at argmax_index of probabilities (noul: 'true' or 'false')",
            "near_tie": "the oracle's top-2 probability margin <= 0.02 (FACTS §7: argmax compared apart)",
            "official_answer": "the publisher's response for this question (`system_one` answer shape)",
            "qtype": QTYPE,
        },
        "sources": sources,
        "media": [media[k] for k in sorted(media)],
        "fixtures": public,
        "rows": rows,
        "summary": summary,
        "built_from": {rel: {"bytes": (WORK / rel).stat().st_size, "sha256": sha256_file(WORK / rel)}
                       for rel in (FIXTURES[0], ENCODE_V3, *(o[0] for o in ORACLES), *(lic[0] for lic in LICENSES.values()))},
    }


def text_of(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def check(doc: dict, staging: Path = STAGING) -> dict:
    """The fixture file against a fresh build and against the HF staging's reference/ (see the module docstring)."""
    out = {"what": "fixtures-d1-omni-600m.json against a fresh build and the HF staging's reference/ (records by sha256 of "
                   "their canonical JSON, media by file sha256, rows by ids / markers / raw logits / probabilities)",
           "file": str(OUT.relative_to(repo_root())), "staging": "<work>/hf_staging/d1-omni-600M-CoreAI/reference"}
    text = text_of(doc)
    out["file_equals_build"] = OUT.exists() and OUT.read_text() == text
    out["file_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    staged = json.loads((staging / "reference" / "records.json").read_text())
    s_by_id = {r["id"]: sha_of(r) for r in staged["records"]}
    f_by_id = {r["id"]: sha_of(r) for r in doc["fixtures"]}
    diff = sorted(i for i in set(s_by_id) | set(f_by_id) if s_by_id.get(i) != f_by_id.get(i))
    out["records"] = {"fixture": len(f_by_id), "staged": len(s_by_id), "equal": len(f_by_id) - len(diff) if not diff else
                      sum(s_by_id.get(i) == h for i, h in f_by_id.items()), "different_or_missing": diff[:20],
                      "set_sha256_fixture": sha_of(sorted(f_by_id.items())), "set_sha256_staged": sha_of(sorted(s_by_id.items())),
                      "staged_fixtures_sha256": staged.get("fixtures_sha256")}
    media_bad = []
    for m in doc["media"]:
        f = staging / "reference" / m["file"]
        if not f.exists() or sha256_file(f) != m["sha256"] or f.stat().st_size != m["bytes"]:
            media_bad.append(m["file"])
    staged_media = sorted(str(p.relative_to(staging / "reference")) for p in (staging / "reference").rglob("*")
                          if p.suffix in (".png", ".wav"))
    out["media"] = {"fixture": len(doc["media"]), "staged": len(staged_media), "sha256_equal": len(doc["media"]) - len(media_bad),
                    "different": media_bad, "same_names": staged_media == sorted(m["file"] for m in doc["media"])}
    oracle = json.loads((staging / "reference" / "oracle.json").read_text())
    s_rows = {}
    for rec in oracle["records"]:
        for q in rec["questions"]:
            s_rows[f"{rec['id']}/{rec['mode']}/{q['qid']}"] = q
    row_bad = []
    for r in doc["rows"]:
        q = s_rows.get(r["row_id"])
        if q is None or (q["ids"], q["markers"], q["logits_raw"], q["probs_raw"], q["probs"], q["prefix"]) != (
                r["sequence_ids"], r["marker_positions"], r["raw_logits"], r["probabilities_raw"], r["probabilities"],
                r["prefix_rows"]):
            row_bad.append(r["row_id"])
    out["rows"] = {"fixture": len(doc["rows"]), "staged": len(s_rows), "equal": len(doc["rows"]) - len(row_bad),
                   "different_or_missing": row_bad[:20], "same_ids": sorted(s_rows) == sorted(r["row_id"] for r in doc["rows"])}
    out["status"] = "PASS" if (out["file_equals_build"] and not diff and len(f_by_id) == len(s_by_id)
                               and not media_bad and out["media"]["same_names"] and not row_bad
                               and out["rows"]["same_ids"]) else "FAIL"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare the file with a build and the HF staging; exit 1 on FAIL")
    parser.add_argument("--force", action="store_true", help="replace a file that differs from the build")
    parser.add_argument("--staging", type=Path, default=STAGING)
    args = parser.parse_args()
    doc = build()
    if args.check:
        result = check(doc, args.staging)
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    text = text_of(doc)
    if OUT.exists() and OUT.read_text() != text and not args.force:
        raise SystemExit(f"{OUT} differs from the build: pass --force to replace it")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(json.dumps({"file": str(OUT.relative_to(repo_root())), "bytes": len(text.encode()),
                      "sha256": hashlib.sha256(text.encode()).hexdigest()} | doc["summary"] | {"near_ties": len(doc["summary"]["near_ties"])},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
