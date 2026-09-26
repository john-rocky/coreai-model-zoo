"""Lay out the Hugging Face repo mlboydaisuke/GLiNER2.5-Decide-CoreAI in <work>/_gliner25_decide/ship/ file for file,
as it would be uploaded (APFS clones of the gated artifacts), check the staged files against the records of the gates
that ran them, write SHA256SUMS over every file and print the size table. Nothing is uploaded here.

    ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py                # stage, check, SHA256SUMS, sizes
    ~/code/coreai/coreai-models/.venv/bin/python stage_ship.py --check <dir>  # re-hash <dir> against its SHA256SUMS

--check is for a download of the published repo: every listed file present with its sha256, and nothing else
(the Hub's own .gitattributes and a local download's .cache/ excepted). Exit 0 = identical.

What staging checks, on the staged copies, before it writes SHA256SUMS:
  macos/  the exported bundles the Mac gates ran (export_gliner25_decide.py GATE-3, decide-selftest, the kit's
          TextClassifier gate): size equal to the gate record, main.hash == sha256(main.mlirb).
  ios/    the h19p AOT bundles (aot_compile.py) the iPhone 18 Pro gate ran: every file's md5 equal to the MD5SUMS
          pushed to the phone (apps/DecideGate), metadata sourceHash equal to the macos bundle's main.hash.
  gate/   the oracle fixtures both gates read: md5 equal to the same MD5SUMS, model and dataset revisions pinned.
  tokenizer/  tokenizer.json and special_tokens_map.json equal to the source snapshot; tokenizer_config.json the
          source's with one change (tokenizer_class XLMRobertaTokenizer), serialized as write_side_files does.
The ios/ layer holds h19p only (iPhone 18 Pro, iPhone19,2). An h18p compile (iPhone 17 Pro) exists in _work/aot/ but
has not run on a device, so it is not staged. The card is the one file this script does not write: ship/README.md
starts as a one-line placeholder and survives re-runs; run the script again once the card is in place, so that
SHA256SUMS covers it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # importing conversion/_paths must not leave a __pycache__ outside this dir
sys.path.insert(0, str(HERE.parent))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_REPO = "mlboydaisuke/GLiNER2.5-Decide-CoreAI"
REPO_ID = "fastino/GLiNER2.5-Decide"
REVISION = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
MODEL_SHA256 = "40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997"
DATASET_ID = "fastino/fast-decisions"
DATASET_REV = "1a33070cabf94ce2e29105482dd2ef6c157ad7f2"
# The Apache License 2.0 text as GLiNER2 distributes it (its GitHub LICENSE, byte-equal to the gliner2 2.0.0 wheel's).
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MARKER_SHA256 = "8bb12dab02fee45172391da555fb2ba75d2d341e69edb574e59412f2c801f6a7"  # = GLiNER2-PII-CoreAI's config.json
ARCH, DEVICE = "h19p", "iPhone 18 Pro (iPhone19,2)"
SHAPES = (256, 512)
STEM = "gliner25-decide_float16_s{S}_m32"
FIXTURES = {"readme21": (21, 26), "fast_decisions_s256": (340, 580), "fast_decisions_long": (93, 181)}  # cases, tasks
TOKENIZER = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
PLACEHOLDER = "CARD_PLACEHOLDER\n"
IGNORED_ON_CHECK = (".gitattributes",)  # created by the Hub itself; .cache/ is a local download's bookkeeping

GEN = work_path("_gliner25_decide")                        # the oracle's --work-dir, the export's --output-dir parent
EXPORTS, FIXTURE_DIR, RESULTS = GEN / "exports", GEN / "fixtures", GEN / "results"
LEGAL = Path(os.environ.get("GLINER25_SHIP_LEGAL", GEN / "legal"))           # LICENSE (sha256 pinned above)
AOT = HERE / "_work" / "aot"
DEVICE_MD5SUMS = HERE / "_work" / "device_stage" / "DecideAssets" / "MD5SUMS"  # written by apps/DecideGate/_stage.sh

MARKER = {  # the zoo's generic config.json
    "model_type": "coreai-aimodel",
    "format": "aimodel",
    "framework": "Apple Core AI (iOS 27 / macOS 27)",
    "repository": "https://github.com/john-rocky/coreai-model-zoo",
    "note": "Converted .aimodel bundles for Apple's Core AI framework. See README.md for the bundle layout and run "
            "instructions.",
}

NOTICE = f"""\
GLiNER2.5-Decide-CoreAI
Core AI (.aimodel) conversion of the classification path of Fastino's GLiNER2.5-Decide.

Origin
  Model:     {REPO_ID}
  Source:    https://huggingface.co/{REPO_ID}
  Revision:  {REVISION} (2026-09-24)
  Weights:   model.safetensors, sha256 {MODEL_SHA256}
  Provider:  Fastino
  License:   Apache License 2.0 (declared on the model card; the source repository has no LICENSE file)
  Base:      microsoft/deberta-v3-large (MIT License)

Software used
  gliner2 2.0.0 (Apache License 2.0): the fp32 reference the conversion was checked against.
  transformers 4.57.6 (Apache License 2.0): DebertaV2Model, into which the encoder weights were loaded.

Conversion
  The released fp32 weights were loaded strictly into transformers' DebertaV2Model and the classification head
  (Linear 1024->2048, ReLU, Linear 2048->1) and exported as one static graph with Apple coreai-torch 0.4.1 /
  coreai-core 1.0.0b2, at two sequence lengths (256 and 512). The bundles store the released weights in half
  precision. No weights were retrained, pruned or quantized. The span and count heads, which classification does
  not use, are not in the graph. DeBERTa-v3's relative-position bucket table is computed in PyTorch at export time
  and stored in each graph as a constant. The ios/ bundles are the same graphs compiled ahead of time with
  coreai-build for the iPhone 18 Pro (h19p). Converted and published by mlboydaisuke
  (https://huggingface.co/mlboydaisuke), 2026-09.

Changed files
  tokenizer/tokenizer_config.json is the source file with one change: tokenizer_class DebertaV2Tokenizer ->
  XLMRobertaTokenizer, so that swift-transformers loads the same SentencePiece Unigram model. tokenizer.json and
  special_tokens_map.json are unchanged. source/config.json and source/encoder_config.json are the source
  repository's config.json and encoder_config/config.json, unchanged.

Test fixtures (gate/)
  readme21.json: the 21 classify_text examples of the model card at the revision above.
  fast_decisions_s256.json, fast_decisions_long.json: rows of {DATASET_ID} (Apache License 2.0), revision
  {DATASET_REV}.
  Each case carries gliner2 2.0.0's fp32 inputs, logits and decisions.

This distribution is provided under the Apache License 2.0 (see LICENSE). When redistributing any part of it,
keep this NOTICE and the LICENSE file with it.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files_under(p: Path) -> list[Path]:
    return [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())


def size(p: Path) -> tuple[int, int]:
    fs = files_under(p)
    return sum(f.stat().st_size for f in fs), len(fs)


def sources(snap: Path) -> dict[str, Path]:
    """repo path -> the gated artifact it is a clone of."""
    out: dict[str, Path] = {}
    for S in SHAPES:
        stem = STEM.format(S=S)
        out[f"macos/{stem}.aimodel"] = EXPORTS / f"{stem}.aimodel"
        out[f"ios/{stem}.{ARCH}.aimodelc"] = AOT / f"s{S}_ios_gpu_{ARCH}" / f"{stem}.{ARCH}.aimodelc"
    for layer in ("macos", "ios"):
        for name in TOKENIZER:
            out[f"{layer}/tokenizer/{name}"] = EXPORTS / "tokenizer" / name
        for S in SHAPES:
            out[f"{layer}/reference_s{S}.json"] = EXPORTS / f"reference_s{S}.json"
    out["macos/classifier.json"] = EXPORTS / "classifier.json"
    for name in FIXTURES:
        out[f"gate/{name}.json"] = FIXTURE_DIR / f"{name}.json"
    out["source/config.json"] = snap / "config.json"
    out["source/encoder_config.json"] = snap / "encoder_config" / "config.json"
    out["LICENSE"] = LEGAL / "LICENSE"
    return out


def check_staged(ship: Path, snap: Path) -> list[str]:
    """The staged copies against the records of the gates that ran them. Returns the lines to print."""
    said = []
    assert sha256(ship / "LICENSE") == LICENSE_SHA256, "LICENSE is not the pinned Apache-2.0 text"
    assert sha256(ship / "config.json") == MARKER_SHA256, "config.json is not the zoo's generic marker"
    got = sha256(snap / "model.safetensors")
    assert got == MODEL_SHA256, f"snapshot model.safetensors sha256 {got} != {MODEL_SHA256} (NOTICE states it)"
    said.append(f"source snapshot {REVISION[:7]}: model.safetensors sha256 {MODEL_SHA256[:12]}… as NOTICE states; "
                f"LICENSE sha256 {LICENSE_SHA256[:12]}…; config.json = the zoo marker")

    tcfg = json.loads((snap / "tokenizer_config.json").read_text())
    assert tcfg["tokenizer_class"] == "DebertaV2Tokenizer", tcfg["tokenizer_class"]
    tcfg["tokenizer_class"] = "XLMRobertaTokenizer"                      # export_gliner25_decide.write_side_files
    want_cfg = json.dumps(tcfg, indent=2, ensure_ascii=False).encode()
    for layer in ("macos", "ios"):
        tok = ship / layer / "tokenizer"
        for name in ("tokenizer.json", "special_tokens_map.json"):
            assert sha256(tok / name) == sha256(snap / name), f"{layer}/tokenizer/{name} != the source snapshot's"
        assert (tok / "tokenizer_config.json").read_bytes() == want_cfg, \
            f"{layer}/tokenizer/tokenizer_config.json is not the source's with only tokenizer_class changed"
    said.append("tokenizer/ (both layers): tokenizer.json + special_tokens_map.json = source snapshot; "
                "tokenizer_config.json = source with tokenizer_class XLMRobertaTokenizer, byte-exact to write_side_files")
    assert (ship / "source" / "config.json").read_bytes() == (snap / "config.json").read_bytes()
    assert (ship / "source" / "encoder_config.json").read_bytes() == (snap / "encoder_config" / "config.json").read_bytes()

    cfg = json.loads((ship / "macos" / "classifier.json").read_text())
    assert cfg["revision"] == REVISION and cfg["MMAX"] == 32 and cfg["dtype"] == "float16", cfg
    assert [(s["S"], s["bundle"]) for s in cfg["shapes"]] == \
        [(S, f"{STEM.format(S=S)}.aimodel") for S in SHAPES], cfg["shapes"]
    icfg = json.loads((ship / "ios" / "classifier.json").read_text())
    assert (icfg["arch"], icfg["device"]) == (ARCH, DEVICE)
    device = {}
    if DEVICE_MD5SUMS.is_file():
        for line in DEVICE_MD5SUMS.read_text().splitlines():
            digest, rel = line.split(" ", 1)
            device[rel] = digest
    for S, shape, ishape in zip(SHAPES, cfg["shapes"], icfg["shapes"]):
        stem = STEM.format(S=S)
        mac, ios = ship / "macos" / f"{stem}.aimodel", ship / "ios" / f"{stem}.{ARCH}.aimodelc"
        b, _ = size(mac)
        assert b == shape["bytes"], f"{mac.name}: {b} B != classifier.json {shape['bytes']}"
        main_hash = (mac / "main.hash").read_bytes().hex()
        assert main_hash == sha256(mac / "main.mlirb"), f"{mac.name}: main.hash != sha256(main.mlirb)"
        gate = RESULTS / f"gate_s{S}_gpu.json"
        if gate.is_file():
            head = json.loads(gate.read_text())["header"]
            assert head["pass"] and Path(head["bundle"]).name == mac.name and head["bundle_bytes"] == b, \
                f"{gate.name}: the Mac GPU gate record does not match {mac.name}"
            said.append(f"macos/{mac.name}: {b:,} B = {gate.name} (pass); main.hash = sha256(main.mlirb)")
        else:
            said.append(f"macos/{mac.name}: {b:,} B; main.hash = sha256(main.mlirb); {gate} absent, the gate record "
                        f"was NOT checked")
        meta = json.loads((ios / "metadata.json").read_text())
        assert meta["sourceHash"].lower() == main_hash, f"{ios.name}: sourceHash is not {mac.name}'s main.hash"
        aot_rec = json.loads((AOT / f"aot_compile_s{S}.json").read_text())["targets"][f"ios_gpu_{ARCH}"]
        bi, _ = size(ios)
        assert aot_rec["exit"] == 0 and aot_rec["out_bytes"] == bi and aot_rec["ane_regions"]["n_regions"] == 0, \
            f"{ios.name}: not the aot_compile_s{S}.json record"
        assert (ishape["S"], ishape["bundle"], ishape["bytes"]) == (S, ios.name, bi), ishape
        if device:
            rels = {f.relative_to(ios.parent).as_posix() for f in files_under(ios)}
            listed = {r for r in device if r.startswith(f"{ios.name}/")}
            assert rels == listed, f"{ios.name}: files differ from the device MD5SUMS ({sorted(rels ^ listed)})"
            bad = [r for r in sorted(rels) if md5(ios.parent / r) != device[r]]
            assert not bad, f"{ios.name}: md5 differs from the bundle the iPhone gate ran: {bad}"
            said.append(f"ios/{ios.name}: {bi:,} B; {len(rels)} files md5 = the device MD5SUMS (iPhone 18 Pro gate); "
                        f"sourceHash = the macos main.hash; aot_compile_s{S}.json exit 0, ANE regions 0")
        else:
            said.append(f"ios/{ios.name}: {bi:,} B; sourceHash = the macos main.hash; {DEVICE_MD5SUMS} absent, the "
                        f"device md5 was NOT checked")
        for layer in ("macos", "ios"):
            ref = json.loads((ship / layer / f"reference_s{S}.json").read_text())
            assert ref["S"] == S and ref["revision"] == REVISION and ref["MMAX"] == 32, f"{layer}/reference_s{S}.json"

    for name, (n_cases, n_tasks) in FIXTURES.items():
        p = ship / "gate" / f"{name}.json"
        fx = json.loads(p.read_text())
        h = fx["header"]
        assert h["model_rev"] == REVISION and h["model_safetensors_sha256"] == MODEL_SHA256, f"{name}: model pin"
        assert h["set"] == name and len(fx["cases"]) == n_cases, f"{name}: {len(fx['cases'])} cases"
        assert sum(len(c["task_results"]) for c in fx["cases"]) == n_tasks, f"{name}: task count"
        assert h["self_check"]["decisions_equal_tasks"] == n_tasks, f"{name}: oracle self-check"
        if name.startswith("fast_decisions"):
            assert h["dataset_id"] == DATASET_ID and h["dataset_rev"] == DATASET_REV, f"{name}: dataset pin"
        if device:
            assert md5(p) == device[f"fixtures/{name}.json"], f"{name}.json: md5 differs from the device gate's"
        said.append(f"gate/{name}.json: {n_cases} cases / {n_tasks} tasks, model rev pinned"
                    + (f", dataset rev {DATASET_REV[:8]}" if name.startswith("fast_decisions") else "")
                    + (", md5 = the device MD5SUMS" if device else ", the device md5 was NOT checked"))
    return said


def ios_classifier(ship: Path) -> str:
    """macos/classifier.json with each shape pointing at its h19p compile, and the target named."""
    cfg = json.loads((ship / "macos" / "classifier.json").read_text())
    out = {}
    for k, v in cfg.items():
        out[k] = v
        if k == "shapes":
            for s in v:
                s["bundle"] = s["bundle"].removesuffix(".aimodel") + f".{ARCH}.aimodelc"
                s["bytes"] = size(ship / "ios" / s["bundle"])[0]
            out["arch"], out["device"] = ARCH, DEVICE
    return json.dumps(out, indent=2, ensure_ascii=False)


def stage(ship: Path) -> None:
    snap = Path(hf_snapshot(REPO_ID, revision=REVISION))
    src = sources(snap)
    missing = [f"{k} <- {v}" for k, v in src.items() if not v.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(missing))
    ship.mkdir(parents=True, exist_ok=True)
    (ship / "SHA256SUMS").unlink(missing_ok=True)       # never leave the previous run's sums beside a changed tree
    card = ship / "README.md"
    if not card.is_file():
        card.write_text(PLACEHOLDER)
    for name, path in src.items():
        dst = ship / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        subprocess.run(["cp", "-cR" if path.is_dir() else "-c", str(path), str(dst)], check=True)
    generated = {"NOTICE": NOTICE, "config.json": json.dumps(MARKER, indent=2) + "\n"}
    for name, text in generated.items():
        (ship / name).write_text(text)
    generated["ios/classifier.json"] = ios_classifier(ship)
    (ship / "ios" / "classifier.json").write_text(generated["ios/classifier.json"])

    # the staged tree is exactly the listed files: anything else here would be uploaded by mistake
    want = {f"{k}/{f.relative_to(v).as_posix()}" if v.is_dir() else k
            for k, v in src.items() for f in files_under(v)} | set(generated) | {"README.md"}
    have = {f.relative_to(ship).as_posix() for f in files_under(ship)}
    if extra := sorted(have - want):
        raise SystemExit(f"unexpected files in {ship} (remove them or add them to sources()):\n  " + "\n  ".join(extra))
    assert not want - have, sorted(want - have)
    said = check_staged(ship, snap)

    lines = [f"{sha256(ship / rel)}  {rel}" for rel in sorted(have)]
    (ship / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    print("\n".join(said))
    print(f"\n{ship}  ({len(lines)} files + SHA256SUMS)\n")
    print("| path | MB | files |\n|---|---|---|")
    total = 0
    for top in sorted({rel.split("/", 1)[0] for rel in have} | {"SHA256SUMS"}):
        b, n = size(ship / top)
        total += b
        print(f"| `{top}{'/' if (ship / top).is_dir() else ''}` | {b / 1e6:,.3f} | {n} |")
        if top in ("macos", "ios"):
            for child in sorted((ship / top).iterdir()):
                cb, cn = size(child)
                print(f"| &nbsp;&nbsp;`{child.name}{'/' if child.is_dir() else ''}` | {cb / 1e6:,.3f} | {cn} |")
    print(f"| total | {total / 1e6:,.1f} | {len(lines) + 1} |")
    if card.read_text() == PLACEHOLDER:
        print(f"\nNOT READY TO UPLOAD: {card} is the placeholder. Put the card there and run this script again "
              f"(SHA256SUMS covers README.md).")


def check(root: Path) -> int:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"no SHA256SUMS in {root}")
    listed = {}
    for line in sums.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
    have = {f.relative_to(root).as_posix() for f in files_under(root)}
    have = {r for r in have if r != "SHA256SUMS" and r not in IGNORED_ON_CHECK and not r.startswith(".cache/")}
    missing = sorted(set(listed) - have)
    extra = sorted(have - set(listed))
    bad = sorted(r for r in set(listed) & have if sha256(root / r) != listed[r])
    total = sum((root / r).stat().st_size for r in set(listed) & have)
    for tag, rows in (("MISSING", missing), ("SHA256 MISMATCH", bad), ("NOT IN SHA256SUMS", extra)):
        for r in rows:
            print(f"{tag}: {r}")
    ok = not (missing or bad or extra)
    print(f"{'OK' if ok else 'FAIL'}: {len(listed) - len(missing) - len(bad)}/{len(listed)} files match SHA256SUMS "
          f"({total / 1e9:.2f} GB), {len(missing)} missing, {len(bad)} mismatched, {len(extra)} unlisted")
    if (root / "README.md").is_file() and (root / "README.md").read_text() == PLACEHOLDER:
        print("note: README.md is the card placeholder")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ship", type=Path, default=GEN / "ship", help="staging directory (default <work>/_gliner25_decide/ship)")
    ap.add_argument("--check", type=Path, metavar="DIR", help="only verify DIR against its SHA256SUMS")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check(args.check.expanduser().resolve()))
    stage(args.ship.expanduser().resolve())


if __name__ == "__main__":
    main()
