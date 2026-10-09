#!/usr/bin/env python3
"""Stage the Hugging Face repository mlboydaisuke/d1-omni-600M-CoreAI in a local folder. Uploads nothing.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    $PY conversion/d1_omni/stage_hf.py            # -> <work>/hf_staging/d1-omni-600M-CoreAI/ (never replaced)
    $PY conversion/d1_omni/stage_hf.py --ship     # round 10: the stripped bundles (the same folder; move round 8's away)
    $PY conversion/d1_omni/stage_hf.py --ship --with-small   # round 13: + the decision graph at L64 / L128 (round 12)
    $PY conversion/d1_omni/stage_hf.py --ship --final        # round 14: the form to upload (implies --with-small)

--final (round 14) is the form to upload. README.md is the card (<work>/card/README.md, the supervisor's checked text,
copied byte for byte). ios-h19p/ leaves out decide-fp16-L4096: on the iPhone 18 Pro its compile loaded but its first
call was killed at the per-process memory limit (round 11, results/iphone_*.json), so a phone runs
ios/decide-fp16-L4096; NOTICE.md, metadata.json (`platforms`, the decision folders, `staged`) and the record say so.
Nothing that names a measured-only source or a path on the converting machine is published: the audio rule of
reference/records.json loses its clause about the clip that stays measured only, and the paths outside this folder (the
image pools' manifests in records.json, the images' origin files in reference/images/manifest.json, the synthesizing
interpreter in reference/audio/manifest.json) are removed; each decision graph's provenance/runtime-gate.json gives its
cache root relative to the home folder. Each edit is asserted to hit exactly what it names and is listed in
MANIFEST.json `final`. The scans grow (fatal): the measured-only speech corpus's name in any file, binary or text; the
tilde-slash home shorthand in any text file (one line excepted by its exact text: a code comment of the Swift host that
gives the runtime's cache location with placeholders, host/Sources/d1omni-cli/main.swift); in a binary file the
shorthand followed by a home folder name (a bare two-byte check means nothing there: fp16 weights hold that byte pair
hundreds of thousands of times). host/README.md says where the Swift host has run (a Mac, and the iPhone 18 Pro in the
gate app apps/D1OmniGate, round 11).

--with-small (round 13) adds the decision graph's two small buckets (64 and 128 positions, host.SMALL_BUCKETS; round
12: exported, stripped into bundles/.../macos-ship-small/, AOT in compiled/ship-h16c/ and ship-h19p/), which the
supervisor added to the ship form on 2026-10-08: the decision folders become host.ALL_BUCKETS, a host routes a row to
the smallest of them that holds it, and the gates of the small buckets are round 12's (the Python runtime's GPU gate
on the stripped AOT, results/small_runtime_compare.json = the unstripped numbers, the Swift host's parity on the 342
text rows, results/swift_parity_L64_128.json, and the audio end-to-end gate routed by host.ALL_BUCKETS,
results/small_runtime_audio_e2e_fp16.json). Without it --ship stages round 10's five buckets, as before.

--ship (round 10) stages the bundles with their MLIR debug locations stripped (strip_ship.py: bundles/.../macos-ship/,
checked against provenance/strip.json) and their AOT (aot_ship.py: compiled/ship-h19p/), gated by the round 10 runs on
them (results/ship_runtime_*.json, compiled/ship-h16c/<name>/provenance/runtime-gate.json, results/ship_swift_*.json,
results/ship_runtime_compare.json: the same numbers as the unstripped bundles); each graph's provenance gains
strip.json. It adds host/Package.swift, host/Package.resolved and host/Sources/ (a copy of apps/D1Omni, the Swift host)
with host/README.md (what the folder holds, the JPEG rule), reference/LICENSE-MMLU-MIT.txt (fixtures/, sha256 pinned)
and the MMLU line of NOTICE.md, and the graphs' hashes in metadata.json. Its binary check is fatal: no staged file,
binary or text, holds "/Users/", the home directory or the user name.

The ship form (round 7's speed ladder, decided by the supervisor on 2026-10-08): the fp16 decision graph at L = 256 /
512 / 1024 / 2048 / 4096, the fp16 vision graph and the fp16 audio graph at 5 / 10 / 20 / 30 s. The layout is the
zoo's (knowledge/jit-distribution.md): one self-contained folder per graph and platform.

  macos/<name>/     the export folder's .aimodel and metadata.json, its host file (vision: position_table.f32; audio:
                    mel_filters_128x257_f32.bin), tokenizer/ (decision graphs: metadata.json names `tokenizer/`, and
                    the Swift host apps/D1Omni reads the bucket folder's), provenance/
  ios/<name>/       the same files as macos/<name>/: each phone specializes the .aimodel on its first load
  ios-h19p/<name>/  the iPhone 18 Pro AOT compile of the same bundle (aot_ios.py, <stem>.h19p.aimodelc), metadata.json
                    with assets.main naming it, the same host file and tokenizer/, provenance/aot-manifest.json
                    (--final: every graph but decide-fp16-L4096)
  host/           the repository metadata.json and the two host files again (the kit lists the Hub's tree by folder)
  tokenizer/        the publisher's tokenizer.json and tokenizer_config.json, unmodified
  metadata.json     the repository's contract: the three graphs' inputs and outputs, the nine token ids, the
                    temperatures, the buckets, the mel and the image tiling, the noul order, source / revision / license
  reference/        the public fixture records (fixtures/records.json, public true), the publisher's fp32 model's numbers
                    on their rows (ref/records_ref*.json), their 3 images and 15 clips, the SemIf licence
  LICENSE           the publisher's LFM Open License v1.0, unmodified
  NOTICE.md         what is converted and changed, the base's attribution, the fixtures' sources
  README.md         a placeholder (the card is written later); --final: the card
  MANIFEST.json     every other file (path, bytes, sha256), the checks below, the upload command (not run)

Every graph and host file is checked against its export / AOT manifest and copied with `cp -c` (an APFS clone: no new
blocks). The lane's JSON files hold absolute paths and the rows of records that are not published: the staged
provenance keeps the export and gate summaries with every local path replaced by a token (<work>, <zoo>, <venv>,
<hf-cache>, ~) and every item that names an unpublished record removed; the per-row lists and the bundles'
reference.json (rows of every fixture record) are not staged. The published records, the oracle rows and the media
manifests only get their paths replaced (nothing in them is removed). Fatal checks (recorded in MANIFEST.json): no text file
holds a local path (`grep -rIl /Users/` = 0), the id of an unpublished record, COCO, a .flac, tv4x or tv4s; every
oracle row's ids equal the v3 host row's (results/encode_rows.v3.json); every media file matches its record's sha256;
ios/ equals macos/ byte for byte. Recorded, not fatal: the binary files that hold a local path (the graphs' MLIR debug
locations name the file of the module that was exported).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import hf_cache, hf_snapshot, repo_root, work_path  # noqa: E402
import host  # noqa: E402

MODEL_ID = "LiquidAI/d1-omni-600M"
MODEL_SHA = "414f8d6438174f5b2133a9c21a478fc42625e308"
HF_REPO = "mlboydaisuke/d1-omni-600M-CoreAI"
WORK = work_path("_d1_omni")
MACOS = WORK / "bundles" / "d1-omni-600m" / "macos"
MACOS_SMALL = WORK / "bundles" / "d1-omni-600m" / "macos-ship-small"  # round 12's stripped L64 / L128
IOS_AOT = WORK / "compiled" / "ios-h19p"
OUT = WORK / "hf_staging" / "d1-omni-600M-CoreAI"
BUCKETS = host.BUCKETS  # (256, 512, 1024, 2048, 4096); --with-small: host.ALL_BUCKETS
assert BUCKETS == (256, 512, 1024, 2048, 4096) and host.SMALL_BUCKETS == (64, 128)
AUDIO_S = (5, 10, 20, 30)


def ship_names(buckets) -> dict[str, str]:
    """Staged folder name -> the lane's bundle folder name."""
    return {**{f"decide-fp16-L{L}": f"fp16-L{L}" for L in buckets}, "vision-fp16": "vision-fp16",
            **{f"audio-fp16-{s}s": f"audio-fp16-{s}s" for s in AUDIO_S}}


SHIP = ship_names(BUCKETS)
FIXTURES = ("fixtures/records.json", "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b")  # v3
ORACLES = (  # latest first: a (record, mode) is taken from the first file that has it
    ("ref/records_ref_audio.json", "c483f8c511cfcdb05a8211e495a7476b11f9318899a1125c387c9e722a6d8fbc"),
    ("ref/records_ref_images.json", "681d2d10eef35ecb8980a3caf93d922a77a17e88ee58795468e78b915ac91b08"),
    ("ref/records_ref.json", "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"),
)
ENCODE_V3 = "results/encode_rows.v3.json"
# the ship pair's end-to-end gates (SPEED_LADDER rows 23 and 31) and the Python runtime gates of the decision AOTs
E2E = {"vision-fp16": "results/vision_e2e_fp16.run4.json", "audio": "results/audio_e2e_fp16.run4.json"}
RUNTIME_GATE = "compiled/fp16-L{L}-h16c/provenance/runtime-gate.json"
SWIFT_PARITY = "results/swift_parity_gate.json"
SWIFT_LOADS = {"aot": "results/swift_parity_aot.json", "jit": "results/swift_parity_jit.json",
               "jit_second_process": "results/swift_parity_jit.warm.json"}
L1024_PROBE = "results/swift_l1024_probe.json"
MARKERS = re.compile(r"(?i)cocodataset|\bcoco(?:_|\b)|\.flac\b|tv4x|tv4s")
# never staged from the lane's JSON: other lanes' processes and the machine around a run, compile-cache listings,
# and every per-row list (a list under "rows")
PRUNE = {"other_gpu_jobs", "cache_entries_created", "measurement_lock", "busiest", "snapshots", "example_row"}
DROP = object()
SHIP_MODE = False  # --ship (round 10): configure_ship() repoints the inputs above at the stripped bundles' runs
SWIFT_MEDIA = None  # --ship: the Swift media verdicts (results/ship_swift_media_{image,audio}.json)
STRIP_COMPARE = None  # --ship: results/ship_runtime_compare.json
SWIFT_SHIP = None  # --ship: results/ship_swift_parity.json
MMLU_LICENSE = ("fixtures/LICENSE-MMLU-MIT.txt", "5d841b9eee8bf0d721359b79fbae956a8b426d3139bfefee32149c7d4f2a1a9c")
MMLU = {"hf_id": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe", "split": "test",
        "license": "MIT, Copyright (c) 2020 Dan Hendrycks (github.com/hendrycks/test, LICENSE)",
        "license_file": "LICENSE-MMLU-MIT.txt"}
SWIFT_HOST = repo_root() / "apps" / "D1Omni"
SWIFT_HOST_FILES = ("Package.swift", "Package.resolved")
SWIFT_BINARY = WORK / "swift" / ".build" / "out" / "Products" / "Release" / "d1omni"
# --with-small (round 13): round 12's runs on the small decision buckets
SMALL_MODE = False
SMALL = {"compare": "results/small_runtime_compare.json",  # stripped vs unstripped, L64 / L128
         "swift_ship": "results/swift_parity_L64_128.json",  # the Swift host's verdict on the 342 text rows
         "swift_parity": "results/small_swift_parity_gate.json",
         "swift_loads": {"aot": "results/small_swift_parity_aot.json", "jit": "results/small_swift_parity_jit.json"},
         "audio_e2e": "results/small_runtime_audio_e2e_fp16.json"}  # the audio rows routed by host.ALL_BUCKETS


# --final (round 14): the form to upload
FINAL_MODE = False
CARD = WORK / "card" / "README.md"  # the card, checked by the supervisor: README.md byte for byte
NOT_H19P = {"decide-fp16-L4096": "on the iPhone 18 Pro (iOS 27.2, round 11) the compile loaded but its first call was "
                                 "killed at the per-process memory limit (footprint 3,306 MB, JetsamEvent reason "
                                 "per-process-limit; the .aimodel in ios/ passed there): a phone runs "
                                 "ios/decide-fp16-L4096"}
NOT_H19P_EVIDENCE = ("device/runs/r11-210603/memory.tsv", "device/runs/r11-210603/crash/JetsamEvent-2026-10-08-210652.ips")
# the scans' needles, spelled here and nowhere in the staged files (MANIFEST.json names them in words)
MEASURED_ONLY_CORPUS = re.compile(rb"(?i)librispeech")
TILDE = "~" + "/"
TILDE_HOME_FOLDER = re.compile(rb"~/(?:code|Library|\.cache|\.config|\.local|Documents|Desktop|Downloads|Applications)/")
# a text line allowed to hold the tilde shorthand: the runtime's cache location with placeholders, in a code comment of
# the Swift host (host/ is apps/D1Omni file for file), not a path of the converting machine
TILDE_ALLOWED = {("host/Sources/d1omni-cli/main.swift",
                  "/// The runtime's specialization cache of this process (" + TILDE
                  + "Library/Caches/coreai-cache/<os build>/<process name>).")}


def configure_final() -> None:
    """--final (round 14): the decision folders of --with-small, and the edits and scans of the module docstring."""
    global FINAL_MODE
    if not SHIP_MODE:
        raise SystemExit("--final needs --ship")
    if not SMALL_MODE:
        configure_small()
    FINAL_MODE = True


def h19p_names() -> list[str]:
    """The graphs staged in ios-h19p/ (--final: all but NOT_H19P)."""
    return [n for n in SHIP if not (FINAL_MODE and n in NOT_H19P)]


def edit_once(doc: dict, path: tuple, old, new, edits: list, file: str, why: str, at: str = "") -> None:
    """Replace (new is not DROP) or remove (new is DROP) doc[path...] after asserting it holds exactly `old`; `at` = where
    doc sits in its file (the recorded key is at + the path)."""
    cur = doc
    for k in path[:-1]:
        cur = cur[k]
    if path[-1] not in cur or cur[path[-1]] != old:
        raise SystemExit(f"{file}: {'.'.join(map(str, path))} is not what --final edits ({str(cur.get(path[-1]))[:80]!r})")
    if new is DROP:
        del cur[path[-1]]
    else:
        cur[path[-1]] = new
    edits.append({"file": file, "key": at + ".".join(map(str, path)), "change": "removed" if new is DROP else "rewritten",
                  "why": why})


def is_small(name: str) -> bool:
    return name.startswith("decide-") and int(name.rsplit("L", 1)[1]) in host.SMALL_BUCKETS


def source_folder(name: str) -> Path:
    """The lane's bundle folder of a staged graph (--with-small: L64 / L128 live in macos-ship-small/, round 12)."""
    return (MACOS_SMALL if SMALL_MODE and is_small(name) else MACOS) / SHIP[name]


def configure_ship() -> None:
    """--ship: the stripped bundles, their AOT and the round 10 gates on them."""
    global SHIP_MODE, MACOS, IOS_AOT, E2E, RUNTIME_GATE, SWIFT_PARITY, SWIFT_LOADS, L1024_PROBE, SWIFT_MEDIA
    global STRIP_COMPARE, SWIFT_SHIP
    SHIP_MODE = True
    MACOS = WORK / "bundles" / "d1-omni-600m" / "macos-ship"
    IOS_AOT = WORK / "compiled" / "ship-h19p"
    E2E = {"vision-fp16": "results/ship_runtime_vision_e2e_fp16.json", "audio": "results/ship_runtime_audio_e2e_fp16.json"}
    RUNTIME_GATE = "compiled/ship-h16c/fp16-L{L}/provenance/runtime-gate.json"
    SWIFT_PARITY = "results/ship_swift_parity_gate.json"
    SWIFT_LOADS = {"aot": "results/ship_swift_parity_aot.json", "jit": "results/ship_swift_parity_jit.json"}
    L1024_PROBE = "results/ship_swift_l1024_probe.json"
    SWIFT_MEDIA = {"vision-fp16": "results/ship_swift_media_image.json", "audio": "results/ship_swift_media_audio.json"}
    STRIP_COMPARE = "results/ship_runtime_compare.json"
    SWIFT_SHIP = "results/ship_swift_parity.json"


def configure_small() -> None:
    """--with-small (round 13): the decision folders become host.ALL_BUCKETS (L64 / L128 from macos-ship-small/)."""
    global SMALL_MODE, BUCKETS, SHIP
    if not SHIP_MODE:
        raise SystemExit("--with-small needs --ship (the small buckets exist only stripped)")
    SMALL_MODE = True
    BUCKETS = host.ALL_BUCKETS
    SHIP = ship_names(BUCKETS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def disk_free() -> str:
    return subprocess.run(["df", "-h", "/System/Volumes/Data"], capture_output=True, text=True).stdout.splitlines()[-1]


def clone(src: Path, dst: Path) -> None:
    """cp -c (clonefile); a directory recursively. Symlinks are followed (the HF cache's snapshot files are links)."""
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-c", "-R", "-L", str(src), str(dst)], check=True)


def write_json(path: Path, value) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# =========================================================================== what may not be published
class Scrub:
    """Local paths -> tokens; an item that names an unpublished record (or COCO / a .flac / tv4x / tv4s) -> removed."""

    def __init__(self, unpublished: set[str], groups: set[str] = frozenset()):
        """`groups`: the "<source>/<mode>" keys of the gates' per-source aggregates whose rows are all unpublished (a
        source without a public record, and the card's measured-only image and clip)."""
        self.groups = set(groups)
        home = str(Path.home())
        venv = str(Path(sys.executable).resolve().parent.parent)
        pairs = [(str(WORK), "<work>"), (str(repo_root()), "<zoo>"), (venv, "<venv>"),
                 (str(Path(sys.executable).parent.parent), "<venv>"), (str(hf_cache()), "<hf-cache>"), (home, "~")]
        self.pairs = sorted({p for p in pairs if p[0]}, key=lambda p: -len(p[0]))
        ids = sorted(unpublished, key=len, reverse=True)
        self.ids = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, ids)) + r")(?![A-Za-z0-9])")
        self.removed = 0

    def text(self, s: str) -> str:
        for a, b in self.pairs:
            s = s.replace(a, b)
        return s

    def bad(self, s: str) -> bool:
        return bool(self.ids.search(s) or MARKERS.search(s))

    def walk(self, v):
        if isinstance(v, dict):
            for k in ("id", "row", "record", "file"):
                if isinstance(v.get(k), str) and self.bad(v[k]):
                    self.removed += 1
                    return DROP
            out = {}
            for k, x in v.items():
                if k in PRUNE or (k == "rows" and isinstance(x, list)):
                    continue
                if k in self.groups or k.split("/")[0] in self.groups:
                    self.removed += 1
                    continue
                if self.bad(k):
                    self.removed += 1
                    continue
                y = self.walk(x)
                if y is not DROP:
                    out[self.text(k)] = y
            return out
        if isinstance(v, list):
            return [y for y in (self.walk(x) for x in v) if y is not DROP]
        if isinstance(v, str):
            s = self.text(v)
            if self.bad(s):
                self.removed += 1
                return DROP
            return s
        return v

    def paths(self, v):
        """Local paths -> tokens only (nothing removed): for the published records, whose text must stay as sent."""
        if isinstance(v, dict):
            return {self.text(k): self.paths(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.paths(x) for x in v]
        return self.text(v) if isinstance(v, str) else v

    def json(self, value, drop_keys=()) -> dict:
        value = copy.deepcopy(value)
        if isinstance(value, dict):
            for k in drop_keys:
                value.pop(k, None)
        out = self.walk(value)
        return {} if out is DROP else out


# =========================================================================== inputs, checked
def check_export(folder: Path) -> dict:
    """The bundle's files against its export manifest. The manifest's status is the export gate's verdict, which for
    the fp16 graphs is measured, not decided (its bar_rule: "converted whatever it says; the runtime gate decides";
    fp16 L1024 / L2048 read FAIL there on the CPU's fp16, mean 2.1e-3): the ship gate is checked in ship_gate()."""
    manifest = json.loads((folder / "provenance" / "export-manifest.json").read_text())
    if manifest.get("measure_only"):
        raise SystemExit(f"{folder}: a measure-only export")
    bundle = folder / manifest["bundle"]
    files = manifest["files"]
    if SHIP_MODE:  # the stripped bundle: its files are strip.json's "after" (the export manifest lists the source's)
        strip = json.loads((folder / "provenance" / "strip.json").read_text())
        if strip["status"] != "PASS" or strip["bundle"] != manifest["bundle"]:
            raise SystemExit(f"{folder}: strip {strip['status']} ({strip['bundle']})")
        files = strip["after"]["files"]
    for item in files:
        f = bundle / item["path"]
        if f.stat().st_size != item["bytes"] or sha256_file(f) != item["sha256"]:
            raise SystemExit(f"{f} changed since its export" + (" and strip" if SHIP_MODE else ""))
    return manifest | {"bytes": sum(f["bytes"] for f in files)}


def check_aot(folder: Path) -> dict:
    aot = json.loads((folder / "provenance" / "aot-manifest.json").read_text())
    if aot.get("status") != "COMPILED" or aot.get("architecture") != "h19p" or aot.get("platform", "iOS") != "iOS":
        raise SystemExit(f"{folder}: {aot.get('status')} {aot.get('architecture')}")
    aimodelc = folder / aot["aimodelc"]
    for item in aot["files"]:
        f = aimodelc / item["path"]
        if f.stat().st_size != item["bytes"] or sha256_file(f) != item["sha256"]:
            raise SystemExit(f"{f} changed since its compile")
    return aot


def ship_gate(name: str) -> dict:
    """The gate that made the form shippable: the Python runtime's GPU gate of the bucket's Mac AOT (decision), the
    end-to-end GPU gate of the media graph with the fp16 decision graphs (vision / audio). PASS is required."""
    if name.startswith("decide-"):
        rel = RUNTIME_GATE.format(L=int(name.rsplit("L", 1)[1]))
        status = json.loads((WORK / rel).read_text())["gpu"]["status"]
    else:
        rel = E2E["vision-fp16"] if name == "vision-fp16" else E2E["audio"]
        status = json.loads((WORK / rel).read_text())["status"]
    if status != "PASS":
        raise SystemExit(f"{name}: the ship gate {rel} says {status}")
    out = {"file": rel, "status": status}
    if SHIP_MODE:  # the stripped bundles: the same numbers as the unstripped ones, and the Swift host's parity on them
        small = is_small(name)
        compare_rel = SMALL["compare"] if small else STRIP_COMPARE
        swift_rel = SMALL["swift_ship"] if small else SWIFT_SHIP
        compare = json.loads((WORK / compare_rel).read_text())
        keys = [name] if name.startswith("decide-") else (["vision-fp16", "vision-e2e"] if name == "vision-fp16" else
                                                          [name, "audio-e2e"])
        same = {k: compare["pairs"][k]["status"] for k in keys}
        swift = json.loads((WORK / swift_rel).read_text())["status"]
        if any(s not in ("SAME", "WITHIN") for s in same.values()) or swift != "PASS":
            raise SystemExit(f"{name}: stripped vs unstripped {same}, Swift parity {swift}")
        out |= {"stripped_vs_unstripped": {"file": compare_rel, "status": same},
                "swift_parity": {"file": swift_rel, "status": swift}}
        if SMALL_MODE and name.startswith("audio-"):  # the audio rows routed by host.ALL_BUCKETS (round 12)
            status_all = json.loads((WORK / SMALL["audio_e2e"]).read_text())["status"]
            if status_all != "PASS":
                raise SystemExit(f"{name}: {SMALL['audio_e2e']} says {status_all}")
            out["e2e_all_buckets"] = {"file": SMALL["audio_e2e"], "status": status_all}
    return out


def host_files(meta: dict) -> list[str]:
    files = []
    if "vision" in meta:
        files.append(meta["vision"]["position_table"]["file"])
    if "audio" in meta:
        files.append(meta["audio"]["filterbank"]["file"])
    return files


# =========================================================================== the folders
def final_runtime_gate(doc: dict, name: str, edits: list) -> dict:
    """--final: the gate's cache root as a path relative to the home folder (the runtime's cache location; Scrub had
    written the home folder as the tilde shorthand)."""
    old = doc.get("gpu", {}).get("cache_root")
    if not isinstance(old, str) or not old.startswith(TILDE + "Library/Caches/coreai-cache/"):
        raise SystemExit(f"{name}: runtime-gate.json gpu.cache_root is not the expected cache path")
    edit_once(doc, ("gpu", "cache_root"), old, old[len(TILDE):], edits,
              f"macos/{name}/provenance/runtime-gate.json (and its ios/ copy)",
              "the cache root relative to the home folder (gpu.cache_root_relative_to)")
    gpu = {}
    for k, v in doc["gpu"].items():  # the note right after the value it qualifies
        gpu[k] = v
        if k == "cache_root":
            gpu["cache_root_relative_to"] = "the home folder of the account that ran the gate"
    doc["gpu"] = gpu
    return doc


def stage_graph(name: str, out: Path, scrub: Scrub, record: dict, edits: list) -> None:
    src = source_folder(name)
    manifest = check_export(src)
    gate = ship_gate(name)
    meta = json.loads((src / "metadata.json").read_text())
    if scrub.walk(meta) != meta:
        raise SystemExit(f"{src}/metadata.json holds a local path or an unpublished id")
    files = host_files(meta)
    for f in files:
        want = (meta.get("vision", {}).get("position_table") or meta.get("audio", {}).get("filterbank"))["sha256"]
        if sha256_file(src / f) != want:
            raise SystemExit(f"{src / f} differs from metadata.json's sha256")
    m = out / "macos" / name
    clone(src / manifest["bundle"], m / manifest["bundle"])
    clone(src / "metadata.json", m / "metadata.json")
    for f in files:
        clone(src / f, m / f)
    if (src / "tokenizer").exists():
        clone(src / "tokenizer", m / "tokenizer")
    prov = {"export-manifest.json": scrub.json(manifest),
            "export-gate.json": scrub.json(json.loads((src / "provenance" / "export-gate.json").read_text()),
                                           drop_keys=("rows", "example_row"))}
    if name.startswith("decide-"):
        L = int(name.rsplit("L", 1)[1])
        prov["runtime-gate.json"] = scrub.json(json.loads((WORK / RUNTIME_GATE.format(L=L)).read_text()))
        if FINAL_MODE:
            prov["runtime-gate.json"] = final_runtime_gate(prov["runtime-gate.json"], name, edits)
        prov["swift-parity.json"] = swift_parity(L, scrub)
    else:
        e2e = WORK / (E2E["vision-fp16"] if name == "vision-fp16" else E2E["audio"])
        prov["e2e-gate.json"] = scrub.json(json.loads(e2e.read_text()), drop_keys=("rows",))
        if SHIP_MODE:
            prov["swift-parity.json"] = swift_media_parity(name, scrub)
        if "e2e_all_buckets" in gate:  # --with-small: the same clips, the rows routed by host.ALL_BUCKETS (round 12)
            prov["e2e-gate-all-buckets.json"] = scrub.json(json.loads((WORK / SMALL["audio_e2e"]).read_text()),
                                                           drop_keys=("rows",))
    if SHIP_MODE:
        prov["strip.json"] = strip_record(src, scrub)
        prov["strip-equivalence.json"] = scrub.json({k: v for k, v in gate["stripped_vs_unstripped"].items()} | {
            "pairs": {k: json.loads((WORK / gate["stripped_vs_unstripped"]["file"]).read_text())["pairs"][k]
                      for k in gate["stripped_vs_unstripped"]["status"]}})
    for fname, doc in prov.items():
        write_json(m / "provenance" / fname, doc)
    # ios/: the same files (the JIT IR; the phone specializes it)
    clone(m, out / "ios" / name)
    # ios-h19p/: the AOT of the same bundle, the same host files and tokenizer (--final: not for NOT_H19P)
    a_src = IOS_AOT / (SHIP[name] if SHIP_MODE else name)
    aot = check_aot(a_src)
    if Path(aot["source"]["folder"]).resolve() != src.resolve():
        raise SystemExit(f"{a_src}: compiled from {aot['source']['folder']}, not {src}")
    if name in h19p_names():
        a = out / "ios-h19p" / name
        clone(a_src / aot["aimodelc"], a / aot["aimodelc"])
        meta_ios = copy.deepcopy(meta)
        meta_ios["assets"]["main"] = aot["aimodelc"]
        meta_ios["compiled"] = {"platform": "iOS", "min_deployment_version": aot.get("min_deployment_version", "27.0"),
                                "architecture": "h19p", "preferred_compute": aot.get("preferred_compute", "gpu"),
                                "from": f"../../macos/{name}/{manifest['bundle']}",
                                "note": "an iPhone 18 Pro (h19p) GPU compile; another phone generation refuses it at "
                                        "load (incompatibleCompiledAssetArchitecture) and takes ios/ instead"}
        write_json(a / "metadata.json", meta_ios)
        for f in files:
            clone(src / f, a / f)
        if (src / "tokenizer").exists():
            clone(src / "tokenizer", a / "tokenizer")
        write_json(a / "provenance" / "aot-manifest.json", scrub.json(aot))
        h19p = {"aimodelc_h19p": aot["aimodelc"], "aimodelc_h19p_bytes": aot["bytes"],
                "aimodelc_h19p_main_hash": aot["hashes"].get("main.hash")}
    else:  # --final: compiled (round 10) and run on the phone (round 11), not shipped
        h19p = {"ios_h19p_not_shipped": {"why": NOT_H19P[name], "evidence": list(NOT_H19P_EVIDENCE),
                                         "compile": {"aimodelc": aot["aimodelc"], "bytes": aot["bytes"],
                                                     "main_hash": aot["hashes"].get("main.hash")}}}
    record[name] = {"source": str(src.relative_to(WORK)), "bundle": manifest["bundle"], "bundle_bytes": manifest["bytes"],
                    "main_hash": (src / manifest["bundle"] / "main.hash").read_bytes().hex(),
                    "export_gate": manifest.get("status"), "ship_gate": gate, **h19p, "host_files": files,
                    "provenance": sorted(prov)}
    if SHIP_MODE:
        strip = json.loads((src / "provenance" / "strip.json").read_text())
        record[name]["unstripped"] = {"bundle_bytes": strip["before"]["bytes"], "main_hash": strip["before"]["main_hash"],
                                      "source": str((WORK / "bundles" / "d1-omni-600m" / "macos" / SHIP[name]).relative_to(WORK))}


def strip_record(folder: Path, scrub: Scrub) -> dict:
    """provenance/strip.json as staged: the bytes, sha256 and main.hash before / after, the op counts, the byte scans
    (counts per file), the inspect verdict; not the inspect outputs, nor the scan's needles (the home directory and
    the user name of the machine that ran it)."""
    strip = json.loads((folder / "provenance" / "strip.json").read_text())
    strip["inspect"] = {"equal": strip["inspect"]["equal"]}
    strip.pop("local_path_needles", None)
    return scrub.json(strip)


def swift_media_parity(name: str, scrub: Scrub) -> dict:
    """--ship: the Swift host's media path on the stripped bundles (gate_swift.py --ship media, the 46 rows of this
    graph's mode against the Python runtime on the same AOT and the oracle; AOT and JIT)."""
    doc = json.loads((WORK / SWIFT_MEDIA["vision-fp16" if name == "vision-fp16" else "audio"]).read_text())
    e2e = doc["e2e"]
    out = {"what": "the Swift host (apps/D1Omni `d1omni parity-media`) on the Mac GPU, the stripped bundles, against the "
                   "Python host's dump on the same AOT (host_dump_media.py --ship) and the publisher's fp32 model",
           "status": doc["status"], "swift": e2e["swift"], "control_next_item_prefix": e2e["control_next_item_prefix"],
           "wrong_pairing_oracle_swap": e2e["wrong_pairing_oracle_swap"], "jit_vs_aot": doc["jit_vs_aot"],
           "rows_bit_equal_python": e2e.get("rows_bit_equal_python", e2e.get("rows_bit_equal_python_numpy_mel"))}
    for key in ("decode", "graph_inputs", "prefix", "samples", "mel", "masks"):
        if key in doc:
            out[key] = doc[key]
    return scrub.json(out)


def swift_parity(L: int, scrub: Scrub) -> dict:
    """The Swift host's parity on this bucket (apps/D1Omni, round 8): the gate's summary for every bucket, this
    bucket's rows and loads; for L1024 (no fixture row lands there) the one-request probe."""
    small = L in host.SMALL_BUCKETS
    gate = json.loads((WORK / (SMALL["swift_parity"] if small else SWIFT_PARITY)).read_text())
    if small:  # round 13 (--with-small)
        what = ("round 12, the stripped L64 / L128 bundles, the 342 text rows whose bucket under host.ALL_BUCKETS is 64 or "
                "128); the summary covers both small buckets")
    else:  # the round 8 / 10 text, unchanged (the staged files of those buckets stay byte-identical)
        what = ("round 10, the stripped bundles" if SHIP_MODE else "round 8") + "); the summary covers all buckets"
    out = {"what": "the Swift host (apps/D1Omni `d1omni parity`) on the Mac GPU against the publisher's fp32 model and the "
                   "Python Core AI runtime on the same AOT (" + what,
           "summary": {k: {f: gate[k][f] for f in ("status", "rows", "text_rows", "argmax_equal", "max_abs_dp",
                                                    "mean_row_max_abs_dp", "python_runtime", "drift_marker_logits_max")}
                       for k in ("aot", "jit")},
           "jit_vs_aot": gate.get("jit_vs_aot"), "this_bucket": {}}
    for kind, rel in (SMALL["swift_loads"] if small else SWIFT_LOADS).items():
        doc = json.loads((WORK / rel).read_text())
        rows = [r for r in doc["rows"] if r["bucket"] == L]
        load = doc["loads"].get(str(L))
        out["this_bucket"][kind] = {
            "rows": len(rows), "python_bit_equal": sum(r["python_bit_equal"] for r in rows),
            "load": {k: load[k] for k in ("kind", "model_seconds", "function_seconds", "wall_seconds", "main_hash")} if load else None}
    if L == 1024:
        probe = json.loads((WORK / L1024_PROBE).read_text())
        out["this_bucket"]["probe"] = {"what": "no fixture row has 513..1024 positions: one request (own_L01's state cut to "
                                               "its first 34 lines, 3 rows of 961..1009 positions) through `d1omni ask` "
                                               "(AOT, JIT cold, JIT warm) against the Python runtime's response",
                                       "status": probe["status"],
                                       "response_equal_python_bytes": {k: v["response_equal_python_bytes"]
                                                                       for k, v in probe["swift"].items()}}
    return scrub.json(out)


# =========================================================================== the repository files
def repo_metadata(record: dict) -> dict:
    dec = json.loads((MACOS / "fp16-L256" / "metadata.json").read_text())
    vis = json.loads((MACOS / "vision-fp16" / "metadata.json").read_text())
    aud = {s: json.loads((MACOS / f"audio-fp16-{s}s" / "metadata.json").read_text()) for s in AUDIO_S}
    d = dec["decision"]

    def folders(names):
        return {p: {n: f"{p}/{n}" for n in names if p != "ios-h19p" or n in h19p_names()}
                for p in ("macos", "ios", "ios-h19p")}

    decision_inputs = json.loads(json.dumps(d["inputs"]).replace("256", '"L"'))
    return {
        "schema": "d1-omni-coreai/1",
        "model": {"source": MODEL_ID, "revision": MODEL_SHA, "license": "LFM Open License v1.0",
                  "conversion": "coreai-model-zoo conversion/d1_omni (re-authored in plain PyTorch, coreai-torch 0.4.1"
                                + ("; the graphs' debug locations stripped with coreai-torch strip_debug_info, "
                                   "conversion/d1_omni/strip_ship.py)" if SHIP_MODE else ")")},
        "runtime": "Core AI (macOS 27 / iOS 27)",
        "precision": "fp16 (weights and compute; the exceptions are listed in each graph's metadata.json)",
        "graphs": {
            "decision": {"buckets": list(BUCKETS),
                         "bucket_rule": ("the smallest L that holds P + len(ids) (host.py bucket_for(positions, "
                                         "ALL_BUCKETS))" if SMALL_MODE else
                                         "the smallest L that holds P + len(ids) (host.py bucket_for)"),
                         "inputs": decision_inputs, "outputs": json.loads(json.dumps(d["outputs"]).replace("256", '"L"')),
                         "max_length": d["max_length"], "image_text_length": d["image_text_length"],
                         "audio_text_length": d["audio_text_length"], "min_text_positions": d["min_text_positions"],
                         "prefix_hidden": d["prefix_hidden"],
                         "folders": folders([f"decide-fp16-L{L}" for L in BUCKETS])},
            "vision": {k: vis["vision"][k] for k in ("role", "inputs", "outputs", "max_patches", "max_tokens", "patch",
                                                       "crops", "prefix", "position_table", "precision_detail")}
                      | {"folders": folders(["vision-fp16"])},
            "audio": {"buckets_s": list(AUDIO_S),
                      "per_bucket": {f"{s}s": {"bucket": aud[s]["audio"]["bucket"],
                                               "inputs": aud[s]["audio"]["inputs"], "outputs": aud[s]["audio"]["outputs"]}
                                     for s in AUDIO_S},
                      **{k: aud[10]["audio"][k] for k in ("role", "waveform", "mel", "filterbank", "prefix", "precision_detail")},
                      "folders": folders([f"audio-fp16-{s}s" for s in AUDIO_S])},
        },
        "token_ids": d["token_ids"], "delimiters": d["delimiters"], "temperatures": d["temperatures"],
        "temperature_rule": d["temperature_rule"], "noul_order": d["noul_order"], "noul_reported": d["noul_reported"],
        "tokenizer": {"folder": "tokenizer/", "encode": "tok(escape(s), add_special_tokens=False): no <|startoftext|> "
                      "from the post-processor; escape() rewrites every <|name|> in caller text as <¦name¦>"},
        "host": {"python": "coreai-model-zoo conversion/d1_omni/host.py (request_rows, graph_inputs, "
                           "probabilities_from_logits, image_crops_inputs, audio_inputs)",
                 "swift": ("coreai-model-zoo apps/D1Omni (library D1Omni, CLI d1omni; a copy in host/Package.swift and "
                           "host/Sources): text, image and audio requests; the JPEG rule is in host/README.md"
                           if SHIP_MODE else "coreai-model-zoo apps/D1Omni (library D1Omni, CLI d1omni): text requests; "
                                             "the image and audio preprocessing in Swift is not written yet")},
        "platforms": {"macos": "the .aimodel; the Mac specializes it at load (the Swift host's JIT and AOT gave the same "
                               "marker logits bit for bit and the same ms)",
                      "ios": "the same .aimodel; every phone specializes it on its first load",
                      "ios-h19p": ("iPhone 18 Pro (h19p) GPU AOT of the same bundles except decide-fp16-L4096 (its "
                                   "compile exceeds the phone's per-process memory limit at the first call: a phone "
                                   "runs ios/decide-fp16-L4096); another phone generation refuses them at load and "
                                   "takes ios/" if FINAL_MODE else
                                   "iPhone 18 Pro (h19p) GPU AOT of the same bundles; another phone generation refuses "
                                   "them at load and takes ios/" if SMALL_MODE else
                                   "iPhone 18 Pro (h19p) GPU AOT of the same bundles; not yet loaded on a phone")},
        "staged": {name: {k: v for k, v in r.items() if k in ("bundle", "bundle_bytes", "main_hash", "aimodelc_h19p",
                                                               "aimodelc_h19p_bytes", "aimodelc_h19p_main_hash")}
                   for name, r in record.items()},
        "hash_note": "main_hash = the .aimodel's main.hash, sha256 of its main.mlirb (the name of the compile-cache entry a "
                     "JIT load creates); aimodelc_h19p_main_hash = the AOT's main.hash",
    }


NOTICE = """# NOTICE

This repository holds Core AI conversions of [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
(revision `{sha}`) by Liquid AI, Inc., distributed under the LFM Open License v1.0. `LICENSE` is the publisher's
text, unmodified. Use is subject to that licence, including its Commercial Use Limitation (Section 5).

## What was changed (LFM Open License v1.0, Section 4(b))

Every graph here is a converted and modified form of the publisher's `model.safetensors`:

- The network was re-authored in plain PyTorch from the checkpoint's weights (coreai-model-zoo
  `conversion/d1_omni/d1_omni_model.py`, `d1_omni_vision.py`, `d1_omni_audio.py`), exported with Apple's
  coreai-torch 0.4.1 and stored in fp16 (weights and compute; the operations kept in fp32 are listed in each
  `metadata.json`).
- The decision graph is cut into five static lengths (256, 512, 1,024, 2,048 and 4,096 positions; the 4,096 graph
  computes attention in blocks of 2,048 keys) and returns a score per position, which the host reads at the
  marker positions before the publisher's temperatures and softmax. The vision graph takes one crop per call; the
  audio graph takes one clip in a 5, 10, 20 or 30 s bucket.
- `ios-h19p/` holds ahead-of-time compiles of the same graphs for the iPhone 18 Pro GPU
  (`coreai-build compile --platform iOS --architecture h19p`).
- Every `metadata.json`, `position_table.f32` (the vision tower's position embeddings from the checkpoint, float32)
  and `mel_filters_128x257_f32.bin` (the publisher's Slaney filterbank, float32) were written for this repository.

Unmodified files from the publisher's repository: `LICENSE`, `tokenizer/tokenizer.json`,
`tokenizer/tokenizer_config.json` (and the copies of the two tokenizer files beside each decision graph).

## Reference fixtures (`reference/`)

`reference/records.json` holds the public fixture records and `reference/oracle.json` the publisher's model's
numbers on them (CPU, fp32). Their sources:

- Records written for these ports (`own_*`, `long_3400`) and the publisher's README examples (`card_text`,
  `card_batch_00`, `card_batch_01`: the "How to use" requests of LiquidAI/d1-omni-600M's README at the pinned
  revision).
- `semif_*`: 144 items of [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) `benchmarks/data/authored144.jsonl`
  (commit `ca3ba65f`), MIT licence, Copyright (c) 2026 TheoLeeCJ: `reference/LICENSE-SemIf`.
- `tv4_000` … `tv4_059`: the first 60 lines of jaredpalmer/kev's `evals/v4/transfer-v4/development.jsonl` (tag
  `kev-1.0`); the questions come from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), MIT licence per its
  dataset card.
- `reference/images/img_01.png`: own model output (FLUX.2 klein 4B, Apache-2.0); `img_02.png`, `img_03.png`: CC0 1.0
  photographs (Wikimedia Commons, no attribution required), downscaled. Origin and resize: `reference/images/manifest.json`.
- `reference/audio/aud_01.wav` … `aud_15.wav`: speech synthesized with hexgrad/Kokoro-82M (Apache-2.0) from scripts
  written for these ports (`reference/audio/script.json`).
"""
# --ship (round 10): the strip is a change to the graphs; the MMLU questions' licence text is now in reference/
NOTICE_STRIP = """  `metadata.json`).
- The graphs' debug locations, which named source files on the machine that exported them, were removed with
  coreai-torch's `strip_debug_info`. Their operations are unchanged.
"""
NOTICE_TV4 = """- `tv4_000` … `tv4_059`: the first 60 lines of jaredpalmer/kev's `evals/v4/transfer-v4/development.jsonl` (tag
  `kev-1.0`). The questions are MMLU test items from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (revision
  `c30699e8`), MIT licence, Copyright (c) 2020 Dan Hendrycks: `reference/LICENSE-MMLU-MIT.txt` (from
  [hendrycks/test](https://github.com/hendrycks/test)).
"""


def notice_text() -> str:
    text = NOTICE.format(sha=MODEL_SHA)
    if not SHIP_MODE:
        return text
    swaps = [("  `metadata.json`).\n- The decision graph", NOTICE_STRIP + "- The decision graph"),
             (NOTICE[NOTICE.index("- `tv4_000`"):NOTICE.index("- `reference/images/img_01.png`")], NOTICE_TV4)]
    if SMALL_MODE:  # round 13: the decision graph at seven lengths
        swaps.append(("five static lengths (256, 512, 1,024, 2,048 and 4,096 positions;",
                      "seven static lengths (64, 128, 256, 512, 1,024, 2,048 and 4,096 positions;"))
    if FINAL_MODE:  # round 14: no h19p compile of decide-fp16-L4096
        swaps.append(("- `ios-h19p/` holds ahead-of-time compiles of the same graphs for the iPhone 18 Pro GPU\n",
                      "- `ios-h19p/` holds ahead-of-time compiles of the same graphs, except `decide-fp16-L4096`, for the "
                      "iPhone 18 Pro GPU\n"))
    for old, new in swaps:
        if text.count(old) != 1:
            raise SystemExit(f"NOTICE: {old[:40]!r} found {text.count(old)} times")
        text = text.replace(old, new)
    return text


AUDIO_RULE_MEASURED_ONLY = "; the card's flac (" + "Libri" + "Speech, CC BY 4.0) stays measured only"


def final_sources(sources: dict, edits: list) -> None:
    """--final: reference/records.json's source list without the measured-only clip's clause and without the image pools'
    manifest paths (files outside this folder, on the converting machine)."""
    rule = sources["audio"]["rule"]
    if rule.count(AUDIO_RULE_MEASURED_ONLY) != 1 or not rule.endswith(AUDIO_RULE_MEASURED_ONLY):
        raise SystemExit("reference/records.json: sources.audio.rule is not the rule --final rewrites")
    edit_once(sources, ("audio", "rule"), rule, rule[:-len(AUDIO_RULE_MEASURED_ONLY)], edits, "reference/records.json",
              "the clause about a measured-only clip removed (not named in a published file)", at="sources.")
    for pool in ("cc0", "flux"):
        old = sources["images"]["pools"][pool].get("manifest", "")
        if not old.startswith(TILDE):
            raise SystemExit(f"reference/records.json: sources.images.pools.{pool}.manifest is not a home path")
        edit_once(sources, ("images", "pools", pool, "manifest"), old, DROP, edits, "reference/records.json",
                  "a path outside this folder (the pool's list on the converting machine; its sha256 stays)", at="sources.")


def final_media_manifests(img_doc: dict, aud_doc: dict, edits: list) -> None:
    """--final: the images' origin files and the synthesizing interpreter (paths on the converting machine) removed."""
    for i, im in enumerate(img_doc["images"]):
        old = im["made"].get("from", "")
        if not old.startswith(TILDE):
            raise SystemExit(f"reference/images/manifest.json: {im['id']} made.from is not a home path")
        edit_once(img_doc, ("images", i, "made", "from"), old, DROP, edits, "reference/images/manifest.json",
                  f"{im['id']}: a path outside this folder (the source file; its sha256 stays as made.source_sha256)")
    old = aud_doc.get("python", "")
    if not old.startswith(TILDE):
        raise SystemExit("reference/audio/manifest.json: python is not a home path")
    edit_once(aud_doc, ("python",), old, DROP, edits, "reference/audio/manifest.json",
              "a path outside this folder (the interpreter that ran the synthesis; the versions stay)")


def stage_reference(out: Path, scrub: Scrub, unpublished: set[str], edits: list) -> dict:
    fixtures = json.loads((WORK / FIXTURES[0]).read_text())
    if sha256_file(WORK / FIXTURES[0]) != FIXTURES[1]:
        raise SystemExit(f"{FIXTURES[0]} is not fixtures v3")
    public = [r for r in fixtures["records"] if r["public"] is True]
    ids = {r["id"] for r in public}
    assert not ids & unpublished
    # the records, with a source list that names only what they come from
    sources = copy.deepcopy(fixtures["sources"])
    card = sources.pop("card")
    sources["card"] = {k: card[k] for k in ("file", "sha256", "section", "repo", "revision")}
    sources.pop("kev_records", None)
    tv4 = sources.pop("transfer_v4")
    sources["transfer_v4"] = {k: tv4[k] for k in ("path_in_repo", "sha256", "lines", "author_repo")} | {
        "records_here": "tv4_000 ... tv4_059: the file's first 60 lines (all mmlu)", "license": tv4["source_licenses"]["mmlu"]}
    if SHIP_MODE:
        revisions = {r["provenance"]["_meta"]["revision"] for r in public if r["id"].startswith("tv4_")}
        if revisions != {MMLU["revision"]}:
            raise SystemExit(f"tv4 records from cais/mmlu revisions {revisions}, not {MMLU['revision']}")
        sources["transfer_v4"]["mmlu"] = MMLU
    sources["semif_authored144"]["license_file"] = "LICENSE-SemIf"
    sources["own"] = {k: v for k, v in sources["own"].items() if not k.startswith("withheld")}
    records_doc = {"schema": "d1-omni-public-fixtures/1",
                   "what": "the public records of the d1-omni-600M Core AI port's fixture (v3, round 6): requests as sent "
                           "to the model, media, the expected answer where one exists, provenance",
                   "fixtures_sha256": FIXTURES[1], "records": public, "sources": scrub.json(sources)}
    records_doc = scrub.paths(records_doc)
    if FINAL_MODE:
        final_sources(records_doc["sources"], edits)
    write_json(out / "reference" / "records.json", records_doc)
    # the oracle rows of those records, each (record, mode) from the latest oracle file that has it
    enc = {(r["id"], r["mode"], r["qid"]): r["ids_sha256"] for r in json.loads((WORK / ENCODE_V3).read_text())["rows"]}
    rows, metas, bad = {}, [], []
    for rel, sha in ORACLES:
        if sha256_file(WORK / rel) != sha:
            raise SystemExit(f"{rel} is not the pinned oracle")
        doc = json.loads((WORK / rel).read_text())
        metas.append({"file": Path(rel).name, "sha256": sha, "schema": doc["schema"], "written": doc["written"],
                      "fixtures_sha256": doc["fixtures"]["sha256"]})
        for rec in doc["records"]:
            key = (rec["id"], rec["mode"])
            if rec["id"] not in ids or key in rows:
                continue
            for q in rec["questions"]:
                if enc.get((rec["id"], rec["mode"], q["qid"])) != hashlib.sha256(json.dumps(q["ids"]).encode()).hexdigest():
                    bad.append(f"{rec['id']}/{rec['mode']}/{q['qid']}")
            rows[key] = {k: rec[k] for k in ("id", "mode", "source", "native", "prefix", "usage_input_tokens", "questions",
                                              "response")} | {"oracle_file": Path(rel).name}
    if bad:
        raise SystemExit(f"oracle rows whose ids differ from the v3 host rows: {bad[:10]}")
    want = {(r["id"], "text") for r in public} | {(r["id"], "image" if "images" in r["media"] else "audio")
                                                    for r in public if r.get("media")}
    if set(rows) != want:
        raise SystemExit(f"oracle rows missing {sorted(want - set(rows))[:10]}, extra {sorted(set(rows) - want)[:10]}")
    model = json.loads((WORK / ORACLES[-1][0]).read_text())
    oracle_doc = {"schema": "d1-omni-public-oracle/1",
                  "what": "the publisher's model (AutoModel.from_pretrained(trust_remote_code=True), CPU, fp32) on every "
                          "public record in text mode and, when it has media, in its media mode: per question the ids, the "
                          "marker positions, the raw marker logits, the temperature, the probabilities, the response",
                  "model": {k: model["model"][k] for k in ("hf_id", "revision", "source_sha256", "weights_sha256")},
                  "versions": model["versions"], "reference": model["reference"], "oracle_files": metas,
                  "rows_checked": "every row's ids equal the v3 host rows (encode_rows.v3.json, the publisher's encode)",
                  "records": [rows[k] for k in sorted(rows)]}
    write_json(out / "reference" / "oracle.json", scrub.paths(oracle_doc))
    # media
    media = []
    for r in public:
        prov = r["provenance"]
        for rel in (r.get("media") or {}).get("images", []):
            want_sha = prov["image"]["sha256"]
            media.append((rel, want_sha))
        if (r.get("media") or {}).get("audio"):
            media.append((r["media"]["audio"], (prov.get("clip") or prov.get("audio") or {}).get("sha256")))
    for rel, want_sha in media:
        if sha256_file(WORK / rel) != want_sha:
            raise SystemExit(f"{rel} differs from its record's sha256")
        clone(WORK / rel, out / "reference" / rel)
    img_manifest = json.loads((WORK / "images" / "manifest.json").read_text())
    img_doc = {"schema": img_manifest["schema"], "rule": img_manifest["rule"],
               "images": [i for i in img_manifest["images"] if i["id"] in ids]}
    img_doc = scrub.paths(img_doc)
    aud_manifest = json.loads((WORK / "audio" / "manifest.json").read_text())
    aud_doc = scrub.paths({k: aud_manifest[k] for k in ("clips", "kokoro_version", "synth", "torch", "python", "written")})
    if FINAL_MODE:
        final_media_manifests(img_doc, aud_doc, edits)
    write_json(out / "reference" / "images" / "manifest.json", img_doc)
    write_json(out / "reference" / "audio" / "manifest.json", aud_doc)
    write_json(out / "reference" / "audio" / "script.json", scrub.paths(json.loads((WORK / "audio" / "script.json").read_text())))
    lic = WORK / "fixtures" / "LICENSE-SemIf"
    if sha256_file(lic) != fixtures["sources"]["semif_authored144"]["license_sha256"]:
        raise SystemExit("LICENSE-SemIf differs from its pin")
    clone(lic, out / "reference" / "LICENSE-SemIf")
    if SHIP_MODE:
        mmlu = WORK / MMLU_LICENSE[0]
        if sha256_file(mmlu) != MMLU_LICENSE[1]:
            raise SystemExit(f"{MMLU_LICENSE[0]} differs from its pin")
        clone(mmlu, out / "reference" / MMLU["license_file"])
    out_ref = {"records": len(public), "oracle_request_modes": len(rows),
               "oracle_questions": sum(len(r["questions"]) for r in rows.values()), "media_files": len(media)}
    if SHIP_MODE:
        out_ref["mmlu_license"] = {"file": f"reference/{MMLU['license_file']}", "sha256": MMLU_LICENSE[1],
                                   "provenance": json.loads((WORK / "fixtures" / "LICENSE-MMLU-MIT.json").read_text())}
    return out_ref


# =========================================================================== host/ (--ship)
HOST_README = """# host/

What a host reads besides the graphs:

- `metadata.json`: the repository's contract, the same file as `../metadata.json`.
- `position_table.f32`: the vision tower's 16 × 16 × 768 position embeddings (float32), read by the image path.
- `mel_filters_128x257_f32.bin`: the 128 × 257 Slaney mel filterbank (float32), read by the audio path.
- `Package.swift`, `Package.resolved` and `Sources/`: the Swift host, a copy of coreai-model-zoo `apps/D1Omni`.

## The Swift host

The library `D1Omni` and its Mac command-line tool `d1omni` take a state and typed questions (`noul`, `choice`,
`score`), with images or a 16 kHz mono 16-bit WAV clip if the request has them, and return the publisher's
probabilities. `D1Omni.folders(macos:)` and `D1Omni.MediaFolders.find(macos:)` find the bucket and media folders in a
folder laid out like `macos/` (`ios/` and `ios-h19p/` use the same folder names). `systemOne(state:questions:)` answers
a request, and its `images:` and `audio:` forms take the media. The package declares macOS 27 and iOS 27; it has been
built and run on a Mac only, and the iOS build has not been tried.

Its specification is the Python host, coreai-model-zoo `conversion/d1_omni/host.py`. On the fixtures, the Swift token
rows and graph inputs are bit-equal to the Python host's on every row. On all 562 fixture rows run through these graphs
on the Mac GPU, the Swift host's marker logits are bit-equal to the Python host's, with the AOT compile and with the
`.aimodel`.

## JPEG

`JPEGBaseline` decodes a baseline JPEG whose components share one sampling factor (no chroma subsampling, such as
4:4:4). It copies libjpeg-turbo's integer IDCT and colour conversion, so its pixels equal PIL's, which the publisher's
preprocessing reads. It is the default decoder for JPEG files.

Any other JPEG (4:2:0 or 4:2:2 chroma subsampling, progressive) falls back to ImageIO. ImageIO's pixels can differ from
PIL's: on one 4:4:4 baseline JPEG, 19 % of the bytes differed, by at most 3 levels. The fallback's effect on answers
was not measured. The images in `../reference/` are PNG, and ImageIO decodes the fixtures' PNG files bit-equal to PIL.
For pixel-exact input, pass a PNG or a baseline JPEG without chroma subsampling.
"""


HOST_README_PARITY = """Its specification is the Python host, coreai-model-zoo `conversion/d1_omni/host.py`. On the fixtures, the Swift token
rows and graph inputs are bit-equal to the Python host's on every row. On all 562 fixture rows run through these graphs
on the Mac GPU, the Swift host's marker logits are bit-equal to the Python host's, with the AOT compile and with the
`.aimodel`.
"""
# --with-small (round 13): what was run through which graph, now that the folders hold L64 and L128 too
HOST_README_PARITY_SMALL = """Its specification is the Python host, coreai-model-zoo `conversion/d1_omni/host.py`. A host picks the smallest
decision folder that holds a row (`D1Omni.folders(macos:)` lists every `decide-fp16-L<L>` folder). On the fixtures, the
Swift token rows and graph inputs are bit-equal to the Python host's on every row. Run through these graphs on the Mac
GPU, the Swift host's marker logits are bit-equal to the Python host's, with the AOT compile and with the `.aimodel`, on
562 fixture rows with the 256 to 4,096 graphs (text 470, image 46, audio 46) and on the 342 text rows of 128 positions
or fewer with the 64 and 128 graphs. Nine audio rows of 128 positions or fewer were run at 64 and 128 by the Python
host only.
"""


HOST_README_PLATFORMS = """The package declares macOS 27 and iOS 27; it has been
built and run on a Mac only, and the iOS build has not been tried.
"""
# --final (round 14): round 11 ran the package on the iPhone 18 Pro in the gate app
HOST_README_PLATFORMS_FINAL = """The package declares macOS 27 and iOS 27. It has
run on a Mac and, inside the gate app `apps/D1OmniGate` of coreai-model-zoo, on an iPhone 18 Pro (iOS 27.2).
"""


def stage_host_swift(out: Path) -> dict:
    """--ship: the Swift host (apps/D1Omni: Package.swift, Package.resolved, Sources/**/*.swift) copied into host/ file
    for file, and host/README.md. Recorded: each file's sha256 and whether it is older than the d1omni binary that ran
    the round 10 parity (Package.swift's comment was edited after the build, round 9)."""
    files = [SWIFT_HOST / f for f in SWIFT_HOST_FILES] + sorted((SWIFT_HOST / "Sources").rglob("*.swift"))
    built = SWIFT_BINARY.stat().st_mtime
    listing = []
    for f in files:
        rel = f.relative_to(SWIFT_HOST)
        clone(f, out / "host" / rel)
        listing.append({"path": f"host/{rel}", "sha256": sha256_file(f), "older_than_binary": f.stat().st_mtime < built})
    readme = HOST_README
    if SMALL_MODE:
        if readme.count(HOST_README_PARITY) != 1:
            raise SystemExit("host/README.md: the parity paragraph is not where it was")
        readme = readme.replace(HOST_README_PARITY, HOST_README_PARITY_SMALL)
    if FINAL_MODE:  # round 11 built the package for iOS (the gate app) and ran it on the phone
        if readme.count(HOST_README_PLATFORMS) != 1:
            raise SystemExit("host/README.md: the platforms sentence is not where it was")
        readme = readme.replace(HOST_README_PLATFORMS, HOST_README_PLATFORMS_FINAL)
    write_text(out / "host" / "README.md", readme)
    return {"from": "coreai-model-zoo apps/D1Omni", "files": listing, "binary": {
        "sha256": sha256_file(SWIFT_BINARY),
        "built": datetime.datetime.fromtimestamp(built).astimezone().isoformat(timespec="seconds")},
            "readme": "host/README.md"}


# =========================================================================== checks and the manifest
def text_file(path: Path) -> str | None:
    if path.suffix in (".mlirb", ".bin", ".f32", ".png", ".wav") or path.stat().st_size > 64 << 20:
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def checks(out: Path, files: list[Path], sha: dict[str, str], scrub: Scrub) -> dict:
    local_text, unpublished_hits, binary_local, user_name = [], [], [], []
    corpus_hits, tilde_text, tilde_allowed, tilde_binary = [], [], [], []  # --final
    home = str(Path.home()).encode()
    user = Path.home().name.encode()
    for f in files:
        t = text_file(f)
        rel = str(f.relative_to(out))
        if SHIP_MODE and user in f.read_bytes():
            user_name.append(rel)
        if FINAL_MODE:
            data = f.read_bytes()
            if MEASURED_ONLY_CORPUS.search(data):
                corpus_hits.append(rel)
            if t is None and TILDE_HOME_FOLDER.search(data):
                tilde_binary.append(rel)
            for n, line in enumerate((t or "").splitlines(), 1):
                if TILDE in line:
                    if (rel, line.strip()) in TILDE_ALLOWED:
                        tilde_allowed.append({"file": rel, "line": n,
                                              "line_sha256": hashlib.sha256(line.strip().encode()).hexdigest()})
                    else:
                        tilde_text.append(f"{rel}:{n}")
        if t is None:
            data = f.read_bytes()
            if b"/Users/" in data or home in data:
                binary_local.append(rel)
            continue
        if "/Users/" in t or str(Path.home()) in t:
            local_text.append(rel)
        # the publisher's tokenizer files are vocabularies (a BPE piece such as "coco" is not a record): path check only
        if "tokenizer" not in Path(rel).parts and (scrub.bad(t) or any(f'"{g}' in t for g in scrub.groups)):
            unpublished_hits.append(rel)
    # ios/ = macos/
    mismatch = []
    for f in files:
        rel = f.relative_to(out)
        if rel.parts[0] == "ios":
            twin = str(Path("macos", *rel.parts[1:]))
            if sha.get(twin) != sha[str(rel)]:
                mismatch.append(str(rel))
    result = {"text_files_with_a_local_path": local_text, "text_files_naming_unpublished_material": unpublished_hits,
              "ios_equals_macos": not mismatch, "ios_differs": mismatch[:20],
              "binary_files_with_a_local_path": binary_local,
              "binary_note": "MLIR debug locations in the graphs name the file of the exported module (the exporting "
                             "machine's path); not read by a loader. Stripping them (coreai-torch strip_debug_info) changes "
                             "the bundles' bytes and main.hash, so it is a re-export decision, not a staging step."}
    if SHIP_MODE:
        # the note must not spell the needle out: MANIFEST.json is itself grepped for it
        result["binary_note"] = ("the stripped bundles (strip_ship.py) and their AOT: every staged file, binary or text, "
                                 "was scanned for the macOS users-folder prefix, the home directory and the user name; "
                                 "any hit is fatal")
        result["files_with_the_user_name"] = user_name
        result["files_scanned"] = len(files)
    if FINAL_MODE:  # named in words: MANIFEST.json is scanned too
        result["final_scans"] = {
            "what": "round 14 (--final), fatal: the name of the measured-only speech corpus (case-insensitive) in any "
                    "file, binary or text; the tilde-slash home shorthand on any line of a text file, except the lines "
                    "listed under allowed (matched by their exact text, recorded here by sha256); in a binary file the "
                    "shorthand followed by a home folder name (code, Library, .cache, .config, .local, Documents, "
                    "Desktop, Downloads, Applications). A bare two-byte search is no test in a binary file: the fp16 "
                    "weights hold that byte pair hundreds of thousands of times",
            "files_naming_the_measured_only_corpus": corpus_hits,
            "text_lines_with_the_shorthand": tilde_text,
            "allowed": {"lines": tilde_allowed,
                        "why": "a code comment of the Swift host (host/ is coreai-model-zoo apps/D1Omni file for file) "
                               "that gives the Core AI runtime's cache location with placeholders for the OS build and "
                               "the process name: not a path of the converting machine"},
            "binary_files_with_a_home_folder_path": tilde_binary,
            "files_scanned": len(files)}
        if len(tilde_allowed) != len(TILDE_ALLOWED):
            raise SystemExit(f"staging check: the allowed line(s) not found as listed: {tilde_allowed}")
    if local_text or unpublished_hits or mismatch or (SHIP_MODE and (binary_local or user_name)) \
            or corpus_hits or tilde_text or tilde_binary:
        raise SystemExit(f"staging check failed: {json.dumps(result)[:2000]}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ship", action="store_true", help="round 10: the stripped bundles and their gates (see the "
                        "module docstring)")
    parser.add_argument("--with-small", action="store_true", help="round 13 (with --ship): + the decision graph at L64 "
                        "/ L128 (round 12's stripped bundles and gates)")
    parser.add_argument("--final", action="store_true", help="round 14 (with --ship; implies --with-small): the form to "
                        "upload (see the module docstring)")
    args = parser.parse_args()
    if args.ship:
        configure_ship()
    if args.with_small:
        configure_small()
    if args.final:
        configure_final()
    out = args.out
    if out.exists():
        raise SystemExit(f"{out} exists: never replaced (move it away first)")
    started, disk_before = now(), disk_free()
    fixtures = json.loads((WORK / FIXTURES[0]).read_text())
    unpublished = {r["id"] for r in fixtures["records"] if r["public"] is not True}
    public_sources = {r["source"] for r in fixtures["records"] if r["public"] is True}
    groups = ({r["source"] for r in fixtures["records"]} - public_sources) | {"card/image", "card/audio"}
    scrub = Scrub(unpublished, groups)
    record: dict = {}
    edits: list = []  # --final: every rewritten or removed value (MANIFEST.json `final`)
    for name in SHIP:
        stage_graph(name, out, scrub, record, edits)
        print(f"staged {name}", flush=True)
    snap = Path(hf_snapshot(MODEL_ID, revision=MODEL_SHA))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        src = (snap / f).resolve()
        if sha256_file(src) != sha256_file(MACOS / "fp16-L256" / "tokenizer" / f):
            raise SystemExit(f"{f}: the snapshot's and the bundle's differ")
        if SMALL_MODE:  # a Swift host reads the smallest bucket's tokenizer/ (L64 here): the same bytes everywhere
            for name in SHIP:
                if name.startswith("decide-") and sha256_file(source_folder(name) / "tokenizer" / f) != sha256_file(src):
                    raise SystemExit(f"{name}/tokenizer/{f}: differs from the snapshot's")
        clone(src, out / "tokenizer" / f)
    clone((snap / "LICENSE").resolve(), out / "LICENSE")
    meta = repo_metadata(record)
    write_json(out / "metadata.json", meta)
    write_json(out / "host" / "metadata.json", meta)
    clone(MACOS / "vision-fp16" / "position_table.f32", out / "host" / "position_table.f32")
    clone(MACOS / "audio-fp16-10s" / "mel_filters_128x257_f32.bin", out / "host" / "mel_filters_128x257_f32.bin")
    host_swift = stage_host_swift(out) if SHIP_MODE else None
    ref = stage_reference(out, scrub, unpublished, edits)
    write_text(out / "NOTICE.md", notice_text())
    if FINAL_MODE:  # the card the supervisor checked, byte for byte
        clone(CARD, out / "README.md")
        if sha256_file(out / "README.md") != sha256_file(CARD):
            raise SystemExit("README.md differs from the card")
    else:
        write_text(out / "README.md", "# d1-omni-600M-CoreAI\n\nThe card is written later (placeholder; nothing here is "
                                      "uploaded yet).\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    listing = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    result = checks(out, files, {i["path"]: i["sha256"] for i in listing}, scrub)
    tops = {}
    for item in listing:
        top = item["path"].split("/")[0] if "/" in item["path"] else "(root)"
        tops.setdefault(top, [0, 0])
        tops[top][0] += 1
        tops[top][1] += item["bytes"]
    manifest = {
        "schema": "d1-omni-hf-staging/1", "repo": HF_REPO, "uploaded": False, "staged": started, "finished": now(),
        "staged_by": "conversion/d1_omni/stage_hf.py", "source": {"model": MODEL_ID, "revision": MODEL_SHA},
        "files": len(listing), "bytes": sum(i["bytes"] for i in listing),
        "by_top_folder": {k: {"files": v[0], "bytes": v[1]} for k, v in sorted(tops.items())},
        "graphs": record, "reference": ref, "checks": result,
        **({"ship": {"what": "round 10: the bundles with their MLIR debug locations stripped (strip_ship.py), their h19p "
                             "AOT (aot_ship.py), gated again on them (the same numbers as the unstripped bundles)",
                     "gates": {"python_runtime_vs_unstripped": STRIP_COMPARE, "swift_parity": SWIFT_SHIP},
                     "host_swift": host_swift}} if SHIP_MODE else {}),
        **({"small": {"what": "round 13 (--with-small): the decision graph at L64 / L128 (round 12: exported, stripped into "
                              "macos-ship-small/, AOT in compiled/ship-h16c/ and ship-h19p/), added to the ship form by the "
                              "supervisor on 2026-10-08; a host routes a row by host.ALL_BUCKETS",
                      "buckets": list(host.SMALL_BUCKETS),
                      "gates": {"python_runtime_vs_unstripped": SMALL["compare"], "swift_parity": SMALL["swift_ship"],
                                "audio_e2e_all_buckets": SMALL["audio_e2e"]},
                      "not_run": "the Swift host on the 9 audio rows of 128 positions or fewer at L64 / L128 (the Python "
                                 "runtime ran them: audio_e2e_all_buckets)"}} if SMALL_MODE else {}),
        **({"final": {"what": "round 14 (--final): the form to upload. README.md is the card checked by the supervisor "
                              "(card/README.md in the lane, byte for byte); ios-h19p/ leaves out the graphs listed under "
                              "not_in_ios_h19p; the values listed under edits were rewritten or removed so that no file "
                              "names a measured-only source or a path on the converting machine; checks.final_scans",
                      "card": {"file": "card/README.md", "bytes": CARD.stat().st_size, "sha256": sha256_file(CARD)},
                      "not_in_ios_h19p": {n: record[n]["ios_h19p_not_shipped"] for n in NOT_H19P if n in record},
                      "edits": edits,
                      "host_readme": "the platforms sentence says where the package has run (a Mac; the iPhone 18 Pro "
                                     "in the gate app, round 11)"}} if FINAL_MODE else {}),
        "provenance_items_removed": scrub.removed,
        "not_staged": ["each bundle folder's reference.json (rows of every fixture record, published or not)",
                       "the per-row lists of the export and gate files", "coreai-build logs"],
        "disk": {"before": disk_before, "after": disk_free(), "note": "every graph and media file is an APFS clone"},
        "upload": {"run": False, "needs": "the user's GO", "command": f"HF_HUB_DISABLE_XET=1 hf upload-large-folder {HF_REPO} "
                   "<this folder> --repo-type model"},
        "listing": listing,
    }
    text = json.dumps(manifest, ensure_ascii=False)
    if "/Users/" in text or str(Path.home()) in text or (SHIP_MODE and Path.home().name in text):
        raise SystemExit("MANIFEST.json would hold a local path or the user name")
    if FINAL_MODE and (TILDE in text or MEASURED_ONLY_CORPUS.search(text.encode())):
        raise SystemExit("MANIFEST.json would hold the tilde shorthand or the measured-only corpus's name")
    write_json(out / "MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("files", "bytes", "by_top_folder", "reference", "provenance_items_removed")}
                     | {"binary_files_with_a_local_path": len(result["binary_files_with_a_local_path"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
