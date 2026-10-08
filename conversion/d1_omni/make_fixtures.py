#!/usr/bin/env python3
"""Fixture records for d1-omni-600M: text, image and audio requests, read from pinned files or written here.

    python3 conversion/d1_omni/make_fixtures.py [--fetch]   # -> $ZOO_WORK_ROOT/_d1_omni/fixtures/records.json

A record is `{id, source, public, request: {state, questions: {qid: {type, instructions, criteria}}}, media, gold,
note, provenance}`. `request` is what the publisher's `system_one(state, questions, images=, audio=)` takes, with the
media named in `media` (null, {"images": [path, ...]} or {"audio": path}, relative to the work dir); `gold` maps a
question id to the key its probabilities are reported under (choice: the criteria name; noul: "true" / "false";
score: the level index as a string), or is null when the source gives no answer. `public` says whether the record's
text may go into a published file. The request JSON of every record taken from another fixture keeps that record's
keys in that record's order. Sources, in file order:

* `card_*` - the examples of the publisher's model card ("How to use", README.md @ 414f8d64): the text request
  (refund noul / team choice / urgency score), the two batch tickets (team), the two cats (COCO val2017 39769,
  measured only: COCO images carry attribution terms) and the voice note (Narsil/asr_dummy 1.flac, a LibriSpeech
  clip, CC BY 4.0: measured only). gold null: the card shows no answers.
* `tv4_000`..`tv4_059` - the Kev port's records (`$ZOO_WORK_ROOT/_kev/fixtures/records.json`, sha256-pinned): the
  first 60 lines of transfer-v4 `development.jsonl` (github.com/jaredpalmer/kev at tag kev-1.0), all MMLU (MIT).
  request (less Kev's `model` field), gold, note and provenance are copied; `provenance.line_sha256` is added from
  the pinned development.jsonl (sha256 of the line's UTF-8 bytes, the key the LiteRT lane's records carry).
* `semif_<id>` - SemIf authored144 (MIT, github.com/TheoLeeCJ/SemIf at ca3ba65f), mapped as the Kev port maps it
  (state -> state, question -> instructions, options -> criteria {id: description} in file order, one choice
  question `decision`, gold = options[label].id); every request is asserted equal to the Kev record of the same id.
* `own_*` - the 20 records written for the Kev port: the 11 published in `conversion/kev/own_records_public.json`
  (public) and the 9 withheld after that port's name screen, read from `$ZOO_WORK_ROOT/_kev/fixtures/
  own_records_withheld.json` with each request's sha256 asserted against its public stub (public false). A question
  the publisher's `as_question` rejects (no `instructions`) is left out of the request and listed in
  `summary.dropped`; nothing else in the request changes.
* `tv4x_<source>_<k>` (140) and `tv4s_<k>` (20) - the Kev port's other transfer-v4 slices: measured only (their
  sources include tweet_eval, licence unknown, and SciQ, CC BY-NC: their text is not published).
* `long_3400` - own_L01 (a sorter log) continued with more lines of the same log, using only the names, places and
  equipment already in it, to about 3,400 state tokens (the card's long-state column); own_L01's three questions
  plus one whose answer is in the added lines.
* `img_01`..`img_03` - image requests whose questions read the picture (room, bed count, lamp, suitcase,
  brightness, tidiness), public. The pictures (round 5) are copied or downscaled into `images/` from two sieved
  pools, read only: Wikimedia Commons CC0 photographs of hotel rooms (the decider-2b-vision lane's c1_real set,
  every file's licence template cc-zero, no person in frame) and this zoo's own FLUX.2 klein 4B output (prompt,
  seed and revision in its manifest). Each picked file was looked at by eye at full size before its questions were
  written: no people, no legible text or logo, no existing design; gold is the answer seen there.
  `images/manifest.json` (written here) keeps every file's origin, licence, resize and sha256.
* `imgm_01`..`imgm_12` - the same kind of requests over 8 more CC0 photographs (4 at 1024 px = tiled, 4 downscaled
  to 512 px = one crop) and 4 more FLUX rooms (1024 x 1024 = 2 x 2 tiles + thumbnail): the measured image rows
  (public false: they are the gate's, not the card's).
* `aud_01`..`aud_03` - voice notes synthesised with Kokoro-82M from the scripts here (`audio/script.json`; the
  synthesis writes `audio/manifest.json` and the 16 kHz mono int16 wav files). No names.
* `aud_04`..`aud_15` (round 6) - twelve more voice notes the same way (K/scripts/synth_aud_r6.py appends them to
  `audio/manifest.json`), English requests from a user to an assistant, no names, at lengths across the audio
  graph's clip buckets (5 / 10 / 20 / 30 s): about 3 s x2, 8 s x2, 15 s x2, 22 s x2, 29 s x2, one under 0.5 s (the
  publisher pads it to 0.5 s) and one over 30 s (the publisher cuts it to 30 s). Each asks aud_01's three
  questions; gold = the answer the script was written to carry (aud_14, one word, has no urgency gold).
* `red_arm_000` - tv4_000 with "correctly" -> "incorrectly" in its instructions: a gate arm that must fail when
  compared with tv4_000's reference, not a fixture (public false).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import code_path, hf_snapshot, repo_root, work_path  # noqa: E402
from host import (MODEL_ID, MODEL_SHA, QTYPES, RawTokenizer, as_question, escape, image_crops,  # noqa: E402
                  image_prefix_length, layout, serialize)

import numpy as np  # noqa: E402

WORK = work_path("_d1_omni")
SCHEMA = "d1-omni-fixtures/1"
CARD = {"file": "README.md", "sha256": "a18d4aa959198a001bc35be2309b3b21a265473fa59275e23823862c9544b411",
        "section": "How to use"}
COCO = {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "path": "images/coco_000000039769.jpg",
        "sha256": "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e", "bytes": 173131,
        "license": "COCO: images under their Flickr authors' terms (attribution) - measured only, never published"}
FLAC = {"url": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac", "path": "audio/card_1.flac",
        "sha256": "30885601173f96b0d8ddd020dc959b055c6c1582b85a33e3fcab8c4b08ed94c2", "bytes": 183318,
        "license": "LibriSpeech (CC BY 4.0) via Narsil/asr_dummy - measured only, never published"}
KEV = {"path": str(work_path("_kev", "fixtures", "records.json")),
       "sha256": "a8ee72eb06fc7522b5f719d01b53e3686c902536f5bcda3071835dc9ac425969", "records": 384}
TRANSFER_V4 = {"path": str(work_path("_kev", "kev-src", "evals", "v4", "transfer-v4", "development.jsonl")),
               "path_in_repo": "evals/v4/transfer-v4/development.jsonl",
               "sha256": "ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e", "lines": 764,
               "author_repo": {"repo": "github.com/jaredpalmer/kev", "tag": "kev-1.0",
                               "commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc"}}
TV4_SOURCE_LICENSES = {  # per-source licence of the underlying dataset (the Kev port's record of the Hub cards)
    "mmlu": {"dataset": "cais/mmlu", "license": "mit"},
    "emotion": {"dataset": "dair-ai/emotion", "license": "other (card)"},
    "tweet_offensive": {"dataset": "cardiffnlp/tweet_eval", "license": "unknown (card)"},
    "qnli": {"dataset": "nyu-mll/glue", "license": "other (card; QNLI derives from SQuAD, CC BY-SA 4.0)"},
    "paws": {"dataset": "google-research-datasets/paws", "license": "other (card)"},
    "sciq": {"dataset": "allenai/sciq", "license": "cc-by-nc-3.0"},
    "legacy_holdout": {"dataset": None, "license": "generated by the author (Apache-2.0 repository)"},
    "composition_holdout": {"dataset": None, "license": "generated by the author (Apache-2.0 repository)"},
}
SEMIF = {"path": str(code_path("codex-conversions", "2026-09-21", "semif-ondevice", "fixtures", "authored144.jsonl")),
         "license_path": str(code_path("codex-conversions", "2026-09-21", "semif-ondevice", "semif", "LICENSE")),
         "sha256": "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079",
         "license_sha256": "f765f2140f8507a8f0d81ec0fd2c4bd72fe6a066841ef27883ff876a76bf61be",
         "upstream_repo": "github.com/TheoLeeCJ/SemIf", "upstream_path": "benchmarks/data/authored144.jsonl",
         "upstream_commit": "ca3ba65f142967030ecb453346e94d6f476a69df", "license": "MIT",
         "copyright_notice": "Copyright (c) 2026 TheoLeeCJ", "rows": 144, "question_id": "decision"}
OWN = {"public_path": str(repo_root() / "conversion" / "kev" / "own_records_public.json"),
       "public_sha256": "f7a3bb2cef5f3379aa7d0f6bf82df89c62824c4b9e87010d5f3f50027791b2b7",
       "withheld_path": str(work_path("_kev", "fixtures", "own_records_withheld.json")),
       "withheld_sha256": "8ee259e9b567ab61b7199fbc6b2c475d2a5263bd31ba76451b34c29d595d6d29"}
LONG_TOKENS = (3300, 3500)        # long_3400: state tokens (d1 tokenizer, escape()d, no special tokens)
LONG_1500_TOKENS = (1300, 1700)   # an own_L0x state in this range stands in for a 1.5k long_1500 record
RED_ARM = {"from": "tv4_000", "old": "correctly", "new": "incorrectly"}

# ---------------------------------------------------------------------------------------------- card examples
CARD_STATE = "I was charged twice this month, please refund one of them."
CARD_QUESTIONS = {
    "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
    },
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}
CARD_CATS = {
    "type": "choice",
    "instructions": "How many cats are there?",
    "criteria": {"one": "One", "two": "Two", "more": "Three or more"},
}
CARD_AUDIO_STATE = "Voice note from a user."
CARD_TOPIC = {
    "type": "choice",
    "instructions": "What is the speaker talking about?",
    "criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": "The weather"},
}
CARD_TICKETS = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
# Strings that must appear in the card verbatim (checked against the pinned README).
CARD_SNIPPETS = ['print(model.system_one("I was charged twice this month, please refund one of them.", questions))',
                 '"instructions": "Is the customer asking for a refund?"', '"instructions": "Which team should handle this?"',
                 '"billing": "Charges, refunds, invoices"', '"technical": "App or site faults"',
                 '"fraud": "Suspected unauthorised use"', '"instructions": "How urgent is this?"',
                 '"criteria": ["Can wait", "Today", "Blocking the customer now"]',
                 '"instructions": "How many cats are there?"',
                 '"criteria": {"one": "One", "two": "Two", "more": "Three or more"}',
                 '"instructions": "What is the speaker talking about?"',
                 '"criteria": {"food": "Food and meals", "travel": "Travel and transport", "weather": "The weather"}',
                 'print(model.system_one("Voice note from a user.", {"topic": topic}, audio=audio))',
                 'tickets = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]',
                 'print(model.system_one_batch([(t, {"team": questions["team"]}) for t in tickets]))',
                 'image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")',
                 'url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"',
                 'print(model.system_one(None, {"cats": cats}, images=[image]))']

# ---------------------------------------------------------------------------------------------- image requests
ROOMS = {"kitchen": "A kitchen", "bedroom": "A bedroom", "office": "An office", "bathroom": "A bathroom"}
BRIGHTNESS = ["Dark", "Dim", "Bright"]
TIDY = ["Very cluttered", "Some clutter", "Tidy"]
BEDS = {"one": "One bed", "two": "Two beds", "more": "Three or more beds"}
Q_ROOM = {"type": "choice", "instructions": "What kind of room is this?", "criteria": ROOMS}
Q_BEDS = {"type": "choice", "instructions": "How many beds are in the room?", "criteria": BEDS}
Q_BRIGHT = {"type": "score", "instructions": "How bright is the room?", "criteria": BRIGHTNESS}
Q_TIDY = {"type": "score", "instructions": "How tidy is the room?", "criteria": TIDY}
Q_LAMP = {"type": "noul", "instructions": "Is a lamp switched on?"}
Q_WINDOW = {"type": "noul", "instructions": "Is there a window in the picture?"}
Q_SUITCASE = {"type": "noul", "instructions": "Is there a suitcase in the room?"}
Q_UNMADE = {"type": "noul", "instructions": "Is a bed unmade?"}
CHECK_STATE = "Photo from a room check."

# The two picture pools (round 5), read only. CC0: the decider-2b-vision lane's Wikimedia Commons hotel-room set
# (every file's licence template in the cc-zero family, no person in frame, re-saved as 1024 px PNG; the manifest
# lists the Commons file, page, artist, original sha256 and resize, not the PNG's sha256, which is pinned here).
# FLUX: this zoo's FLUX.2 klein 4B (Core AI int4) rooms; rooms-manifest.json holds the six looked at by eye before use
# (prompt, seed, hub revision, sha256), the gen manifest all twelve.
IMAGE_POOLS = {
    "cc0": {"dir": work_path("_decidervision_demo", "pretest", "c1_real", "images"),
            "manifest": code_path("standup", "drafts", "2026-09-29-decidervision-demo-attachments", "pretest",
                                  "c1_real-images-manifest.json"),
            "manifest_sha256": "eb05cad78ac038f27eb99e8a109b035fbb3194d9698f3bc6609e09120ae6cb7f",
            "license": "CC0 1.0 (Wikimedia Commons licence template in the cc-zero family; no attribution required)"},
    "flux": {"dir": work_path("_decidervision_demo", "demo", "gen", "images"),
             "manifest": code_path("standup", "drafts", "2026-09-29-decidervision-demo-attachments", "demo-runs",
                                   "rooms-manifest.json"),
             "manifest_sha256": "99f26f8a961dfd168df20855cc1c47bb17ff5a5c1ad6b9306b114305b1e9a5aa",
             "license": "own model output: FLUX.2 klein 4B (Apache-2.0) on Core AI, no third-party image involved"},
}
CHECKED = ("looked at by eye at full size (round 5, 2026-10-08) before the questions were written: no person, "
           "no legible text or logo, no existing design")
# id, public, pool, source file, its sha256, the size written (None = the file as it is), what it shows, the request
IMAGES = [
    {"id": "img_01", "public": True, "pool": "flux", "name": "room3_05",
     "source_sha256": "c4848b5361ecac7ec0e09ebd5a38e11c12b62d42a9c1443f76fed2487f9ee913", "size": [384, 384],
     "shows": "a made double bed seen from a doorway in daylight, folded towels, one bedside lamp on, a closed "
              "suitcase on the floor",
     "state": None, "questions": {"room": Q_ROOM, "suitcase": Q_SUITCASE, "brightness": Q_BRIGHT},
     "gold": {"room": "bedroom", "suitcase": "true", "brightness": "2"}},
    {"id": "img_02", "public": True, "pool": "cc0", "name": "real_14",
     "source_sha256": "cf5d06af977fc100fee35e73c1d5281a1c23a543df9448252f68a9eef5f99bd7", "size": [384, 288],
     "shows": "a hotel bedroom at night lit by two switched-on table lamps; a made bed, an armchair, a desk phone "
              "(keypad and clock face not legible)",
     "state": CHECK_STATE, "questions": {"room": Q_ROOM, "lamp": Q_LAMP, "brightness": Q_BRIGHT},
     "gold": {"room": "bedroom", "lamp": "true", "brightness": "1"}},
    {"id": "img_03", "public": True, "pool": "cc0", "name": "real_18",
     "source_sha256": "28080dd775920ac0b701d6640eb08e0500542946a19c84ab80e263e595224312", "size": None,
     "shows": "two made single beds side by side, a folded towel on the nearer one, a desk lamp (off) and a wooden "
              "lattice on the wall; portrait 768 x 1024 (tiled)",
     "state": None,
     "questions": {"beds": Q_BEDS, "towel": {"type": "noul", "instructions": "Is there a folded towel on a bed?"},
                   "tidy": Q_TIDY},
     "gold": {"beds": "two", "towel": "true", "tidy": "2"}},
    # measured only: 4 CC0 at 1024 px (tiled), 4 CC0 downscaled to 512 px (one crop), 4 FLUX at 1024 x 1024 (tiled)
    {"id": "imgm_01", "public": False, "pool": "cc0", "name": "real_10",
     "source_sha256": "7a57a5d4bceb2107661dca9d49190519eddfd792373d35e4038a9456884510ef", "size": None,
     "shows": "a hotel room in daylight with two unmade beds, a desk with things left on it and an office chair",
     "state": None, "questions": {"beds": Q_BEDS, "unmade": Q_UNMADE, "tidy": Q_TIDY},
     "gold": {"beds": "two", "unmade": "true", "tidy": "1"}},
    {"id": "imgm_02", "public": False, "pool": "cc0", "name": "real_17",
     "source_sha256": "4c398e2557ac20ea126dbcfb39d68b1c53e1b2fa13368bc7b436a0dba1beca48", "size": None,
     "shows": "a hostel dormitory with three bunk beds, a jacket and a towel hanging from them (small bed labels, not "
              "legible)",
     "state": None,
     "questions": {"beds": Q_BEDS, "clothes": {"type": "noul", "instructions": "Is there clothing hanging on a bed?"},
                   "tidy": Q_TIDY},
     "gold": {"beds": "more", "clothes": "true", "tidy": "1"}},
    {"id": "imgm_03", "public": False, "pool": "cc0", "name": "real_19",
     "source_sha256": "9c0b2ca1b09ae1ee4f6c9d7a49d6be0fa216be39df5efdac020d267db497bce9", "size": None,
     "shows": "a tidy hotel room: one made double bed, a curtained window, a desk with a yellow chair, an armchair "
              "(a card and wrappers on the desk, a wall screen: nothing legible)",
     "state": CHECK_STATE, "questions": {"beds": Q_BEDS, "window": Q_WINDOW, "tidy": Q_TIDY},
     "gold": {"beds": "one", "window": "true", "tidy": "2"}},
    {"id": "imgm_04", "public": False, "pool": "cc0", "name": "real_21",
     "source_sha256": "9bd2b4e01b5af86deef83f3e7fd5179029d187dc69f01c99044eb41d87cc93e1", "size": None,
     "shows": "a bedroom at night: a made bed between two switched-on table lamps, two abstract paintings, a phone",
     "state": None, "questions": {"room": Q_ROOM, "lamp": Q_LAMP, "brightness": Q_BRIGHT},
     "gold": {"room": "bedroom", "lamp": "true", "brightness": "1"}},
    {"id": "imgm_05", "public": False, "pool": "cc0", "name": "real_02",
     "source_sha256": "6f863bef6c2cdef41aa595feed3f93999f983899673dda042f4b2ea7d28f0019", "size": [512, 342],
     "shows": "a single hotel room: one made bed, a leather armchair, a glass table, two lamps on, a bright window",
     "state": None,
     "questions": {"beds": Q_BEDS, "armchair": {"type": "noul", "instructions": "Is there an armchair in the room?"},
                   "tidy": Q_TIDY},
     "gold": {"beds": "one", "armchair": "true", "tidy": "2"}},
    {"id": "imgm_06", "public": False, "pool": "cc0", "name": "real_07",
     "source_sha256": "81f1e99f5f6ea6d4917bdbd8b26f213a34dbd878ae6104e9d18cc7920c7110f0", "size": [384, 512],
     "shows": "a small single room in daylight: one bed, a window with curtains, a chair and table, a desk with a "
              "monitor (printed sheets on the desk and bed, not legible); portrait",
     "state": None, "questions": {"beds": Q_BEDS, "window": Q_WINDOW, "brightness": Q_BRIGHT},
     "gold": {"beds": "one", "window": "true", "brightness": "2"}},
    {"id": "imgm_07", "public": False, "pool": "cc0", "name": "real_08",
     "source_sha256": "3ce11ab8425eb7a0f960f0c998c3e8ba8fa327bc392618c166a58e50c6389434", "size": [512, 384],
     "shows": "a twin room in the evening: two made single beds, a pendant lamp on, an armchair (a phone screen and "
              "a clock on the nightstand, not legible)",
     "state": CHECK_STATE, "questions": {"beds": Q_BEDS, "lamp": Q_LAMP, "brightness": Q_BRIGHT},
     "gold": {"beds": "two", "lamp": "true", "brightness": "1"}},
    {"id": "imgm_08", "public": False, "pool": "cc0", "name": "real_13",
     "source_sha256": "32b868460ceb28a1fa04099c48d43e106a1fd14aa53787fb99179e9edf8baec5", "size": [512, 339],
     "shows": "a low double bed in a dark room with warm accent lights on a wooden wall; no window",
     "state": None, "questions": {"room": Q_ROOM, "window": Q_WINDOW, "brightness": Q_BRIGHT},
     "gold": {"room": "bedroom", "window": "false", "brightness": "1"}},
    {"id": "imgm_09", "public": False, "pool": "flux", "name": "room3_01",
     "source_sha256": "42a59f4ad46be290f8dad7f7d2a91d74b3b48cd69b9aaf8b84639fffe98d6c9c", "size": None,
     "shows": "a made double bed seen from a doorway in daylight, folded towels, the bedside lamps off",
     "state": CHECK_STATE, "questions": {"beds": Q_BEDS, "lamp": Q_LAMP, "tidy": Q_TIDY},
     "gold": {"beds": "one", "lamp": "false", "tidy": "2"}},
    {"id": "imgm_10", "public": False, "pool": "flux", "name": "room3_03",
     "source_sha256": "7d4ed0f2652a25a4901a6aa05f837b9978b5b85db9a9a2f22e37536f4304dd9f", "size": None,
     "shows": "a made double bed on a wooden floor seen from a doorway, folded towels, a bedside lamp glowing",
     "state": None, "questions": {"room": Q_ROOM, "lamp": Q_LAMP, "brightness": Q_BRIGHT},
     "gold": {"room": "bedroom", "lamp": "true", "brightness": "2"}},
    {"id": "imgm_11", "public": False, "pool": "flux", "name": "room3_04",
     "source_sha256": "0d256b4662ed9aa1f9c92328e200d0ac6429e3f54d2f25be5d97c14dc891fbba", "size": None,
     "shows": "a bright white bedroom seen from a doorway: a made bed with folded towels, no suitcase",
     "state": None, "questions": {"beds": Q_BEDS, "suitcase": Q_SUITCASE, "brightness": Q_BRIGHT},
     "gold": {"beds": "one", "suitcase": "false", "brightness": "2"}},
    {"id": "imgm_12", "public": False, "pool": "flux", "name": "room3_06",
     "source_sha256": "284d7bc44a35d03397399488476f2db8fd0c5cd531d2017074558bf3b7c94413", "size": None,
     "shows": "an unmade double bed with rumpled sheets and towels seen from a doorway, a window, the lamp off",
     "state": CHECK_STATE, "questions": {"beds": Q_BEDS, "unmade": Q_UNMADE, "tidy": Q_TIDY},
     "gold": {"beds": "one", "unmade": "true", "tidy": "1"}},
]
IMAGE_RESIZE = {"filter": "PIL.Image.BICUBIC", "call": "Image.resize(size, resample=Image.BICUBIC)",
                "png": "Image.save(format='PNG', icc_profile=<the source file's, if any>)"}
# The fixture version this one replaces (kept next to it byte for byte, _fixtures.py): every record but the rewritten
# ones is asserted unchanged. Round 5 (v2) rewrote img_01..03 over v1; round 6 (v3) only adds aud_04..aud_15 to v2.
PREVIOUS = {"file": "records.v2.json", "sha256": "07ea38a2c0f3d8da4afe4dd4ae79e083c46404cb8ab0ba150b9c0a3fca83ae62",
            "rewritten": []}

# ---------------------------------------------------------------------------------------------- audio requests
TOPICS = {"delivery": "A delivery or a damaged item", "billing": "Charges and payments", "account": "Account access",
          "feedback": "Feedback about the service"}
AUDIO_QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the speaker talking about?", "criteria": TOPICS},
    "wants": {"type": "noul", "instructions": "Is the speaker asking for something to be done?"},
    "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["Can wait", "Soon", "Right now"]},
}
AUDIO = [  # Kokoro-82M voices in the local cache; speed 1.0; English; no names
    {"id": "aud_01", "voice": "am_michael", "state": "Voice note from a user.",
     "text": "Hi, the blender I ordered last week arrived with a cracked jug. Could you send me a replacement "
             "before Friday? I need it for the weekend.",
     "gold": {"topic": "delivery", "wants": "true", "urgency": "1"}},
    {"id": "aud_02", "voice": "bf_emma", "state": "Voice note from a user.",
     "text": "I just wanted to say that the new update works really well. The app opens much faster now, and I "
             "have not had a single problem this week. Thanks.",
     "gold": {"topic": "feedback", "wants": "false", "urgency": "0"}},
    {"id": "aud_03", "voice": "af_heart", "state": None,
     "text": "I can't log in to my account, and I have a payment due in an hour. Please reset my password right "
             "away. The reset email never arrived.",
     "gold": {"topic": "account", "wants": "true", "urgency": "2"}},
]
SYNTH = {"model": "hexgrad/Kokoro-82M", "package": "kokoro 0.9.4 + misaki", "lang_code": "a", "speed": 1.0,
         "seed": 0, "native_rate": 24000, "rate": 16000, "resample": "ffmpeg -ar 16000 -ac 1 -sample_fmt s16",
         "seconds": [6.0, 12.0]}
# Round 6: the same synthesis per clip (speed and target length per clip; the voices are the five in the local cache).
# A clip with `trim` keeps only its speech: the 16 kHz int16 samples from 20 ms before the first to 20 ms after the
# last sample with |x| >= 328 (0.01 of full scale).
SYNTH_R6 = {k: v for k, v in SYNTH.items() if k not in ("speed", "seconds")} | {
    "round": 6, "speed": "per clip", "seconds": "per clip (target_seconds)",
    "trim": {"threshold_abs_int16": 328, "margin_samples": 320}}
VOICE_NOTE = "Voice note from a user."
AUDIO_R6 = [  # gold: topic / wants / urgency, the answer each script was written to carry
    {"id": "aud_04", "voice": "af_bella", "speed": 1.0, "target_seconds": [2.5, 3.5], "state": VOICE_NOTE,
     "text": "Please reset my password right away.",
     "gold": {"topic": "account", "wants": "true", "urgency": "2"}},
    {"id": "aud_05", "voice": "am_fenrir", "speed": 1.0, "target_seconds": [2.5, 3.5], "state": None,
     "text": "No rush, but please update my billing address.",
     "gold": {"topic": "billing", "wants": "true", "urgency": "0"}},
    {"id": "aud_06", "voice": "bf_emma", "speed": 1.0, "target_seconds": [7.0, 9.0], "state": VOICE_NOTE,
     "text": "Hello, my parcel was marked as delivered yesterday, but it is not at my door or with my neighbours. "
             "Could you look into it this week?",
     "gold": {"topic": "delivery", "wants": "true", "urgency": "1"}},
    {"id": "aud_07", "voice": "af_heart", "speed": 1.0, "target_seconds": [7.0, 9.0], "state": None,
     "text": "I just want to say that the support team was very kind on the phone today. Everything is sorted now, "
             "and I am very happy. Thank you.",
     "gold": {"topic": "feedback", "wants": "false", "urgency": "0"}},
    {"id": "aud_08", "voice": "am_michael", "speed": 1.0, "target_seconds": [13.5, 16.5], "state": VOICE_NOTE,
     "text": "Hi there. I looked at my statement this morning, and I was charged twice for the same monthly plan. The "
             "second payment went out on the third. I would like the extra charge refunded, ideally before my next "
             "statement at the end of the month.",
     "gold": {"topic": "billing", "wants": "true", "urgency": "1"}},
    {"id": "aud_09", "voice": "af_bella", "speed": 1.0, "target_seconds": [13.5, 16.5], "state": None,
     "text": "Someone changed the email address on my account last night, and it was not me. I have already changed "
             "my password, but I am still worried. Please lock the account for now, and call me back as soon as you "
             "possibly can.",
     "gold": {"topic": "account", "wants": "true", "urgency": "2"}},
    {"id": "aud_10", "voice": "bf_emma", "speed": 1.0, "target_seconds": [20.5, 23.5], "state": VOICE_NOTE,
     "text": "I wanted to leave some feedback about the new delivery tracking page. It is much clearer than the old "
             "one. I can see where my order is on the map, and the time window has been right every time so far. The "
             "messages are short and friendly, and I no longer have to call to ask where things are. My neighbours "
             "have started using it too, and they all say the same thing. Keep up the good work, and thank you.",
     "gold": {"topic": "feedback", "wants": "false", "urgency": "0"}},
    {"id": "aud_11", "voice": "am_fenrir", "speed": 1.0, "target_seconds": [20.5, 23.5], "state": None,
     "text": "Good morning. I ordered a set of kitchen chairs two weeks ago, and the delivery date has moved three "
             "times now. Each time I took a morning off work to wait for the van, and each time nobody came, and "
             "nobody called to tell me. I am having family over next Saturday, and I really need the chairs by then. "
             "Could you tell me the real delivery date and, if you can, put them on the earliest van this week? A "
             "short text message with the date would be perfect. Thank you very much for your help.",
     "gold": {"topic": "delivery", "wants": "true", "urgency": "1"}},
    {"id": "aud_12", "voice": "af_heart", "speed": 0.95, "target_seconds": [27.5, 29.8], "state": VOICE_NOTE,
     "text": "Hi, I am calling about my bill. This morning I got a message saying that my card was declined and that "
             "my service will be switched off at noon today. The card works everywhere else, and I paid the last "
             "three bills on time. I have tried to update the card details in the app twice, and both times it "
             "showed an error, so I am stuck. Please take the payment by hand or keep my service on, because I work "
             "from home and I need it for a meeting today. Thank you, and please be quick, because noon is not far "
             "away.",
     "gold": {"topic": "billing", "wants": "true", "urgency": "2"}},
    {"id": "aud_13", "voice": "am_michael", "speed": 1.0, "target_seconds": [27.5, 29.8], "state": None,
     "text": "Hello. This is not urgent at all, so please take your time. I would like to change the name on my "
             "account to the name I use now, and I would also like to add a second phone number. I looked in the "
             "settings, but I could not find where to add a second number, and the help page did not mention it "
             "either. Whenever you have a moment this month, please tell me what I need to send you, and I will "
             "reply as soon as I can. There is no hurry, and next month would also be fine. Thanks a lot, and have a "
             "good day.",
     "gold": {"topic": "account", "wants": "true", "urgency": "0"}},
    {"id": "aud_14", "voice": "af_bella", "speed": 1.5, "target_seconds": [0.2, 0.45], "state": None, "trim": True,
     "text": "Refund.",
     "gold": {"topic": "billing", "wants": "true"}},
    {"id": "aud_15", "voice": "bf_emma", "speed": 1.0, "target_seconds": [33.0, 37.0], "state": VOICE_NOTE,
     "text": "Hello, I hope you can help me quickly. My grocery order was meant to arrive between eight and nine this "
             "morning, and it is now almost eleven. The tracking page still says the van is on its way, but it has "
             "not moved for an hour. I need the milk in that order for my baby's bottles, so please send the driver "
             "now, or tell me where the order is so that I can collect it myself. I have already waited two hours, "
             "and I cannot go out and leave the baby alone for long. If the van has broken down, I would rather have "
             "a refund than wait for another slot tomorrow. My phone is on, and I will keep it next to me all "
             "morning. Please call me back as soon as you see this. Thank you.",
     "gold": {"topic": "delivery", "wants": "true", "urgency": "2"}},
]

# ---------------------------------------------------------------------------------------------- long_3400
LONG_QUESTION = {"tension_10": {"type": "noul",
                                "instructions": "At the 10:00 check, was the lane 7 belt tension within specification?"}}
LONG_GOLD = {"tension_10": "true"}
LONG_STEP = 4          # minutes between the continued log's throughput lines (own_L01 uses 3; 4 reaches the
                       # 10:00 check the shift note announces within the ~3,400-token budget)
LONG_UNTIL = (10, 20)  # the last routine line of the continued log (chosen once for ~3,400 state tokens)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_pinned(path: str | Path, sha256: str, what: str) -> bytes:
    data = Path(path).read_bytes()
    got = _sha256(data)
    if got != sha256:
        raise SystemExit(f"{what}: {path} sha256 {got} != pinned {sha256}")
    return data


def record(rid, source, public, state, questions, media=None, gold=None, note="", provenance=None):
    return {"id": rid, "source": source, "public": public, "request": {"state": state, "questions": questions},
            "media": media, "gold": gold, "note": note, "provenance": provenance or {}}


def strip_model(request: dict) -> dict:
    """A Kev request without its `model` field (d1 takes no model name), every other key in its order."""
    return {k: copy.deepcopy(v) for k, v in request.items() if k != "model"}


# ---------------------------------------------------------------------------------------------- media
def fetch(spec: dict, allow: bool) -> tuple[Path | None, str]:
    path = WORK / spec["path"]
    if not path.exists():
        if not allow:
            return None, f"{spec['path']} missing (run with --fetch)"
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec["url"], timeout=60) as r:
            data = r.read()
        path.write_bytes(data)
    data = path.read_bytes()
    if _sha256(data) != spec["sha256"] or len(data) != spec["bytes"]:
        raise SystemExit(f"{path}: sha256 / size differ from the pinned download")
    return path, ""


def image_info(path: Path) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
        mode = im.mode
    return {"px": [width, height], "mode": mode, "prefix_positions": image_prefix_length([(width, height)])}


def audio_info(path: Path) -> dict:
    import soundfile as sf

    info = sf.info(str(path))
    data, rate = sf.read(str(path), dtype="int16")  # as the card reads it
    return {"sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames,
            "seconds": round(info.frames / info.samplerate, 4), "subtype": info.subtype, "format": info.format,
            "read_dtype": str(data.dtype), "read_shape": list(data.shape), "read_rate": rate,
            "note": "the card passes sf.read(dtype='int16') as is; waveform() expects 16 kHz mono and does not resample"}


# ---------------------------------------------------------------------------------------------- sources
def card_records(readme: str, fetch_media: bool) -> list[dict]:
    for snippet in CARD_SNIPPETS:
        if snippet not in readme:
            raise SystemExit(f"the card no longer contains: {snippet}")
    prov = {"file": CARD["file"], "repo": MODEL_ID, "revision": MODEL_SHA, "sha256": CARD["sha256"],
            "section": CARD["section"]}
    out = [record("card_text", "card", True, CARD_STATE, copy.deepcopy(CARD_QUESTIONS),
                  note="the card's text example (three named questions over one state)", provenance=prov)]
    for i, ticket in enumerate(CARD_TICKETS):
        out.append(record(f"card_batch_{i:02d}", "card", True, ticket, {"team": copy.deepcopy(CARD_QUESTIONS["team"])},
                          note=f"the card's system_one_batch example, ticket {i}", provenance=prov))
    path, why = fetch(COCO, fetch_media)
    media = {"images": [COCO["path"]]} if path else None
    out.append(record("card_cats", "card", False, None, {"cats": copy.deepcopy(CARD_CATS)}, media=media,
                      note="the card's image example: two cats on a sofa (measured only)" + (f"; {why}" if why else ""),
                      provenance={**prov, "image": {k: COCO[k] for k in ("url", "sha256", "bytes", "license")},
                                  **({"image_info": image_info(path)} if path else {})}))
    path, why = fetch(FLAC, fetch_media)
    media = {"audio": FLAC["path"]} if path else None
    out.append(record("card_audio", "card", False, CARD_AUDIO_STATE, {"topic": copy.deepcopy(CARD_TOPIC)}, media=media,
                      note="the card's audio example (measured only)" + (f"; {why}" if why else ""),
                      provenance={**prov, "audio": {k: FLAC[k] for k in ("url", "sha256", "bytes", "license")},
                                  **({"audio_info": audio_info(path)} if path else {})}))
    return out


def kev_records() -> dict[str, dict]:
    doc = json.loads(_read_pinned(KEV["path"], KEV["sha256"], "Kev records"))
    records = {r["id"]: r for r in doc["records"]}
    if len(records) != KEV["records"]:
        raise SystemExit(f"Kev records: {len(records)} != {KEV['records']}")
    return records


def transfer_v4_lines() -> list[str]:
    raw = _read_pinned(TRANSFER_V4["path"], TRANSFER_V4["sha256"], "transfer-v4 development.jsonl")
    lines = raw.decode().splitlines()
    if len(lines) != TRANSFER_V4["lines"]:
        raise SystemExit("transfer-v4 line count changed")
    return lines


def tv4_family(kev: dict, lines: list[str], prefix: str, public: bool) -> list[dict]:
    out = []
    for rid, r in kev.items():
        if not rid.startswith(prefix + "_"):  # "tv4_" / "tv4x_" / "tv4s_" do not overlap
            continue
        request = strip_model(r["request"])
        if list(request) != ["state", "questions"]:
            raise SystemExit(f"{rid}: request keys {list(r['request'])}")
        prov = copy.deepcopy(r["provenance"])
        line = lines[prov["line"]]
        if json.loads(line)["_meta"] != prov["_meta"]:
            raise SystemExit(f"{rid}: provenance line {prov['line']} is not this record's line")
        prov["line_sha256"] = _sha256(line.encode())
        out.append(record(rid, r["source"], public, request["state"], request["questions"],
                          gold=copy.deepcopy(r["gold"]), note=r["note"], provenance=prov))
    return out


def semif_records(kev: dict) -> list[dict]:
    raw = _read_pinned(SEMIF["path"], SEMIF["sha256"], "SemIf authored144")
    out = []
    for n, line in enumerate(raw.decode().splitlines()):
        r = json.loads(line)
        criteria = {o["id"]: o["description"] for o in r["options"]}
        if len(criteria) != len(r["options"]):
            raise SystemExit(f"semif {r['id']}: duplicate option ids")
        question = {"type": "choice", "instructions": r["question"], "criteria": criteria}
        rid = f"semif_{r['id']}"
        rec = record(rid, "semif_authored144", True, r["state"], {SEMIF["question_id"]: question},
                     gold={SEMIF["question_id"]: r["options"][r["label"]]["id"]},
                     note=f"family={r['family']} variant={r['provenance']['variant']} group={r['group_id']}",
                     provenance={"file": SEMIF["upstream_path"], "repo": SEMIF["upstream_repo"],
                                 "commit": SEMIF["upstream_commit"], "file_sha256": SEMIF["sha256"], "license": "MIT",
                                 "line": n, "line_sha256": _sha256(line.encode()), "semif_id": r["id"],
                                 "family": r["family"], "variant": r["provenance"]["variant"],
                                 "group_id": r["group_id"], "label": r["label"],
                                 "option_order": [o["id"] for o in r["options"]]})
        mine = json.dumps(rec["request"], ensure_ascii=False)
        theirs = json.dumps(strip_model(kev[rid]["request"]), ensure_ascii=False)
        if mine != theirs or rec["gold"] != kev[rid]["gold"]:
            raise SystemExit(f"{rid}: the request / gold differs from the Kev record of the same id")
        out.append(rec)
    if len(out) != SEMIF["rows"]:
        raise SystemExit(f"SemIf rows {len(out)} != {SEMIF['rows']}")
    return out


def d1_questions(rid: str, questions: dict, dropped: list) -> dict:
    """The questions the publisher's as_question accepts; the others are listed in `dropped`."""
    keep = {}
    for qid, q in questions.items():
        try:
            as_question(q)
        except ValueError as error:
            dropped.append({"id": rid, "qid": qid, "reason": str(error),
                            "question_keys": list(q), "type": q.get("type")})
            continue
        keep[qid] = q
    return keep


def own_records(dropped: list) -> list[dict]:
    public = json.loads(_read_pinned(OWN["public_path"], OWN["public_sha256"], "own_records_public.json"))["records"]
    withheld = {r["id"]: r for r in json.loads(_read_pinned(OWN["withheld_path"], OWN["withheld_sha256"],
                                                            "own_records_withheld.json"))["records"]}
    out = []
    for stub in public:
        if stub.get("withheld"):
            full = withheld[stub["id"]]
            got = _sha256(json.dumps(full["request"], sort_keys=True, ensure_ascii=False).encode())
            if got != stub["request_sha256"]:
                raise SystemExit(f"{stub['id']}: the withheld request does not match its public stub")
            r, is_public, how = full, False, "withheld by the Kev port's name screen (round 9): measured only"
        else:
            r, is_public, how = stub, True, "published in conversion/kev/own_records_public.json"
        request = strip_model(r["request"])
        questions = d1_questions(r["id"], request["questions"], dropped)
        gold = {qid: g for qid, g in r["gold"].items() if qid in questions}
        out.append(record(r["id"], r["source"], is_public, request["state"], questions, gold=gold, note=r["note"],
                          provenance={"from": "the Kev port's own records", "status": how,
                                      "request_sha256": _sha256(json.dumps(r["request"], sort_keys=True,
                                                                           ensure_ascii=False).encode())}))
    if len(out) != 20:
        raise SystemExit(f"own records: {len(out)} != 20")
    return out


def long_log_lines(until=LONG_UNTIL) -> list[str]:
    """own_L01's sorter log continued after 07:33:05 with the same kinds of lines, the same names and the same
    equipment; the lane 7 belt tension check the shift note announces for 10:00 finds it within specification;
    no new stoppage, and the log ends normally."""
    rates = [1708, 1716, 1699, 1722, 1704, 1711, 1693, 1719, 1702, 1714, 1697, 1725, 1706, 1690, 1718, 1701]
    lines, i = [], 0
    hour, minute = 7, 36
    parcel = 50113
    while (hour, minute) <= until:
        stamp = f"{hour:02d}:{minute:02d}"
        lines.append(f"{stamp}:00 INFO  throughput {rates[i % len(rates)]} parcels/h; lanes 1-12 active; "
                     f"scanner read rate {99.0 + (i % 8) * 0.1:.1f}%")
        if i % 6 == 2:
            lines.append(f"{stamp}:41 INFO  diverter D4 calibration check passed (offset {0.31 + (i % 5) * 0.06:.2f} mm)")
        if i % 7 == 4:
            lines.append(f"{stamp}:17 WARN  scanner T{1 + i % 2} no-read on parcel {parcel}; parcel sent to manual lane")
            parcel += 397
        if i % 10 == 6:
            lines.append(f"{stamp}:30 INFO  lane 7 drive motor current {5.8 + (i % 3) * 0.1:.1f} A (limit 8.0 A); "
                         f"belt speed 1.80 m/s")
        if i % 12 == 9:
            lines.append(f"{stamp}:50 INFO  accumulation belts at {4 + i % 5}% capacity")
        if (hour, minute) == (10, 0):
            lines.append("10:00:40 INFO  lane 7 belt tension checked by Malik (as planned in the 07:33 shift note): "
                         "within specification; no adjustment needed")
        i += 1
        minute += LONG_STEP
        if minute >= 60:
            hour, minute = hour + 1, minute - 60
    if not any(line.startswith("10:00:40") for line in lines):
        raise SystemExit("long_3400: the log must reach the 10:00 check")
    last = until[0] * 60 + until[1] + 1
    lines.append(f"{last // 60:02d}:{last % 60:02d}:05 INFO  sorter S1 running normally; no active alarms")
    return lines


def long_record(own: list[dict], tok: RawTokenizer) -> tuple[dict, dict]:
    base = next(r for r in own if r["id"] == "own_L01")
    if not base["public"]:
        raise SystemExit("own_L01 is expected to be public")
    state = base["request"]["state"].rstrip("\n") + "\n" + "\n".join(long_log_lines())
    if "Brindlecourt" in state or "http" in state or "@" in state:
        raise SystemExit("long_3400: unexpected spelling")
    tokens = len(tok(escape(serialize(state)), add_special_tokens=False)["input_ids"])
    lo, hi = LONG_TOKENS
    if not lo <= tokens <= hi:
        raise SystemExit(f"long_3400: state is {tokens} tokens, outside {LONG_TOKENS} (move LONG_UNTIL)")
    questions = {**copy.deepcopy(base["request"]["questions"]), **copy.deepcopy(LONG_QUESTION)}
    gold = {**base["gold"], **LONG_GOLD}
    rec = record("long_3400", "own_long_extended", True, state, questions, gold=gold,
                 note="own_L01's sorter log continued to ~3.4k state tokens (same names, places and equipment); "
                      "own_L01's questions keep their answers (no new stoppage, the log ends normally); "
                      "tension_10 is answered in the added lines",
                 provenance={"from": "own_L01", "own_L01_state_sha256": _sha256(base["request"]["state"].encode()),
                             "added_lines": len(long_log_lines()), "until": f"{LONG_UNTIL[0]:02d}:{LONG_UNTIL[1]:02d}",
                             "state_tokens": tokens})
    return rec, {"state_tokens": tokens, "range": list(LONG_TOKENS)}


def long_1500_check(own: list[dict], tok: RawTokenizer) -> dict:
    counts = {r["id"]: len(tok(escape(serialize(r["request"]["state"])), add_special_tokens=False)["input_ids"])
              for r in own if r["source"] == "own_long"}
    lo, hi = LONG_1500_TOKENS
    hits = [rid for rid, n in counts.items() if lo <= n <= hi]
    public_hits = [rid for rid in hits if next(r for r in own if r["id"] == rid)["public"]]
    return {"state_tokens": counts, "range": list(LONG_1500_TOKENS), "in_range": hits, "public_in_range": public_hits,
            "long_1500": "not written: " + ", ".join(public_hits or hits) + " is in range" if hits else "needed"}


def _pool_entries() -> dict[str, dict]:
    """{pool: {file stem: the pool manifest's entry}} for the two picture pools (manifests sha256-pinned)."""
    out = {}
    for pool, spec in IMAGE_POOLS.items():
        doc = json.loads(_read_pinned(spec["manifest"], spec["manifest_sha256"], f"{pool} picture manifest"))
        if pool == "cc0":
            out[pool] = {Path(e["file"]).stem: {**e, "_pool": {k: doc[k] for k in ("source", "license_rule", "people")}}
                         for e in doc["images"]}
        else:
            top = {k: doc[k] for k in ("model", "hub_revision", "model_license", "runner_source", "settings",
                                       "output_license", "selection")}
            out[pool] = {e["name"]: {**e, "_pool": top} for e in doc["items"]}
    return out


def _source_record(pool: str, entry: dict) -> dict:
    """What the image manifest keeps of a pool entry (the pool's own fields, unchanged)."""
    if pool == "cc0":
        keep = ("commons_file", "commons_page", "original_url", "license_templates", "license_short_name", "artist",
                "retrieved_at", "original_resolution", "original_bytes", "original_sha256", "exif_orientation",
                "upright_resolution", "icc_profile", "crop_bottom_px_of_upright", "resize", "saved_resolution")
        if not any("cc-zero" in t for t in entry["license_templates"]) or entry.get("excluded_at_review"):
            raise SystemExit(f"{entry['file']}: not a usable CC0 entry {entry['license_templates']}")
        return {k: entry[k] for k in keep} | {"pool": entry["_pool"]}
    keep = ("prompt", "seed", "intent", "seconds", "sha256")
    return {k: entry[k] for k in keep} | {"pool": entry["_pool"]}


def make_images() -> dict:
    """Copy or downscale each IMAGES source into images/<id>.png (the pools are only read) and write
    images/manifest.json. An existing file is kept when its pixels equal what this would write, else refused."""
    import io

    import PIL
    from PIL import Image

    entries = _pool_entries()
    out_dir = WORK / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for spec in IMAGES:
        pool = IMAGE_POOLS[spec["pool"]]
        source = pool["dir"] / f"{spec['name']}.png"
        data = _read_pinned(source, spec["source_sha256"], f"{spec['id']} source")
        entry = entries[spec["pool"]].get(spec["name"])
        if entry is None:
            raise SystemExit(f"{spec['name']} is not in the {spec['pool']} manifest")
        if spec["pool"] == "flux" and entry["sha256"] != spec["source_sha256"]:
            raise SystemExit(f"{spec['name']}: rooms manifest sha256 {entry['sha256']} != {spec['source_sha256']}")
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            icc = im.info.get("icc_profile")
            if im.mode != "RGB":
                raise SystemExit(f"{source}: mode {im.mode}, expected RGB")
            if spec["size"] is None:
                written, pixels, op = data, np.asarray(im), {"op": "copy (the source file, byte for byte)"}
            else:
                small = im.resize(tuple(spec["size"]), resample=Image.BICUBIC)
                buffer = io.BytesIO()
                small.save(buffer, format="PNG", icc_profile=icc)
                written, pixels = buffer.getvalue(), np.asarray(small)
                op = {"op": f"resize {im.size[0]}x{im.size[1]} -> {spec['size'][0]}x{spec['size'][1]}",
                      **IMAGE_RESIZE, "pillow": PIL.__version__}
        target = out_dir / f"{spec['id']}.png"
        if target.exists():
            kept = target.read_bytes()
            if kept != written:
                with Image.open(target) as old:
                    if not np.array_equal(np.asarray(old.convert("RGB")), pixels):
                        raise SystemExit(f"{target} exists with other pixels: not replaced")
                written = kept  # same pixels, another encoder's bytes: keep the file that is there
        else:
            target.write_bytes(written)
        with Image.open(target) as im:
            width, height = im.size
        items.append({
            "id": spec["id"], "file": f"images/{spec['id']}.png", "public": spec["public"], "bytes": len(written),
            "sha256": _sha256(written), "px": [width, height], "mode": "RGB",
            "icc_profile": None if icc is None else f"embedded ({len(icc)} B, kept from the source)",
            "made": {"from": str(source), "source_sha256": spec["source_sha256"], **op},
            "license": pool["license"], "pool": spec["pool"], "source": _source_record(spec["pool"], entry),
            "layout": layout(width, height), "crops_hw": [list(c) for c in image_crops(width, height)],
            "prefix_positions": image_prefix_length([(width, height)]),
            "shows": spec["shows"], "checked": CHECKED})
    manifest = {
        "schema": "d1-omni-images/1", "written_by": "conversion/d1_omni/make_fixtures.py (make_images)",
        "rule": "pictures only from the two sieved pools below, read only; nothing generated or downloaded here; "
                "no person, no legible text or logo, no existing design (each file looked at by eye at full size)",
        "pools": {k: {"dir": str(v["dir"]), "manifest": str(v["manifest"]), "manifest_sha256": v["manifest_sha256"],
                      "license": v["license"]} for k, v in IMAGE_POOLS.items()},
        "images": items,
        "summary": {"public": sum(i["public"] for i in items), "measured_only": sum(not i["public"] for i in items),
                    "by_pool": dict(Counter(i["pool"] for i in items)),
                    "tiled": sum(i["layout"]["tiled"] for i in items),
                    "prefix_positions": {i["id"]: i["prefix_positions"] for i in items}},
    }
    data = (json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode()
    path = out_dir / "manifest.json"
    if not path.exists() or path.read_bytes() != data:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return {"path": "images/manifest.json", "sha256": _sha256(data), "items": {i["id"]: i for i in items}}


def image_records(made: dict) -> list[dict]:
    out = []
    for spec in IMAGES:
        item = made["items"][spec["id"]]
        path = WORK / item["file"]
        out.append(record(
            spec["id"], "own_image" if spec["public"] else "measure_image", spec["public"], spec["state"],
            copy.deepcopy(spec["questions"]), media={"images": [item["file"]]}, gold=dict(spec["gold"]),
            note=f"{spec['shows']} ({'CC0 photograph' if spec['pool'] == 'cc0' else 'own FLUX.2 klein output'}; "
                 f"gold = the answer seen in the picture)",
            provenance={"image": {"file": item["file"], "sha256": item["sha256"], "bytes": item["bytes"],
                                  "px": item["px"], "license": item["license"], "pool": item["pool"],
                                  "from": Path(item["made"]["from"]).name, "op": item["made"]["op"],
                                  "manifest": made["path"]},
                        "image_info": image_info(path)}))
    return out


def previous_version_check(records: list[dict], out_dir: Path) -> dict:
    """Every record of the previous version but the rewritten ones is here unchanged, in the same order."""
    path = out_dir / PREVIOUS["file"]
    previous = json.loads(_read_pinned(path, PREVIOUS["sha256"], "previous fixtures"))["records"]
    now = {r["id"]: r for r in records}
    kept = [r["id"] for r in previous if r["id"] not in PREVIOUS["rewritten"]]
    changed = [rid for rid in kept if json.dumps(now.get(rid), ensure_ascii=False) != json.dumps(
        next(r for r in previous if r["id"] == rid), ensure_ascii=False)]
    if changed:
        raise SystemExit(f"records changed from {PREVIOUS['file']}: {changed[:5]}")
    order = [r["id"] for r in records if r["id"] in {p["id"] for p in previous}]
    if order != [r["id"] for r in previous]:
        raise SystemExit("the previous records are not in their previous order")
    added = [r["id"] for r in records if r["id"] not in {p["id"] for p in previous}]
    rewritten = {}
    for rid in PREVIOUS["rewritten"]:
        old, new = next(r for r in previous if r["id"] == rid), now[rid]
        rewritten[rid] = {"fields_changed": [k for k in new if new[k] != old[k]],
                          "questions_before": list(old["request"]["questions"]),
                          "questions_after": list(new["request"]["questions"]),
                          "media_before": old["media"], "file_present_before": old["provenance"].get("file_present"),
                          "media_after": new["media"]}
    return {"previous": {"file": PREVIOUS["file"], "sha256": PREVIOUS["sha256"], "records": len(previous)},
            "unchanged_records": len(kept), "unchanged_check": "json.dumps equal, same order", "added": added,
            "rewritten": rewritten}


def audio_records() -> tuple[list[dict], dict]:
    manifest_path = WORK / "audio" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"clips": {}}
    out = []
    for spec in AUDIO + AUDIO_R6:
        r6 = spec in AUDIO_R6
        clip = manifest["clips"].get(spec["id"])
        wav = WORK / "audio" / f"{spec['id']}.wav"
        made = clip is not None and wav.exists() and _sha256(wav.read_bytes()) == clip["sha256"]
        if made and r6 and (clip["text"], clip["voice"], clip["speed"]) != (spec["text"], spec["voice"], spec["speed"]):
            raise SystemExit(f"{spec['id']}: the manifest's clip was made from another script, voice or speed")
        note = ("synthesised: Kokoro-82M voice " + spec["voice"]) if made else \
               ("not synthesised yet (audio/script.json holds the script; see manifest.json status)")
        if r6 and made:
            note += f", speed {spec['speed']}, {clip['seconds']} s" + (", speech only (trim)" if spec.get("trim") else "")
        provenance = {"script": spec["text"], "voice": spec["voice"], "synth": SYNTH_R6 if r6 else SYNTH}
        if r6:
            provenance |= {"speed": spec["speed"], "target_seconds": spec["target_seconds"],
                           "trim": bool(spec.get("trim"))}
        provenance |= {"file_present": made, **({"clip": clip} if made else {})}
        out.append(record(spec["id"], "own_audio", True, spec["state"], copy.deepcopy(AUDIO_QUESTIONS),
                          media={"audio": f"audio/{spec['id']}.wav"}, gold=dict(spec["gold"]), note=note,
                          provenance=provenance))
    return out, manifest


def write_script_json() -> Path:
    """audio/script.json: what the Kokoro runs synthesise (round 1: K/scripts/synth_aud.py, aud_01..03; round 6:
    K/scripts/synth_aud_r6.py, aud_04..15, appended to manifest.json; each run writes the 16 kHz wav files)."""
    path = WORK / "audio" / "script.json"
    doc = {"note": "Kokoro-82M scripts for aud_01..aud_15 (make_fixtures.py AUDIO, AUDIO_R6); no names",
           "synth": SYNTH,
           "clips": [{k: spec[k] for k in ("id", "voice", "text")} for spec in AUDIO],
           "synth_r6": SYNTH_R6,
           "clips_r6": [{k: spec[k] for k in ("id", "voice", "speed", "target_seconds", "text")}
                        | {"trim": bool(spec.get("trim"))} for spec in AUDIO_R6]}
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def red_arm(tv4: list[dict]) -> dict:
    base = next(r for r in tv4 if r["id"] == RED_ARM["from"])
    request = copy.deepcopy(base["request"])
    (qid, q), = request["questions"].items()
    if q["instructions"].count(RED_ARM["old"]) != 1:
        raise SystemExit(f"red arm: {RED_ARM['old']!r} is not in {RED_ARM['from']}'s instructions exactly once")
    q["instructions"] = q["instructions"].replace(RED_ARM["old"], RED_ARM["new"])
    if RED_ARM["new"] not in q["instructions"]:
        raise SystemExit("red arm: the replacement did not happen")
    return record("red_arm_000", "red_arm", False, request["state"], request["questions"], gold=copy.deepcopy(base["gold"]),
                  note=f"{RED_ARM['from']} with '{RED_ARM['old']}' -> '{RED_ARM['new']}' in the instructions; compared "
                       f"against {RED_ARM['from']}'s reference it must fail the gate (not a fixture)",
                  provenance={**copy.deepcopy(base["provenance"]), "arm": RED_ARM})


# ---------------------------------------------------------------------------------------------- checks + summary
def gold_keys(q: dict) -> list[str]:
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def check(records: list[dict]) -> None:
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate record id")
    for r in records:
        if list(r) != ["id", "source", "public", "request", "media", "gold", "note", "provenance"]:
            raise SystemExit(f"{r['id']}: record keys {list(r)}")
        if list(r["request"]) != ["state", "questions"] or not r["request"]["questions"]:
            raise SystemExit(f"{r['id']}: request keys {list(r['request'])}")
        for qid, q in r["request"]["questions"].items():
            as_question(q)
            if set(q) - {"type", "instructions", "criteria"}:
                raise SystemExit(f"{r['id']}.{qid}: question keys {list(q)}")
            if r["gold"] is not None and qid in r["gold"] and r["gold"][qid] not in gold_keys(q):
                raise SystemExit(f"{r['id']}.{qid}: gold {r['gold'][qid]!r} not in {gold_keys(q)}")
        if r["gold"] is not None and set(r["gold"]) - set(r["request"]["questions"]):
            raise SystemExit(f"{r['id']}: gold for a question that is not in the request")
        if r["media"] is not None and set(r["media"]) not in ({"images"}, {"audio"}):
            raise SystemExit(f"{r['id']}: media {r['media']}")
        if r["source"].startswith(("own_", "card")) and r["public"]:
            text = json.dumps(r["request"], ensure_ascii=False)
            for bad in ("http", "www.", "@", ".com", "Jev"):
                if bad in text:
                    raise SystemExit(f"{r['id']}: {bad!r} in a public written record")


def summarize(records: list[dict], dropped: list, extra: dict) -> dict:
    by = {}
    for r in records:
        s = by.setdefault(r["source"], {"records": 0, "public": 0, "questions": 0, "types": Counter(),
                                        "options": Counter(), "media": Counter()})
        s["records"] += 1
        s["public"] += bool(r["public"])
        media = "none" if r["media"] is None else ("images" if "images" in r["media"] else "audio")
        present = "" if r["media"] is None else (":present" if (WORK / (r["media"].get("audio") or
                                                                           r["media"]["images"][0])).exists() else ":planned")
        s["media"][media + present] += 1
        for q in r["request"]["questions"].values():
            s["questions"] += 1
            s["types"][q["type"]] += 1
            s["options"][len(gold_keys(q))] += 1
    for s in by.values():
        for k in ("types", "options", "media"):
            s[k] = {str(key): v for key, v in sorted(s[k].items(), key=lambda kv: (len(str(kv[0])), str(kv[0])))}
    public = [r for r in records if r["public"]]
    return {
        "records": len(records), "questions": sum(len(r["request"]["questions"]) for r in records),
        "public_records": len(public), "public_questions": sum(len(r["request"]["questions"]) for r in public),
        "by_source": by,
        "types": dict(Counter(q["type"] for r in records for q in r["request"]["questions"].values())),
        "options": {str(k): v for k, v in sorted(Counter(len(gold_keys(q)) for r in records
                                                         for q in r["request"]["questions"].values()).items())},
        "state_types": dict(Counter(type(r["request"]["state"]).__name__ for r in records)),
        "dropped": dropped, **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="download the card's image and flac if missing")
    ap.add_argument("--out", default=None, help="default $ZOO_WORK_ROOT/_d1_omni/fixtures/records.json")
    args = ap.parse_args()

    snapshot = Path(hf_snapshot(MODEL_ID, revision=MODEL_SHA))
    readme = _read_pinned(snapshot / CARD["file"], CARD["sha256"], "model card").decode()
    tok = RawTokenizer(snapshot / "tokenizer.json")
    write_script_json()

    card = card_records(readme, args.fetch)
    kev = kev_records()
    lines = transfer_v4_lines()
    tv4 = tv4_family(kev, lines, "tv4", True)
    tv4x = tv4_family(kev, lines, "tv4x", False)
    tv4s = tv4_family(kev, lines, "tv4s", False)
    if (len(tv4), len(tv4x), len(tv4s)) != (60, 140, 20):
        raise SystemExit(f"transfer-v4 slices {len(tv4)}/{len(tv4x)}/{len(tv4s)} != 60/140/20")
    semif = semif_records(kev)
    dropped: list = []
    own = own_records(dropped)
    long, long_stats = long_record(own, tok)
    long_1500 = long_1500_check(own, tok)
    made = make_images()
    images = image_records(made)
    audio, manifest = audio_records()
    records = card + tv4 + semif + own + tv4x + tv4s + [long] + images + audio + [red_arm(tv4)]
    check(records)

    license_bytes = Path(SEMIF["license_path"]).read_bytes()
    out_dir = Path(args.out).parent if args.out else WORK / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    version = previous_version_check(records, WORK / "fixtures")
    (out_dir / "LICENSE-SemIf").write_bytes(license_bytes)
    license_sha = _sha256(license_bytes)

    doc = {
        "schema": SCHEMA,
        "model": {"hf_id": MODEL_ID, "revision": MODEL_SHA},
        "record_form": ("request = the publisher's system_one(state, questions) arguments; media = the images or the "
                        "audio clip (paths relative to the work dir; the files are not in this document); gold = the "
                        "key a question's probabilities are reported under (choice: criteria name, noul: true/false, "
                        "score: level index); public = the record's text may be published"),
        "records": records,
        "summary": summarize(records, dropped, {"long_3400": long_stats, "long_1500": long_1500,
                                                "audio_manifest": manifest.get("status", "absent"),
                                                "images_manifest": {k: made[k] for k in ("path", "sha256")},
                                                "version": version}),
        "sources": {
            "card": {**CARD, "repo": MODEL_ID, "revision": MODEL_SHA, "image": COCO, "audio": FLAC},
            "kev_records": {k: KEV[k] for k in ("sha256", "records")} | {
                "path": "$ZOO_WORK_ROOT/_kev/fixtures/records.json",
                "note": "the Core AI Kev port's fixture (zoo conversion/kev/make_fixtures.py); requests copied less `model`"},
            "transfer_v4": {k: TRANSFER_V4[k] for k in ("path_in_repo", "sha256", "lines", "author_repo")} | {
                "source_licenses": TV4_SOURCE_LICENSES,
                "publication": "tv4 (MMLU, MIT) public; tv4x / tv4s measured only (do not publish their text)"},
            "semif_authored144": {k: SEMIF[k] for k in ("sha256", "upstream_repo", "upstream_path", "upstream_commit",
                                                        "license", "copyright_notice", "rows", "question_id")} | {
                "license_file": "LICENSE-SemIf", "license_sha256": license_sha,
                "license_sha256_expected": SEMIF["license_sha256"]},
            "own": {k: OWN[k] for k in ("public_sha256", "withheld_sha256")} | {
                "public_file": "conversion/kev/own_records_public.json",
                "withheld_file": "$ZOO_WORK_ROOT/_kev/fixtures/own_records_withheld.json",
                "license": "written for the Kev port (same licence as the conversion code)"},
            "long_3400": {"from": "own_L01", "written": "here (long_log_lines)"},
            "images": {"status": "made (round 5)", "manifest": made["path"], "manifest_sha256": made["sha256"],
                       "rule": "CC0 photographs or own FLUX.2 klein output from two sieved pools, read only; no "
                               "person, no legible text or logo, no existing design (looked at by eye)",
                       "pools": {k: {"manifest": str(v["manifest"]), "manifest_sha256": v["manifest_sha256"],
                                     "license": v["license"]} for k, v in IMAGE_POOLS.items()}},
            "audio": {"synth": SYNTH, "synth_r6": SYNTH_R6, "script_file": "audio/script.json",
                      "manifest": "audio/manifest.json",
                      "rule": "own Kokoro-82M output from scripts written here (English, no names, no real person, "
                              "company or product); the card's flac (LibriSpeech, CC BY 4.0) stays measured only"},
        },
    }
    path = Path(args.out) if args.out else out_dir / "records.json"
    data = (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    s = doc["summary"]
    print(f"{path}: {s['records']} records / {s['questions']} questions (public {s['public_records']} / "
          f"{s['public_questions']}), sha256 {_sha256(data)}")
    for source, v in s["by_source"].items():
        print(f"  {source}: {v['records']} records (public {v['public']}), {v['questions']} questions {v['types']} "
              f"options {v['options']} media {v['media']}")
    print(f"  dropped {len(dropped)}: {[(d['id'], d['qid']) for d in dropped]}")
    print(f"  long_3400 {long_stats}; long_1500 {long_1500}")
    print(f"  LICENSE-SemIf sha256 {license_sha} (expected {SEMIF['license_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
