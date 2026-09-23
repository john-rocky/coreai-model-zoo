#!/usr/bin/env python3
"""Write models/laya-multilingual/fixtures-laya-multilingual.json from the frozen fixtures.

    python3 conversion/laya/fixtures_laya.py            # (re)write it
    python3 conversion/laya/fixtures_laya.py --check    # exit 1 if the file differs from the frozen sources

Self-contained, schema `coreai-encoder-fixtures/1` (what coreai-kit's `decide-cli parity` reads): the
44 states of ml_fixtures.json and the 402 frozen question rows of both windows, each with its token ids,
marker positions and the official answer at T = 1. No model runs here; oracle_laya.py proves the rows
are what the publisher's builder and model produce (ids 402/402, answers and logits bit-exact).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HEAD_MAX_LEN, MODEL_ID, MODEL_SHA, SUBFOLDER, WINDOWS, load_fixture_states, load_rows  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

ROW_FIELDS = ("row_id", "fixture_id", "window", "question_id", "question", "sequence_ids", "marker_positions", "qtype",
              "K", "sequence_length", "raw_logits", "raw_act_logits", "probabilities", "act_probability", "official_answer")


def build() -> str:
    states = load_fixture_states()
    rows = [{k: row[k] for k in ROW_FIELDS} for window in WINDOWS for row in load_rows(window)]
    document = {"schema": "coreai-encoder-fixtures/1", "model": MODEL_ID, "subfolder": SUBFOLDER, "revision": MODEL_SHA,
                "head_max_len": HEAD_MAX_LEN, "temperature": [1, 1, 1],
                "fixtures": [{"id": fixture_id, "state": fixture["state"]} for fixture_id, fixture in states.items()],
                "rows": rows}
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = repo_root() / "models" / "laya-multilingual" / "fixtures-laya-multilingual.json"
    text = build()
    if args.check:
        same = out.exists() and out.read_text() == text
        print("up to date" if same else f"{out} differs from the frozen fixtures")
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out, len(text.encode()), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
