#!/usr/bin/env python3
"""Write models/julia-1/fixtures-julia-1.json from the oracle's rows.

    python3 conversion/julia/fixtures_julia.py            # (re)write it
    python3 conversion/julia/fixtures_julia.py --check    # exit 1 if the file differs from the oracle's rows

Self-contained: the publisher's 100 Julia-1-ONNX parity requests, every 20th typed-decisions test question
(100 of 2,000) and the first 5 window-filling rows of each window, each with its request (state, question,
options, type), the token ids and marker positions the publisher's builder gives, and the publisher's raw
logits (one question per forward, CPU fp32). A host port is checked against these rows: ids and markers
exactly, then the logits through its bundle (argmax, max |dp| at T = 1). No model runs here;
oracle_julia.py proves the rows are what the publisher's builder and model produce.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HEAD_LENGTH, MODEL_ID, MODEL_SHA, oracle_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402

FIELDS = ("row_id", "set", "type", "keys", "gold", "request", "ids", "markers", "publisher_logits")


def build() -> str:
    rows = json.loads((oracle_dir() / "rows.json").read_text())["rows"]
    typed = [r for r in rows if r["set"] == "typed"][::20]
    parity = [r for r in rows if r["set"] == "parity"]
    fills = [r for w in (512, 1024) for r in [x for x in rows if x["set"] == "fill" and x["fill_window"] == w][:5]]
    out = []
    for row in parity + typed + fills:
        entry = {k: row.get(k) for k in FIELDS}
        entry["builder"] = ({"max_length": row["fill_window"], "head_length": HEAD_LENGTH[row["fill_window"]], "strict": False}
                            if row["set"] == "fill" else
                            {"max_length": 1024, "head_length": 256 if row["set"] == "parity" else 512, "strict": True})
        out.append(entry)
    document = {"schema": "julia-fixtures/1", "model": MODEL_ID, "revision": MODEL_SHA, "temperature": 1,
                "reference": "the publisher's julia package: FastEngine.logits, one question per forward, CPU fp32",
                "rows": out}
    return json.dumps(document, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = repo_root() / "models" / "julia-1" / "fixtures-julia-1.json"
    text = build()
    if args.check:
        same = out.exists() and out.read_text() == text
        print("up to date" if same else f"{out} differs from the oracle's rows")
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out, len(text.encode()), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
