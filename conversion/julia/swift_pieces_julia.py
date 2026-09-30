#!/usr/bin/env python3
"""Write the input of the Swift host's parity run (swift/parity/main.swift) from the fixture rows.

    python3 conversion/julia/swift_pieces_julia.py <out.json>

For every strict fixture row (models/julia-1/fixtures-julia-1.json: parity requests and typed questions), the
request with its state as text (json.dumps(state, ensure_ascii=False), as the publisher's builder writes it),
the publisher's ids, markers and raw logits, and — once per distinct text piece the builder encodes (the head
`{type} question: {question}`, each " " + option, the state) — the ids the checkpoint's tokenizer.json gives
it through the `tokenizers` library. The Swift run encodes through this table, so it checks everything after
the tokenizer: the row assembly, the graph call and the readout.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import source_dir  # noqa: E402
from _julia_host import MASK_TEXT, Tokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import repo_root  # noqa: E402


def main():
    out = Path(sys.argv[1])
    tokenizer = Tokenizer(source_dir() / "tokenizer" / "tokenizer.json")
    fixture = json.loads((repo_root() / "models" / "julia-1" / "fixtures-julia-1.json").read_text())
    pieces, rows = {}, []
    for row in fixture["rows"]:
        if not row["builder"]["strict"]:
            continue
        request = row["request"]
        state = request["state"] if isinstance(request["state"], str) else json.dumps(request["state"], ensure_ascii=False)
        clean = lambda text: text.replace(MASK_TEXT, " ")  # noqa: E731
        for text in [f"{request.get('type', 'choice')} question: {clean(request['question'])}",
                     *[" " + clean(o) for o in request["options"]], clean(state)]:
            if text and text not in pieces:
                pieces[text] = tokenizer(text)
        rows.append({"row_id": row["row_id"], "type": request.get("type", "choice"), "question": request["question"],
                     "options": request["options"], "state": state, "ids": row["ids"], "markers": row["markers"],
                     "publisher_logits": row["publisher_logits"]})
    out.write_text(json.dumps({"pieces": [{"text": t, "ids": ids} for t, ids in pieces.items()], "rows": rows},
                              ensure_ascii=False))
    print(out, len(rows), "rows,", len(pieces), "pieces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
