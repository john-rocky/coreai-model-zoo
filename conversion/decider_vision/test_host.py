#!/usr/bin/env python3
"""Test: `host.build_ids` reproduces the round-1 oracle rows — ids, slots, M-RoPE — for all 111 runs.

For every (row, arm) run in `fixture_oracle.json` (35 image rows x g256 / g448 / native + 6 text
rows), rebuilt from `fixtures/rows.json` with each tokenizer under test:

  * ids   — V+k mapped back to <|image_pad|> equals the processor's `input_ids`, token for token;
  * slots — `find_slots` on the rebuilt row equals the oracle's `slot_idx`;
  * rope  — `rope_positions(ids, start, amount, W)` equals the three planes the text rotary
            actually received (the oracle's forward-hook capture, `npz/<row>__<arm>.npz rope_pos`),
            and amount equals the oracle's `rope_shift`;
  * the tokenizers agree with each other (the oracle ran transformers 5.x; the host may not).

The native arm takes H, W from the oracle's own `image_grid_thw` (the processor picked it).

Run from the worktree root (shared venv, offline):
    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 \\
        ../coreai-models/.venv/bin/python conversion/decider_vision/test_host.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "Mapika/decider-2b-vision"
REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
LANE = work_path("_decider2bv")


def tokenizers_under_test(snapshot: Path) -> dict:
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    return {
        f"tokenizers {tokenizers.__version__}": tokenizers.Tokenizer.from_file(str(snapshot / "tokenizer.json")),
        f"transformers {transformers.__version__} AutoTokenizer": AutoTokenizer.from_pretrained(str(snapshot)),
    }


def check_run(run: dict, fx_row: dict, tok) -> tuple[list[str], list[int]]:
    """Mismatch messages for one oracle run (empty = pass) and the rebuilt ids."""
    arm = run["arm"]
    has_image = arm != "text"
    hw = None
    if has_image:
        t, gh, gw = run["grid_thw"]
        assert t == 1, run["grid_thw"]
        hw = (gh // host.MERGE, gw // host.MERGE)
    ids, slots, start, amount = host.build_ids(has_image, fx_row["context"], fx_row["questions"], tok, hw)
    bad = []
    proc = host.to_processor_ids(ids)
    if proc != run["ids"]:
        i = next((k for k, (a, b) in enumerate(zip(proc, run["ids"])) if a != b), min(len(proc), len(run["ids"])))
        bad.append(f"ids differ at {i} (len {len(proc)} vs {len(run['ids'])})")
    if slots != run["slot_idx"]:
        bad.append(f"slots {slots} vs {run['slot_idx']}")
    if has_image and amount != run["rope_shift"]:
        bad.append(f"amount {amount} vs rope_shift {run['rope_shift']}")
    z = np.load(LANE / "oracle" / "npz" / f"{run['id']}__{arm}.npz")
    want = z["rope_pos"]
    got = host.rope_positions(ids, start, amount, hw[1] if hw else 1)
    if got.shape != want.shape or not np.array_equal(got, want):
        bad.append("rope planes differ")
    if not np.array_equal(z["input_ids"], np.asarray(run["ids"])):
        bad.append("npz input_ids differ from fixture_oracle.json ids")
    return bad, ids


def test_build_ids_matches_oracle() -> dict:
    fx_json = LANE / "oracle" / "fixture_oracle.json"
    oracle = json.loads(fx_json.read_text())
    rows_path = LANE / "fixtures" / "rows.json"
    assert oracle["fixture"]["rows_json_sha256"] == hashlib.sha256(rows_path.read_bytes()).hexdigest(), \
        "fixtures/rows.json changed since the oracle ran"
    fx_rows = {r["id"]: r for r in json.loads(rows_path.read_text())["rows"]}
    snapshot = Path(hf_snapshot(HF_ID, revision=REVISION))
    toks = tokenizers_under_test(snapshot)
    runs = oracle["rows"]
    report = {"runs": len(runs), "arms": {}, "tokenizers": {}, "fixture_oracle_sha256":
              hashlib.sha256(fx_json.read_bytes()).hexdigest(), "snapshot": str(snapshot)}
    for a in sorted({r["arm"] for r in runs}):
        report["arms"][a] = sum(r["arm"] == a for r in runs)
    first_ids: dict[tuple, list[int]] = {}
    for name, tok in toks.items():
        fails = []
        for run in runs:
            bad, ids = check_run(run, fx_rows[run["id"]], tok)
            key = (run["id"], run["arm"])
            if key in first_ids and first_ids[key] != ids:
                bad.append("ids differ between tokenizers")
            first_ids.setdefault(key, ids)
            if bad:
                fails.append(f"{run['id']}/{run['arm']}: {'; '.join(bad)}")
        report["tokenizers"][name] = {"pass": len(runs) - len(fails), "fail": fails}
        print(f"{name}: {len(runs) - len(fails)}/{len(runs)} runs match (ids, slots, rope)")
        for f in fails[:10]:
            print("   ", f)
    ok = all(not v["fail"] for v in report["tokenizers"].values())
    report["pass"] = ok
    report["negative_controls"] = negative_controls(runs, fx_rows, next(iter(toks.values())))
    print("negative controls:", json.dumps(report["negative_controls"]))
    assert ok, report["tokenizers"]
    assert all(v["red"] == v["runs"] > 0 for v in report["negative_controls"].values()), \
        report["negative_controls"]
    return report


def negative_controls(runs: list[dict], fx_rows: dict, tok) -> dict:
    """The rope check must go red: shift amount by one (every image run), and swap H/W on the
    native arm's non-square grids (the only runs where a row/col transposition is visible)."""
    out = {"amount_plus_1": {"runs": 0, "red": 0}, "hw_swapped_nonsquare": {"runs": 0, "red": 0}}
    for run in runs:
        if run["arm"] == "text":
            continue
        _, gh, gw = run["grid_thw"]
        h, w = gh // host.MERGE, gw // host.MERGE
        fx = fx_rows[run["id"]]
        ids, _, start, amount = host.build_ids(True, fx["context"], fx["questions"], tok, (h, w))
        want = np.load(LANE / "oracle" / "npz" / f"{run['id']}__{run['arm']}.npz")["rope_pos"]
        out["amount_plus_1"]["runs"] += 1
        out["amount_plus_1"]["red"] += not np.array_equal(host.rope_positions(ids, start, amount + 1, w), want)
        if h != w:
            ids_t, _, start_t, amount_t = host.build_ids(True, fx["context"], fx["questions"], tok, (w, h))
            out["hw_swapped_nonsquare"]["runs"] += 1
            out["hw_swapped_nonsquare"]["red"] += not np.array_equal(
                host.rope_positions(ids_t, start_t, amount_t, h), want)
    return out


if __name__ == "__main__":
    rep = test_build_ids_matches_oracle()
    print(json.dumps({k: v for k, v in rep.items() if k != "tokenizers"}))
    print("PASS")
