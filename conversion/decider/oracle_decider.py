#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==5.17.0",
#     "safetensors>=0.8.0",
#     "huggingface_hub>=1.5.0,<2",
#     "numpy",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""Fixtures + fp32 oracle for the decider System One readout, from the author's own code.

Mapika/decider-0.8b answers typed questions (Choice / Score / Noul) by the probability of
option-letter tokens at an answer slot; it never generates text. This script downloads the
checkpoint's `decider/` inference package (the author's prompt builder, row planner and slot
readout, Apache-2.0), builds the zoo's fixture requests through it exactly as
`Decider.system_one(state, questions)` plans them — state-first layout, one independent row per
question, one yes/no row per isolated Score level, no option shuffle — runs the fp32 model on
the CPU, and writes one JSON with every row's token ids, slot, label ids, fp32 slot logits and
probabilities (softmax at the checkpoint's temperature, 1.03). The two Core AI gates in this
directory compare a bundle against that file, so it is published next to the card and the
gates re-run without rebuilding the oracle.

    uv run conversion/decider/oracle_decider.py --out models/decider-0.8b/fixtures-decider-0.8b.json

The author's row contract, as the file records it: the slot is the LAST token of the row; the
label table is A..Z followed by the first 229 two-letter strings that are single tokens (255);
`p = softmax(slot_logits[:nopts] / 1.03)`. Rows stay under 1,024 tokens except the 255-option
row (1,965 tokens; the bundle's context is 4,096). No real people, companies or products appear
in the requests.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_HF_ID = "Mapika/decider-0.8b"
DEFAULT_REVISION = "1ea54127d3bd52f6d753d9257b32a6380b873907"


def choice(instructions, criteria):
    crit = dict.fromkeys(criteria) if isinstance(criteria, list) else criteria
    return dict(type="choice", instructions=instructions, criteria=crit)


def noul(instructions, criteria=None):
    return dict(type="noul", instructions=instructions, criteria=criteria or {})


def score(instructions, criteria):
    return dict(type="score", instructions=instructions, criteria=criteria)


def request(id, state, questions):
    return dict(id=id, state=state, questions=questions)


# Deterministic, synthetic English requests. 13 requests -> 44 rows: 23 Choice rows with
# 3-10 options, 9 Noul rows, 10 isolated yes/no rows from 2 Score questions, one 11-option
# row (the first "wide" rendering) and one 255-option row.
REQUESTS = [
    request("r01", "A parcel has a broken latch. Its contents are intact. The sender asks for a replacement latch, not a refund.", {
        "route": choice("Which team should handle the request?", {"repair": "Replace damaged parts", "billing": "Handle payments", "delivery": "Find missing parcels"}),
        "condition": choice("What is the condition of the contents?", ["missing", "intact", "wet", "broken"]),
        "refund": noul("Does the sender request a refund?", {"false": "No money back is requested", "true": "Money back is requested"})}),
    request("r02", {"room": {"temperature": "cold", "lamp": "off", "window": "open"}, "request": "Close the window."}, {
        "action": choice("What action is requested?", ["open the window", "close the window", "turn on the lamp"]),
        "lamp": choice("What is the current lamp state?", ["on", "flashing", "off"]),
        "cold": noul("Is the room described as cold?")}),
    request("r03", "The marked tile is a green hexagon. It is dry and has no border.", {
        "color": choice("What color is the marked tile?", ["red", "blue", "yellow", "purple", "green", "orange"]),
        "shape": choice("What shape is the marked tile?", ["circle", "square", "triangle", "pentagon", "hexagon", "oval", "star", "rectangle"]),
        "wet": noul("Is the marked tile wet?", {"false": "The tile is dry", "true": "The tile has water on it"})}),
    request("r04", {"cabinet": {"drawer": "seven", "material": "wood", "status": "locked"}, "key": "present"}, {
        "drawer": choice("Which drawer is named?", ["one", "two", "three", "four", "five", "six", "seven"]),
        "material": choice("What material is the cabinet made from?", ["glass", "stone", "steel", "paper", "clay", "wood", "rubber", "copper", "fabric", "plastic"]),
        "key": noul("Is the key present?")}),
    request("r05", "A garden bed is dry. The seedlings have green leaves. The next task is watering.", {
        "task": choice("What is the next task?", ["harvesting", "watering", "painting"]),
        "leaves": choice("What color are the leaves?", ["brown", "yellow", "green", "red"]),
        "dry": noul("Is the garden bed dry?")}),
    request("r06", {"job": {"phase": "review", "result": "pending", "priority": "low"}, "approved": False}, {
        "phase": choice("Which phase is the job in?", ["draft", "review", "complete", "cancelled"]),
        "priority": choice("What is the job priority?", ["high", "medium", "low"]),
        "approved": noul("Has the job been approved?", {"false": "Approval has not been granted", "true": "Approval has been granted"})}),
    request("r07", "A container holds sand. Its lid is blue. A label says to keep it indoors, and the seal is unbroken.", {
        "contents": choice("What does the container hold?", ["water", "sand", "salt", "seeds"]),
        "location": choice("Where should the container be kept?", ["outdoors", "underground", "indoors", "on a roof", "in a pond"]),
        "seal": noul("Is the seal broken?")}),
    request("r08", {"route": {"direction": "west", "surface": "gravel", "closed": True}, "weather": "clear"}, {
        "direction": choice("Which direction does the route lead?", ["north", "south", "east", "west", "up"]),
        "surface": choice("What is the route surface?", ["gravel", "ice", "asphalt"]),
        "open": noul("Is the route open?", {"false": "The route is closed", "true": "Travel is permitted"})}),
    request("r09", "The storage tank is half full of clean water. The valve is closed. Its fill level is exactly halfway between empty and full.", {
        "liquid": choice("What liquid is in the tank?", ["oil", "water", "ink"]),
        "valve": choice("What is the valve position?", ["open", "closed", "missing"]),
        "clarity": choice("How is the water described?", ["dirty", "cloudy", "frozen", "clean"]),
        "fill": score("How full is the tank?", ["empty", "one quarter full", "half full", "three quarters full", "completely full"])}),
    request("r10", {"inspection": {"damage": "none", "surface": "smooth", "color": "white", "count": 4}}, {
        "surface": choice("How is the inspected surface described?", ["rough", "smooth", "cracked", "sticky"]),
        "color": choice("What is the surface color?", ["black", "white", "gray"]),
        "count": choice("How many items were inspected?", ["two", "four", "six"]),
        "damage": score("How much damage is present?", ["no damage", "minor scratches", "several shallow dents", "large cracks", "completely broken"])}),
    request("r11", "The selected storage bin is violet. Its lid is closed.", {
        "bin": choice("Which storage bin is selected?", ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "black", "white", "gray", "brown"])}),
    request("r12", "The requested archive slot is 173. Retrieve that slot.", {
        "slot": choice("Which archive slot is requested?", [str(i) for i in range(1, 256)])}),
    request("r13", {"notice": {"status": "paused", "reason": "inspection"}, "restart_allowed": False}, {
        "status": choice("What is the current status?", ["running", "paused", "finished"]),
        "restart": noul("Is a restart currently allowed?")}),
]

ROW_TOKEN_LIMIT = 1024       # every row except the 255-option one
WIDE_ROW_TOKEN_LIMIT = 2048  # the 255-option row needs >= 1,275 tokens by construction


class _Keep:
    """`Decider.system_one` passes a no-op RNG so option order is the request's order."""

    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", default="models/decider-0.8b/fixtures-decider-0.8b.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(args.threads)
    snapshot = snapshot_download(
        args.hf_id, revision=args.revision,
        allow_patterns=["config.json", "model.safetensors", "tokenizer*", "decider/*.py",
                        "decider_config.json", "generation_config.json"])
    sys.path.insert(0, snapshot)
    from decider.infer import Decider, Example, Q          # noqa: E402  (author's code)
    from decider.model import collate                      # noqa: E402
    from decider.prompt import build, label_table, letter_ids  # noqa: E402
    from decider.systemone import assemble, plan_rows, render_question, render_state  # noqa: E402

    t0 = time.monotonic()
    d = Decider(snapshot, device="cpu", dtype=torch.float32, use_graphs=False)
    assert d.T == 1.03 and d.neutralize_none is False and d.isolated_levels is True, (
        d.T, d.neutralize_none, d.isolated_levels)
    assert len(letter_ids(d.m.tok)) == 255
    labels, label_ids, _ = label_table(d.m.tok)
    assert all(d.m.tok.encode(labels[i], add_special_tokens=False) == [label_ids[i]] for i in range(255))

    rows, planned = [], []
    for req in REQUESTS:
        ctx = render_state(req["state"])
        rqs = {k: render_question(v) for k, v in req["questions"].items()}
        planned_rows, index = plan_rows(rqs, isolated=True)
        items = [build(Example(ctx, [Q(r["question"], r["options"], 0)]), d.m.tok, _Keep(),
                       max_options=255, max_ctx_tokens=32768, layout="state_first")
                 for r in planned_rows]
        recs = []
        for qi, kind, first, n in index:
            for j in range(n):
                k = first + j
                item = items[k]
                nopts, slot = item["nopts"][0], item["slots"][0]
                assert slot == len(item["ids"]) - 1, "the slot must be the last token"
                limit = WIDE_ROW_TOKEN_LIMIT if nopts == 255 else ROW_TOKEN_LIMIT
                assert len(item["ids"]) <= limit, (req["id"], qi, len(item["ids"]), limit)
                rec = {
                    "id": f"{req['id']}-{qi}" + (f"-level{j}" if kind == "iso" else ""),
                    "request_id": req["id"], "question_id": qi, "kind": kind,
                    "type": rqs[qi]["type"], "level_index": j if kind == "iso" else None,
                    "question": planned_rows[k]["question"], "options": planned_rows[k]["options"],
                    "ids": item["ids"], "slot": slot, "nopts": nopts,
                    "label_ids": label_ids[:nopts], "label_strings": labels[:nopts],
                    "tokens": len(item["ids"]),
                }
                rows.append(rec)
                recs.append(rec)
        planned.append((req, rqs, index, items, recs))

    api_assembly = []
    for req, rqs, index, items, recs in planned:
        batch = collate(items, d.m.tok.pad_token_id)
        with torch.no_grad():
            logits = d.m.slot_logits(*[batch[k].to(d.dev) for k in
                                       ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
            probs = torch.softmax(logits / d.T, -1).cpu()
        for i, rec in enumerate(recs):
            n = rec["nopts"]
            values = logits[i, :n].float().cpu()
            p = probs[i, :n]
            assert torch.isfinite(values).all() and torch.isfinite(p).all()
            # The explicit restricted softmax equals the author's masked head.
            assert float((torch.softmax(values / d.T, -1) - p).abs().max()) < 1e-6
            top = torch.topk(p, 2)
            rec.update({
                "slot_logits": values.tolist(), "p_oracle": p.tolist(),
                "argmax": int(p.argmax()), "argmax_label": labels[int(p.argmax())],
                "argmax_token_id": label_ids[int(p.argmax())],
                "top2_margin": float(top.values[0] - top.values[1]),
            })
        expected = assemble(rqs, index, probs.tolist())
        api = d.system_one(req["state"], req["questions"])
        api_assembly.append({"request_id": req["id"], "exact_equal": api["answers"] == expected,
                             "system_one": api})
        assert api["answers"] == expected, req["id"]
        print(f"{req['id']}: {len(recs)} rows, API assembly exact", flush=True)

    out = {
        "schema": "coreai-decider-fixtures/1",
        "source": {"hf_id": args.hf_id, "revision": args.revision,
                   "oracle": "author's decider/model.py slot_logits, fp32, CPU",
                   "torch": torch.__version__},
        "temperature": d.T,
        "layout": "state_first", "independent_rows": True, "isolated_levels": True,
        "neutralize_none": False,
        "label_table": [[labels[i], label_ids[i]] for i in range(255)],
        "requests": REQUESTS,
        "rows": rows,
        "api_assembly": api_assembly,
        "summary": {
            "requests": len(REQUESTS), "rows": len(rows),
            "min_top2_margin": min(r["top2_margin"] for r in rows),
            "max_tokens": max(r["tokens"] for r in rows),
            "wall_seconds": time.monotonic() - t0,
        },
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out["summary"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
