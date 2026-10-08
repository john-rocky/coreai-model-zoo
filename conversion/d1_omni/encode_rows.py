#!/usr/bin/env python3
"""Gate the host (host.py) against the publisher's own code on every fixture row, with no weights.

    python3 conversion/d1_omni/encode_rows.py      # -> $ZOO_WORK_ROOT/_d1_omni/results/encode_rows.json

Runs in a private venv with the checkpoint's requirement (transformers >= 5.15) and reads the pinned snapshot's
code (`prompt.py`, `modeling_d1.py`, `vision.py`, `audio.py`, `encoder.py`) with importlib; nothing is downloaded
(HF_HUB_OFFLINE=1). Four checks, each an assert:

1. copies: host.py's copied functions are the publisher's source text (prompt.py, vision.layout).
2. encode: for every record x question x mode (text for every record; image / audio for the records with media,
   at the media's prefix length), host.request_rows' ids and markers == the publisher's `prompt.encode` with the
   same tokenizer (AutoTokenizer, the publisher's path), and == host.request_rows over the raw `tokenizers` file.
3. dispatch: the publisher's `D1OmniModel.system_one_batch` run over every request at once with the networks
   replaced by stubs (embedding width 1; vision / audio return a zero prefix of the media's length; the head
   returns a fixed integer-derived function of the marker position and the question type) gives the same
   answers and usage as host.py on the same stub logits: the mode dispatch (max_len, noul wording, audio option
   text, state None), the token-budget batching, the temperature and the noul flip.
4. records (no assert): per row the length, the marker positions, the delimiter positions, the truncation of
   the state / instructions / options, the temperature; per source the length percentiles, the bucket of each
   row (prefix + text positions; buckets 256 / 512 / 1024 / 2048 / 4096) and the rows within 16 of a boundary.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

WORK = work_path("_d1_omni")
PROMPT_NAMES = ("Question", "as_question", "escape", "serialize", "_criterion", "render_options", "encode", "answer",
                "temperature_key")
NEAR = 16


def load_upstream(snapshot: Path) -> types.SimpleNamespace:
    """The snapshot's .py files as a package `d1_upstream` (their relative imports resolve inside it)."""
    package = types.ModuleType("d1_upstream")
    package.__path__ = [str(snapshot)]
    sys.modules["d1_upstream"] = package
    mods = {name: importlib.import_module(f"d1_upstream.{name}")
            for name in ("prompt", "vision", "audio", "encoder", "modeling_d1")}
    return types.SimpleNamespace(**mods)


def check_copies(up) -> dict:
    out = {}
    for name in PROMPT_NAMES:
        out[name] = inspect.getsource(getattr(host, name)) == inspect.getsource(getattr(up.prompt, name))
    for name in ("QTYPES", "DELIM", "MARKER"):
        out[name] = getattr(host, name) == getattr(up.prompt, name)
    out["_SPECIAL"] = host._SPECIAL.pattern == up.prompt._SPECIAL.pattern
    out["vision.layout"] = inspect.getsource(host.layout) == inspect.getsource(up.vision.layout)
    out["vision constants"] = (host.TILE, host.PATCH, host.MAX_PATCHES) == (up.vision.TILE, up.vision.PATCH,
                                                                             up.vision.MAX_PATCHES)
    out["audio constants"] = (host.SAMPLE_RATE, host.MIN_SAMPLES, host.MAX_SECONDS) == (
        up.audio.SAMPLE_RATE, up.audio.MIN_SAMPLES, up.audio.MAX_SECONDS)
    out["YES_NO"] = host.YES_NO == up.modeling_d1.YES_NO
    bad = [k for k, v in out.items() if not v]
    assert not bad, f"host copies differ from the publisher's source: {bad}"
    return out


# --------------------------------------------------------------------------- media prefix lengths of the fixtures
def media_of(record: dict) -> tuple[str | None, int, dict]:
    """(mode, prefix length, what it was computed from) for a record's media; (None, 0, {}) without media."""
    media = record["media"]
    if media is None:
        return None, 0, {}
    prov = record["provenance"]
    if "images" in media:
        size = (prov.get("image_info") or {}).get("px") or prov.get("px")
        return "image", host.image_prefix_length([tuple(size)]), {"px": size, "file_present": (WORK / media["images"][0]).exists()}
    if "clip" in prov:
        samples = prov["clip"]["frames"]
    elif "audio_info" in prov:
        samples = prov["audio_info"]["frames"]
    else:
        raise ValueError(f"{record['id']}: audio without a sample count")
    return "audio", host.audio_prefix_length(samples), {"samples": samples}


# --------------------------------------------------------------------------- stub logits (exact in torch and numpy)
def stub_logits_torch(marker_pos: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
    z = (marker_pos.to(torch.int64) * 7919 + qtype.to(torch.int64)[:, None] * 104729 + 13) % 997
    return z.to(torch.float32) / 100.0 - 5.0


def stub_logits_numpy(markers: list[int], qtype: int) -> np.ndarray:
    z = (np.asarray(markers, dtype=np.int64) * 7919 + qtype * 104729 + 13) % 997
    return z.astype(np.float32) / np.float32(100.0) - np.float32(5.0)


def make_spy(up, tok, config_json: dict):
    md = up.modeling_d1

    class StubTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(65536, 1)

        def forward(self, h, pad, prefix):
            return h

    class StubHead(nn.Module):
        def forward(self, h, pad, marker_pos, marker_mask, qtype):
            return stub_logits_torch(marker_pos, qtype).masked_fill(~marker_mask, -1e4)

    class Spy(md.D1OmniModel):
        device = torch.device("cpu")

        def __init__(self, config):
            nn.Module.__init__(self)
            self.config = config
            self.encoder = StubTrunk()
            self.head = StubHead()
            self.vision = lambda images: torch.zeros(1, host.image_prefix_length(images), 1)
            self.audio = lambda audio: torch.zeros(1, host.audio_prefix_length(int(audio)), 1)

    spy = Spy(md.D1OmniConfig.from_dict(config_json))
    spy.__dict__["tokenizer"] = tok
    spy.eval()
    return spy


def request_args(record: dict, mode: str):
    """The (state, questions[, images[, audio]]) tuple system_one_batch takes, with media placeholders the stubs
    read: a list of (width, height) for images, the sample count for audio."""
    state, questions = record["request"]["state"], record["request"]["questions"]
    if mode == "text":
        return (state, questions)
    if mode == "image":
        prov = record["provenance"]
        size = (prov.get("image_info") or {}).get("px") or prov.get("px")
        return (state, questions, [tuple(size)])
    prov = record["provenance"]
    samples = prov["clip"]["frames"] if "clip" in prov else prov["audio_info"]["frames"]
    return (state, questions, None, samples)


# --------------------------------------------------------------------------- per-row record
def row_record(record, mode, native, row: host.Row, tok) -> dict:
    enc = lambda s: tok(host.escape(s), add_special_tokens=False)["input_ids"]  # noqa: E731
    q, ids, markers = row.question, row.ids, row.markers
    settings = host.mode_settings(mode, row.prefix_len)
    opts = host.render_options(q, settings["noul_default"], settings["audio"])
    k = len(opts)
    budget = max(96, min(k * 24 + 32, settings["max_len"] // 2))
    per = max(2, (budget - 3 * k) // k)
    question_full = 1 + len(enc(q.instructions))
    state = record["request"]["state"]
    if mode == "audio" and state is None:
        state = {}
    state = "" if state is None else state
    state_full = len(enc(host.serialize(state)))
    question_len = min(question_full, max(16, budget)) + sum(2 + min(len(enc(" " + o)), per) + 1 for o in opts) + 1
    room = max(0, settings["max_len"] - question_len - 2)
    q_pos = 1 + 1 + min(state_full, room)
    assert ids[0] == host.BOS_ID and ids[1] == host.TOKEN_IDS["<|reserved_7|>"], record["id"]
    assert ids[q_pos] == host.TOKEN_IDS["<|reserved_8|>"], (record["id"], q_pos)
    assert all(ids[m] == host.MASK_ID and ids[m - 1] == host.TOKEN_IDS["<|reserved_9|>"] for m in markers), record["id"]
    assert ids[-1] == host.TOKEN_IDS["<|reserved_11|>"] and ids.count(host.MASK_ID) == k, record["id"]
    option_cuts = [max(0, len(enc(" " + o)) - per) for o in opts]
    out = {
        "id": record["id"], "qid": row.qid, "source": record["source"], "public": record["public"], "mode": mode,
        "native": native, "type": q.type, "K": k, "prefix": row.prefix_len, "n_ids": len(ids),
        "positions": row.positions, "bucket": host.bucket_for(row.positions), "max_len": settings["max_len"],
        "markers": markers, "q_delim_pos": q_pos, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "state_tokens": state_full, "state_room": room, "state_cut": max(0, state_full - room),
        "instructions_tokens": question_full - 1, "instructions_cut": max(0, question_full - max(16, budget)),
        "budget": budget, "per_option": per, "option_cut": option_cuts, "row_cut": len(ids) == settings["max_len"],
        "calibrate": row.calibrate,
    }
    if row.calibrate:
        out["temperature_key"] = host.temperature_key(q)
        out["T"] = host.temperature(q)
    return out


def nearest_rank(values: list[int], q: float) -> int:
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def aggregate(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["source"], r["mode"])].append(r)
    out = {}
    for (source, mode), group in sorted(by.items()):
        pos = [r["positions"] for r in group]
        out[f"{source}/{mode}"] = {
            "rows": len(group), "native_rows": sum(r["native"] for r in group),
            "n_ids": {"p50": nearest_rank([r["n_ids"] for r in group], 0.5),
                      "p99": nearest_rank([r["n_ids"] for r in group], 0.99), "max": max(r["n_ids"] for r in group)},
            "positions": {"p50": nearest_rank(pos, 0.5), "p99": nearest_rank(pos, 0.99), "max": max(pos)},
            "buckets": dict(Counter(str(r["bucket"]) for r in group)),
        }
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=WORK / "results" / "encode_rows.json",
                    help="a new file (round 5: results/encode_rows.v2.json for fixtures v2)")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: not replaced")
    t0 = time.time()
    from transformers import AutoTokenizer
    import transformers

    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    up = load_upstream(snapshot)
    copies = check_copies(up)
    auto = AutoTokenizer.from_pretrained(host.MODEL_ID, revision=host.MODEL_SHA, trust_remote_code=True)
    raw = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(auto), host.check_token_ids(raw)
    fixtures_path = WORK / "fixtures" / "records.json"
    fixtures_bytes = fixtures_path.read_bytes()
    records = json.loads(fixtures_bytes)["records"]
    config_json = json.loads((snapshot / "config.json").read_text())

    rows_out, n_encode, requests, host_rows_by_request = [], 0, [], []
    for record in records:
        media_mode, prefix_len, media_from = media_of(record)
        modes = ["text"] + ([media_mode] if media_mode else [])
        for mode in modes:
            p = prefix_len if mode != "text" else 0
            native = mode == (media_mode or "text")
            rows_auto = host.request_rows(auto, record["request"]["state"], record["request"]["questions"], mode, p)
            rows_raw = host.request_rows(raw, record["request"]["state"], record["request"]["questions"], mode, p)
            settings = host.mode_settings(mode, p)
            state = record["request"]["state"]
            if mode == "audio" and state is None:
                state = {}
            state = "" if state is None else state
            for row, row_raw in zip(rows_auto, rows_raw):
                theirs = up.prompt.encode(auto, state, up.prompt.as_question(record["request"]["questions"][row.qid]),
                                          settings["max_len"], settings["noul_default"], settings["audio"])
                assert (row.ids, row.markers) == tuple(theirs), (record["id"], row.qid, mode)
                assert (row_raw.ids, row_raw.markers) == tuple(theirs), (record["id"], row.qid, mode, "raw")
                n_encode += 1
                rec = row_record(record, mode, native, row, auto)
                if media_from and mode != "text":
                    rec["media_from"] = media_from
                rows_out.append(rec)
            requests.append(request_args(record, mode))
            host_rows_by_request.append(rows_auto)

    # dispatch: the publisher's batched API on stubs vs host.py on the same stub logits
    spy = make_spy(up, auto, config_json)
    with torch.no_grad():
        theirs = spy.system_one_batch(requests)
    max_dp, n_answers, mismatches = 0.0, 0, []
    for rows, resp in zip(host_rows_by_request, theirs):
        probs = [host.probabilities_from_logits(stub_logits_numpy(r.markers, host.QTYPES[r.question.type]), r.question,
                                                r.calibrate) for r in rows]
        mine = host.response(rows, probs)
        if mine["usage"] != resp["usage"] or list(mine["answers"]) != list(resp["answers"]):
            mismatches.append({"usage": [mine["usage"], resp["usage"]]})
        for name, a in mine["answers"].items():
            b = resp["answers"][name]
            n_answers += 1
            if set(a) != set(b) or a["type"] != b["type"] or a.get("choice") != b.get("choice"):
                mismatches.append({"name": name, "mine": a, "theirs": b})
                continue
            pa = [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
            pb = [b["noul"]] if b["type"] == "noul" else list(b["probabilities"].values())
            max_dp = max(max_dp, max(abs(x - y) for x, y in zip(pa, pb)))
            if a["type"] == "score":
                max_dp = max(max_dp, abs(a["score"] - b["score"]))
            if a["type"] != "noul" and (list(a["probabilities"]) != list(b["probabilities"])):
                mismatches.append({"name": name, "keys": [list(a["probabilities"]), list(b["probabilities"])]})
    assert not mismatches, mismatches[:3]
    assert max_dp <= 1e-6, max_dp

    native = [r for r in rows_out if r["native"]]
    near = sorted({(r["id"], r["qid"], r["mode"], r["positions"], b) for r in rows_out for b in host.BUCKETS
                   if abs(r["positions"] - b) <= NEAR})
    doc = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fixtures_sha256": hashlib.sha256(fixtures_bytes).hexdigest(),
        "model": {"hf_id": host.MODEL_ID, "revision": host.MODEL_SHA}, "transformers": transformers.__version__,
        "torch": torch.__version__, "tokenizer_class": type(auto).__name__,
        "checks": {
            "copies_identical": copies,
            "encode_rows_equal": f"{n_encode}/{n_encode}",
            "encode_compared": "host.request_rows (AutoTokenizer) == prompt.encode (AutoTokenizer) == host.request_rows (raw tokenizers)",
            "dispatch": {"requests": len(requests), "answers": n_answers, "answers_equal": n_answers,
                         "usage_equal": len(requests), "max_abs_dp_numpy_vs_torch": max_dp,
                         "stub": "embedding width 1, prefix zeros, head logits = ((pos * 7919 + qtype * 104729 + 13) % 997) / 100 - 5"},
            "row_asserts": "ids[0] = 1, ids[1] = 17, ids[q_delim_pos] = 18, every marker = 16 after 19, last = 21, K markers",
        },
        "summary": {
            "rows": len(rows_out), "native_rows": len(native),
            "by_source_mode": aggregate(rows_out),
            "native_buckets": dict(Counter(str(r["bucket"]) for r in native)),
            "native_buckets_by_mode": {m: dict(Counter(str(r["bucket"]) for r in native if r["mode"] == m))
                                       for m in host.MODES},
            "truncated": {
                "state": sorted({(r["id"], r["mode"], r["state_cut"]) for r in rows_out if r["state_cut"]}),
                "instructions": sorted({(r["id"], r["qid"], r["mode"], r["instructions_cut"]) for r in rows_out
                                        if r["instructions_cut"]}),
                "options": sorted({(r["id"], r["qid"], r["mode"], sum(r["option_cut"])) for r in rows_out
                                   if any(r["option_cut"])}),
                "row_at_max_len": sorted({(r["id"], r["qid"], r["mode"]) for r in rows_out if r["row_cut"]}),
            },
            f"near_bucket_boundary_pm{NEAR}": near,
            "over_4096": sorted({(r["id"], r["qid"], r["mode"], r["positions"]) for r in rows_out if r["bucket"] is None}),
            "temperature_keys": dict(Counter(r.get("temperature_key") for r in rows_out if r["calibrate"])),
        },
        "rows": rows_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = args.out
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    s = doc["summary"]
    print(f"{out}: encode {n_encode}/{n_encode} rows equal; dispatch {len(requests)} requests / {n_answers} answers equal, "
          f"max |dp| {max_dp:.2e}; {doc['elapsed_s']} s")
    print("native buckets:", s["native_buckets_by_mode"])
    for key, v in s["by_source_mode"].items():
        print(f"  {key}: {v['rows']} rows, positions p50/p99/max {v['positions']['p50']}/{v['positions']['p99']}/"
              f"{v['positions']['max']}, buckets {v['buckets']}")
    print("truncated:", {k: len(v) for k, v in s["truncated"].items()}, "| near boundary:", len(near),
          "| over 4096:", s["over_4096"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
