#!/usr/bin/env python3
"""d1_omni_model.D1Decide in fp32 with the checkpoint's weights, one row at its bucket length, against the
publisher's model on every fixture row (ref/records_ref.json from reference_d1_omni.py), CPU.

    python3 conversion/d1_omni/eager_check.py   # -> $ZOO_WORK_ROOT/_d1_omni/results/eager_check.json

Weights: load_d1_decide(snapshot, 256, "fp32", verify_sha256=True) once; the other buckets (512 / 1024 / 2048 /
4096) and the variants below are modules of the same class holding the same tensors (load_state_dict(assign=True)).
Every row runs at host.bucket_for(prefix + text positions) through host.graph_inputs (a prefix row takes the
reference's own prefix tensor from ref/npz), its scores are read at P + markers, and host.probabilities_from_logits
gives the answer distribution.

Bar: argmax equal on every row (near ties included), max |dp| <= 2e-5 after the temperature, marker logits max |d|
<= 1e-3. The max |dp| bar was 1e-5 before round 2's run and was set to 2e-5 by the supervisor after it (BAR_BASIS):
twice the fp32 reference's own distance from the publisher's code run in float64. Two fp32 computations of the same
function in a different order (and at a different padded length) cannot be held below the sum of their rounding:
the reference is 9.3e-6 from float64, this module 7.9e-6, and with every fp32 step in float64 and the RoPE tables
built for the same length the two agree to 1.2e-13 in the logits.

Also measured:
  (a) the trunk output (after embedding_norm) against the reference's on the npz rows: real positions, max |d| and
      max |d| / max |ref|
  (b) pad content: pad ids 1 / 100 / random, prefix_embeds random at the pad positions, prefix_embeds random at
      every non-prefix position: the real-position scores stay bit-identical (20 rows)
  (c) bucket: the same row at two lengths (256 and 512: 10 rows; 2048 and 4096: 3 rows), marker logits max |d|
  (d) GQA as repeat_interleave vs reshape-broadcast at full size (10 rows): bit-identical or max |d|
  (e) the three mutations (keep_right_ones / no_media_text_mask / no_head_key_mask) on the 5 prefix rows: max |dp|
      against the reference (> 0.02 was expected) and marker logits max |d| (> the 1e-3 logit bar). On rows whose
      distribution is saturated a mutation moves the logits but hardly the probabilities, so a prefix-row gate
      reads both.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _fixtures import fixtures_path  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402

WORK = work_path("_d1_omni")
BARS = {"argmax": "all rows", "max_abs_dp": 2e-5, "max_abs_dlogit": 1e-3}
MUTATION_BAR = 0.02
# Why max |dp| <= 2e-5 (supervisor's ruling 2026-10-08, after round 2's first run at 1e-5): measured by round 2's
# one-off diagnostics in the work dir (results/dp_float64.json: 470 rows; dp_isolate.json: 17 rows;
# dp_float64_pure.json: 50 rows; rope_length.json; ref_summary.json). pub64 = the publisher's trunk + head in float64
# (its RMSNorm normalises in fp32 as written); mine64 = this module in float64.
BAR_BASIS = {
    "rule": "max |dp| <= 2 x the fp32 reference's max |dp| against the publisher's code in float64",
    "ref_vs_pub64_max_abs_dp": 9.302171096958745e-06,
    "mine_vs_pub64_max_abs_dp": 7.897718024424405e-06,
    "mine64_vs_pub64_max_abs_dp": 2.114485346538242e-06,
    "mine64_vs_pub64_note": "dp_isolate.json (17 rows): the module in float64 keeping its fp32 attention softmax, "
                            "LayerNorm and scorer; dp_float64.json (470 rows) with those in float64: 1.161161186202797e-06",
    "pure_float64_same_rope_length_max_abs_dlogit": 1.1901590823981678e-13,
    "pure_float64_note": "dp_float64_pure.json (the 50 rows furthest apart): every fp32 step of both sides in float64 "
                         "and the publisher's trunk padded to the module's length; unpadded 2.54e-07 (the RoPE table)",
    "publisher_padded_to_bucket_max_abs_dp": 9.713121706333983e-06,
    "publisher_head_fast_path_max_abs_dp": 2.2091844864569055e-07,
    "publisher_batch_partners_max_abs_dp": 4.9173831939697266e-06,
    "rope_table": "the publisher's Trunk.rope(n) differs from rope(L)[:n] by one fp32 ulp in 18 (n 191) to 729 "
                  "(n 3497) entries at 12 threads, none at 1 or 4 threads; this module's table at L equals the "
                  "publisher's at L",
}
SEED = 20261008
PAD_ROWS_EXTRA = [("card_text", "refund", "text"), ("red_arm_000", "answer", "text"),
                  ("aud_01", "wants", "audio"), ("aud_01", "urgency", "audio")]
BUCKET_ROWS_256 = [("card_text", "team", "text"), ("tv4_000", "answer", "text"), ("semif_FIRST", "decision", "text"),
                   ("own_t01", "upset", "text"), ("own_m01", "furnished", "text"), ("tv4x_sciq_02", "answer", "text"),
                   ("tv4s_00", "decision", "text"), ("red_arm_000", "answer", "text"),
                   ("card_audio", "topic", "audio"), ("aud_02", "urgency", "audio")]
BUCKET_ROWS_2048 = [("own_L01", "stoppage", "text"), ("own_L03", "fee_motion", "text"),
                    ("own_L02", "subcontract", "text")]
GQA_ROWS = [("card_text", "urgency", "text"), ("tv4_000", "answer", "text"), ("semif_FIRST", "decision", "text"),
            ("own_t04", "blocked", "text"), ("own_j02", "next_action", "text"), ("tv4x_sciq_15", "answer", "text"),
            ("card_audio", "topic", "audio"), ("aud_03", "wants", "audio"), ("tv4_053", "answer", "text"),
            ("own_L03", "quorate", "text")]
PREFIX_ROWS = [("card_cats", "cats", "image"), ("card_audio", "topic", "audio"), ("aud_01", "topic", "audio"),
               ("aud_02", "topic", "audio"), ("aud_03", "topic", "audio")]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


class Models:
    """One D1Decide per (length, mutation, gqa), all holding the base module's tensors."""

    def __init__(self, base: dm.D1Decide, config: dict):
        self.base, self.config = base, config
        self.state = base.state_dict()
        self.cache = {(base.seq_len, "none", "repeat"): base}

    def get(self, seq_len: int, mutation: str = "none", gqa: str = "repeat") -> dm.D1Decide:
        key = (seq_len, mutation, gqa)
        if key not in self.cache:
            m = dm.build(self.config, seq_len, mutation, gqa)
            m.load_state_dict(self.state, strict=True, assign=True)
            m.eval().requires_grad_(False)
            assert m.trunk.embed_tokens.weight.data_ptr() == self.base.trunk.embed_tokens.weight.data_ptr()
            self.cache[key] = m
        return self.cache[key]


@contextmanager
def trunk_output(model: dm.D1Decide):
    """Capture the trunk output (the embedding_norm call's result) while the module runs unchanged."""
    box = {}
    original = dm.D1Decide._rms

    def rms(norm, x):
        y = original(norm, x)
        if norm is model.trunk.embedding_norm:
            box["h"] = y.detach().clone()
        return y

    model._rms = rms
    try:
        yield box
    finally:
        del model._rms


def to_tensors(inputs: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(inputs[k]) for k in ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask",
                                                  "keep_right", "qtype_onehot")]


def run_row(model: dm.D1Decide, row: host.Row, prefix, mutate=None):
    inputs, markers = host.graph_inputs(row, model.seq_len, prefix)
    if mutate is not None:
        mutate(inputs, row)
    scores = model(*to_tensors(inputs))
    return scores, markers


def bar_basis() -> dict:
    """BAR_BASIS, checked against the diagnostics' files when they are in the work dir."""
    out = dict(BAR_BASIS)
    files = {name: WORK / "results" / name for name in ("dp_float64.json", "dp_isolate.json", "dp_float64_pure.json")}
    if all(p.exists() for p in files.values()):
        f64 = json.loads(files["dp_float64.json"].read_text())["summary"]
        iso = json.loads(files["dp_isolate.json"].read_text())["aggregate"]
        pure = json.loads(files["dp_float64_pure.json"].read_text())
        seen = {"ref_vs_pub64_max_abs_dp": f64["ref-pub64"]["dp"]["max"],
                "mine_vs_pub64_max_abs_dp": f64["mine-pub64"]["dp"]["max"],
                "mine64_vs_pub64_max_abs_dp": iso["pub64-mine64"]["max_dp"],
                "pure_float64_same_rope_length_max_abs_dlogit": pure["max_dlogit_pure_same_rope_length"],
                "publisher_padded_to_bucket_max_abs_dp": iso["pubB1-pubL"]["max_dp"],
                "publisher_head_fast_path_max_abs_dp": iso["pubL-pubPL"]["max_dp"]}
        out["files_checked"] = {k: v == BAR_BASIS[k] for k, v in seen.items()}
    else:
        out["files_checked"] = "diagnostic files not in this work dir"
    out["twice_ref_vs_pub64"] = 2 * BAR_BASIS["ref_vs_pub64_max_abs_dp"]
    out["bar_max_abs_dp"] = BARS["max_abs_dp"]  # 2e-5: twice the reference's float64 distance, rounded up
    return out


def main() -> int:
    t_start = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    ref_path = WORK / "ref" / "records_ref.json"
    ref = json.loads(ref_path.read_text())
    assert ref["model"]["revision"] == host.MODEL_SHA
    fixtures_file = fixtures_path(ref["fixtures"]["sha256"])  # the version the reference read (_fixtures.py)
    fixtures = {r["id"]: r for r in json.loads(fixtures_file.read_text())["records"]}
    semif_first = next(i for i in fixtures if i.startswith("semif_"))

    def resolve(rows):
        return [(semif_first if rid == "semif_FIRST" else rid, qid, mode) for rid, qid, mode in rows]

    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok)
    config = json.loads((snapshot / "config.json").read_text())
    t0 = time.perf_counter()
    base, load_record = dm.load_d1_decide(snapshot, 256, "fp32", verify_sha256=True)
    load_seconds = time.perf_counter() - t0
    models = Models(base, config)

    prefixes, npz_hidden = {}, {}
    for name in sorted(p.stem for p in (WORK / "ref" / "npz").glob("*.npz")):
        with np.load(WORK / "ref" / "npz" / f"{name}.npz") as z:
            mode = str(z["mode"])
            rid = name.split("__")[0]
            if mode != "text":
                prefixes[rid] = z["prefix"].copy()
            for k, qid in enumerate(z["qids"].tolist()):
                npz_hidden[(rid, qid, mode)] = {"hidden": z[f"q{k}_hidden"].copy(), "file": f"{name}.npz",
                                                "prefix_len": int(z["prefix_len"])}

    # every reference row through the module
    rows_out, row_cache = [], {}
    t_rows = time.perf_counter()
    for entry in ref["records"]:
        record = fixtures[entry["id"]]
        req = record["request"]
        hrows = host.request_rows(tok, req["state"], req["questions"], entry["mode"], entry["prefix"])
        prefix = prefixes.get(entry["id"]) if entry["mode"] != "text" else None
        for row, q in zip(hrows, entry["questions"]):
            assert row.qid == q["qid"] and row.ids == q["ids"] and row.markers == q["markers"], (entry["id"], q["qid"])
            L = host.bucket_for(row.positions)
            t1 = time.perf_counter()
            scores, markers = run_row(models.get(L), row, prefix)
            seconds = time.perf_counter() - t1
            z = scores[0, markers].numpy()
            p = host.probabilities_from_logits(z, row.question, row.calibrate)
            dlogit = float(np.max(np.abs(z.astype(np.float64) - np.asarray(q["logits_raw"], dtype=np.float64))))
            dp = float(max(abs(a - b) for a, b in zip(p, q["probs"])))
            key = (entry["id"], q["qid"], entry["mode"])
            row_cache[key] = {"row": row, "prefix": prefix, "L": L, "scores": scores.clone(), "markers": markers, "ref": q}
            rows_out.append({"id": entry["id"], "qid": q["qid"], "mode": entry["mode"], "source": entry["source"],
                             "type": q["type"], "K": q["K"], "positions": row.positions, "bucket": L,
                             "max_abs_dlogit": dlogit, "max_abs_dp": dp,
                             "argmax_equal": int(np.argmax(p)) == q["argmax_index"], "near_tie": q["near_tie"],
                             "ref_top2_margin": q["top2_margin"], "seconds": round(seconds, 4)})
    rows_seconds = time.perf_counter() - t_rows
    print(f"rows: {len(rows_out)} in {rows_seconds:.1f} s; max |dlogit| {max(r['max_abs_dlogit'] for r in rows_out):.3e} "
          f"max |dp| {max(r['max_abs_dp'] for r in rows_out):.3e} argmax {sum(r['argmax_equal'] for r in rows_out)}/"
          f"{len(rows_out)}", flush=True)

    # (a) trunk output vs the reference's (npz rows)
    hidden = []
    for key, saved in sorted(npz_hidden.items()):
        c = row_cache[key]
        model = models.get(c["L"])
        with trunk_output(model) as box:
            scores, _ = run_row(model, c["row"], c["prefix"])
        assert torch.equal(scores, c["scores"])
        n = c["row"].positions
        mine = box["h"][0, :n].numpy().astype(np.float64)
        theirs = saved["hidden"].astype(np.float64)
        assert theirs.shape == mine.shape, (key, theirs.shape, mine.shape)
        diff = np.abs(mine - theirs)
        pos = int(np.unravel_index(np.argmax(diff), diff.shape)[0])
        ref_pos = int(np.unravel_index(np.argmax(np.abs(theirs)), theirs.shape)[0])
        p = saved["prefix_len"]
        hidden.append({"id": key[0], "qid": key[1], "mode": key[2], "file": saved["file"], "positions": n, "prefix": p,
                       "bucket": c["L"], "max_abs_d": float(diff.max()), "max_abs_ref": float(np.abs(theirs).max()),
                       "relative": float(diff.max() / np.abs(theirs).max()), "argmax_d_position": pos,
                       "argmax_ref_position": ref_pos,
                       "max_abs_d_text_positions": float(diff[p:].max()),
                       "max_abs_d_prefix_positions": float(diff[:p].max()) if p else None,
                       "max_abs_d_excluding_position_0": float(diff[1:].max()),
                       "mean_abs_d": float(diff.mean())})
    print("(a) hidden: max |d|", max(h["max_abs_d"] for h in hidden), "max rel", max(h["relative"] for h in hidden),
          flush=True)

    # (b) pad content
    gen = np.random.default_rng(SEED)

    def pad_ids(value):
        def mutate(inputs, row):
            n = row.positions
            inputs["input_ids"][0, n:] = value if value is not None else gen.integers(0, 65536, inputs["input_ids"].shape[1] - n)
        return mutate

    def pad_embeds(inputs, row):
        n = row.positions
        inputs["prefix_embeds"][0, n:] = gen.standard_normal((inputs["prefix_embeds"].shape[1] - n, 1024)).astype(np.float32)

    def nonprefix_embeds(inputs, row):
        p = row.prefix_len
        inputs["prefix_embeds"][0, p:] = gen.standard_normal((inputs["prefix_embeds"].shape[1] - p, 1024)).astype(np.float32)

    variants = {"pad_ids_1": pad_ids(1), "pad_ids_100": pad_ids(100), "pad_ids_random": pad_ids(None),
                "prefix_embeds_random_at_pad": pad_embeds, "prefix_embeds_random_at_non_prefix": nonprefix_embeds}
    pad_rows = sorted(npz_hidden)
    pad_rows = [k for k in pad_rows if not (k[2] != "text" and k[1] != "topic")] + resolve(PAD_ROWS_EXTRA)
    pad_rows = list(dict.fromkeys(pad_rows))
    pad = []
    for key in pad_rows:
        c = row_cache[key]
        n = c["row"].positions
        out = {"id": key[0], "qid": key[1], "mode": key[2], "positions": n, "bucket": c["L"], "pad_positions": c["L"] - n}
        for name, mutate in variants.items():
            scores, _ = run_row(models.get(c["L"]), c["row"], c["prefix"], mutate)
            out[name] = bool(torch.equal(scores[0, :n], c["scores"][0, :n]))
        pad.append(out)
    pad_ok = all(all(v for k, v in r.items() if k in variants) for r in pad)
    print(f"(b) pad content: {len(pad)} rows, bit-identical {pad_ok}", flush=True)

    # (c) bucket: the same row at two lengths
    bucket = []
    for key, other in [(k, 512) for k in resolve(BUCKET_ROWS_256)] + [(k, 4096) for k in resolve(BUCKET_ROWS_2048)]:
        c = row_cache[key]
        assert c["L"] < other, (key, c["L"], other)
        scores, markers = run_row(models.get(other), c["row"], c["prefix"])
        a, b = scores[0, markers], c["scores"][0, c["markers"]]
        n = c["row"].positions
        bucket.append({"id": key[0], "qid": key[1], "mode": key[2], "positions": n, "lengths": [c["L"], other],
                       "marker_max_abs_d": float((a - b).abs().max()), "marker_bit_identical": bool(torch.equal(a, b)),
                       "real_scores_max_abs_d": float((scores[0, :n] - c["scores"][0, :n]).abs().max())})
    print("(c) bucket: marker max |d|", max(r["marker_max_abs_d"] for r in bucket), flush=True)

    # (d) GQA forms at full size
    gqa = []
    for key in resolve(GQA_ROWS):
        c = row_cache[key]
        scores, markers = run_row(models.get(c["L"], gqa="broadcast"), c["row"], c["prefix"])
        a, b = scores[0, markers], c["scores"][0, c["markers"]]
        n = c["row"].positions
        gqa.append({"id": key[0], "qid": key[1], "mode": key[2], "bucket": c["L"],
                    "marker_max_abs_d": float((a - b).abs().max()), "marker_bit_identical": bool(torch.equal(a, b)),
                    "real_scores_bit_identical": bool(torch.equal(scores[0, :n], c["scores"][0, :n]))})
    print("(d) GQA broadcast vs repeat: bit-identical", sum(r["real_scores_bit_identical"] for r in gqa), "/", len(gqa),
          "max |d|", max(r["marker_max_abs_d"] for r in gqa), flush=True)

    # (e) mutations on the prefix rows
    mutations = {}
    for mutation in ("keep_right_ones", "no_media_text_mask", "no_head_key_mask"):
        per_row = []
        for key in PREFIX_ROWS:
            c = row_cache[key]
            scores, markers = run_row(models.get(c["L"], mutation=mutation), c["row"], c["prefix"])
            z = scores[0, markers].numpy()
            p = host.probabilities_from_logits(z, c["row"].question, c["row"].calibrate)
            mine = host.probabilities_from_logits(c["scores"][0, c["markers"]].numpy(), c["row"].question, c["row"].calibrate)
            per_row.append({"id": key[0], "qid": key[1], "mode": key[2],
                            "max_abs_dp_vs_reference": float(max(abs(a - b) for a, b in zip(p, c["ref"]["probs"]))),
                            "max_abs_dp_vs_unmutated": float(max(abs(a - b) for a, b in zip(p, mine))),
                            "max_abs_dlogit_vs_reference": float(np.max(np.abs(z - np.asarray(c["ref"]["logits_raw"])))),
                            "argmax_equal_reference": int(np.argmax(p)) == c["ref"]["argmax_index"]})
        mutations[mutation] = {"rows": per_row, "max_abs_dp": max(r["max_abs_dp_vs_reference"] for r in per_row),
                               "rows_over_bar": sum(r["max_abs_dp_vs_reference"] > MUTATION_BAR for r in per_row),
                               "max_abs_dlogit": max(r["max_abs_dlogit_vs_reference"] for r in per_row),
                               "min_abs_dlogit": min(r["max_abs_dlogit_vs_reference"] for r in per_row),
                               "rows_over_logit_bar": sum(r["max_abs_dlogit_vs_reference"] > BARS["max_abs_dlogit"]
                                                          for r in per_row)}
    print("(e) mutations:", {k: round(v["max_abs_dp"], 4) for k, v in mutations.items()}, flush=True)

    # summary
    max_dp = max(r["max_abs_dp"] for r in rows_out)
    max_dl = max(r["max_abs_dlogit"] for r in rows_out)
    argmax_all = all(r["argmax_equal"] for r in rows_out)
    worst = {}
    by_source = defaultdict(list)
    for r in rows_out:
        by_source[f"{r['source']}/{r['mode']}"].append(r)
    for s, rs in sorted(by_source.items()):
        wp = max(rs, key=lambda r: r["max_abs_dp"])
        wl = max(rs, key=lambda r: r["max_abs_dlogit"])
        worst[s] = {"rows": len(rs), "argmax_equal": sum(r["argmax_equal"] for r in rs),
                    "max_abs_dp": wp["max_abs_dp"], "max_abs_dp_row": f"{wp['id']}/{wp['qid']}",
                    "max_abs_dlogit": wl["max_abs_dlogit"], "max_abs_dlogit_row": f"{wl['id']}/{wl['qid']}"}
    by_bucket = defaultdict(list)
    for r in rows_out:
        by_bucket[r["bucket"]].append(r)
    buckets = {str(b): {"rows": len(rs), "max_abs_dp": max(r["max_abs_dp"] for r in rs),
                        "max_abs_dlogit": max(r["max_abs_dlogit"] for r in rs),
                        "seconds_p50": pct([r["seconds"] for r in rs], 0.5), "seconds_max": max(r["seconds"] for r in rs)}
               for b, rs in sorted(by_bucket.items())}
    ties = [r for r in rows_out if r["near_tie"]]
    verdict = {"argmax": argmax_all, "max_abs_dp": max_dp <= BARS["max_abs_dp"],
               "max_abs_dlogit": max_dl <= BARS["max_abs_dlogit"], "b_pad_content_bit_identical": pad_ok,
               "e_mutations_over_bar": {k: v["max_abs_dp"] > MUTATION_BAR for k, v in mutations.items()},
               "e_mutations_over_logit_bar_every_row": {k: v["rows_over_logit_bar"] == len(v["rows"])
                                                        for k, v in mutations.items()}}
    doc = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "reference": {"path": str(ref_path), "sha256": sha256_file(ref_path), "rows": len(rows_out),
                      "skipped": "img_01..03 image mode (9 rows): no picture files (the reference skipped them)"},
        "weights": load_record, "load_seconds": round(load_seconds, 2), "bars": BARS, "bar_basis": bar_basis(),
        "mutation_bar": MUTATION_BAR,
        "pass": argmax_all and max_dp <= BARS["max_abs_dp"] and max_dl <= BARS["max_abs_dlogit"],
        "verdict": verdict,
        "summary": {"rows": len(rows_out), "argmax_equal": sum(r["argmax_equal"] for r in rows_out),
                    "max_abs_dp": max_dp, "max_abs_dlogit": max_dl,
                    "p99_abs_dp": pct([r["max_abs_dp"] for r in rows_out], 0.99),
                    "p99_abs_dlogit": pct([r["max_abs_dlogit"] for r in rows_out], 0.99),
                    "near_ties": {"rows": len(ties), "argmax_equal": sum(r["argmax_equal"] for r in ties),
                                  "max_abs_dp": max((r["max_abs_dp"] for r in ties), default=None)},
                    "by_source_mode": worst, "by_bucket": buckets, "rows_seconds": round(rows_seconds, 1)},
        "a_hidden": {"rows": len(hidden), "max_abs_d": max(h["max_abs_d"] for h in hidden),
                     "max_relative": max(h["relative"] for h in hidden), "per_row": hidden},
        "b_pad_content": {"rows": len(pad), "variants": list(variants), "all_bit_identical": pad_ok, "per_row": pad},
        "c_bucket": {"rows": len(bucket), "marker_max_abs_d": max(r["marker_max_abs_d"] for r in bucket),
                     "bit_identical_rows": sum(r["marker_bit_identical"] for r in bucket), "per_row": bucket},
        "d_gqa": {"rows": len(gqa), "real_scores_bit_identical_rows": sum(r["real_scores_bit_identical"] for r in gqa),
                  "marker_max_abs_d": max(r["marker_max_abs_d"] for r in gqa), "per_row": gqa},
        "e_mutations": mutations,
        "rows": rows_out,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    out = WORK / "results" / "eager_check.json"
    write_atomic(out, (json.dumps(doc, indent=1) + "\n").encode())
    print(json.dumps({"pass": doc["pass"], "verdict": verdict,
                      **{k: doc["summary"][k] for k in ("rows", "argmax_equal", "max_abs_dp", "max_abs_dlogit",
                                                        "p99_abs_dp", "p99_abs_dlogit", "near_ties")},
                      "elapsed_s": doc["elapsed_s"]}, indent=1))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
