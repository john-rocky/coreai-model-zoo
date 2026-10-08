#!/usr/bin/env python3
"""The key-block attention (d1_omni_model, "Key blocks") against the plain form, in torch on the CPU.

    python3 conversion/d1_omni/block_softmax_check.py [--out results/block_softmax_check.v2.json]
        # default -> $ZOO_WORK_ROOT/_d1_omni/results/block_softmax_check.json

Above KEY_BLOCK (2048) keys every attention of D1Decide runs in key blocks sharing one max; at L <= 2048 the graph is
unchanged. The blocked softmax is the plain one summed in another order: the same function, so in float64 the two
forms agree to float64 rounding, and in fp32 each sits about as far from the exact value as the other. Measured here:

  toy        random toy64 modules (hidden 64, 4 heads over 2 key/value heads, one attention layer, the 2-layer head)
             at L=256 with key_block 64 (4 blocks) and at L=4096 with key_block 2048 (2 blocks), both GQA forms, in
             float64, fp32 and fp16 compute: every real position's score, blocked vs plain
  full       the checkpoint's module (sha256-verified weights) at L=4096, blocked (key_block 2048) vs plain (key_block
             above L) in fp32 on the 4 long_3400 rows (bucket 4096), the 11 bucket-2048 rows and 5 bucket-256 rows (4
             text, 1 audio prefix) padded to 4096, and card_cats (image prefix, bucket 512): marker logits and
             real-position scores; each form against the publisher's fp32 model (ref/records_ref.json); the 2048 rows
             also against the same module at L=2048 (bucket invariance)
  full64     both forms in float64 (every fp32 step of the module in float64: round 2's dp_float64_pure.py frame) on
             the 6 rows with the largest fp32 blocked-vs-plain gap and the 2 with the smallest; each fp32 form against
             its own float64 run (the fp32 rounding of each form)

Bars (supervisor's ruling, round 4, 2026-10-08): the same function = float64 blocked vs plain, marker logits max |d|
<= 1e-12 (toy and full64); blocked fp32 vs the reference = round 2's eager bar (argmax on every row, max |dp| <= 2e-5,
marker logits <= 1e-3). The fp32 blocked-vs-plain gap is recorded, not a bar: the plain form itself sits about 3e-5
(marker logits) from its float64 run on these rows, so no blocked form can hold the two fp32 forms within 1e-5 (the
first run's bar). Run with the reference venv (torch 2.9.0).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _fixtures import fixtures_path as fixtures_path_for  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402

WORK = work_path("_d1_omni")
OUT = WORK / "results" / "block_softmax_check.json"
REFERENCE_SHA256 = "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"  # ref/records_ref.json (round 2)
SEED = 20261008
PLAIN = 1 << 30  # a key_block no length reaches: the plain form at any L
BAR_SAME_FUNCTION_F64 = 1e-12
EAGER_BAR = {"argmax": "every row", "max_abs_dp": 2e-5, "max_abs_dlogit": 1e-3}
F64_ROWS = (6, 2)  # the rows with the largest and the smallest fp32 gap that also run in float64
INPUT_NAMES = ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot")


def to_float64(m: dm.D1Decide) -> dm.D1Decide:
    """Every step in float64: compute and softmax dtypes, and RMSNorm / LayerNorm / the scorer (which normalise in fp32
    by design) re-run in float64. The caller puts the parameters in float64."""
    m.compute_dtype = m.softmax_dtype = torch.float64

    def rms(norm, x):
        return norm.weight.to(x.dtype) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps))

    def ln(norm, x):
        return F.layer_norm(x, norm.normalized_shape, norm.weight.to(x.dtype), norm.bias.to(x.dtype), norm.eps)

    def scorer(h, m=m):
        norm, first, _, last = m.head.scorer
        return m._lin(F.gelu(m._lin(ln(norm, h), first)), last).reshape(1, m.seq_len)

    m._rms, m._ln, m._scorer = rms, ln, scorer
    return m
TOY = {"vocab_size": 128, "hidden_size": 64, "intermediate_size": 192, "num_hidden_layers": 4,
       "num_attention_heads": 4, "num_key_value_heads": 2, "block_multiple_of": 32, "norm_eps": 1e-05,
       "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "rope_theta": 1000000.0,
       "layer_types": ["conv", "conv", "full_attention", "conv"]}
# (seq_len, key_block, rows as (prefix positions, text positions, markers))
TOY_CASES = [(256, 64, [(0, 40, [10, 20, 30]), (0, 130, [50, 90]), (0, 250, [100, 180, 240]),
                        (20, 200, [60, 150]), (100, 150, [30, 120, 140])]),
             (4096, 2048, [(0, 600, [100, 400]), (0, 2500, [1000, 2400]), (0, 4000, [1500, 2100, 3900]),
                           (300, 3000, [500, 2900]), (1900, 2100, [1000, 2050])])]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- toy
def randomize(model: dm.D1Decide, gen: torch.Generator) -> None:
    def normal(shape, scale):
        return torch.randn(shape, generator=gen) * scale
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, dm.RMSNorm)):
            m.weight.data = 1.0 + normal(m.weight.shape, 0.2)
            if getattr(m, "bias", None) is not None:
                m.bias.data = normal(m.bias.shape, 0.1)
        elif isinstance(m, nn.Linear):
            m.weight.data = normal(m.weight.shape, m.weight.shape[1] ** -0.5)
            if m.bias is not None:
                m.bias.data = normal(m.bias.shape, 0.1)
        elif isinstance(m, nn.Embedding):
            m.weight.data = normal(m.weight.shape, 1.0 if m.weight.shape[0] > 3 else 0.5)
        elif isinstance(m, dm._DepthwiseFilter):
            m.weight.data = normal(m.weight.shape, 0.5)
        elif isinstance(m, dm._HeadAttention):
            m.in_proj_weight.data = normal(m.in_proj_weight.shape, m.in_proj_weight.shape[1] ** -0.5)
            m.in_proj_bias.data = normal(m.in_proj_bias.shape, 0.1)


def toy_inputs(L: int, p: int, n: int, markers: list[int], qtype: int, d: int, vocab: int, gen: torch.Generator):
    ids = torch.randint(22, vocab, (n,), generator=gen)
    ids[0], ids[1], ids[2], ids[-1] = 1, 17, 18, 21
    for m in markers:
        ids[m - 1], ids[m] = 19, 16
    input_ids = torch.zeros(1, L, dtype=torch.int32)
    input_ids[0, p:p + n] = ids.to(torch.int32)
    prefix_embeds = torch.zeros(1, L, d)
    if p:
        prefix_embeds[0, :p] = torch.randn(p, d, generator=gen)
    pad = torch.zeros(1, L)
    pad[0, :p + n] = 1.0
    prefix = torch.zeros(1, L)
    prefix[0, :p] = 1.0
    keep = torch.ones(1, L)
    if p:
        keep[0, p - 1] = 0.0
    onehot = torch.zeros(1, 3)
    onehot[0, qtype] = 1.0
    return (input_ids, prefix_embeds, pad, prefix, keep, onehot), [p + m for m in markers]


def run_toy() -> dict:
    out = {"config": TOY, "head_layers": 2, "cases": []}
    for L, key_block, rows in TOY_CASES:
        for gqa in dm.GQA_FORMS:
            for precision in ("float64", "fp32", "fp16"):
                gen = torch.Generator().manual_seed(SEED + L)
                plain = dm.D1Decide(TOY, 2, L, gqa=gqa, key_block=PLAIN).eval()
                randomize(plain, gen)
                blocked = dm.D1Decide(TOY, 2, L, gqa=gqa, key_block=key_block).eval()
                blocked.load_state_dict(plain.state_dict(), strict=True)
                if precision == "float64":
                    plain, blocked = to_float64(plain.double()), to_float64(blocked.double())
                else:
                    plain.set_precision(precision)
                    blocked.set_precision(precision)
                records = []
                for i, (p, n, markers) in enumerate(rows):
                    args, marker_pos = toy_inputs(L, p, n, markers, i % 3, TOY["hidden_size"], TOY["vocab_size"], gen)
                    if precision == "float64":
                        args = (args[0], args[1].double(), *args[2:])
                    with torch.no_grad():
                        a = plain(*args)[0, :p + n].double()
                        b = blocked(*args)[0, :p + n].double()
                    records.append({"prefix": p, "text": n, "markers": markers,
                                    "max_abs_d_scores": float((a - b).abs().max()),
                                    "max_abs_d_markers": float((a[marker_pos] - b[marker_pos]).abs().max()),
                                    "max_abs_scores": float(a.abs().max()),
                                    "bit_identical": bool(torch.equal(a, b))})
                case = {"seq_len": L, "key_block": key_block, "blocks": -(-L // key_block), "gqa": gqa,
                        "precision": precision, "rows": records,
                        "max_abs_d_scores": max(r["max_abs_d_scores"] for r in records),
                        "bit_identical_rows": sum(r["bit_identical"] for r in records)}
                if precision == "float64":
                    case["status"] = "PASS" if case["max_abs_d_scores"] <= BAR_SAME_FUNCTION_F64 else "FAIL"
                else:
                    case["status"] = "recorded (no bar)"
                out["cases"].append(case)
                print(f"toy L={L} kb={key_block} {gqa} {precision}: max |d| {case['max_abs_d_scores']:.3e}, "
                      f"bit-identical {case['bit_identical_rows']}/{len(records)}", flush=True)
    f64_cases = [c for c in out["cases"] if c["precision"] == "float64"]
    out["status"] = "PASS" if all(c["status"] == "PASS" for c in f64_cases) else "FAIL"
    out["max_abs_d_scores_float64"] = max(c["max_abs_d_scores"] for c in f64_cases)
    out["max_abs_d_scores_fp32"] = max(c["max_abs_d_scores"] for c in out["cases"] if c["precision"] == "fp32")
    return out


# --------------------------------------------------------------------------- full model
def load_reference(tok) -> tuple[dict, dict, dict]:
    ref_path = WORK / "ref" / "records_ref.json"
    if sha256_file(ref_path) != REFERENCE_SHA256:
        raise SystemExit(f"{ref_path} is not round 2's reference")
    ref = json.loads(ref_path.read_text())
    fixtures_path = fixtures_path_for(ref["fixtures"]["sha256"])  # the version the reference read (_fixtures.py)
    fixtures = {r["id"]: r for r in json.loads(fixtures_path.read_text())["records"]}
    rows = {}
    for entry in ref["records"]:
        req = fixtures[entry["id"]]["request"]
        built = host.request_rows(tok, req["state"], req["questions"], entry["mode"], entry["prefix"])
        prefix = None
        if entry["mode"] != "text":
            meta = ref["npz"][entry["id"]]
            path = WORK / "ref" / "npz" / f"{entry['id']}.npz"
            if sha256_file(path) != meta["sha256"]:
                raise SystemExit(f"{path} differs from the reference's npz")
            with np.load(path) as z:
                prefix = np.asarray(z["prefix"], dtype=np.float32).copy()
        for row, q in zip(built, entry["questions"], strict=True):
            assert row.qid == q["qid"] and row.ids == q["ids"] and row.markers == q["markers"], (entry["id"], q["qid"])
            rows[(entry["id"], q["qid"], entry["mode"])] = {"row": row, "q": q, "prefix": prefix,
                                                             "source": entry["source"]}
    return ref, fixtures, rows


def pick_rows(rows: dict) -> list[tuple]:
    by_bucket = {}
    for key, r in rows.items():
        by_bucket.setdefault(r["q"]["bucket"], []).append(key)
    text256 = [k for k in by_bucket[256] if k[2] == "text"]
    picked_256 = [text256[round(i * (len(text256) - 1) / 3)] for i in range(4)]
    picked_256.append(next(k for k in by_bucket[256] if k[0] == "card_audio"))
    image512 = [k for k in by_bucket[512] if k[2] == "image"]
    return by_bucket[4096] + by_bucket[2048] + picked_256 + image512


def oracle_record(z: np.ndarray, r: dict) -> dict:
    q, row = r["q"], r["row"]
    p = host.probabilities_from_logits(z, row.question, row.calibrate)
    return {"max_abs_dp": float(max(abs(a - b) for a, b in zip(p, q["probs"]))),
            "max_abs_dlogit": float(np.max(np.abs(z.astype(np.float64) - np.asarray(q["logits_raw"], np.float64)))),
            "argmax_equal": int(np.argmax(p)) == q["argmax_index"]}


def run_full() -> dict:
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok)
    config = json.loads((snapshot / "config.json").read_text())
    _, _, rows = load_reference(tok)
    t0 = time.perf_counter()
    blocked, load_record = dm.load_d1_decide(snapshot, 4096, "fp32", verify_sha256=True)
    load_s = time.perf_counter() - t0
    assert blocked.key_block == dm.KEY_BLOCK == 2048
    state = blocked.state_dict()
    others = {}
    for name, (L, kb) in {"plain4096": (4096, PLAIN), "native2048": (2048, dm.KEY_BLOCK)}.items():
        m = dm.build(config, L, key_block=kb)
        m.load_state_dict(state, strict=True, assign=True)
        others[name] = m.eval().requires_grad_(False)
    plain, native = others["plain4096"], others["native2048"]
    records = []
    for key in pick_rows(rows):
        r = rows[key]
        row = r["row"]
        n = row.positions
        inputs, markers = host.graph_inputs(row, 4096, r["prefix"])
        args = [torch.from_numpy(inputs[k]) for k in ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask",
                                                      "keep_right", "qtype_onehot")]
        t1 = time.perf_counter()
        with torch.no_grad():
            sb = blocked(*args)[0].numpy()
        tb = time.perf_counter() - t1
        t1 = time.perf_counter()
        with torch.no_grad():
            sp = plain(*args)[0].numpy()
        tp = time.perf_counter() - t1
        zb, zp = sb[markers], sp[markers]
        rec = {"id": key[0], "qid": key[1], "mode": key[2], "source": r["source"], "positions": n,
               "bucket": r["q"]["bucket"], "near_tie": r["q"]["near_tie"], "K": len(markers),
               "blocked_vs_plain": {
                   "max_abs_d_markers": float(np.max(np.abs(zb.astype(np.float64) - zp))),
                   "max_abs_d_scores": float(np.max(np.abs(sb[:n].astype(np.float64) - sp[:n]))),
                   "bit_identical_scores": bool(np.array_equal(sb[:n], sp[:n]))},
               "blocked_vs_reference": oracle_record(zb, r), "plain_vs_reference": oracle_record(zp, r),
               "logits_blocked": [float(v) for v in zb], "logits_plain": [float(v) for v in zp],
               "seconds": {"blocked": tb, "plain": tp}}
        if r["q"]["bucket"] == 2048:
            inputs2, markers2 = host.graph_inputs(row, 2048, r["prefix"])
            args2 = [torch.from_numpy(inputs2[k]) for k in ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask",
                                                            "keep_right", "qtype_onehot")]
            with torch.no_grad():
                s2 = native(*args2)[0].numpy()
            rec["blocked4096_vs_native2048_max_abs_d_markers"] = float(
                np.max(np.abs(zb.astype(np.float64) - s2[markers2])))
        records.append(rec)
        print(f"{key[0]}/{key[1]} ({key[2]}, {n} pos): blocked-plain {rec['blocked_vs_plain']['max_abs_d_markers']:.2e}"
              f" | ref dp {rec['blocked_vs_reference']['max_abs_dp']:.2e} dlogit "
              f"{rec['blocked_vs_reference']['max_abs_dlogit']:.2e} argmax {rec['blocked_vs_reference']['argmax_equal']}"
              f" | {tb:.1f} s / {tp:.1f} s", flush=True)

    def agg(side: str) -> dict:
        vals = [x[side] for x in records]
        return {"rows": len(vals), "argmax_equal": sum(v["argmax_equal"] for v in vals),
                "max_abs_dp": max(v["max_abs_dp"] for v in vals),
                "max_abs_dlogit": max(v["max_abs_dlogit"] for v in vals)}

    vs_ref = agg("blocked_vs_reference")
    eager_pass = (vs_ref["argmax_equal"] == vs_ref["rows"] and vs_ref["max_abs_dp"] <= EAGER_BAR["max_abs_dp"]
                  and vs_ref["max_abs_dlogit"] <= EAGER_BAR["max_abs_dlogit"])
    bvp = max(x["blocked_vs_plain"]["max_abs_d_markers"] for x in records)
    inv = [x["blocked4096_vs_native2048_max_abs_d_markers"] for x in records if "blocked4096_vs_native2048_max_abs_d_markers" in x]

    # full64: both forms in float64 on the rows with the largest / smallest fp32 gap, and each fp32 form against it
    ranked = sorted(records, key=lambda x: -x["blocked_vs_plain"]["max_abs_d_markers"])
    chosen = ranked[:F64_ROWS[0]] + ranked[-F64_ROWS[1]:]
    state64 = {k: v.to(torch.float64) for k, v in state.items()}
    f64 = {}
    for name, kb in (("blocked", dm.KEY_BLOCK), ("plain", PLAIN)):
        m = dm.build(config, 4096, key_block=kb)
        m.load_state_dict(state64, strict=True, assign=True)
        f64[name] = to_float64(m.eval().requires_grad_(False))
    full64 = []
    for x in chosen:
        key = (x["id"], x["qid"], x["mode"])
        r = rows[key]
        inputs, markers = host.graph_inputs(r["row"], 4096, r["prefix"])
        args = [torch.from_numpy(inputs[k]) for k in INPUT_NAMES]
        args[1] = args[1].double()
        with torch.no_grad():
            zb64 = f64["blocked"](*args)[0, markers].numpy()
            zp64 = f64["plain"](*args)[0, markers].numpy()
        zb32, zp32 = np.asarray(x["logits_blocked"]), np.asarray(x["logits_plain"])
        full64.append({"id": x["id"], "qid": x["qid"], "mode": x["mode"], "positions": x["positions"],
                       "blocked64_vs_plain64": float(np.max(np.abs(zb64 - zp64))),
                       "blocked32_vs_blocked64": float(np.max(np.abs(zb32 - zb64))),
                       "plain32_vs_plain64": float(np.max(np.abs(zp32 - zp64))),
                       "blocked32_vs_plain32": x["blocked_vs_plain"]["max_abs_d_markers"]})
        print(f"float64 {x['id']}/{x['qid']}: blocked-plain {full64[-1]['blocked64_vs_plain64']:.2e}; fp32 to float64 "
              f"blocked {full64[-1]['blocked32_vs_blocked64']:.2e} plain {full64[-1]['plain32_vs_plain64']:.2e}", flush=True)
    same64 = max(y["blocked64_vs_plain64"] for y in full64)
    return {
        "status": "PASS" if (eager_pass and same64 <= BAR_SAME_FUNCTION_F64) else "FAIL",
        "seq_len": 4096, "key_block": dm.KEY_BLOCK, "precision": "fp32 (torch CPU), float64 on the full64 rows",
        "rows": len(records), "rows_by_bucket": {str(b): sum(x["bucket"] == b for x in records) for b in (256, 512, 2048, 4096)},
        "same_function_float64": {"bar_max_abs_d_markers": BAR_SAME_FUNCTION_F64, "max_abs_d_markers": same64,
                                  "status": "PASS" if same64 <= BAR_SAME_FUNCTION_F64 else "FAIL", "rows": full64,
                                  "fp32_to_float64_max": {
                                      "blocked": max(y["blocked32_vs_blocked64"] for y in full64),
                                      "plain": max(y["plain32_vs_plain64"] for y in full64)}},
        "blocked_vs_plain": {"bar": "recorded, not a bar (fp32 rounding floor)", "max_abs_d_markers": bvp,
                             "max_abs_d_scores": max(x["blocked_vs_plain"]["max_abs_d_scores"] for x in records),
                             "bit_identical_rows": sum(x["blocked_vs_plain"]["bit_identical_scores"] for x in records)},
        "blocked_vs_reference": {**vs_ref, "bar": EAGER_BAR, "status": "PASS" if eager_pass else "FAIL"},
        "plain_vs_reference": agg("plain_vs_reference"),
        "bucket_invariance_2048_rows": {"rows": len(inv), "max_abs_d_markers": max(inv) if inv else None,
                                        "note": "blocked graph at 4096 vs the plain graph at 2048 (the RoPE table is "
                                                "built per length: round 2 measured 3.3e-5 between buckets)"},
        "load_s": load_s, "weights": load_record, "records": records,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="result file (relative to the work dir, or absolute)")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else WORK / args.out
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    if out.exists():
        raise SystemExit(f"{out} exists: an existing result is never replaced")
    toy = run_toy()
    full = run_full()
    result = {
        "status": "PASS" if toy["status"] == full["status"] == "PASS" else "FAIL",
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "what": "D1Decide's key-block attention (above 2048 keys) against the plain form, torch CPU (fp32, float64)",
        "bar": {"same_function": f"float64: blocked vs plain marker logits max |d| <= {BAR_SAME_FUNCTION_F64:g} (toy, full64)",
                "blocked_vs_reference": "fp32: round 2's eager bar", "blocked_vs_plain_fp32": "recorded, not a bar"},
        "toy": toy, "full": full,
        "code_sha256": {f: sha256_file(HERE / f) for f in ("block_softmax_check.py", "d1_omni_model.py", "host.py")},
        "environment": {"python": platform.python_version(), "executable": sys.executable, "torch": torch.__version__,
                        "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "pid": os.getpid()},
        "seconds": time.perf_counter() - started,
    }
    out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"{result['status']}: toy float64 {toy['max_abs_d_scores_float64']:.3e} (fp32 {toy['max_abs_d_scores_fp32']:.3e}); "
          f"full float64 {full['same_function_float64']['max_abs_d_markers']:.3e}, fp32 blocked-plain "
          f"{full['blocked_vs_plain']['max_abs_d_markers']:.3e}, vs reference max |dp| "
          f"{full['blocked_vs_reference']['max_abs_dp']:.3e} max |dlogit| {full['blocked_vs_reference']['max_abs_dlogit']:.3e}"
          f" argmax {full['blocked_vs_reference']['argmax_equal']}/{full['rows']} -> {out}", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
