#!/usr/bin/env python3
"""d1_omni_model.D1Decide against the publisher's encoder.py (Trunk + DecisionHead) on small random configurations,
CPU fp32, no weights. The publisher's side is run the way its `D1OmniModel._forward` runs it (pad_sequence, the
trunk over [prefix, text], the head over the text positions, under no_grad in eval mode); this graph's side is
driven through host.graph_inputs.

    python3 conversion/d1_omni/toy_module_check.py   # -> $ZOO_WORK_ROOT/_d1_omni/results/toy_parity.json

Two configurations: `toy64` (hidden 64, layers conv/conv/attention/conv, 4 query / 2 key-value heads, vocab 128;
its head has 64 // 64 = 1 head, so PyTorch's encoder-layer fast path is off: odd head count) and `toy256` (hidden
256, head_dim 64 as in the checkpoint, 6 layers with 2 attentions, a 4-head head: the fast path is on, as in the
publisher's 16-head model). Every parameter is drawn from a seeded distribution (norm weights 1 + 0.2 N, biases
0.1 N, ...) so none is left at a value that hides its use, and (h) shows that perturbing each one moves the output.

  (a) text rows (7 / 12 / 20 ids, L = 24, one row per question type): marker logits, max |d| <= 1e-5
  (b) prefix rows (5 + 9, 3 + 15): the same bar
  (c) pad content: pad ids 1 / 100 and random prefix_embeds at pad positions leave every marker logit bit-identical
  (d) batch: the publisher's B = 3 forward vs one row at a time, max |d| <= 1e-6 (text rows; prefix + text mix)
  (e) mask value: the publisher's NEG -1e9 -> -1e4 (module attribute) leaves its outputs bit-identical
  (f) negative controls: keep_right all ones / no media x text mask / no head key mask move the prefix rows'
      marker logits by more than 1e-3 (the instrument can go red); their effect on text rows is recorded
  (g) GQA as repeat_interleave vs reshape-broadcast: recorded (max |d|, bit-identical or not)
  (h) parameter coverage: each parameter tensor perturbed alone moves some marker logit (no dead parameter)
  (i) the publisher's head through PyTorch's fast path vs the plain path (recorded)
  (j) the full-size module (meta device) has exactly the checkpoint header's trunk / head keys and shapes, and the
      RoPE tables equal the publisher's Trunk.rope at L 24 and 4096
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402
from _paths import code_path, hf_snapshot, work_path  # noqa: E402

WORK = work_path("_d1_omni")
BASE = {"norm_eps": 1e-05, "conv_L_cache": 3, "block_ffn_dim_multiplier": 1.0, "rope_theta": 1000000.0}
TOYS = {
    "toy64": {**BASE, "vocab_size": 128, "hidden_size": 64, "intermediate_size": 192, "num_hidden_layers": 4,
              "num_attention_heads": 4, "num_key_value_heads": 2, "block_multiple_of": 32,
              "layer_types": ["conv", "conv", "full_attention", "conv"]},
    "toy256": {**BASE, "vocab_size": 128, "hidden_size": 256, "intermediate_size": 768, "num_hidden_layers": 6,
               "num_attention_heads": 4, "num_key_value_heads": 2, "block_multiple_of": 32,
               "layer_types": ["conv", "conv", "full_attention", "conv", "full_attention", "conv"]},
}
L = 24
SEED = 20261008
BARS = {"a": 1e-5, "b": 1e-5, "d": 1e-6, "f": 1e-3}
MUTATIONS = ("keep_right_ones", "no_media_text_mask", "no_head_key_mask")


def load_publisher_encoder(snapshot: Path):
    spec = importlib.util.spec_from_file_location("d1_upstream_encoder", snapshot / "encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def randomize(modules: list[nn.Module], generator: torch.Generator, rms_type) -> None:
    """Every parameter from a seeded distribution that makes its use visible."""
    def normal(shape, scale):
        return torch.randn(shape, generator=generator) * scale
    for root in modules:
        for m in root.modules():
            if isinstance(m, (nn.LayerNorm, rms_type)):
                m.weight.data = 1.0 + normal(m.weight.shape, 0.2)
                if getattr(m, "bias", None) is not None:
                    m.bias.data = normal(m.bias.shape, 0.1)
            elif isinstance(m, nn.Linear):
                m.weight.data = normal(m.weight.shape, m.weight.shape[1] ** -0.5)
                if m.bias is not None:
                    m.bias.data = normal(m.bias.shape, 0.1)
            elif isinstance(m, nn.Embedding):
                m.weight.data = normal(m.weight.shape, 1.0 if m.weight.shape[0] > 3 else 0.5)
            elif isinstance(m, nn.Conv1d):
                m.weight.data = normal(m.weight.shape, 0.5)
            elif isinstance(m, nn.MultiheadAttention):
                m.in_proj_weight.data = normal(m.in_proj_weight.shape, m.in_proj_weight.shape[1] ** -0.5)
                m.in_proj_bias.data = normal(m.in_proj_bias.shape, 0.1)


def make_rows(cfg: dict, generator: torch.Generator) -> list[dict]:
    """Three text rows and two prefix rows shaped like real ones: [1, 17, ..., 18, ..., (19, 16, ..., 20)*K, 21]."""
    vocab, d = cfg["vocab_size"], cfg["hidden_size"]
    specs = [("text7", 0, 7, [3, 5], 2), ("text12", 0, 12, [4, 7, 10], 0), ("text20", 0, 20, [5, 9, 13, 17], 1),
             ("prefix5_text9", 5, 9, [3, 6], 0), ("prefix3_text15", 3, 15, [4, 8, 12], 2)]
    rows = []
    for name, p, n, markers, qtype in specs:
        ids = torch.randint(22, vocab, (n,), generator=generator).tolist()
        ids[0], ids[1], ids[-1] = 1, 17, 21
        ids[2] = 18
        for m in markers:
            ids[m - 1], ids[m] = 19, 16
        prefix = torch.randn(p, d, generator=generator) if p else None
        rows.append({"name": name, "ids": ids, "markers": markers, "qtype": qtype, "prefix": prefix})
    return rows


def publisher_forward(trunk, head, rows: list[dict]) -> list[torch.Tensor]:
    """D1OmniModel._forward's tensor steps for a batch of rows -> each row's K marker logits."""
    seqs, lengths, offsets = [], [], []
    for r in rows:
        text = trunk.embed_tokens(torch.tensor(r["ids"]))
        seqs.append(text if r["prefix"] is None else torch.cat([r["prefix"], text]))
        offsets.append(0 if r["prefix"] is None else r["prefix"].shape[0])
        lengths.append(len(r["ids"]))
    h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    total = torch.tensor([len(s) for s in seqs])
    pad = torch.arange(h.shape[1])[None] < total[:, None]
    h = trunk(h, pad, torch.tensor(offsets))
    text = torch.nn.utils.rnn.pad_sequence([h[i, o:o + n] for i, (o, n) in enumerate(zip(offsets, lengths))],
                                           batch_first=True)
    k = max(len(r["markers"]) for r in rows)
    text_pad = torch.arange(text.shape[1])[None] < torch.tensor(lengths)[:, None]
    marker_pos = torch.zeros(len(rows), k, dtype=torch.long)
    marker_mask = torch.zeros(len(rows), k, dtype=torch.bool)
    for i, r in enumerate(rows):
        marker_pos[i, :len(r["markers"])] = torch.tensor(r["markers"])
        marker_mask[i, :len(r["markers"])] = True
    logits = head(text, text_pad, marker_pos, marker_mask, torch.tensor([r["qtype"] for r in rows]))
    return [logits[i, :len(r["markers"])].clone() for i, r in enumerate(rows)]


def graph_forward(model: dm.D1Decide, row: dict, pad_id: int = 0, pad_noise: torch.Generator | None = None) -> torch.Tensor:
    question = host.Question(("choice", "score", "noul")[row["qtype"]], "x",
                             {"a": "", "b": ""} if row["qtype"] == 0 else (["l0", "l1"] if row["qtype"] == 1 else None))
    p = 0 if row["prefix"] is None else row["prefix"].shape[0]
    hrow = host.Row(question, row["ids"], row["markers"], False, p, "image" if p else "text", 0)
    inputs, markers = host.graph_inputs(hrow, L, None if p == 0 else row["prefix"].numpy(), model.hidden)
    t = {k: torch.from_numpy(v) for k, v in inputs.items()}
    n = len(row["ids"])
    if pad_id:
        t["input_ids"][0, p + n:] = pad_id
    if pad_noise is not None:
        t["prefix_embeds"][0, p + n:] = torch.randn(L - p - n, model.hidden, generator=pad_noise)
    scores = model(t["input_ids"], t["prefix_embeds"], t["pad_mask"], t["prefix_mask"], t["keep_right"],
                   t["qtype_onehot"])
    return scores[0, markers]


def max_abs(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def bit_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def run_toy(name: str, cfg: dict, enc) -> dict:
    gen = torch.Generator().manual_seed(SEED + (0 if name == "toy64" else 1))
    trunk = enc.Trunk(cfg)
    head = enc.DecisionHead(cfg["hidden_size"], 2)
    randomize([trunk, head], gen, enc.RMSNorm)
    trunk.eval(), head.eval()
    state = {f"encoder.{k}": v for k, v in trunk.state_dict().items()}
    state.update({f"head.{k}": v for k, v in head.state_dict().items()})
    table = {k: dm.map_key(k) for k in state}
    model = dm.D1Decide(cfg, head_layers=2, seq_len=L).eval()
    model.load_state_dict({table[k]: v for k, v in state.items()}, strict=True)
    rows = make_rows(cfg, gen)
    text, pre = rows[:3], rows[3:]
    out = {"config": cfg, "rows": {r["name"]: {"ids": r["ids"], "markers": r["markers"], "qtype": r["qtype"],
                                               "prefix": 0 if r["prefix"] is None else r["prefix"].shape[0]} for r in rows},
           "checkpoint_style_keys": len(table), "head_heads": cfg["hidden_size"] // 64,
           "publisher_head_fast_path_expected": (cfg["hidden_size"] // 64) % 2 == 0}
    with torch.no_grad():
        ref = [publisher_forward(trunk, head, [r])[0] for r in rows]
        mine = [graph_forward(model, r) for r in rows]
        out["a_text_max_abs"] = max_abs(mine[:3], ref[:3])
        out["b_prefix_max_abs"] = max_abs(mine[3:], ref[3:])
        out["ab_bit_identical_rows"] = [bool(torch.equal(x, y)) for x, y in zip(mine, ref)]
        out["marker_logits_ref"] = {r["name"]: v.tolist() for r, v in zip(rows, ref)}
        # (c) pad content
        noise = torch.Generator().manual_seed(SEED + 7)
        variants = {"pad_id_1": [graph_forward(model, r, pad_id=1) for r in rows],
                    "pad_id_100": [graph_forward(model, r, pad_id=100) for r in rows],
                    "pad_id_100_random_prefix_embeds": [graph_forward(model, r, pad_id=100, pad_noise=noise) for r in rows]}
        out["c_pad_content_bit_identical"] = {k: bit_equal(v, mine) for k, v in variants.items()}
        # (d) batch
        batch_text = publisher_forward(trunk, head, text)
        batch_mix = publisher_forward(trunk, head, [pre[0], pre[1], text[2]])
        out["d_batch_text_max_abs"] = max_abs(batch_text, ref[:3])
        out["d_batch_mix_max_abs"] = max_abs(batch_mix, [ref[3], ref[4], ref[2]])
        out["d_batch_bit_identical"] = bit_equal(batch_text, ref[:3]) and bit_equal(batch_mix, [ref[3], ref[4], ref[2]])
        # (e) NEG
        saved = enc.NEG
        enc.NEG = -1e4
        try:
            ref_neg = [publisher_forward(trunk, head, [r])[0] for r in rows]
        finally:
            enc.NEG = saved
        out["e_neg_1e4_bit_identical"] = bit_equal(ref_neg, ref)
        out["e_neg_1e4_max_abs"] = max_abs(ref_neg, ref)
        # (f) mutations
        out["f_mutations"] = {}
        for mutation in MUTATIONS:
            mutant = dm.D1Decide(cfg, head_layers=2, seq_len=L, mutation=mutation).eval()
            mutant.load_state_dict(model.state_dict(), strict=True)
            got = [graph_forward(mutant, r) for r in rows]
            out["f_mutations"][mutation] = {"prefix_rows_max_abs": max_abs(got[3:], mine[3:]),
                                            "text_rows_max_abs": max_abs(got[:3], mine[:3])}
        # (g) GQA forms
        broadcast = dm.D1Decide(cfg, head_layers=2, seq_len=L, gqa="broadcast").eval()
        broadcast.load_state_dict(model.state_dict(), strict=True)
        got = [graph_forward(broadcast, r) for r in rows]
        out["g_gqa_broadcast_vs_repeat"] = {"max_abs": max_abs(got, mine), "bit_identical": bit_equal(got, mine),
                                            "broadcast_vs_publisher_max_abs": max_abs(got, ref)}
        # (h) coverage
        coverage = {}
        pert = torch.Generator().manual_seed(SEED + 11)
        for pname, param in model.named_parameters():
            saved_param = param.data.clone()
            scale = 0.1 * (float(torch.std(saved_param, correction=0)) + 0.1)
            param.data = saved_param + torch.randn(saved_param.shape, generator=pert) * scale
            got = [graph_forward(model, r) for r in rows]
            coverage[pname] = max_abs(got, mine)
            param.data = saved_param
        assert bit_equal([graph_forward(model, r) for r in rows], mine), "coverage did not restore the parameters"
        out["h_coverage"] = {"parameters": len(coverage), "min_effect": min(coverage.values()),
                             "min_effect_parameter": min(coverage, key=coverage.get),
                             "dead": [k for k, v in coverage.items() if v == 0.0], "per_parameter": coverage}
        # (i) the publisher's head: fast path vs plain path
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            ref_plain = [publisher_forward(trunk, head, [r])[0] for r in rows]
        finally:
            torch.backends.mha.set_fastpath_enabled(True)
        out["i_fast_vs_plain"] = {"max_abs": max_abs(ref_plain, ref), "bit_identical": bit_equal(ref_plain, ref),
                                  "graph_vs_plain_max_abs": max_abs(mine, ref_plain)}
    return out


def checkpoint_header(snapshot: Path) -> dict:
    """model.safetensors' JSON header: from the file when it is downloaded, else from the lane's copy of the header
    (round 1 runs before the weights are fetched)."""
    weights = snapshot / dm.WEIGHTS["file"]
    if weights.exists():
        with open(weights, "rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            return json.loads(handle.read(size))
    return json.loads(code_path("standup", "handoffs", "assets", "2026-10-08-d1", "source",
                                "d1-omni-600m_safetensors_header.json").read_text())


def full_size_checks(enc, snapshot: Path) -> dict:
    config = json.loads((snapshot / "config.json").read_text())
    dm.validate_config(config)
    header = checkpoint_header(snapshot)
    expected = dm.expected_shapes(header)
    with torch.device("meta"):
        model = dm.build(config, 256)
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert shapes == expected, (sorted(set(shapes) ^ set(expected))[:5],
                                [k for k in shapes if k in expected and shapes[k] != expected[k]][:5])
    trunk_count = sum(v.numel() for k, v in model.state_dict().items() if k.startswith("trunk."))
    head_count = sum(v.numel() for k, v in model.state_dict().items() if k.startswith("head."))
    assert (trunk_count, head_count) == (dm.PARAMETERS["encoder"], dm.PARAMETERS["head"])
    rope = {}
    publisher_trunk = enc.Trunk.__new__(enc.Trunk)  # only rope() is used: it reads rope_theta and head_dim
    publisher_trunk.rope_theta = config["text_config"]["rope_theta"]
    publisher_trunk.head_dim = 64
    for length in (24, 4096):
        cos, sin = dm.rope_tables(config["text_config"], length)
        pc, ps = enc.Trunk.rope(publisher_trunk, length, torch.zeros(1))
        rope[str(length)] = {"cos_bit_identical": bool(torch.equal(cos, pc)), "sin_bit_identical": bool(torch.equal(sin, ps))}
    key_table = dm.key_table(header)
    return {"module_tensors": len(shapes), "checkpoint_trunk_head_tensors": len(expected), "shapes_equal": True,
            "trunk_parameters": trunk_count, "head_parameters": head_count, "rope": rope,
            "ignored_checkpoint_tensors": sum(1 for k in header if k != "__metadata__" and dm.map_key(k) is None),
            "key_table": key_table}


def main() -> int:
    t0 = time.time()
    torch.manual_seed(SEED)
    snapshot = Path(hf_snapshot(dm.MODEL_ID, revision=dm.MODEL_SHA))
    enc = load_publisher_encoder(snapshot)
    results = {name: run_toy(name, cfg, enc) for name, cfg in TOYS.items()}
    full = full_size_checks(enc, snapshot)
    verdict = {}
    for name, r in results.items():
        verdict[name] = {
            "a": r["a_text_max_abs"] <= BARS["a"], "b": r["b_prefix_max_abs"] <= BARS["b"],
            "c": all(r["c_pad_content_bit_identical"].values()),
            "d": max(r["d_batch_text_max_abs"], r["d_batch_mix_max_abs"]) <= BARS["d"],
            "e": r["e_neg_1e4_bit_identical"],
            "f": all(v["prefix_rows_max_abs"] > BARS["f"] for v in r["f_mutations"].values()),
            "h": not r["h_coverage"]["dead"],
        }
    rope_ok = all(v["cos_bit_identical"] and v["sin_bit_identical"] for v in full["rope"].values())
    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "seed": SEED, "L": L, "bars": BARS,
           "pass": all(all(v.values()) for v in verdict.values()) and rope_ok, "verdict": verdict,
           "toys": results, "full_size": full, "elapsed_s": round(time.time() - t0, 2)}
    out = WORK / "results" / "toy_parity.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{out}: pass {doc['pass']} ({doc['elapsed_s']} s)")
    for name, r in results.items():
        print(f"  {name}: (a) {r['a_text_max_abs']:.2e} (b) {r['b_prefix_max_abs']:.2e} bit-identical rows "
              f"{sum(r['ab_bit_identical_rows'])}/5 | (c) {r['c_pad_content_bit_identical']} | (d) text "
              f"{r['d_batch_text_max_abs']:.2e} mix {r['d_batch_mix_max_abs']:.2e} | (e) {r['e_neg_1e4_bit_identical']} | "
              f"(g) {r['g_gqa_broadcast_vs_repeat']['max_abs']:.2e} bit {r['g_gqa_broadcast_vs_repeat']['bit_identical']} | "
              f"(i) fast-vs-plain {r['i_fast_vs_plain']['max_abs']:.2e} (expected fast {r['publisher_head_fast_path_expected']})")
        print("     (f)", {k: (f"{v['prefix_rows_max_abs']:.3g}", f"{v['text_rows_max_abs']:.3g}") for k, v in r["f_mutations"].items()},
              "| (h)", r["h_coverage"]["parameters"], "params, min effect", f"{r['h_coverage']['min_effect']:.2e}",
              r["h_coverage"]["min_effect_parameter"], "dead", r["h_coverage"]["dead"])
    print("  full size:", {k: v for k, v in full.items() if k != "key_table"})
    print("  verdict:", verdict)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
