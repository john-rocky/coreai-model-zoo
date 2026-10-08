#!/usr/bin/env python3
"""Activation ranges of d1_omni_model.D1Decide (fp32, the checkpoint's weights) on every fixture row against fp16's
largest finite value, then its fp16 and wfp16 frames in torch eager on a 60-row subset against the reference. CPU.

    python3 conversion/d1_omni/absmax_scan.py   # -> $ZOO_WORK_ROOT/_d1_omni/results/absmax.json

Scan. The module runs unchanged; wrappers on its own helpers (`_rms`, `_ln`, `_lin`, `_attend`, `_short_conv`) read
the residual stream entering and leaving every trunk layer and head layer, the trunk output, every linear's output,
the MLP product silu(w1 x) * w3 x (w2's input), the short convolution's b * u, filter output y and c * y, the
attention products q k^T (before the 1/8 scale; the logits are that times 1/8), and the scorer's input. Each is
max |x| over the row's real positions (prefix + text for the trunk, the text for the head) and over all L positions
(pad included: an fp16 graph computes those too, and an inf there reaches real rows through a masked key). The
attention products are taken over all entries and over real query x open key. Every maximum keeps the row and
position it came from; margin = 65,504 / max. The wrappers compute what the helpers compute, op for op (asserted:
every row's scores are bit-identical to a plain module's).

fp16 / wfp16 (set_precision; fp16 = fp16 compute with RMSNorm / LayerNorm / softmax / scorer in fp32, wfp16 = fp16
weights with fp32 compute): 60 rows — every source, own_L01..03, long_3400, the 5 prefix rows and the 13 near ties
— against the reference: max |dp| after the temperature, argmax, marker logits max |d|, NaN / inf in the scores,
seconds; also against the fp32 module. Values only: the bar is the supervisor's call.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _fixtures import fixtures_path  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402

WORK = work_path("_d1_omni")
FP16_MAX = 65504.0
# The 60-row subset: (source/mode, quota); near ties and the npz rows of a source are taken first, then fixture order.
SUBSET_QUOTA = {"card/text": 3, "transfer_v4_dev_head/text": 9, "semif_authored144/text": 9, "own_text/text": 8,
                "own_json/text": 4, "own_long/text": 3, "own_long_extended/text": 1, "transfer_v4_dev_sources/text": 11,
                "transfer_v4_dev_score/text": 4, "red_arm/text": 1, "own_image/text": 1, "own_audio/text": 1,
                "card/image": 1, "card/audio": 1, "own_audio/audio": 3}
SUBSET_FIRST = [("own_L01", "stoppage", "text"), ("own_L02", "subcontract", "text"), ("own_L03", "fee_motion", "text"),
                ("long_3400", "stoppage", "text"), ("card_cats", "cats", "image"), ("card_audio", "topic", "audio"),
                ("aud_01", "topic", "audio"), ("aud_02", "topic", "audio"), ("aud_03", "topic", "audio"),
                ("tv4_009", "answer", "text"), ("tv4_053", "answer", "text")]


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


def shared(base: dm.D1Decide, config: dict, seq_len: int, state: dict | None = None) -> dm.D1Decide:
    m = dm.build(config, seq_len)
    m.load_state_dict(base.state_dict() if state is None else state, strict=True, assign=True)
    m.eval().requires_grad_(False)
    m.precision, m.compute_dtype = base.precision, base.compute_dtype
    return m


class Scan:
    """Wrappers on one module instance that record max |x| at every point, per row."""

    def __init__(self, model: dm.D1Decide):
        self.model = model
        self.points: dict[str, dict] = {}
        self.ctx = None
        self.attn_calls = 0
        mod_names = {id(m): n for n, m in model.named_modules()}
        par_names = {id(p): n for n, p in model.named_parameters()}
        role = {}
        for i, layer in enumerate(model.trunk.layers):
            role[id(layer.operator_norm)] = f"trunk.L{i:02d}.resid_in"
            role[id(layer.ffn_norm)] = f"trunk.L{i:02d}.resid_mid"
        role[id(model.trunk.embedding_norm)] = "trunk.resid_out"
        for j, layer in enumerate(model.head.head.layers):
            role[id(layer.norm1)] = f"head.H{j}.resid_in"
            role[id(layer.norm2)] = f"head.H{j}.resid_mid"
        role[id(model.head.scorer[0])] = "head.scorer_in"
        self.attn_names = [f"trunk.L{i:02d}.attn" for i, layer in enumerate(model.trunk.layers)
                           if layer.is_attention_layer] + [f"head.H{j}.attn" for j in range(len(model.head.head.layers))]
        cls = dm.D1Decide
        rms0, lin0 = cls._rms, cls._lin
        ln0, masks0 = model._ln, model.masks

        def rms(norm, x):
            if id(norm) in role:
                name = role[id(norm)]
                self.see(name, x, "trunk")
                if name == "trunk.L00.resid_in":
                    p = self.ctx["prefix"]
                    self.see("trunk.embed.text", x, "trunk", lo=p)
                    if p:
                        self.see("trunk.embed.prefix", x, "trunk", hi=p)
            y = rms0(norm, x)
            if norm is model.trunk.embedding_norm:
                self.see("trunk.out", y, "trunk")
            return y

        def ln(norm, x):
            if id(norm) in role:
                self.see(role[id(norm)], x, "head")
            return ln0(norm, x)

        def lin(x, module=None, weight=None, bias=None):
            y = lin0(x, module, weight, bias)
            name = mod_names[id(module)] if module is not None else par_names[id(weight)]
            region = "trunk" if name.startswith("trunk.") else "head"
            self.see(f"{name}:out", y, region)
            if name.endswith(("feed_forward.w2", "conv.out_proj")):
                self.see(f"{name}:in", x, region)
            return y

        def short_conv(conv, x, pad3, keep3):  # D1Decide._short_conv, op for op, with two reads
            L = model.seq_len
            b, c, u = model._lin(x * pad3, conv.in_proj).chunk(3, dim=-1)
            bx = b * u
            name = mod_names[id(conv)]
            self.see(f"{name}:b*u", bx, "trunk")
            w = conv.conv.weight[:, 0, :].to(bx.dtype)
            xp = F.pad(bx, (0, 0, 1, 1))
            right = xp[:, 2:2 + L] * keep3
            y = xp[:, 0:L] * w[:, 0]
            y = y + xp[:, 1:1 + L] * w[:, 1]
            y = y + right * w[:, 2]
            self.see(f"{name}:y", y, "trunk")
            return model._lin(c * y, conv.out_proj)

        def attend(q, k, v, mask, scale):  # D1Decide._attend, op for op, with the product read
            raw = torch.matmul(q, k.transpose(-1, -2))
            self.see_attention(self.attn_names[self.attn_calls], raw, mask)
            self.attn_calls += 1
            scores = raw.float() * scale + mask
            return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)

        def masks(pad_mask, prefix_mask, keep_right):
            self.attn_calls = 0
            return masks0(pad_mask, prefix_mask, keep_right)

        model._rms, model._ln, model._lin = rms, ln, lin
        model._short_conv, model._attend, model.masks = short_conv, attend, masks

    def _keep(self, name: str, kind: str, value: float, pos: int):
        slot = self.points.setdefault(name, {}).setdefault(kind, {"max": -1.0})
        if value > slot["max"]:
            slot.update(max=value, row=self.ctx["key"], position=pos)

    def see(self, name: str, x: torch.Tensor, region: str, lo: int | None = None, hi: int | None = None):
        L, p, n = self.model.seq_len, self.ctx["prefix"], self.ctx["n"]
        v = x.detach().abs().reshape(L, -1).amax(-1).float()
        real_lo = (p if region == "head" else 0) if lo is None else lo
        real_hi = n if hi is None else hi
        seg = v[real_lo:real_hi]
        i = int(torch.argmax(seg))
        self._keep(name, "real", float(seg[i]), real_lo + i)
        if lo is None and hi is None:
            j = int(torch.argmax(v))
            self._keep(name, "all", float(v[j]), j)
            if not bool(torch.isfinite(v).all()):
                self.points[name]["nonfinite"] = True

    def see_attention(self, name: str, raw: torch.Tensor, mask: torch.Tensor):
        L, p, n = self.model.seq_len, self.ctx["prefix"], self.ctx["n"]
        lo = p if name.startswith("head") else 0
        everything = float(torch.maximum(raw.amax(), -raw.amin()))
        self._keep(name + ".qk", "all", everything, -1)
        open_ = (mask.expand(1, 1, L, L)[0, 0, lo:n] == 0)
        sel = torch.where(open_[None], raw[0, :, lo:n], torch.zeros((), dtype=raw.dtype)).abs().amax(dim=(0, 2)).float()
        i = int(torch.argmax(sel))
        self._keep(name + ".qk", "real_open", float(sel[i]), lo + i)


def to_tensors(inputs: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(inputs[k]) for k in ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask",
                                                  "keep_right", "qtype_onehot")]


def spread(rows: list[tuple]) -> list[tuple]:
    """Fixture-order rows interleaved across sub-sources (the id without its last `_part`), and within one
    sub-source across records before any record's second question."""
    if not rows:
        return []
    subs: dict[str, dict[str, list]] = {}
    for r in rows:
        subs.setdefault(r[0].rsplit("_", 1)[0], {}).setdefault(r[0], []).append(r)
    per_sub = []
    for recs in subs.values():
        lists = list(recs.values())
        per_sub.append([lst[i] for i in range(max(map(len, lists))) for lst in lists if i < len(lst)])
    return [lst[i] for i in range(max(map(len, per_sub))) for lst in per_sub if i < len(lst)]


def pick_subset(ref: dict) -> list[tuple]:
    rows = [(e["id"], q["qid"], e["mode"], f"{e['source']}/{e['mode']}", q["near_tie"])
            for e in ref["records"] for q in e["questions"]]
    chosen = []
    for group, quota in SUBSET_QUOTA.items():
        members = [r for r in rows if r[3] == group]
        first = [r for r in members if r[:3] in SUBSET_FIRST or r[4]]
        rest = spread([r for r in members if r not in first])
        take = (first + rest)[:max(quota, len(first))]
        chosen += take
    assert len(chosen) == len(set(chosen))
    return [r[:3] for r in chosen]


def main() -> int:
    t_start = time.perf_counter()
    torch.set_grad_enabled(False)
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    ref_path = WORK / "ref" / "records_ref.json"
    ref = json.loads(ref_path.read_text())
    fixtures = {r["id"]: r for r in json.loads(fixtures_path(ref["fixtures"]["sha256"]).read_text())["records"]}
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    config = json.loads((snapshot / "config.json").read_text())
    base, load_record = dm.load_d1_decide(snapshot, 256, "fp32", verify_sha256=True)
    plain = {256: base}
    scanned = {}
    prefixes = {}
    for p in (WORK / "ref" / "npz").glob("*.npz"):
        with np.load(p) as z:
            if str(z["mode"]) != "text":
                prefixes[p.stem] = z["prefix"].copy()

    rows = []
    for entry in ref["records"]:
        rec = fixtures[entry["id"]]
        hrows = host.request_rows(tok, rec["request"]["state"], rec["request"]["questions"], entry["mode"], entry["prefix"])
        for row, q in zip(hrows, entry["questions"]):
            assert row.ids == q["ids"] and row.markers == q["markers"]
            rows.append((entry, row, q, prefixes.get(entry["id"]) if entry["mode"] != "text" else None))

    # ---- scan (fp32, every row)
    t0 = time.perf_counter()
    fp32_scores = {}
    for entry, row, q, prefix in rows:
        L = host.bucket_for(row.positions)
        if L not in plain:
            plain[L] = shared(base, config, L)
        if L not in scanned:
            scanned[L] = Scan(shared(base, config, L))
        scan = scanned[L]
        key = f"{entry['id']}/{q['qid']}/{entry['mode']}"
        inputs, markers = host.graph_inputs(row, L, prefix)
        t = to_tensors(inputs)
        for s in scanned.values():
            s.ctx = {"key": key, "prefix": row.prefix_len, "n": row.positions}
        got = scan.model(*t)
        want = plain[L](*t)
        assert torch.equal(got, want), key
        fp32_scores[key] = want[0, markers].numpy().copy()
    scan_seconds = time.perf_counter() - t0
    points: dict[str, dict] = {}
    for s in scanned.values():
        for name, kinds in s.points.items():
            dst = points.setdefault(name, {})
            for kind, slot in kinds.items():
                if kind == "nonfinite":
                    dst["nonfinite"] = True
                elif kind not in dst or slot["max"] > dst[kind]["max"]:
                    dst[kind] = dict(slot)
    for name, kinds in points.items():
        for kind, slot in kinds.items():
            if isinstance(slot, dict):
                slot["margin"] = FP16_MAX / slot["max"] if slot["max"] > 0 else None
    print(f"scan: {len(rows)} rows in {scan_seconds:.1f} s, {len(points)} points", flush=True)

    def worst(names):
        best = None
        for name in names:
            for kind in ("real", "all"):
                slot = points.get(name, {}).get(kind)
                if slot and (best is None or slot["max"] > best[1]["max"]):
                    best = (f"{name} [{kind}]", slot)
        return None if best is None else {"point": best[0], **best[1]}

    layers = []
    for i, layer in enumerate(base.trunk.layers):
        pre = f"trunk.layers.{i}"
        entry = {"layer": i, "kind": "attention" if layer.is_attention_layer else "conv",
                 "resid_in": points[f"trunk.L{i:02d}.resid_in"], "resid_mid": points[f"trunk.L{i:02d}.resid_mid"],
                 "mlp_w1_w3_out": worst([f"{pre}.feed_forward.w1:out", f"{pre}.feed_forward.w3:out"]),
                 "mlp_product": points[f"{pre}.feed_forward.w2:in"],
                 "mlp_out": points[f"{pre}.feed_forward.w2:out"]}
        if layer.is_attention_layer:
            entry["qkv_out"] = worst([f"{pre}.self_attn.{n}:out" for n in ("q_proj", "k_proj", "v_proj")])
            entry["qk_product"] = points[f"trunk.L{i:02d}.attn.qk"]
            entry["attn_out"] = points[f"{pre}.self_attn.out_proj:out"]
        else:
            entry["conv_in_proj_out"] = points[f"{pre}.conv.in_proj:out"]
            entry["conv_b*u"] = points[f"{pre}.conv:b*u"]
            entry["conv_y"] = points[f"{pre}.conv:y"]
            entry["conv_c*y"] = points[f"{pre}.conv.out_proj:in"]
            entry["conv_out"] = points[f"{pre}.conv.out_proj:out"]
        layers.append(entry)
    head = []
    for j in range(len(base.head.head.layers)):
        pre = f"head.head.layers.{j}"
        head.append({"layer": j, "resid_in": points[f"head.H{j}.resid_in"], "resid_mid": points[f"head.H{j}.resid_mid"],
                     "qkv_out": points[f"{pre}.self_attn.in_proj_weight:out"],
                     "qk_product": points[f"head.H{j}.attn.qk"],
                     "ffn_linear1_out": points[f"{pre}.linear1:out"], "ffn_linear2_out": points[f"{pre}.linear2:out"]})
    overall = sorted(((slot["max"], f"{name} [{kind}]", slot["row"], slot["position"])
                      for name, kinds in points.items() for kind, slot in kinds.items() if isinstance(slot, dict)),
                     reverse=True)
    scan_doc = {"rows": len(rows), "seconds": round(scan_seconds, 1), "fp16_max": FP16_MAX,
                "embeddings": {k: points[k] for k in ("trunk.embed.text", "trunk.embed.prefix", "trunk.L00.resid_in")},
                "trunk_layers": layers,
                "final_norm": {"resid_out (input)": points["trunk.resid_out"], "trunk.out": points["trunk.out"]},
                "head_layers": head,
                "scorer": {"input": points["head.scorer_in"], "linear_1_out": points["head.scorer.1:out"],
                           "scores": points["head.scorer.3:out"]},
                "largest_10": [{"max": m, "point": n, "row": r, "position": p, "margin": FP16_MAX / m}
                               for m, n, r, p in overall[:10]],
                "largest_real_10": [{"max": m, "point": n, "row": r, "position": p, "margin": FP16_MAX / m}
                                    for m, n, r, p in overall if "[all]" not in n][:10],
                "points": points}

    # ---- fp16 / wfp16 eager on the subset
    subset = pick_subset(ref)
    by_key = {(e["id"], q["qid"], e["mode"]): (e, row, q, prefix) for e, row, q, prefix in rows}
    frames = {}
    for precision in ("fp16", "wfp16"):
        first = shared(base, config, 256, {k: v.clone() for k, v in base.state_dict().items()}).set_precision(precision)
        mods = {256: first}
        out = []
        t0 = time.perf_counter()
        for key in subset:
            entry, row, q, prefix = by_key[key]
            L = host.bucket_for(row.positions)
            if L not in mods:
                mods[L] = shared(first, config, L)
            inputs, markers = host.graph_inputs(row, L, prefix)
            t1 = time.perf_counter()
            scores = mods[L](*to_tensors(inputs))
            seconds = time.perf_counter() - t1
            n = row.positions
            finite = bool(torch.isfinite(scores[0, :n]).all())
            z = scores[0, markers].numpy()
            p = host.probabilities_from_logits(z, row.question, row.calibrate)
            p32 = host.probabilities_from_logits(fp32_scores[f"{key[0]}/{key[1]}/{key[2]}"], row.question, row.calibrate)
            out.append({"id": key[0], "qid": key[1], "mode": key[2], "source": entry["source"], "bucket": L,
                        "positions": n, "near_tie": q["near_tie"], "finite": finite,
                        "max_abs_dp": float(max(abs(a - b) for a, b in zip(p, q["probs"]))),
                        "max_abs_dlogit": float(np.max(np.abs(z.astype(np.float64) - np.asarray(q["logits_raw"])))),
                        "argmax_equal": int(np.argmax(p)) == q["argmax_index"],
                        "max_abs_dp_vs_fp32_module": float(max(abs(a - b) for a, b in zip(p, p32))),
                        "seconds": round(seconds, 3)})
        dps = [r["max_abs_dp"] for r in out]
        frames[precision] = {
            "rows": len(out), "argmax_equal": sum(r["argmax_equal"] for r in out),
            "near_ties": sum(r["near_tie"] for r in out),
            "argmax_equal_non_near_tie": sum(r["argmax_equal"] for r in out if not r["near_tie"]),
            "max_abs_dp": max(dps), "mean_of_row_max_abs_dp": float(np.mean(dps)),
            "max_abs_dlogit": max(r["max_abs_dlogit"] for r in out),
            "max_abs_dp_vs_fp32_module": max(r["max_abs_dp_vs_fp32_module"] for r in out),
            "nonfinite_rows": [f"{r['id']}/{r['qid']}/{r['mode']}" for r in out if not r["finite"]],
            "worst_row": max(out, key=lambda r: r["max_abs_dp"]), "seconds_total": round(time.perf_counter() - t0, 1),
            "seconds_by_bucket": {str(b): round(sum(r["seconds"] for r in out if r["bucket"] == b), 2)
                                  for b in sorted({r["bucket"] for r in out})},
            "per_row": out}
        print(f"{precision}: argmax {frames[precision]['argmax_equal']}/{len(out)} max |dp| "
              f"{frames[precision]['max_abs_dp']:.3e} mean {frames[precision]['mean_of_row_max_abs_dp']:.3e} "
              f"nonfinite {len(frames[precision]['nonfinite_rows'])} ({frames[precision]['seconds_total']} s)", flush=True)
        del mods, first

    doc = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "reference": {"path": str(ref_path), "sha256": sha256_file(ref_path)}, "weights": load_record,
           "scan": scan_doc, "subset": [list(k) for k in subset], "eager_fp16": frames["fp16"],
           "eager_wfp16": frames["wfp16"], "elapsed_s": round(time.perf_counter() - t_start, 1)}
    write_atomic(WORK / "results" / "absmax.json", (json.dumps(doc, indent=1) + "\n").encode())
    print(json.dumps({"largest_10": scan_doc["largest_10"], "largest_real_10": scan_doc["largest_real_10"]}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_row", "worst_row")} for k, v in frames.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
