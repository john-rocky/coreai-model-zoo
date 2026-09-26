# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.1",
#     "torch==2.9.0",
#     "transformers==4.57.6",
#     "safetensors",
#     "numpy",
# ]
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# GLiNER2.5-Decide (fastino/GLiNER2.5-Decide) -> Core AI: the classification path of the GLiNER2.5
# family as one static graph. Zero-shot: the label set enters through input_ids, so one bundle answers
# any task/label set up to MMAX labels per call.
#
# FUSED STATIC GRAPH (the classification version of conversion/export_gliner2_pii.py):
#   forward(input_ids[1,S] i32, attention_mask[1,S] i32, label_idx[1,MMAX] i32) -> logits[1,MMAX] f32
# DeBERTa-v3-large (transformers DebertaV2Model, weights strict-loaded from model.safetensors
# `encoder.*`) -> gather the [L] marker rows -> classifier Linear(1024,2048) > ReLU > Linear(2048,1).
# span_rep / count_embed / count_pred are not used by classification and are not in the graph; gliner2
# itself is not imported (the export venv has gliner2 1.3.2, which cannot read Decide).
# Relative positions are baked (--relpos baked, default): coreai-torch 0.4.1 lowers aten.div.Tensor(int, int)
# to integer division, so make_log_bucket_position's `abs_pos / mid` puts every distance >= 129 in bucket +-128
# (probe: _gliner25_decide/logs/diag_div_probe.log). The table depends only on S; --relpos runtime keeps the stock path.
# rel_embeddings is a Parameter, so half() takes it to fp16 (measured max|dlogit| 0.014).
# Host contract: the host linearizes the schema exactly like gliner2's SchemaTransformer
#   ( [P] task[: prompt][ [DESCRIPTION] label: desc]* ( [L] l1 [L] l2 ... ) ) [SEP_STRUCT] ( ... ) [SEP_TEXT] words .
# label_idx = the [L] positions of all tasks concatenated in task order (the host keeps each task's
# slice); unused slots repeat the first [L] position (a safe gather the host never reads).
# input_ids pad = 0 ([PAD]), attention_mask 1 = real / 0 = pad. Softmax (single-label) / sigmoid +
# cls_threshold (multi-label; none above -> argmax) run on the host, as does dropping the [P] row.
#
# Gates, all against the gliner2 2.0.0 fp32 oracle fixtures (conversion/gliner25_decide_oracle.py):
#   GATE-1  fp32 re-authored module (CPU): decisions 100% equal, max|dlogit| <= 1e-3
#   GATE-2  poisoned label_idx (+1: reads each label's first sub-word instead of [L]) must change
#           decisions — proves the comparator can go red
#   GATE-3  fp16 bundle on the Mac GPU engine: decisions, max|dlogit|, max|dprob|, per-call wall time
#           (--poison also runs the poisoned inputs through the engine)
import argparse
import asyncio
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from _paths import exports_dir, hf_snapshot

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REV = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
MARKER_IDS = {"[MASK]": 128000, "[SEP_STRUCT]": 128001, "[SEP_TEXT]": 128002, "[P]": 128003,
              "[C]": 128004, "[E]": 128005, "[R]": 128006, "[L]": 128007, "[EXAMPLE]": 128008,
              "[OUTPUT]": 128009, "[DESCRIPTION]": 128010}
PAD_ID = 0
DEFAULT_THRESHOLD = 0.5
XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"


# ------------------------------------------------------------------ graph
class GLiNER25Classifier(torch.nn.Module):
    def __init__(self, encoder, classifier, s, relpos="baked"):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.relpos = relpos
        if relpos == "baked":
            from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position

            enc = encoder.encoder
            dummy = torch.zeros(1, s, 1)
            table = build_relative_position(dummy, dummy, bucket_size=enc.position_buckets,
                                            max_position=enc.max_relative_positions)
            stock = enc.get_rel_pos(torch.zeros(1, s, encoder.config.hidden_size))
            assert table.shape == (1, s, s) and table.dtype == torch.int64 and torch.equal(table, stock)
            self.register_buffer("rel_pos", table, persistent=False)                          # int64 [1,S,S]

    def forward(self, input_ids, attention_mask, label_idx):
        if self.relpos == "baked":                    # DebertaV2Model.forward (z_steps 0) with the table passed in
            emb = self.encoder.embeddings(input_ids=input_ids, mask=attention_mask)
            hidden = self.encoder.encoder(emb, attention_mask, output_hidden_states=False,
                                          relative_pos=self.rel_pos).last_hidden_state[0]      # [S,1024]
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        label_embs = hidden.index_select(0, label_idx[0].to(torch.long))                      # [MMAX,1024]
        logits = self.classifier(label_embs).squeeze(-1)                                      # [MMAX]
        return logits.unsqueeze(0).to(torch.float32)                                          # [1,MMAX]


def build_module(snapshot, s, relpos):
    from safetensors.torch import load_file
    from transformers import DebertaV2Config, DebertaV2Model

    cfg = DebertaV2Config.from_json_file(str(Path(snapshot) / "encoder_config" / "config.json"))
    encoder = DebertaV2Model(cfg)
    sd = load_file(str(Path(snapshot) / "model.safetensors"))
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    assert len(enc_sd) == 390, len(enc_sd)
    res = encoder.load_state_dict(enc_sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    classifier = torch.nn.Sequential(torch.nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size), torch.nn.ReLU(),
                                     torch.nn.Linear(2 * cfg.hidden_size, 1))
    clf_sd = {k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")}
    res = classifier.load_state_dict(clf_sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    unused = sorted({k.split(".")[0] for k in sd} - {"encoder", "classifier"})
    n_unused = sum(v.numel() for k, v in sd.items() if k.split(".")[0] in unused)
    module = GLiNER25Classifier(encoder, classifier, s, relpos).eval()
    print(f"[INFO] encoder: DebertaV2Model {cfg.num_hidden_layers}L h{cfg.hidden_size} vocab {cfg.vocab_size} "
          f"attn={encoder.config._attn_implementation} eps={cfg.layer_norm_eps}; loaded {len(enc_sd)} encoder + "
          f"{len(clf_sd)} classifier tensors (strict); not in graph: {unused} ({n_unused:,} params); "
          f"relative positions: {relpos}"
          + (f" (table [1,{s},{s}] == encoder.get_rel_pos)" if relpos == "baked" else ""))
    return module


# ------------------------------------------------------------------ host (NumPy; the Swift host mirrors this)
def graph_inputs(case, S, MMAX, poison=False):
    ids = case["input_ids"]
    n = len(ids)
    assert n <= S, (case["id"], n)
    input_ids = np.full((1, S), PAD_ID, np.int32)
    input_ids[0, :n] = ids
    attention_mask = np.zeros((1, S), np.int32)
    attention_mask[0, :n] = 1
    l_pos, slices = [], []
    for ssi in case["schema_special_indices"]:
        assert ids[ssi[0]] == MARKER_IDS["[P]"], (case["id"], ssi)
        assert all(ids[p] == MARKER_IDS["[L]"] for p in ssi[1:]), (case["id"], ssi)
        slices.append((len(l_pos), len(l_pos) + len(ssi) - 1))
        l_pos.extend(ssi[1:])
    assert 0 < len(l_pos) <= MMAX, (case["id"], len(l_pos))
    label_idx = np.full((1, MMAX), l_pos[0], np.int32)
    label_idx[0, :len(l_pos)] = l_pos
    if poison:
        label_idx = label_idx + 1
    return input_ids, attention_mask, label_idx, slices


def probs_of(logits, multi, act="auto"):
    x = np.asarray(logits, np.float64)
    if act == "sigmoid" or (act == "auto" and multi):
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x - x.max())
    return e / e.sum()


def decide(probs, multi, thr):
    if multi:
        return [k for k in range(len(probs)) if probs[k] >= thr] or [int(np.argmax(probs))]
    return [int(np.argmax(probs))]


def compare_case(case, logits_row, slices):
    """Per-task comparison of a [MMAX] logits row against the oracle's fp32 task results."""
    tasks = []
    for tr, (a, b) in zip(case["task_results"], slices):
        got = np.asarray(logits_row[a:b], np.float64)
        ref = np.asarray(tr["logits"], np.float64)
        assert got.shape == ref.shape
        p = probs_of(got, tr["multi_label"], tr["class_act"])
        dec = decide(p, tr["multi_label"], tr["cls_threshold"])
        ref_dec = tr["decision"] if tr["multi_label"] else [tr["decision"]]
        ref_idx = [tr["labels"].index(lbl) for lbl in ref_dec]
        if tr["multi_label"]:
            margin = float(np.min(np.abs(np.asarray(tr["probs"]) - tr["cls_threshold"])))
        else:
            top = np.sort(ref)[::-1]
            margin = float(top[0] - top[1]) if len(top) > 1 else float("inf")
        tasks.append({
            "task": tr["task"], "multi_label": tr["multi_label"], "n_labels": b - a,
            "decision_equal": dec == ref_idx,
            "decision": [tr["labels"][k] for k in dec], "oracle_decision": ref_dec,
            "max_abs_dlogit": float(np.max(np.abs(got - ref))),
            "max_abs_dprob": float(np.max(np.abs(p - np.asarray(tr["probs"], np.float64)))),
            "oracle_margin": margin,
        })
    return tasks


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def summarize(rows, key="tasks"):
    tasks = [t for r in rows for t in r[key]]
    return {
        "cases": len(rows), "tasks": len(tasks),
        "decisions_equal_tasks": sum(t["decision_equal"] for t in tasks),
        "decisions_equal_pct": round(100.0 * sum(t["decision_equal"] for t in tasks) / len(tasks), 3),
        "cases_with_changed_decision": sum(not all(t["decision_equal"] for t in r[key]) for r in rows),
        "max_abs_dlogit": max(t["max_abs_dlogit"] for t in tasks),
        "max_abs_dprob": max(t["max_abs_dprob"] for t in tasks),
        "min_cos": min(r["cos"] for r in rows),
    }


def load_fixtures(paths, only):
    cases, headers = [], []
    for p in paths:
        d = json.loads(Path(p).read_text())
        assert d["header"]["model_rev"] == MODEL_REV, (p, d["header"]["model_rev"])
        headers.append({"path": str(p), "set": d["header"]["set"], "created": d["header"]["created"],
                        "versions": d["header"]["versions"], "self_check": d["header"]["self_check"]})
        cases.extend(d["cases"])
    if only:
        keep = set(only.split(","))
        cases = [c for c in cases if c["id"] in keep]
        assert len(cases) == len(keep), f"unknown case ids: {keep - {c['id'] for c in cases}}"
    return cases, headers


def torch_gate(module, cases, S, MMAX, poison):
    rows = []
    with torch.no_grad():
        for c in cases:
            ii, am, li, slices = graph_inputs(c, S, MMAX, poison=poison)
            out = module(torch.from_numpy(ii), torch.from_numpy(am), torch.from_numpy(li))
            assert out.shape == (1, MMAX) and out.dtype == torch.float32
            row = out[0].numpy().astype(np.float64)
            n = slices[-1][1]
            ref_all = [x for tr in c["task_results"] for x in tr["logits"]]
            rows.append({"id": c["id"], "seq_len": c["seq_len"], "n_labels": n,
                         "tasks": compare_case(c, row, slices), "cos": cos(row[:n], ref_all),
                         "logits": row[:n].tolist()})
    return rows


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(f"[INFO] wrote {path}")


def bundle_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def env_info():
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine()}
    try:
        from importlib.metadata import version
        info["coreai-torch"] = version("coreai-torch")
        info["coreai-core"] = version("coreai-core")
    except Exception as e:  # noqa: BLE001 - informational only
        info["coreai_version_error"] = str(e)
    info["os_build"] = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return info


# ------------------------------------------------------------------ convert
def convert(module, example, model_path, optimize):
    from coreai.runtime import AIModelAssetMetadata
    from coreai_torch import TorchConverter, get_decomp_table

    t = {}
    t0 = time.time()
    print(f"[INFO] torch.export start {time.strftime('%H:%M:%S')}", flush=True)
    ep = torch.export.export(module, args=(), kwargs=example)
    ep = ep.run_decompositions(get_decomp_table())
    t["export_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    print(f"[INFO] convert start {time.strftime('%H:%M:%S')} (export {t['export_s']}s)", flush=True)
    conv = TorchConverter().add_exported_program(
        exported_program=ep, input_names=["input_ids", "attention_mask", "label_idx"], output_names=["logits"])
    prog = conv.to_coreai()
    t["to_coreai_s"] = round(time.time() - t1, 1)
    if optimize:
        t2 = time.time()
        print(f"[INFO] optimize() start {time.strftime('%H:%M:%S')} (to_coreai {t['to_coreai_s']}s)", flush=True)
        prog.optimize()
        t["optimize_s"] = round(time.time() - t2, 1)
    else:
        print("[WARN] optimize() skipped (--no-optimize)", flush=True)
        t["optimize_s"] = None
    print(f"[INFO] === CONVERT OK === {t}", flush=True)
    if model_path.exists():
        shutil.rmtree(model_path)                                  # save_asset will not overwrite
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta = AIModelAssetMetadata()
    meta.author = "fastino (GLiNER2.5-Decide); DeBERTa-v3-large (Microsoft, MIT)"
    meta.license = "Apache-2.0"
    meta.model_description = ("GLiNER2.5-Decide zero-shot classification — DeBERTa-v3-large encoder, [L]-marker "
                              "gather and 1024-2048-1 label head in one static graph; labels are host input.")
    meta.creation_date = int(time.time())
    t3 = time.time()
    prog.save_asset(model_path, meta)
    t["save_s"] = round(time.time() - t3, 1)
    t["total_s"] = round(time.time() - t0, 1)
    return t


def bundle_name(dtype, s, mmax, relpos="baked"):
    return f"gliner25-decide_{dtype}_s{s}_m{mmax}{'' if relpos == 'baked' else '_runtime-relpos'}.aimodel"


def pick_reference(cases, S):
    """One host smoke case per shape: prefer the long-distance range (S/2 < len <= S), then multi-label,
    then the most tasks, then the longest."""
    pool = [c for c in cases if S // 2 < c["seq_len"] <= S] or [c for c in cases if c["seq_len"] <= S]
    return max(pool, key=lambda c: (any(tr["multi_label"] for tr in c["task_results"]),
                                    len(c["task_results"]), c["seq_len"], c["id"]))


def write_side_files(out_dir, snapshot, S, MMAX, dtype, ref_case, ref_inputs, ref_slices):
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(Path(snapshot) / name, tok_dir / name)
    tcfg = json.loads((tok_dir / "tokenizer_config.json").read_text())
    added = {v["content"]: int(k) for k, v in tcfg["added_tokens_decoder"].items()}
    for m, i in MARKER_IDS.items():
        assert added.get(m) == i, (m, added.get(m), i)
    ii, am, li = ref_inputs
    ref_name = f"reference_s{S}.json"
    (out_dir / ref_name).write_text(json.dumps({
        "model": MODEL_ID, "revision": MODEL_REV, "S": S, "MMAX": MMAX, "case_id": ref_case["id"],
        "selection": "longest-distance range first (S/2 < seq_len <= S), then multi-label, most tasks, longest",
        "seq_len": ref_case["seq_len"], "text": ref_case["text"], "tasks": ref_case["tasks"],
        "input_ids": ii[0].tolist(), "attention_mask": am[0].tolist(), "label_idx": li[0].tolist(),
        "task_slices": [{"task": tr["task"], "start": a, "end": b}
                        for tr, (a, b) in zip(ref_case["task_results"], ref_slices)],
        "oracle_fp32": [{"task": tr["task"], "labels": tr["labels"], "multi_label": tr["multi_label"],
                         "cls_threshold": tr["cls_threshold"], "logits": tr["logits"], "probs": tr["probs"],
                         "decision": tr["decision"]} for tr in ref_case["task_results"]],
    }, indent=1, ensure_ascii=False))
    shapes = []                                          # every baked bundle of this dtype/MMAX in out_dir
    for p in out_dir.glob(f"gliner25-decide_{dtype}_s*_m{MMAX}.aimodel"):
        m = re.fullmatch(rf"gliner25-decide_{dtype}_s(\d+)_m{MMAX}\.aimodel", p.name)
        if m:
            ref = out_dir / f"reference_s{m.group(1)}.json"
            shapes.append({"S": int(m.group(1)), "bundle": p.name, "bytes": bundle_bytes(p),
                           "reference": ref.name if ref.exists() else None})
    shapes.sort(key=lambda x: x["S"])
    (out_dir / "classifier.json").write_text(json.dumps({
        "model": MODEL_ID, "revision": MODEL_REV, "dtype": dtype, "MMAX": MMAX,
        "shapes": shapes,
        "host_shape_rule": "use the smallest S in `shapes` with len(input_ids) <= S; longer inputs are outside "
                           "these bundles (gliner2's classify_text_long chunks such text on the host)",
        "graph": {"relative_position": "bucket table baked at export (S fixed); coreai-torch 0.4.1 lowers "
                                       "aten.div.Tensor(int,int) to integer division — logs/diag_div_probe.log"},
        "inputs": {"input_ids": ["int32", [1, "S"]], "attention_mask": ["int32", [1, "S"]],
                   "label_idx": ["int32", [1, MMAX]]},
        "outputs": {"logits": ["float32", [1, MMAX]]},
        "marker_ids": MARKER_IDS,
        "pad": {"input_ids": PAD_ID, "attention_mask": "1 for real tokens, 0 for padding",
                "label_idx": "unused slots repeat the first [L] position (host never reads them)"},
        "label_idx": "the [L] marker positions of every task, concatenated in task order; the host keeps each "
                     "task's slice",
        "layout": "( [P] <task>[: <prompt>][ [DESCRIPTION] <label>: <desc>]* ( [L] <label> [L] <label> ... ) ) "
                  "[SEP_STRUCT] ( ... ) [SEP_TEXT] <text words>; every piece tokenized with tokenizer.tokenize(piece) "
                  "on its own, no CLS/SEP",
        "text": "append '.' unless the text ends with . ! ? (empty -> '.'); split with gliner2 "
                "WhitespaceTokenSplitter regex (URL | email | @handle | \\w+(?:[-_]\\w+)* | \\S, IGNORECASE); "
                "lowercase the word values only; schema strings keep their case",
        "decision": {"single_label": "softmax over the task's logits, argmax",
                     "multi_label": "sigmoid, every label with prob >= cls_threshold; none -> argmax",
                     "cls_threshold_default": DEFAULT_THRESHOLD, "temperature": 1.0},
    }, indent=2, ensure_ascii=False))
    print(f"[INFO] wrote tokenizer/ (3 files), {ref_name} ({ref_case['id']}, len {ref_case['seq_len']}), "
          f"classifier.json shapes={[x['S'] for x in shapes]}")


def aot_compile(model_path, out_dir):
    env = dict(os.environ, DEVELOPER_DIR=XCODE)
    tool = subprocess.run(["xcrun", "-f", "coreai-build"], env=env, capture_output=True, text=True,
                          check=True).stdout.strip()
    aot_dir = out_dir / "aot_macos_h16c"
    shutil.rmtree(aot_dir, ignore_errors=True)
    cmd = [tool, "compile", str(model_path), "--output", str(aot_dir), "--platform", "macOS",
           "--architecture", "h16c", "--preferred-compute", "gpu"]           # fixed shape: no efr
    print("[INFO] AOT:", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    hits = sorted(aot_dir.rglob("*.aimodelc"))
    assert hits, f"no .aimodelc under {aot_dir}"
    print(f"[INFO] AOT done in {time.time() - t0:.0f}s -> {hits[0]}", flush=True)
    return hits[0]


# ------------------------------------------------------------------ engine gate
async def engine_gate(path, cases, S, MMAX, aot, poison):
    import coreai.runtime as rt

    opts = (rt.SpecializationOptions.default() if aot else
            rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    print(f"\n[GATE-3] engine load ({'AOT .aimodelc, default()' if aot else 'JIT, preferred GPU'}) "
          f"{time.strftime('%H:%M:%S')} ...", flush=True)
    t0 = time.perf_counter()
    model = await rt.AIModel.load(str(path), opts)                 # keep `model` alive for every call
    load_s = time.perf_counter() - t0
    fn = model.load_function("main")
    print(f"[GATE-3] loaded in {load_s:.1f}s; functions={model.function_names}", flush=True)

    async def run(ii, am, li):
        res = await asyncio.wait_for(fn(inputs={
            "input_ids": rt.NDArray(np.ascontiguousarray(ii)),
            "attention_mask": rt.NDArray(np.ascontiguousarray(am)),
            "label_idx": rt.NDArray(np.ascontiguousarray(li)),
        }), timeout=900)
        return np.asarray(res["logits"].numpy(), np.float32)

    passes = [("clean", False)] + ([("poison", True)] if poison else [])
    out = {"load_s": round(load_s, 2)}
    for name, poisoned in passes:
        rows = []
        for i, c in enumerate(cases):
            ii, am, li, slices = graph_inputs(c, S, MMAX, poison=poisoned)
            t1 = time.perf_counter()
            logits = await run(ii, am, li)
            ms = (time.perf_counter() - t1) * 1e3
            assert logits.shape == (1, MMAX), logits.shape
            row = logits[0].astype(np.float64)
            n = slices[-1][1]
            ref_all = [x for tr in c["task_results"] for x in tr["logits"]]
            finite = bool(np.all(np.isfinite(row)))
            tasks = compare_case(c, row, slices)
            rows.append({"id": c["id"], "seq_len": c["seq_len"], "n_labels": n, "call_ms": round(ms, 2),
                         "first_call": i == 0, "finite": finite, "all_zero": bool(np.all(row[:n] == 0)),
                         "tasks": tasks, "cos": cos(row[:n], ref_all), "logits": row[:n].tolist()})
            ok = all(t["decision_equal"] for t in tasks)
            if name == "clean" and (not ok or i < 3 or i % 50 == 0):
                print(f"   [{i:3d}] {'OK ' if ok else 'DIFF'} {c['id']:34s} len={c['seq_len']:3d} "
                      f"max|dlogit|={max(t['max_abs_dlogit'] for t in tasks):.4f} "
                      f"max|dprob|={max(t['max_abs_dprob'] for t in tasks):.5f} {ms:7.1f} ms", flush=True)
            if name == "clean" and not ok:
                for t in tasks:
                    if not t["decision_equal"]:
                        print(f"         task {t['task']}: engine {t['decision']} vs oracle "
                              f"{t['oracle_decision']} (oracle margin {t['oracle_margin']:.5f})", flush=True)
        s = summarize(rows)
        ms_all = [r["call_ms"] for r in rows]
        warm = [r["call_ms"] for r in rows if not r["first_call"]] or ms_all
        s.update({"nonfinite_cases": sum(not r["finite"] for r in rows),
                  "all_zero_cases": sum(r["all_zero"] for r in rows),
                  "first_call_ms": ms_all[0], "warm_call_ms_median": round(float(np.median(warm)), 2),
                  "warm_call_ms_p90": round(float(np.percentile(warm, 90)), 2),
                  "warm_call_ms_max": round(float(np.max(warm)), 2)})
        out[name] = {"summary": s, "cases": rows}
        print(f"[GATE-3:{name}] {json.dumps(s)}", flush=True)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="+", required=True)
    ap.add_argument("-S", type=int, default=256)
    ap.add_argument("--mmax", type=int, default=32)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--output-dir", default=str(exports_dir()))
    ap.add_argument("--results-dir", default=None, help="default: <output-dir>/../results")
    ap.add_argument("--no-convert", action="store_true", help="GATE-1/2 only")
    ap.add_argument("--no-optimize", action="store_true", help="skip prog.optimize() (if it runs > 30 min)")
    ap.add_argument("--engine-only", action="store_true", help="GATE-3 on an existing bundle")
    ap.add_argument("--poison", action="store_true", help="also run poisoned label_idx through the engine")
    ap.add_argument("--aot", action="store_true", help="coreai-build compile (macOS h16c gpu) and gate the .aimodelc")
    ap.add_argument("--cases", default=None, help="comma-separated case ids (single-case re-runs)")
    ap.add_argument("--gate-name", default=None, help="results json/log stem; default gate_s<S>_gpu")
    ap.add_argument("--relpos", choices=["baked", "runtime"], default="baked",
                    help="baked: bucket table computed in torch at export (default); runtime: stock DebertaV2Model "
                         "forward, kept as negative evidence")
    args = ap.parse_args()
    S, MMAX = args.S, args.mmax
    out_dir = Path(args.output_dir).expanduser()
    res_dir = Path(args.results_dir).expanduser() if args.results_dir else out_dir.parent / "results"
    model_path = out_dir / bundle_name(args.dtype, S, MMAX, args.relpos)
    suffix = "" if args.relpos == "baked" else "_runtime_relpos"
    gate_name = args.gate_name or f"gate_s{S}_gpu{suffix}"
    snapshot = hf_snapshot(MODEL_ID, revision=MODEL_REV)
    cases, fx_headers = load_fixtures(args.fixtures, args.cases)
    print(f"[INFO] {len(cases)} fixture cases, {sum(len(c['task_results']) for c in cases)} tasks; "
          f"S={S} MMAX={MMAX} dtype={args.dtype} relpos={args.relpos}; snapshot {snapshot}")
    info = env_info()
    print(f"[INFO] env {info}")
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    convert_times, bundle_size, gate1 = None, None, None

    if not args.engine_only:
        module = build_module(snapshot, S, args.relpos)
        t0 = time.time()
        g1 = torch_gate(module, cases, S, MMAX, poison=False)
        s1 = summarize(g1)
        print(f"\n[GATE-1] fp32 re-authored vs oracle: {json.dumps(s1)}  ({time.time() - t0:.0f}s)")
        for r in g1:
            for t in r["tasks"]:
                if not t["decision_equal"] or t["max_abs_dlogit"] > 1e-3:
                    print(f"   FAIL {r['id']} {t['task']}: {t['decision']} vs {t['oracle_decision']} "
                          f"max|dlogit|={t['max_abs_dlogit']:.2e}")
        g1_pass = s1["decisions_equal_tasks"] == s1["tasks"] and s1["max_abs_dlogit"] <= 1e-3
        print(f"[GATE-1] {'PASS' if g1_pass else 'FAIL'}")
        t0 = time.time()
        g2 = torch_gate(module, cases, S, MMAX, poison=True)
        s2 = summarize(g2)
        print(f"[GATE-2] poisoned label_idx (+1) vs oracle: {json.dumps(s2)}  ({time.time() - t0:.0f}s)")
        g2_pass = s2["cases_with_changed_decision"] >= 1
        print(f"[GATE-2] {'PASS (the comparator goes red)' if g2_pass else 'FAIL: poisoned run changed nothing'}")
        gate1 = {"gate1_fp32": {"pass": g1_pass, "summary": s1, "cases": g1},
                 "gate2_poison": {"pass": g2_pass, "summary": s2, "cases": g2}}
        write_json(res_dir / f"gate_s{S}_fp32{suffix}.json", {
            "header": {"kind": "gliner25-decide-gate-fp32", "created": started, "model_rev": MODEL_REV,
                       "S": S, "MMAX": MMAX, "relpos": args.relpos, "device": "cpu", "dtype": "float32", "env": info,
                       "fixtures": fx_headers,
                       "gate1": "decisions 100% equal and max|dlogit| <= 1e-3 vs the gliner2 2.0.0 fp32 oracle",
                       "gate2": "label_idx + 1 (first label sub-word instead of [L]) must change >= 1 decision"},
            **gate1})
        if not (g1_pass and g2_pass):
            raise SystemExit("[STOP] GATE-1/2 not green")
        if args.no_convert:
            print("[SKIP] convert")
            return

        ref_case = pick_reference(cases, S)
        ii, am, li, ref_slices = graph_inputs(ref_case, S, MMAX)
        if args.dtype == "float16":
            module.half()                                  # rel_pos stays int64 (half() casts floating tensors only)
        example = {"input_ids": torch.from_numpy(ii), "attention_mask": torch.from_numpy(am),
                   "label_idx": torch.from_numpy(li)}
        if model_path.exists():
            print(f"[INFO] replacing {model_path}")
        convert_times = convert(module, example, model_path, optimize=not args.no_optimize)
        bundle_size = bundle_bytes(model_path)
        print(f"[INFO] saved {model_path} ({bundle_size / 1e6:.1f} MB) times={convert_times}")
        if args.relpos == "baked":                         # side files describe the shipped (baked) bundles only
            write_side_files(out_dir, snapshot, S, MMAX, args.dtype, ref_case, (ii, am, li), ref_slices)
        del module

    if not model_path.exists():
        raise SystemExit(f"[STOP] no bundle at {model_path}")
    bundle_size = bundle_size or bundle_bytes(model_path)
    gate_path = model_path
    if args.aot:
        gate_path = aot_compile(model_path, out_dir)
    g3 = asyncio.run(engine_gate(gate_path, cases, S, MMAX, args.aot, args.poison))
    s3 = g3["clean"]["summary"]
    g3_pass = s3["decisions_equal_tasks"] == s3["tasks"] and s3["nonfinite_cases"] == 0
    print(f"[GATE-3] {'PASS' if g3_pass else 'FAIL'}: decisions {s3['decisions_equal_tasks']}/{s3['tasks']} "
          f"max|dlogit| {s3['max_abs_dlogit']:.4f} max|dprob| {s3['max_abs_dprob']:.5f} min cos {s3['min_cos']:.6f}")
    if args.poison:
        sp = g3["poison"]["summary"]
        print(f"[GATE-3:poison] cases with a changed decision {sp['cases_with_changed_decision']}/{sp['cases']} "
              f"max|dlogit| {sp['max_abs_dlogit']:.3f}")
    write_json(res_dir / f"{gate_name}.json", {
        "header": {"kind": "gliner25-decide-gate-engine", "created": started, "model_rev": MODEL_REV,
                   "bundle": str(gate_path), "bundle_bytes": bundle_size, "dtype": args.dtype, "S": S,
                   "MMAX": MMAX, "relpos": args.relpos,
                   "path": "AOT .aimodelc (coreai-build macOS h16c gpu), default()" if args.aot
                   else "JIT, SpecializationOptions.from_preferred_compute_unit_kind(gpu)",
                   "timing_note": "contended: the Mac GPU is shared with other sessions; call_ms is wall time "
                                  "per call, not a benchmark",
                   "env": info, "fixtures": fx_headers, "convert_times_s": convert_times,
                   "gate3": "decisions 100% equal vs the gliner2 2.0.0 fp32 oracle; max|dlogit| and max|dprob| "
                            "reported (no threshold)",
                   "pass": g3_pass},
        **({"gate1_gate2_summary": {k: {"pass": v["pass"], "summary": v["summary"]} for k, v in gate1.items()}}
           if gate1 else {}),
        "engine": g3})


if __name__ == "__main__":
    main()
