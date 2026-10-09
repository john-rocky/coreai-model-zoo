#!/usr/bin/env python3
"""Which layers carry the int8lin body's error? fp32 torch with the exported int8 weights, per config (d1-3B).

conversion/clef_flash/int8_bisect_torch.py's and conversion/kev/int8_bisect_torch.py's instrument on this port. The
module is `lfm2_d1_decoder.Lfm2D1Decoder` in fp32 on the CPU (`parity_decoder_torch.real_runner`, 1 thread: round 4's
P1 instrument), driven the way the graph runs: one row per question (the oracle's `row_ids`), fresh zero states,
S = 16 chunks, the last chunk padded with <|pad|> 124893; the slot's hidden row goes through the host's readout on the
module's tied embedding rows and the per-question softmax is compared with the oracle (`oracle/records_oracle.json`).
What changes per config is which of the 134 int8lin linears (feed_forward.gate_proj / up_proj / down_proj of the 30
layers, conv.in_proj / out_proj of the 22 conv layers) are int8: each holds either its checkpoint weight (bf16, exact
in fp32) or the weight the exported op computes - the exporter's own `export_decoder.quantize` on the exporter's own
fp16 load (`export_decoder.load_real`, the S = 16 export spec), read back through the finalized module's
dequantization (`coreai::constexpr_blockwise_shift_scale` on the int8 codes and fp16 block scales, fp16 out), cast to
fp32. The attention projections, the embedding, the conv1d and the norms hold the checkpoint's weights in every
config. Only those weights differ between configs; activations stay fp32.

    rule   the bisect rows (the int8lin gate's worst), the configs and the selection rule -> <work>/rule.json, written
           once, before any bisect result
    dump   the exact (bf16) and the int8 (dequantized fp16) weights of the 134 linears, per layer, once
    run    evaluate configs on the bisect rows; one process, one part file, resumable; at the end the process puts every
           weight back to exact and re-runs its first row (swap-back proof)
    plan   what the rule asks for next, from the evaluated configs (JSON on stdout)
    merge  collect the part files, apply the rule -> one transcript

Configs: `exact` (no int8: the instrument's floor), `all_int8` (the exported int8lin body), `only_mlp_int8` /
`only_conv_int8` (that kind int8, the other exact: a map of the error by kind, never chosen), `layer_NN_fp16` (layer
NN's linears exact, every other int8; 30) and layer sets `set_L03_L07` (those layers exact, every other int8).

    cd conversion/d1
    PY=<coreai-models venv>/bin/python Q="$HOME/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 --"
    $PY int8_bisect_torch.py rule --gate $K/results/<int8lin gate>.json
    $Q $PY int8_bisect_torch.py dump
    $Q $PY int8_bisect_torch.py run --part $K/bisect/parts/part_a.json --configs all_int8,exact,..
    $PY int8_bisect_torch.py plan
    $Q $PY int8_bisect_torch.py merge --out $K/results/<bisect>.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

LANE = work_path("_d1_3b")
ORACLE = LANE / "oracle" / "records_oracle.json"
PARITY = LANE / "parity"
WORK = LANE / "bisect"
DUMP = WORK / "dump" / "int8lin"
EXACT = WORK / "dump" / "exact"
DUMP_META = WORK / "dump" / "int8lin.json"
RULE_PATH = WORK / "rule.json"
PARTS = WORK / "parts"
N_LAYERS = 30
CHUNK = 16
MAX_CTX = 4096
N_ROWS = 6
OUTSIDE_BATCH = 8          # out-of-rule sets `plan` hands out at once (parallel processes; decided in order)
LIMITS = {"worst_max_abs_dp": 0.010, "max_fp16_layers": 6, "floor_max_abs_dp": 1e-4, "reproduction_ratio": 0.5}
MLP = ("feed_forward.gate_proj", "feed_forward.up_proj", "feed_forward.down_proj")
CONV = ("conv.in_proj", "conv.out_proj")
RULE = {
    "rows": f"the {N_ROWS} runs with the highest max|dp| in the int8lin gate transcript (ties: the oracle's record and "
            "question order); a row whose row_ids equal an earlier pick's is counted once (the fixture holds identical "
            "records)",
    "instrument": "fp32 torch on the CPU, 1 thread (parity_decoder_torch.real_runner / Runner: S = 16 chunks from "
                  "fresh zero states, the oracle's row ids, the host's readout on the module's tied embedding); per "
                  "config each of the 134 int8lin linears holds its checkpoint weight (bf16, exact) or the exporter's "
                  "int8 weight (export_decoder.quantize on export_decoder.load_real's fp16 model, the finalized "
                  "module's dequantization, fp16 -> fp32); every other weight is the checkpoint's",
    "reproduction_check": f"before any selection: all_int8's worst max|dp| over the bisect rows >= "
                          f"{LIMITS['reproduction_ratio']} x the int8lin gate's worst over the same rows; otherwise stop "
                          "and report (the graph's error would not be the int8 weights')",
    "instrument_floor": f"exact's worst max|dp| against the oracle <= {LIMITS['floor_max_abs_dp']} (P1's bar); a bisect "
                        "row of the round-4 P1 subset reproduces its P1 probabilities bit for bit; every process's "
                        "swap-back re-run (every weight exact again, its first row) reproduces exact's probabilities "
                        "bit for bit",
    "1": f"worst bisect-row max|dp| <= {LIMITS['worst_max_abs_dp']} (every option of every question, near-ties "
         "included; this instrument)",
    "2": "no bisect row's max|dp| above its all_int8 value (this instrument)",
    "3": f"at most {LIMITS['max_fp16_layers']} fp16 layers (a layer = every int8lin linear of it: gate / up / down, "
         "and in_proj / out_proj in a conv layer)",
    "4": "the smallest layer set meeting 1-3; between sets of one size the lower mean over the bisect rows of the row's "
         "mean |dp|, then the lower worst, then the lexicographically smallest sorted layer list",
    "ranking": "the 30 layers by their layer_NN_fp16 config's mean over the bisect rows of the row's mean |dp| "
               "(ascending; ties: the lower worst, then the lower index); not by the worst row (a worst-row ranking "
               "puts layers that change nothing ahead: Kev, lane memory)",
    "search": [
        "1. fixed: exact, all_int8, only_mlp_int8, only_conv_int8, layer_NN_fp16 for NN = 0..29",
        "2. a single layer meeting 1-3 ends the search at size 1 (every single is evaluated)",
        "3. else L_k = the top k layers of the ranking for k = 2..6, taken in order; at the first k whose L_k meets "
        "1-3, also every other k-subset of the top k+1 layers; stop there (sets may be evaluated together in "
        "parallel processes; a set evaluated past the stop is recorded and never chosen)",
    ],
    "outside_rule": "only if steps 1-3 find no set: L_k for k = 7, 8, .. (the same ranking), taken in order, until the "
                    "first L_k meeting 1-2; that set is the out-of-rule candidate, exported and gated, reported as "
                    "needing a held-out set (fixture rows chose it), never chosen",
    "map_only": "only_mlp_int8 and only_conv_int8 map the error by kind; they are never chosen",
    "then": "the chosen set (or the out-of-rule candidate): export_decoder.py int8mix --fp16-layers <set> --aot, then "
            "readout_gate.py run --red --compare-with <the fp16 transcript> on every row with the gate's own bar",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def layer_of(name: str) -> int:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def kind_of(name: str) -> str:
    leaf = ".".join(name.split(".")[3:])
    return "mlp" if leaf in MLP else "conv" if leaf in CONV else "other"


def row_key(rid: str, k: int) -> str:
    return f"{rid}:q{k}"


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
def set_name(layers) -> str:
    ls = sorted(set(layers))
    return f"layer_{ls[0]:02d}_fp16" if len(ls) == 1 else "set_" + "_".join(f"L{i:02d}" for i in ls)


def fixed_configs() -> list[str]:
    return ["all_int8", "exact", "only_mlp_int8", "only_conv_int8"] + [f"layer_{i:02d}_fp16" for i in range(N_LAYERS)]


def config(names: list[str], name: str) -> dict:
    """name -> {"int8": [module names that are int8], "layers": [fp16 layers] | None, "why": ..}."""
    every = sorted(names)
    if name == "all_int8":
        return {"int8": every, "layers": [], "why": "the export's int8lin body: every linear int8"}
    if name == "exact":
        return {"int8": [], "layers": None, "why": "no int8: the instrument's floor vs the oracle"}
    if name in ("only_mlp_int8", "only_conv_int8"):
        k = name.split("_")[1]
        return {"int8": [n for n in every if kind_of(n) == k], "layers": None,
                "why": f"the {k} linears int8, the rest exact (map only)"}
    if m := re.fullmatch(r"layer_(\d\d)_fp16", name):
        layers = [int(m.group(1))]
    elif name.startswith("set_L"):
        layers = sorted(int(t[1:]) for t in name[len("set_"):].split("_"))
    else:
        raise SystemExit(f"unknown config {name}")
    if any(not 0 <= i < N_LAYERS for i in layers) or set_name(layers) != name:
        raise SystemExit(f"{name}: not a layer set of 0..{N_LAYERS - 1} in canonical form")
    return {"int8": [n for n in every if layer_of(n) not in set(layers)], "layers": layers,
            "why": f"layers {layers} exact, every other int8lin linear int8"}


# --------------------------------------------------------------------------- #
# rule
# --------------------------------------------------------------------------- #
def rule(args) -> None:
    from parity_decoder_torch import all_rows, load_oracle

    if RULE_PATH.exists():
        sys.exit(f"{RULE_PATH} exists (the rule is written once, before any result)")
    if (PARTS.exists() and any(PARTS.iterdir())) or DUMP_META.exists():
        sys.exit(f"{PARTS} or {DUMP_META} exists: the rule comes first")
    gate_path = Path(args.gate).resolve()
    gate = json.loads(gate_path.read_text())
    if gate["oracle"]["sha256"] != sha256(ORACLE):
        sys.exit(f"{gate_path} was read against another oracle")
    if (gate["bundle"].get("compression") or {}).get("scheme") != "int8lin":
        sys.exit(f"{gate_path} is not an int8lin gate: {gate['bundle']['name']}")
    _, recs = load_oracle(ORACLE)
    order = {rk: n for n, rk in enumerate(all_rows(recs))}
    runs = sorted(gate["runs"], key=lambda r: (-r["max_abs_dp"], order[(r["id"], r["k"])]))
    rows, seen, skipped = [], {}, []
    for r in runs:
        ids = tuple(recs[r["id"]]["questions"][r["k"]]["row_ids"])
        if ids in seen:
            skipped.append({"row": row_key(r["id"], r["k"]), "same_row_ids_as": seen[ids], "max_abs_dp": r["max_abs_dp"]})
            continue
        seen[ids] = row_key(r["id"], r["k"])
        rows.append(r)
        if len(rows) == N_ROWS:
            break
    lens = [recs[r["id"]]["questions"][r["k"]]["row_len"] for r in rows]
    doc = {"schema": "d1-int8-bisect-rule/1", "rule": RULE, "limits": LIMITS,
           "rows": [[r["id"], r["k"]] for r in rows],
           "rows_detail": [{"row": row_key(r["id"], r["k"]), "name": r["name"], "source": r["source"],
                            "row_len": recs[r["id"]]["questions"][r["k"]]["row_len"],
                            "near_tie": bool(recs[r["id"]]["questions"][r["k"]]["near_tie"]),
                            "oracle_top2_margin": recs[r["id"]]["questions"][r["k"]]["top2_margin"],
                            "gate_int8lin_max_abs_dp": r["max_abs_dp"], "gate_int8lin_mean_abs_dp": r["mean_abs_dp"]}
                           for r in rows],
           "skipped_identical_rows": skipped,
           "gate_transcript": {"path": str(gate_path), "sha256": sha256(gate_path), "result": gate["result"],
                               "max_abs_dp": gate["summary"]["max_abs_dp"], "bundle": gate["bundle"]["name"],
                               "aimodelc_tree_sha256": gate["bundle"]["aimodelc"]["tree_sha256"]},
           "oracle": {"path": str(ORACLE), "sha256": sha256(ORACLE)},
           "configs_fixed": fixed_configs(), "chunk": CHUNK, "tokens": int(sum(lens)),
           "script_sha256": sha256(Path(__file__).resolve()), "written_at": now()}
    RULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE_PATH.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"rule -> {RULE_PATH} ({sha256(RULE_PATH)}): rows {[row_key(r['id'], r['k']) for r in rows]} "
          f"({sum(lens):,} tokens); skipped identical {[s['row'] for s in skipped]}", flush=True)


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def dump(args) -> None:
    import torch
    import torch.nn.utils.parametrize as P
    from lfm2_d1_decoder import N_IMAGE_TOKENS, Lfm2D1Decoder
    from safetensors.torch import save_file

    import export_decoder as ed
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN

    if DUMP_META.exists():
        sys.exit(f"{DUMP_META} exists: the dump is written once")
    if not RULE_PATH.exists():
        sys.exit(f"no {RULE_PATH}: the rule comes first")
    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    # the exact weights: the fp32 module's own (the checkpoint's bf16, widened exactly), as bf16
    m32 = Lfm2D1Decoder.from_hf(str(ed.snapshot()), target_dtype=torch.float32, fp32_attn_proj=True)
    names = ed.intended_quantized(m32, [])
    mods32 = dict(m32.named_modules())
    EXACT.mkdir(parents=True, exist_ok=True)
    fp16_inexact, files_exact = {}, {}
    for li in sorted({layer_of(n) for n in names}):
        ws = {}
        for n in names:
            if layer_of(n) != li:
                continue
            w = mods32[n].weight.detach()
            b = w.to(torch.bfloat16)
            if not torch.equal(b.float(), w):
                sys.exit(f"{n}: the fp32 weight is not bf16-exact")
            fp16_inexact[n] = int((w.half().float() != w).sum())
            ws[n] = b.contiguous()
        p = EXACT / f"layer_{li:02d}.safetensors"
        save_file(ws, str(p))
        files_exact[str(li)] = {"file": str(p), "sha256": sha256(p), "modules": sorted(ws)}
    del m32, mods32
    t_exact = time.monotonic() - t0
    # the int8 weights: the exporter's own fp16 load, export spec and quantize call
    model, load = ed.load_real(argparse.Namespace(n_image_tokens=N_IMAGE_TOKENS))
    spec = model.build_export_spec(torch.float16, MAX_CTX, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=CHUNK)
    before = {n: m.weight.detach().clone() for n, m in model.named_modules() if n in set(names)}
    t1 = time.monotonic()
    model, quant = ed.quantize(model, spec, "int8lin", 32, [])
    t_q = time.monotonic() - t1
    DUMP.mkdir(parents=True, exist_ok=True)
    per_layer: dict[int, dict] = {}
    info = {}
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod, "weight"):
            continue
        deq = [p for p in mod.parametrizations["weight"] if hasattr(p, "quantized_data")]
        assert len(deq) == 1, (name, [type(p).__name__ for p in mod.parametrizations["weight"]])
        d = deq[0]
        w = mod.weight.detach().clone()                    # the finalized module's own dequantization
        again = torch.ops.coreai.constexpr_blockwise_shift_scale(
            d.quantized_data, d.scale, zero_point=d.zero_point, minval=d.minval,
            input_dtype=d.input_dtype, output_dtype=d.output_dtype)
        if not torch.equal(w, again):
            sys.exit(f"{name}: the module's weight differs from constexpr_blockwise_shift_scale on its codes")
        q, ref = d.quantized_data, before[name].float()
        info[name] = {"kind": kind_of(name), "shape": list(w.shape), "dtype": str(w.dtype), "codes_dtype": str(q.dtype),
                      "codes_min": int(q.min()), "codes_max": int(q.max()), "scale_shape": list(d.scale.shape),
                      "scale_dtype": str(d.scale.dtype),
                      "zero_point_absmax": None if d.zero_point is None else int(d.zero_point.abs().max()),
                      "rel_err": float((w.float() - ref).norm() / ref.norm()),
                      "max_abs_err": float((w.float() - ref).abs().max()),
                      "bf16_to_fp16_inexact_elements": fp16_inexact[name]}
        per_layer.setdefault(layer_of(name), {})[name] = w.contiguous()
    if sorted(info) != names:
        sys.exit(f"the dumped set differs from export_decoder.intended_quantized: {len(info)} vs {len(names)}")
    files = {}
    for li, ws in sorted(per_layer.items()):
        p = DUMP / f"layer_{li:02d}.safetensors"
        save_file(ws, str(p))
        files[str(li)] = {"file": str(p), "sha256": sha256(p), "modules": sorted(ws)}
    rel = [v["rel_err"] for v in info.values()]
    meta = {"schema": "d1-int8-bisect-dump/1", "quantization": {k: v for k, v in quant.items() if k != "fp16_linear_modules"},
            "load_report": {k: load["load_report"][k] for k in ("checkpoint_keys_under_prefix", "module_tensors",
                                                                 "unread_checkpoint_keys_under_prefix",
                                                                 "module_tensors_not_in_checkpoint")},
            "weights": load["weights"], "export_spec_query_len": CHUNK, "max_ctx": MAX_CTX,
            "quantized_modules": len(info), "modules_by_kind": {k: sum(v["kind"] == k for v in info.values())
                                                               for k in ("mlp", "conv")},
            "params": int(sum(int(np.prod(v["shape"])) for v in info.values())),
            "codes_range": [min(v["codes_min"] for v in info.values()), max(v["codes_max"] for v in info.values())],
            "rel_err": {"median": float(np.median(rel)), "max": float(np.max(rel)), "min": float(np.min(rel))},
            "bf16_to_fp16_inexact_elements": int(sum(fp16_inexact.values())),
            "seconds": {"exact": t_exact, "quantize": t_q, "total": time.monotonic() - t0},
            "files": files, "files_exact": files_exact, "modules": info,
            "exact_weights": "the fp32 module's own weights (Lfm2D1Decoder.from_hf fp32 = the checkpoint's bf16 "
                             "widened), stored as bf16 (checked exact)",
            "export_decoder_sha256": sha256(HERE / "export_decoder.py"),
            "script_sha256": sha256(Path(__file__).resolve()), "generated_at": now()}
    DUMP_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dumped {len(info)} modules ({meta['params']:,} params): codes {meta['codes_range']}, rel err median "
          f"{meta['rel_err']['median']:.4e} max {meta['rel_err']['max']:.4e}; exact {t_exact:.0f}s, quantize "
          f"{t_q:.0f}s -> {DUMP.parent}", flush=True)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
class Weights:
    """Moves the 134 linears' weights between exact (the bf16 dump) and int8 (the int8 dump), read per tensor through
    safetensors' mmap: the process keeps one fp32 module and nothing else."""

    def __init__(self, model):
        import torch
        from safetensors import safe_open

        self.torch = torch
        meta = json.loads(DUMP_META.read_text())
        self.names = sorted(meta["modules"])
        mods = dict(model.named_modules())
        self.mods = {n: mods[n] for n in self.names}
        assert all(isinstance(m, torch.nn.Linear) for m in self.mods.values())
        layers = sorted({layer_of(n) for n in self.names})
        self.files = {s: {li: safe_open(str(d / f"layer_{li:02d}.safetensors"), framework="pt", device="cpu")
                          for li in layers} for s, d in (("int8", DUMP), ("exact", EXACT))}
        self.params = {n: int(self.mods[n].weight.numel()) for n in self.names}
        self.state = {n: "exact" for n in self.names}
        # the module holds the checkpoint's own values: the first and the last layer against the exact dump
        probe = [n for n in self.names if layer_of(n) in (layers[0], layers[-1])]
        self.probe_bit_equal = {n: bool(torch.equal(self.mods[n].weight,
                                                    self.files["exact"][layer_of(n)].get_tensor(n).float()))
                                for n in probe}
        if not all(self.probe_bit_equal.values()):
            raise SystemExit(f"the module's weights differ from the exact dump: {self.probe_bit_equal}")

    def apply(self, int8: list[str]) -> int:
        want = set(int8)
        changed = 0
        with self.torch.no_grad():
            for n, m in self.mods.items():
                s = "int8" if n in want else "exact"
                if self.state[n] == s:
                    continue
                m.weight.copy_(self.files[s][layer_of(n)].get_tensor(n).to(self.torch.float32))
                self.state[n] = s
                changed += 1
        return changed


def p1_probs() -> dict:
    out = {}
    for p in sorted(PARITY.glob("p1_shard*of*.jsonl")):
        for line in p.read_text().splitlines():
            x = json.loads(line)
            if x.get("kind") == "p1":
                out[(x["id"], x["q"])] = x["probs"]
    return out


def run(args) -> None:
    from parity_decoder_torch import load_oracle, real_runner, score_probs

    rdoc = json.loads(RULE_PATH.read_text())
    rows = [(rid, int(k)) for rid, k in rdoc["rows"]]
    _, recs = load_oracle(ORACLE)
    p1 = p1_probs()
    part_path = Path(args.part)
    part = json.loads(part_path.read_text()) if part_path.exists() else {"configs": {}}
    t0 = time.monotonic()
    runner = real_runner(args.threads)
    W = Weights(runner.model)
    load_s = time.monotonic() - t0
    body = sum(W.params.values())
    todo = [c for c in args.configs.split(",") if c]
    cfgs = {c: config(W.names, c) for c in todo}
    part.update({"pid": os.getpid(), "threads": args.threads, "chunk": CHUNK, "rows": [list(k) for k in rows],
                 "rule_sha256": sha256(RULE_PATH), "script_sha256": sha256(Path(__file__).resolve()),
                 "dump_meta_sha256": sha256(DUMP_META), "load_seconds": load_s, "body_linear_params": body,
                 "probe_bit_equal_exact_dump": all(W.probe_bit_equal.values()),
                 "load_report_ok": not any(runner.load_report[k] for k in ("unread_checkpoint_keys_under_prefix",
                                                                           "module_tensors_not_in_checkpoint")),
                 "started": part.get("started", now())})

    def save() -> None:
        tmp = part_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, part_path)

    def one_row(rid: str, k: int) -> dict:
        q = recs[rid]["questions"][k]
        t1 = time.monotonic()
        h, info = runner.chunked(q["row_ids"], CHUNK)
        sc = score_probs(runner.readout(h[q["slot"]], q["groups"]), q)
        out = {x: sc[x] for x in ("max_abs_dp", "mean_abs_dp", "argmax", "argmax_oracle", "argmax_equal", "near_tie",
                                  "probs")}
        out.update({"tokens": info["T"], "chunks": info["chunks"], "finite": bool(np.isfinite(h).all()),
                    "seconds": time.monotonic() - t1})
        ref = p1.get((rid, k))
        out["bit_equal_p1"] = None if ref is None else bool(sc["probs"] == ref)
        return out

    print(f"[{os.getpid()}] loaded in {load_s:.0f}s; rows {[row_key(*r) for r in rows]}; configs {todo}", flush=True)
    for name in todo:
        if name in part["configs"] and not args.redo:
            print(f"[{os.getpid()}] {name}: done already", flush=True)
            continue
        c = cfgs[name]
        t1 = time.monotonic()
        changed = W.apply(c["int8"])
        int8_params = int(sum(W.params[n] for n in c["int8"]))
        rec = {"why": c["why"], "layers": c["layers"], "int8_modules": len(c["int8"]),
               "exact_modules": len(W.names) - len(c["int8"]), "int8_params": int8_params,
               "fp16_params": body - int8_params, "modules_swapped": changed, "swap_seconds": time.monotonic() - t1,
               "rows": {}}
        for rid, k in rows:
            rec["rows"][row_key(rid, k)] = one_row(rid, k)
        rr = list(rec["rows"].values())
        rec["worst_max_abs_dp"] = max(v["max_abs_dp"] for v in rr)
        rec["mean_of_row_mean_abs_dp"] = float(np.mean([v["mean_abs_dp"] for v in rr]))
        rec["argmax_equal_non_near_tie"] = all(v["argmax_equal"] for v in rr if not v["near_tie"])
        rec["finite"] = all(v["finite"] for v in rr)
        rec["seconds"] = time.monotonic() - t1
        rec["finished"] = now()
        part["configs"][name] = rec
        save()
        worst = max(rec["rows"].items(), key=lambda kv: kv[1]["max_abs_dp"])
        print(f"[{os.getpid()}] {name}: worst {rec['worst_max_abs_dp']:.5f} ({worst[0]}) mean "
              f"{rec['mean_of_row_mean_abs_dp']:.6f} ({rec['seconds']:.0f}s, {changed} swapped)", flush=True)
    # swap-back proof: every weight exact again, the first row again (merge compares it with `exact`)
    W.apply([])
    rid, k = rows[0]
    again = one_row(rid, k)
    part["swap_back_check"] = {"row": row_key(rid, k), "probs": again["probs"], "bit_equal_p1": again["bit_equal_p1"],
                               "max_abs_dp_vs_oracle": again["max_abs_dp"], "at": now()}
    part["finished"] = now()
    save()
    print(f"[{os.getpid()}] swap-back {row_key(rid, k)}: max|dp| {again['max_abs_dp']:.3e}, bit-equal P1 "
          f"{again['bit_equal_p1']}", flush=True)


# --------------------------------------------------------------------------- #
# plan / merge
# --------------------------------------------------------------------------- #
def load_parts() -> tuple[dict, list[dict]]:
    table, procs = {}, []
    for p in sorted(PARTS.glob("part_*.json")):
        d = json.loads(p.read_text())
        procs.append({"part": str(p), **{k: d.get(k) for k in ("pid", "threads", "load_seconds", "rule_sha256",
                                                               "script_sha256", "dump_meta_sha256", "started",
                                                               "finished", "probe_bit_equal_exact_dump",
                                                               "load_report_ok", "swap_back_check")},
                      "configs": list(d["configs"])})
        for name, rec in d["configs"].items():
            if name in table:
                rec["repeat_bit_equal"] = ({r: v["probs"] for r, v in table[name]["rows"].items()}
                                           == {r: v["probs"] for r, v in rec["rows"].items()})
            table[name] = rec
    return table, procs


def meets(rec: dict, base: dict) -> dict:
    """Rules 1-3 for one evaluated config against all_int8 (the same rows, this instrument)."""
    rows = rec["rows"]
    worst = max(v["max_abs_dp"] for v in rows.values())
    not_worse = all(v["max_abs_dp"] <= base["rows"][r]["max_abs_dp"] for r, v in rows.items())
    n = len(rec["layers"]) if rec.get("layers") is not None else None
    return {"worst_max_abs_dp": worst, "mean_of_row_mean_abs_dp": rec["mean_of_row_mean_abs_dp"],
            "rule_1": worst <= LIMITS["worst_max_abs_dp"], "rule_2": not_worse,
            "rule_3": n is not None and 0 < n <= LIMITS["max_fp16_layers"], "fp16_layer_count": n}


def ok13(rec: dict, base: dict) -> bool:
    m = meets(rec, base)
    return m["rule_1"] and m["rule_2"] and m["rule_3"]


def ranking(table: dict) -> list[int]:
    return sorted(range(N_LAYERS), key=lambda i: (table[f"layer_{i:02d}_fp16"]["mean_of_row_mean_abs_dp"],
                                                  table[f"layer_{i:02d}_fp16"]["worst_max_abs_dp"], i))


def rule4_key(name: str, rec: dict) -> tuple:
    return (len(rec["layers"]), rec["mean_of_row_mean_abs_dp"], rec["worst_max_abs_dp"], sorted(rec["layers"]))


def checks(table: dict, rdoc: dict) -> dict:
    """The reproduction check and the instrument floor (before any selection)."""
    out: dict = {}
    if "all_int8" in table:
        gate = json.loads(Path(rdoc["gate_transcript"]["path"]).read_text())
        g_by = {row_key(r["id"], r["k"]): r["max_abs_dp"] for r in gate["runs"]}
        rows = [row_key(rid, k) for rid, k in rdoc["rows"]]
        x = np.array([g_by[r] for r in rows])
        y = np.array([table["all_int8"]["rows"][r]["max_abs_dp"] for r in rows])
        out["reproduction"] = {"gate_worst": float(x.max()), "torch_all_int8_worst": float(y.max()),
                               "ratio": float(y.max() / x.max()),
                               "reproduced": bool(y.max() >= LIMITS["reproduction_ratio"] * x.max()),
                               "pearson_per_row": float(np.corrcoef(x, y)[0, 1]) if len(rows) > 1 else None,
                               "per_row": [{"row": r, "gate": float(a), "torch": float(b)} for r, a, b in zip(rows, x, y)]}
    if "exact" in table:
        ex = table["exact"]["rows"]
        out["floor"] = {"worst_max_abs_dp": table["exact"]["worst_max_abs_dp"],
                        "within": table["exact"]["worst_max_abs_dp"] <= LIMITS["floor_max_abs_dp"],
                        "p1_rows": {r: v["bit_equal_p1"] for r, v in ex.items() if v["bit_equal_p1"] is not None},
                        "p1_bit_equal": all(v["bit_equal_p1"] for v in ex.values() if v["bit_equal_p1"] is not None)}
    return out


def plan_steps(table: dict, rdoc: dict, procs: list[dict] | None = None) -> dict:
    """What the rule has evaluated and what it asks for next."""
    missing = [c for c in fixed_configs() if c not in table]
    if missing:
        return {"stage": "fixed", "next": missing}
    ck = checks(table, rdoc)
    if not ck["reproduction"]["reproduced"] or not ck["floor"]["within"] or not ck["floor"]["p1_bit_equal"]:
        return {"stage": "stop", "next": [], "checks": ck, "why": "the reproduction check or the instrument floor failed"}
    base = table["all_int8"]
    rank = ranking(table)
    out = {"ranking": rank, "ranking_values": [{"layer": i, **{k: meets(table[f"layer_{i:02d}_fp16"], base)[k] for k in
                                                               ("mean_of_row_mean_abs_dp", "worst_max_abs_dp",
                                                                "rule_2")}} for i in rank],
           "checks": ck, "next": []}
    singles = [f"layer_{i:02d}_fp16" for i in range(N_LAYERS) if ok13(table[f"layer_{i:02d}_fp16"], base)]
    if singles:
        chosen = min(singles, key=lambda n: rule4_key(n, table[n]))
        return {**out, "stage": "chosen", "chosen": chosen, "meeting": singles, "search_sets": []}

    def first_in_order(names: list[str], ok) -> tuple[int | None, list[str]]:
        """The index of the first name meeting `ok` when every name before it is evaluated; else the unevaluated."""
        for i, name in enumerate(names):
            if name not in table:
                return None, [n for n in names[i:] if n not in table]
            if ok(table[name]):
                return i, []
        return None, []

    tops = [set_name(rank[:k]) for k in range(2, LIMITS["max_fp16_layers"] + 1)]
    i, todo = first_in_order(tops, lambda rec: ok13(rec, base))
    if todo:
        return {**out, "stage": "top-k", "next": todo, "search_sets": tops}
    if i is not None:
        k = i + 2
        name = tops[i]
        subs = [s for s in (set_name(c) for c in itertools.combinations(sorted(rank[:k + 1]), k)) if s != name]
        search = tops[:i + 1] + subs
        todo = [s for s in subs if s not in table]
        if todo:
            return {**out, "stage": "subsets", "next": todo, "search_sets": search}
        meeting = [s for s in [name] + subs if ok13(table[s], base)]
        chosen = min(meeting, key=lambda n: rule4_key(n, table[n]))
        return {**out, "stage": "chosen", "chosen": chosen, "meeting": meeting, "search_sets": search,
                "evaluated_past_stop": [s for s in tops[i + 1:] if s in table]}
    outside = [set_name(rank[:k]) for k in range(LIMITS["max_fp16_layers"] + 1, N_LAYERS)]
    out["search_sets"] = tops

    def ok12(rec: dict) -> bool:
        m = meets(rec, base)
        return m["rule_1"] and m["rule_2"]

    i, todo = first_in_order(outside, ok12)
    if todo:
        return {**out, "stage": "outside", "next": todo[:OUTSIDE_BATCH], "outside_sets": outside}
    if i is not None:
        return {**out, "stage": "outside_candidate", "chosen": None, "outside_rule_candidate": outside[i],
                "outside_sets": outside[:i + 1], "evaluated_past_stop": [s for s in outside[i + 1:] if s in table]}
    return {**out, "stage": "none", "chosen": None, "outside_sets": outside}


def plan(args) -> None:
    rdoc = json.loads(RULE_PATH.read_text())
    table, procs = load_parts()
    print(json.dumps(plan_steps(table, rdoc, procs), indent=1))


def merge(args) -> None:
    rdoc = json.loads(RULE_PATH.read_text())
    table, procs = load_parts()
    p = plan_steps(table, rdoc, procs)
    base = table.get("all_int8")
    exact = table.get("exact")
    swap = [{"part": x["part"], "row": (x["swap_back_check"] or {}).get("row"),
             "bit_equal_exact": (None if not exact or not x["swap_back_check"] else
                                 x["swap_back_check"]["probs"] == exact["rows"][x["swap_back_check"]["row"]]["probs"])}
            for x in procs]
    rows = [row_key(rid, k) for rid, k in rdoc["rows"]]
    search = set(p.get("search_sets", [])) | {f"layer_{i:02d}_fp16" for i in range(N_LAYERS)}
    tab = []
    for name, rec in table.items():
        r = {"config": name, "layers": rec["layers"], "int8_params": rec["int8_params"], "fp16_params": rec["fp16_params"],
             "worst_max_abs_dp": rec["worst_max_abs_dp"], "mean_of_row_mean_abs_dp": rec["mean_of_row_mean_abs_dp"],
             "argmax_equal_non_near_tie": rec["argmax_equal_non_near_tie"], "finite": rec["finite"],
             **{x: rec["rows"][x]["max_abs_dp"] for x in rows}, "seconds": rec["seconds"],
             "in_rule_search": name in search, "outside_rule": name in set(p.get("outside_sets", [])),
             "repeat_bit_equal": rec.get("repeat_bit_equal")}
        if base:
            m = meets(rec, base)
            r.update({k: m[k] for k in ("rule_1", "rule_2", "rule_3", "fp16_layer_count")})
            r["rows_not_worse_than_all_int8"] = sum(rec["rows"][x]["max_abs_dp"] <= base["rows"][x]["max_abs_dp"]
                                                    for x in rows)
        tab.append(r)
    tab.sort(key=lambda t: (t["config"] not in ("all_int8", "exact", "only_mlp_int8", "only_conv_int8"), t["config"]))
    out = {"schema": "d1-int8-bisect/1", "rule": rdoc["rule"], "limits": rdoc["limits"], "rule_file": str(RULE_PATH),
           "rule_sha256": sha256(RULE_PATH), "rule_written_at": rdoc["written_at"], "rows": rdoc["rows"],
           "rows_detail": rdoc["rows_detail"], "skipped_identical_rows": rdoc["skipped_identical_rows"],
           "gate_transcript": rdoc["gate_transcript"],
           "instrument": RULE["instrument"],
           "script": {"path": "conversion/d1/int8_bisect_torch.py", "sha256": sha256(Path(__file__).resolve())},
           "dump": {"meta": str(DUMP_META), "sha256": sha256(DUMP_META)},
           "oracle": {"path": str(ORACLE), "sha256": sha256(ORACLE)},
           "processes": procs, "swap_back": swap, "plan": p, "stage": p.get("stage"), "chosen": p.get("chosen"),
           "outside_rule_candidate": p.get("outside_rule_candidate"), "table": tab, "configs": table,
           "generated_at": now()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        sys.exit(f"{args.out} exists: never overwritten")
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} configs -> {args.out}; stage {p.get('stage')} chosen {p.get('chosen')} out-of-rule "
          f"{p.get('outside_rule_candidate')}; swap-back {[s['bit_equal_exact'] for s in swap]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rule")
    r.add_argument("--gate", required=True, help="the int8lin readout gate transcript")
    d = sub.add_parser("dump")
    d.add_argument("--threads", type=int, default=8)
    b = sub.add_parser("run")
    b.add_argument("--part", required=True)
    b.add_argument("--configs", required=True, help="comma list")
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--redo", action="store_true")
    sub.add_parser("plan")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"rule": rule, "dump": dump, "run": run, "plan": plan, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
