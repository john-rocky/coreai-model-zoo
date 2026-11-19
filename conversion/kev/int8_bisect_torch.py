#!/usr/bin/env python3
"""Which linears carry the int8 body's error? fp32 torch with the exported int8 weights, per config (Kev-0.8B).

conversion/clef_flash/int8_bisect_torch.py's instrument, text-only. The module is
`qwen3_5_kev_decoder.Qwen3_5KevDecoder` in fp32 (the merged checkpoint's own fp32 weights), driven by
`parity_decoder_torch.Runner` the way the graph runs: one row per question (the oracle's `row_ids`), fresh zero
states, S = 16 chunks (the shipped width), the last chunk padded with 248044, the bmm depthwise conv (checked
against F.conv1d at the last chunk of the first row of every process). The hidden rows go through the author's
fp32 pointer head (`parity_decoder_torch.KevHead`) and the per-question softmax is compared with the oracle
(`oracle/records_oracle.json`). What changes per config is which linear weights are int8: every module the
exporter's int8lin config quantizes (`export_decoder.linear_quant_config`: the 186 decoder linears; the
embedding, conv1d and norms excluded) holds either its fp32 checkpoint weight or the weight the exported op
computes - the exporter's own `quantize_pytorch_model` call on the fp16 model (same config, the S = 16 export
spec's reference inputs, the same GDN flags), read back through the finalized module's dequantization
(`coreai::constexpr_blockwise_shift_scale` on the int8 codes and fp16 block scales, fp16 out), cast to fp32.
Only the weights differ between configs; activations stay fp32.

    dump   quantize the fp16 model, save every quantized module's dequantized fp16 weight
           (<work>/dump/int8_clipping/layer_NN.safetensors + <work>/dump/int8_clipping.json)
    rule   write the bisect rows, the configs and the selection rule (<work>/rule.json) before any bisect result
    run    evaluate configs on the bisect rows; one process, one part file, resumable; at the end the process
           puts every weight back to fp32 and re-runs its first row against the round-2 P1 run (swap-back proof)
    plan   from the evaluated configs, what the rule asks for next (the candidate sets, in the rule's order)
    merge  collect the part files, apply the rule -> one transcript

Configs (`--configs`, comma list): `exact` (no int8: the instrument's floor; must reproduce P1's fp32 run of the
same rows bit for bit), `all_int8` (the exported int8lin body), `layer_NN_fp16` (layer NN's linears fp32, every
other int8; 24), `gdn_fp16` / `attn_fp16` / `mlp_fp16` (that kind fp32 in every layer), `proj_<kind>_fp16` (one
projection kind fp32 in every layer; kinds qkv z a b out q k v o gate up down), and candidate sets `set_L03_L07`
(layers), `set_K_down_up` (kinds), `set_L03__K_down` (both).

    cd conversion/kev
    export HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1
    PY=<coreai-models venv>/bin/python
    $PY int8_bisect_torch.py rule --gate $ZOO_WORK_ROOT/_kev/results/readout_int8lin_pf16.json
    $PY int8_bisect_torch.py dump
    $PY int8_bisect_torch.py run --part $ZOO_WORK_ROOT/_kev/bisect/parts/part_a.json --configs all_int8,exact,..
    $PY int8_bisect_torch.py plan
    $PY int8_bisect_torch.py merge --out $ZOO_WORK_ROOT/_kev/results/r3_int8_bisect.json
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import from_record_path, hf_snapshot, work_path  # noqa: E402

LANE = work_path("_kev")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WORK = LANE / "bisect"
DUMP = WORK / "dump" / "int8_clipping"
DUMP_META = WORK / "dump" / "int8_clipping.json"
RULE_PATH = WORK / "rule.json"
PARTS = WORK / "parts"
PARITY = LANE / "parity"
N_LAYERS = 24
CHUNK = 16
SEED = 20261003
N_TOP, N_RANDOM = 30, 30
BAR_MAX_ABS_DP = 0.02
TARGET = 0.5 * BAR_MAX_ABS_DP          # the rule's target on the bisect rows (strictly below)
CAP_FRACTION = 0.5                     # at most half the body's linear params stay fp32 / fp16
# projection kinds: config code -> (block, leaf module name)
KINDS = {"qkv": ("linear_attn", "in_proj_qkv"), "z": ("linear_attn", "in_proj_z"), "a": ("linear_attn", "in_proj_a"),
         "b": ("linear_attn", "in_proj_b"), "out": ("linear_attn", "out_proj"), "q": ("self_attn", "q_proj"),
         "k": ("self_attn", "k_proj"), "v": ("self_attn", "v_proj"), "o": ("self_attn", "o_proj"),
         "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"), "down": ("mlp", "down_proj")}
GROUPS = {"gdn": ("qkv", "z", "a", "b", "out"), "attn": ("q", "k", "v", "o"), "mlp": ("gate", "up", "down")}
RULE = {
    "target": f"worst max|dp| over the {N_TOP + N_RANDOM} bisect rows < {TARGET:.3f} (0.5 x the gate bar "
              f"{BAR_MAX_ABS_DP}; every option of every question, near-ties included), this instrument",
    "reproduction_check": "before any selection: all_int8's worst max|dp| over the bisect rows (this instrument) >= "
                          "0.5 x the int8lin gate transcript's worst over the same rows; otherwise stop and report "
                          "(the gate's error would not be the int8 weights')",
    "instrument_floor": "exact must reproduce the round-2 P1 run (parity/p1_shard*.jsonl) of every bisect row bit for "
                        "bit (probabilities), and every process's swap-back re-run likewise",
    "size": "a set's size = the params of the linears it keeps out of int8 (the bytes it costs)",
    "cap": f"a candidate keeps at most {CAP_FRACTION:.0%} of the body's linear params out of int8",
    "search": [
        "1. fixed configs: exact, all_int8, layer_NN_fp16 (24), gdn_fp16, attn_fp16, mlp_fp16, proj_<kind>_fp16 (12)",
        "2. layer sets: the 24 layers ranked by their layer_NN_fp16 worst (ascending; ties: the lower index); "
        "L_k = the top k, for k = 2, 3, ... while within the cap; at the first k whose L_k meets the target (k = 1: "
        "the best single), also every k-subset of the top k+1 layers",
        "3. kind sets: the 12 kinds ranked by their proj_<kind>_fp16 worst (ascending; ties: the order qkv z a b out "
        "q k v o gate up down); K_j = the top j, for j = 2, 3, ... while within the cap; at the first j whose K_j "
        "meets the target, also every j-subset of the top j+1 kinds; gdn_fp16 / attn_fp16 / mlp_fp16 are kind sets "
        "too",
        "4. only if no set of steps 1-3 meets the target: unions L_k + K_j (k, j >= 1, the top-k / top-j sets of 2 "
        "and 3) within the cap, in ascending size (ties: smaller k, then smaller j), evaluated in that order until "
        "the first that meets the target, at most 12 of them",
    ],
    "choose": "among the layer and kind sets of steps 1-3 that meet the target, the smallest size (ties: the layer "
              "set, then the lexicographically smallest sorted list); if none, the first union of step 4 that meets "
              "it; if none, no set (report the numbers; the ship form is chosen separately)",
    "then": "the chosen set is exported (export_decoder.py int8mix) and gated on all 434 rows with the gate's own bar, "
            "then on the 130 held-out rows (never used to choose)",
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
    parts = name.split(".")
    for code, (block, leaf) in KINDS.items():
        if len(parts) == 5 and parts[3] == block and parts[4] == leaf:
            return code
    return "other"


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
def set_name(layers=(), kinds=()) -> str:
    parts = []
    if layers:
        parts.append("_".join(f"L{i:02d}" for i in sorted(layers)))
    if kinds:
        parts.append("K_" + "_".join(k for k in KINDS if k in set(kinds)))
    return "set_" + "__".join(parts)


def parse_set(name: str) -> tuple[list[int], list[str]]:
    layers, kinds = [], []
    for part in name[len("set_"):].split("__"):
        if part.startswith("K_"):
            kinds = part[2:].split("_")
        else:
            layers = [int(t[1:]) for t in part.split("_")]
    bad = [k for k in kinds if k not in KINDS] + [i for i in layers if not 0 <= i < N_LAYERS]
    if bad:
        raise SystemExit(f"{name}: unknown layers / kinds {bad}")
    return sorted(layers), [k for k in KINDS if k in set(kinds)]


def config(names: list[str], name: str) -> dict:
    """name -> {"fp16": [module names kept out of int8], "layers": [..], "kinds": [..], "why": ..}."""
    every = sorted(names)
    if name == "exact":
        layers, kinds, why = list(range(N_LAYERS)), list(KINDS), "no int8: the instrument's floor vs the oracle"
        keep = set(every)
    elif name == "all_int8":
        layers, kinds, why, keep = [], [], "the export's int8lin body: every linear int8", set()
    elif m := re.fullmatch(r"layer_(\d\d)_fp16", name):
        i = int(m.group(1))
        layers, kinds, why = [i], [], f"layer {i}'s linears fp32, every other linear int8"
        keep = {n for n in every if layer_of(n) == i}
    elif name in ("gdn_fp16", "attn_fp16", "mlp_fp16"):
        g = name[:-len("_fp16")]
        layers, kinds, why = [], list(GROUPS[g]), f"every {g} linear fp32, the rest int8"
        keep = {n for n in every if kind_of(n) in GROUPS[g]}
    elif m := re.fullmatch(r"proj_([a-z]+)_fp16", name):
        k = m.group(1)
        if k not in KINDS:
            raise SystemExit(f"unknown kind {k}")
        layers, kinds, why = [], [k], f"{KINDS[k][1]} fp32 in every layer, the rest int8"
        keep = {n for n in every if kind_of(n) == k}
    elif name.startswith("set_"):
        layers, kinds = parse_set(name)
        keep = {n for n in every if layer_of(n) in set(layers) or kind_of(n) in set(kinds)}
        why = f"layers {layers} and kinds {kinds} fp32, the rest int8"
    else:
        raise SystemExit(f"unknown config {name}")
    return {"fp16": sorted(keep), "layers": layers, "kinds": kinds, "why": why}


def fixed_configs() -> list[str]:
    return (["all_int8", "exact"] + [f"layer_{i:02d}_fp16" for i in range(N_LAYERS)] + ["gdn_fp16", "attn_fp16", "mlp_fp16"]
            + [f"proj_{k}_fp16" for k in KINDS])


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def dump(args) -> None:
    import torch
    import torch.nn.utils.parametrize as P
    from export_decoder import HF_ID, MERGED_SAFETENSORS_SHA256, linear_quant_config
    from qwen3_5_kev_decoder import Qwen3_5KevDecoder, set_unrolled_scan
    from safetensors.torch import save_file

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.compression import quantize_pytorch_model

    if DUMP_META.exists():
        sys.exit(f"{DUMP_META} exists: the dump is written once")
    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    model = Qwen3_5KevDecoder.from_hf(HF_ID, target_dtype=torch.float16, max_context_length=4096)
    n_unrolled = set_unrolled_scan(model)          # the exporter's flags, before quantizing
    spec = model.build_export_spec(torch.float16, 4096, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=CHUNK)
    cfg = linear_quant_config("int8")
    qspec = json.loads(json.dumps(cfg["global_config"]["op_state_spec"]["weight"]))
    linears = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    before = {n: m.weight.detach().clone() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg)
    t_q = time.monotonic() - t0
    DUMP.mkdir(parents=True, exist_ok=True)
    per_layer: dict[int, dict] = {}
    info, other = {}, []
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod):
            continue
        if not (isinstance(mod, torch.nn.Linear) and P.is_parametrized(mod, "weight")):
            other.append(f"{name} ({type(mod).__name__})")
            continue
        deq = [p for p in mod.parametrizations["weight"] if hasattr(p, "quantized_data")]
        assert len(deq) == 1, (name, [type(p).__name__ for p in mod.parametrizations["weight"]])
        d = deq[0]
        w = mod.weight.detach().clone()                    # the finalized module's own dequantization
        again = torch.ops.coreai.constexpr_blockwise_shift_scale(
            d.quantized_data, d.scale, zero_point=d.zero_point, minval=d.minval,
            input_dtype=d.input_dtype, output_dtype=d.output_dtype)
        assert torch.equal(w, again), name
        q = d.quantized_data
        ref = before[name].float()
        info[name] = {"shape": list(w.shape), "dtype": str(w.dtype), "codes_dtype": str(q.dtype),
                      "codes_min": int(q.min()), "codes_max": int(q.max()),
                      "scale_shape": list(d.scale.shape), "scale_dtype": str(d.scale.dtype),
                      "zero_point": None if d.zero_point is None else int(d.zero_point.abs().max()),
                      "rel_err": float((w.float() - ref).norm() / ref.norm()),
                      "max_abs_err": float((w.float() - ref).abs().max())}
        per_layer.setdefault(layer_of(name), {})[name] = w.contiguous()
    expected = [n for n in linears if n.startswith("model.layers.")]
    if other or sorted(info) != expected or len(expected) != 186 or set(linears) != set(expected):
        sys.exit(f"quantized set differs: other {other}, {len(info)} quantized, {len(expected)} decoder linears, "
                 f"linears outside the decoder layers {sorted(set(linears) - set(expected))}")
    files = {}
    for li, ws in sorted(per_layer.items()):
        p = DUMP / f"layer_{li:02d}.safetensors"
        save_file(ws, str(p))
        files[str(li)] = {"file": str(p), "sha256": sha256(p), "modules": sorted(ws)}
    rel = [v["rel_err"] for v in info.values()]
    meta = {"weight_spec": qspec, "excluded_types": sorted(cfg["module_type_configs"]), "hf_id": HF_ID,
            "export_spec_query_len": CHUNK, "unrolled_gdn_layers": n_unrolled,
            "quantized_modules": len(info), "modules_by_kind": {k: sum(kind_of(n) == k for n in info) for k in KINDS},
            "params": int(sum(int(np.prod(v["shape"])) for v in info.values())),
            "codes_range": [min(v["codes_min"] for v in info.values()), max(v["codes_max"] for v in info.values())],
            "rel_err": {"median": float(np.median(rel)), "max": float(np.max(rel)), "min": float(np.min(rel))},
            "quantize_seconds": t_q, "files": files, "modules": info,
            "exact_weights": {"what": "the fp32 module's own weights (Qwen3_5KevDecoder.from_hf fp32 = the merged "
                                      "checkpoint, read unchanged), kept in memory by every run process",
                              "checkpoint_model_safetensors_sha256": MERGED_SAFETENSORS_SHA256},
            "script_sha256": sha256(Path(__file__).resolve()), "generated_at": now()}
    DUMP_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dumped {len(info)} modules ({meta['params']:,} params) in {t_q:.0f}s to {DUMP}; codes "
          f"{meta['codes_range']}; rel err median {meta['rel_err']['median']:.4e} max {meta['rel_err']['max']:.4e}",
          flush=True)


# --------------------------------------------------------------------------- #
# rule
# --------------------------------------------------------------------------- #
def rule(args) -> None:
    from parity_decoder_torch import load_oracle

    if RULE_PATH.exists():
        sys.exit(f"{RULE_PATH} exists (the rule is written once, before any result)")
    if PARTS.exists() and any(PARTS.iterdir()):
        sys.exit(f"{PARTS} has results already: the rule comes first")
    gate_path = Path(args.gate).resolve()
    gate = json.loads(gate_path.read_text())
    _, recs = load_oracle()
    order = {(rid, k): n for n, (rid, k) in enumerate((rid, k) for rid, r in recs.items()
                                                       for k in range(len(r["questions"])))}
    runs = sorted(gate["runs"], key=lambda r: (-r["max_abs_dp"], order[(r["id"], r["q"])]))
    top = [(r["id"], r["q"]) for r in runs[:N_TOP]]
    rest = [r for r in runs[N_TOP:]]
    by_src: dict = {}
    for r in sorted(rest, key=lambda r: order[(r["id"], r["q"])]):
        by_src.setdefault(r["source"], []).append((r["id"], r["q"]))
    # proportional allocation (largest remainder), at least one row per source
    total = sum(len(v) for v in by_src.values())
    quota = {s: max(1, int(N_RANDOM * len(v) / total)) for s, v in by_src.items()}
    rema = sorted(by_src, key=lambda s: (-(N_RANDOM * len(by_src[s]) / total - int(N_RANDOM * len(by_src[s]) / total)), s))
    i = 0
    while sum(quota.values()) < N_RANDOM:
        quota[rema[i % len(rema)]] += 1
        i += 1
    while sum(quota.values()) > N_RANDOM:          # only if the minimum of one pushed it over
        s = max(quota, key=lambda s: (quota[s], s))
        quota[s] -= 1
    rng = np.random.default_rng(SEED)
    rand = []
    for s in sorted(by_src):
        pool = by_src[s]
        pick = sorted(rng.choice(len(pool), size=quota[s], replace=False).tolist())
        rand += [pool[j] for j in pick]
    rows = top + rand
    g_by = {(r["id"], r["q"]): r for r in gate["runs"]}
    lens = [recs[rid]["questions"][k]["row_len"] for rid, k in rows]
    doc = {"schema": "kev-int8-bisect-rule/1", "rule": RULE, "target": TARGET, "cap_fraction": CAP_FRACTION,
           "rows": [list(x) for x in rows],
           "rows_detail": [{"id": rid, "q": k, "set": "top" if n < N_TOP else "random", "source": recs[rid]["source"],
                            "row_len": recs[rid]["questions"][k]["row_len"],
                            "near_tie": bool(recs[rid]["questions"][k]["near_tie"]),
                            "gate_int8lin_max_abs_dp": g_by[(rid, k)]["max_abs_dp"]} for n, (rid, k) in enumerate(rows)],
           "why_rows": f"the top {N_TOP} rows of the int8lin gate by max|dp| (ties: oracle order), then {N_RANDOM} rows "
                       f"drawn from the other {len(rest)} rows by source (proportional, largest remainder, at least one "
                       f"per source; numpy default_rng({SEED}).choice without replacement inside each source, sources "
                       f"in sorted order)",
           "random_quota": quota, "seed": SEED,
           "gate_transcript": {"path": str(gate_path), "sha256": sha256(gate_path), "result": gate["result"],
                               "max_abs_dp": gate["summary"]["max_abs_dp"],
                               "bundle": gate["bundle"]["name"],
                               "aimodelc_tree_sha256": gate["bundle"]["aimodelc"]["tree_sha256"]},
           "configs_fixed": fixed_configs(), "chunk": CHUNK, "tokens": int(sum(lens)),
           "estimate": f"{sum(lens):,} tokens per config; at round 2's 7.5 ms per token (1 thread, bmm conv) about "
                       f"{sum(lens) * 7.5e-3:.0f} s per config",
           "script_sha256": sha256(Path(__file__).resolve()), "written_at": now()}
    RULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RULE_PATH.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"rule -> {RULE_PATH} ({sha256(RULE_PATH)}): {len(rows)} rows, {sum(lens):,} tokens, quota {quota}; "
          f"{doc['estimate']}", flush=True)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
class Weights:
    """Swaps the 186 linears' weights between their fp32 checkpoint values and the int8 dump (fp16 -> fp32)."""

    def __init__(self, model):
        import torch
        from safetensors import safe_open

        self.torch = torch
        meta = json.loads(DUMP_META.read_text())
        self.names = sorted(meta["modules"])
        mods = dict(model.named_modules())
        self.mods = {n: mods[n] for n in self.names}
        assert all(isinstance(m, torch.nn.Linear) for m in self.mods.values())
        self.exact = {n: m.weight.detach().clone() for n, m in self.mods.items()}
        self.int8 = {}
        for li in sorted({layer_of(n) for n in self.names}):
            with safe_open(str(DUMP / f"layer_{li:02d}.safetensors"), framework="pt", device="cpu") as f:
                for n in f.keys():  # noqa: SIM118
                    self.int8[n] = f.get_tensor(n)
        assert sorted(self.int8) == self.names
        for n in self.names:
            assert self.int8[n].shape == self.exact[n].shape and self.exact[n].dtype == torch.float32, n
        self.params = {n: int(self.mods[n].weight.numel()) for n in self.names}
        self.state = {n: "exact" for n in self.names}

    def apply(self, keep: list[str]) -> int:
        """Every module in `keep` gets its fp32 weight, every other one the int8 dump's."""
        want = set(keep)
        changed = 0
        with self.torch.no_grad():
            for n, m in self.mods.items():
                s = "exact" if n in want else "int8"
                if self.state[n] == s:
                    continue
                m.weight.copy_(self.exact[n] if s == "exact" else self.int8[n].to(self.torch.float32))
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
    from parity_decoder_torch import KevHead, Runner, load_oracle, score_question

    rdoc = json.loads(RULE_PATH.read_text())
    rows = [tuple(x) for x in rdoc["rows"]]
    _, recs = load_oracle()
    p1 = p1_probs()
    part_path = Path(args.part)
    part = json.loads(part_path.read_text()) if part_path.exists() else {"configs": {}}
    t0 = time.monotonic()
    head = KevHead()
    runner = Runner(args.threads, "bmm")
    W = Weights(runner.model)
    load_s = time.monotonic() - t0
    body = sum(W.params.values())
    todo = [c for c in args.configs.split(",") if c]
    cfgs = {c: config(W.names, c) for c in todo}
    part.update({"pid": os.getpid(), "threads": args.threads, "conv": "bmm", "chunk": CHUNK,
                 "rows": [list(k) for k in rows], "rule_sha256": sha256(RULE_PATH),
                 "script_sha256": sha256(Path(__file__).resolve()), "load_seconds": load_s,
                 "dump_meta_sha256": sha256(DUMP_META), "body_linear_params": body,
                 "started": part.get("started", now())})

    def save() -> None:
        tmp = part_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, part_path)

    first = [True]

    def one_row(rid: str, k: int) -> dict:
        q = recs[rid]["questions"][k]
        t1 = time.monotonic()
        res = runner.run(q["row_ids"], CHUNK, conv_check=first[0])
        first[0] = False
        sc = score_question(head, res["hidden"], q)
        out = {k_: sc[k_] for k_ in ("max_abs_dp", "mean_abs_dp", "argmax", "argmax_oracle", "argmax_equal",
                                       "near_tie", "probs")}
        out.update({"tokens": res["T"], "seconds": time.monotonic() - t1})
        ref = p1.get((rid, k))
        out["bit_equal_p1"] = None if ref is None else bool(sc["probs"] == ref)
        return out

    print(f"[{os.getpid()}] loaded in {load_s:.0f}s; {len(rows)} rows; configs {todo}", flush=True)
    for name in todo:
        if name in part["configs"] and not args.redo:
            print(f"[{os.getpid()}] {name}: done already", flush=True)
            continue
        c = cfgs[name]
        t1 = time.monotonic()
        changed = W.apply(c["fp16"])
        keep_params = int(sum(W.params[n] for n in c["fp16"]))
        rec = {"why": c["why"], "layers": c["layers"], "kinds": c["kinds"], "fp16_modules": len(c["fp16"]),
               "int8_modules": len(W.names) - len(c["fp16"]), "fp16_params": keep_params,
               "int8_params": body - keep_params, "fp16_fraction": keep_params / body,
               "modules_swapped": changed, "swap_seconds": time.monotonic() - t1, "rows": {}}
        for rid, k in rows:
            rec["rows"][f"{rid}:q{k}"] = one_row(rid, k)
        rr = rec["rows"].values()
        rec["worst_max_abs_dp"] = max(v["max_abs_dp"] for v in rr)
        rec["mean_of_row_mean_abs_dp"] = float(np.mean([v["mean_abs_dp"] for v in rr]))
        rec["argmax_equal_non_near_tie"] = all(v["argmax_equal"] for v in rr if not v["near_tie"])
        rec["seconds"] = time.monotonic() - t1
        rec["finished"] = now()
        part["configs"][name] = rec
        part["conv"] = runner.conv.report()
        save()
        worst = max(rec["rows"].items(), key=lambda kv: kv[1]["max_abs_dp"])
        print(f"[{os.getpid()}] {name}: worst {rec['worst_max_abs_dp']:.4f} ({worst[0]}) mean "
              f"{rec['mean_of_row_mean_abs_dp']:.5f} fp16 params {keep_params:,} ({rec['fp16_fraction']:.1%}) "
              f"{rec['seconds']:.0f}s", flush=True)
    # swap-back proof: every weight back to fp32, the first row again, against P1's fp32 run
    W.apply(list(W.names))
    rid, k = rows[0]
    again = one_row(rid, k)
    part["swap_back_check"] = {"row": f"{rid}:q{k}", "bit_equal_p1": again["bit_equal_p1"],
                               "max_abs_dp_vs_oracle": again["max_abs_dp"], "at": now()}
    part["conv"] = runner.conv.report()
    part["finished"] = now()
    save()
    print(f"[{os.getpid()}] swap-back {rid}:q{k} bit-equal P1 {again['bit_equal_p1']}; conv {json.dumps(part['conv'])}",
          flush=True)


# --------------------------------------------------------------------------- #
# plan / merge
# --------------------------------------------------------------------------- #
def load_parts() -> tuple[dict, list[dict]]:
    table, procs = {}, []
    for p in sorted(PARTS.glob("part_*.json")):
        d = json.loads(p.read_text())
        procs.append({"part": str(p), "pid": d.get("pid"), "threads": d.get("threads"), "load_seconds": d.get("load_seconds"),
                      "rule_sha256": d.get("rule_sha256"), "script_sha256": d.get("script_sha256"),
                      "started": d.get("started"), "finished": d.get("finished"), "conv": d.get("conv"),
                      "swap_back_check": d.get("swap_back_check"), "configs": list(d["configs"])})
        for name, rec in d["configs"].items():
            if name in table:
                a = {k: v["probs"] for k, v in table[name]["rows"].items()}
                b = {k: v["probs"] for k, v in rec["rows"].items()}
                rec["repeat_bit_equal"] = a == b
            table[name] = rec
    return table, procs


def meets(rec: dict) -> bool:
    return rec["worst_max_abs_dp"] < TARGET


def rankings(table: dict) -> tuple[list[int], list[str]]:
    layers = sorted(range(N_LAYERS), key=lambda i: (table[f"layer_{i:02d}_fp16"]["worst_max_abs_dp"], i))
    kinds = sorted(KINDS, key=lambda k: (table[f"proj_{k}_fp16"]["worst_max_abs_dp"], list(KINDS).index(k)))
    return layers, kinds


def set_size(names: list[str], params: dict, layers=(), kinds=()) -> int:
    return int(sum(params[n] for n in names if layer_of(n) in set(layers) or kind_of(n) in set(kinds)))


def module_params() -> dict:
    meta = json.loads(DUMP_META.read_text())
    return {n: int(np.prod(v["shape"])) for n, v in meta["modules"].items()}


def plan_steps(table: dict) -> dict:
    """What the rule has evaluated and what it asks for next (names in the rule's order)."""
    params = module_params()
    names = sorted(params)
    body = sum(params.values())
    cap = CAP_FRACTION * body
    missing_fixed = [c for c in fixed_configs() if c not in table]
    if missing_fixed:
        return {"stage": 1, "next": missing_fixed}
    lay, kin = rankings(table)
    out = {"layer_ranking": lay, "kind_ranking": kin, "body_linear_params": body, "cap_params": cap, "next": []}

    def walk(order: list, mk) -> dict:
        """top-k sets for k = 1.. within the cap; the first meeting k, then its subsets of the top k+1."""
        seq, first = [], None
        for k in range(1, len(order) + 1):
            name, size = mk(order[:k])
            if size > cap:
                break
            seq.append({"k": k, "set": name, "size": size})
            rec = table.get(name)
            if rec is None:
                return {"seq": seq, "next": [name], "first_meeting_k": None}
            if meets(rec):
                first = k
                break
        if first is None:
            return {"seq": seq, "next": [], "first_meeting_k": None, "exhausted": True}
        subs = []
        if first < len(order):
            subs = [mk(sorted(s, key=order.index))[0] for s in itertools.combinations(order[:first + 1], first)]
        return {"seq": seq, "first_meeting_k": first, "subsets": subs, "next": [s for s in subs if s not in table]}

    def mk_layers(ls):
        nm = f"layer_{ls[0]:02d}_fp16" if len(ls) == 1 else set_name(layers=ls)
        return nm, set_size(names, params, layers=ls)

    def mk_kinds(ks):
        nm = f"proj_{ks[0]}_fp16" if len(ks) == 1 else set_name(kinds=ks)
        return nm, set_size(names, params, kinds=ks)

    wl, wk = walk(lay, mk_layers), walk(kin, mk_kinds)
    out["layers"], out["kinds"] = wl, wk
    out["next"] = wl["next"] + wk["next"]
    if out["next"]:
        return {**out, "stage": "2-3"}
    # candidates of steps 1-3 that meet the target: only the sets the rule's search generates (an informative config
    # evaluated outside the search is never chosen)
    allowed = ({f"layer_{i:02d}_fp16" for i in range(N_LAYERS)} | {f"proj_{k}_fp16" for k in KINDS}
               | {f"{g}_fp16" for g in GROUPS} | {s["set"] for s in wl["seq"] + wk["seq"]}
               | set(wl.get("subsets", [])) | set(wk.get("subsets", [])))
    out["rule_search_sets"] = sorted(allowed)
    cands = []
    for name, rec in table.items():
        if name not in allowed or not meets(rec) or rec["fp16_params"] > cap:
            continue
        cands.append({"config": name, "size": rec["fp16_params"], "is_layer_set": bool(rec["layers"]),
                      "sorted": sorted(rec["layers"]) if rec["layers"] else sorted(rec["kinds"])})
    out["meeting_steps_1_3"] = sorted(cands, key=lambda c: (c["size"], not c["is_layer_set"], c["sorted"]))
    if cands:
        return {**out, "stage": "chosen", "chosen": out["meeting_steps_1_3"][0]["config"]}
    # step 4: unions L_k + K_j within the cap, ascending size
    ls_seq = [s for s in wl["seq"]]
    ks_seq = [s for s in wk["seq"]]
    unions = []
    for a in ls_seq:
        for b in ks_seq:
            ls_, ks_ = lay[:a["k"]], kin[:b["k"]]
            size = set_size(names, params, layers=ls_, kinds=ks_)
            if size <= cap:
                unions.append({"k": a["k"], "j": b["k"], "set": set_name(layers=ls_, kinds=ks_), "size": size})
    unions.sort(key=lambda u: (u["size"], u["k"], u["j"]))
    unions = unions[:12]
    out["unions"] = unions
    for u in unions:
        rec = table.get(u["set"])
        if rec is None:
            return {**out, "stage": 4, "next": [u["set"]]}
        if meets(rec):
            return {**out, "stage": "chosen", "chosen": u["set"]}
    return {**out, "stage": "none", "chosen": None}


def plan(args) -> None:
    table, _ = load_parts()
    p = plan_steps(table)
    print(json.dumps({k: v for k, v in p.items()}, indent=1))


def merge(args) -> None:
    rdoc = json.loads(RULE_PATH.read_text())
    table, procs = load_parts()
    p = plan_steps(table)
    gate = json.loads(from_record_path(rdoc["gate_transcript"]["path"]).read_text())
    g_by = {f"{r['id']}:q{r['q']}": r["max_abs_dp"] for r in gate["runs"]}
    rows = [f"{a}:q{b}" for a, b in rdoc["rows"]]
    top, rand = rows[:N_TOP], rows[N_TOP:]
    base = table.get("all_int8")
    repro = None
    if base:
        gw = max(g_by[r] for r in rows)
        tw = base["worst_max_abs_dp"]
        x = np.array([g_by[r] for r in rows])
        y = np.array([base["rows"][r]["max_abs_dp"] for r in rows])
        repro = {"gate_worst": gw, "torch_all_int8_worst": tw, "ratio": tw / gw, "reproduced": tw >= 0.5 * gw,
                 "pearson_per_row": float(np.corrcoef(x, y)[0, 1]),
                 "rows_above_bar_gate": int((x > BAR_MAX_ABS_DP).sum()), "rows_above_bar_torch": int((y > BAR_MAX_ABS_DP).sum()),
                 "per_row": [{"row": r, "gate": g_by[r], "torch": base["rows"][r]["max_abs_dp"]} for r in rows]}
    floor = None
    if "exact" in table:
        ex = table["exact"]["rows"]
        floor = {"worst_max_abs_dp": table["exact"]["worst_max_abs_dp"],
                 "bit_equal_p1_rows": sum(bool(v["bit_equal_p1"]) for v in ex.values()), "rows": len(ex)}
    tab = []
    for name, rec in table.items():
        rr = rec["rows"]
        tab.append({"config": name, "layers": rec["layers"], "kinds": rec["kinds"], "fp16_params": rec["fp16_params"],
                    "fp16_fraction": rec["fp16_fraction"], "worst_max_abs_dp": rec["worst_max_abs_dp"],
                    "worst_top30": max(rr[r]["max_abs_dp"] for r in top),
                    "worst_random30": max(rr[r]["max_abs_dp"] for r in rand),
                    "mean_of_row_mean_abs_dp": rec["mean_of_row_mean_abs_dp"],
                    "mean_of_row_mean_abs_dp_random30": float(np.mean([rr[r]["mean_abs_dp"] for r in rand])),
                    "argmax_equal_non_near_tie": rec["argmax_equal_non_near_tie"],
                    "rows_not_worse_than_all_int8": (sum(rr[r]["max_abs_dp"] <= base["rows"][r]["max_abs_dp"] for r in rows)
                                                     if base else None),
                    "meets_target": meets(rec), "seconds": rec["seconds"],
                    "in_rule_search": (name in ("exact", "all_int8") or name in p.get("rule_search_sets", [])
                                       or name in {u["set"] for u in p.get("unions", [])})})
    tab.sort(key=lambda t: (not t["config"].startswith(("all_int8", "exact")), t["config"]))
    out = {"schema": "kev-int8-bisect/1", "rule": rdoc["rule"], "rule_file": str(RULE_PATH), "rule_sha256": sha256(RULE_PATH),
           "rule_written_at": rdoc["written_at"], "rows": rdoc["rows"], "why_rows": rdoc["why_rows"],
           "instrument": ("fp32 torch on the CPU (parity_decoder_torch.Runner, 1 thread, bmm depthwise conv), S = 16 chunks "
                          "from fresh zero states, the oracle's row ids, the author's fp32 pointer head; per config the "
                          "listed linears carry their fp32 checkpoint weight, every other one the exporter's int8 weight "
                          "(quantize_pytorch_model on the fp16 model, dequantized by the finalized module)"),
           "script": {"path": "conversion/kev/int8_bisect_torch.py", "sha256": sha256(Path(__file__).resolve())},
           "dump": {"meta": str(DUMP_META), "sha256": sha256(DUMP_META)},
           "reproduction_check": repro, "instrument_floor": floor, "processes": procs, "plan": p,
           "chosen": p.get("chosen"), "table": tab, "configs": table, "generated_at": now()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} configs -> {args.out}; reproduction {None if repro is None else (round(repro['ratio'], 3), repro['reproduced'])}; "
          f"floor {floor}; stage {p.get('stage')} chosen {p.get('chosen')}")


# --------------------------------------------------------------------------- #
# Kev-4B (round 6): `--model kev-4b`. The same instrument on the 4B checkpoint (32 layers, 248 linears), with
# quantization variants of the linears (per-block 16, an asymmetric zero point), the embedding table in int8, and
# four-layer groups; its own rule (40 rows, configs ranked by the rows' mean |dp|, at most 25 % of the body's linear
# params kept fp16, target worst <= 0.006 and mean <= 0.0012). Weights move by module from the merged checkpoint
# (exact) or the dumps (int8) through safetensors' mmap: a process holds one fp32 model, nothing else.
# --------------------------------------------------------------------------- #
R6_LAYERS, R6_GROUP, R6_LINEARS = 32, 4, 248
R6_WORK = LANE / "bisect_4b"
R6_DUMP = R6_WORK / "dump"
R6_RULE = R6_WORK / "rule.json"
R6_PARTS = R6_WORK / "parts"
R6_PARITY = LANE / "parity_4b"
R6_GATE = LANE / "results" / "readout_int8lin_pf16_4b.json"
R6_N_TOP, R6_N_RANDOM = 20, 20
R6_TARGET_WORST, R6_TARGET_MEAN, R6_CAP = 0.006, 0.0012, 0.25
R6_EMBED_SLACK = 0.001
R6_FP16_MAIN_MLIRB = 8_414_007_636          # round 4 fp16 pf16 4B main.mlirb (the bytes estimate's anchor)
# variant -> (block, qscheme, stored bytes per weight: int8 code + fp16 scale per block (+ int8 zero point))
R6_VARIANTS = {"b32": (32, "symmetric_with_clipping", 1 + 2 / 32), "b16": (16, "symmetric_with_clipping", 1 + 2 / 16),
               "aff32": (32, "asymmetric", 1 + 2 / 32 + 1 / 32)}
R6_EMBED_BPP = 1 + 2 / 32                   # the table int8 per-block-32 symmetric_with_clipping (axis 1)
R6_ALIASES = {"exact": "set_fp32", "all_int8_b32": "set_b32", "all_int8_b16": "set_b16",
              "all_int8_affine_b32": "set_aff32", "embed_int8_only": "set_fp32__E"}
R6_LIMITS = {"ii": 4, "iv": 4, "v": 8, "beyond_cap": 2}


def r6_group_name(g: int) -> str:
    return f"layers_{R6_GROUP * g}-{R6_GROUP * g + R6_GROUP - 1}_fp16"


def r6_fixed() -> list[str]:
    return (["exact", "all_int8_b32", "all_int8_b16", "all_int8_affine_b32", "embed_int8_only"]
            + [f"proj_{k}_fp16" for k in KINDS] + [r6_group_name(g) for g in range(R6_LAYERS // R6_GROUP)])


def r6_set_name(variant: str, groups=(), kinds=(), embed: bool = False) -> str:
    parts = [f"set_{variant}"]
    if groups:
        parts.append("G_" + "_".join(str(g) for g in sorted(groups)))
    if kinds:
        parts.append("K_" + "_".join(k for k in KINDS if k in set(kinds)))
    if embed:
        parts.append("E")
    return "__".join(parts)


def r6_parse(name: str) -> dict:
    """config name -> {variant, groups, kinds, embed}; the fixed names are spelled through the set grammar."""
    if name in R6_ALIASES:
        name = R6_ALIASES[name]
    elif m := re.fullmatch(r"proj_([a-z]+)_fp16", name):
        name = f"set_b32__K_{m.group(1)}"
    elif m := re.fullmatch(r"layers_(\d+)-(\d+)_fp16", name):
        a, b = int(m.group(1)), int(m.group(2))
        if a % R6_GROUP or b != a + R6_GROUP - 1:
            raise SystemExit(f"{name}: not a {R6_GROUP}-layer group")
        name = f"set_b32__G_{a // R6_GROUP}"
    if not name.startswith("set_"):
        raise SystemExit(f"unknown config {name}")
    parts = name[len("set_"):].split("__")
    out = {"variant": parts[0], "groups": [], "kinds": [], "embed": False}
    for p in parts[1:]:
        if p.startswith("G_"):
            out["groups"] = sorted(int(x) for x in p[2:].split("_"))
        elif p.startswith("K_"):
            out["kinds"] = [k for k in KINDS if k in set(p[2:].split("_"))]
            if len(out["kinds"]) != len(p[2:].split("_")):
                raise SystemExit(f"{name}: unknown kinds")
        elif p == "E":
            out["embed"] = True
        else:
            raise SystemExit(f"{name}: unknown part {p}")
    if out["variant"] not in R6_VARIANTS and out["variant"] != "fp32":
        raise SystemExit(f"{name}: unknown variant {out['variant']}")
    if any(not 0 <= g < R6_LAYERS // R6_GROUP for g in out["groups"]):
        raise SystemExit(f"{name}: unknown group")
    return out


def r6_canonical(name: str) -> str:
    c = r6_parse(name)
    return r6_set_name(c["variant"], c["groups"], c["kinds"], c["embed"])


def r6_states(c: dict, linears: list[str]) -> dict:
    """module -> 'exact' | variant, and the embedding -> 'exact' | 'embed32'."""
    out = {}
    for n in linears:
        keep = c["variant"] == "fp32" or layer_of(n) // R6_GROUP in set(c["groups"]) or kind_of(n) in set(c["kinds"])
        out[n] = "exact" if keep else c["variant"]
    out["model.embed_tokens"] = "embed32" if c["embed"] else "exact"
    return out


def r6_module_params() -> tuple[dict, int]:
    """Body linear params from the merged checkpoint's safetensors headers (no tensor read), and the table's."""
    import export_decoder as ed
    from safetensors import safe_open

    ed.select_model("kev-4b")
    snap = Path(hf_snapshot(ed.HF_ID))
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    params, table = {}, None
    for key, fname in sorted(wm.items()):
        with safe_open(str(snap / fname), framework="pt") as f:
            shape = f.get_slice(key).get_shape()
        name = "model." + key[: -len(".weight")] if key.endswith(".weight") else None
        if name and name.startswith("model.layers.") and kind_of(name) != "other":
            params[name] = int(np.prod(shape))
        if key == "embed_tokens.weight":
            table = int(np.prod(shape))
    if len(params) != R6_LINEARS:
        raise SystemExit(f"{len(params)} body linears in the checkpoint headers, want {R6_LINEARS}")
    return params, table


def r6_size(c: dict, params: dict, table: int) -> dict:
    """fp16 params (the cap's measure) and the estimated main.mlirb bytes of a config's bundle."""
    body = sum(params.values())
    st = r6_states(c, sorted(params))
    fp16 = sum(p for n, p in params.items() if st[n] == "exact")
    lin_bytes = sum(2 * p if st[n] == "exact" else R6_VARIANTS[st[n]][2] * p for n, p in params.items())
    est = R6_FP16_MAIN_MLIRB - 2 * body + lin_bytes - (2 - R6_EMBED_BPP) * table * c["embed"]
    return {"fp16_params": int(fp16), "fp16_fraction": fp16 / body, "est_main_mlirb_bytes": int(round(est))}


R6_RULE_TEXT = {
    "rows": f"the top {R6_N_TOP} rows of the 4B int8lin gate (results/readout_int8lin_pf16_4b.json) by max|dp| (ties: "
            f"oracle order), then {R6_N_RANDOM} rows drawn from the other rows by source (proportional, largest "
            f"remainder, at least one per source; numpy default_rng({SEED}).choice without replacement inside each "
            f"source, sources in sorted order) - round 3's procedure with 20 + 20",
    "instrument": "fp32 torch on the CPU (parity_decoder_torch.Runner --model kev-4b, 1 thread, bmm depthwise conv), "
                  "S = 16 chunks from fresh zero states, the 4B oracle's row ids, the 4B fp32 pointer head; per config "
                  "every body linear and the embedding table carry either the merged checkpoint's fp32 weight or the "
                  "weight the exporter's quantize_pytorch_model call computes for that variant (the finalized module's "
                  "dequantization, fp16, cast to fp32)",
    "reproduction_check": "before any selection: all_int8_b32's worst max|dp| over the 40 rows >= 0.5 x the int8lin gate "
                          "transcript's worst over the same rows (0.0601); otherwise stop and report",
    "instrument_floor": "exact reproduces the round-4 P1 run (parity_4b/p1_shard*of2.jsonl) of every row bit for bit "
                        "(probabilities), and every process's swap-back re-run likewise",
    "target": f"worst max|dp| over the 40 rows <= {R6_TARGET_WORST} AND the mean over the rows of the row's mean |dp| "
              f"<= {R6_TARGET_MEAN} (every option of every question, near-ties included), this instrument",
    "ranking": "configs ranked by the mean (ascending; ties: the worst, then the name); the worst is recorded",
    "size": "a config's size = the params of the body linears it keeps out of int8 (fp16); the cap = 25 % of the 248 "
            "body linears' params; the embedding table's precision does not count against the cap; the estimated "
            "main.mlirb = the round-4 fp16 main.mlirb - 2 B x body params + per-linear bytes (fp16 2 B, b32 1 + 2/32, "
            "b16 1 + 2/16, aff32 1 + 2/32 + 1/32) - (2 - (1 + 2/32)) B x table params when the table is int8",
    "search": [
        "(fixed) exact, all_int8_b32, all_int8_b16, all_int8_affine_b32, embed_int8_only, proj_<kind>_fp16 x 12 (that "
        "kind fp16, every other linear int8 b32), layers_<a>-<b>_fp16 x 8 (that four-layer group fp16, the rest b32)",
        "(i) the variants alone: all_int8_b32 / b16 / affine_b32 (0 % fp16)",
        "(ii) kind sets K_j (int8 part b32): the 12 kinds in ranking order, each added while the set stays within the "
        "cap (a kind that would cross it is skipped and the walk goes on); K_j = the first j added; K_2, K_3, ... "
        f"evaluated in order, at most {R6_LIMITS['ii']}",
        "(iii) group sets G_k (int8 part b32): the 8 groups in ranking order; G_k = the top k, k = 2, 3, ... while "
        "within the cap",
        "(iv) only if the best variant by mean is not b32: that variant as the int8 part of every K_j / G_k (j, k >= 1) "
        f"evaluated in (ii)-(iii), ascending size, at most {R6_LIMITS['iv']}",
        "(v) unions K_j + G_k (j, k >= 1, int8 part = the best variant) within the cap, ascending size (ties: smaller k, "
        f"then smaller j), at most {R6_LIMITS['v']}",
        "a stage is entered only while no evaluated config within the cap meets the target",
        f"(reference, never chosen) at most {R6_LIMITS['beyond_cap']} sets beyond the cap: G_3 and G_4 by ranking "
        "(int8 part b32)",
    ],
    "choose": "among the evaluated configs within the cap that meet the target, the smallest size (ties: the smaller "
              "estimated main.mlirb, then the lower mean); if none, the best two within the cap by the ranking "
              "(exact and embed_int8_only excluded: they keep the whole body fp32)",
    "embed": f"then the chosen set(s) with the table int8 (+E), one config each; +E is adopted when its worst exceeds the "
             f"set's own by at most {R6_EMBED_SLACK}",
    "then": "the chosen set(s) (and their +E when adopted) are exported (export_decoder.py int8mix / int8lin with "
            "--fp16-layers / --fp16-kinds / --quant-block / --quant-scheme / --embed-int8, --model kev-4b) and gated on "
            "all 434 rows with the gate's own bar, the 130 held-out rows (never used to choose) and the v2 red arms",
}


def r6_rule(args) -> None:
    import parity_decoder_torch as pdt

    pdt.configure("kev-4b")
    if R6_RULE.exists():
        sys.exit(f"{R6_RULE} exists (the rule is written once, before any result)")
    if R6_PARTS.exists() and any(R6_PARTS.iterdir()):
        sys.exit(f"{R6_PARTS} has results already: the rule comes first")
    gate_path = Path(args.gate).resolve()
    gate = json.loads(gate_path.read_text())
    _, recs = pdt.load_oracle()
    order = {(rid, k): n for n, (rid, k) in enumerate((rid, k) for rid, r in recs.items()
                                                       for k in range(len(r["questions"])))}
    runs = sorted(gate["runs"], key=lambda r: (-r["max_abs_dp"], order[(r["id"], r["q"])]))
    top = [(r["id"], r["q"]) for r in runs[:R6_N_TOP]]
    rest = runs[R6_N_TOP:]
    by_src: dict = {}
    for r in sorted(rest, key=lambda r: order[(r["id"], r["q"])]):
        by_src.setdefault(r["source"], []).append((r["id"], r["q"]))
    total = sum(len(v) for v in by_src.values())
    quota = {s: max(1, int(R6_N_RANDOM * len(v) / total)) for s, v in by_src.items()}
    rema = sorted(by_src, key=lambda s: (-(R6_N_RANDOM * len(by_src[s]) / total
                                           - int(R6_N_RANDOM * len(by_src[s]) / total)), s))
    i = 0
    while sum(quota.values()) < R6_N_RANDOM:
        quota[rema[i % len(rema)]] += 1
        i += 1
    while sum(quota.values()) > R6_N_RANDOM:
        s = max(quota, key=lambda s: (quota[s], s))
        quota[s] -= 1
    rng = np.random.default_rng(SEED)
    rand = []
    for s in sorted(by_src):
        pool = by_src[s]
        pick = sorted(rng.choice(len(pool), size=quota[s], replace=False).tolist())
        rand += [pool[j] for j in pick]
    rows = top + rand
    g_by = {(r["id"], r["q"]): r for r in gate["runs"]}
    lens = [recs[rid]["questions"][k]["row_len"] for rid, k in rows]
    params, table = r6_module_params()
    body = sum(params.values())
    sizes = {name: r6_size(r6_parse(name), params, table) for name in r6_fixed()}
    doc = {"schema": "kev-int8-bisect-rule-4b/1", "model": "kev-4b", "rule": R6_RULE_TEXT,
           "target": {"worst_max_abs_dp": R6_TARGET_WORST, "mean_of_row_mean_abs_dp": R6_TARGET_MEAN},
           "cap_fraction": R6_CAP, "cap_params": R6_CAP * body, "limits": R6_LIMITS, "embed_slack": R6_EMBED_SLACK,
           "rows": [list(x) for x in rows],
           "rows_detail": [{"id": rid, "q": k, "set": "top" if n < R6_N_TOP else "random", "source": recs[rid]["source"],
                            "row_len": recs[rid]["questions"][k]["row_len"],
                            "near_tie": bool(recs[rid]["questions"][k]["near_tie"]),
                            "gate_int8lin_max_abs_dp": g_by[(rid, k)]["max_abs_dp"]} for n, (rid, k) in enumerate(rows)],
           "random_quota": quota, "seed": SEED,
           "gate_transcript": {"path": str(gate_path), "sha256": sha256(gate_path), "result": gate["result"],
                               "max_abs_dp": gate["summary"]["max_abs_dp"], "bundle": gate["bundle"]["name"],
                               "aimodelc_tree_sha256": gate["bundle"]["aimodelc"]["tree_sha256"]},
           "configs_fixed": r6_fixed(), "configs_fixed_canonical": {n: r6_canonical(n) for n in r6_fixed()},
           "variants": {k: {"block": v[0], "qscheme": v[1], "bytes_per_weight": v[2]} for k, v in R6_VARIANTS.items()},
           "embedding_variant": {"embed32": "per-block-32 symmetric_with_clipping along axis 1 (hidden)",
                                 "bytes_per_weight": R6_EMBED_BPP},
           "body_linear_params": body, "table_params": table,
           "params_by_kind": {k: sum(p for n, p in params.items() if kind_of(n) == k) for k in KINDS},
           "params_by_group": {str(g): sum(p for n, p in params.items() if layer_of(n) // R6_GROUP == g)
                               for g in range(R6_LAYERS // R6_GROUP)},
           "sizes_fixed": sizes, "chunk": CHUNK, "tokens": int(sum(lens)),
           "estimate": f"{sum(lens):,} tokens per config; at round 4's 38 ms per token (1 thread, bmm conv, alone) about "
                       f"{sum(lens) * 38e-3:.0f} s per config",
           "script_sha256": sha256(Path(__file__).resolve()), "written_at": now()}
    R6_RULE.parent.mkdir(parents=True, exist_ok=True)
    R6_RULE.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"rule -> {R6_RULE} ({sha256(R6_RULE)}): {len(rows)} rows, {sum(lens):,} tokens, quota {quota}; "
          f"{doc['estimate']}", flush=True)


def r6_dump(args) -> None:
    """One variant: the exporter's quantize_pytorch_model on the fp16 4B model (S = 16 spec, GDN flags), every quantized
    module's dequantized fp16 weight per layer (or the table) + a meta file."""
    import torch
    import torch.nn.utils.parametrize as P
    from qwen3_5_kev_decoder import Qwen3_5KevDecoder, set_unrolled_scan
    from safetensors.torch import save_file

    import export_decoder as ed
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.compression import quantize_pytorch_model

    ed.select_model("kev-4b")
    v = args.variant
    out_dir, meta_path = R6_DUMP / v, R6_DUMP / f"{v}.json"
    if meta_path.exists():
        sys.exit(f"{meta_path} exists: a dump is written once")
    if v == "embed32":
        cfg = ed.quant_config(linear=False, embed=True)
    else:
        block, scheme, _ = R6_VARIANTS[v]
        cfg = ed.quant_config(block, scheme)
    cfg_rec = json.loads(json.dumps(cfg))
    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    model = Qwen3_5KevDecoder.from_hf(ed.HF_ID, target_dtype=torch.float16, max_context_length=4096)
    n_unrolled = set_unrolled_scan(model)
    spec = model.build_export_spec(torch.float16, 4096, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=CHUNK)
    before = {n: m.weight.detach().clone() for n, m in model.named_modules()
              if isinstance(m, (torch.nn.Linear, torch.nn.Embedding))}
    t_load = time.monotonic() - t0
    model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg)
    t_q = time.monotonic() - t0 - t_load
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file: dict[str, dict] = {}
    info, other = {}, []
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod, "weight"):
            if P.is_parametrized(mod):
                other.append(name)
            continue
        deq = [p for p in mod.parametrizations["weight"] if hasattr(p, "quantized_data")]
        assert len(deq) == 1, name
        d = deq[0]
        w = mod.weight.detach().clone()
        again = torch.ops.coreai.constexpr_blockwise_shift_scale(
            d.quantized_data, d.scale, zero_point=d.zero_point, minval=d.minval,
            input_dtype=d.input_dtype, output_dtype=d.output_dtype)
        assert torch.equal(w, again), name
        q, ref = d.quantized_data, before[name].float()
        info[name] = {"type": type(mod).__name__, "shape": list(w.shape), "dtype": str(w.dtype), "codes_dtype": str(q.dtype),
                      "codes_min": int(q.min()), "codes_max": int(q.max()), "scale_shape": list(d.scale.shape),
                      "zero_point_absmax": None if d.zero_point is None else int(d.zero_point.abs().max()),
                      "rel_err": float((w.float() - ref).norm() / ref.norm()),
                      "max_abs_err": float((w.float() - ref).abs().max())}
        fname = "embed.safetensors" if isinstance(mod, torch.nn.Embedding) else f"layer_{layer_of(name):02d}.safetensors"
        per_file.setdefault(fname, {})[name] = w.contiguous()
    lin = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    want = ["model.embed_tokens"] if v == "embed32" else lin
    if other or sorted(info) != sorted(want) or (v != "embed32" and len(lin) != R6_LINEARS):
        sys.exit(f"quantized set differs: other {other}, {len(info)} quantized, want {len(want)}")
    files = {}
    for fname, ws in sorted(per_file.items()):
        p = out_dir / fname
        save_file(ws, str(p))
        files[fname] = {"file": str(p), "sha256": sha256(p), "modules": sorted(ws)}
    rel = [x["rel_err"] for x in info.values()]
    meta = {"variant": v, "config": cfg_rec, "model": "kev-4b", "hf_id": ed.HF_ID, "export_spec_query_len": CHUNK,
            "unrolled_gdn_layers": n_unrolled, "quantized_modules": len(info),
            "params": int(sum(int(np.prod(x["shape"])) for x in info.values())),
            "codes_range": [min(x["codes_min"] for x in info.values()), max(x["codes_max"] for x in info.values())],
            "rel_err": {"median": float(np.median(rel)), "max": float(np.max(rel)), "min": float(np.min(rel))},
            "load_seconds": t_load, "quantize_seconds": t_q, "files": files, "modules": info,
            "merged_files_sha256": ed.MERGED_FILES, "script_sha256": sha256(Path(__file__).resolve()),
            "generated_at": now()}
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dump {v}: {len(info)} modules ({meta['params']:,} params) quantized in {t_q:.0f}s; codes {meta['codes_range']}; "
          f"rel err median {meta['rel_err']['median']:.4e} max {meta['rel_err']['max']:.4e} -> {out_dir}", flush=True)


class R6Weights:
    """Moves the 248 body linears' and the table's weights between the merged checkpoint (exact, fp32) and the dumps
    (fp16 dequantized, cast to fp32) through safetensors' mmap; the process keeps one fp32 model and nothing else."""

    def __init__(self, model, hf_id: str):
        import torch

        self.torch = torch
        snap = Path(hf_snapshot(hf_id))
        wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        mods = dict(model.named_modules())
        self.linears = sorted(n for n, m in mods.items() if isinstance(m, torch.nn.Linear) and n.startswith("model.layers."))
        assert len(self.linears) == R6_LINEARS, len(self.linears)
        self.names = self.linears + ["model.embed_tokens"]
        self.mods = {n: mods[n] for n in self.names}
        self.key = {n: n[len("model."):] + ".weight" for n in self.names}
        self.ckpt = {n: snap / wm[self.key[n]] for n in self.names}
        self.params = {n: int(self.mods[n].weight.numel()) for n in self.linears}
        self.state = {n: "exact" for n in self.names}
        self.dump_meta = {}
        for v in list(R6_VARIANTS) + ["embed32"]:
            p = R6_DUMP / f"{v}.json"
            if p.exists():
                self.dump_meta[v] = {"path": str(p), "sha256": sha256(p)}
        # the module holds the checkpoint's own values: check the first and last layer and the table
        probe = [n for n in self.names if layer_of(n) in (0, R6_LAYERS - 1)] + ["model.embed_tokens"]
        self.probe_bit_equal = self._compare(probe)
        assert all(self.probe_bit_equal.values()), self.probe_bit_equal

    def _source(self, n: str, state: str) -> tuple[Path, str]:
        if state == "exact":
            return self.ckpt[n], self.key[n]
        if state == "embed32":
            return R6_DUMP / "embed32" / "embed.safetensors", n
        return R6_DUMP / state / f"layer_{layer_of(n):02d}.safetensors", n

    def _compare(self, names: list[str]) -> dict:
        from safetensors import safe_open

        out, by_file = {}, {}
        for n in names:
            by_file.setdefault(str(self.ckpt[n]), []).append(n)
        for f_, ns in by_file.items():
            with safe_open(f_, framework="pt", device="cpu") as f:
                for n in ns:
                    out[n] = bool(self.torch.equal(self.mods[n].weight, f.get_tensor(self.key[n])))
        return out

    def apply(self, want: dict) -> int:
        from safetensors import safe_open

        todo: dict[str, list] = {}
        for n in self.names:
            s = want[n]
            if self.state[n] != s:
                path, key = self._source(n, s)
                todo.setdefault(str(path), []).append((n, key, s))
        changed = 0
        with self.torch.no_grad():
            for path, items in todo.items():
                with safe_open(path, framework="pt", device="cpu") as f:
                    for n, key, s in items:
                        t = f.get_tensor(key)
                        assert t.shape == self.mods[n].weight.shape, (n, s)
                        self.mods[n].weight.copy_(t.to(self.torch.float32))
                        self.state[n] = s
                        changed += 1
        return changed


def r6_run(args) -> None:
    import parity_decoder_torch as pdt

    pdt.configure("kev-4b")
    rdoc = json.loads(R6_RULE.read_text())
    rows = [tuple(x) for x in rdoc["rows"]]
    _, recs = pdt.load_oracle()
    p1 = {}
    for p in sorted(R6_PARITY.glob("p1_shard*of*.jsonl")):
        for line in p.read_text().splitlines():
            x = json.loads(line)
            if x.get("kind") == "p1":
                p1[(x["id"], x["q"])] = x["probs"]
    part_path = Path(args.part)
    part = json.loads(part_path.read_text()) if part_path.exists() else {"configs": {}}
    t0 = time.monotonic()
    head = pdt.KevHead()
    runner = pdt.Runner(args.threads, "bmm")
    W = R6Weights(runner.model, pdt.HF_ID)
    load_s = time.monotonic() - t0
    params = W.params
    table = int(W.mods["model.embed_tokens"].weight.numel())
    body = sum(params.values())
    todo = [c for c in args.configs.split(",") if c]
    canon = {c: r6_canonical(c) for c in todo}
    part.update({"pid": os.getpid(), "threads": args.threads, "conv": "bmm", "chunk": CHUNK, "model": "kev-4b",
                 "rows": [list(k) for k in rows], "rule_sha256": sha256(R6_RULE),
                 "script_sha256": sha256(Path(__file__).resolve()), "load_seconds": load_s,
                 "dumps": W.dump_meta, "probe_bit_equal_checkpoint": all(W.probe_bit_equal.values()),
                 "body_linear_params": body, "table_params": table, "started": part.get("started", now())})

    def save() -> None:
        tmp = part_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, part_path)

    first = [True]

    def one_row(rid: str, k: int) -> dict:
        q = recs[rid]["questions"][k]
        t1 = time.monotonic()
        res = runner.run(q["row_ids"], CHUNK, conv_check=first[0])
        first[0] = False
        sc = pdt.score_question(head, res["hidden"], q)
        out = {k_: sc[k_] for k_ in ("max_abs_dp", "mean_abs_dp", "argmax", "argmax_oracle", "argmax_equal",
                                       "near_tie", "probs")}
        out.update({"tokens": res["T"], "seconds": time.monotonic() - t1})
        ref = p1.get((rid, k))
        out["bit_equal_p1"] = None if ref is None else bool(sc["probs"] == ref)
        return out

    print(f"[{os.getpid()}] loaded in {load_s:.0f}s; {len(rows)} rows; configs {todo}", flush=True)
    for name in todo:
        cname = canon[name]
        if any(r6_canonical(k) == cname for k in part["configs"]) and not args.redo:
            print(f"[{os.getpid()}] {name}: done already", flush=True)
            continue
        c = r6_parse(name)
        need = sorted({s for s in r6_states(c, W.linears).values() if s != "exact"})
        t_wait = time.monotonic()
        while [v for v in need if v not in W.dump_meta]:
            for v in need:
                p_ = R6_DUMP / f"{v}.json"
                if v not in W.dump_meta and p_.exists():
                    W.dump_meta[v] = {"path": str(p_), "sha256": sha256(p_)}
            if [v for v in need if v not in W.dump_meta]:
                if not args.wait_dumps or time.monotonic() - t_wait > 1800:
                    sys.exit(f"{name}: no dump for {[v for v in need if v not in W.dump_meta]}")
                time.sleep(10)
        part["dumps"] = W.dump_meta
        t1 = time.monotonic()
        changed = W.apply(r6_states(c, W.linears))
        sz = r6_size(c, params, table)
        rec = {"canonical": cname, "variant": c["variant"], "groups": c["groups"], "kinds": c["kinds"], "embed": c["embed"],
               "fp16_modules": sum(W.state[n] == "exact" for n in W.linears),
               "int8_modules": sum(W.state[n] != "exact" for n in W.linears), **sz,
               "modules_swapped": changed, "swap_seconds": time.monotonic() - t1, "rows": {}}
        for rid, k in rows:
            rec["rows"][f"{rid}:q{k}"] = one_row(rid, k)
        rr = rec["rows"].values()
        rec["worst_max_abs_dp"] = max(v["max_abs_dp"] for v in rr)
        rec["mean_of_row_mean_abs_dp"] = float(np.mean([v["mean_abs_dp"] for v in rr]))
        rec["argmax_equal_non_near_tie"] = all(v["argmax_equal"] for v in rr if not v["near_tie"])
        rec["seconds"] = time.monotonic() - t1
        rec["finished"] = now()
        part["configs"][name] = rec
        part["conv"] = runner.conv.report()
        save()
        worst = max(rec["rows"].items(), key=lambda kv: kv[1]["max_abs_dp"])
        print(f"[{os.getpid()}] {name}: worst {rec['worst_max_abs_dp']:.4f} ({worst[0]}) mean "
              f"{rec['mean_of_row_mean_abs_dp']:.5f} fp16 {sz['fp16_fraction']:.1%} est {sz['est_main_mlirb_bytes']:,} B "
              f"{rec['seconds']:.0f}s", flush=True)
    W.apply({n: "exact" for n in W.names})
    rid, k = rows[0]
    again = one_row(rid, k)
    part["swap_back_check"] = {"row": f"{rid}:q{k}", "bit_equal_p1": again["bit_equal_p1"],
                               "max_abs_dp_vs_oracle": again["max_abs_dp"], "at": now()}
    part["conv"] = runner.conv.report()
    part["finished"] = now()
    save()
    print(f"[{os.getpid()}] swap-back {rid}:q{k} bit-equal P1 {again['bit_equal_p1']}; conv {json.dumps(part['conv'])}",
          flush=True)


def r6_load_parts() -> tuple[dict, list[dict]]:
    table, procs = {}, []
    for p in sorted(R6_PARTS.glob("part_*.json")):
        d = json.loads(p.read_text())
        procs.append({"part": str(p), "pid": d.get("pid"), "threads": d.get("threads"), "load_seconds": d.get("load_seconds"),
                      "rule_sha256": d.get("rule_sha256"), "script_sha256": d.get("script_sha256"), "dumps": d.get("dumps"),
                      "probe_bit_equal_checkpoint": d.get("probe_bit_equal_checkpoint"),
                      "started": d.get("started"), "finished": d.get("finished"), "conv": d.get("conv"),
                      "swap_back_check": d.get("swap_back_check"), "configs": list(d["configs"])})
        for name, rec in d["configs"].items():
            cname = rec["canonical"]
            if cname in table:
                a = {k: v["probs"] for k, v in table[cname]["rows"].items()}
                b = {k: v["probs"] for k, v in rec["rows"].items()}
                rec["repeat_bit_equal"] = a == b
            rec["name"] = name
            table[cname] = rec
    return table, procs


def r6_meets(rec: dict) -> bool:
    return rec["worst_max_abs_dp"] <= R6_TARGET_WORST and rec["mean_of_row_mean_abs_dp"] <= R6_TARGET_MEAN


def r6_rank_key(rec: dict):
    return (rec["mean_of_row_mean_abs_dp"], rec["worst_max_abs_dp"], rec["canonical"])


def r6_plan(table: dict) -> dict:
    """What the rule has evaluated, what it asks for next, and the choice (canonical names)."""
    rdoc = json.loads(R6_RULE.read_text())
    params, table_params = r6_module_params()
    cap = rdoc["cap_params"]
    fixed = [r6_canonical(n) for n in r6_fixed()]
    missing = [n for n, c in zip(r6_fixed(), fixed) if c not in table]
    if missing:
        return {"stage": "fixed", "next": missing}
    out: dict = {"cap_params": cap, "next": []}
    # reproduction
    gate = json.loads(from_record_path(rdoc["gate_transcript"]["path"]).read_text())
    g_by = {f"{r['id']}:q{r['q']}": r["max_abs_dp"] for r in gate["runs"]}
    rows = [f"{a}:q{b}" for a, b in rdoc["rows"]]
    gw = max(g_by[r] for r in rows)
    tw = table["set_b32"]["worst_max_abs_dp"]
    out["reproduction"] = {"gate_worst": gw, "torch_all_int8_b32_worst": tw, "ratio": tw / gw, "reproduced": tw >= 0.5 * gw}
    if not out["reproduction"]["reproduced"]:
        return {**out, "stage": "stop: all_int8_b32 does not reproduce the gate"}
    kin = sorted(KINDS, key=lambda k: r6_rank_key(table[f"set_b32__K_{k}"]))
    grp = sorted(range(R6_LAYERS // R6_GROUP), key=lambda g: r6_rank_key(table[f"set_b32__G_{g}"]))
    var = sorted(R6_VARIANTS, key=lambda v: (table[f"set_{v}"]["mean_of_row_mean_abs_dp"], list(R6_VARIANTS).index(v)))
    out.update({"kind_ranking": kin, "group_ranking": grp, "variant_ranking": var, "best_variant": var[0]})

    def size(c: dict) -> dict:
        return r6_size(c, params, table_params)

    def within(name: str) -> bool:
        return size(r6_parse(name))["fp16_params"] <= cap

    # (ii) kind sets K_j: the ranking walked, kinds that would cross the cap skipped
    ksets, cur = [], []
    for k in kin:
        if size({"variant": "b32", "groups": [], "kinds": cur + [k], "embed": False})["fp16_params"] <= cap:
            cur = cur + [k]
            ksets.append(list(cur))
    k_names = [r6_set_name("b32", kinds=s) for s in ksets]          # K_1 = a fixed proj config
    # (iii) group sets G_k
    gsets = []
    for n in range(1, len(grp) + 1):
        s = sorted(grp[:n])
        if size({"variant": "b32", "groups": s, "kinds": [], "embed": False})["fp16_params"] > cap:
            break
        gsets.append(s)
    g_names = [r6_set_name("b32", groups=s) for s in gsets]
    stages = [("ii", k_names[1:1 + R6_LIMITS["ii"]]), ("iii", g_names[1:])]
    if var[0] != "b32":
        iv = sorted({r6_set_name(var[0], kinds=s) for s in ksets[:1 + R6_LIMITS["ii"]]}
                    | {r6_set_name(var[0], groups=s) for s in gsets},
                    key=lambda n: (size(r6_parse(n))["fp16_params"], n))
        stages.append(("iv", iv[:R6_LIMITS["iv"]]))
    unions = []
    for kj, ks in enumerate(ksets[:1 + R6_LIMITS["ii"]], 1):
        for gk, gs in enumerate(gsets, 1):
            nm = r6_set_name(var[0], groups=gs, kinds=ks)
            sz = size(r6_parse(nm))
            if sz["fp16_params"] <= cap:
                unions.append((sz["fp16_params"], gk, kj, nm))
    stages.append(("v", [u[3] for u in sorted(unions)][:R6_LIMITS["v"]]))
    out["stages"] = {s: names for s, names in stages}

    def meeting() -> list[str]:
        return [n for n, r in table.items() if r6_meets(r) and within(n) and not r6_parse(n)["embed"]
                and n != "set_fp32"]

    for stage, names in stages:
        if meeting():
            break
        todo = [n for n in names if n not in table]
        if todo:
            # within a stage the sets are evaluated in order; stop at the first that meets the target
            evaluated = [n for n in names if n in table]
            if not any(r6_meets(table[n]) for n in evaluated):
                return {**out, "stage": stage, "next": todo}
    beyond = [r6_set_name("b32", groups=sorted(grp[:n])) for n in (3, 4)][:R6_LIMITS["beyond_cap"]]
    out["beyond_cap_reference"] = beyond
    m = meeting()
    if m:
        best = sorted(m, key=lambda n: (size(r6_parse(n))["fp16_params"], size(r6_parse(n))["est_main_mlirb_bytes"],
                                        table[n]["mean_of_row_mean_abs_dp"]))
        chosen = [best[0]]
    else:
        pool = [n for n in table if within(n) and n not in ("set_fp32", "set_fp32__E") and not r6_parse(n)["embed"]]
        chosen = sorted(pool, key=lambda n: r6_rank_key(table[n]))[:2]
    out["meeting"] = m
    out["chosen"] = chosen
    out["chosen_meets_target"] = bool(m)
    plus_e = [r6_set_name(**{**r6_parse(n), "embed": True}) for n in chosen]
    out["embed_candidates"] = plus_e
    todo = [n for n in plus_e if n not in table]
    if todo:
        return {**out, "stage": "embed", "next": todo}
    out["embed_adopted"] = {n: table[e]["worst_max_abs_dp"] - table[n]["worst_max_abs_dp"] <= R6_EMBED_SLACK
                            for n, e in zip(chosen, plus_e)}
    return {**out, "stage": "chosen", "next": [b for b in beyond if b not in table]}


def r6_plan_cmd(args) -> None:
    table, _ = r6_load_parts()
    print(json.dumps(r6_plan(table), indent=1))


def r6_merge(args) -> None:
    rdoc = json.loads(R6_RULE.read_text())
    table, procs = r6_load_parts()
    p = r6_plan(table)
    params, table_params = r6_module_params()
    gate = json.loads(from_record_path(rdoc["gate_transcript"]["path"]).read_text())
    g_by = {f"{r['id']}:q{r['q']}": r["max_abs_dp"] for r in gate["runs"]}
    rows = [f"{a}:q{b}" for a, b in rdoc["rows"]]
    top, rand = rows[:R6_N_TOP], rows[R6_N_TOP:]
    base = table.get("set_b32")
    repro = None
    if base:
        x = np.array([g_by[r] for r in rows])
        y = np.array([base["rows"][r]["max_abs_dp"] for r in rows])
        repro = {"gate_worst": float(x.max()), "torch_all_int8_b32_worst": base["worst_max_abs_dp"],
                 "ratio": base["worst_max_abs_dp"] / float(x.max()), "reproduced": base["worst_max_abs_dp"] >= 0.5 * float(x.max()),
                 "pearson_per_row": float(np.corrcoef(x, y)[0, 1]),
                 "rows_above_bar_gate": int((x > BAR_MAX_ABS_DP).sum()), "rows_above_bar_torch": int((y > BAR_MAX_ABS_DP).sum()),
                 "per_row": [{"row": r, "gate": g_by[r], "torch": base["rows"][r]["max_abs_dp"]} for r in rows]}
    floor = None
    if "set_fp32" in table:
        ex = table["set_fp32"]["rows"]
        floor = {"worst_max_abs_dp": table["set_fp32"]["worst_max_abs_dp"],
                 "bit_equal_p1_rows": sum(bool(v["bit_equal_p1"]) for v in ex.values()), "rows": len(ex)}
    cap = rdoc["cap_params"]
    tab = []
    for cname, rec in table.items():
        rr = rec["rows"]
        tab.append({"config": rec["name"], "canonical": cname, "variant": rec["variant"], "groups": rec["groups"],
                    "kinds": rec["kinds"], "embed": rec["embed"], "fp16_params": rec["fp16_params"],
                    "fp16_fraction": rec["fp16_fraction"], "within_cap": rec["fp16_params"] <= cap,
                    "est_main_mlirb_bytes": rec["est_main_mlirb_bytes"],
                    "worst_max_abs_dp": rec["worst_max_abs_dp"], "mean_of_row_mean_abs_dp": rec["mean_of_row_mean_abs_dp"],
                    "worst_top20": max(rr[r]["max_abs_dp"] for r in top),
                    "worst_random20": max(rr[r]["max_abs_dp"] for r in rand),
                    "mean_random20": float(np.mean([rr[r]["mean_abs_dp"] for r in rand])),
                    "argmax_equal_non_near_tie": rec["argmax_equal_non_near_tie"],
                    "rows_not_worse_than_all_int8_b32": (sum(rr[r]["max_abs_dp"] <= base["rows"][r]["max_abs_dp"]
                                                             for r in rows) if base else None),
                    "meets_target": r6_meets(rec), "seconds": rec["seconds"]})
    tab.sort(key=lambda t: (t["mean_of_row_mean_abs_dp"], t["worst_max_abs_dp"], t["canonical"]))
    for i, t in enumerate(tab, 1):
        t["rank_by_mean"] = i
    out = {"schema": "kev-int8-bisect-4b/1", "model": "kev-4b", "rule": rdoc["rule"], "rule_file": str(R6_RULE),
           "rule_sha256": sha256(R6_RULE), "rule_written_at": rdoc["written_at"], "rows": rdoc["rows"],
           "target": rdoc["target"], "cap_params": cap, "body_linear_params": rdoc["body_linear_params"],
           "script": {"path": "conversion/kev/int8_bisect_torch.py", "sha256": sha256(Path(__file__).resolve())},
           "dumps": {v: {"meta": str(R6_DUMP / f"{v}.json"), "sha256": sha256(R6_DUMP / f"{v}.json")}
                     for v in list(R6_VARIANTS) + ["embed32"] if (R6_DUMP / f"{v}.json").exists()},
           "reproduction_check": repro, "instrument_floor": floor, "processes": procs, "plan": p,
           "chosen": p.get("chosen"), "table": tab, "configs": table, "generated_at": now()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} configs -> {args.out}; reproduction {None if repro is None else (round(repro['ratio'], 3), repro['reproduced'])}; "
          f"floor {floor}; stage {p.get('stage')} chosen {p.get('chosen')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="kev-0.8b", choices=["kev-0.8b", "kev-4b"],
                    help="kev-4b: the round-6 instrument (bisect_4b/, its own rule, variants and groups)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--variant", choices=list(R6_VARIANTS) + ["embed32"], help="kev-4b: which variant to dump")
    r = sub.add_parser("rule")
    r.add_argument("--gate", required=True, help="the int8lin gate transcript (its per-row max|dp| picks the top rows)")
    b = sub.add_parser("run")
    b.add_argument("--part", required=True)
    b.add_argument("--configs", required=True, help="comma list of config names")
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--redo", action="store_true")
    b.add_argument("--wait-dumps", action="store_true", help="kev-4b: wait (up to 30 min) for a dump a config needs")
    sub.add_parser("plan")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.model == "kev-4b":
        if args.cmd == "dump" and not args.variant:
            ap.error("kev-4b dump needs --variant")
        {"dump": r6_dump, "rule": r6_rule, "run": r6_run, "plan": r6_plan_cmd, "merge": r6_merge}[args.cmd](args)
        return
    {"dump": dump, "rule": rule, "run": run, "plan": plan, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
