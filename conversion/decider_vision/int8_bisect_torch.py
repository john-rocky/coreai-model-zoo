#!/usr/bin/env python3
"""Which layers carry the int8 body's error? fp32 torch with the exported int8 weights, per config.

The instrument is `parity_decoder_torch.py`'s: the ids-input decoder in fp32 on the CPU, S=1 from
fresh zero states, the bmm depthwise conv (checked against F.conv1d on the first run of every
process), oracle ids and the oracle's fp32 image_embeds, letter softmax (T=1) at every slot vs
`fixture_oracle.json`. What changes per config is which linear weights are int8:

  every module the exporter's int8lin config quantizes (`export_decoder.linear_quant_config`: the
  body's linears; head, embeddings, norms, conv excluded) holds either the checkpoint's own weight
  (bf16 in the file, so fp32 here is exact) or the weight the exported op computes — the
  exporter's own `quantize_pytorch_model` call on the fp16 model (same config, same reference
  inputs, same GDN flags), finalized, and read back through the finalized module's own
  dequantization (`coreai::constexpr_blockwise_shift_scale` on the int8 codes and fp16 block
  scales, fp16 out), then cast to fp32. A config names the modules that are int8; the rest are
  exact. Only the weights differ between configs; activations stay fp32.

    dump   quantize the fp16 model (qscheme `symmetric_with_clipping` = the export, or `symmetric`)
           and save the dequantized fp16 weight of every quantized module
    run    evaluate configs (default: all of `configs()`) on the bisect runs; one process, one part
           file, resumable
    merge  collect the part files into one table (per config: the runs' letter p and |dp|)

    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 python int8_bisect_torch.py dump --qscheme clipping
    ... int8_bisect_torch.py run --configs int8lin,exact --part <dir>/part_a.json
    ... int8_bisect_torch.py merge --parts <dir> --out ~/code/coreai/_decider2bv/logs/r6_bisect.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_decider2bv", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

LANE = work_path("_decider2bv")
ORACLE = LANE / "oracle"
DUMPS = LANE / "scratch" / "r6"
N_LAYERS = 24
RUNS = [("r29", "g256"), ("r26", "g448"), ("r28", "g256"), ("r20", "g256")]
GDN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def layer_of(name: str) -> int:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def kind_of(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if ".mlp." in name and leaf in MLP:
        return "mlp"
    if ".self_attn." in name and leaf in ATTN:
        return "attn"
    if ".linear_attn." in name and leaf in GDN:
        return "gdn"
    return "other"


def configs(names: list[str]) -> dict[str, dict]:
    """name -> {"int8": [module names], "dump": qscheme, "why": ...}. `names` = every quantized
    module (the dump's keys)."""
    every = sorted(names)
    out = {
        "exact": {"int8": [], "dump": "clipping", "why": "no int8 (instrument floor vs the oracle)"},
        "int8lin": {"int8": every, "dump": "clipping", "why": "the export's int8lin body (A)"},
        "only_mlp": {"int8": [n for n in every if kind_of(n) == "mlp"], "dump": "clipping",
                     "why": "B1: MLP gate/up/down int8, the rest exact"},
        "only_attn": {"int8": [n for n in every if kind_of(n) == "attn"], "dump": "clipping",
                      "why": "B1: full-attention q/k/v/o int8, the rest exact"},
        "only_gdn": {"int8": [n for n in every if kind_of(n) == "gdn"], "dump": "clipping",
                     "why": "B1: GDN in_proj_qkv/z/b/a + out_proj int8, the rest exact"},
    }
    for i in range(N_LAYERS):
        out[f"fp16_L{i:02d}"] = {"int8": [n for n in every if layer_of(n) != i], "dump": "clipping",
                                 "fp16_layers": [i], "why": f"B2: layer {i} exact, every other layer int8"}
    out["fp16_L00-11"] = {"int8": [n for n in every if layer_of(n) >= 12], "dump": "clipping",
                          "fp16_layers": list(range(12)), "why": "B2: first 12 layers exact, last 12 int8"}
    out["fp16_L12-23"] = {"int8": [n for n in every if layer_of(n) < 12], "dump": "clipping",
                          "fp16_layers": list(range(12, 24)), "why": "B2: last 12 layers exact, first 12 int8"}
    out["int8lin_symmetric"] = {"int8": every, "dump": "symmetric",
                                "why": "B3: the int8lin body with qscheme symmetric (grid -128..127, scale "
                                       "absmax/127.5) instead of symmetric_with_clipping (-127..127, absmax/127)"}
    for k in (kind_of(n) for n in every):
        assert k != "other", "a quantized module outside mlp/attn/gdn"
    return out


def fp16_set_config(every: list[str], layers: list[int]) -> dict:
    return {"int8": [n for n in every if layer_of(n) not in set(layers)], "dump": "clipping",
            "fp16_layers": sorted(layers), "why": f"layers {sorted(layers)} exact, every other layer int8"}


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def dump(args) -> None:
    import torch
    import torch.nn.utils.parametrize as P
    from export_decoder import HF_ID, linear_quant_config
    from qwen3_5_vl_pipelined import Qwen3_5VLPipelinedForCausalLM

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.compression import quantize_pytorch_model

    torch.set_num_threads(args.threads)
    t0 = time.monotonic()
    model = Qwen3_5VLPipelinedForCausalLM.from_hf(HF_ID, target_dtype=torch.float16, max_context_length=4096,
                                                  n_image_max=256)
    for layer in model.model.layers:          # the exporter's flags, before quantizing
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            if args.unroll:
                layer.linear_attn.use_loopfree_unroll = True
    spec = model.build_export_spec(torch.float16, 4096, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
    cfg = linear_quant_config("int8")
    if args.qscheme == "symmetric":
        cfg["global_config"]["op_state_spec"]["weight"]["qscheme"] = "symmetric"
    qspec = json.loads(json.dumps(cfg["global_config"]["op_state_spec"]["weight"]))
    before = {n: m.weight.detach().clone() for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg)
    t_q = time.monotonic() - t0
    weights, info = {}, {}
    for name, mod in model.named_modules():
        if not P.is_parametrized(mod, "weight"):
            continue
        plist = mod.parametrizations["weight"]
        deq = [p for p in plist if hasattr(p, "quantized_data")]
        assert len(deq) == 1, (name, [type(p).__name__ for p in plist])
        d = deq[0]
        w = mod.weight.detach().clone()                    # the finalized module's own dequantization
        again = torch.ops.coreai.constexpr_blockwise_shift_scale(
            d.quantized_data, d.scale, zero_point=d.zero_point, minval=d.minval,
            input_dtype=d.input_dtype, output_dtype=d.output_dtype)
        assert torch.equal(w, again), name
        q = d.quantized_data
        info[name] = {"shape": list(w.shape), "dtype": str(w.dtype), "codes_dtype": str(q.dtype),
                      "codes_min": int(q.min()), "codes_max": int(q.max()),
                      "scale_shape": list(d.scale.shape), "scale_dtype": str(d.scale.dtype),
                      "zero_point": None if d.zero_point is None else int(d.zero_point.abs().max()),
                      "minval": d.minval is not None,
                      "rel_err": float((w.float() - before[name].float()).norm() / before[name].float().norm())}
        weights[name] = w
    missing = sorted(set(before) - set(weights))
    out = DUMPS / f"int8_{args.qscheme}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    meta = {"qscheme_arg": args.qscheme, "weight_spec": qspec, "hf_id": HF_ID, "unroll": args.unroll,
            "quantized_modules": len(weights), "linear_modules_not_quantized": missing,
            "params": int(sum(w.numel() for w in weights.values())), "quantize_seconds": t_q,
            "file": str(out), "sha256": sha256(out), "modules": info,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"dumped {len(weights)} modules ({meta['params']:,} params) to {out}; not quantized: {missing}; "
          f"rel err median {np.median([v['rel_err'] for v in info.values()]):.4e}", flush=True)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
class Weights:
    """Swaps the quantizable modules' weights between exact (bf16-exact fp32) and int8 (dump)."""

    def __init__(self, model, qschemes: list[str]):
        import torch
        self.torch = torch
        self.dumps = {q: torch.load(DUMPS / f"int8_{q}.pt") for q in qschemes}
        names = sorted(next(iter(self.dumps.values())))
        for q, d in self.dumps.items():
            assert sorted(d) == names, f"dump {q} has a different module set"
        mods = dict(model.named_modules())
        self.mods = {n: mods[n] for n in names}
        self.exact = {}
        for n, m in self.mods.items():
            w = m.weight.detach()
            b = w.to(torch.bfloat16)
            assert torch.equal(b.float(), w), f"{n}: fp32 weight is not bf16-exact"
            self.exact[n] = b
        self.state = {n: "exact" for n in names}
        self.names = names

    def apply(self, int8: list[str], qscheme: str) -> int:
        want = set(int8)
        changed = 0
        with self.torch.no_grad():
            for n, m in self.mods.items():
                s = qscheme if n in want else "exact"
                if self.state[n] == s:
                    continue
                src = self.exact[n] if s == "exact" else self.dumps[s][n]
                m.weight.copy_(src.float())
                self.state[n] = s
                changed += 1
        return changed


def run_row(runner, ids, emb, rc, start, amount, read_steps, check_conv: bool) -> dict:
    torch = runner.torch
    T = int(ids.shape[0])
    st = runner.q.build_decode_state(runner.cfg, max_seq_len=T + 8, dtype=torch.float32)
    last = max(read_steps)
    read = {}
    try:
        with torch.inference_mode():
            for t in range(T):
                runner.conv.check = check_conv and t == last
                pos = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
                out = runner.model(ids[t].reshape(1, 1), pos, emb, rc, start, amount, st["k_cache"],
                                   st["v_cache"], st["conv_state"], st["rec_state"])
                if t in read_steps:
                    read[t] = out[0, -1].clone()
    finally:
        runner.conv.check = False
    return read


def score(row: dict, read: dict) -> dict:
    import torch
    slots = []
    for s, (t, nopts) in enumerate(zip(row["slot_idx"], row["nopts"])):
        lg = read[t][32:32 + nopts].double()
        p = torch.softmax(lg, -1).numpy()
        po = np.asarray(row["probs"][s], dtype=np.float64)
        d = np.abs(p - po)
        slots.append({"t": t, "nopts": nopts, "p": p.tolist(), "p_oracle": po.tolist(),
                      "dp": (p - po).tolist(), "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                      "argmax": int(p.argmax()), "argmax_oracle": row["argmax"][s],
                      "full_vocab_top1": int(torch.argmax(read[t])),
                      "letter_logits": read[t][32:32 + nopts].tolist()})
    return {"max_abs_dp": max(x["max_abs_dp"] for x in slots),
            "argmax_equal": all(x["argmax"] == x["argmax_oracle"] for x in slots),
            "full_vocab_top1_is_letter": all(x["full_vocab_top1"] == 32 + x["argmax_oracle"] for x in slots),
            "slots": slots}


def run(args) -> None:
    import torch
    from parity_decoder_torch import Runner

    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    by_key = {(r["id"], r["arm"]): r for r in fx["rows"]}
    runs = [tuple(k.split(":")) for k in args.runs.split(",")] if args.runs else RUNS
    part_path = Path(args.part)
    part = json.loads(part_path.read_text()) if part_path.exists() else {"configs": {}}
    runner = Runner(args.threads, need_plain=False)
    qs = sorted(set(args.dumps.split(",")))
    W = Weights(runner.model, qs)
    cfgs = configs(W.names)
    for extra in (args.sets or "").split(";"):
        if extra.strip():
            layers = [int(x) for x in extra.split(",")]
            cfgs["fp16_set_" + "_".join(f"L{i:02d}" for i in sorted(layers))] = fp16_set_config(W.names, layers)
    todo = args.configs.split(",") if args.configs else list(cfgs)
    unknown = [c for c in todo if c not in cfgs]
    if unknown:
        sys.exit(f"unknown configs {unknown}")
    inputs = {}
    for k in runs:
        npz = np.load(ORACLE / "npz" / f"{k[0]}__{k[1]}.npz")
        inputs[k] = runner.static_inputs(by_key[k], npz)
    part.update({"pid": os.getpid(), "threads": args.threads, "runs": [list(k) for k in runs],
                 "dumps": {q: {"file": str(DUMPS / f"int8_{q}.pt"),
                               "meta_sha256": sha256(DUMPS / f"int8_{q}.json")} for q in qs}})
    first = True
    for name in todo:
        if name in part["configs"] and not args.redo:
            print(f"[{os.getpid()}] {name}: done already", flush=True)
            continue
        c = cfgs[name]
        if c["dump"] not in W.dumps:
            sys.exit(f"{name} needs the {c['dump']} dump (--dumps)")
        t0 = time.monotonic()
        changed = W.apply(c["int8"], c["dump"])
        rec = {"why": c["why"], "dump": c["dump"], "fp16_layers": c.get("fp16_layers"),
               "int8_modules": len(c["int8"]), "exact_modules": len(W.names) - len(c["int8"]),
               "int8_params": int(sum(W.mods[n].weight.numel() for n in c["int8"])),
               "modules_swapped": changed, "runs": {}}
        for k in runs:
            ids, emb, rc, start, amount = inputs[k]
            row = by_key[k]
            t1 = time.monotonic()
            read = run_row(runner, ids, emb, rc, start, amount, set(row["slot_idx"]), check_conv=first)
            sc = score(row, read)
            sc["seconds"] = time.monotonic() - t1
            rec["runs"][f"{k[0]}:{k[1]}"] = sc
            first = False
        rec["seconds"] = time.monotonic() - t0
        part["configs"][name] = rec
        part["conv"] = runner.conv.report()
        tmp = part_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, part_path)
        msg = " ".join(f"{k}={v['max_abs_dp']:.4f}" for k, v in rec["runs"].items())
        p29 = rec["runs"].get("r29:g256", {}).get("slots", [{}])[0].get("p")
        print(f"[{os.getpid()}] {name}: {msg} r29 p={None if p29 is None else [round(x, 4) for x in p29]} "
              f"({rec['seconds']:.0f}s, {changed} swapped)", flush=True)


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge(args) -> None:
    parts = sorted(Path(args.parts).glob("part_*.json"))
    table, procs = {}, []
    for p in parts:
        d = json.loads(p.read_text())
        procs.append({"part": str(p), "pid": d.get("pid"), "threads": d.get("threads"), "conv": d.get("conv"),
                      "configs": list(d["configs"])})
        for name, rec in d["configs"].items():
            if name in table:
                a = {k: v["max_abs_dp"] for k, v in table[name]["runs"].items()}
                b = {k: v["max_abs_dp"] for k, v in rec["runs"].items()}
                rec["repeat_equal"] = a == b
            table[name] = rec
    base = table.get("int8lin")
    rows = []
    for name, rec in table.items():
        r = {"config": name, "fp16_layers": rec.get("fp16_layers"), "dump": rec["dump"],
             "int8_params": rec["int8_params"]}
        for k, v in rec["runs"].items():
            r[k] = v["max_abs_dp"]
            r[k + ":p_argmax"] = v["slots"][0]["p"][v["slots"][0]["argmax_oracle"]]
        if base:
            r["no_run_worse_than_int8lin"] = all(rec["runs"][k]["max_abs_dp"] <= base["runs"][k]["max_abs_dp"]
                                                 for k in rec["runs"] if k in base["runs"])
        r["argmax_all_equal"] = all(v["argmax_equal"] for v in rec["runs"].values())
        rows.append(r)
    out = {"schema": "coreai-decider-vision-int8-bisect/1",
           "instrument": ("fp32 torch on the CPU, S=1 from fresh zero states, oracle ids + oracle fp32 image_embeds; "
                          "per config the listed modules carry the exporter's int8 weights (quantize_pytorch_model "
                          "on the fp16 model, dequantized by the finalized module), every other quantizable "
                          "module its exact checkpoint weight"),
           "script": {"path": "conversion/decider_vision/int8_bisect_torch.py", "sha256": sha256(Path(__file__))},
           "oracle": {"path": str(ORACLE / "fixture_oracle.json"), "sha256": sha256(ORACLE / "fixture_oracle.json")},
           "selection_rule": ("smallest fp16 layer set with r29 |dp| <= 0.010 and no bisect run worse than int8lin "
                              "(this instrument), at most 4 layers; fixed before the results"),
           "processes": procs, "table": rows, "configs": table,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(table)} configs -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("--qscheme", choices=["clipping", "symmetric"], default="clipping")
    a.add_argument("--unroll", action="store_true", help="also set use_loopfree_unroll (the pf16 exporter)")
    a.add_argument("--threads", type=int, default=4)
    b = sub.add_parser("run")
    b.add_argument("--configs", help="comma list (default: every config)")
    b.add_argument("--sets", help="extra fp16 layer sets, ';'-separated lists of layer indices, e.g. '9;9,21'")
    b.add_argument("--runs", help="comma list id:arm (default: r29:g256,r26:g448,r28:g256,r20:g256)")
    b.add_argument("--dumps", default="clipping", help="comma list of dumps to load (clipping,symmetric)")
    b.add_argument("--part", required=True)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--redo", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--parts", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"dump": dump, "run": run, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
