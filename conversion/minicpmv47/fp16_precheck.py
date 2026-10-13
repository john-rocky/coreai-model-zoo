#!/usr/bin/env python3
"""fp16 precheck before any fp16 export: activation absmax of the vision tower and the VL decoder, fp32 CPU.

fp16 tops out at 65504; this records, per stage, the largest |activation| over the prompts so an
export knows which stages need fp32 arithmetic or a rescaled residual before it is built. Runs
the re-authored tower (`minicpmv47_vision.py`, baked path (b), the export path) on P1..P4 and the
round-1 VL decoder (`minicpmv47_vl_pipelined.py`) on the same prompts with that tower's rows, both
in fp32 with forward hooks (nothing is changed in either module).

  tower  embeddings; per encoder layer: residual in / after attention / out, q / k / v, attention
         scores (q.k^T * 72**-0.5, recomputed from the hooked q / k), attention context and
         out_proj, fc1 (pre-GELU), fc2; window merger: input, q / k / v, window scores, out_proj,
         the 2x2 concat (pre_norm input), pre_norm out, linear_1 (pre-GELU), linear_2, output;
         post_layernorm in / out; merger: pre_norm in / out, linear_1 (pre-GELU), GELU, output.
  text   input embeddings; per layer: residual in / mid / out; full attention: q_proj (query |
         gate), q and k after their norms, v, the scores SDPA computes (q.k^T / 16 over the causal
         prefix, from SDPA's own inputs = after RoPE), the gated context (o_proj input), o_proj;
         GatedDeltaNet: its input, in_proj_qkv, in_proj_z, in_proj_a, the recurrence output
         (RMSNormGated input), out_proj; MLP: gate_proj, up_proj, gate x up (down_proj input),
         down_proj; final norm in / out; logits; the states after the prompt.
  norms  for every LayerNorm / RMSNorm input also the largest row sum of squares and row mean of
         squares — an fp16 norm that squares before it averages overflows at sum(x^2) > 65504 even
         when every |x| is far below it.
  cpu16  for scale only (CPU torch, not the device): the tower with fp16 weights and fp16 math, and
         with fp16 weights read in fp32 math ("fp16w32"), against the fp32 tower — per-token cosine.

Writes `fp16_precheck.json`: per stage the absmax per prompt and the max over prompts, the table
sorted by value, and the verdict line (largest stage, every stage above 30,000).

    QW coreai-models/.venv/bin/python -I conversion/minicpmv47/fp16_precheck.py [--threads 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
PROMPTS = ("P1", "P2", "P3", "P4")
RISK = 30000.0
FP16_MAX = 65504.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Recorder:
    """stage -> prompt -> max |x| (and, for norm inputs, the largest row sum / mean of squares)."""

    def __init__(self, torch):
        self.torch = torch
        self.absmax = defaultdict(dict)
        self.prompt = None
        self.hooks = []

    def put(self, stage: str, t, norm_input: bool = False) -> None:
        t = t.detach().float()
        v = float(t.abs().max())
        d = self.absmax[stage]
        d[self.prompt] = max(d.get(self.prompt, 0.0), v)
        if norm_input:
            sq = (t.double() ** 2).reshape(-1, t.shape[-1])
            for name, val in ((f"{stage}.row_sumsq", float(sq.sum(-1).max())),
                              (f"{stage}.row_meansq", float(sq.mean(-1).max()))):
                e = self.absmax[name]
                e[self.prompt] = max(e.get(self.prompt, 0.0), val)

    def out(self, mod, stage, norm_input=False):
        self.hooks.append(mod.register_forward_hook(lambda m, a, o: self.put(stage, o, norm_input)))

    def inp(self, mod, stage, norm_input=False):
        self.hooks.append(mod.register_forward_pre_hook(lambda m, a: self.put(stage, a[0], norm_input)))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def tower_hooks(rec: Recorder, tower, torch) -> None:
    vt = tower.vision_tower
    rec.out(vt.embeddings, "tower.embeddings")
    for i, layer in enumerate(vt.encoder.layers):
        p = f"tower.L{i:02d}"
        rec.inp(layer.layer_norm1, f"{p}.residual_in", norm_input=True)
        rec.inp(layer.layer_norm2, f"{p}.residual_mid", norm_input=True)
        rec.out(layer, f"{p}.residual_out")
        att = layer.self_attn
        qk = {}
        for name in ("q", "k", "v"):
            proj = getattr(att, f"{name}_proj")
            rec.out(proj, f"{p}.{name}")
            if name != "v":
                rec.hooks.append(proj.register_forward_hook(
                    lambda m, a, o, n=name, d=qk: d.__setitem__(n, o.detach())))

        def scores(m, a, o, p=p, qk=qk, att=att):
            q = qk["q"].reshape(-1, att.num_heads, att.head_dim).transpose(0, 1)    # [H, N, hd]
            k = qk["k"].reshape(-1, att.num_heads, att.head_dim).transpose(0, 1)
            rec.put(f"{p}.scores", torch.matmul(q, k.transpose(-2, -1)) * att.scaling)
        rec.hooks.append(att.k_proj.register_forward_hook(scores))
        rec.inp(att.out_proj, f"{p}.attn_ctx")
        rec.out(att.out_proj, f"{p}.attn_out")
        rec.out(layer.mlp.fc1, f"{p}.fc1_pre_gelu")
        rec.out(layer.mlp.fc2, f"{p}.fc2")
    wm = vt.vit_merger
    rec.inp(wm.layer_norm1, "tower.wm.input", norm_input=True)
    qk = {}
    for name in ("q", "k", "v"):
        proj = getattr(wm.self_attn, f"{name}_proj")
        rec.out(proj, f"tower.wm.{name}")
        if name != "v":
            rec.hooks.append(proj.register_forward_hook(
                lambda m, a, o, n=name, d=qk: d.__setitem__(n, o.detach())))

    def wscores(m, a, o, att=wm.self_attn, qk=qk):
        q = qk["q"].reshape(-1, 4, att.num_heads, att.head_dim).transpose(1, 2)        # [W, H, 4, hd]
        k = qk["k"].reshape(-1, 4, att.num_heads, att.head_dim).transpose(1, 2)
        rec.put("tower.wm.scores", torch.matmul(q, k.transpose(-2, -1)) * att.scaling)
    rec.hooks.append(wm.self_attn.k_proj.register_forward_hook(wscores))
    rec.out(wm.self_attn.out_proj, "tower.wm.attn_out")
    rec.inp(wm.pre_norm, "tower.wm.concat_2x2", norm_input=True)
    rec.out(wm.pre_norm, "tower.wm.pre_norm")
    rec.out(wm.linear_1, "tower.wm.linear_1_pre_gelu")
    rec.out(wm.linear_2, "tower.wm.linear_2")
    rec.inp(vt.encoder.layers[7], "tower.wm.output")
    rec.inp(vt.post_layernorm, "tower.post_layernorm.input", norm_input=True)
    rec.out(vt.post_layernorm, "tower.post_layernorm")
    mlp = tower.merger.mlp[0]
    rec.inp(mlp.pre_norm, "tower.merger.pre_norm_input", norm_input=True)
    rec.out(mlp.pre_norm, "tower.merger.pre_norm")
    rec.out(mlp.linear_1, "tower.merger.linear_1_pre_gelu")
    rec.out(mlp.act, "tower.merger.gelu")
    rec.out(mlp.linear_2, "tower.merger.output")


def text_hooks(rec: Recorder, vlm, torch) -> None:
    m = vlm.model
    rec.inp(m.layers[0], "text.input_embeds")
    for i, layer in enumerate(m.layers):
        p = f"text.L{i:02d}.{'full' if layer.is_full else 'gdn'}"
        rec.inp(layer.input_layernorm, f"{p}.residual_in", norm_input=True)
        rec.inp(layer.post_attention_layernorm, f"{p}.residual_mid", norm_input=True)
        rec.out(layer, f"{p}.residual_out")
        if layer.is_full:
            att = layer.self_attn
            rec.out(att.q_proj, f"{p}.q_proj_query_gate")
            rec.out(att.q_norm, f"{p}.q_normed")
            rec.out(att.k_norm, f"{p}.k_normed")
            rec.out(att.v_proj, f"{p}.v")

            def sdpa_scores(mod, a, p=p, att=att):
                q, k = a[0].detach(), a[1].detach()                       # [b, H, s, D], [b, HKV, seq, D]
                rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(rep, dim=1)
                s = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
                n_q, n_k = s.shape[-2], s.shape[-1]
                causal = torch.ones(n_q, n_k, dtype=torch.bool).tril(diagonal=n_k - n_q)
                rec.put(f"{p}.scores", s.masked_select(causal.expand_as(s)))
            rec.hooks.append(att.sdpa.register_forward_pre_hook(sdpa_scores))
            rec.inp(att.o_proj, f"{p}.gated_ctx")
            rec.out(att.o_proj, f"{p}.o_proj")
        else:
            g = layer.linear_attn
            rec.inp(g, f"{p}.gdn_input")
            rec.out(g.in_proj_qkv, f"{p}.in_proj_qkv")
            rec.out(g.in_proj_z, f"{p}.in_proj_z")
            rec.out(g.in_proj_a, f"{p}.in_proj_a")
            rec.inp(g.norm, f"{p}.recurrence_out", norm_input=True)
            rec.out(g.out_proj, f"{p}.out_proj")
        mlp = layer.mlp
        rec.out(mlp.gate_proj, f"{p}.mlp_gate_proj")
        rec.out(mlp.up_proj, f"{p}.mlp_up_proj")
        rec.inp(mlp.down_proj, f"{p}.mlp_gate_x_up")
        rec.out(mlp.down_proj, f"{p}.mlp_down_proj")
    rec.inp(m.norm, "text.final_norm.input", norm_input=True)
    rec.out(m.norm, "text.final_norm")
    rec.out(vlm.lm_head, "text.logits")


def table(absmax: dict) -> list:
    rows = [{"stage": s, "max": max(d.values()), "prompt": max(d, key=d.get), "per_prompt": d}
            for s, d in absmax.items()]
    return sorted(rows, key=lambda r: -r["max"])


def run(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from minicpmv47_vision import GRID, MiniCPMV47Vision, unpack_patches
    from minicpmv47_vl_pipelined import MiniCPMV47VLPipelinedForCausalLM, host_static_inputs

    from coreai_models.models.macos import qwen3_5 as q

    snap = hf_snapshot(HF_ID, revision=REVISION)
    z = np.load(Path(args.oracle_dir) / "oracle.npz")
    tower = MiniCPMV47Vision.from_checkpoint(snap, torch.float32).bake_constants(GRID)
    vlm = MiniCPMV47VLPipelinedForCausalLM.from_checkpoint(snap, torch.float32)
    cfg = vlm.config

    trec, xrec = Recorder(torch), Recorder(torch)
    tower_hooks(trec, tower, torch)
    text_hooks(xrec, vlm, torch)
    feats, states = {}, {}
    for P in PROMPTS:
        trec.prompt = xrec.prompt = P
        pv = unpack_patches(torch.from_numpy(z[f"{P}__pixel_values"]), GRID, GRID)
        with torch.inference_mode():
            feats[P] = tower(pv)
            ids = torch.from_numpy(z[f"{P}__input_ids"])
            mapped, start, amount = host_static_inputs(ids)
            S = int(ids.numel())
            st = q.build_decode_state(cfg, max_seq_len=S, dtype=torch.float32)
            vlm(mapped.reshape(1, -1).to(torch.int32), torch.arange(S, dtype=torch.int32).unsqueeze(0), feats[P],
                start, amount, st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
        for name in ("k_cache", "v_cache", "conv_state", "rec_state"):
            xrec.put(f"text.state.{name}", st[name])
        xrec.put("text.image_embeds", feats[P])
        print(f"[{P}] tower + decoder prompt (S={S}) recorded", flush=True)
    trec.clear()
    xrec.clear()

    cpu16 = {}
    if not args.skip_cpu16:
        variants = (("fp16", torch.float16, torch.float16), ("fp16w32", torch.float16, torch.float32))
        for name, wdt, cdt in variants:
            t = MiniCPMV47Vision.from_checkpoint(snap, wdt).bake_constants(GRID)
            if cdt != wdt:
                t = t.to(cdt)                                           # fp16-rounded weights, fp32 math
            rows = {}
            for P in PROMPTS:
                pv = unpack_patches(torch.from_numpy(z[f"{P}__pixel_values"]), GRID, GRID).to(cdt)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    f = t(pv).double()
                secs = time.perf_counter() - t0
                r = feats[P].double()
                cos = (f * r).sum(-1) / (f.norm(dim=-1) * r.norm(dim=-1))
                o = torch.from_numpy(z[f"{P}__image_features"]).double()
                cos_o = (f * o).sum(-1) / (f.norm(dim=-1) * o.norm(dim=-1))
                rows[P] = {"cos_min_vs_fp32_tower": float(cos.min()), "cos_mean_vs_fp32_tower": float(cos.mean()),
                           "rows_below_0.999": int((cos < 0.999).sum()), "max_abs_diff": float((f - r).abs().max()),
                           "finite": bool(torch.isfinite(f).all()), "cos_min_vs_oracle": float(cos_o.min()),
                           "seconds": secs}
                print(f"[cpu16] {name} {P}: cos min {rows[P]['cos_min_vs_fp32_tower']:.6f} mean "
                      f"{rows[P]['cos_mean_vs_fp32_tower']:.6f} rows<0.999 {rows[P]['rows_below_0.999']} max|d| "
                      f"{rows[P]['max_abs_diff']:.3e} finite {rows[P]['finite']} ({secs:.1f}s)", flush=True)
            cpu16[name] = rows
            del t
    return {"tower": trec.absmax, "text": xrec.absmax, "cpu16": cpu16}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-cpu16", action="store_true")
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "fp16_precheck.json")))
    args = ap.parse_args()
    t_all = time.perf_counter()
    res = run(args)
    for side in ("tower", "text"):        # every hooked stage fired on every prompt (no silent gap)
        short = {s: sorted(d) for s, d in res[side].items() if set(d) != set(PROMPTS)}
        assert not short, (side, short)
    n_stages = {side: sum(not s.endswith((".row_sumsq", ".row_meansq")) for s in res[side]) for side in ("tower", "text")}
    print(f"[stages] {n_stages}", flush=True)

    def is_act(stage):  # activations, not the squared norm statistics
        return not stage.endswith((".row_sumsq", ".row_meansq"))

    out = {}
    for side in ("tower", "text"):
        rows = table(res[side])
        acts = [r for r in rows if is_act(r["stage"])]
        sumsq = [r for r in rows if r["stage"].endswith(".row_sumsq")]
        out[side] = {
            "largest": {k: acts[0][k] for k in ("stage", "max", "prompt")},
            "above_30000": [{k: r[k] for k in ("stage", "max", "prompt")} for r in acts if r["max"] > RISK],
            "norm_row_sumsq_above_65504": [{k: r[k] for k in ("stage", "max", "prompt")} for r in sumsq
                                           if r["max"] > FP16_MAX],
            "table": rows,
        }
        print(f"[{side}] largest {acts[0]['stage']} {acts[0]['max']:.1f} ({acts[0]['prompt']}); > 30,000: "
              f"{len(out[side]['above_30000'])}; norm row sum(x^2) > 65,504: {len(out[side]['norm_row_sumsq_above_65504'])}",
              flush=True)
        for r in acts[:8]:
            print(f"    {r['stage']:40s} {r['max']:12.2f} ({r['prompt']})", flush=True)
    top = max((out[s]["largest"] for s in ("tower", "text")), key=lambda r: r["max"])
    risky = out["tower"]["above_30000"] + out["text"]["above_30000"]
    verdict = (f"largest activation {top['max']:.1f} at {top['stage']} ({top['prompt']}); "
               f"{'no stage' if not risky else str(len(risky)) + ' stage(s)'} above 30,000"
               + ("" if not risky else ": " + ", ".join(f"{r['stage']} {r['max']:.0f}" for r in risky)))
    record = {
        "schema": "minicpmv47-fp16-precheck/1",
        "purpose": "activation absmax per stage (fp32 CPU torch, P1..P4) before an fp16 export; tower = baked path "
                   "(b), text = round-1 VL decoder with that tower's rows",
        "verdict": verdict,
        "fp16_risk": bool(risky),
        "stages_recorded": n_stages,
        "threshold": RISK,
        "fp16_max": FP16_MAX,
        "tower": out["tower"],
        "text": out["text"],
        "cpu16_tower": {"note": "CPU torch (not the device): fp16 = fp16 weights + fp16 math; fp16w32 = fp16 weights "
                                "read in fp32 math; both vs the fp32 tower, P1..P4", **res["cpu16"]},
        "modules": {"tower": {"path": "conversion/minicpmv47/minicpmv47_vision.py",
                              "sha256": sha256(HERE / "minicpmv47_vision.py")},
                    "decoder": {"path": "conversion/minicpmv47/minicpmv47_vl_pipelined.py",
                                "sha256": sha256(HERE / "minicpmv47_vl_pipelined.py")}},
        "script": {"path": "conversion/minicpmv47/fp16_precheck.py", "sha256": sha256(Path(__file__))},
        "versions": {"python": sys.version.split()[0], "torch": __import__("torch").__version__,
                     "numpy": np.__version__, "platform": platform.platform(), "threads": args.threads},
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"[fp16_precheck] {verdict}", flush=True)
    print(f"[fp16_precheck] -> {args.out}")


if __name__ == "__main__":
    main()
