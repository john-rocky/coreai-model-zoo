"""Steps (2)-(4): metalize the MoE (sym8), export the R-SWA decode graph, gate the
engine logits vs the fp32 oracle via coreai.runtime.

THE decode graph is the constant-window gather: each decode step
gathers prefix[0,Lm) ++ tail[max(seq_len-W,0), +W) -> a CONSTANT [Lm+W] key set ->
constant-shape SDPA + MetalSwitchGLU MoE -> the whole graph is shape-static, so the
engine compiles it ONCE and decode latency is FLAT (no per-step Metal recompile, the
Qwen3-Coder-Next freeze that a growing-KV full-mask graph hits).

Prefill (writing the global prefix [0,Lm) into the KV cache) is run in TORCH here
(fp16, the non-metalized SwitchGLU handles the q_len=Lm block; K/V do not depend on
the MoE quant) to seed the cache; the gate then drives engine decode one token at a
time and compares logits to the oracle at the sampled steps.

Run (coreai venv):
    cd ~/code/coreai/coreai-models && PYTHONPATH=../_unlimited_ocr \
        .venv/bin/python ../_unlimited_ocr/export_decoder.py [--layers N] [--max-step S]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from model import (
    DECODE_STATE_NAMES,
    OcrPrefillForCausalLM,
    build_decode_state,
    ocr_decoder_from_safetensors,
)
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.moe_metal import metalize_moe, metalize_moe_batched

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"
REF = HERE / "out/_oracle/oracle_tensors.npz"
MAX_CTX = 4096       # dynamic seq upper bound
CACHE_LEN = 2048     # allocated KV cache length (>= TRACE_KV_CACHE_SEQ_LEN)
DTYPE = torch.float16


def export_unified(decode_model, prefill_model, kernel, cfg, prefix_len: int, out_dir: Path):
    """ONE bundle with two functions — 'prefill' (q_len=Lm) and 'decode' (q=1) — sharing
    a single copy of the sym8 decoder weights and the keyCache/valueCache state. Both
    stage into one TorchConverter (multi-entrypoint); the custom kernel is registered once."""
    import coreai_torch
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization

    st_p = build_decode_state(cfg, CACHE_LEN, DTYPE)
    st_d = build_decode_state(cfg, CACHE_LEN, DTYPE)
    pref_inputs = {"inputs_embeds": torch.zeros(1, prefix_len, cfg.hidden_size, dtype=DTYPE),
                   "k_cache": st_p["k_cache"], "v_cache": st_p["v_cache"]}
    dec_inputs = {"inputs_embeds": torch.zeros(1, 1, cfg.hidden_size, dtype=DTYPE),
                  "pos": torch.tensor([prefix_len], dtype=torch.int32),
                  "k_cache": st_d["k_cache"], "v_cache": st_d["v_cache"]}
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in {"gated_delta_update", "scaled_dot_product_attention", "gather_mm"}]

    def mk_export_fn(ref):
        def export_fn(m):
            with torch.no_grad():
                ep = torch.export.export(m, args=(), kwargs=ref, dynamic_shapes=None)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            return ep
        return export_fn

    print(f"exporting UNIFIED bundle (prefill q={prefix_len} + decode q=1, shared sym8) ...", flush=True)
    conv = coreai_torch.TorchConverter()
    conv.register_custom_kernels([kernel])
    conv.add_pytorch_module(
        prefill_model, export_fn=mk_export_fn(pref_inputs), externalize_modules=specs,
        input_names=("inputs_embeds",), output_names=("logits",),
        state_names=DECODE_STATE_NAMES, entrypoint_name="prefill")
    conv.add_pytorch_module(
        decode_model, export_fn=mk_export_fn(dec_inputs), externalize_modules=specs,
        input_names=("inputs_embeds", "pos"), output_names=("logits",),
        state_names=DECODE_STATE_NAMES, entrypoint_name="decode")
    register_custom_torch_lowering(conv)
    prog = conv.to_coreai()
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return aimodel


def export_prefill(model, cfg, prefix_len: int, out_dir: Path, kernel):
    """Export the batched prefill graph (q_len=Lm, static). With ``kernel`` set the MoE
    runs the sym8 BatchedMetalSwitchGLU (routed-only read, same weights as decode);
    with ``kernel`` None it falls back to the engine GatherMM (fp16, dense)."""
    pf = OcrPrefillForCausalLM(model, prefix_len).eval()
    inputs_embeds = torch.zeros(1, prefix_len, cfg.hidden_size, dtype=DTYPE)
    state = build_decode_state(cfg, max_seq_len=CACHE_LEN, dtype=DTYPE)
    reference_inputs = {"inputs_embeds": inputs_embeds,
                        "k_cache": state["k_cache"], "v_cache": state["v_cache"]}
    drop = {"gated_delta_update", "scaled_dot_product_attention"}
    if kernel is not None:
        drop.add("gather_mm")
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in drop]
    moe = "batched-metal sym8" if kernel is not None else "engine GatherMM"
    print(f"exporting batched prefill graph (q_len={prefix_len}, {moe}) ...", flush=True)
    if kernel is not None:
        prog = export_to_coreai_with_kernels(
            model=pf, reference_inputs=reference_inputs, custom_kernels=[kernel],
            dynamic_shapes=None, input_names=("inputs_embeds",), output_names=("logits",),
            state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    else:
        prog = export_to_coreai(
            model=pf, reference_inputs=reference_inputs, dynamic_shapes=None,
            input_names=("inputs_embeds",), output_names=("logits",),
            state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    prog.optimize()
    import coreai.runtime as rt
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return aimodel


def export(model, kernel, cfg, prefix_len: int, out_dir: Path, externalize_sdpa: bool = False):
    # FULLY-STATIC decode graph: inputs_embeds [1,1,hidden], pos [1] int32 (the abs
    # query position, a runtime VALUE). No dynamic shapes -> the engine compiles once
    # -> flat decode latency (a growing position_ids input instead recompiles the
    # Metal shader every step and faults on Metal 4 after the first shape).
    inputs_embeds = torch.zeros(1, 1, cfg.hidden_size, dtype=DTYPE)
    pos = torch.tensor([prefix_len], dtype=torch.int32)  # representative decode position
    state = build_decode_state(cfg, max_seq_len=CACHE_LEN, dtype=DTYPE)
    reference_inputs = {
        "inputs_embeds": inputs_embeds, "pos": pos,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
    }
    # SDPA NOT externalized (we feed a runtime R-SWA mask the engine-native SDPA op
    # can't take); RMSNorm stays externalized. With the metal kernel, gather_mm is
    # replaced by it; without (kernel is None), gather_mm stays the engine-native MoE.
    drop = {"gated_delta_update"}
    if not externalize_sdpa:
        drop.add("scaled_dot_product_attention")
    if kernel is not None:
        drop.add("gather_mm")
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in drop]
    print(f"exporting static R-SWA decode graph ({'metal kernel' if kernel else 'engine GatherMM'}) ...", flush=True)
    if kernel is not None:
        prog = export_to_coreai_with_kernels(
            model, reference_inputs=reference_inputs, custom_kernels=[kernel],
            dynamic_shapes=None, input_names=("inputs_embeds", "pos"),
            output_names=("logits",), state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    else:
        prog = export_to_coreai(
            model, reference_inputs, dynamic_shapes=None,
            input_names=("inputs_embeds", "pos"), output_names=("logits",),
            state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return aimodel


async def gate(model, prefill_aimodel, decode_aimodel, cfg, prefix_len, gaps_path, max_step_cap, unified=False):
    import asyncio
    import coreai.runtime as rt

    z = np.load(REF)
    dec = torch.from_numpy(z["dec_embeds"]).to(DTYPE)  # [1,Lm,1280] prefix embeddings
    ids = torch.from_numpy(z["token_ids"]).long()  # [1,T]
    Lm = prefix_len
    save_steps = sorted(int(k[len("logits_step"):]) for k in z.files if k.startswith("logits_step"))
    if max_step_cap is not None:
        save_steps = [s for s in save_steps if s <= max_step_cap]
    max_step = max(save_steps)
    Tneed = Lm + max_step + 1  # decode positions Lm .. Lm-1+max_step

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    state = {DECODE_STATE_NAMES[0]: rt.NDArray(np.ascontiguousarray(
                 build_decode_state(cfg, CACHE_LEN, DTYPE)["k_cache"].numpy())),
             DECODE_STATE_NAMES[1]: rt.NDArray(np.ascontiguousarray(
                 build_decode_state(cfg, CACHE_LEN, DTYPE)["v_cache"].numpy()))}

    # ---- load: one unified bundle (prefill/decode functions) or two single-fn bundles ----
    if unified:
        m = await rt.AIModel.load(str(prefill_aimodel), gpu)
        fnp, fn = m.load_function("prefill"), m.load_function("decode")
        mp = None
    else:
        mp = await rt.AIModel.load(str(prefill_aimodel), gpu)
        fnp = mp.load_function("main")

    # ---- ENGINE PREFILL: write prefix [0,Lm) into the cache, get step 0 ----
    print(f"[gate] engine prefill ({'unified' if unified else prefill_aimodel.name}) ...", flush=True)
    t0 = time.time()
    resp = await asyncio.wait_for(
        fnp(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(dec.numpy()))}, state=state),
        timeout=600)
    prefill_ms = (time.time() - t0) * 1000.0
    got = {0: resp["logits"].numpy().astype(np.float32).reshape(-1)}  # step 0 (engine)
    print(f"  prefill {prefill_ms:.0f} ms (q_len={Lm}); cache seeded", flush=True)

    # ---- ENGINE DECODE: steps 1.. via the static decode graph, reusing the cache ----
    if not unified:
        del fnp, mp
        m = await rt.AIModel.load(str(decode_aimodel), gpu)
        fn = m.load_function("main")
    print(f"[gate] engine decode ({'unified' if unified else decode_aimodel.name}) ...", flush=True)
    with torch.no_grad():
        gen_emb = model.model.embed_tokens(ids[:, Lm:Tneed]).to(DTYPE)  # [1,*,1280]

    gaps = []
    for p in range(Lm, Tneed):
        emb = gen_emb[:, p - Lm:p - Lm + 1, :]
        pos = torch.tensor([p], dtype=torch.int32)
        inputs = {"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb.numpy())),
                  "pos": rt.NDArray(np.ascontiguousarray(pos.numpy()))}
        t0 = time.time()
        res = await asyncio.wait_for(fn(inputs=inputs, state=state), timeout=600)
        gaps.append((time.time() - t0) * 1000.0)
        step = p - (Lm - 1)
        if step in save_steps:
            got[step] = res["logits"].numpy().astype(np.float32).reshape(-1)
        if (p - Lm) % 64 == 0:
            print(f"  pos {p:4d}/{Tneed - 1}  gap {gaps[-1]:.1f} ms", flush=True)

    allcos, allflip = [], 0
    print("\n[gate] engine prefill + engine decode vs fp32 oracle:", flush=True)
    for s in save_steps:
        ref = z[f"logits_step{s}"].astype(np.float32).reshape(-1)
        mine = got[s]
        cos = float(np.dot(mine, ref) / (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-12))
        flip = int(np.argmax(mine) != np.argmax(ref))
        allflip += flip
        allcos.append(cos)
        tag = " (prefill)" if s == 0 else ""
        print(f"  step{s:4d}: cos={cos:.6f}  argmax {'MATCH' if not flip else 'FLIP'} "
              f"(mine={int(np.argmax(mine))} ref={int(np.argmax(ref))}){tag}", flush=True)
    # Correctness gate = argmax (flip<=2); cos is a quality metric. Full sym8 (prefix+decode
    # both sym8) sits at the sym8 floor: a couple early-decode steps dip just under 0.999
    # (the sym8 prefix), but 0 flips = identical output. cos>=0.998 floor catches real breakage.
    n_below = sum(c < 0.999 for c in allcos)
    ok = allflip <= 2 and min(allcos) >= 0.998
    print(f"\nVERDICT: cos min={min(allcos):.6f} mean={np.mean(allcos):.6f}  flips={allflip}/{len(save_steps)}"
          f"  ({n_below} step(s) cos<0.999, sym8 floor)")
    print("✅ PASS" if ok else "❌ FAIL")

    if gaps_path and gaps:
        Path(gaps_path).write_text("\n".join(f"{g:.3f}" for g in gaps))
        steady = gaps[cfg.sliding_window:] if len(gaps) > cfg.sliding_window else gaps
        print(f"[latency] decode gap (steady, n={len(steady)}): min {min(steady):.1f} / "
              f"med {np.median(steady):.1f} / max {max(steady):.1f} ms -> {gaps_path}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", type=int, default=None, help="prefix length Lm (default: oracle's)")
    ap.add_argument("--layers", type=int, default=None, help="truncate to N layers (fast debug)")
    ap.add_argument("--max-step", type=int, default=None, help="cap gate decode steps (fast debug)")
    ap.add_argument("--externalize-sdpa", action="store_true")
    ap.add_argument("--no-metal", action="store_true", help="engine-native GatherMM decode MoE (no custom kernel)")
    ap.add_argument("--unified", action="store_true", help="one bundle w/ prefill+decode functions (shared sym8 weights)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(REF)
    prefix_len = args.lm if args.lm is not None else int(z["dec_embeds"].shape[1])
    out_dir = Path(args.out) if args.out else (HERE / "out/_dec_static")
    prefill_dir = out_dir.parent / f"{out_dir.name}_prefill"

    t0 = time.time()
    print(f"loading Unlimited-OCR decoder fp16 (Lm={prefix_len}) ...", flush=True)
    model = ocr_decoder_from_safetensors(str(CKPT), target_dtype=DTYPE)
    model.prefix_len = prefix_len
    model.last_token_only = True
    cfg = model.config
    if args.layers is not None:
        model.model.layers = model.model.layers[: args.layers]
        cfg.num_hidden_layers = args.layers
        print(f"  [debug] truncated to {args.layers} layers", flush=True)
    print(f"  {cfg.num_hidden_layers}L E={cfg.n_routed_experts}/top{cfg.num_experts_per_tok} "
          f"moe_inter={cfg.moe_intermediate_size} vocab={cfg.vocab_size}", flush=True)

    # Metalize ONCE with the batched variant: BatchedMetalSwitchGLU serves BOTH prefill
    # (q_len=Lm, sort->routed-only read) AND decode (q=1, falls back to the proven q=1
    # MetalSwitchGLU path) from the SAME sym8 weights -> both bundles sym8, ~3.2GB each.
    if args.no_metal:
        print("MoE: engine-native GatherMM fp16 (no custom kernel) ...", flush=True)
        kernel = None
    else:
        print("metalizing routed MoE -> gather_qmm sym8 BATCHED (prefill+decode share) ...", flush=True)
        kernel = metalize_moe_batched(model, scheme="sym8")

    import asyncio
    if args.unified and kernel is not None:
        prefill_model = OcrPrefillForCausalLM(model, prefix_len).eval()
        unified_aimodel = export_unified(model, prefill_model, kernel, cfg, prefix_len, out_dir)
        print(f"[export] done in {time.time() - t0:.1f}s", flush=True)
        gaps = str(out_dir / "token_gaps.txt")
        ok = asyncio.run(gate(model, unified_aimodel, unified_aimodel, cfg, prefix_len, gaps,
                              args.max_step, unified=True))
    else:
        # two single-function bundles (prefill q_len=Lm, decode q=1)
        prefill_aimodel = export_prefill(model, cfg, prefix_len, prefill_dir, kernel)
        decode_aimodel = export(model, kernel, cfg, prefix_len, out_dir, externalize_sdpa=args.externalize_sdpa)
        print(f"[export] done in {time.time() - t0:.1f}s", flush=True)
        gaps = str(out_dir / "token_gaps.txt")
        ok = asyncio.run(gate(model, prefill_aimodel, decode_aimodel, cfg, prefix_len, gaps, args.max_step))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
