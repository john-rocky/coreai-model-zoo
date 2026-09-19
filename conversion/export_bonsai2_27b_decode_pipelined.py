"""Export Bonsai 2 27B (ternary g128 + Hadamard, PQ2_0 GGUF) as a decode-only Core AI bundle.

The zoo's Qwen3.5 hybrid decoder with every folded linear on the PQ2_0 ternary matvec
(`bonsai_ternary_metal`), one shared Hadamard transform per activation site
(`bonsai_hadamard_metal.HadamardSite`), a packed embedding with the inverse transform fused
into the gather, and the untied ternary head. Weights are the shipped ternary words — nothing
is re-quantised. Static `input_ids` [1,1] (custom kernels are M=1), dynamic position/KV, the
SSM conv/rec states as fixed-shape extra states — the same contract as the qwen3.5 decode
bundles, so it rides the pipelined engine with `COREAI_CHUNK_THRESHOLD=1`.

  # smoke (2 layers): export + load on the GPU + one decode step
  .venv/bin/python conversion/export_bonsai2_27b_decode_pipelined.py --num-layers 2 --run-check
  # full export
  .venv/bin/python conversion/export_bonsai2_27b_decode_pipelined.py --run-check --prompt "..."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from _bundle import write_bundle_metadata

import coreai_torch
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from coreai_models.models.macos.bonsai2 import HF_ID, load_bonsai2_from_gguf, set_gdn_mode
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.qwen3_5 import DECODE_STATE_NAMES, build_decode_state
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
GGUF_REPO, GGUF_FILE = "prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf"


def gguf_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    from huggingface_hub import hf_hub_download
    return hf_hub_download(GGUF_REPO, GGUF_FILE)


_ORIG_FORWARD = {}


def set_next_token_output(model, enabled: bool) -> None:
    """Trace-time switch: with `enabled`, forward returns (logits, argmax int32 [1,S]) instead of
    logits. torch.export binds the *class* forward, so the patch goes on the class and is restored
    before the next entrypoint traces; the module (and its shared weights) is the same throughout.
    `functools.wraps` keeps the signature export binds kwargs against."""
    import functools
    cls = type(model)
    orig = _ORIG_FORWARD.setdefault(cls, cls.forward)
    if not enabled:
        cls.forward = orig
        return

    # explicit signature (not functools.wraps: torch.export follows __wrapped__ back to the original)
    def forward(self, input_ids, position_ids, k_cache, v_cache, conv_state, rec_state):
        logits = orig(self, input_ids, position_ids, k_cache, v_cache, conv_state, rec_state)
        return logits, logits.argmax(dim=-1).to(torch.int32)
    cls.forward = forward


def decode_spec(cfg, max_ctx: int, query: int = 1):
    """Static query length `query` (1 = decode, C = prefill chunk), dynamic position/KV."""
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, query), dtype=torch.int32)
    position_ids = torch.arange(max(trace_past + 1, query + 1), dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {"input_ids": input_ids, "position_ids": position_ids, **state}
    dynamic_shapes = {
        "input_ids": None,
        # min = the query length, NOT max(2, query): iOS MPSGraph asserts `Failed to resolve dynamic
        # dimensions for memref.alloc` when the S=1 entrypoint is driven at position 0 (length 1 <
        # min 2). macOS tolerates it; the device does not (see ternary-chunked-prefill.md, and the
        # same fix in export_bitcpm8b_chunked_prefill.py).
        "position_ids": {1: torch.export.Dim("seq_pos", min=query, max=max_ctx - 1)},
        "k_cache": {KVCache.seq_len_dim(): torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "v_cache": {KVCache.seq_len_dim(): torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "conv_state": None,
        "rec_state": None,
    }
    return reference_inputs, dynamic_shapes


def export_multifunction_with_kernels(model, entries, custom_kernels, specs):
    """`export_to_coreai_multifunction` + the custom-kernel hook (mirrors
    export_bitcpm8b_chunked_prefill.py); `entries` = [(name, spec, prepare_fn)] where prepare_fn
    flips the model's trace-time switches (GDN step vs chunk) before that entrypoint is traced."""
    model.eval()
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels(custom_kernels)
    for entrypoint_name, spec, prepare, output_names in entries:
        # The converter traces lazily, at to_coreai(): the trace-time switches must be flipped
        # inside the closure, right before this entrypoint's trace, not here in the loop.
        def export_fn(module, _inputs=spec[0], _dyn=spec[1], _prepare=prepare, _name=entrypoint_name):
            _prepare(module)
            with torch.no_grad():
                ep = torch.export.export(module, args=(), kwargs=_inputs, dynamic_shapes=_dyn)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            print(f"  [{_name}] traced: {len(ep.graph_signature.user_outputs)} outputs", flush=True)
            return ep

        converter.add_pytorch_module(
            model, export_fn=export_fn, externalize_modules=list(specs) or None,
            input_names=("input_ids", "position_ids"), output_names=output_names,
            state_names=DECODE_STATE_NAMES, entrypoint_name=entrypoint_name)
    register_custom_torch_lowering(converter)
    return converter.to_coreai()


async def s1_walk(fn, rt, cfg, ids: list[int]):
    """Logits at every prompt position from a fresh-state S=1 walk (the decode-only contract)."""
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    st = {"keyCache": rt.NDArray(state["k_cache"].numpy()), "valueCache": rt.NDArray(state["v_cache"].numpy()),
          "convState": rt.NDArray(state["conv_state"].numpy()), "recState": rt.NDArray(state["rec_state"].numpy())}
    out = []
    for step, t in enumerate(ids):
        feed = {"input_ids": rt.NDArray(np.array([[t]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(step + 1, dtype=np.int32)[None])}
        res = await fn(inputs=feed, state=st)
        out.append(res["logits"].numpy().astype(np.float32).reshape(-1))
    return out


async def run_check(aimodel: Path, cfg, tokenizer_dir: Path | None, prompt: str, new: int,
                    torch_model=None, ref: dict | None = None, chunk: int = 0,
                    self_check: bool = False):
    """Load on the GPU and decode greedily, S=1 per step (the engine's chunkThreshold=1 walk).

    With `torch_model` (the same truncated model in torch), every step's logits are compared
    against the torch reference path driven with identical states: argmax agreement and
    max|delta| per step — the numeric gate for the composed graph before a full export.
    """
    import coreai.runtime as rt
    t0 = time.perf_counter()
    if aimodel.suffix == ".aimodelc":      # AOT-compiled: the compile fixed the compute placement
        opts = rt.SpecializationOptions.default()
    else:
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = await rt.AIModel.load(str(aimodel), opts)
    fn = m.load_function("main")
    pf = m.load_function("prefill") if chunk and "prefill" in list(m.function_names) else None
    print(f"[check] loaded in {time.perf_counter() - t0:.1f}s; functions {list(m.function_names)}; {fn.desc}", flush=True)

    tok = None
    if tokenizer_dir is not None and (tokenizer_dir / "tokenizer.json").exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if ref is not None:                      # gate: same ids as the reference run
        ids = list(ref["ids"])
        new = len(ref["steps"]) + 1 - len(ids)
    elif tok is not None:
        ids = list(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=True))
    else:
        ids = [1, 2, 3]
    ref_steps = {st["step"]: st for st in ref["steps"]} if ref else {}
    ref_agree = 0
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    st = {
        "keyCache": rt.NDArray(state["k_cache"].numpy()),
        "valueCache": rt.NDArray(state["v_cache"].numpy()),
        "convState": rt.NDArray(state["conv_state"].numpy()),
        "recState": rt.NDArray(state["rec_state"].numpy()),
    }
    out, times = [], []
    seq = list(ids)
    if torch_model is not None:
        tstate = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
        agree, worst = 0, 0.0
    start = 0
    walk = None
    if pf is not None and self_check:        # the same bundle's S=1 walk is the yardstick
        walk = await s1_walk(fn, rt, cfg, ids)
        sc_agree, sc_n, sc_worst = 0, 0, 0.0
    if pf is not None:                       # prefill whole chunks first, then fall through to S=1
        n_chunks = (len(ids) - 1) // chunk   # keep the last prompt token for the S=1 loop below
        pf_agree, pf_n, pf_t = 0, 0, 0.0
        for c in range(n_chunks):
            lo = c * chunk
            feed = {"input_ids": rt.NDArray(np.array([ids[lo:lo + chunk]], dtype=np.int32)),
                    "position_ids": rt.NDArray(np.arange(lo + chunk, dtype=np.int32)[None])}
            t1 = time.perf_counter()
            res = await pf(inputs=feed, state=st)
            pf_t += time.perf_counter() - t1
            lg = res["logits"].numpy().astype(np.float32).reshape(chunk, -1)
            for j in range(chunk):
                stp = lo + j
                if stp in ref_steps:
                    pf_n += 1
                    pf_agree += int(lg[j].argmax()) == ref_steps[stp]["argmax"]
                if walk is not None:
                    sc_n += 1
                    sc_agree += int(lg[j].argmax()) == int(walk[stp].argmax())
                    sc_worst = max(sc_worst, float(np.abs(lg[j] - walk[stp]).max()))
        start = n_chunks * chunk
        print(f"[check] prefill {start} tok in {n_chunks} chunks of {chunk}: {start / max(pf_t, 1e-9):.1f} tok/s"
              + (f"; argmax vs ref {pf_agree}/{pf_n}" if ref else ""), flush=True)
    for step in range(start, len(ids) + new - 1):
        t = seq[step]
        feed = {"input_ids": rt.NDArray(np.array([[t]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(step + 1, dtype=np.int32)[None])}
        t1 = time.perf_counter()
        res = await fn(inputs=feed, state=st)
        times.append(time.perf_counter() - t1)
        logits = res["logits"].numpy().astype(np.float32).reshape(-1)
        if walk is not None and step < len(walk):   # S=1 steps after the chunk(s): chunk-primed state vs walk
            sc_n += 1
            sc_agree += int(logits.argmax()) == int(walk[step].argmax())
            sc_worst = max(sc_worst, float(np.abs(logits - walk[step]).max()))
        if step in ref_steps:
            r = ref_steps[step]
            same = int(logits.argmax()) == r["argmax"]
            ref_agree += same
            top = np.argsort(logits)[::-1][:2]
            print(f"  [ref] step {step:2d}: argmax {'==' if same else '!='} (rt {int(top[0])} / mlx {r['argmax']}) "
                  f"rt margin {float(logits[top[0]] - logits[top[1]]):.3f} mlx margin {r['margin']:.3f}  "
                  f"top1 logit rt {float(logits[top[0]]):.3f} mlx {r['top5_logits'][0]:.3f}", flush=True)
            if not same:                      # teacher-force the reference token so the walk stays comparable
                pass
        if torch_model is not None:
            with torch.no_grad():
                ref = torch_model(torch.tensor([[t]], dtype=torch.int32),
                                  torch.arange(step + 1, dtype=torch.int32)[None],
                                  tstate["k_cache"], tstate["v_cache"], tstate["conv_state"],
                                  tstate["rec_state"]).float().reshape(-1).numpy()
            d = float(np.abs(logits - ref).max())
            same = int(logits.argmax()) == int(ref.argmax())
            agree += same
            worst = max(worst, d)
            top = np.argsort(ref)[::-1][:2]
            margin = float(ref[top[0]] - ref[top[1]])
            print(f"  [parity] step {step:2d} tok {t:6d}: argmax {'==' if same else '!='} "
                  f"(rt {int(logits.argmax())} / torch {int(ref.argmax())}, margin {margin:.3f})  "
                  f"max|delta| {d:.4f}  |logits| {np.abs(ref).max():.2f}", flush=True)
        if step >= len(ids) - 1:
            nxt = int(logits.argmax())
            out.append(nxt)
            # under a reference, feed the reference's own continuation (teacher forcing) so every
            # step compares the same context even after a divergence
            seq.append(ref_steps[step + 1]["token"] if (step + 1) in ref_steps else nxt)
            print(f"  -> {nxt} {tok.decode([nxt])!r}" if tok else f"  -> {nxt}", flush=True)
    n_walk = len(ids) - start
    dec = times[n_walk:] if len(times) > n_walk else times
    if n_walk:
        print(f"[check] S=1 walk {n_walk} tok @ {n_walk / max(sum(times[:n_walk]), 1e-9):.1f} tok/s, ", end="")
    print(f"decode {len(dec)} tok @ {len(dec) / max(sum(dec), 1e-9):.1f} tok/s", flush=True)
    if tok:
        print("[check] GENERATION:", repr(tok.decode(out)), flush=True)
    if torch_model is not None:
        n = len(ids) + new - 1
        print(f"[parity] argmax agreement {agree}/{n}, worst max|delta| {worst:.4f}", flush=True)
    if walk is not None:
        print(f"[self] prefill-vs-S=1-walk over {sc_n} prompt positions: argmax {sc_agree}/{sc_n}, "
              f"worst max|delta| {sc_worst:.4f}", flush=True)
    if ref is not None:
        n = len(ref_steps)
        if pf is not None:
            ref_agree += pf_agree            # chunk positions were scored in the prefill loop
        print(f"[ref] argmax agreement vs MLX reference: {ref_agree}/{n} "
              f"(generated {out} vs mlx {ref['generated']})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=None, help="path to the PQ2_0 GGUF (default: HF cache)")
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer build")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--run-check", action="store_true", help="load the bundle on the GPU and decode")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--check-only", default=None, metavar="ASSET",
                    help="skip export; run the decode check on this .aimodel/.aimodelc (needs --num-layers to match)")
    ap.add_argument("--torch-parity", action="store_true",
                    help="with --check-only: also run the torch reference per step and compare logits")
    ap.add_argument("--tokenizer", default=None, help="bundle tokenizer dir for --check-only")
    ap.add_argument("--ref-json", default=None, help="mlx_reference.py output to gate against (teacher-forced)")
    ap.add_argument("--self-check", action="store_true",
                    help="with --check-only --chunk: gate prefill against the same bundle's S=1 walk")
    ap.add_argument("--chunk", type=int, default=0,
                    help="add a `prefill` entrypoint at static S=chunk (tiled GEMM + fp32 GDN chunk scan); 0 = decode-only")
    ap.add_argument("--chunks", default=None,
                    help="comma list of further static prefill lengths, each its own entrypoint `prefill<S>` (e.g. 16)")
    ap.add_argument("--no-next-token", action="store_true",
                    help="omit the `next_token` argmax output on `main` (the host-side-argmax contract)")
    ap.add_argument("--head-rows", type=int, default=None, help="debug: truncate lm_head to N rows (timing probes)")
    ap.add_argument("--gdn-main", choices=("fused", "chunk", "step"), default="fused",
                    help="GDN path traced into `main` (S=1): fused step kernel (round 3), chunk-scan kernel, or torch step")
    args = ap.parse_args()

    path = gguf_path(args.gguf)
    if args.check_only:
        import gguf
        from coreai_models.models.macos.bonsai2 import config_from_gguf
        cfg = config_from_gguf(gguf.GGUFReader(path), args.num_layers)
        asset = Path(args.check_only)
        tok_dir = Path(args.tokenizer) if args.tokenizer else asset.parent / "tokenizer"
        tm = None
        if args.torch_parity:
            tm, _ = load_bonsai2_from_gguf(path, num_layers=args.num_layers, dtype=DTYPE)
        ref = json.loads(Path(args.ref_json).read_text()) if args.ref_json else None
        asyncio.run(run_check(asset, cfg, tok_dir if tok_dir.exists() else None, args.prompt, args.new, tm, ref,
                              chunk=args.chunk, self_check=args.self_check))
        return 0
    t0 = time.perf_counter()
    print(f"loading {path} (layers={args.num_layers or 'all'}) ...", flush=True)
    extra_chunks = sorted({int(c) for c in args.chunks.split(",") if c.strip()}, reverse=True) if (args.chunks and args.chunk) else []
    if any(c >= args.chunk for c in extra_chunks):
        raise SystemExit(f"--chunks {extra_chunks} must all be smaller than --chunk {args.chunk}")
    model, kern = load_bonsai2_from_gguf(path, num_layers=args.num_layers, dtype=DTYPE,
                                         chunk=([args.chunk, *extra_chunks] if args.chunk else None),
                                         head_rows=args.head_rows)
    cfg = model.config
    n_buf = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9
    print(f"loaded in {time.perf_counter() - t0:.0f}s: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"full={cfg.num_full_layers} linear={cfg.num_linear_layers} vocab={cfg.vocab_size} "
          f"buffers={n_buf:.2f} GB", flush=True)

    name = "bonsai2_27b_decode_pq2_0" + (f"_pf{args.chunk}" if args.chunk else "") + \
           (f"_l{args.num_layers}" if args.num_layers else "")
    functions = ("main", "prefill", *(f"prefill{c}" for c in extra_chunks)) if args.chunk else ("main",)
    t0 = time.perf_counter()
    if args.chunk:
        # static S>1 trips the SDPA composite's auto-Dim (min=2) — decompose attention in-graph for
        # both entrypoints so the shared model carries one externalization (verify export precedent)
        specs = [s for s in _EXTERNALIZE_SPECS
                 if s.composite_op_name not in ("gated_delta_update", "scaled_dot_product_attention")]
        # `main` also emits its greedy choice as an int32 [1,1] output: a host that chains it
        # straight into the next step's input_ids never waits for a logits readback, and can
        # encode step N+1 while N runs (the round-2 host lever). Trace-time switch, like GDN mode.
        main_outputs = ("logits",) if args.no_next_token else ("logits", "next_token")

        def prepare_main(m):
            # "fused" (round 3): the whole GDN block between the projections in one kernel.
            # "chunk" is the round-0..2 path (the fp32 chunk-scan kernel at S=1, the proven one);
            # the loop-free torch step ("step") is untested on device.
            set_gdn_mode(m, args.gdn_main)
            set_next_token_output(m, not args.no_next_token)

        def prepare_chunk(m):
            set_gdn_mode(m, "chunk")
            set_next_token_output(m, False)

        entries = [("main", decode_spec(cfg, args.max_ctx, 1), prepare_main, main_outputs),
                   ("prefill", decode_spec(cfg, args.max_ctx, args.chunk), prepare_chunk, ("logits",))]
        for c in extra_chunks:
            entries.append((f"prefill{c}", decode_spec(cfg, args.max_ctx, c), prepare_chunk, ("logits",)))
        print(f"exporting multifunction (main S=1 + prefill S={args.chunk}"
              + "".join(f" + prefill{c} S={c}" for c in extra_chunks)
              + f") with {len(kern.all())} kernels ...", flush=True)
        prog = export_multifunction_with_kernels(model, entries, kern.all(), specs)
    else:
        specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
        reference_inputs, dynamic_shapes = decode_spec(cfg, args.max_ctx)
        print("exporting decode graph to Core AI (4 custom kernels) ...", flush=True)
        prog = export_to_coreai_with_kernels(
            model, reference_inputs, custom_kernels=kern.all(), dynamic_shapes=dynamic_shapes,
            input_names=("input_ids", "position_ids"), output_names=("logits",),
            state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    print(f"exported in {time.perf_counter() - t0:.0f}s; optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, name, HF_ID, cfg.vocab_size, args.max_ctx, functions=functions,
                          weights=f"{GGUF_REPO}/{GGUF_FILE}", mode="ternary-g128-pq2_0-hadamard1024",
                          language_extra={"extra_states": ["convState", "recState"],
                                          **({"prefill_chunk": args.chunk,
                                              "prefill_chunks": [args.chunk, *extra_chunks]} if args.chunk else {}),
                                          **({} if args.no_next_token else {"next_token_output": "next_token"})})
    tok_dir = out_dir / "tokenizer"
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(HF_ID).save_pretrained(tok_dir)
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer not saved ({type(e).__name__}: {e}); bundle is still runnable", flush=True)
        tok_dir = None
    print(f"bundle ready: {out_dir}", flush=True)

    if args.run_check:
        asyncio.run(run_check(aimodel, cfg, tok_dir, args.prompt, args.new, chunk=args.chunk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
