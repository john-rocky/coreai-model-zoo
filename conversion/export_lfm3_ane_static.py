#!/usr/bin/env python3
"""LFM3 (conv + full + sliding attention hybrid) on Apple's STATIC iOS path -> Neural Engine: the static builder
`coreai_models.models.ios.lfm3` (CM, community file) traced and converted by Apple's own iOS export functions
(`coreai_models.export.ios._export_ios_programs` / `_convert_to_coreai`: the four entrypoints, the static
(cache_len, query_len) ladder 256 .. max ctx x {8, 16, 64}, the IOSurface constraints on the KV rows and the
embedding table), palettized by Apple's coreai-opt k-means call (`export.compression.palettize_pytorch_model`),
then AOT for the Neural Engine with the ANE regions counted by unique name.

    python conversion/export_lfm3_ane_static.py --src <LFM3 checkpoint dir> --out-dir <dir> [--name NAME] \\
        [--qconfig <kmeans_palettization_config yaml> | --no-palettize] [--max-context-length 4096] \\
        [--act tanh|stock] [--history block|column] [--num-layers N] [--aot h16c,h18p,h19p] \\
        [--tokenizer-src <dir>] [--chat-template <file>]
    python conversion/export_lfm3_ane_static.py --tiny --out-dir <dir> --aot h16c     # random 5-layer probe

Why not `coreai.llm.export --platform iOS` with a runtime `register()` (the LFM2.5 path,
export_lfm25_ane_static.py): on this checkout (coreai-models zoo-0.2) the stock iOS path sizes the cache rows from
n_kv x head_dim (`_build_ios_reference_inputs`) and transformers 4.57 has no lfm3 config. The builder's rows are
hidden/2 wide (a conv layer's gated-input column lives in its key + value rows); `_convert_to_coreai` takes the row
width as an argument, so the bundle is Apple's contract for Apple's runner (StaticShapeEngine, unmodified): two
states key_cache / value_cache, functions extend_<ctx>_<q> / prompt_opt_<ctx>_<q> / gather_embeddings_<q> /
load_embeddings.

The default palettization (no --qconfig) = 8-bit k-means per-grouped-channel g32 on every Conv2d (LFM2.5's and
MiniCPM5-1B's ANE recipe; 4-bit g8 does not compile for the ANE), the embedding table int8 per tensor (Apple's
iOS table, also the tied head), the attention output gate (16 rows < one group of 32) left fp16, the depthwise conv
taps and the norms are bare parameters (never palettized).

AOT (`--aot`): h16c (Mac) = `--platform macOS`, h18p / h19p (iPhone 17 / 18 Pro) = `--platform iOS`, all
`--preferred-compute neural-engine` (Xcode 27.0.0 RC's coreai-build) = the placement check (ANE regions by unique
name). Nothing here loads a Neural Engine compile on the Mac: an iOS-platform static bundle run on the Mac hung the
machine twice (watchdog panic, 2026-06-24) — the static ANE bundle is checked on the iPhone. `--aot-gpu h16c` compiles
the same `.aimodel` with `--preferred-compute gpu` (it must keep 0 ANE regions) = the Mac check of the static graph's
numbers (palettized weights, int8 table, the builder's reads) on the GPU, not the Neural Engine's.
`--check-form`: if the iOS form cannot run there, the Mac check form of WeVisOne r7 (its worktree's
`conversion/wevisone/ane/export_vl.py`): the same palettized weights in the same process, only
`extend_<ctx>_8` / `extend_<ctx>_64` at one context (`--check-ctx`, default 2048 = every line-1 sequence fits), no
hardware constraints.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

AOT_XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"
ARCH_PLATFORM = {"h16c": "macOS", "h18p": "iOS", "h19p": "iOS"}
HF_ID = "LiquidAI/LFM3-1.7B"
LICENSE = "LFM Open License v1.0"

PAL8_G32 = {
    "global_config": {"op_state_spec": {"weight": {
        "n_bits": 8, "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 32}}}},
    "module_type_configs": {
        "torch.nn.modules.sparse.Embedding": None,
        "coreai_models.primitives.ios.embedding.LoadEmbeddings": None,
    },
    "module_name_configs": {r"extend\.model\.layers\.\d+\.self_attn\.gate_proj": None},
}

TINY_CONFIG = {
    "hidden_size": 256, "num_hidden_layers": 5, "vocab_size": 1024, "intermediate_size": 512,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128, "norm_eps": 1e-5, "conv_L_cache": 7,
    "conv_bias": False, "sliding_window": 256, "use_sandwich_norm": True, "tie_word_embeddings": True,
    "max_position_embeddings": 4096, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention"],
    "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0,
                                           "partial_rotary_factor": 0.5},
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000}},
    "model_type": "lfm3",
}


def write_tiny_checkpoint(out: Path, seed: int = 0) -> Path:
    """A random LFM3-shaped checkpoint (HF names, bf16 like the real one) for compile probes."""
    from safetensors.torch import save_file

    c = TINY_CONFIG
    g = torch.Generator().manual_seed(seed)
    d, f, v, n = c["hidden_size"], c["intermediate_size"], c["vocab_size"], c["num_hidden_layers"]
    hd, nh, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def ones(dim):
        return (1.0 + 0.1 * torch.randn(dim, generator=g)).to(torch.bfloat16)

    sd = {"model.embed_tokens.weight": rnd(v, d, scale=0.05), "model.embedding_norm.weight": ones(d)}
    for i, t in enumerate(c["layer_types"]):
        p = f"model.layers.{i}."
        for norm in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
            sd[p + norm + ".weight"] = ones(d)
        sd[p + "feed_forward.w1.weight"] = rnd(f, d)
        sd[p + "feed_forward.w3.weight"] = rnd(f, d)
        sd[p + "feed_forward.w2.weight"] = rnd(d, f)
        if t == "conv":
            sd[p + "conv.in_proj.weight"] = rnd(3 * d, d)
            sd[p + "conv.out_proj.weight"] = rnd(d, d)
            sd[p + "conv.conv.weight"] = rnd(d, 1, c["conv_L_cache"], scale=0.3)
        else:
            sd[p + "self_attn.q_proj.weight"] = rnd(nh * hd, d)
            sd[p + "self_attn.k_proj.weight"] = rnd(nkv * hd, d)
            sd[p + "self_attn.v_proj.weight"] = rnd(nkv * hd, d)
            sd[p + "self_attn.o_proj.weight"] = rnd(d, nh * hd)
            sd[p + "self_attn.gate_proj.weight"] = rnd(nh, d, scale=0.1)
            sd[p + "self_attn.q_norm.weight"] = ones(hd)
            sd[p + "self_attn.k_norm.weight"] = ones(hd)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(c, indent=2))
    return out


def load_qconfig(path: str | None):
    from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig

    if path is None:
        return KMeansPalettizerConfig.from_dict({"kmeans_palettization_config": PAL8_G32})
    import yaml

    data = yaml.safe_load(open(path))
    if list(data) != ["kmeans_palettization_config"]:
        raise SystemExit(f"{path}: expected one top-level key kmeans_palettization_config, got {sorted(data)}")
    return KMeansPalettizerConfig.from_dict(data)


class _CheckExtend(torch.nn.Module):
    """The decode graph with the int8 embedding table (the tied head) as a constant instead of an input: the
    Python runtime cannot hand the iOS form's IOSurface inputs over (`embedding_table has an invalid interleave:
    [1, 1, 1], expected: [8, 1, 1]`, and the KV rows' interleave 8), so the check form takes neither the table
    nor a gather function — the host gathers the rows (int8 x fp16 scale in fp16, the iOS gather's arithmetic),
    as WeVisOne r7's Mac form did. Same extend module, same weights."""

    def __init__(self, extend, table):
        super().__init__()
        self.extend = extend
        self.register_buffer("table", table, persistent=False)

    def forward(self, transformer_input, position_ids, in_step, causal_mask, key_cache, value_cache):
        return self.extend(transformer_input, position_ids, in_step, causal_mask, key_cache, value_cache, self.table)


async def convert_check_form(model, inputs, *, cache_len: int, hidden_size: int, kv_cached_embed_size: int,
                             num_layers: int, query_lengths=(8, 64)):
    """Apple's `_export_ios_programs` + `_convert_to_coreai` (export/ios.py) for the Mac check form: the decode
    entrypoint only (no prompt_opt, no gather / load), the table inside the graph, one context, no IOSurface /
    interleave constraints."""
    from coreai_torch import TorchConverter

    from coreai_models.export import ios as X
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization

    model.set_prefill_mode(False)
    check = _CheckExtend(model.extend, model.load_embeddings.embedding_table.data)
    kwargs = {k: v for k, v in inputs["forward_inputs"].items() if k != X.EMBEDDING_TABLE_INPUT_NAME}
    dyn = {k: v for k, v in inputs["forward_dynamic_shapes"].items() if k != X.EMBEDDING_TABLE_INPUT_NAME}
    decomp = torch.export.default_decompositions()
    decomp.pop(torch.ops.aten.silu.default)
    decomp.pop(torch.ops.aten.silu.out)
    with torch.no_grad():
        program = torch.export.export(check, args=(), kwargs=kwargs, dynamic_shapes=dyn).run_decompositions(decomp)
    remove_functionalization(program)

    converter = TorchConverter()
    register_custom_torch_lowering(converter)
    converter.add_exported_program(
        program,
        input_names=[X.TRANSFORMER_INPUT_NAME, X.POSITION_IDS_INPUT_NAME, X.IN_STEP_INPUT_NAME,
                     X.CAUSAL_MASK_INPUT_NAME],
        state_names=[X.KEY_CACHE_INPUT_NAME, X.VALUE_CACHE_INPUT_NAME], output_names=[X.OUTPUT_LOGITS_NAME],
        entrypoint_name=X.EXTEND_FUNCTION_NAME)
    prog = converter.to_coreai()
    prog.set_static_shape_config(X.EXTEND_FUNCTION_NAME, {
        f'"{cache_len}_{q}"': {
            X.TRANSFORMER_INPUT_NAME: (1, q, 1, hidden_size),
            X.POSITION_IDS_INPUT_NAME: (1, q),
            X.CAUSAL_MASK_INPUT_NAME: (1, cache_len, 1, q),
            X.KEY_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
            X.VALUE_CACHE_INPUT_NAME: (num_layers, 1, kv_cached_embed_size, 1, cache_len),
        } for q in query_lengths})
    prog.optimize()
    return prog


def save_tokenizer(src: Path, dest: Path, chat_template: str | None) -> None:
    """The source tokenizer files verbatim (+ ``chat_template`` in place of the source template), as the lane's
    dynamic exporter does (export_lfm3_decode_pipelined.save_tokenizer)."""
    from _bundle import is_tokenizer_file

    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and is_tokenizer_file(f.name):
            shutil.copy2(f, dest / f.name)
    if chat_template:
        shutil.copy2(chat_template, dest / "chat_template.jinja")


def _save_form(prog, out: Path, name: str, *, cfg, cache_width: int, max_ctx: int, hf_id: str, revision, src,
               palettize: bool, act: str, history: str, form: str, tokenizer_src, chat_template) -> Path:
    import coreai.runtime as rt

    from _bundle import write_bundle_metadata

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    aimodel = out / f"{name}.aimodel"
    md = rt.AIModelAssetMetadata()
    md.author = "LiquidAI"
    md.license = LICENSE
    md.model_description = (f"{hf_id} — Apple static iOS export for the Neural Engine (LFM3 static builder: conv "
                            f"history in the KV rows, sliding window from the runner's mask, act {act}, "
                            f"history {history}, {'8-bit k-means g32' if palettize else 'fp16'}; form {form})")
    md.creation_date = int(time.time())
    assert md.author == "LiquidAI" and md.license == LICENSE, "AIModelAssetMetadata dropped a field"
    prog.save_asset(aimodel, md)
    write_bundle_metadata(out, name, hf_id, cfg.vocab_size, max_ctx, revision=revision,
                          weights=str(src), mode="pal8_g32" if palettize else "fp16",
                          extra={"lfm3_static": {"form": form,
                                                 "cache_rows": [cfg.num_hidden_layers, 1, cache_width, 1],
                                                 "kv_width": cfg.num_key_value_heads * cfg.head_dim,
                                                 "attention_layers": cfg.attention_layer_ids,
                                                 "conv_layers": cfg.conv_layer_ids,
                                                 "sliding_window": cfg.sliding_window, "act": act,
                                                 "history": history}})
    if tokenizer_src:
        save_tokenizer(Path(tokenizer_src), out / "tokenizer", chat_template)
    return aimodel


def export_lfm3_ane_static(src: str, out_dir: Path, name: str, *, qconfig: str | None = None, palettize: bool = True,
                           max_ctx: int = 4096, num_layers: int | None = None, act: str = "tanh",
                           history: str = "block", tokenizer_src: str | None = None, chat_template: str | None = None,
                           hf_id: str = HF_ID, revision: str | None = None, seed: int = 0,
                           check_form: bool = False, check_ctx: int = 2048) -> tuple[dict, dict]:
    """Export the static bundle dir `out_dir/name` (`name.aimodel` + metadata.json + tokenizer/), and with
    ``check_form`` the Mac check form `out_dir/name_maccheck` from the same palettized model. Returns
    ({form: aimodel path}, timing / size record)."""
    from coreai_models.export import ios as ios_export
    from coreai_models.export.compression import palettize_pytorch_model
    from coreai_models.models.ios import lfm3 as L

    torch.manual_seed(seed)
    rec: dict = {"src": str(src), "name": name, "max_ctx": max_ctx, "num_layers": num_layers, "act": act,
                 "history": history, "palettize": palettize, "qconfig": qconfig or ("PAL8_G32" if palettize else None),
                 "norm_k": L.NORM_K, "builder": "coreai_models.models.ios.lfm3"}
    t0 = time.time()
    model = L.lfm3_ios_from_hf(src, dtype=torch.float16, max_context_length=max_ctx, act=act, history=history,
                               num_layers=num_layers)
    cfg = model.config
    rec["load_s"] = time.time() - t0
    if palettize:
        t1 = time.time()
        model = palettize_pytorch_model(model, L.palettization_example_inputs(model), load_qconfig(qconfig))
        rec["palettize_s"] = time.time() - t1
    save = dict(cfg=cfg, cache_width=model.cache_width, max_ctx=max_ctx, hf_id=hf_id, revision=revision, src=src,
                palettize=palettize, act=act, history=history, tokenizer_src=tokenizer_src,
                chat_template=chat_template)
    forms: dict[str, Path] = {}

    t2 = time.time()
    inputs = L.build_ios_reference_inputs(model, max_ctx)
    programs = ios_export._export_ios_programs(model, inputs)
    rec["trace_s"] = time.time() - t2
    t3 = time.time()
    prog = asyncio.run(ios_export._convert_to_coreai(
        *programs, max_context_length=max_ctx, kv_cached_embed_size=model.cache_width,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_hidden_layers))
    rec["convert_s"] = time.time() - t3
    forms["ios"] = _save_form(prog, out_dir / name, name, form="ios", **save)
    del prog, programs

    if check_form:
        # (Apple's _export_ios_programs left the model in prefill mode; convert_check_form resets it)
        t5 = time.time()
        prog = asyncio.run(convert_check_form(
            model, inputs, cache_len=check_ctx, hidden_size=cfg.hidden_size,
            kv_cached_embed_size=model.cache_width, num_layers=cfg.num_hidden_layers))
        rec["check_form_s"] = time.time() - t5
        forms["maccheck"] = _save_form(prog, out_dir / f"{name}_maccheck", f"{name}_maccheck", form="maccheck",
                                       **save)
        # the host gather of the check form: the iOS table (int8 [V, d]) and its fp16 scale
        import numpy as np

        table = model.load_embeddings.embedding_table.data
        np.save(forms["maccheck"].parent / "emb_int8.npy", table.reshape(table.shape[0], -1).numpy())
        (forms["maccheck"].parent / "emb.json").write_text(json.dumps(
            {"scale_f16": float(model.gather_embeddings.scale.data.to(torch.float16)),
             "dtype": str(table.dtype), "shape": list(table.shape)}) + "\n")
    rec["aimodel_bytes"] = {f: sum(p.stat().st_size for p in a.rglob("*") if p.is_file()) for f, a in forms.items()}
    rec["export_total_s"] = time.time() - t0
    return forms, rec


def aot(aimodel: Path, dest: Path, arch: str, log: Path, compute: str = "neural-engine") -> dict:
    """coreai-build compile (`compute` = neural-engine for the placement check, gpu for the Mac check); `dest` =
    the .aimodelc path. Regions counted by unique name (memory reference_coreai_ane_requires_palettization:
    `find` lists each region twice)."""
    tmp = dest.with_name(dest.name + ".tmpout")
    if dest.exists() or tmp.exists():
        raise SystemExit(f"{dest} (or its tmp) exists")
    tmp.mkdir(parents=True)
    cmd = ["xcrun", "coreai-build", "compile", str(aimodel), "--output", str(tmp), "--platform", ARCH_PLATFORM[arch],
           "--preferred-compute", compute, "--architecture", arch]
    t0 = time.time()
    with log.open("w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(["/usr/bin/time", "-l", *cmd], stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "DEVELOPER_DIR": AOT_XCODE}).returncode
    secs = time.time() - t0
    rec = {"arch": arch, "platform": ARCH_PLATFORM[arch], "compute": compute, "rc": rc, "seconds": round(secs, 1),
           "log": str(log)}
    outs = list(tmp.glob("*.aimodelc"))
    if rc != 0 or not outs:
        rec["error"] = "no .aimodelc"
        return rec
    outs[0].rename(dest)
    tmp.rmdir()
    names = sorted({p.name.split(".")[0] for p in dest.rglob("*_ANE_region_*")})
    rec.update({
        "aimodelc": str(dest),
        "aimodelc_bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
        "resources_bin_bytes": sum(p.stat().st_size for p in dest.rglob("resources.bin")),
        "ane_regions_unique": len(names),
        "ane_region_names": names,
        "main_hash": (dest / "main.hash").read_bytes().hex() if (dest / "main.hash").exists() else None,
    })
    text = log.read_text(errors="replace")
    rec["anec_failures"] = text.count("ANECCompileOffline() failed")
    rec["log_error_lines"] = [ln for ln in text.splitlines() if "rror" in ln or "ailed" in ln][:20]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="LFM3 checkpoint dir (config.json + *.safetensors)")
    ap.add_argument("--tiny", action="store_true", help="random 5-layer LFM3-shaped checkpoint (compile probe)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--qconfig", default=None, help="kmeans_palettization_config yaml (default: 8-bit g32)")
    ap.add_argument("--no-palettize", action="store_true", help="fp16 weights (the int8 embedding table stays)")
    ap.add_argument("--max-context-length", type=int, default=4096)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--act", default="tanh", choices=("tanh", "stock"))
    ap.add_argument("--history", default="block", choices=("block", "column"))
    ap.add_argument("--aot", default="", help="neural-engine compiles of the iOS form: comma list of h16c / h18p / h19p")
    ap.add_argument("--aot-gpu", default="", help="gpu compiles (Mac check): comma list, h16c")
    ap.add_argument("--check-form", action="store_true", help="also export the Mac check form (see above)")
    ap.add_argument("--check-ctx", type=int, default=2048)
    ap.add_argument("--tokenizer-src", default=None)
    ap.add_argument("--chat-template", default=None)
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = a.src
    if a.tiny:
        src = str(write_tiny_checkpoint(out_dir / "tiny_ckpt", a.seed))
    if not src:
        raise SystemExit("--src or --tiny")
    tag = "pal8" if not a.no_palettize else "fp16"
    name = a.name or f"lfm3_{'tiny' if a.tiny else '1_7b'}_static_{tag}_{a.act}_{a.history}_c{a.max_context_length}"
    forms, rec = export_lfm3_ane_static(src, out_dir, name, qconfig=a.qconfig, palettize=not a.no_palettize,
                                        max_ctx=a.max_context_length, num_layers=a.num_layers, act=a.act,
                                        history=a.history, tokenizer_src=a.tokenizer_src,
                                        chat_template=a.chat_template, hf_id=a.hf_id, revision=a.revision,
                                        seed=a.seed, check_form=a.check_form, check_ctx=a.check_ctx)
    for f, p in forms.items():
        print(f"exported {f}: {p} ({rec['aimodel_bytes'][f]:,} B)", flush=True)
    print(f"export total {rec['export_total_s']:.0f} s", flush=True)
    rec["aot"] = []
    jobs = [("ios", arch, "neural-engine") for arch in a.aot.split(",") if arch]
    jobs += [(f, arch, "gpu") for f in forms for arch in a.aot_gpu.split(",") if arch]
    for form, arch, compute in jobs:
        aimodel = forms[form]
        unit = "ane" if compute == "neural-engine" else "gpu"
        r = aot(aimodel, aimodel.parent / f"{aimodel.stem}.{arch}.{unit}.aimodelc", arch,
                aimodel.parent / f"_aot_{arch}_{unit}.log", compute)
        r["form"] = form
        rec["aot"].append(r)
        print(f"AOT {form} {arch} {compute}: rc {r['rc']} {r['seconds']} s, ANE regions (unique) "
              f"{r.get('ane_regions_unique')}, ANEC failures {r.get('anec_failures')}, "
              f"{r.get('aimodelc_bytes', 0):,} B", flush=True)
    (forms["ios"].parent / "_ane_static.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "aot"}, default=str))


if __name__ == "__main__":
    main()
