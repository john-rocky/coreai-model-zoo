#!/usr/bin/env python3
"""Export the d1-3B decoder (final-norm hidden at every position, no vocabulary head) to a Core AI bundle, one static-S function, and AOT-compile it.

The graph is `lfm2_d1_decoder.Lfm2D1Decoder` — the overlay's LFM2.5-VL text decoder on d1-3B's
`model.language_model.*` weights with an Identity head (contract in that file's header):

    input_ids [1,S] i32 (static), position_ids [1,seq] i32 (dynamic), image_embeds [N,2048] (static, N = 2,816)
    + keyCache / valueCache [8,1,8,ctx,64] (ctx dynamic) / convState [22,1,2048,2] -> hidden [1,S,2048]

One function, `main`, at static S = `--prefill-chunk` (16 by default; no S = 1 function). The externalized composites
leave out gated_delta_update (LFM2 has no recurrent scan) and the static-S causal SDPA's externalize guard is retried
with torch's suggested bounds (`export_qwen38vl_pipelined._install_externalize_dim_retry`). The four attention
projections keep fp32 weights on the fp16 export (the overlay loader's `fp32_attn_proj`). Modes:

    fp16                 the reference
    int8lin              weight-only int8 per block of 32 (symmetric_with_clipping) on every MLP linear
                         (feed_forward.gate_proj / up_proj / down_proj) and conv-mixer projection (conv.in_proj /
                         out_proj); excluded: the four attention projections (by name; fp32), the embedding table, the
                         conv1d and every RMSNorm (by type). export_lfm2_decode_pipelined.py's recipe.
    int8mix --fp16-layers I,J,..
                         int8lin with every linear of decoder layers I, J, .. left fp16 (name `.._int8mix_l<I>-<J>_..`)
    int4lin [--quant-block 32|16]
                         int8lin's set at int4 (block 16: name `.._int4lin_b16_..`)

After a quantized mode the int8 / int4 Linear names must equal the intended set exactly (and nothing but Linear is
quantized), or the export stops. There is no lm_head in this graph; `.*lm_head$` stays in the name exclusions as in
the recipe and matches nothing.

The bundle is `<out-dir>/bundles/<name>/`, `<name>` = `d1_3b_decode_<mode>[_n<N>]_pf<S>` (`_n<N>` when
`--n-image-tokens` is not the default, `lfm2_d1_decoder.N_IMAGE_TOKENS` = 2,816):

    <name>.aimodel
    metadata.json        `_bundle.write_bundle_metadata`'s, `kind` rewritten to `decision-backbone`, `source` to the
                         checkpoint's provenance, `language.contract` (input / output / state names, shapes, dtypes; -1
                         = the dynamic axis), top-level `decision` (host.py's contract: request, prompt, option codes and
                         readout groups, rows and the static-S calls, the readout arithmetic, the response; the option
                         rows), `vision` (extension ids V + slot, the N image rows and how a host fills them from the
                         tower bundle's crops, the limits a host refuses, the crop and token rules in short), and for
                         a quantized mode `compression`
    tokenizer/           the pinned snapshot's tokenizer.json, tokenizer_config.json, chat_template.jinja, verbatim
                         (sha256 checked against the pins)
    head/                option_rows.safetensors + option_rows.json (export_option_rows.py: the tied embedding rows of
                         every readout candidate id, bf16 -> fp32)
    LICENSE              the snapshot's LFM Open License v1.0, verbatim

`--aot` compiles it for the Mac GPU into `<out-dir>/bundles_aotc/<name>.h16c.aimodelc` (`coreai-build compile
--platform macOS --preferred-compute gpu --architecture h16c --expect-frequent-reshapes`; the gate loads only the
`.aimodelc` with `SpecializationOptions.default()`, never the JIT). The record gives the compiled asset's main.hash,
which names the entry the Python runtime makes under ~/Library/Caches/coreai-cache/<build>/python/ when it loads it.

`--toy` runs the same path with no checkpoint: toy_graph_check.py's toy config (hidden 64, layers conv / attention /
conv, vocabulary 256, 8 image slots unless `--n-image-tokens` says otherwise: N adds no weight) and seeded random weights
(`--toy-seed`, 0 = round 1's toy, checked bit for bit), name `d1_toy_decode_<mode>[_n<N>]_pf<S>`, bundles under
`<out-dir>/toy_bundles/` and `<out-dir>/toy_bundles_aotc/`. Its head/
holds the toy embedding's rows (fp32) at the real candidate ids folded into the toy vocabulary (id % 256); tokenizer/,
LICENSE and the metadata writer are the real ones, and metadata.json carries a `toy` block saying so.

    cd conversion/d1
    export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer PY=<coreai-models venv>/bin/python
    ~/code/standup/tools/quiet/quiet_wait.py --max-wait 3600 -- \\
        $PY export_decoder.py fp16 --toy --prefill-chunk 16 --aot --record $ZOO_WORK_ROOT/_d1_3b/results/<json>
    $PY export_decoder.py fp16 --prefill-chunk 16 --aot --record <json>     # round 3: needs model.safetensors
    $PY export_decoder.py int8lin --aot --record <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

LANE = work_path("_d1_3b")
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113",
         "license": "LFM Open License v1.0 (license: other, license_name: lfm1.0)"}
WEIGHTS = {"model.safetensors": {"bytes": 6_247_065_504,
                                 "sha256": "50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c"}}
CONFIG_SHA256 = "0cbac0f580bd8036ef94ac83aa632a57a170e66d20f381ef47c3f6486c9a3e8c"
TOKENIZER_FILES = {"tokenizer.json": "8096ecb9f54599d756c8de728a598a340bc1e43c0deb77ddd62456c38349fcee",
                   "tokenizer_config.json": "ef6770d12dd9ac58d334e979bf160c5cdfd41f13d205fd5d5516857645feb0ea",
                   "chat_template.jinja": "86f4770449a4797c9b4212d110b0cc70fb993c1ce960095bcfb12eea22f61cca"}
LICENSE_SHA256 = "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c"
SPECIAL = {"bos": ("<|startoftext|>", 124894), "im_start": ("<|im_start|>", 124899), "im_end": ("<|im_end|>", 124900),
           "pad": ("<|pad|>", 124893), "image": ("<image>", 124907)}
NAME_PREFIX, TOY_PREFIX = "d1_3b_decode", "d1_toy_decode"
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
DISK = "/System/Volumes/Data"
MODES = ("fp16", "int8lin", "int8mix", "int4lin")
QUANT_LEAVES = ("feed_forward.gate_proj", "feed_forward.up_proj", "feed_forward.down_proj", "conv.in_proj",
                "conv.out_proj")
ATTN_PROJ = r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$"


# --------------------------------------------------------------------------- helpers
def du(path: Path) -> str:
    return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    """sha256 per file of a directory asset + one digest over the sorted listing."""
    files = sorted(p for p in path.rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree, "files": per,
            "file_bytes": {str(p.relative_to(path)): p.stat().st_size for p in files}}


def coreai_cache_dir() -> Path:
    build = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()
    return Path.home() / "Library/Caches/coreai-cache" / build / "python"


def coreai_cache_entries() -> dict:
    cc = coreai_cache_dir()
    if not cc.exists():
        return {}
    return {p.name: du(p) for p in sorted(cc.iterdir()) if p.is_dir()}


def disk_free() -> dict:
    """Free space on the data volume, and the runtime cache the gate's loads grow."""
    free = shutil.disk_usage(DISK).free
    return {"time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "free_bytes": free, "free_gib": round(free / 2**30, 1), "coreai_cache_python": coreai_cache_entries()}


def snapshot() -> Path:
    return Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- the toy
def toy_model(seed: int = 0, n_image_tokens: int | None = None):
    """toy_graph_check.toy_model at any seed and image-row count (N adds no parameter: the same seed gives the same
    weights at every N): the toy config, seeded random weights, the norm gains around 1 (fp32)."""
    import torch
    from lfm2_d1_decoder import Lfm2D1Decoder
    from toy_graph_check import N_IMG, toy_config

    torch.manual_seed(seed)
    model = Lfm2D1Decoder(toy_config(), n_image_tokens=N_IMG if n_image_tokens is None else n_image_tokens).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    return model


def toy_round1_check(model, seed: int) -> dict:
    """At seed 0 the toy is round 1's (toy_graph_check.toy_model): every tensor bit for bit."""
    import torch
    import toy_graph_check

    if seed != toy_graph_check.SEED:
        return {"seed": seed, "round1_seed": toy_graph_check.SEED, "compared": False}
    ref = toy_graph_check.toy_model().state_dict()
    mine = model.state_dict()
    same = set(ref) == set(mine) and all(torch.equal(ref[k], mine[k]) for k in ref)
    if not same:
        raise SystemExit("the seed-0 toy differs from toy_graph_check.toy_model()")
    return {"seed": seed, "compared": True, "bit_equal_round1_toy": True, "tensors": len(ref)}


def toy_fold(i: int, vocab: int) -> int:
    """A real id in the toy vocabulary (the toy gate's rows, pad and option ids)."""
    return int(i) % vocab


def toy_table_ids(vocab: int) -> tuple[list[int], dict]:
    """The real candidate ids (export_option_rows.candidate_table on the snapshot's tokenizer) folded mod vocab ->
    (ascending distinct toy ids, {toy id: the real strings that fold onto it})."""
    import export_option_rows as eor
    import host

    tok = host.load_tokenizer(snapshot() / "tokenizer.json")
    cand = eor.candidate_table(tok)
    strings: dict[str, list[str]] = {}
    for rid, texts in cand["strings_of"].items():
        strings.setdefault(str(toy_fold(int(rid), vocab)), []).extend(texts)
    return sorted(int(k) for k in strings), strings


# --------------------------------------------------------------------------- the recipe
def linear_quant_config(dtype: str = "int8", block: int = 32, fp16_layers: list[int] | None = None) -> dict:
    """Weight-only linear per-block (export_lfm2_decode_pipelined.linear_quant_config's recipe): the attention
    projections excluded by name (fp32 weights, the GPU delegate's precision-critical path), lm_head by name (none in
    this graph), SDPA / RMSNorm / Embedding / Conv1d by type; `fp16_layers` keeps every linear of those layers fp16."""
    names: dict = {r".*lm_head$": None, ATTN_PROJ: None}
    for i in sorted(set(fp16_layers or [])):
        names[rf"model\.layers\.{i}\..*"] = None
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": int(block), "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": names,
    }


def intended_quantized(model, fp16_layers: list[int]) -> list[str]:
    """The Linear names the recipe quantizes: QUANT_LEAVES of every decoder layer outside fp16_layers."""
    import torch

    keep = set(fp16_layers)
    out = []
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or not n.startswith("model.layers."):
            continue
        layer, leaf = int(n.split(".")[2]), ".".join(n.split(".")[3:])
        if leaf in QUANT_LEAVES and layer not in keep:
            out.append(n)
    return sorted(out)


def quantize(model, spec: dict, mode: str, block: int, fp16_layers: list[int]) -> tuple[object, dict]:
    import torch
    import torch.nn.utils.parametrize as P

    from coreai_models.export.compression import quantize_pytorch_model

    dtype = "int4" if mode == "int4lin" else "int8"
    want = intended_quantized(model, fp16_layers)
    cfg_q = linear_quant_config(dtype, block, fp16_layers)
    cfg_rec = json.loads(json.dumps(cfg_q))   # the quantizer rewrites the dict it is given: keep a copy
    t0 = time.monotonic()
    print(f"quantizing ({dtype} per-block-{block} symmetric_with_clipping on {len(want)} linears; attention projections "
          f"fp32, embedding / conv1d / norms fp16; fp16 layers {sorted(set(fp16_layers))}) ...", flush=True)
    model = quantize_pytorch_model(model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
    lin = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    got = sorted(n for n, m in lin if P.is_parametrized(m, "weight"))
    other = sorted({type(m).__name__.removeprefix("Parametrized") for n, m in model.named_modules()
                    if P.is_parametrized(m) and not isinstance(m, torch.nn.Linear)})
    attn = [(n, m) for n, m in lin if n.split(".")[-2] == "self_attn"]
    attn_bad = [n for n, m in attn if P.is_parametrized(m, "weight") or m.weight.dtype != torch.float32]
    if got != want or other or attn_bad:
        raise SystemExit(f"{mode}: the quantized set differs from the intended one: extra {sorted(set(got) - set(want))[:8]}, "
                         f"missing {sorted(set(want) - set(got))[:8]}, other quantized types {other}, attention "
                         f"projections quantized or not fp32 {attn_bad[:8]}")

    def codes(m) -> int:
        return int(next(p for p in m.parametrizations["weight"] if hasattr(p, "quantized_data")).quantized_data.numel())

    fp16_lin = sorted(n for n, m in lin if not P.is_parametrized(m, "weight") and n not in {a for a, _ in attn})
    return model, {
        "mode": mode, "dtype": dtype, "block": block, "linear": cfg_rec["global_config"]["op_state_spec"]["weight"],
        "excluded_names": sorted(cfg_rec["module_name_configs"]),
        "excluded_types": sorted(k for k, v in cfg_rec["module_type_configs"].items() if v is None),
        "fp16_layers": sorted(set(fp16_layers)), "quantized_linear_modules": len(got),
        "quantized_set_equals_intended": True, "quantized_params": int(sum(codes(m) for n, m in lin if n in set(got))),
        "fp32_attention_projections": [n for n, _ in attn], "fp16_linear_modules": fp16_lin,
        "fp16_linear_params": int(sum(m.weight.numel() for n, m in lin if n in set(fp16_lin))),
        "seconds": time.monotonic() - t0}


# --------------------------------------------------------------------------- metadata
def contract_block(cfg, S: int, n_img: int) -> dict:
    hd, H = cfg.head_dim, cfg.hidden_size
    return {"function": "main",
            "inputs": {"input_ids": [[1, S], "int32"], "position_ids": [[1, -1], "int32"],
                       "image_embeds": [[n_img, H], "float16"]},
            "outputs": {"hidden": [[1, S, H], "float16"]},
            "states": {"keyCache": [[cfg.num_full_layers, 1, cfg.num_key_value_heads, -1, hd], "float16"],
                       "valueCache": [[cfg.num_full_layers, 1, cfg.num_key_value_heads, -1, hd], "float16"],
                       "convState": [[cfg.num_conv_layers, 1, H, cfg.conv_state_width], "float16"]},
            "dynamic": "-1 = dynamic: position_ids [1, seq] with seq in S..max_context_length-1; the KV sequence axis "
                       "2048..max_context_length (a host allocates it at max_context_length)"}


def readout_text(S: int, max_ctx: int) -> str:
    pad = SPECIAL["pad"]
    return (f"one row per question, fresh zero states per row; a row of T ids runs as ceil(T / {S}) calls of 'main' "
            f"(static S = {S}): call c gets ids[{S}c : {S}c + {S}] with position_ids 0..{S}c+{S - 1}, image_embeds zero "
            f"for a text row; the last call is padded with {pad[0]} ({pad[1]}) and the hidden rows of the padded "
            f"positions are discarded (causal: they cannot reach a real position). The hidden row of the row's last "
            f"real token (the answer slot, T - 1) is the final-norm hidden the readout reads. A row fits when "
            f"ceil(T / {S}) * {S} <= {max_ctx - 1} (the position input's upper bound; the KV sequence axis is "
            f"allocated at {max_ctx}).")


def option_table_block(table: dict, capacity: dict) -> dict:
    """metadata decision.option_table: the ids a host can read, the label space they cover, the refusal."""
    import export_option_rows as eor

    return {"files": ["head/option_rows.safetensors", "head/option_rows.json"], "n": table.get("n"),
            "layout": "ascending int32 `ids` [n], fp32 `rows` [n, hidden]; row k = the tied embedding row of ids[k] "
                      "(bf16 -> fp32, exact)",
            "label_space": "the single-token strings among the code families " + ", ".join(eor.CODE_FAMILIES)
                           + ", the ' ' + code form of each, and yes / Yes / YES / no / No / NO",
            "covers": f"every noul and score question (1..10 levels); every choice whose labels are all ASCII letters "
                      f"(native one-letter codes) or that takes positional codes, up to {capacity.get('max_options')} "
                      f"options (the alias pool's limit)",
            "refuse": "a request any of whose readout ids is not in the table is refused whole, before any graph call: "
                      "\"questions.<name>: the token id <N> of label '<label>' is not in the option table\" "
                      "(host.option_table_check; host.build_request(..., table_ids=))",
            "refused_examples": "a choice of one-letter native labels outside A..Z / a..z (e.g. 'あ', 'é', 'α'); "
                                f"{capacity.get('first_refused')} options or more are refused before it by the alias "
                                "rule ('no single-token alias left')",
            "provider_difference": "the provider's code reads any id; this refusal is the host's own (the fixture's "
                                   "readout ids are all in the table)"}


def decision_block(S: int, max_ctx: int, hidden: int, table: dict, capacity: dict) -> dict:
    """How a host turns a System One request into the graph's rows and the rows' hidden into the response — host.py's
    contract (sections 1-7), written out for a host that reads only the bundle."""
    sp = {k: {"token": t, "id": i} for k, (t, i) in SPECIAL.items()}
    return {
        "output": f"hidden [1, {S}, {hidden}] per call: the final-norm hidden state at every position (no vocabulary "
                  "head in the graph)",
        "readout": readout_text(S, max_ctx),
        "spec": "conversion/d1/host.py (its docstring is the contract; test_host.py gates it against the provider's code "
                f"at {MODEL['hf_id']}@{MODEL['revision']}: prompt.py, api.py, runner.py)",
        "request": {
            "shape": "{state: JSON | null, questions: {name: question}} in request order",
            "noul": "{type: 'noul', instructions: str, criteria?: {true?: str, false?: str} | null}",
            "choice": "{type: 'choice' (the default when type is absent), instructions: str, criteria: {label: str | "
                      "null}} with 1+ options",
            "score": "{type: 'score', instructions: str, criteria: [str]} with 1..10 levels (the prompt asks for a "
                     "single digit)",
            "refuse": "anything else (no instructions, a non-string instructions or level, a choice without options, "
                      "a score of 0 or more than 10 levels, a criteria of another JSON type)",
        },
        "prompt": {
            "row_text": "prefix + suffix",
            "prefix": "BOS + '<|im_start|>user\\n' + state_block + '\\nQUESTION:\\n' (state null: BOS + "
                      "'<|im_start|>user\\n')",
            "state_block": "a string as is + '\\n\\n'; any other JSON value json.dumps(state, ensure_ascii=False, "
                           "indent=2) + '\\n\\n'",
            "suffix": "question_block + '<|im_end|>\\n<|im_start|>assistant\\n'",
            "choice_block": "instructions + '\\n\\nOptions:\\n' + '\\n'.join(code + ' ' + (desc or "
                            "label.replace('_', ' '))) + '\\n\\nReply with the option code only.'",
            "noul_block": "instructions + ('\\nYes: ' + str(criteria.get('true')) + '\\nNo: ' + "
                          "str(criteria.get('false')) when criteria is a non-empty object) + '\\n\\nReply with yes or "
                          "no only.'",
            "score_block": "instructions + '\\n\\n' + '\\n'.join(str(i) + ' ' + level) + '\\n\\nReply with a single "
                           "digit 0-' + str(K - 1) + ' only.'",
            "special": sp,
        },
        "options": {
            "codes": "the labels when every label (strip) is one alphabetic character, else 'A'.. for at most 26, "
                     "else '00', '01', ..",
            "aliases": "each code takes its own id when it is one token not yet taken, else the first single-token "
                       "untaken entry of A..Z, 00..99, a..z, #0..#199, AA..ZZ; the option line prints the taken code",
            "groups": {"choice": "[alias id] + [encode(' ' + code)[0]] when ' ' + code is one other token",
                       "noul": "[single-token forms of yes, Yes, YES], [the same of no, No, NO]",
                       "score": "[encode(str(i))[0]] for i in 0..K-1"},
            "keys": {"noul": ["yes", "no"], "choice": "the labels in request order", "score": "'0'..'K-1'"},
        },
        "tokens": {
            "tokenizer": "tokenizer/tokenizer.json (BPE, ByteLevel; adds no BOS); add_special_tokens=False with the "
                         "special tokens matched in the text (user text is not escaped)",
            "row": "row_ids = encode(prefix + suffix); slot = len(row_ids) - 1",
            "usage": "input_tokens = len(row) for one question; len(encode(prefix)) + sum(len(encode(suffix_q))) for "
                     "several (the provider's Tree; trunk and branches encoded apart)",
        },
        "arithmetic": {
            "logits": "z[id] = h_slot . E[id] in float64 for the ids of the question's groups (h_slot = the slot's "
                      "hidden row, fp16 -> float64; E[id] = head/option_rows, fp32 -> float64)",
            "pool": "score_k = max over group k of z",
            "softmax": "p = softmax_k(score) in double (Python's math.exp and 3.12 sum()); the provider's vocabulary "
                       "log-sum-exp cancels in a softmax over options",
        },
        "option_table": option_table_block(table, capacity),
        "response": {
            "noul": "{type: 'noul', noul: p[0]} (p = [P(yes), P(no)])",
            "choice": "{type: 'choice', choice: labels[argmax], confidence: p[argmax], probabilities: {label: p}}",
            "score": "{type: 'score', score: sum(i * p_i), confidence: p[argmax], probabilities: {'i': p_i}, legend: "
                     "{'i': level_i}}",
            "argmax": "the first index of the largest p; floats not rounded",
            "body": "{answers: {name: answer} (request order), usage: {input_tokens, output_tokens: 0}}",
        },
    }


def vision_block(cfg, n_img: int) -> dict:
    """How a host turns pictures into the graph's image rows and extension ids — vision_host.py's contract (gated bit for
    bit against transformers 5.19's processor), K/results/vision_rules.md, the tower's in lfm2_vl_tower.py."""
    return {
        "image_token": {"token": SPECIAL["image"][0], "id": SPECIAL["image"][1]},
        "extension_ids": f"the k-th <image> of the row (k counted over every picture and every crop, in text order) is "
                         f"sent as id V + k (V = vocab_size {cfg.vocab_size}, k < {n_img}); the graph reads "
                         "image_embeds[k] for it",
        "n_image_tokens": n_img,
        "image_embeds": [[n_img, cfg.hidden_size], "float16"],
        "rows": "rows 0 .. n - 1 (n = the row's <image> count) = the tower bundle's image_embeds of every crop, each "
                "crop's first h w / 4 rows (its merged grid row-major), concatenated in crop order (pictures in text "
                "order; per picture its tiles row-major, then the thumbnail), cast to float16; rows n .. N - 1 zero; "
                "the same buffer is bound to every call of the row",
        "text_only": "image_embeds zero and no extension id",
        "limits": {"one_picture": "at most 2,810 image tokens (10 tiles + a thumbnail, aspect up to 4:1 after "
                                  "cap_pixels; K/results/vision_grid_table.json)",
                   "refuse": f"a request whose pictures need more than {n_img} image tokens together, or whose row is "
                             "over the position bound (host.graph_context_check), is refused whole before any graph call",
                   "refusal_text": f"images: <n> image tokens over the graph's {n_img} image rows"},
        "tower": {"bundle": "d1_3b_vision_<dtype> (export_vision.py; contract in lfm2_vl_tower.py's header)",
                  "per_crop": "vision_host.tower_inputs(crop, position table) -> patches [1024, 768], pos_table [1024, d], "
                              "key_bias [1024], unshuffle_idx [256, 4] int32 -> image_embeds [256, text hidden]; the "
                              "crop's rows are the first h w / 4"},
        "crops": "per picture (after cap_pixels): one crop at the smart size (aspect kept, sides multiples of 32, 64..256 "
                 "image tokens) when max(16, round32(h)) * max(16, round32(w)) <= 524,288 px, else rows x cols tiles of "
                 "512 x 512 (2 <= rows * cols <= 10, the closest aspect) then a thumbnail at the smart size",
        "tokens": "<|image_start|> + (one crop: <image> x tokens | tiles: <|img_row_r_col_c|> + <image> x 256 per tile, "
                  "then <|img_thumbnail|> + <image> x thumbnail tokens) + <|image_end|>; the prompt holds one '<image>' "
                  "per picture after '<|im_start|>user\\n', replaced by that run (vision_host.prompt_ids)",
        "spec": "conversion/d1/vision_host.py (sections 1-8; test_vision_host.py gates it against the provider's code "
                "and transformers 5.19's processor)",
    }


def compression_block(quant: dict) -> dict:
    return {"scheme": quant["mode"],
            "linear": f"{quant['dtype']} per-block-{quant['block']} symmetric_with_clipping (weight only)",
            "quantized": "every MLP linear (feed_forward.gate_proj / up_proj / down_proj) and conv-mixer projection "
                         "(conv.in_proj / out_proj)" + (f" outside layers {quant['fp16_layers']}" if quant["fp16_layers"]
                                                       else ""),
            "excluded": "fp32: the four attention projections of every full-attention layer; fp16: the embedding "
                        "table (the tied table of head/option_rows), the conv1d, every RMSNorm, SDPA",
            "excluded_names": quant["excluded_names"], "excluded_types": quant["excluded_types"],
            "quantized_linear_modules": quant["quantized_linear_modules"], "fp16_layers": quant["fp16_layers"],
            "fp16_linear_modules": quant["fp16_linear_modules"],
            "head": "none in the graph (the tied rows of the readout ids are head/option_rows)"}


def copy_verbatim(src_dir: Path, dst: Path, files: dict[str, str]) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, want in files.items():
        src = src_dir / name
        if sha256_file(src) != want:
            raise SystemExit(f"{src}: sha256 {sha256_file(src)} != pinned {want}")
        shutil.copyfile(src, dst / name)
        got[name] = sha256_file(dst / name)
        if got[name] != want:
            raise SystemExit(f"the copy of {name} differs from the snapshot")
    return got


def write_metadata(out_dir: Path, name: str, cfg, args, quant: dict | None, table: dict, toy: dict | None,
                   capacity: dict) -> dict:
    """metadata.json (`_bundle.write_bundle_metadata` + kind decision-backbone + source + the d1 blocks), tokenizer/
    and LICENSE into `out_dir` (head/ is written before) -> what was written."""
    from _bundle import write_bundle_metadata

    S, n_img = args.prefill_chunk, args.n_image_tokens
    language_extra = {"prefill_chunk": S, "static_inputs": ["image_embeds"], "image_tokens_max": n_img,
                      "output": f"hidden [1, {S}, {cfg.hidden_size}] fp16, every position",
                      "contract": contract_block(cfg, S, n_img)}
    extra = {"decision": decision_block(S, args.max_ctx, cfg.hidden_size, table, capacity),
             "vision": vision_block(cfg, n_img)}
    if quant:
        extra["compression"] = compression_block(quant)
    if toy:
        extra["toy"] = toy
    write_bundle_metadata(out_dir, name, MODEL["hf_id"], cfg.vocab_size, args.max_ctx, revision=MODEL["revision"],
                          mode=args.mode, functions=("main",), language_extra=language_extra, extra=extra)
    # The graph returns hidden states, not logits: a generation loop must not pick this bundle up as an llm.
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kind"] = "decision-backbone"
    meta["source"] = {
        "model_definition": "torch (conversion/d1/lfm2_d1_decoder.py: coreai_models.models.macos.lfm2_vl."
                            "Lfm2VlPipelinedForCausalLM with an Identity head)",
        "hf_model_id": MODEL["hf_id"], "hf_revision": MODEL["revision"], "license": MODEL["license"],
        "weights": ("random (the toy: NOT the model's weights)" if toy else
                    {"file": "model.safetensors", **WEIGHTS["model.safetensors"], "keys": "model.language_model.*"}),
        "tokenizer": {"files_sha256": TOKENIZER_FILES}}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    snap = snapshot()
    tok = copy_verbatim(snap, out_dir / "tokenizer", TOKENIZER_FILES)
    lic = copy_verbatim(snap, out_dir, {"LICENSE": LICENSE_SHA256})
    return {"metadata_sha256": sha256_file(meta_path), "kind": meta["kind"], "tokenizer_sha256": tok,
            "tokenizer_verbatim_from_snapshot": tok == TOKENIZER_FILES, "license_sha256": lic["LICENSE"]}


# --------------------------------------------------------------------------- export
def default_n_image_tokens(toy: bool) -> int:
    """The image_embeds rows when --n-image-tokens is not given: the contract's N, or the toy's 8."""
    if toy:
        from toy_graph_check import N_IMG
        return N_IMG
    from lfm2_d1_decoder import N_IMAGE_TOKENS
    return N_IMAGE_TOKENS


def bundle_name(args) -> str:
    if args.name:
        return args.name
    suffix = ""
    if args.mode == "int8mix":
        suffix = "_l" + "-".join(str(i) for i in sorted(set(args.fp16_layers)))
    if args.mode == "int4lin" and args.quant_block != 32:
        suffix = f"_b{args.quant_block}"
    if args.n_image_tokens != default_n_image_tokens(args.toy):
        suffix += f"_n{args.n_image_tokens}"
    return f"{TOY_PREFIX if args.toy else NAME_PREFIX}_{args.mode}{suffix}_pf{args.prefill_chunk}"


def load_real(args) -> tuple[object, dict]:
    """The checkpoint's text decoder (fp16, attention projections fp32) with its load report asserted."""
    import torch
    from lfm2_d1_decoder import EXPECTED_TEXT_CONFIG, Lfm2D1Decoder, text_config_record

    snap = snapshot()
    if sha256_file(snap / "config.json") != CONFIG_SHA256:
        raise SystemExit(f"{snap}/config.json differs from the pinned revision's")
    ck = snap / "model.safetensors"
    if not ck.exists():
        raise SystemExit(f"no {ck}: the 6.2 GB checkpoint is not downloaded (round 3)")
    weights = {"file": str(ck), "bytes": ck.stat().st_size, "sha256": sha256_file(ck)}
    if weights["bytes"] != WEIGHTS["model.safetensors"]["bytes"] or weights["sha256"] != WEIGHTS["model.safetensors"]["sha256"]:
        raise SystemExit(f"{ck} differs from the pinned LFS object: {weights}")
    print(f"loading {MODEL['hf_id']}@{MODEL['revision'][:8]} text decoder fp16 (attention projections fp32, no head) ...",
          flush=True)
    model = Lfm2D1Decoder.from_hf(str(snap), target_dtype=torch.float16, n_image_tokens=args.n_image_tokens,
                                  fp32_attn_proj=True)
    rep = model.load_report
    cfg_rec = text_config_record(model.config)
    bad_cfg = {k: (cfg_rec[k], v) for k, v in EXPECTED_TEXT_CONFIG.items() if cfg_rec[k] != v}
    if (rep["unread_checkpoint_keys_under_prefix"] or rep["module_tensors_not_in_checkpoint"] or rep["meta_params"]
            or rep["module_has_lm_head_weight"] or bad_cfg):
        raise SystemExit(f"load mismatch: {json.dumps(rep)} config {bad_cfg}")
    return model, {"load_report": rep, "weights": weights, "text_config": cfg_rec}


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from toy_graph_check import count_ops, typed_module

    import export_option_rows as eor
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an export never overwrites a bundle (remove it first, on purpose)")
    dtype = torch.float16
    S = args.prefill_chunk
    disk = {"before_load": disk_free()}
    t0 = time.monotonic()
    toy = None
    if args.toy:
        base = toy_model(args.toy_seed, args.n_image_tokens)
        load = {"toy": True, **toy_round1_check(base, args.toy_seed),
                "parameters": int(sum(p.numel() for p in base.parameters()))}
        model = typed_module(base, dtype)
        cfg = model.config
        toy = {"what": "a toy: toy_graph_check.py's config with seeded random weights through the export path; NOT the "
                       "model. The gate's toy oracle folds every real id into the toy vocabulary (id % vocab_size): "
                       "the rows, the pad and the option-table ids",
               "seed": args.toy_seed, "fold": f"id % {cfg.vocab_size}", "pad_folded": toy_fold(SPECIAL["pad"][1], cfg.vocab_size),
               "config": {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
                          "layer_types": cfg.layer_types, "vocab_size": cfg.vocab_size, "ff_dim": cfg.ff_dim,
                          "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
                          "head_dim": cfg.head_dim, "conv_L_cache": cfg.conv_L_cache, "rope_theta": cfg.rope_theta,
                          "n_image_tokens": args.n_image_tokens}}
        emb = base.model.embed_tokens.weight.detach().float().numpy()
        table_ids, table_strings = toy_table_ids(cfg.vocab_size)

        def table_fn(ids, emb=emb):
            return emb[list(ids)]
        table_source = {"toy": True, "seed": args.toy_seed, "rows": "the toy embedding (fp32 random), no bf16 step",
                        "ids": f"export_option_rows.candidate_ids of the snapshot's tokenizer, folded id % {cfg.vocab_size}"}
    else:
        model, load = load_real(args)
        cfg = model.config
    if args.n_image_tokens != model.n_image_tokens:
        raise SystemExit(f"n_image_tokens {args.n_image_tokens} != the module's {model.n_image_tokens}")
    spec = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=S)
    t_loaded = time.monotonic()
    disk["after_load"] = disk_free()

    quant = None
    if args.mode != "fp16":
        block = args.quant_block if args.mode == "int4lin" else 32
        model, quant = quantize(model, spec, args.mode, block, args.fp16_layers or [])
        print(f"quantized in {quant['seconds']:.1f}s: {quant['quantized_linear_modules']} {quant['dtype']} linears "
              f"({quant['quantized_params']:,} params), {len(quant['fp16_linear_modules'])} fp16 linears", flush=True)
    t_quantized = time.monotonic()

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    from export_qwen38vl_pipelined import _install_externalize_dim_retry

    _install_externalize_dim_retry()
    print(f"exporting the hidden-output decoder (one function 'main', static S={S}, {args.mode}) ...", flush=True)
    prog = export_to_coreai(model, spec["reference_inputs"], dynamic_shapes=spec["dynamic_shapes"],
                            input_names=spec["input_names"], output_names=spec["output_names"],
                            state_names=spec["state_names"], externalize_modules=specs)
    t_converted = time.monotonic()
    prog.optimize()
    t_exported = time.monotonic()
    ops = count_ops(prog)
    print(f"converted in {t_converted - t_quantized:.1f}s, optimized in {t_exported - t_converted:.1f}s, "
          f"{ops.get('ops')} ops", flush=True)
    disk["after_export"] = disk_free()

    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    disk["after_save"] = disk_free()
    if args.toy:
        table = eor.write_table(table_fn, table_ids, out_dir / "head", strings=table_strings, source=table_source)
    else:
        table = eor.write_checkpoint_table(out_dir / "head", snapshot())
    import host

    capacity = eor.alias_capacity(eor.CachedTok(host.load_tokenizer(snapshot() / "tokenizer.json")))
    meta = write_metadata(out_dir, name, cfg, args, quant, table, toy, capacity)
    mlirb = aimodel / "main.mlirb"
    main_hash = aimodel / "main.hash"
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "toy": bool(args.toy), "load": load,
           "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN, "max_ctx": args.max_ctx, "functions": ["main"], "query_len": S,
           "n_image_tokens": args.n_image_tokens, "spec": {
               "input_names": list(spec["input_names"]), "output_names": list(spec["output_names"]),
               "state_names": list(spec["state_names"]),
               "reference_inputs": {k: [list(v.shape), str(v.dtype).replace("torch.", "")]
                                    for k, v in spec["reference_inputs"].items()},
               "dynamic": {k: (None if v is None else {str(a): getattr(d, "__name__", str(d)) for a, d in v.items()})
                           for k, v in spec["dynamic_shapes"].items()}},
           "fp32_params": sorted(n for n, p in model.named_parameters() if p.dtype == torch.float32),
           "quantization": quant, "ops": ops,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded, "export": t_converted - t_quantized,
                       "optimize": t_exported - t_converted, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "du_aimodel": du(aimodel), "du_bundle": du(out_dir),
           "aimodel_files": {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*")) if p.is_file()},
           "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
           "main_hash_hex": main_hash.read_bytes().hex() if main_hash.exists() else None,
           "head": {k: table[k] for k in ("file", "bytes", "sha256", "n", "hidden", "readback", "source")},
           **meta, "disk": disk}
    print(f"bundle ready: {out_dir} ({rec['du_aimodel']}, main.mlirb {rec['main_mlirb']['bytes']:,} B sha256 "
          f"{rec['main_mlirb']['sha256'][:16]}, head {table['n']} rows, total {rec['seconds']['total']:.1f}s)", flush=True)
    return rec


def aot_compile(aimodel: Path, out_dir: Path) -> tuple[Path, float, dict]:
    target = out_dir / f"{aimodel.stem}.h16c.aimodelc"
    if not os.environ.get("DEVELOPER_DIR"):
        sys.exit("set DEVELOPER_DIR to the Xcode 27 RC (its Metal toolchain carries coreai-build)")
    cb = subprocess.run(["xcrun", "-f", "coreai-build"], capture_output=True, text=True)
    if cb.returncode != 0 or not cb.stdout.strip():
        sys.exit("xcrun -f coreai-build failed:\n" + cb.stderr)
    if target.exists():
        sys.exit(f"{target} exists: an AOT compile never overwrites an asset (remove it first, on purpose)")
    out_dir.mkdir(parents=True, exist_ok=True)
    disk = {"before": disk_free()}
    cmd = [cb.stdout.strip(), "compile", str(aimodel), "--output", str(out_dir), *AOT_FLAGS]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    secs = time.monotonic() - t0
    disk["after"] = disk_free()
    info = {"coreai_build": cb.stdout.strip(), "command": cmd, "disk": disk, "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-20:], "stderr_tail": proc.stderr.splitlines()[-40:]}
    if proc.returncode != 0 or not target.exists():
        info["failed"] = True
        return target, secs, info
    mh = target / "main.hash"
    info["main_hash_hex"] = mh.read_bytes().hex() if mh.exists() else None
    info["runtime_cache_entry"] = (str(coreai_cache_dir() / info["main_hash_hex"]) if info["main_hash_hex"] else None)
    stats = target / "stats.json"
    info["stats"] = json.loads(stats.read_text()) if stats.exists() else None
    return target, secs, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp16", choices=list(MODES))
    ap.add_argument("--fp16-layers", type=lambda s: [int(x) for x in s.split(",") if x != ""],
                    help="int8mix only: comma list of decoder layer indices whose linears stay fp16")
    ap.add_argument("--quant-block", type=int, default=32, choices=[16, 32], help="int4lin only: the per-block size")
    ap.add_argument("--prefill-chunk", type=int, default=16, help="static S of the one function 'main' (name _pf<S>)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--n-image-tokens", type=int, default=None,
                    help="the image_embeds rows N (default lfm2_d1_decoder.N_IMAGE_TOKENS; the toy's 8); a non-default "
                         "N adds _n<N> to the name")
    ap.add_argument("--out-dir", default=str(LANE / "exports"),
                    help="bundles go to <out-dir>/bundles/<name>/ (the toy's to toy_bundles/), AOT assets to "
                         "<out-dir>/bundles_aotc/ (toy_bundles_aotc/)")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--toy", action="store_true",
                    help="toy_graph_check.py's toy config and seeded random weights through the same path (no checkpoint)")
    ap.add_argument("--toy-seed", type=int, default=0, help="--toy: the weights' seed (0 = round 1's toy)")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel (with --aot)")
    ap.add_argument("--aot", action="store_true", help="compile the .aimodel for the Mac GPU (h16c, efr)")
    ap.add_argument("--record", help="write the export / AOT record JSON here (never overwritten)")
    args = ap.parse_args()
    if (args.mode == "int8mix") != bool(args.fp16_layers):
        ap.error("--fp16-layers is required by int8mix and applies to it only")
    if args.quant_block != 32 and args.mode != "int4lin":
        ap.error("--quant-block applies to int4lin only (the int8 modes are block 32)")
    if args.prefill_chunk < 2:
        ap.error("--prefill-chunk must be >= 2 (this graph has no S=1 function)")
    if args.n_image_tokens is None:
        args.n_image_tokens = default_n_image_tokens(args.toy)
    if args.n_image_tokens < 1:
        ap.error("--n-image-tokens must be >= 1")
    name = bundle_name(args)
    sub = "toy_bundles" if args.toy else "bundles"
    out_dir = Path(args.out_dir) / sub / name
    aot_dir = Path(args.out_dir) / f"{sub}_aotc"
    if args.record and Path(args.record).exists():
        sys.exit(f"{args.record} exists: records are never overwritten")
    record: dict = {"mode": args.mode, "name": name, "toy": args.toy, "toy_seed": args.toy_seed if args.toy else None,
                    "hf_id": MODEL["hf_id"], "revision": MODEL["revision"], "prefill_chunk": args.prefill_chunk,
                    "fp16_layers": args.fp16_layers, "quant_block": args.quant_block, "argv": sys.argv[1:],
                    "pid": os.getpid(), "started": now(), "script_sha256": sha256_file(Path(__file__).resolve()),
                    "module_sha256": sha256_file(HERE / "lfm2_d1_decoder.py"),
                    "option_rows_sha256": sha256_file(HERE / "export_option_rows.py")}

    def save_record() -> None:
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1) + "\n")

    if not args.skip_export:
        record["export"] = export(args, out_dir, name)
        save_record()
    if args.aot:
        aimodelc, secs, info = aot_compile(out_dir / f"{name}.aimodel", aot_dir)
        record["aot"] = {"aimodelc": str(aimodelc), "flags": AOT_FLAGS, "seconds": secs, **info}
        if info.get("failed"):
            record["finished"] = now()
            save_record()
            print("\n".join(info["stderr_tail"]), file=sys.stderr)
            sys.exit(f"coreai-build failed (exit {info['returncode']}) after {secs:.1f} s; record {args.record}")
        record["aot"].update({"du_aimodelc": du(aimodelc), "digest": tree_digest(aimodelc)})
        print(f"asset: {aimodelc} (compile {secs:.1f} s, {record['aot']['du_aimodelc']}, "
              f"{record['aot']['digest']['bytes']:,} B, main.hash {info.get('main_hash_hex')})", flush=True)
    record["finished"] = now()
    save_record()
    if args.record:
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
