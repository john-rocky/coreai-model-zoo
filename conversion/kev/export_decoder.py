#!/usr/bin/env python3
"""Export the Kev-0.8B decoder (final-norm hidden at every position) to a Core AI bundle, one static-S function, and AOT-compile it.

The graph is `qwen3_5_kev_decoder.Qwen3_5KevDecoder` — the overlay's stateful Qwen3.5 text decoder on the
merged Kev-0.8B weights with no vocabulary head:

    input_ids [1,S] i32 (static), position_ids [1,seq] i32 (dynamic)
    + keyCache / valueCache (dynamic sequence dim) / convState / recState -> hidden [1,S,1024]

One function, `main` (export_to_coreai's default entry point), at static S = `--prefill-chunk` (16 by
default; there is no S=1 function: the head reads the <decide> and every </opt> position of one row).
Every linear-attention layer runs `use_loopfree_unroll` (the overlay's `_gated_delta_step_unroll`: the S
single steps unrolled in-graph, fp32 inside the call, no doubling inverse), gated_delta_update is left out
of the externalized composites (the loop-free path never calls it), and the static-S causal SDPA's
externalize guard is retried with torch's suggested bounds
(`export_qwen38vl_pipelined._install_externalize_dim_retry`). Modes:

    fp16               the reference
    int8lin            every decoder linear int8 per block of 32 (symmetric_with_clipping, weight only);
                       the embedding table, conv1d and norms stay fp16 (there is no lm_head in this graph)
    int8mix --fp16-layers I,J,..
                       int8lin with every linear of decoder layers I, J, .. left fp16 (excluded by name);
                       the layers go into the bundle metadata (`compression.fp16_layers`)

The weights are the author's full-weight checkpoint (`scripts/merge_lora_checkpoint.py` at tag kev-1.0,
fp32, keys without a prefix), read from a local HF-cache-form id (`kev-local/kev-0.8b-v1.0-merged`, a
symlinked snapshot under $ZOO_WORK_ROOT/_kev/hf; not a Hub repository).

The bundle is `<out-dir>/bundles/<name>/` = `<name>.aimodel` + `metadata.json` + `tokenizer/` + `head/`, with
`<name>` = `kev_0_8b_decode_<mode>_pf<S>`. `metadata.json` is `_bundle.write_bundle_metadata`'s with `kind`
rewritten to `decision-backbone` (the graph returns hidden states, not logits), the `source` block rewritten
to the adapter / base / merge provenance, and a top-level `decision` block (how a host turns the hidden rows
into typed decisions: the row layout and delimiter ids, the API-to-row mapping, the chunk driving, the head,
the per-question softmax, the response shape). `tokenizer/` is the pinned base snapshot's tokenizer.json and
tokenizer_config.json, copied verbatim (sha256 checked; the author's loader reads the tokenizer from the
base). `head/` is the pointer head (`head.safetensors` + `kev_head.json`, from export_head.py).

`--aot` compiles the `.aimodel` ahead of time for the Mac GPU into
`<out-dir>/bundles_aotc/<name>.h16c.aimodelc` (`coreai-build compile --platform macOS --preferred-compute
gpu --architecture h16c --expect-frequent-reshapes`; the gate loads only the `.aimodelc`, never the JIT).
The gate is `readout_gate.py`.

    cd conversion/kev
    HF_HOME=$ZOO_WORK_ROOT/_kev/hf HF_HUB_OFFLINE=1 \\
      DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer \\
      <coreai-models venv>/bin/python export_decoder.py fp16 --prefill-chunk 16 --aot --record <json>

Kev-4B (round 4): `--model kev-4b` selects the other merged checkpoint (`kev-local/kev-4b-v1.0-merged`, four
fp32 shards), its provenance, its base's tokenizer files, the head in `<work>/_kev/oracle_4b/head`, hidden 2560 and
the bundle name `kev_4b_decode_<mode>_pf<S>`; every other line of the recipe is the same.

Round 11 (the GDN scan's form, fp16 only): `--gdn-scan` picks how every linear-attention layer runs its recurrence
inside the call; the graph's inputs, outputs and states do not change.

    unroll   (default, every bundle before round 11) `_gated_delta_step_unroll`; nothing below changes for it
    chunk    the overlay's in-graph chunk scan `_gated_delta_chunk` (S tokens in parallel, the triangular inverse as
             ceil(log2 S) doublings, fp32 inside the call; the overlay records chunk >= 64 breaking even in fp32)
    metal    the overlay's fp32 Metal chunk kernel (`qwen3_5_gdn_metal`: the step recurrence over the whole chunk in one
             GPU dispatch per layer), built with chunk_max = S and registered with the converter through
             `gemma4_metal_mlp.export_to_coreai_with_kernels` (the same externalized composites)

The bundle name gets the form for chunk and metal (`kev_0_8b_decode_fp16_<chunk|metal>_pf<S>`), and their
`metadata.json` a top-level `gdn_scan` block (form, chunk_doublings or chunk_max and kernel name); the export record
carries `gdn_scan` for every form.

Round 14 (`--dynamic-query`, metal only): the one function's query length is dynamic, 2..cap with cap =
`--prefill-chunk` (the kernel is built with chunk_max = cap). The bundle is `kev_0_8b_decode_fp16_metal_dyn<cap>`
(`--dyn-output padded`: `..._dynpad<cap>`, the hidden output a static [1, cap, hidden] whose rows s..cap-1 are zero);
metadata.json's language block names `query_len_range` [2, cap] in place of `prefill_chunk`, and decision.readout
describes how a host cuts a row into calls.

Round 15 (the host's call lengths, and the shipped bundle): `--query-multiple q` writes `language.query_len_multiple`
= q and `--query-call-max L` writes `language.query_len_call_max` = L (<= the cap, a multiple of q): a host cuts a row
into pieces of L ids and pads the last one with <|endoftext|> up to the next multiple of q (conversion/kev/host.py
`plan`), so a process sees at most L / q call lengths, each paying its one-time specialization once; the graph's cap
can be larger than L (a call costs the same on a larger-cap graph, round 14). Without them (round 14's bundles) a host
reads q = 1 and L = the cap: the real ids only, a 1-id remainder folded into the piece before it. `--metadata-only` writes metadata.json, tokenizer/ and head/ beside a `.aimodel`
that already exists (no export, no model load: the numbers come from the checkpoint's config.json); with
`--from-aimodel <dir>` it first clones that `.aimodel` (APFS `cp -c`, every file's sha256 checked against the source)
into the bundle directory, `--bundle-dir` naming it in place of `<out-dir>/bundles/<name>`. The shipped Kev-0.8B
bundle is round 14's gated `.aimodel` bytes with this metadata:

    $PY export_decoder.py fp16 --gdn-scan metal --dynamic-query --prefill-chunk 128 --query-multiple 16 --query-call-max 128 \\
        --metadata-only --from-aimodel $K/exports/bundles/kev_0_8b_decode_fp16_metal_dyn128/kev_0_8b_decode_fp16_metal_dyn128.aimodel \\
        --bundle-dir $K/exports/ship/kev_0_8b_decode_fp16_metal_dyn128 --record $K/results/r15_ship_bundle_dyn128.json

Round 16 (`--kernel-io`, --dynamic-query only): the GDN kernel's edges. `dynamic` (default; every earlier bundle) is the
overlay's kernel, whose conv / g / beta inputs are as long as the call. `static` zero-pads those three inputs in-graph to
the cap and passes the call's length as a 1-element int32 input (the kernel runs t < length); `static-pad` pads them
and runs every one of the cap's steps (a padded step leaves the state as it is). Both use the copy
conversion/kev/qwen3_5_gdn_metal_static.py; the graph's inputs, outputs, states and metadata keys are the dynamic
graph's, the bundle name gets `s` / `p` after the cap (`kev_0_8b_decode_fp16_metal_dyn512s`) and metadata.json's
`gdn_scan` block a `kernel_io` line. Measured in round 16 (Kev-0.8B, Mac): `static` exports, compiles and runs, each call
is 0.9-1.4 ms slower than the dynamic-edge graph's and each new call length adds as much memory or more, so it does not
answer the dynamic graph's footprint growth (a process cycling through four or more call lengths grows on every call,
round 15's probe); the option stays as the record of that experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from _paths import hf_snapshot, record_argv, work_path  # noqa: E402

LANE = work_path("_kev")
os.environ.setdefault("HF_HOME", str(LANE / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "kev-local/kev-0.8b-v1.0-merged"
WEIGHTS_SHA256 = "ab6bd41853ef2a45d56923202e15c8c9c3dcedaec152b1fb8e529f1ec208415f"   # merge_lora_checkpoint.py
MERGED_SAFETENSORS_SHA256 = "b5ebf92a9994a96d5c0c21b52f5eae23049c9e8b5a85fc625b4cea64076408db"
ADAPTER = {"repo": "jaredpalmer/kev-0.8b", "tag": "v1.0", "tag_target": "788ddbdd65715bb03a56788c822f6c632c9a551d",
           "resolved_commit": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
           "weights_identical_to": "9a45d25eb2ab761841196625383fa1dff0e56c1e (the card's weight revision; same LFS oids)",
           "adapter_model_safetensors_sha256": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
           "head_pt_sha256": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
           "license": "apache-2.0"}
BASE = {"repo": "Qwen/Qwen3.5-0.8B-Base", "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "safetensors_sha256": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c", "license": "apache-2.0"}
KEV_CODE = {"repo": "https://github.com/jaredpalmer/kev", "tag": "kev-1.0", "commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc"}
TOKENIZER_FILES = {"tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
                   "tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de"}
HEAD_DIR = LANE / "oracle" / "head"
HEAD_FILES = ("head.safetensors", "kev_head.json")
HIDDEN, TEMPERATURE, ADAPTED_TENSORS, TENSORS, NAME_PREFIX = 1024, 2.3510958125672174, 186, 320, "kev_0_8b_decode"
MERGED_FILES = {"model.safetensors": MERGED_SAFETENSORS_SHA256}
MODEL_PROFILES = {
    "kev-0.8b": None,   # the constants above
    "kev-4b": {
        "HF_ID": "kev-local/kev-4b-v1.0-merged",
        "WEIGHTS_SHA256": "904380cbf0be134e122f2abd7c7bab52f1e2a0f373d3a2e7947a3025e882f3f3",
        "MERGED_FILES": {"model-00001-of-00004.safetensors": "45a919c6e0c4a907324e8735cd3be10ab8bdad2ba67fa5844367e0fe3a4bd18f",
                         "model-00002-of-00004.safetensors": "9e35c50caf92e662b03a092fc6b57db69683f9ae4076661a2aef962ea175aaab",
                         "model-00003-of-00004.safetensors": "c9036c7fed9c87d419f81435955b4a84e6803c1d8b7029e37cd9979aaf6815f3",
                         "model-00004-of-00004.safetensors": "47cde07c0ef1909177ea7c6f6bc75033443ca186f8567d72a9ef5bfd000d2b89"},
        "ADAPTER": {"repo": "jaredpalmer/kev-4b", "tag": "v1.0", "tag_target": "591dcb5bd6d05eb0b5131ea6608f93f10243335c",
                    "resolved_commit": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
                    "weights_identical_to": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101 (the card's weight revision; same LFS oids)",
                    "adapter_model_safetensors_sha256": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
                    "head_pt_sha256": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
                    "license": "apache-2.0"},
        "BASE": {"repo": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                 "safetensors_sha256": {
                     "model.safetensors-00001-of-00002.safetensors": "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712",
                     "model.safetensors-00002-of-00002.safetensors": "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef"},
                 "license": "apache-2.0"},
        "TOKENIZER_FILES": {"tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
                            "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c"},
        "HEAD_DIR": LANE / "oracle_4b" / "head",
        "HIDDEN": 2560, "TEMPERATURE": 2.406050072164233, "ADAPTED_TENSORS": 248, "TENSORS": 426,
        "NAME_PREFIX": "kev_4b_decode",
    },
}


def select_model(model: str) -> None:
    """--model: rebind the per-checkpoint constants (Kev-0.8B's are the module's own)."""
    prof = MODEL_PROFILES[model]
    if prof:
        globals().update(prof)


def check_merged_files(hf_id: str) -> dict:
    """The local id's snapshot holds the merged checkpoint gated in fp32 torch: sha256 of every weight file."""
    snap = Path(hf_snapshot(hf_id))
    got = {n: sha256_file(snap / n) for n in MERGED_FILES}
    if got != MERGED_FILES or sorted(p.name for p in snap.glob("*.safetensors")) != sorted(MERGED_FILES):
        sys.exit(f"{snap}: merged weights differ from the gated checkpoint: {got}")
    return {"snapshot": str(snap), "sha256": got}
DELIM = {"state": ("<|fim_prefix|>", 248060), "q": ("<|fim_middle|>", 248061), "opt": ("<|box_start|>", 248049),
         "opt_end": ("<|box_end|>", 248050), "decide": ("<|fim_suffix|>", 248062)}
PAD_ID = 248044
AOT_FLAGS = ["--platform", "macOS", "--preferred-compute", "gpu", "--architecture", "h16c",
             "--expect-frequent-reshapes"]
DISK = "/System/Volumes/Data"


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
    return {"bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree, "files": per}


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


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8 per-block-32 (decider_vision/export_decoder.py's recipe): SDPA / RoPE / norms /
    Embedding / Conv1d excluded. This graph has no lm_head (the parent's slot is an Identity), so nothing is
    excluded by name."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {},
    }


# Round 6 (Kev-4B): projection kinds a --fp16-kinds list names (code -> the leaf under model.layers.N).
KIND_LEAVES = {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "a": "linear_attn.in_proj_a",
               "b": "linear_attn.in_proj_b", "out": "linear_attn.out_proj", "q": "self_attn.q_proj",
               "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj",
               "up": "mlp.up_proj", "down": "mlp.down_proj"}
QSCHEMES = ("symmetric_with_clipping", "asymmetric", "symmetric")


def quant_config(block: int = 32, scheme: str = "symmetric_with_clipping", linear: bool = True,
                 embed: bool = False) -> dict:
    """linear_quant_config with the round-6 knobs: the linears' per-block size and qscheme, and the embedding table
    int8 per-block-32 symmetric_with_clipping along the hidden axis (axis 1) when `embed`. With the defaults it is
    linear_quant_config("int8") (asserted where it is used). `linear=False` keeps every Linear fp16 (the embedding
    alone, for the round-6 instrument's table dump)."""
    cfg = linear_quant_config("int8")
    cfg["global_config"]["op_state_spec"]["weight"]["granularity"]["block_size"] = int(block)
    cfg["global_config"]["op_state_spec"]["weight"]["qscheme"] = scheme
    if embed:
        cfg["module_type_configs"]["torch.nn.modules.sparse.Embedding"] = {
            "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                                         "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None}
    if not linear:
        cfg["global_config"] = None
        cfg["module_type_configs"]["torch.nn.modules.linear.Linear"] = None
    return cfg


def fp16_patterns(layers: list[int], kinds: list[str]) -> list[str]:
    """module_name_configs keys (fullmatch on the module name) that keep these layers' / kinds' linears fp16."""
    return ([rf"model\.layers\.{i}\..*" for i in layers]
            + [rf"model\.layers\.\d+\.{KIND_LEAVES[k].replace('.', chr(92) + '.')}" for k in kinds])


def readout_text(chunk: int, max_ctx: int) -> str:
    c = chunk
    return (f"one row per question, fresh zero states per row; a row of T ids runs as ceil(T / {c}) calls of 'main' "
            f"(static S = {c}): call k gets ids[{c}k : {c}k + {c}] with position_ids 0..{c}k+{c - 1}; the last call is "
            f"padded with <|endoftext|> ({PAD_ID}) and the hidden rows of the padded positions are discarded (causal: "
            f"they cannot reach a real position). The hidden rows [1, {c}, {HIDDEN}] of every call, concatenated and cut to "
            f"T, are the backbone's final-norm last_hidden_state [T, {HIDDEN}] the head reads. A row fits when "
            f"ceil(T / {c}) * {c} <= {max_ctx - 1} (the position dim's upper bound; the KV sequence dim is "
            f"allocated at {max_ctx}).")


def readout_text_dynamic(cap: int, max_ctx: int, padded_output: bool, q: int | None = None,
                         call_max: int | None = None) -> str:
    """Round 14's text (q and L None: the real ids only) or round 15's (calls of at most L ids, every one a multiple of q,
    the last one padded)."""
    out = (f"hidden [1, {cap}, {HIDDEN}] of which rows 0..s-1 are the call's (rows s..{cap - 1} are zero)" if padded_output
           else f"hidden [1, s, {HIDDEN}]")
    if q is None and call_max is None:
        return (f"one row per question, fresh zero states per row; a row of T ids runs as calls of 'main' with a dynamic "
                f"query length s in 2..{cap} and no padding: the ids are cut into ceil(T / {cap}) pieces of {cap} with the "
                f"remainder last, and a remainder of 1 takes one id from the piece before it (no 1-token call). A call "
                f"with s ids after p earlier ids gets position_ids 0..p+s-1 and returns {out}; the rows of every call, "
                f"concatenated, are the backbone's final-norm last_hidden_state [T, {HIDDEN}] the head reads. Shared "
                f"state: the row's state ids ([<state>] + render(state)) run once in pieces as above, the four states are "
                f"copied, and each question runs its branch (from <q>) on a copy, positions continuing after the state. "
                f"A row fits when T <= {max_ctx - 1} (the position dim's upper bound; the KV sequence dim is allocated at "
                f"{max_ctx}).")
    q = q or 1
    L = call_max or cap
    cut = (f"the last piece is padded with <|endoftext|> ({PAD_ID}) up to the next multiple of {q}; the hidden rows of the "
           f"padded positions are discarded (causal: they cannot reach a real position)" if q > 1 else
           "nothing is padded: a remainder of 1 takes one id from the piece before it (no 1-token call)")
    lengths = f"{q}, {2 * q}, ..., {L}" if q > 1 else f"2, 3, ..., {L}"
    return (f"one row per question, fresh zero states per row; a row of T ids runs as calls of 'main' (the graph takes a "
            f"query length s in 2..{cap}) of at most L = {L} ids (language.query_len_call_max), each a multiple of q = {q} "
            f"(language.query_len_multiple): the ids are cut into ceil(T / {L}) pieces of {L} with the remainder last, and "
            f"{cut}. A call with s ids after p earlier ids gets position_ids 0..p+s-1 and returns {out}; the rows of every "
            f"call, concatenated and cut to T, are the backbone's final-norm last_hidden_state [T, {HIDDEN}] the head "
            f"reads. Shared state: the first floor(Ls / {q}) * {q} ids of the row's state ids ([<state>] + render(state), "
            f"Ls ids) run once in pieces of {L} (no padding), the four states are copied, and each question runs the rest "
            f"of its row on a copy in pieces as above, positions continuing; those calls are cut at other places than a "
            f"direct run's, so the hidden rows can differ from a direct run's in the last bits. Every new call length pays "
            f"a one-time specialization on its first call in a process (a host may run the lengths {lengths} once from "
            f"zero states before its first request). A row fits when its padded end ceil(T / {q}) * {q} <= {max_ctx - 1} "
            f"(the position dim's upper bound; the KV sequence dim is allocated at {max_ctx}).")


def bundle_extra(chunk: int, max_ctx: int) -> dict:
    """Top-level blocks: provenance, and how a host turns the hidden rows into decisions."""
    d = {k: {"token": t, "id": i} for k, (t, i) in DELIM.items()}
    return {
        "decision": {
            "output": f"hidden [1, S, {HIDDEN}] per call: the final-norm hidden state at every position (no vocabulary "
                      "head in the graph)",
            "readout": readout_text(chunk, max_ctx),
            "row": {
                "source": f"the author's kev.model.encode + rows_of (row form, {KEV_CODE['repo']} tag "
                          f"{KEV_CODE['tag']}); conversion/kev/oracle_kev.py records the oracle's rows",
                "layout": "[<state>] + user_tokens(render(state)) + [<q>] + user_tokens(render(instructions)) + for "
                          "each option ([<opt>] + user_tokens(option_text) + [</opt>]) + [<decide>]",
                "delimiters": d,
                "pad": {"token": "<|endoftext|>", "id": PAD_ID},
                "bos": None,
                "positions": "0..L-1 (one causal row = the state ids then the question's branch ids)",
                "user_tokens": "re.sub(r'<\\|([A-Za-z0-9_]+)\\|>', r'<¦\\1¦>', text), then tokenized without special "
                               "tokens (user text can never produce a delimiter)",
                "tokenizer": "tokenizer/ (the base model's pinned tokenizer.json + tokenizer_config.json)",
                "readout_positions": {"decide": "the row's last token (<decide>)",
                                      "options": "each option's </opt> token, in option order"},
                "limits_author_serving": "state <= 65,536 tokens (the <state> token included), a row <= 73,728; this "
                                         "graph: one row <= the readout bound above",
            },
            "request": {
                "source": "kev.api.to_record (SystemOne request -> internal record)",
                "render": "None -> ''; str / int / float / bool -> str(v); array -> one line per item '- ' + "
                          "render(item) (nested items indented by two spaces per level); object -> one line per key "
                          "'key: value', or 'key:' then the nested render indented by two spaces when the value is "
                          "an object or an array",
                "state": "render(state)",
                "instructions": "render(instructions) (omitted -> '': the <q> token is followed by the options)",
                "option_text": "name if description is None or '' else 'name: ' + render(description)",
                "options": {"noul": ["option_text('no', criteria.false)", "option_text('yes', criteria.true)"],
                            "choice": "option_text(name, description) for each criteria entry, in criteria order",
                            "score": "render(level) for each level, in order"},
                "question_keys": {"noul": ["false", "true"], "choice": "the criteria names, in order",
                                  "score": "'0'..'n-1'"},
                "max_options": 255,
            },
            "head": {
                "files": [f"head/{f}" for f in HEAD_FILES],
                "formula": "z_k = ((k.weight @ h_opt_k + k.bias) . (q.weight @ h_decide + q.bias)) * 0.0625; "
                           "p = softmax_k(z / temperature)",
                "scale": 0.0625,
                "temperature": f"kev_head.json temperature ({TEMPERATURE}, the author's calibration in head.pt)",
                "code": f"kev.model.PointerHead.forward (q, k = nn.Linear({HIDDEN}, 256)); weights from head.pt, fp32",
                "dtype_of_record": "fp32 (the gate reads the graph's fp16 hidden through the fp32 head)",
            },
            "softmax": "per question, fp32, over that question's options, after the temperature",
            "response": {
                "shape": "SystemOne-compatible: {model, answers: {question_id: answer}, usage: {input_tokens, "
                         "output_tokens}, latency_ms} (kev.serve.Server._body)",
                "noul": "{type: 'noul', noul: p(true)}",
                "choice": "{type: 'choice', choice: the key of the first max p, confidence: (p_max - 1/K) / (1 - 1/K) "
                          "(1 when K = 1), probabilities: {key: p}}",
                "score": "{type: 'score', score: sum(i * p_i), legend: {key: render(level)}, probabilities: {key: p}, "
                         "confidence: max(0, 1 - E|level - mode| / D), D = mean |i - (L-1)/2| over the L levels, "
                         "mode = the first most likely level (1 when L = 1)}",
                "normalize": "the confidences normalize p to sum 1 first (all zeros -> uniform)",
                "rounding": "4 decimals (kev.api.round_prob)",
                "usage": "input_tokens = the encoded request's tokens (the author's packed form: state once + every "
                         "branch); output_tokens = tokens of json.dumps(answers) (kev.api.output_tokens)",
            },
        },
    }


def source_block() -> dict:
    return {
        "model_definition": "torch (coreai_models.models.macos.qwen3_5, conversion/kev/qwen3_5_kev_decoder.py)",
        "adapter": ADAPTER,
        "base": BASE,
        "merge": {"tool": f"scripts/merge_lora_checkpoint.py ({KEV_CODE['repo']} tag {KEV_CODE['tag']}, commit "
                          f"{KEV_CODE['commit']})",
                  "formula": "W + (B @ A) * alpha / r (alpha / r = 2.0) on every adapted weight, in fp32, written "
                             f"fp32 (12 target kinds, {ADAPTED_TENSORS} adapted tensors); every other weight unchanged",
                  "weights_sha256": WEIGHTS_SHA256,
                  "model_safetensors_sha256": (MERGED_FILES["model.safetensors"] if list(MERGED_FILES) == ["model.safetensors"]
                                               else MERGED_FILES),
                  "tensors": TENSORS},
        "hf_model_id": f"{HF_ID} (derived: a local HF-cache-form id for the merged checkpoint, not a Hub repository)",
        "hf_revision": WEIGHTS_SHA256,
        "tokenizer": {"repo": BASE["repo"], "revision": BASE["revision"], "files_sha256": TOKENIZER_FILES},
    }


def copy_tokenizer(out_dir: Path) -> dict:
    """The pinned base snapshot's tokenizer.json + tokenizer_config.json, verbatim (sha256 asserted)."""
    snap = Path(hf_snapshot(BASE["repo"], revision=BASE["revision"]))
    dst = out_dir / "tokenizer"
    dst.mkdir(parents=True, exist_ok=True)
    got = {}
    for name, want in TOKENIZER_FILES.items():
        src = snap / name
        if sha256_file(src) != want:
            sys.exit(f"{src}: sha256 {sha256_file(src)} != pinned {want}")
        shutil.copyfile(src, dst / name)
        got[name] = sha256_file(dst / name)
        if got[name] != want:
            sys.exit(f"copy of {name} differs from the snapshot")
    return got


def copy_head(out_dir: Path) -> dict:
    dst = out_dir / "head"
    dst.mkdir(parents=True, exist_ok=True)
    got = {}
    for name in HEAD_FILES:
        shutil.copyfile(HEAD_DIR / name, dst / name)
        got[name] = sha256_file(dst / name)
        if got[name] != sha256_file(HEAD_DIR / name):
            sys.exit(f"copy of {name} differs from {HEAD_DIR}")
    info = json.loads((dst / "kev_head.json").read_text())
    if info["head_safetensors_sha256"] != got["head.safetensors"]:
        sys.exit("kev_head.json names another head.safetensors")
    return got


KERNEL_IO = {"dynamic": None, "static": "slen", "static-pad": "pad"}   # round 16: --kernel-io -> static_io


def dynamic_query_record(args) -> dict:
    from qwen3_5_kev_decoder import QUERY_MIN

    return {"min": QUERY_MIN, "max": args.prefill_chunk, "trace_query_len": args.trace_query_len, "output": args.dyn_output}


def metadata_blocks(args, chunk: int, hidden: int, scan: dict, quant: dict | None) -> tuple[dict, dict]:
    """(language_extra, extra) of metadata.json: export() and --metadata-only write the same blocks."""
    language_extra = {"prefill_chunk": chunk, "static_inputs": [],
                      "output": f"hidden [1, {chunk}, {hidden}] fp16, every position"}
    extra = bundle_extra(chunk, args.max_ctx)
    if args.dynamic_query:   # round 14: no prefill_chunk (a host reading it would pad to the cap); the query range instead
        padded = args.dyn_output == "padded"
        language_extra = {"query_len_range": [scan["dynamic_query"]["min"], chunk]}
        if args.query_call_max is not None:   # round 15: the longest call a host makes (<= the graph's cap)
            language_extra["query_len_call_max"] = args.query_call_max
        if args.query_multiple is not None:   # round 15: the host's call lengths are multiples of it
            language_extra["query_len_multiple"] = args.query_multiple
        language_extra.update({"static_inputs": [],
                               "output": (f"hidden [1, {chunk}, {hidden}] fp16, rows 0..s-1 = the call's positions"
                                          if padded else f"hidden [1, s, {hidden}] fp16, every position")})
        extra["decision"]["readout"] = readout_text_dynamic(chunk, args.max_ctx, padded, args.query_multiple, args.query_call_max)
        extra["decision"]["output"] = language_extra["output"] + " (no vocabulary head in the graph)"
    if args.gdn_scan != "unroll":   # round 11; the unroll bundles keep the metadata they always had
        extra["gdn_scan"] = {("form" if k == "gdn_scan" else k): v for k, v in scan.items() if k != "linear_layers"}
    if quant:
        extra["compression"] = {
            "scheme": args.mode,
            "linear": f"int8 per-block-{args.quant_block} {args.quant_scheme} (weight only)",
            "excluded": ("fp16: the embedding table, " if not args.embed_int8 else "fp16: ")
                        + "the GDN conv1d, every RMSNorm (RMSNorm, RMSNormPlusOne, "
                        "RMSNormGated), SDPA, RoPE, the GDN A_log / dt_bias",
            "excluded_types": quant["excluded_types"],
            "int8_linear_modules": quant["int8_linear_modules"],
            "fp16_layers": quant["fp16_layers"],
            "fp16_linear_modules": quant["fp16_linear_modules"],
            "head": "none in the graph (the pointer head runs on the host, head/)",
        }
        if quant["fp16_kinds"] or args.quant_block != 32 or args.quant_scheme != "symmetric_with_clipping" or args.embed_int8:
            extra["compression"].update({
                "linear_spec": quant["linear"], "fp16_kinds": quant["fp16_kinds"],
                "fp16_patterns": quant["fp16_patterns"], "fp16_linear_params": quant["fp16_linear_params"],
                "int8_linear_params": quant["int8_params"],
                "embedding": ("int8 per-block-32 symmetric_with_clipping along the hidden axis (axis 1), dequantized "
                              "in the gather" if args.embed_int8 else "fp16"),
                "embedding_spec": quant["embedding"]})
    return language_extra, extra


def write_metadata(args, out_dir: Path, name: str, vocab_size: int, hidden: int, scan: dict, quant: dict | None) -> dict:
    """metadata.json (`_bundle.write_bundle_metadata` + kind decision-backbone + the provenance `source`), tokenizer/
    and head/ into `out_dir` -> what was written."""
    from _bundle import write_bundle_metadata

    language_extra, extra = metadata_blocks(args, args.prefill_chunk, hidden, scan, quant)
    write_bundle_metadata(
        out_dir, name, args.hf_id, vocab_size, args.max_ctx, revision=WEIGHTS_SHA256, mode=args.mode,
        functions=("main",), language_extra=language_extra, extra=extra,
    )
    # The graph returns hidden states, not logits: a generation loop must not pick this bundle up as an llm.
    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["kind"] = "decision-backbone"
    meta["source"] = source_block()
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"metadata_sha256": sha256_file(meta_path), "kind": meta["kind"], "tokenizer_sha256": copy_tokenizer(out_dir),
            "head_sha256": copy_head(out_dir)}


def metadata_only(args, out_dir: Path, name: str) -> dict:
    """Round 15: metadata.json, tokenizer/ and head/ beside an existing `<name>.aimodel` (no export, no model load).
    `--from-aimodel` clones a gated `.aimodel` into the bundle first (APFS `cp -c`, every file's sha256 equal to the
    source's). The config numbers (vocab, hidden, the GDN grid) come from the checkpoint's config.json."""
    from qwen3_5_kev_decoder import metal_kernel_name, metal_scan_record

    aimodel = out_dir / f"{name}.aimodel"
    rec: dict = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel)}
    if args.from_aimodel:
        src = Path(args.from_aimodel).expanduser().resolve()
        if aimodel.exists():
            sys.exit(f"{aimodel} exists: --from-aimodel never overwrites an asset")
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-c", "-R", str(src), str(aimodel)], check=True)   # APFS clone: no bytes copied
        want, got = tree_digest(src), tree_digest(aimodel)
        if want["files"] != got["files"]:
            sys.exit(f"the clone of {src} differs from its source: {got['files']} vs {want['files']}")
        rec["cloned_from"] = {"path": str(src), "tree_sha256": want["tree_sha256"], "files": want["files"],
                              "clone_tree_sha256": got["tree_sha256"]}
    if not (aimodel / "main.mlirb").exists():
        sys.exit(f"no {aimodel}/main.mlirb: --metadata-only writes beside an existing .aimodel")
    for p in ("metadata.json", "tokenizer", "head"):
        if (out_dir / p).exists():
            sys.exit(f"{out_dir / p} exists: --metadata-only never overwrites (remove it first, on purpose)")
    snap = Path(hf_snapshot(args.hf_id))
    cfg = json.loads((snap / "config.json").read_text())
    if int(cfg["hidden_size"]) != HIDDEN:
        sys.exit(f"{snap}/config.json hidden_size {cfg['hidden_size']} != {HIDDEN} (--model {args.model})")
    layer_types = cfg.get("layer_types") or []
    n_lin = sum(t == "linear_attention" for t in layer_types)
    if args.gdn_scan == "metal":
        static_io = KERNEL_IO[args.kernel_io]
        scan = metal_scan_record(n_lin, int(cfg["linear_num_value_heads"]), int(cfg["linear_value_head_dim"]),
                                 args.prefill_chunk, metal_kernel_name(args.prefill_chunk, args.dynamic_query, static_io),
                                 static_io)
        if args.dynamic_query:
            scan["dynamic_query"] = dynamic_query_record(args)
    elif args.gdn_scan == "chunk":
        scan = {"gdn_scan": "chunk", "linear_layers": n_lin,
                "chunk_doublings": max(1, math.ceil(math.log2(args.prefill_chunk)))}
    else:
        scan = {"gdn_scan": "unroll", "linear_layers": n_lin}
    mlirb = aimodel / "main.mlirb"
    rec.update(write_metadata(args, out_dir, name, int(cfg["vocab_size"]), int(cfg["hidden_size"]), scan, None))
    rec.update({"config": {"path": str(snap / "config.json"), "sha256": sha256_file(snap / "config.json")},
                "gdn_scan": args.gdn_scan, "gdn_scan_detail": scan,
                "query_len": (f"dynamic 2..{args.prefill_chunk}" if args.dynamic_query else args.prefill_chunk),
                "query_len_multiple": args.query_multiple, "query_len_call_max": args.query_call_max,
                "aimodel_files": {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*")) if p.is_file()},
                "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
                "main_hash_file": (aimodel / "main.hash").read_bytes().hex() if (aimodel / "main.hash").exists() else None,
                "tokenizer_verbatim_from_snapshot": None})
    rec["tokenizer_verbatim_from_snapshot"] = rec["tokenizer_sha256"] == TOKENIZER_FILES
    print(f"metadata written beside {aimodel} (main.mlirb {rec['main_mlirb']['bytes']:,} B sha256 "
          f"{rec['main_mlirb']['sha256']})", flush=True)
    return rec


def export(args, out_dir: Path, name: str) -> dict:
    import torch
    from qwen3_5_kev_decoder import (OUTPUT_NAMES, QUERY_MIN, Qwen3_5KevDecoder, metal_kernel_name, set_chunk_scan,
                                     set_metal_scan, set_unrolled_scan)

    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

    dtype = torch.float16
    chunk = args.prefill_chunk
    disk = {"before_load": disk_free()}
    weights_check = check_merged_files(args.hf_id)
    t0 = time.monotonic()
    print(f"loading {args.hf_id} text decoder fp16 (no lm_head) ...", flush=True)
    model = Qwen3_5KevDecoder.from_hf(args.hf_id, target_dtype=dtype, max_context_length=args.max_ctx)
    report = model.load_report
    if (report["unread_checkpoint_keys"] or report["module_tensors_not_in_checkpoint"] or report["meta_params"]
            or report["module_has_lm_head_weight"]):
        sys.exit(f"load mismatch: {json.dumps(report)}")
    n_lin = set_unrolled_scan(model)
    print(f"unrolled step scan on {n_lin} linear layers; load {json.dumps(report)}", flush=True)
    scan: dict = {"gdn_scan": "unroll", "linear_layers": n_lin}
    kernel = None
    if args.gdn_scan == "chunk":
        scan = set_chunk_scan(model, chunk)
        print(f"in-graph chunk scan on {scan['linear_layers']} linear layers, {scan['chunk_doublings']} doublings", flush=True)
    elif args.gdn_scan == "metal":
        static_io = KERNEL_IO[args.kernel_io]
        kernel, scan = set_metal_scan(model, chunk, name=metal_kernel_name(chunk, args.dynamic_query, static_io),
                                      static_io=static_io)
        print(f"fp32 Metal chunk kernel {scan['kernel']} (chunk_max {chunk}) on {scan['linear_layers']} linear layers"
              + (f", edges {args.kernel_io} ({static_io})" if static_io else ""), flush=True)
    cfg = model.config
    if args.dynamic_query:
        spec = model.build_dynamic_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, cap=chunk,
                                               trace_query_len=args.trace_query_len)
        if args.dyn_output == "padded":
            model.pad_output_to = chunk
        scan["dynamic_query"] = dynamic_query_record(args)
        print(f"dynamic query length {QUERY_MIN}..{chunk} (trace {args.trace_query_len}), output {args.dyn_output}",
              flush=True)
    else:
        spec = model.build_export_spec(dtype, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, query_len=chunk)
    if tuple(spec["output_names"]) != tuple(OUTPUT_NAMES):
        sys.exit(f"export spec output names {spec['output_names']} != {OUTPUT_NAMES}")
    t_loaded = time.monotonic()
    disk["after_load"] = disk_free()

    quant: dict | None = None
    if args.mode in ("int8lin", "int8mix"):
        import torch.nn.utils.parametrize as P
        from coreai_models.export.compression import quantize_pytorch_model

        keep = sorted(set(args.fp16_layers or []))
        kinds = [k for k in KIND_LEAVES if k in set(args.fp16_kinds or [])]
        r6 = bool(kinds) or args.quant_block != 32 or args.quant_scheme != "symmetric_with_clipping" or args.embed_int8
        if r6:
            cfg_q = quant_config(args.quant_block, args.quant_scheme, embed=args.embed_int8)
        else:
            cfg_q = linear_quant_config("int8")
        assert quant_config() == linear_quant_config("int8")
        for pat in fp16_patterns(keep, kinds):  # fullmatch on the module name
            cfg_q["module_name_configs"][pat] = None
        cfg_rec = json.loads(json.dumps(cfg_q))   # the quantizer rewrites the dict it is given: keep a copy
        print(f"quantizing (linear int8 per-block-{args.quant_block} {args.quant_scheme}; embedding "
              f"{'int8 per-block-32' if args.embed_int8 else 'fp16'}; conv1d, norms fp16; fp16 layers {keep}, fp16 kinds "
              f"{kinds}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        lin = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        int8 = [n for n, m in lin if P.is_parametrized(m, "weight")]
        fp16 = [n for n, m in lin if not P.is_parametrized(m, "weight")]
        # torch.nn.utils.parametrize renames a parametrized module's class (Embedding -> ParametrizedEmbedding)
        other = sorted({type(m).__name__.removeprefix("Parametrized") for n, m in model.named_modules()
                        if P.is_parametrized(m) and not isinstance(m, torch.nn.Linear)})

        def kept(n: str) -> bool:
            if not n.startswith("model.layers."):
                return False
            leaf = ".".join(n.split(".")[3:])
            return int(n.split(".")[2]) in keep or any(KIND_LEAVES[k] == leaf for k in kinds)

        wrong = [n for n in int8 if not n.startswith("model.layers.") or kept(n)] + \
                [n for n in fp16 if not n.startswith("model.layers.") or not kept(n)]
        other_want = ["Embedding"] if args.embed_int8 else []
        if wrong or other != other_want or not int8 or (args.mode == "int8mix" and not fp16):
            sys.exit(f"{args.mode}: quantized set differs from the request (fp16 layers {keep}, kinds {kinds}): "
                     f"{wrong[:8]}, other quantized types {other} (want {other_want})")

        def codes(m) -> int:
            return int(next(p for p in m.parametrizations["weight"] if hasattr(p, "quantized_data")).quantized_data.numel())

        # The quantizer rewrites the config it was given (dtype strings become torch dtypes): record a fresh copy.
        quant = {"linear": cfg_rec["global_config"]["op_state_spec"]["weight"],
                 "excluded_types": sorted(k for k, v in cfg_rec["module_type_configs"].items() if v is None),
                 "embedding": (cfg_rec["module_type_configs"]["torch.nn.modules.sparse.Embedding"]["op_state_spec"]["weight"]
                               if args.embed_int8 else None),
                 "fp16_layers": keep, "fp16_kinds": kinds, "fp16_layer_patterns": [rf"model\.layers\.{i}\..*" for i in keep],
                 "fp16_patterns": fp16_patterns(keep, kinds),
                 "int8_linear_modules": len(int8), "fp16_linear_modules": fp16,
                 "int8_params": int(sum(codes(m) for n, m in lin if n in set(int8))),
                 "fp16_linear_params": int(sum(m.weight.numel() for n, m in lin if n in set(fp16))),
                 "embedding_int8_params": (int(codes(model.model.embed_tokens)) if args.embed_int8 else 0),
                 "lm_head": "none in the graph (the tied table is the "
                            f"{'int8' if args.embed_int8 else 'fp16'} embedding)",
                 "seconds": time.monotonic() - t_loaded}
        print(f"quantized in {quant['seconds']:.0f}s: {len(int8)} int8 linears, {len(fp16)} fp16"
              f"{', embedding int8' if args.embed_int8 else ''}", flush=True)
    t_quantized = time.monotonic()

    # The loop-free path never calls the GatedDeltaUpdate composite; externalizing the class would mark the
    # uncalled submodules and then fail to find them in the traced program.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    from export_qwen38vl_pipelined import _install_externalize_dim_retry

    _install_externalize_dim_retry()
    shape_txt = f"dynamic S {QUERY_MIN}..{chunk}" if args.dynamic_query else f"static S={chunk}"
    print(f"exporting the hidden-output decoder (one function 'main', {shape_txt}, GDN scan {args.gdn_scan}) ...",
          flush=True)
    if kernel is not None:
        from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels

        prog = export_to_coreai_with_kernels(
            model,
            spec["reference_inputs"],
            custom_kernels=[kernel],
            dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            state_names=spec["state_names"],
            externalize_modules=specs,
        )
    else:
        prog = export_to_coreai(
            model,
            spec["reference_inputs"],
            dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            state_names=spec["state_names"],
            externalize_modules=specs,
        )
    t_converted = time.monotonic()
    print(f"converted in {t_converted - t_quantized:.0f}s; optimizing ...", flush=True)
    prog.optimize()
    t_exported = time.monotonic()
    print(f"optimized in {t_exported - t_converted:.1f}s", flush=True)
    disk["after_export"] = disk_free()

    if out_dir.exists():
        sys.exit(f"{out_dir} exists: an export never overwrites a bundle (remove it first, on purpose)")
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    t_saved = time.monotonic()
    disk["after_save"] = disk_free()
    meta = write_metadata(args, out_dir, name, cfg.vocab_size, cfg.hidden_size, scan, quant)
    mlirb = aimodel / "main.mlirb"
    rec = {"bundle": str(out_dir), "name": name, "aimodel": str(aimodel), "load_report": report, "weights_check": weights_check,
           "loopfree_unrolled_linear_layers": n_lin if args.gdn_scan == "unroll" else 0,
           "trace_kv_len": TRACE_KV_CACHE_SEQ_LEN, "max_ctx": args.max_ctx,
           "functions": ["main"], "query_len": (f"dynamic 2..{chunk}" if args.dynamic_query else chunk),
           "gdn_scan": args.gdn_scan, "gdn_scan_detail": scan,
           "output_names": list(OUTPUT_NAMES),
           **({"kernel_io": args.kernel_io, "kernel_copy_check": kernel_copy_check()} if KERNEL_IO[args.kernel_io] else {}),
           "quantization": quant,
           "seconds": {"load": t_loaded - t0, "quantize": t_quantized - t_loaded,
                       "export": t_converted - t_quantized, "optimize": t_exported - t_converted,
                       "export_optimize": t_exported - t_quantized, "save": t_saved - t_exported,
                       "total": time.monotonic() - t0},
           "du_aimodel": du(aimodel), "du_bundle": du(out_dir),
           "aimodel_files": {str(p.relative_to(aimodel)): p.stat().st_size for p in sorted(aimodel.rglob("*"))
                             if p.is_file()},
           "main_mlirb": {"bytes": mlirb.stat().st_size, "sha256": sha256_file(mlirb)},
           **meta, "tokenizer_verbatim_from_snapshot": meta["tokenizer_sha256"] == TOKENIZER_FILES, "disk": disk}
    print(f"bundle ready: {out_dir} ({rec['du_aimodel']}, main.mlirb {rec['main_mlirb']['bytes']:,} B sha256 "
          f"{rec['main_mlirb']['sha256']}, total {rec['seconds']['total']:.0f}s)", flush=True)
    return rec


def kernel_copy_check() -> dict:
    """Round 16: the static-edge kernel copy's MSL against the overlay's (equal after the length line), and its sha256."""
    from qwen3_5_gdn_metal_static import overlay_body_check

    return {**overlay_body_check(), "copy_file": str(HERE / "qwen3_5_gdn_metal_static.py"),
            "copy_sha256": sha256_file(HERE / "qwen3_5_gdn_metal_static.py")}


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
    subprocess.run(cmd, check=True)
    secs = time.monotonic() - t0
    disk["after"] = disk_free()
    if not target.exists():
        sys.exit(f"coreai-build produced no {target}")
    return target, secs, {"coreai_build": cb.stdout.strip(), "command": cmd, "disk": disk}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp16", choices=["fp16", "int8lin", "int8mix"])
    ap.add_argument("--model", default="kev-0.8b", choices=sorted(MODEL_PROFILES),
                    help="which merged checkpoint, provenance, tokenizer files, head and bundle name prefix")
    ap.add_argument("--fp16-layers", type=lambda s: [int(x) for x in s.split(",") if x != ""],
                    help="int8mix only: comma list of decoder layer indices whose linears stay fp16")
    ap.add_argument("--fp16-kinds", type=lambda s: [x for x in s.split(",") if x != ""],
                    help="int8mix only: comma list of projection kinds kept fp16 in every layer (qkv z a b out q k v o "
                         "gate up down)")
    ap.add_argument("--quant-block", type=int, default=32, choices=[16, 32],
                    help="int8 modes: the linears' per-block size along the input axis")
    ap.add_argument("--quant-scheme", default="symmetric_with_clipping", choices=list(QSCHEMES),
                    help="int8 modes: the linears' qscheme (coreai-opt names; 'asymmetric' = an affine zero point)")
    ap.add_argument("--embed-int8", action="store_true",
                    help="int8 modes: the embedding table int8 per-block-32 symmetric_with_clipping (axis 1)")
    ap.add_argument("--hf-id", help="the merged checkpoint's local id (default: the --model's)")
    ap.add_argument("--out-dir", default=str(LANE / "exports"),
                    help="bundles go to <out-dir>/bundles/<name>/, AOT assets to <out-dir>/bundles_aotc/")
    ap.add_argument("--name", help="override the generated bundle directory and asset name")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=16,
                    help="static S of the one function 'main' (the name gets _pf<S>); S=1 is not built")
    ap.add_argument("--gdn-scan", default="unroll", choices=["unroll", "chunk", "metal"],
                    help="round 11: the GDN recurrence inside the call (unroll = every earlier bundle; chunk = the "
                         "in-graph chunk scan; metal = the fp32 Metal chunk kernel); fp16 only")
    ap.add_argument("--dynamic-query", action="store_true",
                    help="round 14 (metal only): the query length is dynamic 2..--prefill-chunk (the cap)")
    ap.add_argument("--dyn-output", default="dynamic", choices=["dynamic", "padded"],
                    help="round 14: hidden [1, s, H] (dynamic) or a static [1, cap, H] with rows s.. zero (padded)")
    ap.add_argument("--trace-query-len", type=int, default=24, help="round 14: the query length the export traces")
    ap.add_argument("--query-multiple", type=int,
                    help="round 15 (--dynamic-query): metadata language.query_len_multiple, the host's call lengths are "
                         "multiples of it (absent = 1: the real ids only)")
    ap.add_argument("--query-call-max", type=int,
                    help="round 15 (--dynamic-query): metadata language.query_len_call_max, the longest call a host "
                         "makes (<= the cap; absent = the cap)")
    ap.add_argument("--kernel-io", default="dynamic", choices=list(KERNEL_IO),
                    help="round 16 (--dynamic-query): the GDN kernel's edges; dynamic = the overlay's kernel (default), "
                         "static = inputs padded to the cap + the length as an int32 input, static-pad = padded, every "
                         "step run")
    ap.add_argument("--skip-export", action="store_true", help="reuse the saved .aimodel")
    ap.add_argument("--metadata-only", action="store_true",
                    help="round 15: write metadata.json, tokenizer/ and head/ beside an existing <name>.aimodel (no "
                         "export, no model load; fp16 only)")
    ap.add_argument("--from-aimodel", help="round 15, --metadata-only: clone this .aimodel (APFS cp -c, sha256 "
                                           "checked) into the bundle first")
    ap.add_argument("--bundle-dir", help="round 15: the bundle directory (default <out-dir>/bundles/<name>)")
    ap.add_argument("--aot", action="store_true", help="compile the .aimodel for the Mac GPU (h16c)")
    ap.add_argument("--record", help="write the export / AOT record JSON here")
    args = ap.parse_args()
    select_model(args.model)
    args.hf_id = args.hf_id or HF_ID
    if (args.mode == "int8mix") != bool(args.fp16_layers or args.fp16_kinds):
        ap.error("--fp16-layers / --fp16-kinds: int8mix requires one of them and they apply to it only")
    bad_kinds = [k for k in (args.fp16_kinds or []) if k not in KIND_LEAVES]
    if bad_kinds:
        ap.error(f"unknown kinds {bad_kinds}")
    if args.mode == "fp16" and (args.quant_block != 32 or args.quant_scheme != "symmetric_with_clipping" or args.embed_int8):
        ap.error("--quant-block / --quant-scheme / --embed-int8 apply to the int8 modes only")
    if args.prefill_chunk < 2:
        ap.error("--prefill-chunk must be >= 2 (this graph has no S=1 function)")
    if args.gdn_scan != "unroll" and args.mode != "fp16":
        ap.error("--gdn-scan chunk / metal: fp16 only (round 11)")
    if args.dynamic_query and args.gdn_scan != "metal":
        ap.error("--dynamic-query: --gdn-scan metal only (round 14)")
    if (args.query_multiple is not None or args.query_call_max is not None) and not args.dynamic_query:
        ap.error("--query-multiple / --query-call-max: --dynamic-query only (a static-S bundle's calls are all S)")
    if args.query_call_max is not None and not 2 <= args.query_call_max <= args.prefill_chunk:
        ap.error(f"--query-call-max {args.query_call_max}: within the graph's 2..{args.prefill_chunk}")
    if args.query_multiple is not None:
        call_max = args.query_call_max or args.prefill_chunk
        if args.query_multiple < 1 or call_max % args.query_multiple:
            ap.error(f"--query-multiple {args.query_multiple}: a divisor of the call max {call_max}")
    if args.metadata_only and (args.mode != "fp16" or args.skip_export):
        ap.error("--metadata-only: fp16 only, and it replaces the export (no --skip-export)")
    if args.from_aimodel and not args.metadata_only:
        ap.error("--from-aimodel: with --metadata-only")
    if KERNEL_IO[args.kernel_io] and not args.dynamic_query:
        ap.error("--kernel-io static / static-pad: --gdn-scan metal --dynamic-query only (round 16)")

    form = "" if args.gdn_scan == "unroll" else f"_{args.gdn_scan}"
    width = (f"_{'dynpad' if args.dyn_output == 'padded' else 'dyn'}{args.prefill_chunk}" if args.dynamic_query
             else f"_pf{args.prefill_chunk}")
    if KERNEL_IO[args.kernel_io]:   # round 16
        from qwen3_5_kev_decoder import STATIC_IO_SUFFIX

        width += STATIC_IO_SUFFIX[KERNEL_IO[args.kernel_io]]
    name = args.name or f"{NAME_PREFIX}_{args.mode}{form}{width}"
    out_dir = Path(args.bundle_dir).expanduser().resolve() if args.bundle_dir else Path(args.out_dir) / "bundles" / name
    aot_dir = Path(args.out_dir) / "bundles_aotc"
    record: dict = {"mode": args.mode, "model": args.model, "name": name, "hf_id": args.hf_id, "weights_sha256": WEIGHTS_SHA256,
                    "prefill_chunk": args.prefill_chunk, "gdn_scan": args.gdn_scan, "argv": record_argv(sys.argv[1:]), "pid": os.getpid(),
                    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "script_sha256": sha256_file(Path(__file__).resolve()),
                    "module_sha256": sha256_file(HERE / "qwen3_5_kev_decoder.py")}
    if args.dynamic_query:   # round 14 / 15
        record.update({"dynamic_query": True, "query_len_multiple": args.query_multiple,
                       "query_len_call_max": args.query_call_max})
    if KERNEL_IO[args.kernel_io]:   # round 16
        record.update({"kernel_io": args.kernel_io,
                       "kernel_copy_sha256": sha256_file(HERE / "qwen3_5_gdn_metal_static.py")})
    if args.record and Path(args.record).exists():
        sys.exit(f"{args.record} exists: records are never overwritten")

    def save_record() -> None:
        if args.record:
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            Path(args.record).write_text(json.dumps(record, indent=1) + "\n")

    if args.metadata_only:
        record["metadata_only"] = metadata_only(args, out_dir, name)
        save_record()
    elif not args.skip_export:
        record["export"] = export(args, out_dir, name)
        save_record()
    if args.aot:
        aimodelc, secs, info = aot_compile(out_dir / f"{name}.aimodel", aot_dir)
        record["aot"] = {"aimodelc": str(aimodelc), "flags": AOT_FLAGS, "seconds": secs, **info,
                         "du_aimodelc": du(aimodelc), "digest": tree_digest(aimodelc)}
        print(f"asset: {aimodelc} (compile {secs:.1f} s, {record['aot']['du_aimodelc']}, "
              f"{record['aot']['digest']['bytes']:,} B, tree sha256 {record['aot']['digest']['tree_sha256']})",
              flush=True)
    record["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    save_record()
    if args.record:
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
