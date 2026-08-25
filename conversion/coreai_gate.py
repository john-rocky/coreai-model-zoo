#!/usr/bin/env python3
"""Bundle gate: an exported .aimodel decode bundle vs its fp32 eager oracle, N/N greedy.

This is the bar the zoo cards publish ("16/16 oracle"). Unlike the eager numerics gates
(which quantize the PyTorch model and compare to fp32 — they check the *quant recipe*),
this gate drives the *exported bundle* through the Core AI engine and compares its greedy
decode to the overlay model run in fp32. It therefore checks the **conversion** end to end
— which is exactly what the OS-27-beta-2 / coreai-torch-0.4.0 IR-location incident broke and
what re-converting with 0.4.1 fixes. A bundle that loads is not enough; this proves it still
speaks.

Both sides see the SAME pre-tokenized ids (no chat template) and free-run greedy. A first
divergence is a PASS only if the fp32 oracle's top-2 margin there is < 0.1 (a knife-edge
tie, fp16 class) — otherwise FAIL. EOS ends both sides.

Usage:
    python3 coreai_gate.py <bundle-dir> <hf-id> [--arch KEY] [--prompt "..."] [-n 16]
                           [--transcript out.json]

`--arch` is auto-detected from the bundle/repo name for the known families; pass it
explicitly for a new model that reuses an existing family's overlay.

Runnable from outside the maintainer's tree: `--python` / `ZOO_CONVERT_PYTHON` selects an
interpreter carrying the export overlay and `--runner` / `ZOO_LLM_RUNNER` selects the Core AI
CLI; a preflight says which one is missing instead of failing deep inside a subprocess
(`zoo_convert.py doctor` reports whether the overlay is set up).

`--transcript` writes the evidence — pinned revision, the exact input_ids, both sides'
generated tokens, the tie margins, the verdict, the environment. Publish it next to the card.
The asymmetry is deliberate: rebuilding the oracle costs an overlay interpreter and an fp32
checkpoint download, but re-running the *engine* side against a published transcript costs only
the bundle and llm-runner — so the expensive half is published once and the cheap half stays
reproducible by anyone.

Findings baked in here because they are documented nowhere else (2026-07-18 recovery):
  - The engine needs COREAI_CHUNK_THRESHOLD=1 + variant coreai-pipelined. This gate disables
    warmup to isolate the checked generation; the runtime patch makes default warmup honor a
    static-S=1 input descriptor instead of submitting a synthetic 256-token prefill.
  - `llm-runner --inference-engine-variant` help text is stale; the real values are
    auto / coreai-sequential / coreai-pipelined / static-shape.
  - The oracle steps S=1 but position_ids carries the FULL 0..t range each step
    (dynamic full-length positions); passing a single position produces plausible garbage.
  - Each overlay builds its model differently — see ARCH below.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent

# Per-arch overlay wiring, mirrored from each export_*.py. Each value is enough for the
# oracle subprocess (below) to rebuild the fp32 model and step it. `alias` maps extra
# model families onto an existing arch (same overlay, different weights).
ARCH = {
    "qwen3.5": {},
    "lfm2_5": {},
    "lfm2_5_vl": {},
    "granite": {},
    "youtu": {},
    "nanbeige": {},
    "lfm2_moe": {},
    "qwen3_6_moe": {},
    "muse_glimmer": {},
    # Plain dense qwen3 (Qwen3-0.6B and its finetunes). LAST on purpose:
    # detect_arch scans ARCH in order, so "qwen3.5"/"qwen3_6_moe" must be
    # reached first or every hybrid/MoE bundle would route to this branch.
    "qwen3": {},
}
# Dense Qwen3.6-27B reuses the qwen3.5 overlay; the 35B-A3B is MoE (own overlay). Match the
# MoE substrings before the generic qwen3.6->qwen3.5 fallback.
# lfm2_5_vl must beat the generic lfm2_5 substring: the VL checkpoint keeps its decoder
# under `model.language_model.`, so lfm2_from_hf finds no weights at all.
ALIASES = {"ornith": "qwen3.5", "lfm2_moe": "lfm2_moe", "a1b": "lfm2_moe",
           "vl_450m": "lfm2_5_vl", "vl-450m": "lfm2_5_vl", "vl_3b": "lfm2_5_vl",
           "vl-3b": "lfm2_5_vl",
           "35b_a3b": "qwen3_6_moe", "35b-a3b": "qwen3_6_moe", "a3b": "qwen3_6_moe",
           "qwen3_6": "qwen3.5", "qwen3.6": "qwen3.5",
           # Qwen3.8-27B: text_config byte-identical to Qwen3.6-27B (same 851-key
           # text weight map); dense, same overlay.
           "qwen3_8": "qwen3.5", "qwen3.8": "qwen3.5",
           # The Muse-Glimmer repo id is dashed, so the underscore key never
           # matches `hf_id` — only the bundle name. Route both spellings.
           "muse-glimmer": "muse_glimmer", "glimmer": "muse_glimmer",
           # S1-mini (Superwhisper): a Qwen3-0.6B finetune whose bundle name and
           # repo id name neither "qwen" nor "3", so nothing else can route it.
           "s1_mini": "qwen3", "s1-mini": "qwen3"}


def resolve_python(flag: str | None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    sibling = HERE.parent.parent / "coreai-models" / ".venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    return shutil.which("python3") or "python3"


def resolve_runner(flag: str | None = None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_LLM_RUNNER"):
        return env
    base = HERE.parent.parent / "coreai-models"
    for c in (base / ".build" / "release" / "llm-runner",
              base / ".build" / "out" / "Products" / "Release" / "llm-runner"):
        if c.exists():
            return str(c)
    return shutil.which("llm-runner") or "llm-runner"


def preflight(python: str, runner: str) -> None:
    """Fail with an actionable message instead of a confusing subprocess error.

    This gate is meant to be runnable from outside the maintainer's working tree — a
    reader checking a published bundle should not have to reverse-engineer why it broke.
    """
    if not (Path(runner).exists() or shutil.which(runner)):
        sys.exit(
            f"llm-runner not found: {runner}\n"
            "  It is the Core AI CLI that drives the bundle. Either build it from\n"
            "  apple/coreai-models, or point at yours:\n"
            "    --runner /path/to/llm-runner        (or ZOO_LLM_RUNNER=/path/to/llm-runner)"
        )
    probe = subprocess.run([python, "-c", "import coreai_models"], capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(
            f"the oracle interpreter cannot import coreai_models: {python}\n"
            "  The oracle rebuilds the reference model with the zoo's export overlay applied.\n"
            "  Point at an interpreter that has it:\n"
            "    --python /path/to/venv/bin/python   (or ZOO_CONVERT_PYTHON=/path/to/venv/bin/python)\n"
            "  `python3 conversion/zoo_convert.py doctor` reports whether yours is set up."
        )


def detect_arch(bundle: str, hf_id: str) -> str | None:
    name = Path(bundle).name.lower()
    hay = name + " " + hf_id.lower()
    # Aliases first: they carry the specific discriminators (a1b/a3b) that must beat the
    # generic family substring — e.g. "lfm2_5_8b_a1b" must route to lfm2_moe, not lfm2_5.
    a = next((v for sub, v in ALIASES.items() if sub in hay), None)
    if a:
        return a
    return next((k for k in ARCH if k in name or k.replace("_", "") in name or k in hf_id.lower()), None)


# The oracle runs in a child interpreter (the overlay venv). Kept as source text so the
# gate is a single file; each branch is a verbatim transcription of that model's export.
ORACLE_SRC = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch
CTX = 4096
arch, hf_id, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
revision = (sys.argv[6] or None) if len(sys.argv) > 6 else None
# Oracle weight dtype: fp32 is the strict ceiling, but a 35B in fp32 is ~140 GB — past
# most machines. fp16 (the export's own trace dtype) fits and is a valid conversion check.
FP32 = {"fp32": torch.float32, "fp16": torch.float16}[sys.argv[5] if len(sys.argv) > 5 else "fp32"]

def build(arch, hf_id):
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    if arch == "qwen3.5":
        from coreai_models.models.macos.qwen3_5 import Qwen3_5StatefulForCausalLM, build_decode_state
        try:
            m = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32, hf_config_attr="text_config")
        except Exception:
            m = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    elif arch == "lfm2_5":
        from coreai_models.models.macos.lfm2 import lfm2_from_hf, build_decode_state
        m = lfm2_from_hf(hf_id, target_dtype=FP32, stateful=True)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state"]
    elif arch == "lfm2_5_vl":
        # The TEXT CORE of an LFM2.5-VL checkpoint. The VLM bundle itself cannot be
        # gated here: llm-runner has no way to bind its image_embeds buffer, so the
        # image path is gated by _smoke/test_lfm25vl_suite_gate.py instead.
        from coreai_models.models.macos.lfm2 import build_decode_state
        from coreai_models.models.macos.lfm2_vl import lfm2_text_core_from_hf
        m = lfm2_text_core_from_hf(hf_id, target_dtype=FP32, fp32_attn_proj=False)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state"]
    elif arch == "granite":
        from coreai_models.models.macos.granite4h import Granite4HForCausalLMStateful, build_decode_state
        m = Granite4HForCausalLMStateful.from_hf(hf_id, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    elif arch == "youtu":
        from coreai_models.models.macos.youtu_absorbed import youtu_absorbed_from_hf, YoutuAbsorbedStatefulForCausalLM, build_absorbed_decode_state
        m = YoutuAbsorbedStatefulForCausalLM.from_causal_lm(youtu_absorbed_from_hf(hf_id, target_dtype=FP32))
        st = build_absorbed_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["kv_a","kv_b"]
    elif arch == "nanbeige":
        from transformers import AutoConfig
        from coreai_models.models.macos.llama import LlamaForCausalLM
        from coreai_models.models.macos.nanbeige import NanbeigeForCausalLM, create_cache_tensors
        from coreai_models.primitives.macos.cache import KVCache
        source_config = AutoConfig.from_pretrained(hf_id, revision=revision)
        model_classes = {"llama": LlamaForCausalLM, "nanbeige": NanbeigeForCausalLM}
        if source_config.model_type not in model_classes:
            raise ValueError(f"unsupported Nanbeige gate model_type: {source_config.model_type}")
        model_class = model_classes[source_config.model_type]
        # `from_hf_memory_efficient` takes no `revision` and rejects a local snapshot path
        # (it validates its argument as a repo id), so the oracle loads the source model's
        # default branch even when the recipe pins one. The gate reports that rather than
        # implying a pin it cannot honour — see `source_model.weights_pinned` in the transcript.
        m = model_class.from_hf_memory_efficient(
            hf_id, max_context_length=CTX, target_dtype=FP32
        )
        saved = m.config.max_position_embeddings; m.config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        if source_config.model_type == "nanbeige":
            k, v = create_cache_tensors(m.config, dtype=FP32)
        else:
            k, v = KVCache.create_cache_tensors(m.config, dtype=FP32)
        m.config.max_position_embeddings = saved
        st = {"k_cache": k, "v_cache": v}; order = ["k_cache","v_cache"]
    elif arch == "qwen3":
        # Plain dense qwen3 on the stock model definition (fused qkv + fused qk_norm
        # live in Qwen3ForCausalLM._mutate_state_dict). Tied head: the class re-ties
        # lm_head to embed_tokens after load_state_dict, so the oracle matches the
        # export's own weight sharing rather than a stale materialized copy.
        from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
        from coreai_models.primitives.macos.cache import KVCache
        m = Qwen3ForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32)
        saved = m.config.max_position_embeddings; m.config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        k, v = KVCache.create_cache_tensors(m.config, dtype=FP32)
        m.config.max_position_embeddings = saved
        st = {"k_cache": k, "v_cache": v}; order = ["k_cache","v_cache"]
    elif arch == "muse_glimmer":
        # Plain-KV dense text tower of a multimodal checkpoint: same shape as the
        # nanbeige/llama branch, but the text config and weights nest under
        # `text_config` / `model.language_model.` (handled by the model class).
        # fp32 is ~105 GB for the 26.4 B text tower, so run this with --dtype fp16.
        from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLM
        from coreai_models.primitives.macos.cache import KVCache
        m = MuseGlimmerForCausalLM.from_hf_memory_efficient(
            hf_id, max_context_length=CTX, target_dtype=FP32, hf_config_attr="text_config"
        )
        saved = m.config.max_position_embeddings; m.config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        k, v = KVCache.create_cache_tensors(m.config, dtype=FP32)
        m.config.max_position_embeddings = saved
        st = {"k_cache": k, "v_cache": v}; order = ["k_cache","v_cache"]
    elif arch == "lfm2_moe":
        from coreai_models.models.macos.lfm2_moe import lfm2_moe_from_hf, build_decode_state
        m = lfm2_moe_from_hf(hf_id, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state"]
    elif arch == "qwen3_6_moe":
        from coreai_models.models.macos.qwen3_5_moe import Qwen3_5MoeStatefulForCausalLM, build_decode_state
        try:
            m = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32, hf_config_attr="text_config")
        except Exception:
            m = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    else:
        raise SystemExit("unknown arch: " + arch)
    m.eval()
    for layer in getattr(m.model, "layers", []):
        if getattr(layer, "is_full", True) is False and hasattr(layer, "linear_attn"):
            layer.linear_attn.use_loopfree_step = True
    return m, [st[k] for k in order]

from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
except Exception:
    # Some repos (LFM2.5) name a tokenizer_class this transformers build lacks
    # ("TokenizersBackend"); load the fast tokenizer straight from tokenizer.json,
    # bypassing class resolution. config.eos_token_id still drives EOS below.
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(hf_id, "tokenizer.json", revision=revision)
    )
ids = tok(prompt, return_tensors="pt").input_ids.to(torch.int32)
model, states = build(arch, hf_id)
eos = set()
for e in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
    if isinstance(e, int): eos.add(e)
    elif isinstance(e, (list, tuple)): eos.update(int(x) for x in e)
gen, margins, cur = [], [], ids
with torch.no_grad():
    for t in range(ids.shape[1] + n - 1):
        out = model(cur[:, t:t+1], torch.arange(t+1, dtype=torch.int32).unsqueeze(0), *states)
        # Feed the prompt one token at a time; only START collecting once its last token
        # is in (t == len-1 predicts token #1). Dropping this guard emits prompt-position
        # predictions as output and corrupts the sequence.
        if t < ids.shape[1] - 1: continue
        row = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].float()
        nxt = int(row.argmax())
        if nxt in eos: break
        p = torch.softmax(row, dim=-1); top2 = torch.topk(p, 2).values
        margins.append(float(top2[0]-top2[1])); gen.append(nxt)
        if len(gen) >= n: break
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.int32)], dim=1)
print(json.dumps({"input_ids": ids[0].tolist(), "gen_ids": gen, "margins": margins,
                  "gen_text": tok.decode(gen, skip_special_tokens=False)}))
'''


def run_oracle(
    python: str,
    arch: str,
    hf_id: str,
    prompt: str,
    n: int,
    dtype: str,
    revision: str | None,
) -> dict:
    # Named so the child is findable. It runs from a temp file, so it used to appear in `ps` as
    # `python /var/folders/.../tmpXXXX.py` — which `pkill -f coreai_gate` does not match, so
    # killing the gate left the oracle running, holding the Hugging Face cache lock and stalling
    # every later attempt with no visible cause.
    with tempfile.NamedTemporaryFile("w", prefix="coreai_gate_oracle_", suffix=".py",
                                     delete=False) as f:
        f.write(ORACLE_SRC)
        script = f.name
    # The oracle downloads the source checkpoint in a child interpreter, so it needs the same
    # plain-HTTP transfer settings the conversion scripts get from `_paths` — Xet stalls at 0%
    # CPU on large shards, and hf_transfer has no reliable mid-file resume. Set explicitly
    # because this file is standalone by design and does not import `_paths`.
    env = {**os.environ,
           "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
           "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")}
    try:
        r = subprocess.run([python, script, arch, hf_id, prompt, str(n), dtype, revision or ""],
                           capture_output=True, text=True, cwd=tempfile.gettempdir(), env=env)
    finally:
        # `delete=False` is required so the child can read it; removing it here keeps a gate
        # run from leaving a file behind every time.
        Path(script).unlink(missing_ok=True)
    line = next((line for line in r.stdout.splitlines() if line.startswith("{")), None)
    if not line:
        sys.exit("ORACLE FAILED:\n" + r.stdout[-1000:] + r.stderr[-1000:])
    return json.loads(line)


def run_engine(runner: str, bundle: str, input_ids: list[int], n: int) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tokens": input_ids}, f)
        raw = f.name
    env = {"COREAI_CHUNK_THRESHOLD": "1", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([runner, "--model", bundle, "--raw-tokens", raw, "--max-tokens", str(n),
                        "--temperature", "0.0", "--inference-engine-variant", "coreai-pipelined",
                        "--warmup", "off"], capture_output=True, text=True, env=env)
    out = r.stdout
    try:
        body = out.split("Generating...", 1)[1].split("⏱", 1)[0]  # between banner and the ⏱ summary
    except IndexError:
        return None
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n\n"):
        body = body[:-2]
    return body


def finish(args, record: dict, result: str, line: str) -> NoReturn:
    """Print the verdict, optionally write the transcript, exit with the right status.

    The transcript exists so a reader does not have to take the gate on faith. Rebuilding the
    oracle is expensive (an overlay interpreter plus an fp32 checkpoint download); re-running
    the *engine* side against a published transcript is not — it needs the bundle, llm-runner,
    and the input_ids recorded here. That asymmetry is the point: the expensive half is
    published, the cheap half is reproducible by anyone.
    """
    record["schema"] = "coreai-gate-transcript/1"
    record["result"] = result
    record["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["environment"] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "runner": resolve_runner(args.runner),
    }
    record["recheck"] = {
        "engine_side_only": (
            "Download the bundle at the pinned revision, then feed the recorded input_ids to "
            "llm-runner with the protocol above (greedy, warmup off, coreai-pipelined, "
            "COREAI_CHUNK_THRESHOLD=1). The output must equal engine.gen_text. No oracle, no "
            "fp32 weights, no GPU beyond the one running the bundle."
        ),
        "full_gate": (
            "python3 conversion/coreai_gate.py <bundle> "
            f"{record['source_model']['hf_id']}"
            + (f" --revision {record['source_model']['revision']}"
               if record["source_model"]["revision"] else "")
            + f" --arch {record['arch']} --prompt {record['protocol']['prompt']!r}"
              f" -n {record['protocol']['max_new_tokens']}"
        ),
    }
    print(line)
    if args.transcript:
        path = Path(args.transcript)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"  transcript: {path}")
    sys.exit(0 if result.startswith("PASS") else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate an exported decode bundle against its fp32 oracle.")
    ap.add_argument("bundle")
    ap.add_argument("hf_id")
    ap.add_argument("--revision", help="immutable Hugging Face checkpoint revision")
    ap.add_argument("--arch", choices=list(ARCH))
    ap.add_argument("--prompt", default="The alphabet begins A, B, C, D, E, F,",
                    help="deterministic for the WHOLE continuation, not just token 1 — a "
                         "prompt that answers and then free-runs hits ties and gates nothing")
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--oracle-dtype", choices=["fp32", "fp16"], default="fp32",
                    help="fp32 = strict ceiling; fp16 for models too big for fp32 (e.g. 35B needs ~140 GB)")
    ap.add_argument("--python", help="overlay interpreter (default: $ZOO_CONVERT_PYTHON, "
                                     "else sibling coreai-models/.venv)")
    ap.add_argument("--runner", help="llm-runner executable (default: $ZOO_LLM_RUNNER, "
                                     "else sibling coreai-models build, else PATH)")
    ap.add_argument("--artifact", metavar="REPO@REV",
                    help="the published artifact this bundle is a copy of, e.g. "
                         "owner/repo@abcdef123456. Recorded in the transcript: a reader checking "
                         "the evidence needs to know which published bytes were gated, not just "
                         "a local directory name.")
    ap.add_argument("--transcript", metavar="PATH",
                    help="write the gate transcript as JSON: the pinned revision, the exact "
                         "input_ids, both sides' output, and the verdict. Publish it next to "
                         "the card — re-checking a published transcript needs only the bundle "
                         "and llm-runner, not the oracle.")
    args = ap.parse_args()

    arch = args.arch or detect_arch(args.bundle, args.hf_id)
    if not arch:
        sys.exit(f"no arch mapping for {args.bundle} — pass --arch")
    python = resolve_python(args.python)
    runner = resolve_runner(args.runner)
    preflight(python, runner)

    oracle = run_oracle(
        python, arch, args.hf_id, args.prompt, args.n, args.oracle_dtype, args.revision
    )
    engine = run_engine(runner, args.bundle, oracle["input_ids"], args.n)

    print("=== GATE:", Path(args.bundle).name, f"(arch={arch})")
    print("  prompt :", repr(args.prompt), "->", oracle["input_ids"])
    print("  oracle :", repr(oracle["gen_text"]))
    print("  engine :", repr(engine))

    ref, margins = oracle["gen_ids"], oracle.get("margins", [])
    record = {
        "result": None,
        "arch": arch,
        "bundle": Path(args.bundle).name,
        "artifact": args.artifact,
        # `revision` is what the recipe pinned for the *export*. `weights_pinned` says whether
        # this gate's oracle actually loaded that revision: the overlay's loader takes a repo id
        # and no revision, so it reads the default branch. Recorded rather than glossed, because
        # a transcript that implies a pin it did not honour is worse than one that admits the
        # gap — a reader can then decide how much the comparison is worth.
        "source_model": {"hf_id": args.hf_id, "revision": args.revision,
                         "weights_pinned": False,
                         "note": ("the oracle loads the source repo's default branch; the "
                                  "recipe's revision pins the export, not this comparison")},
        "protocol": {
            "prompt": args.prompt,
            "max_new_tokens": args.n,
            "greedy": True,
            "oracle_dtype": args.oracle_dtype,
            "engine_variant": "coreai-pipelined",
            "warmup": "off",
            "env": {"COREAI_CHUNK_THRESHOLD": "1"},
            "tie_rule": "a first divergence passes only where the oracle's top-2 margin < 0.1",
        },
        "input_ids": oracle["input_ids"],
        "oracle": {"gen_ids": ref, "gen_text": oracle["gen_text"], "top2_margins": margins},
        "engine": {"gen_text": engine},
        "match": {},
    }

    if engine is None:
        finish(args, record, "ERROR", "  RESULT: ERROR (engine produced no output)")
    if engine == oracle["gen_text"]:
        record["match"] = {"exact_prefix": len(ref), "of": len(ref), "first_divergence": None}
        finish(args, record, "PASS",
               f"  RESULT: PASS — token-for-token == {args.oracle_dtype} oracle")
    from transformers import AutoTokenizer
    try:
        tk = AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision)
    except Exception:
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tk = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(
                args.hf_id, "tokenizer.json", revision=args.revision
            )
        )
    eng_ids = tk(engine, add_special_tokens=False).input_ids
    d = next((i for i in range(min(len(eng_ids), len(ref))) if eng_ids[i] != ref[i]),
             min(len(eng_ids), len(ref)))
    tie = d < len(margins) and margins[d] < 0.1
    record["engine"]["gen_ids"] = eng_ids
    record["match"] = {"exact_prefix": d, "of": len(ref), "first_divergence": d,
                       "margin_at_divergence": margins[d] if d < len(margins) else None,
                       "tie": tie}
    print(f"  match  : {d}/{len(ref)} exact; first divergence at #{d}"
          + (f", margin {margins[d]:.4f}" if d < len(margins) else ""))
    if tie:
        finish(args, record, "PASS_TIE",
               f"  RESULT: PASS — diverges only at a top-2 tie (margin {margins[d]:.3f} < 0.1), fp16 class")
    finish(args, record, "FAIL",
           "  RESULT: FAIL — bundle diverges from the fp32 oracle at a decisive position")


if __name__ == "__main__":
    main()
