#!/usr/bin/env python3
"""coreai verify — gate an exported bundle against a Hugging Face oracle.

    coreai_verify.py <bundle-dir> [--hf-id X] [-n 16] [--prompt "..."] [--transcript out.json]

A bundle that loads is not a port. This drives the exported bundle and the reference model
over the same token ids and compares them token for token, with the two rules the notes
insist on:

  Validate the PROMPT before the bundle. Every oracle position must clear a top-2 margin
  floor (default 0.1) in fp32. A prompt with a 0.012-margin near-tie is a coin flip that
  healthy int8 noise flips and fp16 passes by luck — a 14/16 there gates nothing, in either
  direction. This refuses such a prompt instead of scoring against it.

  Judge a divergence by the oracle's margin, not by the fact of divergence. A first
  mismatch where the oracle's own top-2 gap is below the floor is a knife-edge tie (fp16
  class), not a failure. Above it, it is a failure.

  The STOP is a token too. When the oracle ends its turn (EOS) at a step whose margin
  clears the floor and the bundle keeps generating, that is a divergence at that step and
  it fails — a bundle that never halts scored 24/24 here for two months because the free-run
  prompts never reached EOS and a longer bundle output was read as "continued past the
  oracle's stop, nothing disagrees". `--chat no-think --prompt "1+1=?"` is the shape that
  reaches EOS within a few tokens; `--must-stop-within N` adds a plain halt check on top
  (the bundle must emit EOS before N tokens), which is what a Think-mode trace needs since
  its long trace rarely clears the margin floor at every position.

Two backends, chosen automatically:

  zoo    The bundle's family has a hand-transcribed fp32 oracle in the zoo's
         conversion/coreai_gate.py. That is the authority for those models and this
         delegates to it rather than reimplementing it.
  stock  Everything else — a bundle exported through Apple's stock recipe, whose reference
         is plain `transformers`. No overlay is involved, so no overlay is required.

The bundle side needs a driver, and which one is available is a property of the GRAPH:
a graph with a dynamic-shaped logits output cannot be executed by the Python runtime at
all, so it must go through llm-runner — which means the GPU, which on this beta means the
exclusive-GPU convention. Both facts are checked before anything long-running starts.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreai_doctor as doctor  # noqa: E402  (shared asset/graph readers)

# A gate prompt has to stay deterministic for the WHOLE continuation, not just its first
# token. "The capital of France is" — the prompt this repo recommended and shipped — answers
# "Paris." and then free-runs into a list of countries where the next one is a coin flip. It
# is rejected by the margin rule below on 2 of the 3 models measured (Qwen3-0.6B 0.0041,
# SmolLM2-360M 0.0172; gemma-3-1b-it clears at 0.3231, which is why it survived — it refuses
# or gates depending on the model under test). A counting sequence is worse: it fails on
# SmolLM2 (0.0289) and gemma-3 (0.0465). Reciting the alphabet is the only one of the three
# that clears everywhere: 0.9585 / 0.9351 / 0.8020. Measured 2026-08-01 at n=16, fp32.
DEFAULT_PROMPT = "The alphabet begins A, B, C, D, E, F,"
MARGIN_FLOOR = 0.1
GPU_LOCK = Path.home() / "code/coreai/_GPU_LOCK"  # workspace-level, not per-repo



# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def find_zoo_root() -> Path | None:
    """The zoo checkout, from wherever this file lives.

    A zoo checkout is the directory holding both `models/` and `conversion/`. Walking up
    finds it when the CLI lives inside the zoo; the explicit siblings cover the CLI living
    beside it. Both layouts are supported on purpose — placement is not settled.
    """
    for base in (HERE, *HERE.parents):
        if (base / "models").is_dir() and (base / "conversion").is_dir():
            return base
    for candidate in (Path.home() / "code/coreai/coreai-models-community",
                      HERE.parent / "coreai-model-zoo"):
        if (candidate / "models").is_dir():
            return candidate
    return None


def find_coreai_models() -> Path | None:
    """Apple's coreai-models checkout — a sibling of the zoo, or of the CLI."""
    roots = [find_zoo_root(), HERE, Path.home() / "code/coreai"]
    for base in filter(None, roots):
        for candidate in (base.parent / "coreai-models", base / "coreai-models"):
            if (candidate / "python").is_dir() or (candidate / ".venv").is_dir():
                return candidate
    return None


def resolve_python(flag: str | None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    if (base := find_coreai_models()) and (venv := base / ".venv/bin/python").exists():
        return str(venv)
    return shutil.which("python3") or "python3"


def resolve_runner(flag: str | None) -> str | None:
    if flag:
        return flag
    if env := os.environ.get("ZOO_LLM_RUNNER"):
        return env
    if base := find_coreai_models():
        for c in (base / ".build/release/llm-runner",
                  base / ".build/out/Products/Release/llm-runner"):
            if c.exists():
                return str(c)
    return shutil.which("llm-runner")


def zoo_root() -> Path | None:
    root = find_zoo_root()
    return root if root and (root / "conversion/coreai_gate.py").exists() else None


# ---------------------------------------------------------------------------
# Bundle facts
# ---------------------------------------------------------------------------


def bundle_facts(bundle: Path) -> dict:
    """Everything the plan depends on, read once."""
    meta = doctor.read_json(bundle / "metadata.json") or {}
    asset_name = (meta.get("assets") or {}).get("main")
    asset = bundle / asset_name if asset_name else None
    info = doctor.inspect_asset(asset) if asset and asset.exists() else None

    facts = {
        "bundle": str(bundle),
        "asset": str(asset) if asset else None,
        "hf_id": (meta.get("source") or {}).get("hf_model_id"),
        "revision": (meta.get("source") or {}).get("hf_revision"),
        "vocab_size": (meta.get("language") or {}).get("vocab_size"),
        "static_query": None, "dynamic_logits": None, "n_states": None,
    }
    if info:
        fns = (info.get("summary") or {}).get("functions") or []
        main = next((f for f in fns if f.get("name") == "main"), fns[0] if fns else {})
        facts["n_states"] = len(main.get("states") or [])
        ins = {i["name"]: i["type"] for i in main.get("inputs") or []}
        outs = {o["name"]: o["type"] for o in main.get("outputs") or []}
        if t := ins.get("input_ids"):
            _, shape = doctor.parse_type(t)
            facts["static_query"] = bool(shape) and all(d is not None for d in shape) and shape[-1] == 1
        logits = next((t for n, t in outs.items() if "logit" in n), None)
        if logits:
            _, lshape = doctor.parse_type(logits)
            facts["dynamic_logits"] = any(d is None for d in lshape)
            facts["logits_width"] = lshape[-1] if lshape else None
    return facts


def zoo_arch(bundle: Path, hf_id: str, root: Path | None) -> str | None:
    """Does the zoo's gate have a hand-transcribed oracle for this family?

    Imported from the gate rather than copied — a second copy of that table would drift,
    and being wrong here means gating a model against another family's reference.
    """
    if root is None:
        return None
    tree = ast.parse((root / "conversion/coreai_gate.py").read_text())
    tables: dict[str, dict] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], "id", None)
        if name in ("ARCH", "ALIASES"):
            tables[name] = ast.literal_eval(node.value)
    hay = bundle.name.lower() + " " + (hf_id or "").lower()
    # Aliases carry the specific discriminators (a1b / a3b) that must beat the generic
    # family substring, so they are matched first — same order as the gate itself.
    for sub, v in (tables.get("ALIASES") or {}).items():
        if sub in hay:
            return v
    return next((k for k in (tables.get("ARCH") or {})
                 if k in bundle.name.lower() or k.replace("_", "") in bundle.name.lower()
                 or k in (hf_id or "").lower()), None)


# ---------------------------------------------------------------------------
# The stock oracle (plain transformers), run in the venv interpreter
# ---------------------------------------------------------------------------

ORACLE_SRC = r'''
"""Teacher-free greedy reference for a stock-recipe bundle: plain transformers, no overlay.

Re-runs the whole prefix each step instead of carrying a KV cache. For a gate of a few
tens of tokens the cost is nothing, and it removes every cache-implementation difference
from the reference — the reference must be boring.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_id, prompt, n, dtype_name, revision, chat = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                                sys.argv[4], (sys.argv[5] or None), (sys.argv[6] or None))
dtype = {"fp32": torch.float32, "fp16": torch.float16}[dtype_name]

tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, torch_dtype=dtype).eval()

eos = set()
for e in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
    if isinstance(e, int): eos.add(e)
    elif isinstance(e, (list, tuple)): eos.update(int(x) for x in e)

if chat:
    # The chat-templated prompt, exactly as the source's own template renders it — one user
    # turn, generation prompt appended; "no-think" passes enable_thinking=False (a no-op on
    # templates without that switch). Both sides then see identical ids.
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"--chat: {hf_id}'s tokenizer has no chat template")
    kw = {} if chat == "think" else {"enable_thinking": False}
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
    if not hasattr(ids, "shape"):
        ids = ids["input_ids"]
else:
    ids = tok(prompt, return_tensors="pt").input_ids
input_ids = ids[0].tolist()
gen, margins = [], []
with torch.no_grad():
    for _ in range(n):
        row = model(ids).logits[0, -1].float()
        p = torch.softmax(row, dim=-1)
        top2 = torch.topk(p, 2).values
        nxt = int(row.argmax())
        margins.append(float(top2[0] - top2[1]))
        if nxt in eos:
            gen.append(nxt); break
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

# The cumulative decode after each step. llm-runner emits decoded TEXT and has no
# token-id output, so this is what makes a text-level comparison locate an exact STEP
# index — and therefore keeps the per-token margin rule meaningful — without
# re-tokenizing the bundle's output and hoping the round trip is faithful.
# The stop token is a step with a margin like any other, but the runner never prints it,
# so the text the bundle can be held to ends before it.
stopped = bool(gen) and gen[-1] in eos
text_ids = gen[:-1] if stopped else gen
prefixes = [tok.decode(gen[:i + 1], skip_special_tokens=False) for i in range(len(gen))]
if stopped:
    prefixes[-1] = tok.decode(text_ids, skip_special_tokens=False)

print("<<<JSON>>>" + json.dumps({
    "input_ids": input_ids, "chat": chat,
    "gen_ids": gen, "margins": margins, "stopped_on_eos": stopped,
    "gen_text": tok.decode(text_ids, skip_special_tokens=False),
    "step_prefixes": prefixes,
    "eos_ids": sorted(eos), "dtype": dtype_name,
}))
'''


def run_stock_oracle(python: str, hf_id: str, prompt: str, n: int, dtype: str,
                     revision: str | None, chat: str | None = None) -> dict:
    with tempfile.NamedTemporaryFile("w", prefix="coreai_verify_oracle_", suffix=".py",
                                     delete=False) as f:
        f.write(ORACLE_SRC)
        script = f.name
    env = {**os.environ,
           "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    try:
        r = subprocess.run([python, script, hf_id, prompt, str(n), dtype, revision or "",
                            chat or ""],
                           capture_output=True, text=True, env=env,
                           cwd=tempfile.gettempdir())
    finally:
        Path(script).unlink(missing_ok=True)
    line = next((x for x in r.stdout.splitlines() if x.startswith("<<<JSON>>>")), None)
    if not line:
        raise SystemExit("oracle failed:\n" + (r.stdout[-800:] + r.stderr[-1500:]))
    return json.loads(line[len("<<<JSON>>>"):])


# ---------------------------------------------------------------------------
# The bundle side
# ---------------------------------------------------------------------------


def driver_plan(facts: dict, runner: str | None) -> tuple[str | None, list[str]]:
    """(driver, blockers). Which driver can execute THIS graph, and may we use it."""
    blockers: list[str] = []
    if facts.get("dynamic_logits"):
        driver = "llm-runner"
        if not runner:
            blockers.append(
                "this graph has a dynamic-shaped logits output, which the Python runtime "
                "cannot execute at all — llm-runner is the only driver, and it was not found. "
                "Build it from apple/coreai-models or pass --runner / ZOO_LLM_RUNNER.")
        elif GPU_LOCK.exists():
            tag = GPU_LOCK.read_text().strip() or "(untagged)"
            blockers.append(
                f"llm-runner drives the graph on the GPU, and {GPU_LOCK} is held: {tag}. "
                f"The beta driver kernel-panics under parallel GPU load, so this stops rather "
                f"than contends. Re-run when the lock clears, or pass --ignore-gpu-lock if you "
                f"know it is stale.")
    else:
        driver = "python-cpu"
    return driver, blockers


def run_llm_runner(runner: str, bundle: Path, input_ids: list[int], n: int,
                   static_query: bool) -> tuple[str | None, str, int | None]:
    """Free-running greedy through the engine.

    Returns (generated text, raw stdout tail, generated-token count). llm-runner has no
    token-id output — it prints decoded text between its banner and the timing summary,
    which is what conversion/coreai_gate.py parses too. The count comes from that summary
    ("Generation: 17ms, 6 tokens"); it includes the stop token, so a count below the cap
    means the engine halted on EOS.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tokens": input_ids}, f)
        raw = f.name
    cmd = [runner, "--model", str(bundle), "--raw-tokens", raw,
           "--max-tokens", str(n), "--temperature", "0.0"]
    env = {"PATH": "/usr/bin:/bin"}
    if static_query:
        # A static S=1 graph cannot serve the default 256-token synthetic warmup, and any
        # multi-token prefill chunk is fatal. A dynamic graph wants neither of these.
        env["COREAI_CHUNK_THRESHOLD"] = "1"
        cmd += ["--inference-engine-variant", "coreai-pipelined", "--warmup", "off"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    finally:
        Path(raw).unlink(missing_ok=True)
    try:
        body = r.stdout.split("Generating...", 1)[1].split("⏱", 1)[0]
    except IndexError:
        return None, (r.stdout[-1200:] + r.stderr[-800:]), None
    m = re.search(r"Generation:\s*[\d.]+\s*ms,\s*(\d+)\s*tokens", r.stdout)
    return body.strip("\n"), r.stdout[-400:], (int(m.group(1)) if m else None)


def run_python_cpu(python: str, bundle: Path, asset: Path, input_ids: list[int],
                   n: int) -> tuple[list[int] | None, str]:
    """Greedy decode through the Core AI Python runtime on CPU.

    cpu_only is the documented choice for a numeric parity check — h16c fp16 compute on
    GPU/ANE gives noisier hidden-state diffs on high-magnitude activations. It also keeps
    this off the exclusive GPU.
    """
    src = r'''
import asyncio, json, sys
import numpy as np
from pathlib import Path
import coreai.runtime as rt

asset, ids_json, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = json.loads(ids_json)

async def main():
    m = await rt.AIModel.load(Path(asset), rt.SpecializationOptions.cpu_only())
    fn = m.load_function("main")
    out = []
    for _ in range(n):
        seq = ids + out
        res = await fn({
            "input_ids": rt.NDArray(np.array([seq], dtype=np.int32)),
            "position_ids": rt.NDArray(np.arange(len(seq), dtype=np.int32)[None, :]),
        })
        logits = next(v for k, v in res.items() if "logit" in k).numpy()
        out.append(int(np.asarray(logits)[0, -1].argmax()))
    print("<<<JSON>>>" + json.dumps(out))

asyncio.run(main())
'''
    with tempfile.NamedTemporaryFile("w", prefix="coreai_verify_cpu_", suffix=".py",
                                     delete=False) as f:
        f.write(src)
        script = f.name
    try:
        r = subprocess.run([python, script, str(asset), json.dumps(input_ids), str(n)],
                           capture_output=True, text=True, timeout=3600)
    finally:
        Path(script).unlink(missing_ok=True)
    line = next((x for x in r.stdout.splitlines() if x.startswith("<<<JSON>>>")), None)
    if line:
        return json.loads(line[len("<<<JSON>>>"):]), r.stdout[-400:]
    return None, (r.stdout[-1000:] + r.stderr[-1200:])


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------


def judge(oracle: dict, got: str, floor: float, bundle_tokens: int | None = None,
          cap: int | None = None) -> tuple[str, str]:
    """Compare the bundle's decoded text against the oracle's, per generation STEP.

    The oracle records its cumulative decode after every step, so the first step whose
    prefix the bundle's text stops matching IS the diverging step — which is what makes
    the per-token margin rule applicable to a text-only driver.

    The stop is judged like any other step. `bundle_tokens` (from the runner's summary,
    stop token included) below `cap` means the engine halted on EOS; the oracle records
    whether it did. Where one side stops and the other continues, the oracle's margin at
    that step decides, exactly as for a token disagreement.
    """
    prefixes, margins = oracle["step_prefixes"], oracle["margins"]
    n = len(prefixes)
    want = oracle["gen_text"]
    o_stop = bool(oracle.get("stopped_on_eos"))
    # None = the runner printed no count, so whether the engine halted is unknowable here;
    # the text-only rules below then apply and say so. An oracle EOS landing exactly on the
    # cap is equally undecidable (both sides produce `cap` tokens either way).
    b_stop: bool | None = None
    if bundle_tokens is not None and cap is not None:
        b_stop = bundle_tokens < cap
    if o_stop and cap is not None and len(oracle["gen_ids"]) >= cap:
        o_stop, undecidable = False, True
    else:
        undecidable = False
    if got == want:
        if o_stop and b_stop:
            return "PASS", (f"{n}/{n} token-exact, and the bundle stopped where the oracle did "
                            f"(EOS at step {n - 1}, oracle margin {margins[-1]:.3f}).")
        if o_stop and b_stop is False:
            m = margins[-1]
            if m >= floor:
                return "FAIL", (f"{n - 1}/{n} exact, but the oracle stops at step {n - 1} (EOS, "
                                f"margin {m:.4f}, above the {floor} floor) and the bundle did "
                                f"not — it ran to the {cap}-token cap.")
        note = ""
        if o_stop and b_stop is None:
            note = " The oracle stopped here; whether the bundle did is not verifiable (no token count from the runner)."
        if undecidable:
            note = f" The oracle's EOS landed on the {cap}-token cap itself; raise -n to gate the stop."
        return "PASS", f"{n}/{n} token-exact (decoded text identical to the fp32 oracle).{note}"

    if got.startswith(want):
        # The bundle produced everything the oracle did, then more.
        if o_stop:
            k, m = n - 1, margins[-1]
            extra = got[len(want):][:40]
            if m >= floor:
                return "FAIL", (f"{k}/{n} exact, then the oracle stops (EOS at step {k}, top-2 "
                                f"margin {m:.4f}, above the {floor} floor) and the bundle keeps "
                                f"generating: {extra!r}. A stop the model is sure of is a token "
                                f"like any other; not emitting it is a disagreement.")
            return "PASS", (f"{k}/{n} exact; the oracle stops at step {k} on a {m:.4f} margin, "
                            f"below the {floor} floor — a knife-edge stop, not a defect. Bundle "
                            f"continued: {extra!r}")
        return "PASS", (f"{n}/{n} exact; the bundle continued past the oracle's {cap}-token cap "
                        f"(different tokenization of the same text). Nothing disagrees.")

    if want.startswith(got):
        # The bundle's text is a strict prefix of the oracle's.
        matched = sum(1 for p in prefixes if got.startswith(p))
        if b_stop is True:
            i = min(matched, len(margins) - 1)
            m = margins[i]
            nxt = want[len(got):][:40]
            if m >= floor:
                return "FAIL", (f"{matched}/{n} exact, then the bundle stopped (EOS) where the "
                                f"oracle continues with {nxt!r} at a {m:.4f} margin, above the "
                                f"{floor} floor.")
            return "PASS", (f"{matched}/{n} exact; the bundle stopped where the oracle's next "
                            f"token sits on a {m:.4f} margin, below the floor — a tie.")
        return "PASS", (f"{matched}/{n} exact; the bundle's output is a prefix of the oracle's "
                        f"(same {cap}-token cap, different tokenization), so nothing they both "
                        f"produced disagrees.")

    # Longest step whose cumulative decode the bundle still reproduces.
    matched = 0
    for p in prefixes:
        if got.startswith(p):
            matched += 1
        else:
            break
    i = min(matched, len(margins) - 1)
    margin = margins[i] if margins else float("nan")
    tail_want = want[len(prefixes[matched - 1]) if matched else 0:][:40]
    tail_got = got[len(prefixes[matched - 1]) if matched else 0:][:40]
    if margin < floor:
        return "PASS", (
            f"first divergence at step {matched} sits on an oracle top-2 margin of "
            f"{margin:.4f}, below the {floor} floor — a knife-edge tie, fp16 class, not a "
            f"conversion defect. {matched}/{n} exact before it. "
            f"oracle {tail_want!r} vs bundle {tail_got!r}")
    return "FAIL", (
        f"diverges at step {matched} at an oracle top-2 margin of {margin:.4f}, above the "
        f"{floor} floor — a real disagreement, not fp16 noise. {matched}/{n} exact before it. "
        f"oracle {tail_want!r} vs bundle {tail_got!r}")


def validate_prompt(oracle: dict, floor: float) -> list[str]:
    weak = [(i, m) for i, m in enumerate(oracle["margins"]) if m < floor]
    if not weak:
        return []
    return [f"position {i}: top-2 margin {m:.4f}" for i, m in weak]


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--hf-id", default=None, help="oracle model (default: the bundle's own "
                                                  "source.hf_model_id)")
    ap.add_argument("-n", type=int, default=16, help="tokens to compare (default: 16)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--chat", choices=["think", "no-think"], default=None,
                    help="wrap --prompt in the tokenizer's chat template as one user turn "
                         "(no-think passes enable_thinking=False); the oracle's EOS then "
                         "becomes a gated step")
    ap.add_argument("--must-stop-within", type=int, default=None, metavar="N",
                    help="also require the bundle to emit EOS before N generated tokens on "
                         "the same prompt (a plain halt check; with -n 0 it is the only check)")
    ap.add_argument("--margin-floor", type=float, default=MARGIN_FLOOR)
    ap.add_argument("--oracle-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="fp32 is the strict ceiling; fp16 for models that will not fit")
    ap.add_argument("--python", default=None)
    ap.add_argument("--runner", default=None)
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--ignore-gpu-lock", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the plan and stop")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    if not (bundle / "metadata.json").exists():
        raise SystemExit(f"{bundle} is not a bundle directory (no metadata.json)")
    python, runner, root = resolve_python(args.python), resolve_runner(args.runner), zoo_root()

    facts = bundle_facts(bundle)
    hf_id = args.hf_id or facts["hf_id"]
    if not hf_id:
        raise SystemExit("no oracle: the bundle records no source.hf_model_id — pass --hf-id")
    arch = zoo_arch(bundle, hf_id, root)
    backend = "zoo" if arch else "stock"
    driver, blockers = driver_plan(facts, runner)
    if args.ignore_gpu_lock:
        blockers = [b for b in blockers if "GPU_LOCK" not in b and "_GPU_LOCK" not in b]

    if args.n <= 0 and args.must_stop_within is None:
        raise SystemExit("-n 0 skips the parity comparison; it only makes sense with "
                         "--must-stop-within N")

    print(f"coreai verify  {bundle}")
    print(f"oracle         {hf_id}" + (f" @ {facts['revision'][:8]}" if facts["revision"] else ""))
    if args.chat:
        print(f"prompt         {args.prompt!r} through the chat template ({args.chat})")
    print(f"graph          {facts['n_states']} states, "
          f"query {'static S=1' if facts['static_query'] else 'dynamic'}, "
          f"logits {'dynamic' if facts['dynamic_logits'] else 'static'}")
    print(f"backend        {backend}" + (f" (zoo arch {arch!r})" if arch else
                                         " (plain transformers reference — no overlay needed)"))
    print(f"driver         {driver}")
    print()

    if backend == "zoo":
        cmd = ["python3", "conversion/coreai_gate.py", str(bundle), hf_id, "--arch", arch,
               "-n", str(args.n)]
        print("--- DELEGATED " + "-" * 58)
        print(f"  The zoo carries a hand-transcribed fp32 oracle for {arch!r}. That is the")
        print("  authority for this family; reimplementing it here would be a second copy to")
        print("  drift. Run:")
        print(f"    $ cd {root} && " + " ".join(cmd))
        raise SystemExit(0)

    if blockers:
        print("--- CANNOT RUN THE BUNDLE SIDE " + "-" * 41)
        for b in blockers:
            print(f"  {b}")
        print()

    if args.plan:
        print("--- PLAN " + "-" * 63)
        print(f"  1. oracle: {hf_id} in {args.oracle_dtype}, greedy, {args.n} tokens, "
              f"prompt {args.prompt!r}" + (f" via chat template ({args.chat})" if args.chat else ""))
        print(f"  2. reject the prompt if any oracle position's top-2 margin < "
              f"{args.margin_floor}")
        print(f"  3. bundle: {driver}")
        print("  4. compare token-for-token; a divergence below the margin floor is a tie;")
        print("     the oracle's EOS is a step — a bundle that runs past it (or stops before")
        print("     it) is judged by the margin at that step like any other token")
        if args.must_stop_within is not None:
            print(f"  5. halt check: the bundle must emit EOS before {args.must_stop_within} "
                  f"tokens on the same prompt")
        raise SystemExit(2 if blockers else 0)

    print(f"running the oracle ({hf_id}, {args.oracle_dtype}) …", flush=True)
    oracle = run_stock_oracle(python, hf_id, args.prompt, args.n, args.oracle_dtype,
                              facts["revision"], args.chat)
    print(f"  prompt: {len(oracle['input_ids'])} ids")
    if args.n > 0:
        print(f"  oracle: {oracle['gen_ids']}" + ("  (stopped on EOS)" if oracle.get("stopped_on_eos") else ""))
        print(f"  text  : {oracle['gen_text']!r}")

    weak = validate_prompt(oracle, args.margin_floor) if args.n > 0 else []
    if weak:
        print()
        print("--- PROMPT REJECTED " + "-" * 52)
        print(f"  {len(weak)} oracle position(s) sit below the {args.margin_floor} top-2 margin "
              f"floor:")
        for w in weak[:6]:
            print(f"    {w}")
        print("  These are statistical coin flips: healthy int8 noise flips them and fp16 passes")
        print("  by luck, so a score against this prompt means nothing in either direction.")
        print("  Pick a more deterministic prompt and re-run. This is computable from the")
        print("  oracle alone, before any bundle exists.")
        raise SystemExit(3)
    if args.n > 0:
        print(f"  margins: min {min(oracle['margins']):.3f} — clears the {args.margin_floor} floor")

    if blockers:
        raise SystemExit(2)

    result, line, got, bundle_tokens = "SKIPPED", "no parity comparison requested (-n 0)", None, None
    if args.n > 0:
        print(f"\nrunning the bundle ({driver}) …", flush=True)
        if driver == "llm-runner":
            got, tail, bundle_tokens = run_llm_runner(runner, bundle, oracle["input_ids"], args.n,
                                                      bool(facts["static_query"]))
        else:
            got, tail = run_python_cpu(python, bundle, Path(facts["asset"]), oracle["input_ids"],
                                       args.n)
        if got is None:
            print("  the driver produced no output:")
            print("  " + tail.replace("\n", "\n  ")[:1200])
            raise SystemExit(4)
        print(f"  bundle: {got!r}" + (f"  ({bundle_tokens} tokens incl. stop)" if bundle_tokens is not None else ""))

        result, line = judge(oracle, got, args.margin_floor, bundle_tokens, args.n)
        print()
        print(f"--- {result} " + "-" * (66 - len(result)))
        print(f"  {line}")

    halt = None
    if args.must_stop_within is not None:
        cap = args.must_stop_within
        if driver != "llm-runner":
            raise SystemExit("--must-stop-within needs the engine (llm-runner); the python-cpu "
                             "driver has no stop logic to check")
        print(f"\nhalt check: {cap}-token cap on the same prompt …", flush=True)
        if args.n == cap and bundle_tokens is not None:
            h_text, h_tokens = got, bundle_tokens
        else:
            h_text, h_tail, h_tokens = run_llm_runner(runner, bundle, oracle["input_ids"], cap,
                                                      bool(facts["static_query"]))
            if h_text is None:
                print("  the driver produced no output:")
                print("  " + h_tail.replace("\n", "\n  ")[:1200])
                raise SystemExit(4)
        if h_tokens is None:
            halt = ("FAIL", "the runner printed no token count, so the stop cannot be verified")
        elif h_tokens < cap:
            halt = ("PASS", f"the bundle stopped after {h_tokens} tokens (cap {cap}); "
                            f"tail {h_text[-60:]!r}")
        else:
            halt = ("FAIL", f"the bundle ran to the {cap}-token cap without emitting EOS; "
                            f"tail {h_text[-60:]!r}")
        print(f"--- HALT {halt[0]} " + "-" * (61 - len(halt[0])))
        print(f"  {halt[1]}")

    if args.transcript:
        record = {
            "schema": "coreai-verify-transcript/1",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bundle": str(bundle), "backend": backend, "driver": driver,
            "oracle_model": hf_id, "oracle_revision": facts["revision"],
            "oracle_dtype": args.oracle_dtype, "prompt": args.prompt, "chat": args.chat,
            "margin_floor": args.margin_floor,
            "input_ids": oracle["input_ids"], "oracle_ids": oracle["gen_ids"],
            "oracle_stopped_on_eos": oracle.get("stopped_on_eos"),
            "oracle_text": oracle["gen_text"], "bundle_text": got,
            "bundle_tokens": bundle_tokens, "cap": args.n,
            "margins": oracle["margins"],
            "result": result, "verdict": line,
            "halt_check": ({"cap": args.must_stop_within, "result": halt[0], "verdict": halt[1]}
                           if halt else None),
            "environment": {"platform": platform.platform(),
                            "python": sys.version.split()[0], "runner": runner},
        }
        Path(args.transcript).write_text(json.dumps(record, indent=2))
        print(f"\n  transcript: {args.transcript}")

    if result == "FAIL" or (halt and halt[0] == "FAIL"):
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
