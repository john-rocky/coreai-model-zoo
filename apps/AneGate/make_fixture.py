"""Build the fp32 HF oracle fixture for the on-device static-bundle gate (AneGateRunner).

Run in the HF env:
    ~/.venv-coreml-llm-py312/bin/python ondevice/_ane_gate/make_fixture.py \
        --hf-id Qwen/Qwen3-0.6B --out fixtures/fixture.json [--poison natural:5]

For every prompt this does a plain-transformers fp32 greedy rollout, re-running the whole
prefix each step (no KV cache: the reference must be boring), and records per step:
  - expected token (fp32 argmax),
  - top-2 softmax-probability gap  (`margins`,       the coreai_verify floor definition),
  - top-2 raw-logit gap            (`margins_logit`, the knowledge/pipelined-engine.md wording),
  - whether the rollout ended on EOS.

Fixture rule (coreai_verify): validate the PROMPT before the bundle. A step whose oracle
margin is below the floor (default 0.1) is a knife-edge tie that gates nothing, so the
script refuses to write a fixture whose gated prompts carry one unless --allow-knife is
given (then the on-device gate simply excludes those positions). Both margin definitions
must clear the floor for a prompt to count as clean.

--poison NAME:K writes a deliberately wrong fixture (expected[K] of prompt NAME replaced by
the oracle's second-best token at that step, a margin-clear position) so the gate's ability
to go RED is demonstrated before its green is trusted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

FLOOR = 0.1

# The gate prompts. Same shapes coreai_verify settled on (2026-08-01 margins, fp32):
#   natural  raw text, free-runs to a deterministic list; clears the floor on Qwen3-0.6B
#            (0.9585 at n=16) where "The capital of France is" fails it (0.0041).
#   chat     a no-think turn that reaches EOS within a few tokens, so the STOP is gated
#            ("1+1=?" from coreai_verify stops on a 0.002-margin tie on Qwen3-0.6B; the
#            capital question stops at 0.993, min 0.891 over its 10 steps).
#   long     a >256-token raw prompt so the static engine has to walk prompt_opt_256_64
#            into the 512-context graphs (a short prompt only ever touches the 256 graphs).
DEFAULT_PROMPTS = [
    {"name": "natural", "text": "The alphabet begins A, B, C, D, E, F,", "chat": None, "n": 24},
    {"name": "chat", "text": "What is the capital of France?", "chat": "no-think", "n": 16},
    {"name": "long",
     "text": "Counting up: " + ", ".join(str(i) for i in range(1, 101)) + ",",
     "chat": None, "n": 16},
    # S5 (2026-09-16): a long free-form chat answer (~110 tokens incl. the stop on MiniCPM5-2B). The 3-prompt
    # gate (48 teacher-forced steps) passed a 4-bit 2B bundle that diverges from fp32 at step 0 of this
    # prompt with the oracle certain (margin 1.0) — knowledge/ane-vs-gpu-iphone-2026-09.md §4b. A long
    # answer has near-tie steps by nature, so this prompt is exempt from the knife-edge refusal
    # (knife_ok): the device gate excludes those positions in the TF sweep and stops judging the
    # free-run at the first sub-floor fork. Its cap is --chat-n (must cover the answer + EOS).
    {"name": "sky", "text": "Explain in a short paragraph why the sky is blue.", "chat": "no-think",
     "n": 16, "knife_ok": True},
]


def render_ids(tok, spec: dict) -> list[int]:
    if spec["chat"]:
        kw = {} if spec["chat"] == "think" else {"enable_thinking": False}
        ids = tok.apply_chat_template([{"role": "user", "content": spec["text"]}],
                                      add_generation_prompt=True, return_tensors="pt", **kw)
        if not hasattr(ids, "shape"):
            ids = ids["input_ids"]
        return [int(x) for x in ids[0]]
    return [int(x) for x in tok(spec["text"], return_tensors="pt").input_ids[0]]


def rollout(model, ids: list[int], n: int, eos: set[int]):
    gen, m_prob, m_logit, second = [], [], [], []
    cur = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(n):
            row = model(cur).logits[0, -1].float()
            p = torch.softmax(row, dim=-1)
            top2p = torch.topk(p, 2)
            top2l = torch.topk(row, 2)
            nxt = int(row.argmax())
            m_prob.append(float(top2p.values[0] - top2p.values[1]))
            m_logit.append(float(top2l.values[0] - top2l.values[1]))
            second.append(int(top2l.indices[1]))
            gen.append(nxt)
            if nxt in eos:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return gen, m_prob, m_logit, second


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "fixtures" / "fixture.json"))
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--only", default="", help="comma list of prompt names to include")
    ap.add_argument("--allow-knife", action="store_true",
                    help="write the fixture even if a gated step is below the floor")
    ap.add_argument("--poison", default="",
                    help="NAME:K — replace expected[K] of prompt NAME by the oracle's 2nd-best token")
    ap.add_argument("--chat-n", type=int, default=16,
                    help="rollout cap for every chat-templated prompt (chat, sky); 16 was the fixed cap of the "
                         "2026-09-15 fixtures, the sky prompt needs ~128 (S5)")
    ap.add_argument("--chat-msg", default=None,
                    help="override the chat prompt text (pick per model with explore_chat.py)")
    ap.add_argument("--must-stop-within", type=int, default=None,
                    help="extra halt rule for every chat prompt: device must emit EOS within N tokens")
    ap.add_argument("--assistant-prefix", default=None,
                    help="text appended (no special tokens) after the chat template's generation prompt, e.g. "
                         "an empty think block '<think>\\n\\n</think>\\n\\n' for a reasoning model whose template "
                         "has no enable_thinking switch, so the turn can reach EOS within the fixture's n")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_id, revision=args.revision, torch_dtype=torch.float32).eval()
    eos: set[int] = set()
    for e in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
        if isinstance(e, int):
            eos.add(e)
        elif isinstance(e, (list, tuple)):
            eos.update(int(x) for x in e)

    only = {s for s in args.only.split(",") if s}
    if args.chat_msg:
        for spec in DEFAULT_PROMPTS:
            if spec["name"] == "chat":
                spec["text"] = args.chat_msg
    for spec in DEFAULT_PROMPTS:
        if spec["chat"]:
            spec["n"] = args.chat_n
    prompts_out = []
    clean = True
    for spec in DEFAULT_PROMPTS:
        if only and spec["name"] not in only:
            continue
        ids = render_ids(tok, spec)
        if spec["chat"] and args.assistant_prefix:
            ids += [int(x) for x in tok(args.assistant_prefix, add_special_tokens=False).input_ids]
        gen, m_prob, m_logit, second = rollout(model, ids, spec["n"], eos)
        stopped = bool(gen) and gen[-1] in eos
        knife = [k for k in range(len(gen)) if min(m_prob[k], m_logit[k]) < args.floor]
        print(f"[{spec['name']}] prompt {len(ids)} ids, rollout {len(gen)} steps, "
              f"stopped_on_eos={stopped}")
        print(f"  text: {tok.decode(gen, skip_special_tokens=False)!r}")
        print(f"  margins_prob : {[round(m, 3) for m in m_prob]}  min={min(m_prob):.4f}")
        print(f"  margins_logit: {[round(m, 3) for m in m_logit]}  min={min(m_logit):.4f}")
        if knife:
            print(f"  KNIFE-EDGE steps (< {args.floor}): {knife}"
                  + ("  (knife_ok: excluded on device, does not block the fixture)" if spec.get("knife_ok") else ""))
            if not spec.get("knife_ok"):
                clean = False
        prompts_out.append({
            "name": spec["name"], "prompt_text": spec["text"], "chat": spec["chat"],
            "prompt_ids": ids, "expected_ids": gen,
            "expected_text": tok.decode(gen, skip_special_tokens=False),
            "margins": [round(m, 6) for m in m_prob],
            "margins_logit": [round(m, 6) for m in m_logit],
            "second_best_ids": second,
            "stopped_on_eos": stopped, "eos_ids": sorted(eos),
            "must_stop_within": args.must_stop_within if spec["chat"] else None,
            "knife_ok": bool(spec.get("knife_ok", False)),
            "assistant_prefix": args.assistant_prefix if spec["chat"] else None,
        })

    poison = None
    if args.poison:
        name, k = args.poison.split(":")
        k = int(k)
        p = next(p for p in prompts_out if p["name"] == name)
        if min(p["margins"][k], p["margins_logit"][k]) < args.floor:
            raise SystemExit(f"--poison {args.poison}: step {k} is a knife-edge, pick a clear one")
        wrong = p["second_best_ids"][k]
        poison = {"prompt": name, "step": k, "was": p["expected_ids"][k], "now": wrong}
        p["expected_ids"][k] = wrong
        p["expected_text"] = tok.decode(p["expected_ids"], skip_special_tokens=False)
        print(f"POISONED {name}[{k}]: {poison['was']} -> {wrong} "
              f"(oracle margin {p['margins'][k]:.3f}); this fixture MUST fail on device")

    if not clean and not args.allow_knife and not poison:
        raise SystemExit(f"MARGIN FAIL: a gated step is below the {args.floor} floor; "
                         "change the prompt or pass --allow-knife (positions are then excluded)")

    payload = {
        "hf_id": args.hf_id, "revision": args.revision, "dtype": "fp32",
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "floor": args.floor, "margin_definition": "top-2 softmax-probability gap (coreai_verify); "
                                                  "margins_logit = raw-logit gap, both >= floor at fixture time",
        "poison": poison, "prompts": prompts_out,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} ({'POISONED' if poison else 'clean' if clean else 'with knife-edges'})")


if __name__ == "__main__":
    main()
