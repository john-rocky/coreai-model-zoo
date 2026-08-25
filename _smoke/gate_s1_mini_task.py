#!/usr/bin/env python3
"""Task gate for an exported S1-mini bundle: does it still NORMALIZE?

`conversion/coreai_gate.py` proves the conversion is faithful token-for-token on a
free-run continuation. That is necessary and not sufficient here: S1-mini does one
job, and what a reader wants to know about an int4 build is whether the JOB
survives — not whether an alphabet continuation does. So this gate drives the
bundle through the model's real input format (system prompt + control line + raw
transcript, `enable_thinking=False`) across the card's documented axes, and
compares greedy output to a reference.

TWO reference columns, and the difference between them is the whole point:

  * `reference` — the same cases run through plain `transformers` on the RELEASED
    weights (bf16, greedy). This is the VERDICT. It is independent of the bundle
    under test (it comes from the source checkpoint), which is what a conversion
    gate needs; a fixture regenerated from the artifact could never fail.

  * `card` — the strings printed on the model card. INFORMATIONAL ONLY. Measured
    2026-08-25: the released weights themselves reproduce 9 of these 14 (the
    misses are stylistic — a retained "So"/"Hmm" opener, "March 3rd" for
    "March 3", and `Structure: lists` staying prose on the card's own list
    example). The card's table appears to predate the released checkpoint, so
    gating on it would fail every faithful build. Recorded so a reader can see
    the disagreement instead of discovering it.

Build the reference once, then gate each bundle against it:
    python3 _smoke/gate_s1_mini_task.py --reference --out _smoke/s1_mini_task_ref.json
    python3 _smoke/gate_s1_mini_task.py <bundle-dir> --out artifacts/<name>_task.json

Exit 0 only if every case matches the reference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REF = HERE / "s1_mini_task_ref.json"

HF_ID = "superwhisper/s1-mini"
SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)

DOG = "hmm im gonna be late theres a cute dog outside i cant just walk past him"
TRIP = ("so for the trip we need to pack sunscreen and then also a first aid kit "
        "and um chargers for everything")
SARAH = ("hey sarah just wanted to follow up on the proposal can you send the "
         "numbers by end of week thanks john")

# (label, styling, structure, context, transcript, card_expected)
CASES = [
    # --- card Examples table (semi-formal / prose / general) ---
    ("ex.report", "semi-formal", "prose", "general",
     "so um i need to like send the the report by uh friday no wait make that thursday",
     "I need to send the report by Thursday."),
    ("ex.number", "semi-formal", "prose", "general",
     "i think the answer is forty two no sorry forty three",
     "I think the answer is 43."),
    ("ex.time", "semi-formal", "prose", "general",
     "let's meet at half past two tomorrow uh actually make it three fifteen p m",
     "Let's meet at 3:15pm tomorrow."),
    ("ex.money", "semi-formal", "prose", "general",
     "the invoice came to twenty three thousand four hundred and fifty dollars and "
     "it's due on march third twenty twenty six",
     "The invoice came to $23,450, and it's due on March 3, 2026."),
    ("ex.email-addr", "semi-formal", "prose", "general",
     "send it to support at superwhisper dot com",
     "Send it to support@superwhisper.com."),
    # Pure filler -> empty string. The case a quantized build is most likely to break
    # by emitting *something*, and the one no free-run continuation gate would catch.
    ("ex.filler", "semi-formal", "prose", "general", "um", ""),
    # --- Styling: one input under all four registers ---
    ("style.casual", "casual", "prose", "general", DOG,
     "hmm im gonna be late. theres a cute dog outside. i cant just walk past him"),
    ("style.semi-casual", "semi-casual", "prose", "general", DOG,
     "hmm, I'm gonna be late. there's a cute dog outside. I can't just walk past him"),
    ("style.semi-formal", "semi-formal", "prose", "general", DOG,
     "I'm going to be late. There's a cute dog outside. I can't just walk past him."),
    ("style.formal", "formal", "prose", "general", DOG,
     "I am going to be late. There is a cute dog outside. I cannot just walk past him."),
    # --- Structure ---
    ("struct.prose", "semi-formal", "prose", "general", TRIP,
     "So for the trip, we need to pack sunscreen and then also a first aid kit and "
     "chargers for everything."),
    ("struct.lists", "semi-formal", "lists", "general", TRIP,
     "So for the trip, we need to pack:\n- Sunscreen\n- A first aid kit\n"
     "- Chargers for everything"),
    # --- Context ---
    ("ctx.general", "semi-formal", "prose", "general", SARAH,
     "Hey Sarah, just wanted to follow up on the proposal. Can you send the numbers "
     "by end of week? Thanks, John."),
    ("ctx.email", "semi-formal", "prose", "email", SARAH,
     "Hey Sarah,\n\nJust wanted to follow up on the proposal. Can you send the "
     "numbers by end of week?\n\nThanks,\nJohn"),
]


def resolve_runner(flag: str | None) -> str:
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


def build_ids(tok, styling: str, structure: str, context: str, transcript: str) -> list[int]:
    """The card's exact input format — verified byte-identical to the literal prompt the
    card documents. `enable_thinking=False` is not optional: with thinking on the model
    emits an empty <think> block and stops, so every case returns "" and the gate goes
    uniformly, plausibly green-on-empty."""
    control = f"[Styling: {styling}] [Structure: {structure}] [Context: {context}]"
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{control}\n{transcript}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]


def run_engine(runner: str, bundle: str, ids: list[int], max_tokens: int) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tokens": ids}, f)
        raw = f.name
    try:
        r = subprocess.run(
            [runner, "--model", bundle, "--raw-tokens", raw, "--max-tokens", str(max_tokens),
             "--temperature", "0.0", "--inference-engine-variant", "coreai-pipelined",
             "--warmup", "off"],
            capture_output=True, text=True,
            env={"COREAI_CHUNK_THRESHOLD": "1", "PATH": "/usr/bin:/bin"},
        )
    finally:
        Path(raw).unlink(missing_ok=True)
    try:
        body = r.stdout.split("Generating...", 1)[1].split("⏱", 1)[0]
    except IndexError:
        return None
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n\n"):
        body = body[:-2]
    return body


_REF_MODEL = None


def run_reference(hf_id: str, revision: str | None, tok, ids: list[int], max_tokens: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM

    global _REF_MODEL
    if _REF_MODEL is None:
        _REF_MODEL = AutoModelForCausalLM.from_pretrained(
            hf_id, revision=revision, dtype="auto").eval()
    inp = torch.tensor([ids])
    with torch.no_grad():
        out = _REF_MODEL.generate(inp, attention_mask=torch.ones_like(inp),
                                  max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", nargs="?", help="bundle dir to gate (omit with --reference)")
    ap.add_argument("--reference", action="store_true",
                    help="run the cases through HF transformers on the released weights and "
                         "write the reference the gate compares against")
    ap.add_argument("--ref", default=str(DEFAULT_REF), help="reference JSON to gate against")
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision")
    ap.add_argument("--runner")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", metavar="PATH", help="write the per-case record")
    args = ap.parse_args()

    if not args.reference and not args.bundle:
        ap.error("pass a bundle dir, or --reference to build the reference")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision)

    ref_map: dict[str, str] = {}
    if not args.reference:
        ref_path = Path(args.ref)
        if not ref_path.exists():
            sys.exit(f"reference not found: {ref_path}\n"
                     f"  build it: python3 {Path(__file__).name} --reference --out {ref_path}")
        ref_map = {c["case"]: c["got"] for c in json.loads(ref_path.read_text())["cases"]}
        runner = resolve_runner(args.runner)
        if not (Path(runner).exists() or shutil.which(runner)):
            sys.exit(f"llm-runner not found: {runner}  (--runner / ZOO_LLM_RUNNER)")
    else:
        runner = None

    who = "transformers reference (released weights, bf16)" if args.reference else Path(args.bundle).name
    print(f"=== S1-mini TASK GATE: {who}")

    rows, passed, card_ok = [], 0, 0
    for label, sty, struct, ctx, src, card in CASES:
        ids = build_ids(tok, sty, struct, ctx, src)
        got = (run_reference(args.hf_id, args.revision, tok, ids, args.max_tokens)
               if args.reference else run_engine(runner, args.bundle, ids, args.max_tokens))
        got_s = "" if got is None else got.strip()
        matches_card = got_s == card.strip()
        card_ok += matches_card
        if args.reference:
            ok = True  # building the reference, nothing to fail against
        else:
            ok = got_s == (ref_map.get(label) or "").strip()
        passed += ok
        rows.append({"case": label, "styling": sty, "structure": struct, "context": ctx,
                     "input": src, "got": got, "card_expected": card,
                     "matches_card": matches_card,
                     "matches_reference": None if args.reference else ok,
                     "prompt_tokens": len(ids)})
        mark = "REF " if args.reference else ("PASS" if ok else "FAIL")
        print(f"  [{mark}] {label:18s} ({len(ids)} tok){'' if matches_card else '  [card differs]'}")
        if not args.reference and not ok:
            print(f"         reference: {(ref_map.get(label) or '').strip()!r}")
            print(f"         bundle   : {got_s!r}")

    if args.reference:
        print(f"  reference built: {len(CASES)} cases; {card_ok}/{len(CASES)} agree with the card")
    else:
        print(f"  RESULT: {passed}/{len(CASES)} match the released-weights reference"
              f"  ({card_ok}/{len(CASES)} agree with the card, informational)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"schema": "s1-mini-task-gate/2", "subject": who,
             "hf_id": args.hf_id, "revision": args.revision,
             "verdict_reference": None if args.reference else str(Path(args.ref)),
             "protocol": {"greedy": True, "max_new_tokens": args.max_tokens,
                          "enable_thinking": False,
                          "engine": None if args.reference else "coreai-pipelined, warmup off, "
                                                               "COREAI_CHUNK_THRESHOLD=1"},
             "passed": None if args.reference else passed, "of": len(CASES),
             "card_agreement": card_ok, "cases": rows}, indent=2, ensure_ascii=False) + "\n")
        print(f"  record: {args.out}")
    sys.exit(0 if args.reference or passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
