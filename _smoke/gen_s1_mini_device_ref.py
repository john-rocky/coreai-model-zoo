#!/usr/bin/env python3
"""Emit the PipelinedBench ModelSpec token arrays for S1-mini.

The device runs the SAME int8lin bundle the Mac does, so the device gate's claim is
`device == Mac == fp32 HF` — the last link is already proven by `conversion/coreai_gate.py`
(16/16 token-exact) and `_smoke/gate_s1_mini_task.py` (13/14), so what the bench must
reproduce is the Mac ENGINE's greedy on this bundle.

Both cases are the model's real input format (system prompt + control line, thinking off),
not a free-run continuation — the same reason this port has a task gate at all. The `oracle`
case is deliberately the currency/date one: it is where int4 corrupted digits, so it is the
case with something to lose.

llm-runner prints text, not ids, so the ids come back through the tokenizer. That round trip
is verified here (`decode(encode(text)) == text`) rather than assumed; an unverified round
trip would silently bake a tokenizer artifact into the device's expected array and read on
device as a model mismatch.

Usage: python3 _smoke/gen_s1_mini_device_ref.py <bundle-dir> [--runner PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gate_s1_mini_task import HF_ID, build_ids, resolve_runner, run_engine  # noqa: E402

# The two long transcripts are verbatim here rather than in a fixture file so this script
# stays the single source for every array the bench carries. `mid` is the ModelSpec `nat` the
# device gate ran; `long` is the ceiling probe that located the iOS 1024 cap. A short case
# cannot reach the shape that matters on iOS, which is why `mid` replaced the original `nat`.
MID = "so um okay let me just get all of this down before i forget it we had the quarterly review this morning and uh there were basically three things that came out of it first the pipeline numbers are actually better than we thought like we closed nineteen deals in q three not seventeen the discrepancy was uh two deals that got booked on october first instead of september thirtieth second thing is the churn rate went from four point two percent down to three point one percent which is uh the lowest it's been since we started tracking it in twenty twenty four sarah thinks that's because of the onboarding changes we shipped in august and i think she's right third and this is the one that i think we need to actually talk about is uh hiring we budgeted for six engineers this half and we've hired two the pipeline is thin i think we've got maybe four candidates at final stage and two of them we're probably going to lose to bigger offers so um my read is we should either raise the bands or we should accept that we're going to be at four not six and if we're at four then the roadmap needs to change because the mobile work assumed five people minimum oh also i need to follow up with legal about the"

LONG = "so um okay let me just get all of this down before i forget it we had the quarterly review this morning and uh there were basically three things that came out of it first the pipeline numbers are actually better than we thought like we closed nineteen deals in q three not seventeen the discrepancy was uh two deals that got booked on october first instead of september thirtieth second thing is the churn rate went from four point two percent down to three point one percent which is uh the lowest it's been since we started tracking it in twenty twenty four sarah thinks that's because of the onboarding changes we shipped in august and i think she's right third and this is the one that i think we need to actually talk about is uh hiring we budgeted for six engineers this half and we've hired two the pipeline is thin i think we've got maybe four candidates at final stage and two of them we're probably going to lose to bigger offers so um my read is we should either raise the bands or we should accept that we're going to be at four not six and if we're at four then the roadmap needs to change because the mobile work assumed five people minimum oh also i need to follow up with legal about the vendor contract the one that expires on march thirty first twenty twenty six they wanted to know if we're renewing at the same tier which is uh twelve thousand four hundred dollars a year or if we're going up to the enterprise tier which is thirty one thousand five hundred my instinct is stay where we are for another year and revisit it once we know the headcount situation um what else oh yeah the incident from last tuesday we lost about forty five minutes of writes on the primary and uh the postmortem is due friday i said i'd write the timeline section so i need to actually do that the root cause was a config change that went out without the usual review and the fix is uh we're adding a required approver on that repo which should have been there already honestly let me also remember to send the deck to marcus before the board meeting on the eleventh he wanted it uh forty eight hours ahead so that means the ninth end of day and the numbers in it need to match what finance published on the seventh not the draft from the second i think that's everything um oh no wait one more thing the customer in berlin asked about data residency again they want confirmation in writing that nothing leaves the eu and right now i can't give them that because of the analytics vendor so either we move that vendor to an eu region or we carve them out for that account i'll ask the platform team what the lift is on the eu region option"

CASES = [
    ("mid", "semi-formal", "prose", "general", MID),
    ("long", "semi-formal", "prose", "general", LONG),
    ("nat", "semi-formal", "prose", "general",
     "so um i need to like send the the report by uh friday no wait make that thursday"),
    ("oracle", "semi-formal", "prose", "general",
     "the invoice came to twenty three thousand four hundred and fifty dollars and "
     "it's due on march third twenty twenty six"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--runner")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=str(HERE / "s1_mini_device_ref.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_id)
    runner = resolve_runner(args.runner)

    out, swift = {}, []
    for label, sty, struct, ctx, src in CASES:
        prompt_ids = build_ids(tok, sty, struct, ctx, src)
        text = run_engine(runner, args.bundle, prompt_ids, args.max_tokens)
        if text is None:
            sys.exit(f"{label}: engine produced no output")
        gen_ids = tok(text, add_special_tokens=False)["input_ids"]
        round_trip = tok.decode(gen_ids, skip_special_tokens=False)
        if round_trip != text:
            sys.exit(f"{label}: tokenizer round trip is not exact\n"
                     f"  engine   : {text!r}\n  re-decoded: {round_trip!r}")
        # Short of the cap => the engine stopped on EOS. Carry it so the device array also
        # proves the halt, not only the words.
        hit_eos = len(gen_ids) < args.max_tokens
        eos = tok.eos_token_id
        if hit_eos and isinstance(eos, int):
            gen_ids = gen_ids + [eos]
        out[label] = {"input": src, "styling": sty, "structure": struct, "context": ctx,
                      "prompt_ids": prompt_ids, "gen_ids": gen_ids, "text": text,
                      "stopped_on_eos": hit_eos}
        print(f"{label}: {len(prompt_ids)} prompt tok -> {len(gen_ids)} gen tok"
              f"{' (incl. EOS)' if hit_eos else ' (hit the cap, no EOS)'}")
        print(f"  {text!r}")

        def fmt(ids: list[int]) -> str:
            rows, per = [], 12
            for i in range(0, len(ids), per):
                rows.append("                " + ", ".join(str(x) for x in ids[i:i + per]) + ",")
            return "\n".join(rows)

        swift.append(f"            {label}Prompt: [{', '.join(str(x) for x in prompt_ids)}],")
        swift.append(f"            {label}Expected: [\n{fmt(gen_ids)}\n            ],")

    Path(args.out).write_text(json.dumps(
        {"schema": "s1-mini-device-ref/1", "hf_id": args.hf_id,
         "bundle": Path(args.bundle).name,
         "source": "Mac Core AI engine greedy (coreai-pipelined, warmup off, "
                   "COREAI_CHUNK_THRESHOLD=1) on the same bundle the device runs",
         "cases": out}, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}\n\n--- ModelSpec arrays ---")
    print("\n".join(swift))


if __name__ == "__main__":
    main()
