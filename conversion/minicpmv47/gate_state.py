#!/usr/bin/env python3
"""State carry-over (round 5): a second turn on the same image — continue the decoder's state vs prefill it again.

The ship decoder `minicpmv47_vlm_decode_int8hu` ("main", S=1) as AOT h16c with `SpecializationOptions.default()`
(gate_common.Decoder), and the two-turn oracle `turn2.npz` (`oracle_turn2.py`): P5 = P1's prompt (88) + A1 (the
oracle's 32-token answer) + <|im_end|> "\\n" + the second user turn ("What colors do you see?") and the generation
prompt, the image rows = the oracle's P1 tower rows as fp16, start / amount = `host_static_inputs(P5)`.

  (a) carry   fresh zero states; P1's prompt one token per call, then A1 one token per call (turn 1 as the state
              saw it; the engine's argmax at every A1 step is recorded against A1), then, with the same state, only
              the new tokens P5[turn1_len:] one per call; free-run greedy (<= 32, stop ids end it) -> A2_carry.
  (b) fresh   fresh zero states; all of P5 one token per call; free-run greedy -> A2_fresh. Every prompt position's
              argmax against the oracle's (`P5__top5_ids[:, 0]`).
  verdicts, fixed before running: A2_carry and A2_fresh against the HF A2 (`P5__gen_ids`) by the first-divergence
              rule (`P5__margins`: a divergence passes only at an oracle softmax top-2 gap < 0.1); (a) = (b): the
              same answer and the last prompt position's logits bit-equal (or, if not bit-equal, the difference and the
              first divergence by the same rule). start / amount of P5 = P1's (73 / 58) = the oracle's, asserted.
  timing      (inside the caller's quiet_hold window) the second turn's prompt part: (a) the new tokens x "main"
              on the carried state (turn 1 replayed untimed before each round) against (b) all of P5 x "main" from
              fresh states, alternated `--rounds` rounds after one untimed warm-up of each; wall from the first call
              to the last prompt call's logits on the host, and every call's ms.
  reset       the process re-runs (b)'s prompt at the end of the gate part; its last logits must be bit-equal.

    QW QH mv47-r5-state -- coreai-models/.venv/bin/python conversion/minicpmv47/gate_state.py <bundle-dir> --out <lane>/r5/state.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as G  # noqa: E402


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


async def feed(dec, ids, t0_pos: int, static: dict, state: dict, keep_all: bool = False):
    """ids one per call at positions t0_pos.. on `state`; returns (argmax list, last logits, kept {pos: logits}, ms list)."""
    argmax, kept, ms, lg = [], {}, [], None
    for i, tk in enumerate(ids):
        t = t0_pos + i
        c0 = time.perf_counter()
        lg = await dec.step(int(tk), t, static, state)
        ms.append((time.perf_counter() - c0) * 1e3)
        argmax.append(int(np.argmax(lg.astype(np.float32))))
        if keep_all:
            kept[t] = lg.copy()
    return argmax, lg, kept, ms


async def greedy(dec, last, t: int, static: dict, state: dict) -> list[int]:
    gen = []
    nxt = int(np.argmax(last.astype(np.float32)))
    while True:
        gen.append(nxt)
        if nxt in G.STOP_IDS or len(gen) >= G.N_FREE:
            return gen
        lg = await dec.step(nxt, t, static, state)
        t += 1
        nxt = int(np.argmax(lg.astype(np.float32)))


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers
    from minicpmv47_vl_pipelined import host_static_inputs

    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    aimodelc = G.aimodelc_for(bundle)
    z = np.load(G.ORACLE / "oracle.npz")
    t2 = np.load(G.ORACLE / "turn2.npz")
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    p5 = t2["P5__input_ids"].astype(np.int64)
    S5, T1 = int(t2["P5__prompt_len"]), int(t2["P5__turn1_len"])
    a1 = t2["P5__a1_ids"].astype(np.int64).tolist()
    a2_hf = t2["P5__gen_ids"].astype(np.int64).tolist()
    margins = t2["P5__margins"]
    p1 = z["P1__input_ids"].astype(np.int64)
    assert np.array_equal(p5[:len(p1)], p1) and p5[len(p1):T1].tolist() == a1 and T1 == len(p1) + len(a1)
    mapped, start, amount = host_static_inputs(p5)
    _, start1, amount1 = host_static_inputs(p1)
    start, amount = int(start[0]), int(amount[0])
    assert (start, amount) == (int(start1[0]), int(amount1[0])) == tuple(int(x) for x in t2["P5__rope_shift"]), \
        (start, amount, start1, amount1, t2["P5__rope_shift"])
    ids = mapped.numpy().astype(np.int64).tolist()
    assert np.array_equal(np.asarray(ids), t2["P5__mapped_ids"]), "host ids = the oracle's mapped ids"
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    dec = await G.Decoder.load(rt, aimodelc, int(meta["language"]["max_context_length"]))
    dec.check_contract()
    static = dec.static(z["P1__image_features"].astype(np.float16), start, amount)
    out = {"schema": "minicpmv47-gate-state/1", "bundle": str(bundle), "name": meta["name"],
           "aimodelc": G.tree_digest(aimodelc), "turn2_npz_sha256": G.sha256_file(G.ORACLE / "turn2.npz"),
           "environment": G.env_record(), "load_seconds": dec.load_seconds, "descriptor": dec.desc,
           "p5": {"tokens": S5, "turn1_tokens": T1, "p1_prompt": len(p1), "a1": len(a1), "new_tokens": S5 - T1,
                  "rope_shift_start": start, "rope_shift_amount": amount, "rope_shift_equal_p1": True,
                  "new_tokens_text": tok.decode(ids[T1:])},
           "bar": "A2_carry and A2_fresh = HF A2 by the first-divergence rule (knife edge = oracle gap < 0.1); "
                  "(a) = (b): same answer, last prompt logits bit-equal or the difference recorded",
           "process_prompts": 0}
    wr_before = G.window_record("mv47-r5-state")

    # (a) carry
    st_a = dec.fresh_state()
    arg1, _, _, _ = await feed(dec, ids[:T1], 0, static, st_a)
    a1_steps = arg1[len(p1) - 1:T1 - 1]                        # argmax at P1's last token .. A1[-2] predicts A1
    _, last_a, _, ms_a = await feed(dec, ids[T1:], T1, static, st_a)
    a2_carry = await greedy(dec, last_a, S5, static, st_a)
    out["process_prompts"] += 1
    # (b) fresh
    st_b = dec.fresh_state()
    arg_b, last_b, kept_b, ms_b = await feed(dec, ids, 0, static, st_b, keep_all=True)
    a2_fresh = await greedy(dec, last_b, S5, static, st_b)
    out["process_prompts"] += 1
    o_top1 = t2["P5__top5_ids"][:, 0]
    mism = [p for p in range(S5) if arg_b[p] != int(o_top1[p])]
    d_top5 = max(float(np.abs(kept_b[p].astype(np.float32)[t2["P5__top5_ids"][p]] - t2["P5__top5_vals"][p]).max())
                 for p in range(S5))
    la, lb = last_a.astype(np.float64), last_b.astype(np.float64)
    out["gate"] = {
        "turn1_engine_argmax_eq_a1": f"{sum(int(a == b) for a, b in zip(a1_steps, a1))}/{len(a1)}",
        "a2_hf": a2_hf, "a2_hf_text": tok.decode(a2_hf, skip_special_tokens=True),
        "a2_carry": a2_carry, "a2_fresh": a2_fresh,
        "a2_carry_text": tok.decode(a2_carry, skip_special_tokens=True),
        "carry_vs_hf": G.first_divergence(a2_carry, a2_hf, margins),
        "fresh_vs_hf": G.first_divergence(a2_fresh, a2_hf, margins),
        "carry_vs_fresh": G.first_divergence(a2_carry, a2_fresh, margins),
        "last_prompt_logits_carry_vs_fresh": {"bit_equal": bool(np.array_equal(last_a, last_b)),
                                              "max_abs_d": float(np.abs(la - lb).max()),
                                              "cos": float(la @ lb / (np.linalg.norm(la) * np.linalg.norm(lb)))},
        "last_prompt_logits_fresh_vs_hf": {"cos": float(lb @ t2["P5__logits"] / (np.linalg.norm(lb) * np.linalg.norm(t2["P5__logits"]))),
                                           "max_abs_d": float(np.abs(lb - t2["P5__logits"]).max()),
                                           "top5_engine": G.topk(last_b, 5)[0], "top5_hf": G.topk(t2["P5__logits"], 5)[0]},
        "fresh_prompt_argmax_eq_hf": f"{S5 - len(mism)}/{S5}", "fresh_prompt_argmax_mismatch_positions": mism,
        "fresh_prompt_max_abs_d_at_hf_top5": d_top5,
    }
    g = out["gate"]
    ok = (g["carry_vs_hf"]["verdict"] != "FAIL" and g["fresh_vs_hf"]["verdict"] != "FAIL"
          and g["carry_vs_fresh"]["verdict"] != "FAIL" and a2_carry == a2_fresh)
    print(f"[state] turn 1 argmax = A1 {g['turn1_engine_argmax_eq_a1']}; A2 carry vs HF {g['carry_vs_hf']['verdict']} "
          f"(div {g['carry_vs_hf']['first_divergence']}), fresh vs HF {g['fresh_vs_hf']['verdict']} (div "
          f"{g['fresh_vs_hf']['first_divergence']}), carry = fresh {a2_carry == a2_fresh}; last logits bit-equal "
          f"{g['last_prompt_logits_carry_vs_fresh']['bit_equal']} (max|d| {g['last_prompt_logits_carry_vs_fresh']['max_abs_d']:.3g}); "
          f"prompt argmax vs HF {g['fresh_prompt_argmax_eq_hf']} | {g['a2_carry_text']!r}", flush=True)
    st_r = dec.fresh_state()
    _, last_r, _, _ = await feed(dec, ids, 0, static, st_r)
    out["process_prompts"] += 1
    out["reset_check"] = {"sequence": "P5 fresh prompt", "bit_equal": bool(np.array_equal(last_r, last_b))}
    ok = ok and out["reset_check"]["bit_equal"]
    out["gate"]["verdict"] = "PASS" if ok else "FAIL"

    # timing
    if args.rounds:
        async def carry_round() -> tuple[float, list]:
            st = dec.fresh_state()
            await feed(dec, ids[:T1], 0, static, st)               # turn 1, untimed
            c0 = time.perf_counter()
            _, lg, _, ms = await feed(dec, ids[T1:], T1, static, st)
            return (time.perf_counter() - c0) * 1e3, ms, lg

        async def fresh_round() -> tuple[float, list]:
            st = dec.fresh_state()
            c0 = time.perf_counter()
            _, lg, _, ms = await feed(dec, ids, 0, static, st)
            return (time.perf_counter() - c0) * 1e3, ms, lg

        await carry_round()
        await fresh_round()
        out["process_prompts"] += 2
        arms = {"carry": [], "fresh": []}
        calls = {"carry": [], "fresh": []}
        order = []
        for r in range(args.rounds):
            for arm in (("carry", "fresh") if r % 2 == 0 else ("fresh", "carry")):
                wall, ms, lg = await (carry_round() if arm == "carry" else fresh_round())
                if int(np.argmax(lg.astype(np.float32))) != a2_hf[0]:
                    raise SystemExit(f"timing {arm} round {r}: first token {int(np.argmax(lg))} != HF {a2_hf[0]}")
                arms[arm].append(wall)
                calls[arm] += ms
                order.append(arm)
                out["process_prompts"] += 1
        out["timing"] = {
            "rounds": args.rounds, "order": order,
            "arms": {"carry": f"main x {S5 - T1} (turn 2's new tokens on the carried state)",
                     "fresh": f"main x {S5} (the whole two-turn prompt from fresh states)"},
            "turn2_prompt_ms": {"carry": stats(arms["carry"]), "fresh": stats(arms["fresh"]),
                                "carry_runs": arms["carry"], "fresh_runs": arms["fresh"]},
            "calls_saved": T1, "ms_saved_median": float(np.median(arms["fresh"]) - np.median(arms["carry"])),
            "speedup_median": float(np.median(arms["fresh"]) / np.median(arms["carry"])),
            "call_ms": {k: stats(v) for k, v in calls.items()},
            "definition": "wall from the first turn-2 prompt call to the last prompt call's logits on the host "
                          "(states allocated and turn 1 replayed before the clock starts)",
        }
        t = out["timing"]
        print(f"[timing] turn-2 prompt: carry {S5 - T1} calls {t['turn2_prompt_ms']['carry']['median']:.1f} ms (p10 "
              f"{t['turn2_prompt_ms']['carry']['p10']:.1f}, p90 {t['turn2_prompt_ms']['carry']['p90']:.1f}) vs fresh {S5} calls "
              f"{t['turn2_prompt_ms']['fresh']['median']:.1f} ms (p10 {t['turn2_prompt_ms']['fresh']['p10']:.1f}, p90 "
              f"{t['turn2_prompt_ms']['fresh']['p90']:.1f}); saved {T1} calls = {t['ms_saved_median']:.1f} ms "
              f"({t['speedup_median']:.2f}x); main call median {t['call_ms']['fresh']['median']:.2f} ms", flush=True)
    out["window_before"], out["window_after"] = wr_before, G.window_record("mv47-r5-state")
    out["result"] = out["gate"]["verdict"]
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/minicpmv47/gate_state.py", "sha256": G.sha256_file(Path(__file__))}
    G.write_json(Path(args.out), out)
    print(f"[gate] {out['name']} state: {out['result']} prompts {out['process_prompts']} wall {out['wall_seconds']:.1f}s "
          f"-> {args.out}", flush=True)
    return 0 if out["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
