#!/usr/bin/env python3
"""Chunked prefill (round 5): the multifunction `minicpmv47_vlm_decode_int8hu_pf16` on the Mac GPU — answers and TTFT.

The bundle (`export_decoder.py int8hu --prefill-chunk 16`) holds "main" (S=1) and "prefill" (static S=16, the
logits of the chunk's last position). Loaded as AOT h16c (`<exports>/minicpmv47_aotc/<name>.h16c.aimodelc`,
efr) with `SpecializationOptions.default()` (gate_common), fresh zero states per sequence, the full position ramp.

  --gate     P1..P4 (ids / start / amount = `host_static_inputs(<P>__input_ids)`, image_embeds = the oracle's
             tower rows as fp16, as r3 G2), each in two orders from fresh states:
               chunk order  "prefill" on ids[c : c + 16] with position_ids 0..c+15 while a whole chunk is left,
                            then "main" one token at a time to the end of the prompt (`vlm.prefill_order`);
               S=1 order    "main" only, one token per call (r3 G2's order, same bundle);
             then free-run greedy (<= 32, stop ids end it) with "main" from the same states.
             Verdicts (the supervisor's rule, fixed before running): the chunk order's answer against the oracle's
             `<P>__gen_ids` by the first-divergence rule (`margins.npz`: a divergence passes only at an oracle
             softmax top-2 gap < 0.1), and against the S=1 order's answer by the same rule; the last prompt
             position's logits chunk vs S=1: cos >= 0.999. Recorded: every chunk boundary's "prefill" logits
             against the S=1 order's logits at the same position (cos min, max |d|, argmax), and the last
             position against `<P>__logits`. The process ends by re-running its first sequence (bit-equal).
  --timing   inside the caller's `quiet_hold` window: P1's 88-token prompt from fresh states, (i) "main" x 88
             against (ii) "prefill" x 5 + "main" x 8, alternated for `--rounds` rounds (the arm that goes first
             swaps every round), plus one untimed warm-up of each. TTFT here = wall from the first call to the
             last prompt call's logits read back on the host (the first generated token's logits); every call's
             ms is kept too. The window record (lock, lsof, load, swap, other processes) goes with the numbers.

Both modes merge into one json (`--out`, sections "gate" / "timing").

    QW         coreai-models/.venv/bin/python conversion/minicpmv47/gate_chunk.py <bundle-dir> --gate --out <lane>/r5/chunk.json
    QW QH ...  coreai-models/.venv/bin/python conversion/minicpmv47/gate_chunk.py <bundle-dir> --timing --out <lane>/r5/chunk.json
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

COS_BAR = 0.999


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class ChunkDecoder(G.Decoder):
    """`main` (S=1, gate_common.Decoder) + `prefill` (static S = C) of one multifunction asset."""

    @classmethod
    async def load(cls, rt, aimodelc: Path, max_ctx: int):
        t0 = time.perf_counter()
        model = await G.maybe(rt.AIModel.load(str(aimodelc), rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await G.maybe(model.load_function("main"))
        t2 = time.perf_counter()
        pf = await G.maybe(model.load_function("prefill"))
        t3 = time.perf_counter()
        self = cls(rt, model, fn, max_ctx)
        self.pf = pf
        self.pf_desc = G.fn_desc(pf)
        self.function_names = list(getattr(model, "function_names", []) or [])
        self.load_seconds = {"model": t1 - t0, "main": t2 - t1, "prefill": t3 - t2}
        return self

    def check_prefill(self, C: int) -> None:
        d = self.pf_desc
        want = dict(G.DECODER_CONTRACT, input_ids=([1, C], "int32"))
        bad = {n: (d["inputs"].get(n), w) for n, w in want.items()
               if d["inputs"].get(n) is None or tuple(d["inputs"][n][0]) != tuple(w[0]) or d["inputs"][n][1] != w[1]}
        if bad or set(d["inputs"]) != set(want):
            raise SystemExit(f"prefill descriptor differs from the contract: {bad} {sorted(d['inputs'])}")
        if [tuple(v[0]) for v in d["outputs"].values()] != [(1, 1, G.V)]:
            raise SystemExit(f"prefill outputs {d['outputs']} != logits [1, 1, {G.V}]")
        if d["states"] != self.desc["states"]:
            raise SystemExit(f"prefill states {d['states']} != main states {self.desc['states']}")

    async def prefill(self, ids_chunk, cursor: int, static: dict, state: dict) -> np.ndarray:
        rt = self.rt
        C = len(ids_chunk)
        res = await G.maybe(self.pf(inputs={"input_ids": G.nd(rt, np.array([ids_chunk], np.int32)),
                                            "position_ids": G.nd(rt, np.arange(cursor + C, dtype=np.int32)[None]),
                                            **static}, state=state))
        lg = np.asarray(res["logits"].numpy())
        assert lg.shape == (1, 1, G.V), lg.shape
        return lg[0, 0]

    async def run_chunk(self, ids, C: int, static: dict, free: int = 0, state: dict | None = None) -> dict:
        """The prompt in chunk order from fresh states (or `state`), then free-run greedy with `main`."""
        state = self.fresh_state() if state is None else state
        ids = [int(t) for t in ids]
        cursor, kinds, ms, boundary = 0, [], [], {}
        lg = None
        t_first = time.perf_counter()
        while len(ids) - cursor >= C:
            t0 = time.perf_counter()
            lg = await self.prefill(ids[cursor:cursor + C], cursor, static, state)
            ms.append((time.perf_counter() - t0) * 1e3)
            kinds.append(1)
            boundary[cursor + C - 1] = lg.copy()
            cursor += C
        while cursor < len(ids):
            t0 = time.perf_counter()
            lg = await self.step(ids[cursor], cursor, static, state)
            ms.append((time.perf_counter() - t0) * 1e3)
            kinds.append(0)
            cursor += 1
        ttft_ms = (time.perf_counter() - t_first) * 1e3
        gen, gen_ms, t = [], [], len(ids)
        if free:
            nxt = int(np.argmax(lg.astype(np.float32)))
            while True:
                gen.append(nxt)
                if nxt in G.STOP_IDS or len(gen) >= free:
                    break
                t0 = time.perf_counter()
                lg2 = await self.step(nxt, t, static, state)
                gen_ms.append((time.perf_counter() - t0) * 1e3)
                t += 1
                nxt = int(np.argmax(lg2.astype(np.float32)))
        return {"last": lg, "boundary": boundary, "kinds": kinds, "call_ms": ms, "ttft_ms": ttft_ms,
                "gen": gen, "gen_ms": gen_ms, "state": state}

    async def run_s1_prompt(self, ids, static: dict) -> dict:
        state = self.fresh_state()
        ms = []
        lg = None
        t_first = time.perf_counter()
        for t, tk in enumerate(ids):
            t0 = time.perf_counter()
            lg = await self.step(int(tk), t, static, state)
            ms.append((time.perf_counter() - t0) * 1e3)
        return {"last": lg, "call_ms": ms, "ttft_ms": (time.perf_counter() - t_first) * 1e3}


def stats(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


async def gate(dec: ChunkDecoder, C: int, z, mz, tok, out: dict) -> bool:
    from minicpmv47_vl_pipelined import host_static_inputs

    res, first = {}, None
    for P in ("P1", "P2", "P3", "P4"):
        mapped, start, amount = host_static_inputs(z[f"{P}__input_ids"].astype(np.int64))
        ids = mapped.numpy().astype(np.int64).tolist()
        S = len(ids)
        static = dec.static(z[f"{P}__image_features"].astype(np.float16), int(start[0]), int(amount[0]))
        rc = await dec.run_chunk(ids, C, static, free=G.N_FREE)
        rs = await dec.run(ids, static, keep=set(rc["boundary"]) | {S - 1}, free=G.N_FREE)
        out["process_prompts"] += 2
        og = z[f"{P}__gen_ids"].astype(np.int64).tolist()
        margins = mz[f"{P}__margins"]
        vs_oracle = G.first_divergence(rc["gen"], og, margins)
        vs_s1 = G.first_divergence(rc["gen"], rs["gen"], margins)
        s1_vs_oracle = G.first_divergence(rs["gen"], og, margins)
        last_c, last_s = rc["last"].astype(np.float32), rs["kept"][S - 1].astype(np.float32)
        bnd = []
        for p, lg in sorted(rc["boundary"].items()):
            ref = rs["kept"][p].astype(np.float32)
            bnd.append({"position": p, "cos": cos(lg, ref), "max_abs_d": float(np.abs(lg.astype(np.float32) - ref).max()),
                        "argmax_chunk": int(np.argmax(lg)), "argmax_s1": int(np.argmax(ref))})
        te, _ = G.topk(last_c, 5)
        to, _ = G.topk(z[f"{P}__logits"], 5)
        rec = {"prompt_tokens": S, "rope_shift_start": int(start[0]), "rope_shift_amount": int(amount[0]),
               "calls_chunk_order": {"prefill": int(sum(rc["kinds"])), "main": len(rc["kinds"]) - int(sum(rc["kinds"]))},
               "chunk_gen": rc["gen"], "s1_gen": rs["gen"], "oracle_gen": og,
               "chunk_vs_oracle": vs_oracle, "chunk_vs_s1": vs_s1, "s1_vs_oracle": s1_vs_oracle,
               "chunk_text": tok.decode(rc["gen"], skip_special_tokens=True),
               "last_position_chunk_vs_s1": {"cos": cos(last_c, last_s), "max_abs_d": float(np.abs(last_c - last_s).max()),
                                             "argmax_equal": int(np.argmax(last_c)) == int(np.argmax(last_s)),
                                             "bit_equal": bool(np.array_equal(rc["last"], rs["kept"][S - 1]))},
               "last_position_chunk_vs_oracle": {"cos": cos(last_c, z[f"{P}__logits"]), "top5_chunk": te, "top5_oracle": to,
                                                 "top5_ordered_equal": te == to},
               "boundaries": bnd,
               "boundary_cos_min": min(b["cos"] for b in bnd) if bnd else None,
               "boundary_max_abs_d": max(b["max_abs_d"] for b in bnd) if bnd else None,
               "boundary_argmax_equal": f"{sum(b['argmax_chunk'] == b['argmax_s1'] for b in bnd)}/{len(bnd)}"}
        ok = (vs_oracle["verdict"] != "FAIL" and vs_s1["verdict"] != "FAIL"
              and rec["last_position_chunk_vs_s1"]["cos"] >= COS_BAR)
        rec["verdict"] = "PASS" if ok else "FAIL"
        res[P] = rec
        if first is None:
            first = (P, ids, static, rc["last"])
        print(f"[chunk {P}] {rec['verdict']}: vs oracle {vs_oracle['verdict']} (div {vs_oracle['first_divergence']}), "
              f"vs S=1 {vs_s1['verdict']} (div {vs_s1['first_divergence']}); last cos {rec['last_position_chunk_vs_s1']['cos']:.6f} "
              f"max|d| {rec['last_position_chunk_vs_s1']['max_abs_d']:.4f}; boundaries cos min {rec['boundary_cos_min']} "
              f"argmax {rec['boundary_argmax_equal']}; calls {rec['calls_chunk_order']} | {rec['chunk_text']!r}", flush=True)
    P, ids, static, last0 = first
    r = await dec.run_chunk(ids, C, static, free=0)
    out["process_prompts"] += 1
    out["reset_check"] = {"sequence": f"{P} chunk order", "bit_equal": bool(np.array_equal(last0, r["last"])),
                          "max_abs_diff": float(np.abs(last0.astype(np.float32) - r["last"].astype(np.float32)).max())}
    n_pass = sum(v["verdict"] == "PASS" for v in res.values())
    out["gate"] = {"prompts": res, "pass": f"{n_pass}/4",
                   "bar": "per prompt: chunk-order answer vs oracle and vs the S=1 order by the first-divergence rule "
                          f"(knife edge = oracle gap < {G.KNIFE_EDGE}); last prompt position chunk vs S=1 cos >= {COS_BAR}",
                   "verdict": "PASS" if n_pass == 4 and out["reset_check"]["bit_equal"] else "FAIL"}
    print(f"[reset] bit-equal {out['reset_check']['bit_equal']}", flush=True)
    return out["gate"]["verdict"] == "PASS"


async def timing(dec: ChunkDecoder, C: int, z, rounds: int, out: dict) -> None:
    from minicpmv47_vl_pipelined import host_static_inputs

    mapped, start, amount = host_static_inputs(z["P1__input_ids"].astype(np.int64))
    ids = mapped.numpy().astype(np.int64).tolist()
    static = dec.static(z["P1__image_features"].astype(np.float16), int(start[0]), int(amount[0]))
    first_tok = int(z["P1__gen_ids"][0])
    wr_before = G.window_record("mv47-r5-chunk")
    await dec.run_s1_prompt(ids, static)        # untimed warm-up of each arm
    await dec.run_chunk(ids, C, static)
    out["process_prompts"] += 2
    arms = {"s1": [], "chunk": []}
    calls = {"s1_main": [], "chunk_prefill": [], "chunk_main": []}
    order_log = []
    for r in range(rounds):
        seq = ("s1", "chunk") if r % 2 == 0 else ("chunk", "s1")
        for arm in seq:
            if arm == "s1":
                x = await dec.run_s1_prompt(ids, static)
                calls["s1_main"] += x["call_ms"]
            else:
                x = await dec.run_chunk(ids, C, static)
                k = np.asarray(x["kinds"])
                cm = np.asarray(x["call_ms"])
                calls["chunk_prefill"] += cm[k == 1].tolist()
                calls["chunk_main"] += cm[k == 0].tolist()
            if int(np.argmax(x["last"].astype(np.float32))) != first_tok:
                raise SystemExit(f"timing {arm} round {r}: first token {int(np.argmax(x['last']))} != oracle {first_tok}")
            arms[arm].append(x["ttft_ms"])
            order_log.append(arm)
            out["process_prompts"] += 1
    wr_after = G.window_record("mv47-r5-chunk")
    n_pf = len(ids) // C
    out["timing"] = {
        "prompt": "P1", "prompt_tokens": len(ids), "chunk": C,
        "arms": {"s1": f"main x {len(ids)}", "chunk": f"prefill x {n_pf} + main x {len(ids) - n_pf * C}"},
        "rounds": rounds, "order": order_log,
        "ttft_ms": {"s1": stats(arms["s1"]), "chunk": stats(arms["chunk"]), "s1_runs": arms["s1"], "chunk_runs": arms["chunk"]},
        "speedup_median": float(np.median(arms["s1"]) / np.median(arms["chunk"])),
        "call_ms": {k: stats(v) for k, v in calls.items()},
        "prefill_call_tokens_per_s_median": float(C / (np.median(calls["chunk_prefill"]) / 1e3)),
        "first_token_equals_oracle": True,
        "definition": "TTFT = wall from the first prompt call to the last prompt call's logits on the host "
                      "(fresh zero states allocated before the clock starts); python runtime, AOT h16c, "
                      "SpecializationOptions.default()",
        "window_before": wr_before, "window_after": wr_after,
    }
    t = out["timing"]
    print(f"[timing] P1 {len(ids)} tok: S=1 {t['ttft_ms']['s1']['median']:.1f} ms (p10 {t['ttft_ms']['s1']['p10']:.1f}, p90 "
          f"{t['ttft_ms']['s1']['p90']:.1f}) vs chunk {t['ttft_ms']['chunk']['median']:.1f} ms (p10 {t['ttft_ms']['chunk']['p10']:.1f}, "
          f"p90 {t['ttft_ms']['chunk']['p90']:.1f}) = {t['speedup_median']:.2f}x; prefill call {t['call_ms']['chunk_prefill']['median']:.2f} ms, "
          f"main call {t['call_ms']['s1_main']['median']:.2f} ms", flush=True)


async def main_async(args) -> int:
    import coreai.runtime as rt
    import tokenizers

    bundle = Path(args.bundle).expanduser().resolve()
    meta = json.loads((bundle / "metadata.json").read_text())
    C = int(meta["language"]["prefill_chunk"])
    if meta["language"]["function_map"]["main"] != ["main", "prefill"]:
        raise SystemExit(f"not a main + prefill bundle: {meta['language']['function_map']}")
    aimodelc = G.aimodelc_for(bundle)
    z = np.load(G.ORACLE / "oracle.npz")
    mz = np.load(G.ORACLE / "margins.npz")
    tok = tokenizers.Tokenizer.from_file(str(bundle / "tokenizer" / "tokenizer.json"))
    outp = Path(args.out)
    doc = json.loads(outp.read_text()) if outp.exists() else {"schema": "minicpmv47-gate-chunk/1"}
    gs = G.GpuSampler().__enter__()
    t_all = time.perf_counter()
    dec = await ChunkDecoder.load(rt, aimodelc, int(meta["language"]["max_context_length"]))
    dec.check_contract()
    dec.check_prefill(C)
    out = {"bundle": str(bundle), "name": meta["name"], "aimodelc": G.tree_digest(aimodelc),
           "metadata_sha256": G.sha256_file(bundle / "metadata.json"), "environment": G.env_record(),
           "load_seconds": dec.load_seconds, "function_names": dec.function_names,
           "descriptor_main": dec.desc, "descriptor_prefill": dec.pf_desc, "process_prompts": 0}
    ok = True
    if args.gate:
        ok = await gate(dec, C, z, mz, tok, out)
    if args.timing:
        await timing(dec, C, z, args.rounds, out)
    out["wall_seconds"] = time.perf_counter() - t_all
    gs.__exit__()
    out["gpu_utilization"] = gs.summary()
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = {"path": "conversion/minicpmv47/gate_chunk.py", "sha256": G.sha256_file(Path(__file__))}
    doc["gate_run" if args.gate else "timing_run"] = out
    G.write_json(outp, doc)
    print(f"[done] {meta['name']} prompts {out['process_prompts']} wall {out['wall_seconds']:.1f}s -> {outp}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.gate == args.timing:
        ap.error("pass exactly one of --gate / --timing (the timing runs in its own window)")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
