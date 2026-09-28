#!/usr/bin/env python3
"""fp32 torch parity: the ids-input Qwen3.5 VL decoder vs the author's fp32 oracle.

`qwen3_5_vl_pipelined.Qwen3_5VLPipelinedForCausalLM` is driven the way its Core AI graph runs:
S=1, fresh zero states per run, every GDN layer on the loop-free single step, ids in (image
tokens rewritten to V + k), the static image inputs fixed for the run, M-RoPE derived in-graph.
The image rows are the oracle's own HF fp32 tower output (`oracle/npz/<row>__<arm>.npz`), so
this isolates the decoder. Compared against `oracle/fixture_oracle.json` (read-only):

  P1  every run (g256 / g448 / native / text): slot count and positions (the author's slot
      finder on the ids), letter argmax at every slot, |dp| of softmax(T=1) over the first nopts
      letters; also the in-graph rope planes vs the planes the oracle's rotary received, and the
      final-norm hidden at each slot vs the oracle's. PASS = slots equal and argmax equal on
      every run, max |dp| <= 1e-4.
  P2  text rows: this module (zero image inputs, shift start 1<<30) vs the overlay's plain
      `Qwen3_5StatefulForCausalLM` (separately loaded, same S=1 stepping) — logits bit-equal at
      every step; else the first differing step and layer.
  P3  r01 / r02 at g256: every decoder layer's output at every position vs the oracle's
      hidden_states (max |d| and cos per layer).
  P4  r14 / r16 at g256, red arms: image_rc row/col swapped, 1-D positions (0..T-1 on all
      planes, the oracle's g256_pos1d definition), image planes collapsed to p_text with the
      host shift kept, image_embeds zero — |dp| and argmax change vs the unperturbed run.

Harness-only substitution (the module is untouched): torch's CPU build here has no oneDNN, so
the GDN depthwise conv (`F.conv1d`, groups = 6144) runs as 6144 per-channel slow convolutions,
~4.5 s per decode step. The conv is evaluated as one `torch.bmm` over the same windows instead,
and at the last slot step of every run all 18 layers' results are checked against `F.conv1d`
itself (`torch.equal`, all channels); the counts go into the transcript.

    HF_HOME=~/code/coreai/_decider2bv/hf HF_HUB_OFFLINE=1 \\
      python parity_decoder_torch.py --shard 0 --nshards 3 --out-dir <dir>   # x3 in parallel
    python parity_decoder_torch.py --merge --out-dir <dir> \\
      --transcript models/decider-2b-vision/gate-decider-2b-vision-torch-parity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

os.environ.setdefault("HF_HOME", str(work_path("_decider2bv", "hf")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "Mapika/decider-2b-vision"
ORACLE = work_path("_decider2bv", "oracle")
VISION_START, IMAGE_PAD, VISION_END = 248053, 248056, 248054
SLOT_TOKEN, COLON, ANSWER = 318, 25, 15666
LETTER_IDS = list(range(32, 42))
MERGE = 2
BAR_MAX_DP = 1e-4
P3_RUNS = [("r01", "g256"), ("r02", "g256")]
P4_RUNS = [("r14", "g256"), ("r16", "g256")]
DETERMINISM_RUN = ("t03", "text")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_slots(ids: list[int]) -> list[int]:
    """The author's slot finder (decider/vision.py VisionDecisionModel.prepare)."""
    return [i for i in range(2, len(ids))
            if ids[i] == SLOT_TOKEN and ids[i - 1] == COLON and ANSWER in ids[max(0, i - 5):i]]


def merged_grid(row: dict) -> tuple[int, int] | None:
    g = row["grid_thw"]
    if g is None:
        return None
    assert g[0] == 1, g
    return g[1] // MERGE, g[2] // MERGE


class DepthwiseConvBmm:
    """F.conv1d replacement for the GDN's valid depthwise conv (batch 1, no bias, stride 1),
    evaluated as one bmm; every other call goes to the original. `check` compares against
    the original on the next calls."""

    def __init__(self, original):
        self.original = original
        self.check = False
        self.fast_calls = self.checked = self.mismatched = 0
        self.max_abs_diff = 0.0

    def __call__(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        C = weight.shape[0]
        if not (bias is None and input.dim() == 3 and input.shape[0] == 1 and input.shape[1] == C
                and groups == C and weight.shape[1] == 1 and stride in (1, (1,))
                and padding in (0, (0,)) and dilation in (1, (1,))):
            return self.original(input, weight, bias, stride, padding, dilation, groups)
        import torch
        K = weight.shape[-1]
        win = input[0].unfold(-1, K, 1)                              # [C, L, K]
        out = torch.bmm(weight, win.transpose(1, 2)).reshape(1, C, -1)  # [C,1,K] @ [C,K,L]
        self.fast_calls += 1
        if self.check:
            ref = self.original(input, weight, bias, stride, padding, dilation, groups)
            self.checked += 1
            if not torch.equal(ref, out):
                self.mismatched += 1
                self.max_abs_diff = max(self.max_abs_diff, float((ref - out).abs().max()))
        return out

    def report(self) -> dict:
        return {"fast_calls": self.fast_calls, "checked_vs_F_conv1d": self.checked,
                "mismatched": self.mismatched, "max_abs_diff": self.max_abs_diff}


class Runner:
    def __init__(self, threads: int, need_plain: bool):
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(threads)
        self.torch = torch
        sys.path.insert(0, str(HERE))
        from qwen3_5_vl_pipelined import Qwen3_5VLPipelinedForCausalLM, host_static_inputs
        from coreai_models.models.macos import qwen3_5 as q

        self.q = q
        self.host_static_inputs = host_static_inputs
        t0 = time.monotonic()
        self.model = Qwen3_5VLPipelinedForCausalLM.from_hf(HF_ID, target_dtype=torch.float32)
        self.load_seconds = time.monotonic() - t0
        self.cls = Qwen3_5VLPipelinedForCausalLM
        self.cfg = self.model.config
        self.V = self.cfg.vocab_size
        self.nmax = self.model.n_image_max
        self.loopfree = self._loopfree(self.model)
        self.plain = None
        if need_plain:
            self.plain = q.Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
                HF_ID, max_context_length=4096, target_dtype=torch.float32,
                hf_config_attr="text_config")
            self.plain.eval()
            self._loopfree(self.plain)
        self.conv = DepthwiseConvBmm(F.conv1d)
        F.conv1d = self.conv
        self.planes_override = None
        self._planes_log = None
        self.model._rope_planes = self._rope_planes

    @staticmethod
    def _loopfree(model) -> int:
        n = 0
        for layer in model.model.layers:
            if not layer.is_full:
                layer.linear_attn.use_loopfree_step = True
                n += 1
        return n

    def _rope_planes(self, is_img, slot, p, image_rc, start, amount):
        planes = self.cls._rope_planes(self.model, is_img, slot, p, image_rc, start, amount)
        if self.planes_override is not None:
            planes = self.planes_override(is_img, slot, p, image_rc, start, amount, planes)
        if self._planes_log is not None:
            self._planes_log.append([int(x.reshape(-1)[0]) for x in planes])
        return planes

    def static_inputs(self, row: dict, npz):
        torch = self.torch
        hw = merged_grid(row)
        ids, rc, start, amount = self.host_static_inputs(
            row["ids"], hw, self.V, IMAGE_PAD, VISION_START, self.nmax)
        emb = torch.zeros(self.nmax, self.cfg.hidden_size, dtype=torch.float32)
        if hw is not None:
            e = torch.from_numpy(np.asarray(npz["image_embeds"], dtype=np.float32))
            assert e.shape == (hw[0] * hw[1], self.cfg.hidden_size), (e.shape, hw)
            emb[: e.shape[0]] = e
        return ids, emb, rc, start, amount

    def step_loop(self, ids, emb, rc, start, amount, read_steps, *, layer_taps=False,
                  plain_lockstep=False):
        """S=1 through one run. Returns (logits at read_steps, final-norm hidden at read_steps,
        in-graph rope planes [3, T], per-position taps for P3, P2 lockstep stats)."""
        torch = self.torch
        T = int(ids.shape[0])
        st = self.q.build_decode_state(self.cfg, max_seq_len=T + 8, dtype=torch.float32)
        st_p = (self.q.build_decode_state(self.cfg, max_seq_len=T + 8, dtype=torch.float32)
                if plain_lockstep else None)
        taps = {"layers": [], "embed": [], "norm": []} if layer_taps else None
        cur, cur_p, norm_last = [], [], [None]
        hooks = [self.model.model.norm.register_forward_hook(
            lambda mod, args, out: norm_last.__setitem__(0, out[0, -1].detach().clone()))]
        if layer_taps or plain_lockstep:
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(
                    lambda mod, args, out: cur.append(out[0, -1].detach().clone())))
        if layer_taps:
            hooks.append(self.model.model.layers[0].register_forward_pre_hook(
                lambda mod, args: taps["embed"].append(args[0][0, -1].detach().clone())))
        if plain_lockstep:
            for layer in self.plain.model.layers:
                hooks.append(layer.register_forward_hook(
                    lambda mod, args, out: cur_p.append(out[0, -1].detach().clone())))
        self._planes_log = []
        read, slot_hidden = {}, {}
        p2 = {"steps": T, "bit_equal_steps": 0, "max_abs_diff": 0.0, "first_diff_step": None,
              "first_diff_layer": None} if plain_lockstep else None
        check_step = max(read_steps) if read_steps else T - 1
        try:
            with torch.inference_mode():
                for t in range(T):
                    self.conv.check = t == check_step
                    cur.clear()
                    cur_p.clear()
                    pos = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
                    tok = ids[t].reshape(1, 1)
                    out = self.model(tok, pos, emb, rc, start, amount, st["k_cache"], st["v_cache"],
                                     st["conv_state"], st["rec_state"])
                    logits = out[0, -1]
                    if t in read_steps:
                        read[t] = logits.clone()
                        slot_hidden[t] = norm_last[0]
                    if layer_taps:
                        taps["layers"].extend(cur)
                        taps["norm"].append(norm_last[0])
                    if plain_lockstep:
                        out_p = self.plain(tok, pos, st_p["k_cache"], st_p["v_cache"],
                                           st_p["conv_state"], st_p["rec_state"])
                        lp = out_p[0, -1]
                        if torch.equal(logits, lp):
                            p2["bit_equal_steps"] += 1
                        else:
                            p2["max_abs_diff"] = max(p2["max_abs_diff"], float((logits - lp).abs().max()))
                            if p2["first_diff_step"] is None:
                                p2["first_diff_step"] = t
                                for li, (a, b) in enumerate(zip(cur, cur_p)):
                                    if not torch.equal(a, b):
                                        p2["first_diff_layer"] = li
                                        break
        finally:
            for h in hooks:
                h.remove()
            self.conv.check = False
        planes = np.array(self._planes_log, dtype=np.int64).T  # [3, T]
        self._planes_log = None
        return read, slot_hidden, planes, taps, p2


def letter_probs(logits_row, nopts: int):
    import torch
    lg = logits_row[LETTER_IDS[0]:LETTER_IDS[0] + 10].float()
    p = torch.softmax(lg[:nopts], -1)
    return lg, p


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def p1_record(runner: Runner, row: dict, npz, read, slot_hidden, planes, seconds) -> dict:
    import torch
    slots_host = find_slots(row["ids"])
    rec = {"id": row["id"], "arm": row["arm"], "tokens": row["tokens"], "grid_thw": row["grid_thw"],
           "merged_grid": merged_grid(row), "slots_host": slots_host, "slots_oracle": row["slot_idx"],
           "slots_equal": slots_host == row["slot_idx"], "rope_planes_equal_oracle": bool(
               planes.shape == npz["rope_pos"].shape and np.array_equal(planes, npz["rope_pos"])),
           "seconds": seconds, "slots": []}
    if not rec["rope_planes_equal_oracle"]:
        bad = np.argwhere(planes != npz["rope_pos"]) if planes.shape == npz["rope_pos"].shape else []
        rec["rope_first_mismatch"] = bad[:4].tolist() if len(bad) else "shape"
    top1 = row.get("full_vocab_top1_id")
    for s, (t, nopts) in enumerate(zip(row["slot_idx"], row["nopts"])):
        lg, p = letter_probs(read[t], nopts)
        po = np.asarray(row["probs"][s], dtype=np.float64)
        pm = p.double().numpy()
        dp = np.abs(pm - po)
        lo = np.asarray(row["letter_logits"][s], dtype=np.float64)
        sh = slot_hidden[t].double().numpy()
        oh = np.asarray(npz["slot_hidden"][s], dtype=np.float64)
        full_top1 = int(torch.argmax(read[t]))
        rec["slots"].append({
            "t": t, "nopts": nopts,
            "letter_logits": [float(v) for v in lg.tolist()],          # all 10, raw
            "probs": [float(v) for v in p.tolist()],
            "probs_oracle": row["probs"][s],
            "argmax": int(pm.argmax()), "argmax_oracle": row["argmax"][s],
            "argmax_equal": int(pm.argmax()) == row["argmax"][s],
            "max_abs_dp": float(dp.max()),
            "max_abs_dlogit": float(np.abs(lg[:nopts].double().numpy() - lo).max()),
            "slot_hidden_max_abs_diff": float(np.abs(sh - oh).max()),
            "slot_hidden_cos": cosine(sh, oh),
            "full_vocab_top1_id": full_top1,
            "full_vocab_top1_equal_oracle": (top1[s] == full_top1) if top1 else None,
            "oracle_top2_margin": row["top2_margin"][s],
        })
    rec["max_abs_dp"] = max(x["max_abs_dp"] for x in rec["slots"])
    rec["argmax_all_equal"] = all(x["argmax_equal"] for x in rec["slots"])
    return rec


def p3_record(row, npz, taps, slot_t) -> dict:
    hs = npz["hidden_states"]                       # [25, T, 2048]; [24] = final-norm output
    T = hs.shape[1]
    n_layers = len(taps["layers"]) // T
    mine = np.stack([x.numpy() for x in taps["layers"]]).reshape(T, n_layers, -1)
    emb = np.stack([x.numpy() for x in taps["embed"]])
    norm = np.stack([x.numpy() for x in taps["norm"]]) if taps["norm"] else None
    out = []

    def cmp(name, a, b):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        coss = [cosine(a[t], b[t]) for t in range(T)]
        out.append({"stage": name, "max_abs_diff": float(d.max()),
                    "max_abs_diff_at_slot": float(d[slot_t].max()),
                    "ref_absmax": float(np.abs(b).max()),
                    "min_cos": float(min(coss)), "cos_at_slot": float(coss[slot_t])})

    cmp("embed (hs[0])", emb, hs[0])
    for li in range(n_layers - 1):                  # layer li output == hs[li + 1]; hs[24] is normed
        cmp(f"layer {li} (hs[{li + 1}])", mine[:, li], hs[li + 1])
    if norm is not None:
        cmp("final norm (hs[24])", norm, hs[24])
    return {"id": row["id"], "arm": row["arm"], "tokens": T, "slot_t": slot_t, "stages": out,
            "worst_max_abs_diff": max(x["max_abs_diff"] for x in out),
            "worst_min_cos": min(x["min_cos"] for x in out)}


def run_shard(args) -> None:
    import torch

    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    rows = fx["rows"]
    by_key = {(r["id"], r["arm"]): r for r in rows}
    if args.runs:
        keys = [tuple(k.split(":")) for k in args.runs.split(",")]
    else:
        keys = [(r["id"], r["arm"]) for r in rows]
    tests = set(args.tests.split(","))

    # Deterministic LPT split by token count; shard 0 carries P2 (text rows in lockstep with the
    # plain model, weight 2), shard 1 (or 0 when alone) carries P4 (base + 4 perturbed arms, weight 5).
    load = [0] * args.nshards
    assign = {i: [] for i in range(args.nshards)}
    text_keys = [k for k in keys if by_key[k]["arm"] == "text"] if "p2" in tests else []
    p4_shard = min(1, args.nshards - 1)
    for k in text_keys:
        assign[0].append(k)
        load[0] += 2 * by_key[k]["tokens"]
    if "p4" in tests:
        load[p4_shard] += sum(5 * by_key[k]["tokens"] for k in P4_RUNS)
    rest = sorted((k for k in keys if k not in text_keys), key=lambda k: -by_key[k]["tokens"])
    for k in rest:
        i = min(range(args.nshards), key=lambda j: load[j])
        assign[i].append(k)
        load[i] += by_key[k]["tokens"]
    mine = assign[args.shard]
    runner = Runner(args.threads, need_plain=("p2" in tests and args.shard == 0 and bool(text_keys)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part = {"shard": args.shard, "nshards": args.nshards, "threads": args.threads,
            "planned_token_steps": load[args.shard], "load_seconds": runner.load_seconds,
            "load_report": runner.model.load_report, "loopfree_linear_layers": runner.loopfree,
            "p1": [], "p2": [], "p3": [], "p4": [], "determinism": None}
    print(f"[shard {args.shard}] {len(mine)} runs, planned {load[args.shard]} token-steps, "
          f"load {runner.load_seconds:.1f}s, {json.dumps(runner.model.load_report)}", flush=True)
    t_all = time.monotonic()

    def save():
        part["conv"] = runner.conv.report()
        part["wall_seconds"] = time.monotonic() - t_all
        tmp = out_dir / f"part_{args.shard}.json.tmp"
        tmp.write_text(json.dumps(part, indent=1) + "\n")
        os.replace(tmp, out_dir / f"part_{args.shard}.json")

    for k in mine:
        row = by_key[k]
        npz = np.load(ORACLE / "npz" / f"{k[0]}__{k[1]}.npz")
        ids, emb, rc, start, amount = runner.static_inputs(row, npz)
        read_steps = set(row["slot_idx"])
        taps_on = "p3" in tests and k in P3_RUNS
        lock = "p2" in tests and row["arm"] == "text" and runner.plain is not None
        t0 = time.monotonic()
        read, sh, planes, taps, p2 = runner.step_loop(ids, emb, rc, start, amount, read_steps,
                                                      layer_taps=taps_on, plain_lockstep=lock)
        secs = time.monotonic() - t0
        rec = p1_record(runner, row, npz, read, sh, planes, secs)
        rec["host"] = {"start": int(start[0]), "amount": int(amount[0]),
                       "image_tokens": int((ids >= runner.V).sum())}
        part["p1"].append(rec)
        if lock:
            p2.update(id=row["id"])
            part["p2"].append(p2)
        msg = (f"[shard {args.shard}] {k[0]}:{k[1]} T={row['tokens']} slots {'ok' if rec['slots_equal'] else 'NO'} "
               f"argmax {'ok' if rec['argmax_all_equal'] else 'NO'} max|dp| {rec['max_abs_dp']:.2e} "
               f"rope {'ok' if rec['rope_planes_equal_oracle'] else 'NO'} {secs:.0f}s")
        if lock:
            msg += f" | P2 bit-equal {p2['bit_equal_steps']}/{p2['steps']} max|d| {p2['max_abs_diff']:.2e}"
        print(msg, flush=True)
        if taps_on:
            part["p3"].append(p3_record(row, npz, taps, row["slot_idx"][-1]))
            w = part["p3"][-1]
            print(f"[shard {args.shard}] P3 {k[0]}:{k[1]} worst max|d| {w['worst_max_abs_diff']:.3e} "
                  f"worst min cos {w['worst_min_cos']:.8f}", flush=True)
        save()

    if "p4" in tests and args.shard == p4_shard:
        for k in P4_RUNS:
            part["p4"].append(run_p4(runner, fx, by_key[k]))
            save()

    # determinism: one short run in every shard, compared across shards at merge
    k = DETERMINISM_RUN
    if k in by_key:
        row = by_key[k]
        npz = np.load(ORACLE / "npz" / f"{k[0]}__{k[1]}.npz")
        ids, emb, rc, start, amount = runner.static_inputs(row, npz)
        read, _, _, _, _ = runner.step_loop(ids, emb, rc, start, amount, set(row["slot_idx"]))
        part["determinism"] = {"id": k[0], "arm": k[1], "letter_logits_hex": [
            np.asarray(read[t][32:42].numpy(), dtype=np.float32).tobytes().hex() for t in row["slot_idx"]]}
    save()
    print(f"[shard {args.shard}] done in {time.monotonic() - t_all:.0f}s; conv {json.dumps(runner.conv.report())}",
          flush=True)


def run_p4(runner: Runner, fx: dict, row: dict) -> dict:
    import torch
    npz = np.load(ORACLE / "npz" / f"{row['id']}__{row['arm']}.npz")
    ids, emb, rc, start, amount = runner.static_inputs(row, npz)
    read_steps = set(row["slot_idx"])

    def probs_of(read):
        return [letter_probs(read[t], n)[1].double().numpy() for t, n in zip(row["slot_idx"], row["nopts"])]

    base_read, _, _, _, _ = runner.step_loop(ids, emb, rc, start, amount, read_steps)
    base = probs_of(base_read)

    def pos1d(is_img, slot, p, image_rc, s, a, planes):
        return p, p, p

    def collapse_to_p_text(is_img, slot, p, image_rc, s, a, planes):
        shift = torch.where(p >= s, a, torch.zeros_like(p))
        pt = p - shift
        return pt, pt, pt

    arms = {
        "image_rc_row_col_swapped": dict(rc=rc[:, [1, 0]].contiguous()),
        "pos1d_0_to_T-1_all_planes": dict(override=pos1d, start=torch.tensor([1 << 30], dtype=torch.int32),
                                          amount=torch.tensor([0], dtype=torch.int32)),
        "planes_collapsed_to_p_text_shift_kept": dict(override=collapse_to_p_text),
        "image_embeds_zero": dict(emb=torch.zeros_like(emb)),
    }
    out = {"id": row["id"], "arm": row["arm"], "base_probs": [b.tolist() for b in base],
           "oracle_probs": row["probs"], "arms": {}}
    red = fx.get("red_arms", {}).get("g256_pos1d", {})
    for name, a in arms.items():
        runner.planes_override = a.get("override")
        try:
            r, _, planes, _, _ = runner.step_loop(ids, a.get("emb", emb), a.get("rc", rc),
                                                  a.get("start", start), a.get("amount", amount), read_steps)
        finally:
            runner.planes_override = None
        pr = probs_of(r)
        rec = {"probs": [x.tolist() for x in pr],
               "max_abs_dp_vs_base": [float(np.abs(x - b).max()) for x, b in zip(pr, base)],
               "argmax": [int(x.argmax()) for x in pr], "base_argmax": [int(b.argmax()) for b in base],
               "argmax_changed": [int(x.argmax()) != int(b.argmax()) for x, b in zip(pr, base)],
               "rope_planes_image_tokens_first4": planes[:, 1:5].tolist()}
        if name.startswith("pos1d") and red.get("row") == row["id"]:
            rec["oracle_red_arm_g256_pos1d_probs"] = red["probs"]
            rec["max_abs_dp_vs_oracle_red_arm"] = [float(np.abs(x - np.asarray(o)).max())
                                                   for x, o in zip(pr, red["probs"])]
        out["arms"][name] = rec
        print(f"P4 {row['id']}:{row['arm']} {name}: max|dp| vs base {rec['max_abs_dp_vs_base']} "
              f"argmax changed {rec['argmax_changed']}", flush=True)
    return out


def versions() -> dict:
    import torch
    import transformers
    import coreai_models
    out = {"python": sys.version.split()[0], "torch": torch.__version__,
           "transformers": transformers.__version__, "numpy": np.__version__,
           "platform": platform.platform()}
    try:
        import importlib.metadata as md
        out["coreai_torch"] = md.version("coreai-torch")
    except Exception:  # noqa: BLE001
        pass
    overlay = Path(coreai_models.__file__).resolve().parents[3]

    def git(*a):  # read-only; --no-optional-locks keeps `status` from rewriting the index
        return subprocess.run(["git", "--no-optional-locks", "-C", str(overlay), *a],
                              capture_output=True, text=True).stdout.rstrip("\n")
    try:
        src = overlay / "python/src/coreai_models"
        out["overlay"] = {
            "path": str(overlay), "branch": git("branch", "--show-current"),
            "rev": git("rev-parse", "--short", "HEAD"),
            "uncommitted_python": git("status", "--porcelain", "--", "python/src").splitlines(),
            "sha256": {f: sha256(src / f) for f in ("models/macos/qwen3_5.py", "models/base.py",
                                                     "primitives/macos/cache.py", "primitives/_ops.py")},
        }
    except Exception as e:  # noqa: BLE001
        out["overlay"] = {"path": str(overlay), "error": str(e)}
    return out


def merge(args) -> None:
    out_dir = Path(args.out_dir)
    parts = [json.loads(p.read_text()) for p in sorted(out_dir.glob("part_*.json"))]
    fx = json.loads((ORACLE / "fixture_oracle.json").read_text())
    expected = {(r["id"], r["arm"]) for r in fx["rows"]}
    p1 = sorted((r for p in parts for r in p["p1"]), key=lambda r: (r["arm"], r["id"]))
    got = {(r["id"], r["arm"]) for r in p1}
    per_arm = {}
    for r in p1:
        a = per_arm.setdefault(r["arm"], {"runs": 0, "slots": 0, "slots_equal_runs": 0, "argmax_equal_slots": 0,
                                          "rope_planes_equal_runs": 0, "max_abs_dp": 0.0, "worst_run": None,
                                          "max_abs_dlogit": 0.0, "min_slot_hidden_cos": 1.0,
                                          "full_vocab_top1_equal_slots": 0})
        a["runs"] += 1
        a["slots"] += len(r["slots"])
        a["slots_equal_runs"] += r["slots_equal"]
        a["rope_planes_equal_runs"] += r["rope_planes_equal_oracle"]
        a["argmax_equal_slots"] += sum(s["argmax_equal"] for s in r["slots"])
        a["full_vocab_top1_equal_slots"] += sum(bool(s["full_vocab_top1_equal_oracle"]) for s in r["slots"])
        a["max_abs_dlogit"] = max(a["max_abs_dlogit"], max(s["max_abs_dlogit"] for s in r["slots"]))
        a["min_slot_hidden_cos"] = min(a["min_slot_hidden_cos"], min(s["slot_hidden_cos"] for s in r["slots"]))
        if r["max_abs_dp"] >= a["max_abs_dp"]:
            a["max_abs_dp"], a["worst_run"] = r["max_abs_dp"], r["id"]
    dps = [s["max_abs_dp"] for r in p1 for s in r["slots"]]
    s1 = {"runs": len(p1), "expected_runs": len(expected), "missing_runs": sorted(":".join(k) for k in expected - got),
          "slots": len(dps), "slots_equal_runs": sum(r["slots_equal"] for r in p1),
          "argmax_equal_slots": sum(s["argmax_equal"] for r in p1 for s in r["slots"]),
          "rope_planes_equal_runs": sum(r["rope_planes_equal_oracle"] for r in p1),
          "max_abs_dp": max(dps) if dps else None, "mean_slot_max_abs_dp": float(np.mean(dps)) if dps else None,
          "bar_max_abs_dp": BAR_MAX_DP, "per_arm": per_arm}
    p1_pass = (s1["runs"] == s1["expected_runs"] and s1["slots_equal_runs"] == s1["runs"]
               and s1["argmax_equal_slots"] == s1["slots"] and s1["max_abs_dp"] is not None
               and s1["max_abs_dp"] <= BAR_MAX_DP)
    p2 = [x for p in parts for x in p["p2"]]
    p2_pass = bool(p2) and all(x["bit_equal_steps"] == x["steps"] for x in p2)
    p3 = [x for p in parts for x in p["p3"]]
    p4 = [x for p in parts for x in p["p4"]]
    det = [p["determinism"] for p in parts if p.get("determinism")]
    det_equal = len(det) > 1 and all(d["letter_logits_hex"] == det[0]["letter_logits_hex"] for d in det)
    conv = {k: sum(p["conv"][k] for p in parts) for k in ("fast_calls", "checked_vs_F_conv1d", "mismatched")}
    conv["max_abs_diff"] = max(p["conv"]["max_abs_diff"] for p in parts)
    here = Path(__file__).resolve()
    record = {
        "schema": "coreai-decider-vision-torch-parity/1",
        "purpose": "ids-input Qwen3.5 VL decoder (S=1, fresh zero states, loop-free GDN step, in-graph "
                   "M-RoPE, fp32 CPU eager) vs the author's fp32 oracle on every fixture run",
        "module": {"path": "conversion/decider_vision/qwen3_5_vl_pipelined.py",
                   "sha256": sha256(here.parent / "qwen3_5_vl_pipelined.py"),
                   "class": "Qwen3_5VLPipelinedForCausalLM", "n_image_max": 256},
        "harness": {"path": "conversion/decider_vision/parity_decoder_torch.py", "sha256": sha256(here),
                    "depthwise_conv": "evaluated as torch.bmm (no oneDNN in this torch build); checked "
                                      "against F.conv1d with torch.equal (all channels, all linear "
                                      "layers) at the last slot step of every run", "conv_check": conv},
        "oracle": {"path": str(ORACLE / "fixture_oracle.json"), "sha256": sha256(ORACLE / "fixture_oracle.json"),
                   "hf_id": fx["source"]["hf_id"], "revision": fx["source"]["revision"],
                   "oracle": fx["oracle"], "image_embeds": "oracle npz image_embeds (HF fp32 tower output), "
                                                          "zero-padded to 256 rows"},
        "contract": {"input_names": ["input_ids", "position_ids", "image_embeds", "image_rc",
                                     "rope_shift_start", "rope_shift_amount"],
                     "image": "ids <|image_pad|> -> V + k (row-major), image_rc[k] = (k // W, k % W), start = "
                              "index of <|vision_end|> = 1 + H*W, amount = H*W - max(H, W)",
                     "text": "image_embeds 0, image_rc 0, start 1<<30, amount 0",
                     "readout": "logits at every slot step of one S=1 pass, letters ids 32..41, softmax(T=1) "
                                "over the first nopts"},
        "versions": versions(),
        "shards": [{k: p[k] for k in ("shard", "threads", "planned_token_steps", "load_seconds", "wall_seconds",
                                      "load_report", "loopfree_linear_layers", "conv")} for p in parts],
        "p1": {"summary": s1, "result": "PASS" if p1_pass else "FAIL", "runs": p1},
        "p2": {"result": "PASS" if p2_pass else "FAIL", "runs": p2},
        "p3": {"runs": p3},
        "p4": {"runs": p4},
        "determinism": {"run": list(DETERMINISM_RUN), "processes": len(det), "letter_logits_bit_equal": det_equal},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.transcript).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transcript).write_text(json.dumps(record, indent=1) + "\n")
    print(f"P1 {record['p1']['result']}: runs {s1['runs']}/{s1['expected_runs']}, slots-equal runs "
          f"{s1['slots_equal_runs']}, argmax {s1['argmax_equal_slots']}/{s1['slots']}, rope planes "
          f"{s1['rope_planes_equal_runs']}/{s1['runs']}, max|dp| {s1['max_abs_dp']}")
    for arm, a in sorted(per_arm.items()):
        print(f"  {arm:7s} runs {a['runs']:3d} slots {a['slots']:3d} argmax {a['argmax_equal_slots']}/{a['slots']} "
              f"max|dp| {a['max_abs_dp']:.3e} (worst {a['worst_run']}) max|dlogit| {a['max_abs_dlogit']:.3e} "
              f"min slot-hidden cos {a['min_slot_hidden_cos']:.9f}")
    print(f"P2 {record['p2']['result']}: " + ", ".join(
        f"{x['id']} {x['bit_equal_steps']}/{x['steps']}" for x in p2))
    for x in p3:
        print(f"P3 {x['id']}:{x['arm']} worst max|d| {x['worst_max_abs_diff']:.3e} worst min cos {x['worst_min_cos']:.9f}")
    for x in p4:
        for name, a in x["arms"].items():
            print(f"P4 {x['id']} {name}: max|dp| {a['max_abs_dp_vs_base']} argmax changed {a['argmax_changed']}")
    print(f"determinism across {len(det)} processes: {det_equal}; conv checks {conv}")
    print(f"transcript: {args.transcript}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", help="comma list id:arm (default: every oracle run)")
    ap.add_argument("--tests", default="p1,p2,p3,p4")
    ap.add_argument("--out-dir", default=str(work_path("_decider2bv", "scratch", "r3_parity")))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--transcript", default=str(
        HERE.parents[1] / "models" / "decider-2b-vision" / "gate-decider-2b-vision-torch-parity.json"))
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
