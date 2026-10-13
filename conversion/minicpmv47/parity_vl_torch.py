#!/usr/bin/env python3
"""fp32 torch parity: the MiniCPM-V-4.7 VL decoder module vs the HF fp32 oracle (P0..P4).

`minicpmv47_vl_pipelined.MiniCPMV47VLPipelinedForCausalLM` is driven with the host contract:
`host_static_inputs` rewrites the processor ids (64 <|image_pad|> -> V + k) and sets
`rope_shift_start` / `rope_shift_amount` by the canvas rule; the image rows are the oracle's own
HF fp32 tower output (`oracle.npz <P>__image_features`), so this isolates the decoder.

  planes   the module's in-graph `_rope_planes` on every position of every prompt, against the
           [3, S] planes the HF rotary received in the oracle's forward (equality per position,
           all three planes), and on the generate rows (prompt + the generated tokens fed back)
           against the planes the rotary received on every generate step; generated positions
           must sit at p - amount on all three planes.
  red      the same plane gate with (a) amount + 1 and (b) every plane collapsed to the 1-D cache
           position (p, p, p) — the 4.6 scheme: both must FAIL on every image prompt.
  logits   one forward over the whole prompt ([1, S], fresh zero states, GDN on the reference
           while_loop path), last position vs the oracle: ordered top-5, cosine, max |d|.
           P0 (text only: start 1 << 30, amount 0, image_embeds 0): the last 48 positions vs the
           oracle, and the same forward through the overlay's plain text graph
           (`Qwen3_5ForCausalLMStateful.forward`, 1-D RoPE) — bit-equal expected.
  Bars (round-1 launch): planes 100 % on P0..P4 + generate rows; red arms FAIL; P1..P4 top-5 4/4,
  cos >= 0.9999.

    QW coreai-models/.venv/bin/python conversion/minicpmv47/parity_vl_torch.py [--threads 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
PROMPTS = ("P0", "P1", "P2", "P3", "P4")
IMAGE_PROMPTS = ("P1", "P2", "P3", "P4")
BAR_COS = 0.9999
N_POS_P0 = 48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plane_cmp(got: np.ndarray, want: np.ndarray, ids: np.ndarray) -> dict:
    """got / want [3, n]: positions where all three planes are equal; the first mismatches."""
    if got.shape != want.shape:
        return {"positions": int(want.shape[1]), "equal": 0, "shape": [list(got.shape), list(want.shape)]}
    eq = (got == want).all(0)
    bad = np.nonzero(~eq)[0]
    return {"positions": int(eq.size), "equal": int(eq.sum()),
            "first_mismatch": [{"index": int(i), "id": int(ids[i]), "oracle_thw": want[:, i].tolist(),
                                "module_thw": got[:, i].tolist()} for i in bad[:4]]}


def logit_cmp(mine, ref) -> dict:
    import torch

    m64, r64 = mine.double().reshape(-1, mine.shape[-1]), ref.double().reshape(-1, ref.shape[-1])
    cos = (m64 * r64).sum(-1) / (m64.norm(dim=-1) * r64.norm(dim=-1))
    t5m = torch.topk(mine.reshape(-1, mine.shape[-1]), 5, dim=-1).indices
    t5r = torch.topk(ref.reshape(-1, ref.shape[-1]), 5, dim=-1).indices
    return {"rows": int(m64.shape[0]), "top5_equal": int((t5m == t5r).all(-1).sum()),
            "argmax_equal": int((t5m[:, 0] == t5r[:, 0]).sum()),
            "cos_min": float(cos.min()), "max_abs_diff": float((m64 - r64).abs().max()),
            "ref_absmax": float(r64.abs().max()),
            "top5": t5m[-1].tolist(), "top5_oracle": t5r[-1].tolist()}


class Runner:
    def __init__(self, threads: int):
        import torch

        torch.set_num_threads(threads)
        self.torch = torch
        from minicpmv47_vl_pipelined import MiniCPMV47VLPipelinedForCausalLM, host_static_inputs

        from coreai_models.models.macos import qwen3_5 as q

        self.q = q
        self.host = host_static_inputs
        t0 = time.perf_counter()
        self.model = MiniCPMV47VLPipelinedForCausalLM.from_checkpoint(
            hf_snapshot(HF_ID, revision=REVISION), torch.float32)
        self.load_seconds = time.perf_counter() - t0
        self.cfg = self.model.config
        self.V = self.cfg.vocab_size

    def planes(self, mapped, start, amount, override=None) -> np.ndarray:
        """In-graph planes for a whole id row at cache positions 0..n-1 -> [3, n] int64."""
        torch = self.torch
        ids = mapped.reshape(1, -1).to(torch.int32)
        n = ids.shape[1]
        p = torch.arange(n, dtype=torch.int32).unsqueeze(0)
        is_img = ids >= self.V
        slot = (ids - self.V).clamp(0, self.model.n_image - 1)
        if override is not None:
            t, h, w = override(ids, is_img, slot, p, start, amount)
        else:
            t, h, w = self.model._rope_planes(ids, is_img, slot, p, start, amount)
        return torch.stack([t[0], h[0], w[0]]).numpy().astype(np.int64)

    def forward(self, mapped, emb, start, amount, plain=False, override=None):
        """One forward over the whole row, fresh zero states -> logits [n, V] fp32. `plain`: the
        overlay's text graph on the same weights (embed gather + 1-D RoPE; text rows only);
        `override`: replaces the module's `_rope_planes` for this call (the red arms)."""
        torch = self.torch
        ids = mapped.reshape(1, -1).to(torch.int32)
        n = ids.shape[1]
        pos = torch.arange(n, dtype=torch.int32).unsqueeze(0)
        st = self.q.build_decode_state(self.cfg, max_seq_len=n, dtype=torch.float32)
        if override is not None:
            self.model._rope_planes = override
        try:
            with torch.inference_mode():
                if plain:
                    out = self.q.Qwen3_5ForCausalLMStateful.forward(
                        self.model, ids, pos, st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
                else:
                    out = self.model(ids, pos, emb, start, amount,
                                     st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
        finally:
            if override is not None:
                del self.model._rope_planes
        return out[0].float()


def run(args) -> dict:
    runner = Runner(args.threads)
    torch = runner.torch
    oracle_dir = Path(args.oracle_dir)
    z = np.load(oracle_dir / "oracle.npz")
    meta = json.loads((oracle_dir / "oracle.json").read_text())
    rep = runner.model.load_report
    print(f"[load] {rep['checkpoint_text_keys']} text tensors, {rep['checkpoint_other_keys_skipped']} skipped, "
          f"{runner.load_seconds:.1f}s", flush=True)
    hidden = runner.cfg.hidden_size
    out = {"prompts": {}, "red": {"amount_plus_1": {}, "planes_1d": {}}}

    def one_d(ids, is_img, slot, p, start, amount):
        return p, p, p

    for P in PROMPTS:
        ids = torch.from_numpy(z[f"{P}__input_ids"])
        mapped, start, amount = runner.host(ids)
        rec = {"tokens": int(ids.numel()), "start": int(start[0]), "amount": int(amount[0]),
               "image_tokens": int((mapped >= runner.V).sum())}
        want = z[f"{P}__planes"]
        got = runner.planes(mapped, start, amount)
        rec["planes_prompt"] = plane_cmp(got, want, ids.numpy())
        if P in IMAGE_PROMPTS:
            rec["canvas_origin"] = int(start[0] - amount[0] - 9)
            rec["image_start_index"] = int((ids == 248078).nonzero()[0, 0])
            gen = torch.from_numpy(z[f"{P}__gen_ids"])
            fed = torch.cat([mapped.long(), gen[:-1]])            # what the generate steps fed back
            g_want = z[f"{P}__gen_rotary_planes"]
            g_got = runner.planes(fed, start, amount)
            rec["planes_generate"] = plane_cmp(g_got, g_want, fed.numpy())
            S = int(ids.numel())
            p_gen = np.arange(S, fed.numel())
            rec["generated_positions_at_p_minus_amount"] = {
                "positions": int(p_gen.size),
                "equal": int((g_want[:, S:] == (p_gen - int(amount[0]))[None]).all(0).sum())}
            for arm, kw in (("amount_plus_1", {"amount": amount + 1}), ("planes_1d", {"override": one_d})):
                bad = runner.planes(mapped, start, kw.get("amount", amount), kw.get("override"))
                c = plane_cmp(bad, want, ids.numpy())
                out["red"][arm][P] = {**c, "gate": "PASS" if c["equal"] == c["positions"] else "FAIL"}
            emb = torch.from_numpy(z[f"{P}__image_features"]).float()
            assert emb.shape == (runner.model.n_image, hidden), emb.shape
            t0 = time.perf_counter()
            logits = runner.forward(mapped, emb, start, amount)
            rec["forward_seconds"] = time.perf_counter() - t0
            rec["logits_last"] = logit_cmp(logits[-1:], torch.from_numpy(z[f"{P}__logits"]).float()[None])
            rec["top5_all_positions_equal_oracle"] = int(
                (torch.topk(logits, 5, dim=-1).indices.numpy() == z[f"{P}__top5_ids"]).all(-1).sum())
            # how much the canvas planes matter: the 1-D arm's last-position logits
            bad_logits = runner.forward(mapped, emb, start, amount, override=one_d)
            rec["logits_last_with_1d_rope"] = logit_cmp(bad_logits[-1:], torch.from_numpy(z[f"{P}__logits"]).float()[None])
            ll = rec["logits_last"]
            print(f"[{P}] S={rec['tokens']} start {rec['start']} amount {rec['amount']} O {rec['canvas_origin']} | "
                  f"planes {rec['planes_prompt']['equal']}/{rec['planes_prompt']['positions']} "
                  f"gen {rec['planes_generate']['equal']}/{rec['planes_generate']['positions']} "
                  f"(p-amount {rec['generated_positions_at_p_minus_amount']['equal']}/"
                  f"{rec['generated_positions_at_p_minus_amount']['positions']}) | top5 {'=' if ll['top5_equal'] else 'NO'} "
                  f"cos {ll['cos_min']:.9f} max|d| {ll['max_abs_diff']:.3e} | 1-D rope: top5 "
                  f"{'=' if rec['logits_last_with_1d_rope']['top5_equal'] else 'changed'} cos "
                  f"{rec['logits_last_with_1d_rope']['cos_min']:.6f} | {rec['forward_seconds']:.1f}s", flush=True)
        else:
            emb = torch.zeros(runner.model.n_image, hidden)
            t0 = time.perf_counter()
            logits = runner.forward(mapped, emb, start, amount)
            rec["forward_seconds"] = time.perf_counter() - t0
            n = logits.shape[0]
            rows = z["P0__logits_positions"]
            assert rows.tolist() == list(range(n - N_POS_P0, n))
            rec["logits_last48"] = logit_cmp(logits[n - N_POS_P0:], torch.from_numpy(z["P0__logits"]).float())
            plain = runner.forward(mapped, emb, start, amount, plain=True)
            rec["text_only_vs_plain_text_graph"] = {"bit_equal": bool(torch.equal(logits, plain)),
                                                    "max_abs_diff": float((logits - plain).abs().max())}
            l48 = rec["logits_last48"]
            print(f"[{P}] L={n} planes {rec['planes_prompt']['equal']}/{rec['planes_prompt']['positions']} | last 48: "
                  f"top5 {l48['top5_equal']}/{l48['rows']} argmax {l48['argmax_equal']}/{l48['rows']} cos min "
                  f"{l48['cos_min']:.9f} max|d| {l48['max_abs_diff']:.3e} | plain text graph bit-equal "
                  f"{rec['text_only_vs_plain_text_graph']['bit_equal']} | {rec['forward_seconds']:.1f}s", flush=True)
        out["prompts"][P] = rec
    for arm, rows in out["red"].items():
        print(f"[red] {arm}: " + ", ".join(f"{P} {r['equal']}/{r['positions']} {r['gate']}" for P, r in rows.items()),
              flush=True)
    out["load_report"] = rep
    out["load_seconds"] = runner.load_seconds
    out["oracle"] = {"npz_sha256": sha256(oracle_dir / "oracle.npz"), "versions": meta["versions"],
                     "threads": meta["oracle"]["threads"], "revision": meta["source"]["revision"]}
    return out


def versions() -> dict:
    import importlib.metadata as md

    import coreai_models
    import torch
    import transformers
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
         "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()}
    for pkg in ("coreai-core", "coreai-torch", "coreai-opt"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    src = Path(coreai_models.__file__).resolve().parent / "models/macos/qwen3_5.py"
    v["overlay_qwen3_5_py"] = {"path": str(src), "sha256": sha256(src)}
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--out", default=str(work_path("_minicpmv47", "oracle", "parity_vl_torch.json")))
    args = ap.parse_args()
    t_all = time.perf_counter()
    res = run(args)
    pr = res["prompts"]
    planes_ok = all(pr[P]["planes_prompt"]["equal"] == pr[P]["planes_prompt"]["positions"] for P in PROMPTS)
    gen_ok = all(pr[P]["planes_generate"]["equal"] == pr[P]["planes_generate"]["positions"]
                 and pr[P]["generated_positions_at_p_minus_amount"]["equal"]
                 == pr[P]["generated_positions_at_p_minus_amount"]["positions"] for P in IMAGE_PROMPTS)
    red_ok = all(r["gate"] == "FAIL" for arm in res["red"].values() for r in arm.values())
    logits_ok = all(pr[P]["logits_last"]["top5_equal"] == 1 and pr[P]["logits_last"]["cos_min"] >= BAR_COS
                    for P in IMAGE_PROMPTS)
    summary = {
        "planes_prompt": {P: f"{pr[P]['planes_prompt']['equal']}/{pr[P]['planes_prompt']['positions']}" for P in PROMPTS},
        "planes_generate": {P: f"{pr[P]['planes_generate']['equal']}/{pr[P]['planes_generate']['positions']}"
                            for P in IMAGE_PROMPTS},
        "generated_at_p_minus_amount": {P: "{equal}/{positions}".format(**pr[P]["generated_positions_at_p_minus_amount"])
                                        for P in IMAGE_PROMPTS},
        "red_arms_fail": {arm: f"{sum(r['gate'] == 'FAIL' for r in rows.values())}/{len(rows)}"
                          for arm, rows in res["red"].items()},
        "top5_equal_last_position": f"{sum(pr[P]['logits_last']['top5_equal'] for P in IMAGE_PROMPTS)}/{len(IMAGE_PROMPTS)}",
        "cos_min_last_position": min(pr[P]["logits_last"]["cos_min"] for P in IMAGE_PROMPTS),
        "max_abs_diff_last_position": max(pr[P]["logits_last"]["max_abs_diff"] for P in IMAGE_PROMPTS),
        "p0_last48": {k: pr["P0"]["logits_last48"][k] for k in ("top5_equal", "argmax_equal", "cos_min", "max_abs_diff")},
        "p0_text_only_bit_equal_plain_text_graph": pr["P0"]["text_only_vs_plain_text_graph"]["bit_equal"],
    }
    record = {
        "schema": "minicpmv47-parity-vl-torch/1",
        "purpose": "ids-input MiniCPM-V-4.7 VL decoder (canvas M-RoPE in-graph, fp32 CPU eager, one forward per "
                   "prompt, fresh zero states, GDN reference while_loop) vs the HF fp32 oracle",
        "result": "PASS" if planes_ok and gen_ok and red_ok and logits_ok else "FAIL",
        "gates": {"planes": planes_ok, "planes_generate": gen_ok, "red_arms_fail": red_ok, "logits": logits_ok},
        "bars": {"planes": "100 % on P0..P4 and the generate rows", "red": "every arm FAILs on P1..P4",
                 "top5_equal_last_position": "4/4", "cos_min": BAR_COS},
        "summary": summary,
        "contract": {"input_names": ["input_ids", "position_ids", "image_embeds", "rope_shift_start", "rope_shift_amount"],
                     "image": "<|image_pad|> x64 -> V + k (row-major 8x8); start = first index after </image> and the "
                              "marker tokens after it; amount = start - (index of <image> + 9)",
                     "text": "image_embeds 0, start 1<<30, amount 0"},
        "module": {"path": "conversion/minicpmv47/minicpmv47_vl_pipelined.py",
                   "sha256": sha256(HERE / "minicpmv47_vl_pipelined.py")},
        "script": {"path": "conversion/minicpmv47/parity_vl_torch.py", "sha256": sha256(Path(__file__))},
        **res,
        "versions": versions(),
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(f"[parity_vl_torch] {record['result']} {json.dumps(summary)}", flush=True)
    print(f"[parity_vl_torch] -> {args.out}")


if __name__ == "__main__":
    main()
