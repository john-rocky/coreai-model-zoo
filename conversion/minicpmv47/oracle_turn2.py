#!/usr/bin/env python3
"""MiniCPM-V-4.7-1B two-turn oracle (round 5, the state lever): a second question on P1's image, fp32 HF greedy.

The question this answers for a chat host: after turn 1 the decoder's state already holds P1's prompt and the
answer A1; can turn 2 continue from that state (feed only the new tokens) and still be the conversation the
model's own chat template describes? It can when the template's render of the two-turn conversation starts
with exactly the ids the state has seen, i.e.

    render(user[image, Q1], assistant A1, user Q2) [: len(P1) + len(A1) + 2]  ==  P1__input_ids + A1 + [<|im_end|>, "\\n"]

(A1 = the oracle's `P1__gen_ids` up to its first stop id, all 32 when it did not stop; Q1 = "What is in this
image?", Q2 = "What colors do you see?"). This script renders the conversation with the processor
(`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` + the 448 earth image,
`max_slice_nums=1`), asserts that prefix (and writes the first difference if it fails), and defines

    P5 = P1__input_ids + A1 + [248046, 198] + the template's second user turn and generation prompt

= the sequence a stateful host feeds. Then, transformers 5.19.0 `MiniCPMV4_7ForConditionalGeneration`, fp32,
CPU, one thread, eager, offline (as `oracle_minicpmv47.py`):
  - `generate(max_new_tokens=32)` greedy on P5 with the image, stop ids honoured -> A2;
  - ONE teacher-forced forward over P5 + A2 (`rope_deltas` reset): the softmax top-2 gap at every A2 step
    (the first-divergence rule's margin), the last prompt position's logits, top-5 at every prompt position,
    turn 1's self-consistency (argmax over A1's positions = A1), and the rotary planes the text model received;
  - the planes against the canvas rule computed here from (ids, start, amount) at every position, and the host
    values `host_static_inputs(P5)` against P1's (start / amount unchanged by the second turn).

Writes `<oracle>/turn2.npz` (keys `P5__*`) and `<oracle>/turn2.json`.

    QW K/.venv-oracle/bin/python -I conversion/minicpmv47/oracle_turn2.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import hf_snapshot, work_path  # noqa: E402
import oracle_minicpmv47 as O  # noqa: E402  (sets HF_HUB_OFFLINE=1)

import numpy as np  # noqa: E402

KNIFE_EDGE = 0.1
Q1 = O.PROMPTS["P1"]["question"]
Q2 = "What colors do you see?"
IM_START, IM_END, NEWLINE = 248045, 248046, 198
USER = 846                       # "user" right after <|im_start|>
GRID = 8


def softmax_top2(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    order = np.argsort(-p, axis=-1, kind="stable")[:, :2]
    rows = np.arange(p.shape[0])
    return order[:, 0], order[:, 1], p[rows, order[:, 0]], p[rows, order[:, 1]]


def host_static_inputs(ids):
    """numpy copy of `minicpmv47_vl_pipelined.host_static_inputs` (that module imports the overlay, absent from the
    oracle venv): (ids with the 64 pads as V + k, start, amount) for one thumbnail after plain text."""
    ids = np.asarray(ids, np.int64)
    i0 = int(np.nonzero(ids == O.IMAGE_START)[0][0])
    pads = np.nonzero(ids == O.IMAGE_PAD)[0]
    assert np.array_equal(pads, np.arange(i0 + 1, i0 + 1 + O.N_IMAGE_TOKENS)) and ids[i0 + 1 + O.N_IMAGE_TOKENS] == O.IMAGE_END
    end = i0 + 2 + O.N_IMAGE_TOKENS
    while end < len(ids) and int(ids[end]) in (248088, 248089, O.IMAGE_START, O.IMAGE_END, NEWLINE):
        end += 1
    mapped = ids.copy()
    mapped[pads] = O.VOCAB + np.arange(O.N_IMAGE_TOKENS)
    return mapped, end, end - (i0 + GRID + 1)


def canvas_planes(ids: np.ndarray, start: int, amount: int) -> np.ndarray:
    """LANE §3 / FEASIBILITY r1 rule for one thumbnail after plain text: [3, n] (t, h, w) per position."""
    n = len(ids)
    O_ = start - amount - (GRID + 1)
    out = np.zeros((3, n), np.int64)
    pads = np.nonzero(ids == O.IMAGE_PAD)[0]
    for p in range(n):
        if p < O_:
            out[:, p] = p
        elif p >= start:
            out[:, p] = p - amount
        elif ids[p] == O.IMAGE_START:
            out[:, p] = (O_, O_ - 1, O_ - 1)
        elif ids[p] == O.IMAGE_END:
            out[:, p] = (O_, O_ + GRID, O_ + GRID)
        elif ids[p] == O.IMAGE_PAD:
            k = int(np.searchsorted(pads, p))
            out[:, p] = (O_, O_ + k // GRID, O_ + k % GRID)
        else:
            out[:, p] = (O_, O_, O_)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle-dir", default=str(work_path("_minicpmv47", "oracle")))
    ap.add_argument("--fixtures", default=str(work_path("_minicpmv47", "oracle", "fixtures")))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    torch.set_num_threads(O.THREADS)
    torch.set_num_interop_threads(O.THREADS)
    oracle_dir = Path(args.oracle_dir)
    z = np.load(oracle_dir / "oracle.npz")
    snap = Path(hf_snapshot(O.HF_ID, revision=O.REVISION))
    t_all = time.perf_counter()
    proc = AutoProcessor.from_pretrained(str(snap))
    tok = proc.tokenizer
    model, load_info, load_secs = O.load_model(snap, torch)
    print(f"[load] {load_secs:.1f}s threads {torch.get_num_threads()} / {torch.get_num_interop_threads()}", flush=True)

    # --- turn 1 as the state saw it -----------------------------------------------------------------
    p1 = z["P1__input_ids"].astype(np.int64)
    g1 = z["P1__gen_ids"].astype(np.int64).tolist()
    stop_at = next((i for i, t in enumerate(g1) if t in O.STOP_IDS), None)
    a1 = g1 if stop_at is None else g1[:stop_at]
    a1_text = tok.decode(a1, skip_special_tokens=False)
    carry_prefix = p1.tolist() + a1 + [IM_END, NEWLINE]

    # --- the template's render of the two-turn conversation ------------------------------------------
    img_path = Path(args.fixtures) / f"{O.PROMPTS['P1']['image']}.448.png"
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q1}]},
            {"role": "assistant", "content": a1_text},
            {"role": "user", "content": Q2}]
    prompt2 = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt2.endswith("<think>\n\n</think>\n\n") and prompt2.count("<|image_pad|>") == 1, prompt2
    inputs = proc(images=[img], text=[prompt2], max_slice_nums=1, return_tensors="pt")
    ids_t = inputs["input_ids"][0].tolist()
    assert inputs["target_sizes"].tolist() == [[32, 32]], inputs["target_sizes"]
    assert sum(t == O.IMAGE_PAD for t in ids_t) == O.N_IMAGE_TOKENS
    pv = inputs["pixel_values"].numpy()
    pixel_equal_p1 = bool(np.array_equal(pv, z["P1__pixel_values"]))

    n_pref = len(carry_prefix)
    prefix_equal = ids_t[:n_pref] == carry_prefix
    first_diff = None
    if not prefix_equal:
        d = next((i for i in range(min(n_pref, len(ids_t))) if ids_t[i] != carry_prefix[i]), min(n_pref, len(ids_t)))
        first_diff = {"index": d, "template_ids": ids_t[max(0, d - 4):d + 8], "carry_ids": carry_prefix[max(0, d - 4):d + 8],
                      "template_text": tok.decode(ids_t[max(0, d - 4):d + 8]),
                      "carry_text": tok.decode(carry_prefix[max(0, d - 4):d + 8])}
    turn2_starts = [i for i in range(len(ids_t) - 1) if ids_t[i] == IM_START and ids_t[i + 1] == USER]
    assert len(turn2_starts) == 2, turn2_starts
    tail = ids_t[turn2_starts[-1]:]
    p5 = np.array(carry_prefix + tail, np.int64)
    if prefix_equal:
        assert p5.tolist() == ids_t, "prefix equal -> P5 is the template's render"
    S = len(p5)
    mapped, start, amount = host_static_inputs(p5)
    _, start1, amount1 = host_static_inputs(p1)
    assert (start, amount) == (start1, amount1) == (73, 58), ((start, amount), (start1, amount1))
    print(f"[P5] P1 {len(p1)} + A1 {len(a1)} + 2 + turn-2 tail {len(tail)} = {S} tokens; template prefix equal "
          f"{prefix_equal}; start/amount {start}/{amount} (P1 {start1}/{amount1}); pixels = P1's {pixel_equal_p1}",
          flush=True)

    kw = {"pixel_values": inputs["pixel_values"], "target_sizes": inputs["target_sizes"]}

    def greedy(ids: np.ndarray) -> tuple[list[int], float]:
        mm = torch.from_numpy((ids == O.IMAGE_PAD).astype(np.int64))[None]
        model.model.rope_deltas = None
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=torch.from_numpy(ids)[None], attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 mm_token_type_ids=mm, max_new_tokens=O.GEN_NEW_TOKENS, do_sample=False, **kw)
        return out[0, len(ids):].tolist(), time.perf_counter() - t0

    a2, gen_secs = greedy(p5)
    print(f"[A2] {len(a2)} tok in {gen_secs:.1f}s: {tok.decode(a2, skip_special_tokens=True)!r}", flush=True)
    a2_template = None
    if not prefix_equal:
        a2_template, _ = greedy(np.array(ids_t, np.int64))

    # --- teacher-forced forward over P5 + A2 -----------------------------------------------------------
    rot, merged = {}, {}
    h_rot = model.model.language_model.rotary_emb.register_forward_pre_hook(
        lambda mod, a: rot.__setitem__("planes", a[1].detach().clone()))
    h_mrg = model.model.merger.register_forward_hook(
        lambda mod, a, out: merged.__setitem__("x", torch.cat(out, 0).detach().clone()))
    full = np.concatenate([p5, np.array(a2, np.int64)])
    mm = np.concatenate([(p5 == O.IMAGE_PAD).astype(np.int64), np.zeros(len(a2), np.int64)])
    model.model.rope_deltas = None
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=torch.from_numpy(full)[None], use_cache=False,
                       mm_token_type_ids=torch.from_numpy(mm)[None], **kw).logits[0].float().numpy()
    fwd_secs = time.perf_counter() - t0
    h_rot.remove()
    h_mrg.remove()
    pos = np.arange(S - 1, S - 1 + len(a2))
    top1, top2, pa, pb = softmax_top2(logits[pos])
    gap = pa - pb
    knife = [int(k) for k in np.nonzero(gap < KNIFE_EDGE)[0]]
    a1_pos = np.arange(len(p1) - 1, len(p1) - 1 + len(a1))
    a1_self = int((logits[a1_pos].argmax(-1) == np.array(a1)).sum())
    p1_last_vs_oracle = float(np.abs(logits[len(p1) - 1] - z["P1__logits"]).max())
    planes = rot["planes"][:, 0].numpy().astype(np.int64)               # [3, S + len(A2)]
    rule = canvas_planes(full, start, amount)
    feats = merged["x"].numpy()
    t5 = np.argsort(-logits[:S], axis=-1, kind="stable")[:, :5]
    t5v = np.take_along_axis(logits[:S], t5, axis=-1)

    arrays = {"P5__input_ids": p5, "P5__mapped_ids": mapped.astype(np.int64), "P5__prompt_len": np.int64(S),
              "P5__turn1_len": np.int64(len(p1) + len(a1)), "P5__a1_ids": np.array(a1, np.int64),
              "P5__rope_shift": np.array([start, amount], np.int64),
              "P5__gen_ids": np.array(a2, np.int64), "P5__margins": gap, "P5__top1": top1, "P5__top2": top2,
              "P5__p1": pa, "P5__p2": pb, "P5__positions": pos, "P5__logits": logits[S - 1],
              "P5__top5_ids": t5, "P5__top5_vals": t5v, "P5__planes": planes[:, :S],
              "P5__image_features": feats.astype(np.float32)}
    if not prefix_equal:
        arrays["P5__template_ids"] = np.array(ids_t, np.int64)
        arrays["P5__template_gen_ids"] = np.array(a2_template, np.int64)
    out_npz = oracle_dir / "turn2.npz"
    np.savez(out_npz, **arrays)
    rec = {
        "schema": "minicpmv47-oracle-turn2/1",
        "conversation": {"turn1_user": ["<image earth.448.png>", Q1], "turn1_assistant_ids": a1,
                         "turn1_assistant_text": a1_text, "turn1_stopped": stop_at is not None,
                         "turn2_user": Q2, "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": False}},
        "template_render": prompt2.replace("<|image_pad|>", "<|image_pad|>"),
        "template_prefix_equals_turn1_state": prefix_equal, "first_difference": first_diff,
        "p5": {"tokens": S, "turn1_tokens": len(p1) + len(a1), "turn1_closing": [IM_END, NEWLINE],
               "turn2_tail_tokens": len(tail), "turn2_tail_text": tok.decode(tail),
               "new_tokens_after_turn1_state": S - (len(p1) + len(a1)),
               "rope_shift_start": start, "rope_shift_amount": amount, "p1_rope_shift": [start1, amount1],
               "rope_shift_unchanged": True, "pixel_values_equal_p1": pixel_equal_p1},
        "a2": {"ids": a2, "text": tok.decode(a2, skip_special_tokens=True), "stopped": bool(a2 and a2[-1] in O.STOP_IDS),
               "generate_seconds": gen_secs},
        "a2_from_template_render": None if a2_template is None else
        {"ids": a2_template, "equal_a2": a2_template == a2, "text": tok.decode(a2_template, skip_special_tokens=True)},
        "teacher_forced": {"forward_seconds": fwd_secs, "steps": len(a2), "knife_edge_steps": knife,
                           "knife_edge_gaps": [float(gap[k]) for k in knife], "min_gap": float(gap.min()),
                           "min_gap_step": int(gap.argmin()), "self_consistency_top1_eq_a2": int((top1 == np.array(a2)).sum()),
                           "turn1_argmax_eq_a1": f"{a1_self}/{len(a1)}",
                           "p1_last_position_vs_oracle_max_abs": p1_last_vs_oracle,
                           "image_features_vs_p1_max_abs": float(np.abs(feats - z["P1__image_features"]).max()),
                           "planes_equal_canvas_rule": bool(np.array_equal(planes, rule)),
                           "planes_mismatch_positions": [int(i) for i in np.nonzero((planes != rule).any(0))[0]][:20],
                           "planes_prompt_prefix_equal_p1_planes": bool(np.array_equal(planes[:, :len(p1)], z["P1__planes"]))},
        "oracle": {"class": type(model).__name__, "dtype": str(model.dtype), "threads": torch.get_num_threads(),
                   "load_seconds": load_secs},
        "versions": O.versions(torch),
        "npz": {"path": str(out_npz), "sha256": O.sha256(out_npz), "keys": sorted(arrays)},
        "script": {"path": "conversion/minicpmv47/oracle_turn2.py", "sha256": O.sha256(Path(__file__))},
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (oracle_dir / "turn2.json").write_text(json.dumps(rec, indent=1) + "\n")
    tf = rec["teacher_forced"]
    print(f"[TF] steps {len(a2)} knife-edge {knife} min gap {tf['min_gap']:.4f}@{tf['min_gap_step']} self "
          f"{tf['self_consistency_top1_eq_a2']}/{len(a2)} turn1 {tf['turn1_argmax_eq_a1']} planes=rule "
          f"{tf['planes_equal_canvas_rule']} P1 last max|d| {p1_last_vs_oracle:.2e} ({fwd_secs:.1f}s)", flush=True)
    print(f"[done] {out_npz} wall {rec['wall_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
