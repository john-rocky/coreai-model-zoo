#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==5.17.0",
#     "torchvision",
#     "pillow",
#     "safetensors>=0.8.0",
#     "huggingface_hub>=1.5.0,<2",
#     "numpy",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""fp32 oracle for the decider-2b-vision readout, from the checkpoint's own code.

Mapika/decider-2b-vision answers lettered questions about an image by the probability of the
option letters at each "Answer: (" slot of one forward pass; it never generates. This script
imports the `decider/` package shipped inside the checkpoint (rev 863e290: `vision.py` sha256
c6cb5350..., `prompt.py` c2fadbe0...; Apache-2.0) unchanged, runs
`VisionDecisionModel.prepare()` -> `slot_logits()` in fp32 on the CPU for every fixture row, and
writes what every later Core AI stage is compared against:

* `npz/<row>__<arm>.npz` per row and arm: input_ids, attention_mask, mm_token_type_ids,
  pixel_values (processor output), image_grid_thw, image_embeds (the merger output the decoder
  receives, [N, 2048]), rope_pos (the three M-RoPE planes the text rotary actually received,
  captured by a hook), slot_idx, slot_hidden, letter logits (masked as the author returns them,
  and raw), probabilities, the full-vocabulary top-5 at each slot; for rows 0 and 1 at g256 also
  the 25 hidden states [25, T, 2048];
* `fixture_oracle.json`: the same per row x arm without the big arrays, plus provenance
  (file sha256s, versions, processor class, which gated-delta / causal-conv implementation ran).

Arms (image rows): `g256` / `g448` = PIL BICUBIC resize to 256x256 / 448x448 before `prepare()`,
so the processor cannot pick its own grid (asserted: image_grid_thw (1,16,16) / (1,28,28));
`native` = the image as drawn, the author's processor chooses the grid. Text rows run once
(arm `text`). The readout is the author's: bare letters A..J (ids 32..41), `-inf` beyond the
option count, softmax at T=1, no BOS, no chat template.

Also asserted on every row: slot count == question count (the author's own assert), the
M-RoPE planes equal the closed form below, the author's masked logits equal the raw letter
logits recomputed from the captured slot hidden state, and a bit-identical re-run of row 0.

M-RoPE closed form (image first; index 0 = <|vision_start|>, merged grid H x W, N = H*W):
  image token k at index 1+k -> (t, h, w) = (1, 1 + k // W, 1 + k % W);
  every later index i -> i - (N - max(H, W)) on all three planes; text-only rows -> i.

    uv run conversion/decider_vision/oracle_decider_vision.py \\
        --public-json models/decider-2b-vision/fixtures-decider-2b-vision.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot, work_path  # noqa: E402

DEFAULT_HF_ID = "Mapika/decider-2b-vision"
DEFAULT_REVISION = "863e290863655f1d6b69324d77d09ac972d21609"
VISION_START, IMAGE_PAD, VISION_END, EOS = 248053, 248056, 248054, 248044
SLOT_TOKEN, COLON, ANSWER = 318, 25, 15666
LETTER_IDS = list(range(32, 42))
MERGE = 2
GRIDS = {"g256": 256, "g448": 448}
NEAR_TIE = 0.02
HIDDEN_ROWS = (0, 1)          # fixture rows whose g256 run also saves every hidden state (round-3 bisection)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> bool:
    """Replace `path` with one rename (a reader never sees a partial file); skip it when the bytes are unchanged."""
    if path.exists() and path.read_bytes() == data:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def save_npz(path: Path, arrays: dict) -> None:
    """np.savez_compressed through a temp file + rename: a reader holding the old file keeps a whole one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def expected_rope(ids: np.ndarray, grid_thw) -> np.ndarray:
    """The closed-form M-RoPE planes for one row (see the module docstring)."""
    T = len(ids)
    if grid_thw is None:
        return np.broadcast_to(np.arange(T, dtype=np.int64), (3, T)).copy()
    gt, gh, gw = (int(v) for v in grid_thw)
    assert gt == 1, grid_thw
    H, W = gh // MERGE, gw // MERGE
    N = H * W
    assert ids[0] == VISION_START and (ids[1:N + 1] == IMAGE_PAD).all() and ids[N + 1] == VISION_END, "image block layout"
    pos = np.zeros((3, T), np.int64)
    k = np.arange(N)
    pos[0, 1:N + 1] = 1
    pos[1, 1:N + 1] = 1 + k // W
    pos[2, 1:N + 1] = 1 + k % W
    pos[:, N + 1:] = np.arange(N + 1, T) - (N - max(H, W))
    return pos


def kernel_report(mq) -> dict:
    """Which implementation each decorated GDN / conv function resolved to, read from its closure."""
    out = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(mq, name)
        env = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in (fn.__closure__ or ()))))
        impl = env.get("implementation")
        out[name] = {
            "implementation": f"{impl.__module__}.{impl.__qualname__}" if impl is not None else None,
            "is_new_implementation": env.get("is_new_implementation"),
        }
    for pkg in ("fla", "causal_conv1d", "kernels"):
        out[f"{pkg}_importable"] = importlib.util.find_spec(pkg) is not None
    return out


class Taps:
    """Hooks on the author's model: rotary input positions, tower output, decoder output, kernel calls."""

    def __init__(self, m, mq):
        self.m, self.mq = m, mq
        self.rope, self.image_embeds, self.vision_last, self.last_hidden = [], [], [], []
        self.override_rope = None
        self.calls = {}
        lm_model = m.lm.model
        lm_model.language_model.rotary_emb.register_forward_pre_hook(self._rope_hook)
        lm_model.register_forward_hook(self._model_hook)
        orig = lm_model.get_image_features

        def get_image_features(*a, **k):
            out = orig(*a, **k)
            self.image_embeds.append([t.detach().clone() for t in out.pooler_output])
            self.vision_last.append(tuple(out.last_hidden_state.shape))
            return out

        lm_model.get_image_features = get_image_features
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                     "causal_conv1d_fn", "causal_conv1d_update"):
            self._count(name)

    def _count(self, name):
        fn = getattr(self.mq, name)
        self.calls[name] = 0

        def wrapped(*a, **k):
            self.calls[name] += 1
            return fn(*a, **k)

        setattr(self.mq, name, wrapped)

    def _rope_hook(self, module, args):
        hidden, pos = args
        self.rope.append(pos.detach().clone())
        if self.override_rope is not None:
            return hidden, self.override_rope(pos)
        return None

    def _model_hook(self, module, args, output):
        self.last_hidden.append(output.last_hidden_state.detach().clone())

    def reset(self):
        self.rope.clear(); self.image_embeds.clear(); self.vision_last.clear(); self.last_hidden.clear()
        for k in self.calls:
            self.calls[k] = 0


def pos1d(pos):
    import torch
    T = pos.shape[-1]
    return torch.arange(T, dtype=pos.dtype, device=pos.device).view(1, 1, T).expand(3, pos.shape[1], T).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--fixtures", default=str(work_path("_decider2bv", "fixtures")),
                    help="dir with rows.json, meta.json, images/ (make_fixture_images.py)")
    ap.add_argument("--out-dir", default=str(work_path("_decider2bv", "oracle")))
    ap.add_argument("--public-json", default=None, help="compact copy for the zoo (models/decider-2b-vision/...)")
    ap.add_argument("--public-max-bytes", type=int, default=1_000_000)
    ap.add_argument("--shared-dir", default=None, help="also copy images/, meta and the oracle JSON here")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rows", default=None, help="comma-separated row ids (debug subset)")
    ap.add_argument("--red-row", default="r14", help="g256 image row for the flip / 1-D-position arms")
    ap.add_argument("--merge", action="store_true",
                    help="with --rows: re-run only those rows and replace them inside the existing fixture_oracle.json; "
                         "their pixel_values / image_embeds must come out byte-identical to the npz already there")
    ap.add_argument("--merge-reason", default="", help="recorded in the revisions list with --merge")
    ap.add_argument("--emit-only", action="store_true",
                    help="skip the model: rewrite --public-json / --shared-dir from an existing fixture_oracle.json")
    args = ap.parse_args()

    if args.emit_only:
        fx = Path(args.fixtures).expanduser()
        out_json = Path(args.out_dir).expanduser() / "fixture_oracle.json"
        full = json.loads(out_json.read_text())
        rows = json.loads((fx / "rows.json").read_text())["rows"]
        assert full["fixture"]["rows_json_sha256"] == sha256_file(fx / "rows.json"), "rows.json changed since the oracle ran"
        meta = {im["name"]: im for im in json.loads((fx / "meta.json").read_text())["images"]}
        header = {k: v for k, v in full.items() if k != "rows"}
        emit(args, header, full["rows"], rows, meta, fx, out_json)
        return 0

    import torch
    import torch.nn.functional as F
    import transformers
    import PIL
    from PIL import Image
    from huggingface_hub import snapshot_download
    import huggingface_hub

    torch.set_num_threads(args.threads)
    t_start = time.monotonic()
    try:                                   # the pinned local snapshot first (works offline) ...
        snapshot = Path(hf_snapshot(args.hf_id, revision=args.revision))
        if not (snapshot / "model.safetensors").exists() or not (snapshot / "decider" / "vision.py").exists():
            raise FileNotFoundError(snapshot)
    except FileNotFoundError:              # ... else fetch exactly that revision
        snapshot = Path(snapshot_download(args.hf_id, revision=args.revision,
                                          allow_patterns=["*.json", "*.jinja", "model.safetensors", "decider/*.py"]))
    sys.path.insert(0, str(snapshot))
    from decider.vision import VisionDecisionModel, IMG            # noqa: E402  (author's code, unchanged)
    from decider.infer import Example, Q                           # noqa: E402
    from decider.prompt import build, LETTERS, MAX_OPTIONS         # noqa: E402
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq       # noqa: E402

    fx = Path(args.fixtures).expanduser()
    all_rows = json.loads((fx / "rows.json").read_text())["rows"]
    meta = {im["name"]: im for im in json.loads((fx / "meta.json").read_text())["images"]}
    rows = all_rows
    if args.rows:
        keep = set(args.rows.split(","))
        rows = [r for r in all_rows if r["id"] in keep]
        assert len(rows) == len(keep), sorted(keep - {r["id"] for r in rows})
    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    prev = None
    if args.merge:
        assert args.rows, "--merge re-runs the rows named by --rows"
        prev = json.loads((out_dir / "fixture_oracle.json").read_text())

    m = VisionDecisionModel(str(snapshot), dtype=torch.float32, grad_ckpt=False).eval()
    torch.set_grad_enabled(False)          # the author's call runs under torch.no_grad(); so does everything here
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert (m.slot_tok, m.colon, m.answer_tok) == (SLOT_TOKEN, COLON, ANSWER), (m.slot_tok, m.colon, m.answer_tok)
    assert m.letters.tolist() == LETTER_IDS and MAX_OPTIONS == 10 and LETTERS == "ABCDEFGHIJ"
    assert m.tok.bos_token_id is None and m.tok.encode("x") == m.tok.encode("x", add_special_tokens=False)
    tied = m.lm.lm_head.weight.data_ptr() == m.lm.model.language_model.embed_tokens.weight.data_ptr()
    kernels = kernel_report(mq)
    taps = Taps(m, mq)
    w_letters = m.lm.lm_head.weight[m.letters]
    ip = m.proc.image_processor
    visual = m.lm.model.visual

    class _Keep:
        def shuffle(self, x): pass
        def sample(self, xs, k): return xs[:k]

    def example(r):
        return Example(r["context"], [Q(qq["text"], qq["options"], qq["gold"]) for qq in r["questions"]])

    def load_image(name):
        return Image.open(fx / meta[name]["path"]).convert("RGB")

    def arm_image(r, arm):
        if r["image"] is None:
            return None
        im = load_image(r["image"])
        if arm in GRIDS:
            im = im.resize((GRIDS[arm], GRIDS[arm]), Image.Resampling.BICUBIC)
        elif arm == "g256_flip":
            im = im.resize((256, 256), Image.Resampling.BICUBIC).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif arm == "g256_pos1d":
            im = im.resize((256, 256), Image.Resampling.BICUBIC)
        return im

    def run(r, arm, hidden_states=False, rope_override=None):
        ex = example(r)
        img = arm_image(r, arm)
        b = build(ex, m.tok, _Keep(), max_ctx_tokens=1536)
        taps.reset()
        taps.override_rope = rope_override
        t0 = time.perf_counter()
        inp = m.prepare([(img, ex)])
        with torch.no_grad():
            lg = m.slot_logits(inp)
        wall = time.perf_counter() - t0
        taps.override_rope = None
        calls = dict(taps.calls)
        ids = inp["input_ids"][0].numpy().astype(np.int64)
        T = len(ids)
        nq = len(r["questions"])
        slot_idx = inp["slot_idx"].numpy().astype(np.int64)
        assert len(slot_idx) == nq and (inp["slot_batch"] == 0).all(), (r["id"], arm, slot_idx)
        nopts = inp["nopts"].numpy().astype(np.int64)
        assert nopts.tolist() == [len(qq["options"]) for qq in r["questions"]]
        assert len(taps.last_hidden) == 1 and len(taps.rope) == 1, (len(taps.last_hidden), len(taps.rope))
        h = taps.last_hidden[0][0]                                   # [T, 2048]
        slot_hidden = h[torch.from_numpy(slot_idx)]
        raw = F.linear(slot_hidden, w_letters).float()
        lg_np = lg.numpy()
        raw_np = raw.numpy()
        valid = np.arange(MAX_OPTIONS)[None, :] < nopts[:, None]
        assert np.array_equal(lg_np[valid], raw_np[valid]) and np.isneginf(lg_np[~valid]).all(), (r["id"], arm)
        probs = torch.softmax(lg, -1).numpy()
        full = m.lm.lm_head(slot_hidden).float()
        top5 = torch.topk(full, 5, dim=-1)
        rope = taps.rope[0][:, 0].numpy().astype(np.int64)          # [3, T]
        grid = inp["image_grid_thw"][0].tolist() if "image_grid_thw" in inp else None
        rec = {
            "id": r["id"], "arm": arm, "form": r["form"], "image": r["image"],
            "image_size_in": meta[r["image"]]["size"] if r["image"] else None,
            "image_size_fed": list(img.size) if img is not None else None,
            "grid_thw": grid, "n_image_tokens": int((ids == IMAGE_PAD).sum()), "tokens": T,
            "ids": ids.tolist(), "slot_idx": slot_idx.tolist(), "nopts": nopts.tolist(),
            "questions": [qq["text"] for qq in r["questions"]], "options": [qq["options"] for qq in r["questions"]],
            "gold": [qq["gold"] for qq in r["questions"]],
            "letter_logits": [lg_np[s, :n].tolist() for s, n in enumerate(nopts)],
            "probs": [probs[s, :n].tolist() for s, n in enumerate(nopts)],
            "wall_s": wall, "kernel_calls": calls,
        }
        am, margin, tie = [], [], []
        for s, n in enumerate(nopts):
            p = probs[s, :n]
            o = np.sort(p)[::-1]
            am.append(int(p.argmax())); margin.append(float(o[0] - o[1])); tie.append(bool(o[0] - o[1] < NEAR_TIE))
        rec.update(argmax=am, argmax_label=[LETTERS[a] for a in am], argmax_token_id=[LETTER_IDS[a] for a in am],
                   gold_correct=[a == g for a, g in zip(am, rec["gold"])], top2_margin=margin, near_tie=tie)
        t1 = top5.indices[:, 0].tolist()
        rec.update(full_vocab_top1_id=t1,
                   full_vocab_top1_is_letter=[i in LETTER_IDS for i in t1],
                   full_vocab_top1_is_argmax_letter=[i == LETTER_IDS[a] for i, a in zip(t1, am)],
                   full_vocab_top5_ids=top5.indices.tolist(), full_vocab_top5_logits=top5.values.tolist())
        # Text part must be the author's build() ids, unchanged by the decode -> re-tokenize round trip.
        off = 0
        if img is not None:
            assert len(taps.image_embeds) == 1 and len(taps.image_embeds[0]) == 1
            emb = taps.image_embeds[0][0]
            n_img = rec["n_image_tokens"]
            gt, gh, gw = grid
            assert tuple(emb.shape) == (n_img, 2048) and n_img == gt * gh * gw // MERGE ** 2, (emb.shape, grid)
            assert taps.vision_last[0] == (gt * gh * gw, 1024), taps.vision_last   # pre-merger rows = patches
            off = n_img + 2
            rec["image_embeds_shape"] = list(emb.shape)
        rec["retokenized_equal"] = ids[off:].tolist() == b["ids"]
        rec["slots_equal_build"] = slot_idx.tolist() == [s + off for s in b["slots"]]
        if rope_override is None:
            exp = expected_rope(ids, grid)
            ok = bool(np.array_equal(rope, exp))
            rec["rope_formula_equal"] = ok
            if not ok:
                bad = np.argwhere(rope != exp)[:8].tolist()
                raise AssertionError(f"M-RoPE planes differ from the closed form on {r['id']}/{arm}: "
                                     f"first mismatches {bad}; captured cols 0..4 {rope[:, :5].tolist()}")
            if grid is not None:
                H, W = grid[1] // MERGE, grid[2] // MERGE
                rec["rope_shift"] = int(rec["n_image_tokens"] - max(H, W))
        arrays = {
            "input_ids": ids, "attention_mask": inp["attention_mask"][0].numpy().astype(np.int64),
            "rope_pos": rope, "slot_idx": slot_idx, "nopts": nopts,
            "slot_hidden": slot_hidden.numpy().astype(np.float32),
            "letter_logits": lg_np.astype(np.float32), "letter_logits_raw": raw_np.astype(np.float32),
            "probs": probs.astype(np.float32),
            "full_top5_ids": top5.indices.numpy().astype(np.int64),
            "full_top5_logits": top5.values.numpy().astype(np.float32),
        }
        if "mm_token_type_ids" in inp:
            arrays["mm_token_type_ids"] = inp["mm_token_type_ids"][0].numpy().astype(np.int64)
        if img is not None:
            arrays["pixel_values"] = inp["pixel_values"].numpy().astype(np.float32)
            arrays["image_grid_thw"] = inp["image_grid_thw"].numpy().astype(np.int64)
            arrays["image_embeds"] = taps.image_embeds[0][0].numpy().astype(np.float32)
        if hidden_states:
            kw = {k: v for k, v in inp.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
            taps.reset()
            with torch.no_grad():
                o = m.lm.model(**kw, use_cache=False, output_hidden_states=True)
            hs = torch.stack(o.hidden_states)[:, 0]                   # [25, T, 2048]
            arrays["hidden_states"] = hs.numpy().astype(np.float32)
            rec["hidden_states"] = {
                "shape": list(hs.shape),
                "rerun_last_hidden_bit_equal": bool(torch.equal(o.last_hidden_state[0], h)),
                "last_equals_final_norm_output": bool(torch.equal(hs[-1], h)),
            }
        return rec, arrays

    print(f"loaded {args.hf_id}@{args.revision[:7]} fp32 on cpu, threads {torch.get_num_threads()}, "
          f"{len(rows)} rows, {time.monotonic() - t_start:.0f}s", flush=True)
    records = []
    hidden_ids = {all_rows[i]["id"] for i in HIDDEN_ROWS}     # fixture rows 0 and 1, whatever subset runs
    for r in rows:
        arms = ["text"] if r["image"] is None else ["g256", "g448", "native"]
        for arm in arms:
            rec, arrays = run(r, arm, hidden_states=(arm == "g256" and r["id"] in hidden_ids))
            if arm in GRIDS:
                g = GRIDS[arm] // 16
                assert rec["grid_thw"] == [1, g, g], ("processor chose its own grid", r["id"], arm, rec["grid_thw"])
            final = out_dir / "npz" / f"{r['id']}__{arm}.npz"
            if args.merge:                       # the image side must not move: same pixels in, same tower out
                with np.load(final) as old:
                    for k in ("pixel_values", "image_embeds"):
                        same = (k in old.files and k in arrays and old[k].dtype == arrays[k].dtype
                                and old[k].shape == arrays[k].shape and old[k].tobytes() == arrays[k].tobytes())
                        rec[f"npz_{k}_byte_identical_to_previous"] = same
                        assert same, (r["id"], arm, k, "changed")
            save_npz(final, arrays)
            records.append(rec)
            ps = " | ".join(f"{LETTERS[a]} {max(p):.3f} m{mg:.3f}" for a, p, mg in zip(rec["argmax"], rec["probs"], rec["top2_margin"]))
            print(f"{r['id']:4s} {arm:6s} T={rec['tokens']:4d} img={rec['n_image_tokens']:3d} grid={rec['grid_thw']} "
                  f"{ps}  {rec['wall_s']:.1f}s", flush=True)

    # Determinism: the first row again, g256 (or text), letter logits must be bit-identical.
    first = rows[0]
    arm0 = "text" if first["image"] is None else "g256"
    again, _ = run(first, arm0)
    base = next(x for x in records if x["id"] == first["id"] and x["arm"] == arm0)
    determinism = {"row": first["id"], "arm": arm0,
                   "letter_logits_bit_equal": again["letter_logits"] == base["letter_logits"]}
    assert determinism["letter_logits_bit_equal"], determinism
    determinism_merge = None
    if prev is not None:                         # keep the full run's record; this run's own check goes beside it
        determinism_merge, determinism = determinism, prev["determinism"]

    # Red arms (can the later gates see a position / content error?): one g256 image row.
    red = {}
    rr = next((x for x in rows if x["id"] == args.red_row), None)
    if rr is not None and rr["image"] is not None:
        base = next(x for x in records if x["id"] == rr["id"] and x["arm"] == "g256")
        for arm, override in (("g256_flip", None), ("g256_pos1d", pos1d)):
            rec, arrays = run(rr, arm, rope_override=override)
            save_npz(out_dir / "npz" / f"{rr['id']}__{arm}.npz", arrays)
            dp = [float(np.abs(np.array(a) - np.array(b)).max()) for a, b in zip(rec["probs"], base["probs"])]
            red[arm] = {"row": rr["id"], "probs": rec["probs"], "argmax": rec["argmax"], "base_probs": base["probs"],
                        "base_argmax": base["argmax"], "max_abs_dp_vs_g256": dp,
                        "argmax_changed": [a != b for a, b in zip(rec["argmax"], base["argmax"])]}
            if arm == "g256_pos1d":
                red[arm]["rope_captured_is_3d"] = True  # the hook saw the full planes and replaced them with 0..T-1
            print(f"red {arm}: probs {[[round(v, 4) for v in p] for p in rec['probs']]} vs g256 "
                  f"{[[round(v, 4) for v in p] for p in base['probs']]} max|dp| {dp}", flush=True)

    if prev is not None:
        red = prev["red_arms"] if not red else red
        new = {(x["id"], x["arm"]): x for x in records}
        old_keys = [(x["id"], x["arm"]) for x in prev["rows"]]
        assert set(new) <= set(old_keys), sorted(set(new) - set(old_keys))
        records = [new.get(k, x) for k, x in zip(old_keys, prev["rows"])]
        rows = all_rows

    # Summary per arm.
    def slots_of(arm):
        return [(x, s) for x in records if x["arm"] == arm for s in range(len(x["nopts"]))]

    summary = {"rows": len(rows), "image_rows": sum(r["image"] is not None for r in rows),
               "text_rows": sum(r["image"] is None for r in rows),
               "slots_per_arm_set": sum(len(r["questions"]) for r in rows), "arms": {}}
    for arm in ("g256", "g448", "native", "text"):
        ss = slots_of(arm)
        if not ss:
            continue
        mg = [x["top2_margin"][s] for x, s in ss]
        summary["arms"][arm] = {
            "row_runs": len({x["id"] for x, _ in ss}), "slots": len(ss),
            "min_top2_margin": min(mg), "near_tie": sum(x["near_tie"][s] for x, s in ss),
            "margin_ge_0.1": sum(v >= 0.1 for v in mg), "frac_margin_ge_0.1": sum(v >= 0.1 for v in mg) / len(mg),
            "full_vocab_top1_is_letter": sum(x["full_vocab_top1_is_letter"][s] for x, s in ss),
            "full_vocab_top1_is_argmax_letter": sum(x["full_vocab_top1_is_argmax_letter"][s] for x, s in ss),
            "gold_correct": sum(x["gold_correct"][s] for x, s in ss),
            "max_tokens": max(x["tokens"] for x, _ in ss),
            "rope_formula_equal_rows": sum(bool(x.get("rope_formula_equal")) for x in records if x["arm"] == arm),
            "retokenized_equal_rows": sum(x["retokenized_equal"] for x in records if x["arm"] == arm),
            "mean_wall_s": float(np.mean([x["wall_s"] for x in records if x["arm"] == arm])),
        }
    # Grid price on the fixture itself: native vs g256 / g448 on the same image rows.
    price = {}
    for arm in ("g256", "g448"):
        dps, agree, n = [], 0, 0
        for x in records:
            if x["arm"] != "native":
                continue
            y = next(z for z in records if z["id"] == x["id"] and z["arm"] == arm)
            for s in range(len(x["nopts"])):
                dps.append(float(np.abs(np.array(x["probs"][s]) - np.array(y["probs"][s])).max()))
                agree += x["argmax"][s] == y["argmax"][s]; n += 1
        price[arm] = {"slots": n, "argmax_agree_with_native": agree, "mean_max_abs_dp": float(np.mean(dps)),
                      "max_abs_dp": float(max(dps))}
    summary["fixture_grid_price_vs_native"] = price
    summary["wall_seconds_total"] = time.monotonic() - t_start
    if prev is not None:
        summary["wall_seconds_total"] = prev["summary"]["wall_seconds_total"]
        summary["merge_run_wall_seconds"] = time.monotonic() - t_start

    snap_files = {n: sha256_file(snapshot / n) for n in
                  ("decider/vision.py", "decider/prompt.py", "decider/infer.py", "config.json",
                   "processor_config.json", "tokenizer.json", "tokenizer_config.json")}
    snap_files["model.safetensors"] = sha256_file(snapshot / "model.safetensors")
    import torchvision
    header = {
        "schema": "coreai-decider-vision-oracle/1",
        "source": {"hf_id": args.hf_id, "revision": args.revision, "sha256": snap_files},
        "oracle": "checkpoint decider/vision.py VisionDecisionModel.prepare() -> slot_logits(), fp32, CPU, torch.no_grad",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "torchvision": torchvision.__version__,
                     "pillow": PIL.__version__, "numpy": np.__version__, "huggingface_hub": huggingface_hub.__version__},
        "device": "cpu", "dtype": "float32", "torch_num_threads": torch.get_num_threads(),
        "platform": {"machine": platform.machine(), "macos": platform.mac_ver()[0], "processor": platform.processor()},
        "processor": {"processor_class": type(m.proc).__name__, "image_processor_class": type(ip).__name__,
                      "image_processor_mro": [c.__name__ for c in type(ip).__mro__],
                      "tokenizer_class": type(m.tok).__name__,
                      "size": str(ip.size),
                      "resample": int(ip.resample), "patch_size": ip.patch_size, "merge_size": ip.merge_size,
                      "temporal_patch_size": ip.temporal_patch_size, "image_mean": list(ip.image_mean),
                      "image_std": list(ip.image_std)},
        "kernels": kernels,
        "attn_implementation": {"text": m.lm.config.text_config._attn_implementation,
                                "vision": m.lm.config.vision_config._attn_implementation},
        "tied_lm_head": tied,
        "tower": {"num_grid_per_side": visual.num_grid_per_side, "pos_embed_interpolation": visual.interpolation_mode,
                  "align_corners": visual.interpolation_align_corners, "patch": 16, "spatial_merge": MERGE},
        "contract": {"image_prefix": IMG, "slot_token": SLOT_TOKEN, "colon": COLON, "answer": ANSWER,
                     "letters": [[LETTERS[i], LETTER_IDS[i]] for i in range(10)], "max_options": MAX_OPTIONS,
                     "temperature": 1.0, "bos": None, "chat_template": None, "max_ctx_tokens": 1536,
                     "special": {"vision_start": VISION_START, "image_pad": IMAGE_PAD, "vision_end": VISION_END, "eos": EOS}},
        "mrope_closed_form": "index 0 = vision_start -> 0; image token k -> (1, 1 + k // W, 1 + k % W); "
                             "later index i -> i - (N - max(H, W)) on all planes (N = H*W merged); text-only -> i",
        "arms": {"g256": "PIL BICUBIC resize to 256x256 before prepare()", "g448": "PIL BICUBIC resize to 448x448",
                 "native": "the image as drawn; the processor picks the grid", "text": "no image",
                 "g256_flip": "g256 image flipped top-bottom (red arm)",
                 "g256_pos1d": "g256, text-rotary positions replaced by 0..T-1 on all three planes (red arm)"},
        "fixture": {"rows_json_sha256": sha256_file(fx / "rows.json"), "meta_json_sha256": sha256_file(fx / "meta.json"),
                    "near_tie_threshold": NEAR_TIE},
        "determinism": determinism,
        "red_arms": red,
        "summary": summary,
    }
    if prev is not None:
        header["determinism_merge_run"] = determinism_merge
        header["revisions"] = prev.get("revisions", []) + [{
            "rows_rerun": [r["id"] for r in (x for x in all_rows if x["id"] in set(args.rows.split(",")))],
            "reason": args.merge_reason, "previous_rows_json_sha256": prev["fixture"]["rows_json_sha256"]}]
    full = dict(header, rows=records)
    out_json = out_dir / "fixture_oracle.json"
    write_atomic(out_json, (json.dumps(full, indent=1, ensure_ascii=False) + "\n").encode())
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_json} ({out_json.stat().st_size} B)")

    emit(args, header, records, rows, meta, fx, out_json)
    return 0


def emit(args, header, records, rows, meta, fx, out_json):
    if args.public_json:
        pub = public_view(header, records, rows)
        text = json.dumps(pub, indent=1, ensure_ascii=False) + "\n"
        if len(text.encode()) > args.public_max_bytes:
            pub = public_view(header, [x for x in records if x["arm"] in ("g256", "text")], rows, trimmed=True)
            text = json.dumps(pub, indent=1, ensure_ascii=False) + "\n"
        p = Path(args.public_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        changed = write_atomic(p, text.encode())
        print(f"{'wrote' if changed else 'unchanged'} {p} ({len(text.encode())} B, arms {sorted({x['arm'] for x in pub['rows']})})")

    if args.shared_dir:
        sd = Path(args.shared_dir).expanduser()
        (sd / "images").mkdir(parents=True, exist_ok=True)
        pairs = [(fx / im["path"], sd / "images" / Path(im["path"]).name) for im in meta.values()]
        pairs += [(fx / "meta.json", sd / "fixtures_meta.json"), (fx / "rows.json", sd / "fixtures_rows.json"),
                  (out_json, sd / "oracle_fixture.json")]
        changed = [dst.name for src, dst in pairs if write_atomic(dst, src.read_bytes())]
        print(f"shared copies -> {sd}: rewritten {changed if changed else 'none (all byte-identical)'}")


PUBLIC_KEYS = {  # fixture_oracle.json key -> decider-0.8b fixture key
    "slot_idx": "slots", "letter_logits": "slot_logits", "probs": "p_oracle",
}


def public_view(header, records, rows, trimmed=False):
    """The zoo copy: the per-row readout without the full-vocab arrays, keys as in decider-0.8b's fixture."""
    keep = ("id", "arm", "form", "image", "image_size_in", "image_size_fed", "grid_thw", "n_image_tokens",
            "questions", "options", "gold", "ids", "tokens", "slot_idx", "nopts", "letter_logits", "probs",
            "argmax", "argmax_label", "argmax_token_id", "top2_margin", "near_tie", "gold_correct",
            "full_vocab_top1_id", "full_vocab_top1_is_letter", "rope_formula_equal", "rope_shift")
    out_rows = []
    for x in records:
        y = {PUBLIC_KEYS.get(k, k): x[k] for k in keep if k in x}
        y["request_id"] = x["id"]
        y["id"] = f"{x['id']}-{x['arm']}"
        y["label_ids"] = [LETTER_IDS[:n] for n in x["nopts"]]
        y["label_strings"] = [list("ABCDEFGHIJ"[:n]) for n in x["nopts"]]
        out_rows.append(y)
    pub = {
        "schema": "coreai-decider-vision-fixtures/1",
        "source": dict(header["source"], oracle=header["oracle"], torch=header["versions"]["torch"],
                       transformers=header["versions"]["transformers"]),
        "temperature": 1.0, "layout": "image_first", "letters": "bare A..J after the ' (' slot token",
        "label_table": header["contract"]["letters"],
        "contract": header["contract"], "mrope_closed_form": header["mrope_closed_form"], "arms": header["arms"],
        "processor": {k: header["processor"][k] for k in ("processor_class", "image_processor_class", "resample")},
        "images_license": "self-made (generated by conversion/decider_vision/make_fixture_images.py), CC0-1.0",
        "requests": rows,
        "rows": out_rows,
        "summary": header["summary"],
        "determinism": header["determinism"],
    }
    if trimmed:
        pub["trimmed"] = "arms g256 + text only; the full oracle (native, g448) is kept outside the repo"
    return pub


if __name__ == "__main__":
    raise SystemExit(main())
