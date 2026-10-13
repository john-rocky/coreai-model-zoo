#!/usr/bin/env python3
"""MiniCPM-V-4.7-1B oracle: transformers' native `minicpmv4_7` in fp32 on the CPU, one thread.

The reference every later gate of this port is measured against. Run in an environment with
transformers 5.19.0 (the first release with `minicpmv4_7`); the zoo's overlay venv (4.57.6) has
no such model. Weights: the pinned snapshot, offline.

Prompts (thinking off: `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`,
so every prompt ends in `<think>\\n\\n</think>\\n\\n`):

  P0  text only, "Write one sentence about the ocean." Greedy 48 tokens with the HF cache, the
      stop tokens NOT honoured (48 positions are needed whatever the answer length; the first
      stop token's index is recorded), then ONE forward over prompt + 48: the logits of the last
      48 positions [48, V] and the argmax / top-5 of every position.
  P1  earth.448.png     "What is in this image?"
  P2  saturn_v.448.png  "Describe this image in one sentence."
  P3  synthetic.448.png "What text is written in the image?"
  P4  system "You are a concise assistant." + earth.448.png "What colors do you see?"

Image prompts: the fixture is already 448x448 (Pillow BICUBIC, `make_fixtures.py`) and goes to
the processor with `max_slice_nums=1` — thumbnail only. Asserted before anything runs:
`target_sizes == [[32, 32]]`, 64 `<|image_pad|>`, and the block `<image_id>0</image_id><image>`
+ pad x64 + `</image>`. Per image prompt: the tower output `image_features` [64, 1024] (the direct
`get_image_features(...).pooler_output`, checked equal to what the merger handed the forward),
the prompt's single forward (last-position logits over the full vocab, top-5 everywhere), and
`generate(max_new_tokens=32)` greedy with the stop tokens honoured (the answer).

M-RoPE planes are taken by hooks, never recomputed: a forward pre-hook on
`model.model.language_model` keeps the `position_ids` it is handed (None for P0, [3, 1, S] for an
image prompt's forward, [4, 1, s] = [text, T, H, W] on every generate step), and a pre-hook on its
`rotary_emb` keeps the [3, 1, s] planes the rotary actually received. `get_rope_index` is never
called here.

Also checked: every checkpoint tensor equals the loaded parameter (bf16 -> fp32 is exact), the
LM head is the checkpoint's own `lm_head.weight` (untied), and the processor's pixel_values
against the plain host formula ((x / 255 - 0.5) / 0.5, 14x14 patches) on the same 448 image.

Writes `<out>/oracle.npz` (keys `<P>__<name>`) and `<out>/oracle.json`.

    K/.venv-oracle/bin/python -I conversion/minicpmv47/oracle_minicpmv47.py [--prompts P0,P1] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import hf_snapshot, work_path  # noqa: E402

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HF_ID = "openbmb/MiniCPM-V-4.7-1B"
REVISION = "984a89aafcd5160164f4f0d04b4be0f4eec03cb8"
VOCAB = 248094
IMAGE_PAD, IMAGE_START, IMAGE_END = 248056, 248078, 248079
STOP_IDS = (248044, 248046)
N_IMAGE_TOKENS = 64
P0_NEW_TOKENS = 48
GEN_NEW_TOKENS = 32
THREADS = 1
PROMPTS = {
    "P0": {"image": None, "system": None, "question": "Write one sentence about the ocean."},
    "P1": {"image": "earth", "system": None, "question": "What is in this image?"},
    "P2": {"image": "saturn_v", "system": None, "question": "Describe this image in one sentence."},
    "P3": {"image": "synthetic", "system": None, "question": "What text is written in the image?"},
    "P4": {"image": "earth", "system": "You are a concise assistant.", "question": "What colors do you see?"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Recorder:
    """Pre-hooks on the text model and its rotary: what position ids each call received."""

    def __init__(self, model):
        self.calls: list[dict] = []
        lm = model.model.language_model
        self.handles = [lm.register_forward_pre_hook(self._lm, with_kwargs=True),
                        lm.rotary_emb.register_forward_pre_hook(self._rotary)]

    def _lm(self, mod, args, kwargs):
        pid = kwargs.get("position_ids")
        emb = kwargs.get("inputs_embeds")
        self.calls.append({"lm_position_ids": None if pid is None else pid.detach().clone(),
                           "seq": None if emb is None else int(emb.shape[1]), "rotary": None})

    def _rotary(self, mod, args):
        self.calls[-1]["rotary"] = args[1].detach().clone()

    def take(self) -> list[dict]:
        out, self.calls = self.calls, []
        return out


def top5(logits):
    import torch
    v, i = torch.topk(logits.float(), 5, dim=-1)
    return i.numpy().astype(np.int64), v.numpy().astype(np.float32)


def host_pixels(img448) -> "np.ndarray":
    """(x / 255 - 0.5) / 0.5 in the processor's packing: [1, 3, 14, 1024 * 14]."""
    x = np.asarray(img448.convert("RGB"), dtype=np.float32)          # [448, 448, 3]
    x = (x / 255.0 - 0.5) / 0.5
    c = x.transpose(2, 0, 1)                                         # [3, 448, 448]
    g = 448 // 14
    t = c.reshape(3, g, 14, g, 14).transpose(0, 2, 1, 3, 4)          # [3, i, py, px, j]
    return t.reshape(3, 14, g * g * 14)[None]


def check_image_block(ids: list[int], tok) -> dict:
    """The thumbnail-only block, asserted: <image_id>0</image_id><image> + pad x64 + </image>."""
    starts = [i for i, t in enumerate(ids) if t == IMAGE_START]
    assert len(starts) == 1, f"<image> count {len(starts)}"
    i0 = starts[0]
    pads = [i for i, t in enumerate(ids) if t == IMAGE_PAD]
    assert pads == list(range(i0 + 1, i0 + 1 + N_IMAGE_TOKENS)), "64 contiguous <|image_pad|> after <image>"
    assert ids[i0 + 1 + N_IMAGE_TOKENS] == IMAGE_END, "</image> right after the pads"
    head = tok.decode(ids[max(0, i0 - 3):i0 + 1])
    assert head == "<image_id>0</image_id><image>", f"block head {head!r}"
    block = tok.decode(ids[i0 - 3:i0 + 2 + N_IMAGE_TOKENS])
    assert block == "<image_id>0</image_id><image>" + "<|image_pad|>" * N_IMAGE_TOKENS + "</image>", block[:80]
    return {"image_start_index": i0, "image_end_index": i0 + 1 + N_IMAGE_TOKENS,
            "block_head_ids": ids[i0 - 3:i0 + 1], "after_end_ids": ids[i0 + 2 + N_IMAGE_TOKENS:i0 + 6 + N_IMAGE_TOKENS],
            "block_text": "<image_id>0</image_id><image>" + "<|image_pad|>x64" + "</image>"}


def load_model(snap: Path, torch):
    from transformers import MiniCPMV4_7ForConditionalGeneration

    t0 = time.perf_counter()
    model, info = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
        str(snap), dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
    model.eval()
    secs = time.perf_counter() - t0
    info = {k: sorted(map(str, v)) if isinstance(v, (list, set, tuple)) else str(v) for k, v in info.items()}
    return model, info, secs


def check_weights(model, snap: Path, torch) -> dict:
    """Every checkpoint tensor equals the loaded parameter; the head is untied."""
    from safetensors import safe_open

    sd = model.state_dict()
    equal, differ, absent = 0, [], []
    with safe_open(str(snap / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k not in sd:
                absent.append(k)
                continue
            if torch.equal(sd[k], f.get_tensor(k).to(torch.float32)):
                equal += 1
            else:
                differ.append(k)
        head = f.get_tensor("lm_head.weight")
        emb = f.get_tensor("model.language_model.embed_tokens.weight")
    lm_head = model.lm_head.weight
    embed = model.model.language_model.embed_tokens.weight
    return {"checkpoint_tensors": len(keys), "equal": equal, "differ": differ, "not_in_model": absent,
            "model_tensors_not_in_checkpoint": sorted(set(sd) - set(keys)),
            "lm_head_untied": bool(lm_head.data_ptr() != embed.data_ptr()),
            "lm_head_equals_checkpoint_lm_head": bool(torch.equal(lm_head, head.to(torch.float32))),
            "checkpoint_lm_head_differs_from_embed": bool(not torch.equal(head, emb))}


def messages_for(spec: dict) -> list[dict]:
    msgs = []
    if spec["system"]:
        msgs.append({"role": "system", "content": spec["system"]})
    if spec["image"] is None:
        msgs.append({"role": "user", "content": spec["question"]})
    else:
        msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": spec["question"]}]})
    return msgs


def run_text(name, spec, model, proc, rec, torch, arrays, record):
    prompt = proc.apply_chat_template(messages_for(spec), tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-40:]
    enc = proc(text=[prompt], return_tensors="pt")
    ids = enc["input_ids"]
    S = int(ids.shape[1])
    model.model.rope_deltas = None
    rec.take()
    gen = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        pkv, nxt = out.past_key_values, out.logits[0, -1].argmax()
        for step in range(P0_NEW_TOKENS):
            gen.append(int(nxt))
            if step == P0_NEW_TOKENS - 1:
                break
            out = model(input_ids=nxt.view(1, 1), past_key_values=pkv, use_cache=True)
            pkv, nxt = out.past_key_values, out.logits[0, -1].argmax()
    greedy_secs = time.perf_counter() - t0
    greedy_calls = rec.take()
    full = torch.cat([ids, torch.tensor([gen], dtype=ids.dtype)], dim=1)
    L = int(full.shape[1])
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=full, use_cache=False).logits[0].float()   # [L, V]
    forward_secs = time.perf_counter() - t0
    calls = rec.take()
    assert len(calls) == 1 and calls[0]["lm_position_ids"] is None, "P0 forward: no position_ids handed in"
    planes = calls[0]["rotary"][:, 0].numpy().astype(np.int64)            # [3, L]
    ar = np.arange(L)
    argmax_all = logits.argmax(-1).numpy().astype(np.int64)
    t5i, t5v = top5(logits)
    first_stop = next((k for k, t in enumerate(gen) if t in STOP_IDS), None)
    arrays.update({f"{name}__input_ids": full[0].numpy().astype(np.int64),
                   f"{name}__prompt_len": np.array(S),
                   f"{name}__gen_ids": np.array(gen, dtype=np.int64),
                   f"{name}__logits": logits[L - P0_NEW_TOKENS:].numpy(),
                   f"{name}__logits_positions": np.arange(L - P0_NEW_TOKENS, L),
                   f"{name}__top5_ids": t5i, f"{name}__top5_vals": t5v,
                   f"{name}__argmax_all": argmax_all,
                   f"{name}__planes": planes})
    greedy_planes_ok = all(c["lm_position_ids"] is None for c in greedy_calls)
    steps_rot = [c["rotary"][:, 0].numpy() for c in greedy_calls]
    record[name] = {
        "kind": "text", "question": spec["question"], "prompt": prompt, "prompt_tokens": S,
        "sequence_tokens": L, "generated_ids": gen,
        "generated_text": proc.tokenizer.decode(gen, skip_special_tokens=False),
        "stop_tokens_honoured": False, "first_stop_token_index": first_stop,
        "greedy_seconds": greedy_secs, "greedy_calls": len(greedy_calls),
        "forward_seconds": forward_secs,
        "logits_positions": [L - P0_NEW_TOKENS, L - 1],
        "self_consistency_argmax_vs_greedy": int(sum(int(argmax_all[S - 1 + k]) == g for k, g in enumerate(gen))),
        "planes_1d": bool(all(np.array_equal(planes[j], ar) for j in range(3))),
        "greedy_steps_planes_1d": bool(greedy_planes_ok and np.array_equal(
            np.concatenate(steps_rot, axis=1), np.broadcast_to(ar[:L - 1], (3, L - 1)))),
        "lm_position_ids_handed": "None (the text model counts positions itself)",
    }
    print(f"[{name}] S={S} L={L} greedy {greedy_secs:.1f}s forward {forward_secs:.1f}s "
          f"self-consistency {record[name]['self_consistency_argmax_vs_greedy']}/{P0_NEW_TOKENS} "
          f"first stop {first_stop} 1D {record[name]['planes_1d']}", flush=True)
    print(f"[{name}] text: {record[name]['generated_text']!r}", flush=True)


def run_image(name, spec, model, proc, rec, torch, arrays, record, fixtures):
    from PIL import Image

    img_path = fixtures / f"{spec['image']}.448.png"
    img = Image.open(img_path).convert("RGB")
    assert img.size == (448, 448), img.size
    prompt = proc.apply_chat_template(messages_for(spec), tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    assert prompt.endswith("<think>\n\n</think>\n\n") and prompt.count("<|image_pad|>") == 1, prompt
    inputs = proc(images=[img], text=[prompt], max_slice_nums=1, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    S = len(ids)
    ts = inputs["target_sizes"].tolist()
    assert ts == [[32, 32]], f"target_sizes {ts}"
    assert sum(t == IMAGE_PAD for t in ids) == N_IMAGE_TOKENS
    block = check_image_block(ids, proc.tokenizer)
    mm = inputs["mm_token_type_ids"][0]
    assert mm.tolist() == [int(t == IMAGE_PAD) for t in ids], "mm_token_type_ids = (ids == <|image_pad|>)"
    hp = host_pixels(img)
    pv = inputs["pixel_values"].numpy()
    assert pv.shape == hp.shape, (pv.shape, hp.shape)

    merged = {}
    hook = model.model.merger.register_forward_hook(
        lambda mod, args, out: merged.__setitem__("x", torch.cat(out, 0).detach().clone()))
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            feats = torch.cat(model.model.get_image_features(
                inputs["pixel_values"], inputs["target_sizes"]).pooler_output, 0)
        tower_secs = time.perf_counter() - t0
        direct = merged.pop("x")
        model.model.rope_deltas = None
        rec.take()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0].float()   # [S, V]
        forward_secs = time.perf_counter() - t0
        in_forward = merged.pop("x")
    finally:
        hook.remove()
    calls = rec.take()
    assert len(calls) == 1
    lm_pid = calls[0]["lm_position_ids"]
    assert lm_pid is not None and tuple(lm_pid.shape) == (3, 1, S), None if lm_pid is None else lm_pid.shape
    planes = calls[0]["rotary"][:, 0].numpy().astype(np.int64)            # [3, S]
    assert np.array_equal(planes, lm_pid[:, 0].numpy()), "rotary planes == planes handed to the text model"
    t5i, t5v = top5(logits)

    model.model.rope_deltas = None
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
    gen_secs = time.perf_counter() - t0
    gen_calls = rec.take()
    new = out_ids[0, S:].tolist()
    # generate: step 0 = prefill [4, 1, S]; step j >= 1 = one token at cache position S + j - 1
    g_lm = [c["lm_position_ids"][:, 0].numpy().astype(np.int64) for c in gen_calls]
    g_rot = [c["rotary"][:, 0].numpy().astype(np.int64) for c in gen_calls]
    assert all(x.shape[0] == 4 for x in g_lm), [x.shape for x in g_lm]
    arrays.update({f"{name}__input_ids": np.array(ids, dtype=np.int64),
                   f"{name}__mm_token_type_ids": mm.numpy().astype(np.int64),
                   f"{name}__target_sizes": np.array(ts, dtype=np.int64),
                   f"{name}__pixel_values": pv.astype(np.float32),
                   f"{name}__image_features": feats.numpy().astype(np.float32),
                   f"{name}__logits": logits[-1].numpy(),
                   f"{name}__top5_ids": t5i, f"{name}__top5_vals": t5v,
                   f"{name}__planes": planes,
                   f"{name}__gen_ids": np.array(new, dtype=np.int64),
                   f"{name}__gen_lm_position_ids": np.concatenate(g_lm, axis=1),
                   f"{name}__gen_rotary_planes": np.concatenate(g_rot, axis=1),
                   f"{name}__gen_step_lens": np.array([x.shape[1] for x in g_lm], dtype=np.int64)})
    record[name] = {
        "kind": "image", "image": img_path.name, "image_sha256": sha256(img_path), "system": spec["system"],
        "question": spec["question"], "prompt": prompt,
        "expanded_prompt": proc.tokenizer.decode(ids, skip_special_tokens=False).replace(
            "<|image_pad|>" * N_IMAGE_TOKENS, "<|image_pad|>x64"),
        "prompt_tokens": S, "target_sizes": ts, "image_pad_count": N_IMAGE_TOKENS, **block,
        "processor_kwargs": {"max_slice_nums": 1, "return_tensors": "pt"},
        "pixel_values_shape": list(pv.shape),
        "pixel_values_vs_host_formula_max_abs": float(np.abs(pv - hp).max()),
        "image_features_shape": list(feats.shape),
        "image_features_equal_merger_output_in_forward": bool(torch.equal(feats, in_forward)),
        "image_features_equal_direct_merger_output": bool(torch.equal(feats, direct)),
        "tower_seconds": tower_secs, "forward_seconds": forward_secs, "generate_seconds": gen_secs,
        "last_logits_top5": t5i[-1].tolist(),
        "generated_ids": new, "answer": proc.tokenizer.decode(new, skip_special_tokens=True),
        "generate_calls": len(gen_calls), "generate_step_lens": [int(x.shape[1]) for x in g_lm],
        "generate_prefill_planes_equal_forward_planes": bool(np.array_equal(g_lm[0][1:], planes)),
        "generate_prefill_text_row_is_arange": bool(np.array_equal(g_lm[0][0], np.arange(S))),
        "generate_rotary_equals_lm_TWH_rows": bool(all(np.array_equal(a[1:], b) for a, b in zip(g_lm, g_rot))),
    }
    print(f"[{name}] S={S} tower {tower_secs:.1f}s forward {forward_secs:.1f}s generate {gen_secs:.1f}s "
          f"({len(new)} tok) pixel host max|d| {record[name]['pixel_values_vs_host_formula_max_abs']:.2e} "
          f"top5 {t5i[-1].tolist()}", flush=True)
    print(f"[{name}] answer: {record[name]['answer']!r}", flush=True)


def versions(torch) -> dict:
    import PIL
    import tokenizers
    import torchvision
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="comma list (default: P0..P4)")
    ap.add_argument("--fixtures", default=str(work_path("_minicpmv47", "oracle", "fixtures")))
    ap.add_argument("--out-dir", default=str(work_path("_minicpmv47", "oracle")))
    args = ap.parse_args()

    import torch

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    from transformers import AutoProcessor

    snap = Path(hf_snapshot(HF_ID, revision=REVISION))
    fixtures, out_dir = Path(args.fixtures), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    proc = AutoProcessor.from_pretrained(str(snap))
    model, load_info, load_secs = load_model(snap, torch)
    weights = check_weights(model, snap, torch)
    assert not weights["differ"] and not weights["not_in_model"], weights
    assert weights["lm_head_untied"] and weights["lm_head_equals_checkpoint_lm_head"], weights
    print(f"[load] {load_secs:.1f}s, {weights['equal']}/{weights['checkpoint_tensors']} tensors equal, "
          f"head untied {weights['lm_head_untied']}, threads {torch.get_num_threads()}", flush=True)
    rec = Recorder(model)
    arrays, record = {}, {}
    for name in args.prompts.split(","):
        spec = PROMPTS[name]
        if spec["image"] is None:
            run_text(name, spec, model, proc, rec, torch, arrays, record)
        else:
            run_image(name, spec, model, proc, rec, torch, arrays, record, fixtures)

    npz = out_dir / "oracle.npz"
    np.savez(npz, **arrays)
    fx_json = fixtures / "fixtures.json"
    meta = {
        "schema": "minicpmv47-oracle/1",
        "source": {"hf_id": HF_ID, "revision": REVISION, "snapshot": str(snap),
                   "model_safetensors_sha256": sha256(snap / "model.safetensors")},
        "oracle": {"class": type(model).__name__, "dtype": str(model.dtype), "device": "cpu",
                   "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "attn_implementation": model.config._attn_implementation,
                   "processor": type(proc).__name__, "image_processor": type(proc.image_processor).__name__,
                   "tokenizer": type(proc.tokenizer).__name__, "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
                   "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": False},
                   "load_seconds": load_secs, "loading_info": load_info, "weights": weights},
        "versions": versions(torch),
        "fixtures": {"dir": str(fixtures), "fixtures_json_sha256": sha256(fx_json) if fx_json.exists() else None},
        "constants": {"vocab": VOCAB, "image_pad": IMAGE_PAD, "image_start": IMAGE_START, "image_end": IMAGE_END,
                      "stop_ids": list(STOP_IDS), "p0_new_tokens": P0_NEW_TOKENS, "gen_new_tokens": GEN_NEW_TOKENS},
        "prompts": record,
        "npz": {"path": str(npz), "sha256": sha256(npz), "keys": sorted(arrays)},
        "script": {"path": "conversion/minicpmv47/oracle_minicpmv47.py", "sha256": sha256(Path(__file__))},
        "wall_seconds": time.perf_counter() - t_all,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "oracle.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[done] {npz} ({npz.stat().st_size / 1e6:.1f} MB), wall {meta['wall_seconds']:.0f}s", flush=True)


import numpy as np  # noqa: E402  (after the env setup above; numpy is needed by the helpers)

if __name__ == "__main__":
    main()
