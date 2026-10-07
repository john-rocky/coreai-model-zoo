#!/usr/bin/env python3
"""Test: vision_host.py rebuilds the provider's image path bit for bit, on the fixture pictures and random ones.

The provider's side is its own code (`K/src/d1` = the .py files of LiquidAI/d1-3B @ da1fe36a, verbatim): `cap_pixels`,
`_image_markup`, `prefix_text` / `suffix_text` and `_image_inputs` exactly as `runner._request` calls them (one
question: the whole row; several: the trunk), with transformers 5.19's `AutoProcessor` on the snapshot (no model, no
weights). The host's side is `vision_host.py` (NumPy + Pillow) with the checkpoint's tokenizer.json through
`tokenizers` (what a Swift host reads). Per request (12 fixture records + 6 random pictures, and one request with two
fixture pictures: markup "<image><image>", slots numbered across both), equal or the test fails:

  * the prompt text and its input_ids, every token (host: the text pieces encoded apart + the image runs; also the
    expanded string encoded whole), usage.input_tokens;
  * pixel_values bit for bit, against the processor's full output [n_crop, 1024, 768] and the provider's cut;
    spatial_shapes; pixel_attention_mask; on a mismatch the stage that splits (each crop's uint8 pixels after the
    resize, the 256-value normalize table, the patch layout);
  * the <image> count == the host's n_image_tokens == the rows get_image_features returns (sum of h/2 * w/2), and
    the extension ids (V + slot) in order.

Also: the position table against `Siglip2VisionEmbeddings.resize_positional_embeddings` (a random [256, 1152] table,
every grid the fixture produces + 5 more, max |d| <= 1e-6, and what HF writes in the padding rows); the unshuffle
index against `Lfm2VlMultiModalProjector.pixel_unshuffle` (arange tensors, every grid); the tower contract on a small
random-weight SigLIP2 + projector built from transformers' own classes (`patches + pos_table (padding rows 0) +
key_bias + unshuffle_idx` -> image_embeds against `get_image_features`' path, and the provider's cut against the
padded batch); negative controls that must go red (the position table with h and w swapped, the unshuffle with
the 4 patches in another order, the patch vector as [c][y][x], the tower without the key mask); and what Pillow's
resampler would change if a host used it in place of torch's.

`--grid-table` (default on) runs every w x h on a 64-px grid with aspect 1:4 .. 4:1 (64 .. 2048 px a side) through
the processor directly and through `cap_pixels` first, and the host's plan beside it.

    cd conversion/d1
    source $ZOO_WORK_ROOT/_d1_3b/venv-oracle/bin/activate && python test_vision_host.py   # transformers 5.19

-> $ZOO_WORK_ROOT/_d1_3b/results/{vision_host.json, vision_grid_table.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import host  # noqa: E402
import vision_host as vh  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
POS_BAR = 1e-6
RANDOM_SIZES = [(384, 384), (384, 256), (640, 480), (1024, 768), (1600, 1200), (2048, 1536)]   # w x h
RANDOM_QUESTION = {"q": {"type": "noul", "instructions": "Is there anything in the picture?"}}
EXTRA_GRIDS = [(32, 32), (16, 16), (8, 24), (64, 16), (16, 64)]
OUTSIDE_SIZES = [(2048, 256), (2048, 128), (2048, 64), (128, 2048), (3000, 100), (4000, 100), (1024, 32)]   # w x h
PAIR = ("img01_shapes_384x384", "img06_grid_1024x768")
PAIR_QUESTIONS = {"same": {"type": "noul", "instructions": "Do the two pictures show the same shapes?"},
                  "count": {"type": "choice", "instructions": "How many pictures have a circle?",
                            "criteria": {"none": "None", "one": "One", "both": "Both"}}}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- the provider's side
def provider_env(snap: Path):
    """The provider's modules and a SystemOne with no model: the vision helpers need only the processor."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    if str(LANE / "src") not in sys.path:
        sys.path.insert(0, str(LANE / "src"))
    import d1.prompt as prompt
    import d1.runner as runner

    proc = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(snap))
    eng = runner.SystemOne.__new__(runner.SystemOne)
    eng.processor, eng.tokenizer, eng.device = proc, tok, torch.device("cpu")
    bos = getattr(tok, "bos_token", None)
    eng.bos = bos if isinstance(bos, str) else ""
    eng.lead, eng.state_style, eng.system, eng.option_style = "", prompt.DEFAULT_STATE_STYLE, prompt.DEFAULT_SYSTEM, "desc"
    return prompt, runner, eng


def provider_request(prompt, runner, eng, state, questions: dict, images: list) -> dict:
    """runner._request up to the forward pass, line for line, plus the processor's uncut output."""
    qs = [prompt.as_question(q) for q in questions.values()]
    pics = [runner.cap_pixels(im) for im in images]
    markup = eng._image_markup(len(pics)) if pics else ""
    prefix = prompt.prefix_text(eng.tokenizer, state, eng.bos, eng.state_style, eng.system, markup)
    suffixes = [prompt.suffix_text(eng.tokenizer, q, eng.lead, eng.option_style) for q in qs]
    if len(qs) == 1:
        text = prefix + suffixes[0]
        inputs = eng._image_inputs(text, pics)
        branches, path = [], "row"
        input_tokens = int(inputs["input_ids"].shape[1])
    else:
        text = prefix
        inputs = eng._image_inputs(prefix, pics)
        branches, path = [eng.tokenizer.encode(s, add_special_tokens=False) for s in suffixes], "tree"
        input_tokens = int(inputs["input_ids"].shape[1]) + sum(map(len, branches))
    raw = eng.processor(text=[text], images=[list(pics)], return_tensors="pt", add_special_tokens=False,
                        return_row_col_info=True)
    return {"markup": markup, "text": text, "path": path, "pics": pics, "inputs": inputs, "raw": raw,
            "branches": branches, "input_tokens": input_tokens}


# --------------------------------------------------------------------------- the host's side
def host_request(rs, state, questions: dict, images: list) -> dict:
    req = host.validate_request({"state": state, "questions": questions})
    built = [host.build_question(rs, req["state"], n, q) for n, q in req["questions"]]
    rgbs = [vh.cap_pixels(vh.to_rgb(im)) for im in images]
    plans = [vh.plan(*rgb.shape[:2]) for rgb in rgbs]
    crops = [c for rgb, p in zip(rgbs, plans) for c in vh.crop_images(rgb, p)]
    pv, masks = zip(*(vh.crop_pixels(c) for c in crops))
    prefix = vh.image_prefix_text(req["state"], len(images))
    text = prefix + built[0]["suffix"] if len(built) == 1 else prefix
    ids = vh.prompt_ids(rs, text, plans)
    branches = [host.token_ids(rs, b["suffix"]) for b in built] if len(built) > 1 else []
    return {"text": text, "ids": ids, "whole_ids": host.token_ids(rs, vh.expanded_text(text, plans)),
            "rgbs": rgbs, "plans": plans, "crops": crops, "pixel_values": np.stack(pv),
            "pixel_attention_mask": np.stack(masks),
            "spatial_shapes": np.array([c.grid for p in plans for c in p.crops], dtype=np.int64),
            "input_tokens": len(ids) + sum(map(len, branches))}


def hf_crops_uint8(eng, pic) -> list[np.ndarray]:
    """The processor's crops as uint8 [H, W, 3], after its resize and split (stage 1 of the diagnosis)."""
    from torchvision.transforms.v2 import functional as tvF

    ip = eng.processor.image_processor
    t = tvF.pil_to_tensor(pic)[None]
    crops, _, _, _ = ip.resize_and_split(
        t, downsample_factor=ip.downsample_factor, min_tiles=ip.min_tiles, max_tiles=ip.max_tiles,
        use_thumbnail=ip.use_thumbnail, min_image_tokens=ip.min_image_tokens, max_image_tokens=ip.max_image_tokens,
        encoder_patch_size=ip.encoder_patch_size, tile_size=ip.tile_size, max_pixels_tolerance=ip.max_pixels_tolerance,
        resample=ip.resample)
    return [c.permute(1, 2, 0).numpy() for c in crops[0]]


def compare_record(rid: str, kind: str, size: list[tuple[int, int]], state, questions: dict, images: list,
                   host_images: list, prompt, runner, eng, rs) -> dict:
    p = provider_request(prompt, runner, eng, state, questions, images)
    h = host_request(rs, state, questions, host_images)
    pid = p["inputs"]["input_ids"][0].tolist()
    raw_pv = p["raw"]["pixel_values"].numpy()
    cut_pv = p["inputs"]["pixel_values"].numpy()
    n_cut = cut_pv.shape[1]
    hv = h["pixel_values"]
    same_shape = raw_pv.shape == hv.shape
    pv_bit = bool(same_shape and np.array_equal(raw_pv.view(np.uint32), hv.view(np.uint32)))
    cut_bit = bool(cut_pv.shape == hv[:, :n_cut].shape and np.array_equal(cut_pv.view(np.uint32), hv[:, :n_cut].view(np.uint32)))
    max_level = float(np.abs(raw_pv.astype(np.float64) - hv.astype(np.float64)).max() * 127.5) if same_shape else None
    # stage 1: the resized crops (uint8), for every picture
    hf_u8 = [c for pic in p["pics"] for c in hf_crops_uint8(eng, pic)]
    resize_levels = [int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else None
                     for a, b in zip(hf_u8, h["crops"])]
    plans = h["plans"]
    n_tok = vh.n_image_tokens(plans)
    feat_rows = int(sum(math.ceil(a / 2) * math.ceil(b / 2) for a, b in p["raw"]["spatial_shapes"].tolist()))
    ext = vh.extension_ids(h["ids"])
    slots = [i for i, t in enumerate(h["ids"]) if t == vh.IMAGE_ID]
    slot_set = set(slots)
    ext_ok = ([ext[i] for i in slots] == [vh.V + k for k in range(len(slots))]
              and all(ext[i] == h["ids"][i] for i in range(len(ext)) if i not in slot_set))
    runs = [vh.image_tokens(pl) for pl in plans]
    starts = [i for i, t in enumerate(pid) if t == vh.IMAGE_START_ID]
    run_ok = len(starts) == len(runs) and all(pid[s:s + len(r)] == r for s, r in zip(starts, runs))
    return {
        "id": rid, "kind": kind, "size_in": [list(s) for s in size],
        "path": p["path"], "markup": p["markup"],
        "pictures": [{"size_capped": [int(rgb.shape[1]), int(rgb.shape[0])], "rows": pl.rows, "cols": pl.cols,
                      "crops": [{"kind": c.kind, "grid": list(c.grid), "tokens": c.n_tokens, "row": c.row, "col": c.col}
                                for c in pl.crops]} for rgb, pl in zip(h["rgbs"], plans)],
        "n_crops": len(h["crops"]), "n_image_tokens": n_tok, "ids_len": len(pid),
        "text_equal": h["text"] == p["text"],
        "ids_equal": h["ids"] == pid, "ids_equal_whole_string": h["whole_ids"] == pid,
        "image_run_equal": run_ok,
        "image_token_count": {"provider_ids": pid.count(vh.IMAGE_ID), "host": n_tok, "feature_rows": feat_rows},
        "image_token_count_equal": pid.count(vh.IMAGE_ID) == n_tok == feat_rows,
        "extension_ids_ok": bool(ext_ok and len(slots) == n_tok),
        "input_tokens": {"provider": p["input_tokens"], "host": h["input_tokens"]},
        "input_tokens_equal": p["input_tokens"] == h["input_tokens"],
        "pixel_values_shape": list(raw_pv.shape), "pixel_values_bit_equal": pv_bit, "pixel_values_max_level": max_level,
        "provider_cut": {"n": int(n_cut), "bit_equal": cut_bit},
        "spatial_shapes_equal": bool(np.array_equal(p["raw"]["spatial_shapes"].numpy(), h["spatial_shapes"])),
        "pixel_attention_mask_equal": bool(np.array_equal(p["raw"]["pixel_attention_mask"].numpy(), h["pixel_attention_mask"])),
        "mask_dtype": str(p["raw"]["pixel_attention_mask"].dtype),
        "stage_resize_max_level": resize_levels,
        "image_rows_cols": [p["raw"]["image_rows"], p["raw"]["image_cols"]] if "image_rows" in p["raw"] else None,
    }


# --------------------------------------------------------------------------- global stages, tables, controls
def stage_checks(eng) -> dict:
    import torch
    from transformers.models.lfm2_vl.image_processing_lfm2_vl import convert_image_to_patches

    ip = eng.processor.image_processor
    x = torch.arange(256, dtype=torch.uint8).repeat(3, 1).reshape(1, 3, 1, 256)
    hf = ip.rescale_and_normalize(x, ip.do_rescale, ip.rescale_factor, ip.do_normalize, ip.image_mean, ip.image_std)
    mine = vh.normalize(np.arange(256, dtype=np.uint8))
    lut_equal = bool(np.array_equal(hf[0, 0, 0].numpy().view(np.uint32), mine.view(np.uint32)))
    rng = np.random.default_rng(7)
    img = rng.standard_normal((48, 80, 3)).astype(np.float32)
    hf_p = convert_image_to_patches(torch.from_numpy(img).permute(2, 0, 1)[None], 16)[0].numpy()
    layout_equal = bool(np.array_equal(hf_p, vh.patchify(img)))
    cyx = vh.patchify(img).reshape(15, 16, 16, 3).transpose(0, 3, 1, 2).reshape(15, 768)   # [c][y][x] inside a patch
    return {"normalize_lut_bit_equal": lut_equal, "normalize_values": [float(mine[0]), float(mine[127]), float(mine[255])],
            "patch_layout_equal": layout_equal,
            "negative_patch_layout_cyx_equal": bool(np.array_equal(hf_p, cyx))}


def pos_table_checks(grids: list[tuple[int, int]], seed: int = 0) -> dict:
    import torch
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    rng = np.random.default_rng(seed)
    table = rng.standard_normal((256, 1152)).astype(np.float32)
    pe = torch.from_numpy(table).reshape(16, 16, 1152)
    rows, ok = [], True
    for gh, gw in grids:
        hf = Siglip2VisionEmbeddings.resize_positional_embeddings(pe, torch.tensor([[gh, gw]]), max_length=1024)[0].numpy()
        n = gh * gw
        mine = vh.pos_table(table, gh, gw)
        plain = vh.pos_table(table, gh, gw, fma=False)
        d = float(np.abs(mine - hf[:n]).max())
        ok &= d <= POS_BAR
        rows.append({"grid": [gh, gw], "max_abs": d, "bit_equal": bool(np.array_equal(mine, hf[:n])),
                     "plain_mul_add_max_abs": float(np.abs(plain - hf[:n]).max()),
                     "hf_padding_rows": ("row 0" if n < 1024 and np.array_equal(hf[n:], np.broadcast_to(hf[0], hf[n:].shape))
                                         else "none" if n == 1024 else "other")})
    # negative control: h and w swapped (a non-square grid)
    gh, gw = 26, 36
    hf = Siglip2VisionEmbeddings.resize_positional_embeddings(pe, torch.tensor([[gh, gw]]), max_length=1024)[0].numpy()
    swapped = vh.pos_table(table, gw, gh)
    neg = float(np.abs(swapped - hf[:gh * gw]).max())
    return {"bar": POS_BAR, "table": "standard normal [256, 1152] float32, seed %d" % seed, "grids": rows,
            "pass": bool(ok), "n_grids": len(rows), "n_pass": sum(r["max_abs"] <= POS_BAR for r in rows),
            "negative_hw_swapped": {"grid": [gh, gw], "max_abs": neg, "red": neg > POS_BAR}}


def unshuffle_checks(grids: list[tuple[int, int]]) -> dict:
    import torch
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector

    me = SimpleNamespace(factor=2)
    rows, ok = [], True
    for gh, gw in grids:
        x = torch.arange(gh * gw * 3, dtype=torch.float64).reshape(gh * gw, 3)
        hf = Lfm2VlMultiModalProjector.pixel_unshuffle(me, x.reshape(1, gh, gw, -1)).reshape(-1, 12).numpy()
        mine = vh.unshuffle(x.numpy(), gh, gw)
        eq = bool(np.array_equal(hf, mine))
        ok &= eq
        rows.append({"grid": [gh, gw], "equal": eq, "tokens": int(hf.shape[0])})
    gh, gw = 26, 36
    x = np.arange(gh * gw * 3, dtype=np.float64).reshape(gh * gw, 3)
    hf = Lfm2VlMultiModalProjector.pixel_unshuffle(me, torch.from_numpy(x).reshape(1, gh, gw, -1)).reshape(-1, 12).numpy()
    idx = vh.unshuffle_index(gh, gw)
    swapped = np.concatenate([x[idx[:, m]] for m in (0, 2, 1, 3)], axis=1)      # (dy, dx) read as (dx, dy)
    col_major = np.concatenate([x[idx.reshape(gh // 2, gw // 2, 4).transpose(1, 0, 2).reshape(-1, 4)[:, m]] for m in range(4)], axis=1)
    return {"grids": rows, "pass": bool(ok), "n_grids": len(rows),
            "channel_order": "m * d + c from offset (dy, dx) = (m // 2, m % 2)",
            "example_26x36_k0": idx[0].tolist(), "example_26x36_k1": idx[1].tolist(), "example_26x36_k18": idx[18].tolist(),
            "negative_patch_order_0213_equal": bool(np.array_equal(hf, swapped)),
            "negative_tokens_column_major_equal": bool(np.array_equal(hf, col_major))}


def tower_contract(fixture_crops: list[tuple[str, np.ndarray]], seed: int = 0) -> dict:
    """A small random-weight SigLIP2 tower + LFM2-VL projector (transformers' classes): HF's path vs the host's
    fixed-shape inputs run through the same weights by hand."""
    import torch
    import torch.nn.functional as F
    from transformers import Siglip2VisionConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

    torch.manual_seed(seed)
    cfg = Siglip2VisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                              num_channels=3, patch_size=16, num_patches=256, layer_norm_eps=1e-6,
                              hidden_act="gelu_pytorch_tanh")
    cfg.vision_use_head = False
    tower = Siglip2VisionModel(cfg).eval()
    for prm in tower.parameters():           # random init leaves some params at constants; spread them
        with torch.no_grad():
            prm.add_(0.02 * torch.randn_like(prm))
    pcfg = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=64), downsample_factor=2, projector_use_layernorm=False,
                           projector_hidden_size=96, projector_bias=True, projector_hidden_act="gelu",
                           text_config=SimpleNamespace(hidden_size=80))
    proj = Lfm2VlMultiModalProjector(pcfg).eval()
    table = tower.embeddings.position_embedding.weight.detach().numpy().astype(np.float32)

    def hf_path(pv, mask, gh, gw):
        out = tower(pixel_values=torch.from_numpy(pv)[None], pixel_attention_mask=torch.from_numpy(mask)[None],
                    spatial_shapes=torch.tensor([[gh, gw]])).last_hidden_state[0]
        n = gh * gw
        return proj(out[:n].reshape(1, gh, gw, -1)).reshape(-1, 80)

    def host_path(ti, use_bias=True):
        x = torch.from_numpy(ti["patches"]) @ tower.embeddings.patch_embedding.weight.T + tower.embeddings.patch_embedding.bias
        x = x + torch.from_numpy(ti["pos_table"])
        bias = torch.from_numpy(ti["key_bias"]) if use_bias else torch.zeros(1024)
        for layer in tower.encoder.layers:
            a = layer.self_attn
            hdn = layer.layer_norm1(x)
            q = a.q_proj(hdn).view(1024, 4, 16).transpose(0, 1)
            k = a.k_proj(hdn).view(1024, 4, 16).transpose(0, 1)
            v = a.v_proj(hdn).view(1024, 4, 16).transpose(0, 1)
            s = (q @ k.transpose(1, 2)) * a.scale + bias[None, None, :]
            o = (torch.softmax(s, dim=-1) @ v).transpose(0, 1).reshape(1024, 64)
            x = x + a.out_proj(o)
            x = x + layer.mlp.fc2(F.gelu(layer.mlp.fc1(layer.layer_norm2(x)), approximate="tanh"))
        x = tower.post_layernorm(x)
        idx = torch.from_numpy(ti["unshuffle_idx"]).long()
        u = torch.cat([x[idx[:, m]] for m in range(4)], dim=-1)
        return proj.linear_2(proj.act(proj.linear_1(u)))[: ti["n_tokens"]]

    def cmp(a, b):
        a64, b64 = a.double().reshape(-1), b.double().reshape(-1)
        return {"max_abs": float((a64 - b64).abs().max()), "cos": float(a64 @ b64 / (a64.norm() * b64.norm()))}

    rows = []
    with torch.no_grad():
        for name, crop in fixture_crops:
            ti = vh.tower_inputs(crop, table)
            gh, gw = ti["grid"]
            n = ti["n_patches"]
            hf_full = hf_path(ti["patches"], ti["mask"], gh, gw)
            hf_cut = hf_path(np.ascontiguousarray(ti["patches"][:n]), np.ascontiguousarray(ti["mask"][:n]), gh, gw)
            mine = host_path(ti)
            row = {"crop": name, "grid": [gh, gw], "n_patches": n, "padding_rows": 1024 - n,
                   "host_vs_hf_padded": cmp(mine, hf_full), "provider_cut_vs_padded": cmp(hf_cut, hf_full)}
            if n < 1024:
                row["negative_no_key_mask"] = cmp(host_path(ti, use_bias=False), hf_full)
            rows.append(row)
    return {"model": "Siglip2VisionModel(hidden 64, 2 layers, 4 heads, 16x16 table) + Lfm2VlMultiModalProjector(64*4 -> 96 -> 80), "
                     "random weights seed %d, float32 CPU, HF attention = %s" % (seed, tower.config._attn_implementation),
            "crops": rows}


def pillow_instead(fixtures: list[tuple[str, np.ndarray]]) -> list[dict]:
    """What a host that used Pillow's resampler (22-bit weights) for the processor's resize would change."""
    out = []
    for name, rgb in fixtures:
        p = vh.plan(*rgb.shape[:2])
        worst, n_diff, n = 0, 0, 0
        for c in p.crops:
            a = vh.torch_bicubic(rgb, *c.resize_to)
            b = vh.pillow_bicubic(rgb, *c.resize_to)
            y0, x0, y1, x1 = c.box
            d = np.abs(a[y0:y1, x0:x1].astype(int) - b[y0:y1, x0:x1].astype(int))
            worst, n_diff, n = max(worst, int(d.max())), n_diff + int((d > 0).sum()), n + d.size
        out.append({"id": name, "max_level": worst, "n_diff": n_diff, "n": n})
    return out


# --------------------------------------------------------------------------- the grid table
def grid_table(runner, eng) -> dict:
    from PIL import Image

    ip = eng.processor.image_processor
    sizes = [(w, h) for w in range(64, 2049, 64) for h in range(64, 2049, 64) if 0.25 <= w / h <= 4.0]

    def run(w, h, capped):
        im = Image.new("RGB", (w, h), (0, 0, 0))
        pic = runner.cap_pixels(im) if capped else im
        pw, ph = pic.size
        hp = vh.plan(ph, pw)
        try:
            out = ip(images=[[pic]], return_tensors="pt", return_row_col_info=True)
        except Exception as e:  # noqa: BLE001 - recorded, not raised (outside the 1024-patch range)
            return {"size": [pw, ph], "error": f"{type(e).__name__}: {e}"[:200], "host_crops": len(hp.crops)}
        ss = out["spatial_shapes"].tolist()
        tokens = int(sum(math.ceil(a / 2) * math.ceil(b / 2) for a, b in ss))
        return {"size": [pw, ph], "rows": int(out["image_rows"][0]), "cols": int(out["image_cols"][0]),
                "n_crops": len(ss), "grids": sorted({tuple(g) for g in ss}), "tokens": tokens,
                "max_patches": int(max(a * b for a, b in ss)),
                "pixel_values_shape": list(out["pixel_values"].shape),
                "host_equal": ([list(c.grid) for c in hp.crops] == ss and (hp.rows, hp.cols) ==
                               (int(out["image_rows"][0]), int(out["image_cols"][0])) and vh.n_image_tokens([hp]) == tokens)}

    t0 = time.time()
    rows = []
    for w, h in sizes:
        d = run(w, h, False)
        c = d if w * h <= vh.VISION_MAX_PIXELS else run(w, h, True)
        rows.append({"w": w, "h": h, "direct": d, "capped": c})
    outside = []
    for w, h in OUTSIDE_SIZES:
        outside.append({"w": w, "h": h, "direct": run(w, h, False), "capped": run(w, h, True)})
    secs = time.time() - t0

    def best(key):
        m = max(r[key]["tokens"] for r in rows if "tokens" in r[key])
        return m, [[r["w"], r["h"]] for r in rows if r[key].get("tokens") == m]

    cap_max, cap_arg = best("capped")
    dir_max, dir_arg = best("direct")
    single = [r for r in rows if r["direct"].get("n_crops") == 1]
    single_max_px = max(r["w"] * r["h"] for r in single)

    def single_limit(max_aspect: float):
        """The largest w x h (any integers, w / h within [1, max_aspect]) the host plans as one crop."""
        r32 = [max(vh.PATCH, vh.round_by_factor(v, vh.PATCH * vh.FACTOR)) for v in range(0, 8192)]
        top = (0, None)
        for hh in range(1, 4096):
            if r32[hh] * vh.PATCH > 524288:
                break
            lim = 524288 // r32[hh]
            ww = min(int(hh * max_aspect), len(r32) - 1)
            while ww > 0 and r32[ww] > lim:
                ww -= 1
            if ww >= hh and hh * ww > top[0]:
                top = (hh * ww, [ww, hh])
        return {"pixels": top[0], "w_h": top[1], "host_single": not vh.is_too_large(top[1][1], top[1][0])}
    return {
        "sizes": "w, h in 64..2048 step 64 with 1/4 <= w/h <= 4 (%d sizes), a black picture each" % len(sizes),
        "seconds": round(secs, 1),
        "host_equal": f"{sum(r['direct'].get('host_equal', False) and r['capped'].get('host_equal', False) for r in rows)}/{len(rows)}",
        "summary": {
            "n_image_tokens_max_capped": cap_max, "argmax_capped": cap_arg,
            "n_image_tokens_max_direct": dir_max, "argmax_direct": dir_arg,
            "crops_max_capped": max(r["capped"].get("n_crops", 0) for r in rows),
            "max_patches_per_crop": max(max(r["direct"].get("max_patches", 0), r["capped"].get("max_patches", 0)) for r in rows),
            "errors": [[r["w"], r["h"]] for r in rows if "error" in r["direct"] or "error" in r["capped"]],
            "single_crop_max_pixels_in_table": single_max_px,
            "single_crop_max_pixels_sizes": [[r["w"], r["h"]] for r in single if r["w"] * r["h"] == single_max_px],
            "single_crop_rule": "one crop iff max(16, round32(h)) * max(16, round32(w)) <= 524,288 (round half to even)",
            "single_crop_max_any_size": {k: single_limit(a) for k, a in
                                         (("square", 1.0), ("4:3", 4 / 3), ("16:9", 16 / 9), ("4:1", 4.0))},
            "smallest_tiled_in_table": min((r["w"] * r["h"], [r["w"], r["h"]]) for r in rows if r["direct"].get("n_crops", 0) > 1)[1],
        },
        "rows": rows, "outside_aspect": outside,
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=str(LANE / "fixtures" / "image_records.json"))
    ap.add_argument("--out", default=str(LANE / "results" / "vision_host.json"))
    ap.add_argument("--grid-out", default=str(LANE / "results" / "vision_grid_table.json"))
    ap.add_argument("--no-grid-table", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import PIL
    import tokenizers
    import torch
    import torchvision
    import transformers
    from PIL import Image

    torch.set_num_threads(max(1, torch.get_num_threads()))
    snap = Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
    prompt, runner, eng = provider_env(snap)
    rs = host.load_tokenizer(snap / "tokenizer.json")
    ip = eng.processor.image_processor
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "pillow": PIL.__version__,
                "numpy": np.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "machine": platform.machine(), "snapshot": str(snap)},
        "processor": {"class": type(eng.processor).__name__, "image_processor": type(ip).__name__,
                      "image_processor_mro": [c.__name__ for c in type(ip).__mro__][:3],
                      "backend": getattr(ip, "backend", None), "resample": int(ip.resample),
                      "image_markup_1": eng._image_markup(1), "image_markup_2": eng._image_markup(2)},
    }
    t0 = time.time()
    fx = json.loads(Path(args.fixtures).read_text())
    base = Path(args.fixtures).parent
    records = []
    fixture_rgbs = []
    for r in fx["records"]:
        if r["source"] != "own_image":
            continue
        path = base / r["images"][0]
        size = Image.open(path).size
        rec = compare_record(r["id"], "fixture", [size], r["request"]["state"], r["request"]["questions"],
                             [Image.open(path)], [path], prompt, runner, eng, rs)
        rec["png_sha256"] = sha256_bytes(path.read_bytes())
        records.append(rec)
        fixture_rgbs.append((r["id"], vh.cap_pixels(vh.to_rgb(path))))
        print(f"{r['id']}: ids {rec['ids_equal']} pixels {rec['pixel_values_bit_equal']} tokens {rec['n_image_tokens']}", flush=True)
    rng = np.random.default_rng(args.seed)
    for w, h in RANDOM_SIZES:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        rec = compare_record(f"random_{w}x{h}", "random", [(w, h)], None, RANDOM_QUESTION, [Image.fromarray(arr)], [arr],
                             prompt, runner, eng, rs)
        rec["rgb_sha256"] = sha256_bytes(arr.tobytes())
        records.append(rec)
        print(f"random {w}x{h}: ids {rec['ids_equal']} pixels {rec['pixel_values_bit_equal']} tokens {rec['n_image_tokens']}", flush=True)
    # two pictures in one request: markup "<image><image>", slots numbered across both pictures' crops
    pair = [base / r["images"][0] for r in fx["records"] if r["id"] in PAIR]
    rec = compare_record("pair_" + "_".join(i.split("_")[0] for i in PAIR), "pair", [Image.open(q).size for q in pair],
                         "Two drawings.", PAIR_QUESTIONS, [Image.open(q) for q in pair], pair, prompt, runner, eng, rs)
    records.append(rec)
    print(f"pair: ids {rec['ids_equal']} pixels {rec['pixel_values_bit_equal']} tokens {rec['n_image_tokens']}", flush=True)
    report["records"] = records
    keys = ["text_equal", "ids_equal", "ids_equal_whole_string", "image_run_equal", "image_token_count_equal",
            "extension_ids_ok", "input_tokens_equal", "pixel_values_bit_equal", "spatial_shapes_equal",
            "pixel_attention_mask_equal"]

    def summarize(rs_):
        s = {k: f"{sum(bool(r[k]) for r in rs_)}/{len(rs_)}" for k in keys}
        s["provider_cut_bit_equal"] = f"{sum(r['provider_cut']['bit_equal'] for r in rs_)}/{len(rs_)}"
        s["stage_resize_all_zero"] = f"{sum(all(v == 0 for v in r['stage_resize_max_level']) for r in rs_)}/{len(rs_)}"
        return s

    summ = summarize([r for r in records if r["kind"] in ("fixture", "random")])
    report["summary"] = summ
    report["summary_pair"] = summarize([r for r in records if r["kind"] == "pair"])
    report["stages"] = stage_checks(eng)

    grids = sorted({tuple(c["grid"]) for r in records for pc in r["pictures"] for c in pc["crops"]} | set(EXTRA_GRIDS))
    report["pos_table"] = pos_table_checks(grids, args.seed)
    report["unshuffle"] = unshuffle_checks(grids)
    picks = {}
    for (rid, rgb) in fixture_rgbs:
        p = vh.plan(*rgb.shape[:2])
        for c, u8 in zip(p.crops, vh.crop_images(rgb, p)):
            picks.setdefault((c.kind, c.grid), (f"{rid}/{c.kind}{c.row}{c.col}", u8))
    chosen = [picks[k] for k in [("tile", (32, 32)), ("thumbnail", (26, 36)), ("single", (18, 18)), ("single", (18, 54))]
              if k in picks]
    report["tower_contract"] = tower_contract(chosen, args.seed)
    report["pillow_resampler_instead_of_torch"] = pillow_instead(fixture_rgbs)
    report["n_image_tokens"] = {r["id"]: r["n_image_tokens"] for r in records}
    report["seconds"] = round(time.time() - t0, 1)

    tc = report["tower_contract"]["crops"]
    fails = [k for k in keys if not all(r[k] for r in records)]
    fails += ["provider_cut"] if not all(r["provider_cut"]["bit_equal"] for r in records) else []
    fails += ["normalize_lut"] if not report["stages"]["normalize_lut_bit_equal"] else []
    fails += ["patch_layout"] if not report["stages"]["patch_layout_equal"] else []
    fails += ["pos_table"] if not report["pos_table"]["pass"] else []
    fails += ["unshuffle"] if not report["unshuffle"]["pass"] else []
    fails += ["neg_pos_hw"] if not report["pos_table"]["negative_hw_swapped"]["red"] else []
    fails += ["neg_unshuffle"] if (report["unshuffle"]["negative_patch_order_0213_equal"]
                                   or report["unshuffle"]["negative_tokens_column_major_equal"]) else []
    fails += ["neg_layout"] if report["stages"]["negative_patch_layout_cyx_equal"] else []
    fails += ["tower_contract"] if not all(c["host_vs_hf_padded"]["cos"] > 0.999999 for c in tc) else []
    fails += ["neg_tower_mask"] if not all(c["negative_no_key_mask"]["max_abs"] > 1e-3 for c in tc if "negative_no_key_mask" in c) else []
    report["pass"] = not fails
    report["fails"] = fails
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(json.dumps(summ))
    print("pos_table", report["pos_table"]["n_pass"], "/", report["pos_table"]["n_grids"], "unshuffle", report["unshuffle"]["pass"],
          "tower", [round(c["host_vs_hf_padded"]["max_abs"], 9) for c in tc])
    print("PASS" if not fails else f"FAIL: {fails}", flush=True)

    if not args.no_grid_table:
        gt = grid_table(runner, eng)
        gt["summary"]["fixture"] = {
            r["id"]: {"size": r["size_in"][0], "capped": r["pictures"][0]["size_capped"], "rows": r["pictures"][0]["rows"],
                      "cols": r["pictures"][0]["cols"], "crops": len(r["pictures"][0]["crops"]),
                      "grids": sorted({tuple(c["grid"]) for c in r["pictures"][0]["crops"]}), "tokens": r["n_image_tokens"]}
            for r in records if r["kind"] == "fixture"}
        Path(args.grid_out).write_text(json.dumps(gt, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        print("grid table", gt["host_equal"], json.dumps(gt["summary"]), f"{gt['seconds']} s")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
