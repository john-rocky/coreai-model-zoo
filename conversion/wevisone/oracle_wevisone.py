#!/usr/bin/env python3
"""fp32 CPU oracle for WeVisOne-VL-2B: the package's own transformers path, recorded for every gate.

Runs in a venv carrying the package's pins (transformers 5.12.1, torch 2.11.0) with the reference
package imported by path (appended to sys.path, never in front of it):

    <oracle venv>/bin/python -I oracle_wevisone.py --pkg <dir holding wevisone_vl/> \
        --model <snapshot dir of tencent/WeVisOne-VL-2B> --fixtures <fixtures.json> --out <oracle dir> \
        --jobs O2-448:syn_a_det,O2-512:syn_a_det [--dtype float32|bfloat16] [--hidden-ids a,b]

Forms (the fp32 ones can share a process; O2-448-bf16 needs --dtype bfloat16):
  O2-448 / O2-512  the original image resized with PIL BICUBIC to S x S (both sides, aspect not kept) and handed to
                   the package; asserts image_grid_thw == [[1, S/16, S/16]]; generation = the package's call with
                   repetition_penalty 1.0 (pure greedy), max_new_tokens 768
  O2-448-rp105     O2-448 with repetition_penalty 1.05 (the package default)
  O1               the author's form: the original image, the package's min / max pixels, repetition_penalty 1.05,
                   max_new_tokens 768 (the package default is 4096; the cap is the same as O2's)
  O2-448-bf16      O2-448 with the VLM cast to bfloat16 (the mask decoder stays fp32, as the package does)

Per run, `<out>/<form>/<id>.npz` + `<id>.json`: the chat-templated prompt and its ids, pixel_values, image_grid_thw,
the language model's position_ids as passed to it (prompt call and every generation step; rows = text index, t, h, w),
the generated ids and text (raw and the package's cut), per-step top-5 (id, logit) and the softmax / logit top-2 gaps,
the prompt-last full-vocabulary logits, the wall seconds. O2-448 / O2-512 also keep the vision outputs (image_embeds
and the three deepstack rows) and the first 16 steps' full logits (gate D); `--hidden-ids` rows keep the input
embeddings and the 28 decoder-layer outputs of the prompt call. segmentation / matting rows (O2-448, O2-512, O1) also
record the mask decoder's inputs and outputs the way `WeVisOneVL.decode_masks` builds them: the merger inputs of ViT
layers 5 / 11 / 17 / 23 (block-major and as [1, 1024, 2g, 2g] maps), the teacher-forced ids and the last decoder
layer's output at every `<OBJ>`, the same hidden state taken during generation, the boxes, the 1024 x 1024 RGB
canvas, the four logit maps and the alpha at the original size. The RGB canvas is the ORIGINAL image resized to
1024 x 1024 (the package resizes the image it was given; for O2 that would be the S x S copy) — the deployed form.

A run whose json exists is skipped (resume).
"""
from __future__ import annotations

import os

for _k, _v in (("TOKENIZERS_PARALLELISM", "false"), ("OMP_NUM_THREADS", "1"), ("VECLIB_MAXIMUM_THREADS", "1"),
               ("MKL_NUM_THREADS", "1"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    os.environ[_k] = _v

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from PIL import Image  # noqa: E402

FORMS = {
    "O2-448": {"side": 448, "rp": 1.0, "dtype": "float32"},
    "O2-512": {"side": 512, "rp": 1.0, "dtype": "float32"},
    "O2-448-rp105": {"side": 448, "rp": 1.05, "dtype": "float32"},
    "O1": {"side": None, "rp": 1.05, "dtype": "float32"},
    "O2-448-bf16": {"side": 448, "rp": 1.0, "dtype": "bfloat16"},
}
KEEP_VISION = {"O2-448", "O2-512"}
KEEP_STEPS = 16
DENSE_TASKS = {"segmentation", "matting"}
DENSE_FORMS = {"O2-448", "O2-512", "O1"}
IMAGE_PAD, VISION_START, VISION_END, IM_END, OBJ = 151655, 151652, 151653, 151645, 152678


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def as_np(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return t.detach().to(torch.float32 if dtype == np.float32 else t.dtype).cpu().numpy().astype(dtype, copy=False)


def step_stats(logits: list[torch.Tensor], chosen: list[int]) -> dict:
    """Per generation step: top-5 ids / logits, softmax top-2 gap (p1 - p2), logit top-2 gap, raw argmax."""
    n = len(logits)
    top_ids = np.zeros((n, 5), np.int64)
    top_val = np.zeros((n, 5), np.float32)
    gap_p = np.zeros(n, np.float64)
    gap_l = np.zeros(n, np.float64)
    for i, row in enumerate(logits):
        r = row[0].double()
        v, ix = torch.topk(r, 5)
        lse = torch.logsumexp(r, 0)
        p = torch.exp(v[:2] - lse)
        top_ids[i] = ix.numpy()
        top_val[i] = v.float().numpy()
        gap_p[i] = float(p[0] - p[1])
        gap_l[i] = float(v[0] - v[1])
    raw_argmax = top_ids[:, 0]
    return {"top5_ids": top_ids, "top5_logits": top_val, "gap_softmax": gap_p, "gap_logit": gap_l,
            "raw_argmax_differs_from_chosen": int(sum(int(a) != int(b) for a, b in zip(raw_argmax, chosen)))}


class Recorder:
    """Forward hooks on the language model, its decoder layers and the vision tower, switched per phase."""

    def __init__(self, model):
        self.model = model
        base = model.model
        self.lm = base.language_model
        self.layers = self.lm.layers
        self.visual = base.visual
        self.on = False
        self.layers_on = False
        self.last_on = False
        self.vision_on = False
        self.clear()
        self.lm.register_forward_pre_hook(self._lm_pre, with_kwargs=True)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._layer_hook(i))
        self.visual.register_forward_hook(self._vis_hook)

    def clear(self):
        self.calls = 0
        self.pos: list[np.ndarray | None] = []
        self.embeds = None
        self.layer_out: dict[int, np.ndarray] = {}
        self.last: list[np.ndarray] = []
        self.vision = None

    def _lm_pre(self, _mod, _args, kwargs):
        if not self.on:
            return None
        p = kwargs.get("position_ids")
        self.pos.append(None if p is None else p.detach().to(torch.int64).cpu().numpy()[:, 0, :].copy())
        if self.calls == 0 and self.layers_on and kwargs.get("inputs_embeds") is not None:
            self.embeds = as_np(kwargs["inputs_embeds"][0])
        self.calls += 1
        return None

    def _layer_hook(self, i):
        def hook(_mod, _args, out):
            if not self.on:
                return
            h = out[0] if isinstance(out, (tuple, list)) else out
            call = self.calls - 1
            if self.layers_on and call == 0:
                self.layer_out[i] = as_np(h[0])
            if self.last_on and i == len(self.layers) - 1:
                self.last.append(as_np(h[0]))
        return hook

    def _vis_hook(self, _mod, _args, out):
        if not (self.on and self.vision_on):
            return
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        deep = out.deepstack_features if hasattr(out, "deepstack_features") else out[2]
        self.vision = {"image_embeds": as_np(pooled), "deepstack_embeds": np.stack([as_np(d) for d in deep])}


def form_image(original: Image.Image, form: str) -> Image.Image:
    side = FORMS[form]["side"]
    if side is None:
        return original
    return original.resize((side, side), resample=Image.Resampling.BICUBIC)


def host_patches(img: Image.Image) -> np.ndarray:
    """NumPy reference of the processor for an image already at a multiple of 32: (x/255 - 0.5)/0.5, frame
    duplicated (temporal 2), block-major [H/32, W/32, 2, 2] patch order, each patch flattened as [C, T, 16, 16]."""
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    h, w, _ = x.shape
    p, m = 16, 2
    gh, gw = h // p, w // p
    x = x.transpose(2, 0, 1)                                  # C, H, W
    x = np.stack([x, x], axis=0)                              # T, C, H, W
    x = x.reshape(2, 3, gh // m, m, p, gw // m, m, p)         # T C bh mh ph bw mw pw
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)                   # bh bw mh mw C T ph pw
    return x.reshape(gh * gw, 3 * 2 * p * p)


def env_record() -> dict:
    import importlib.metadata as md
    import subprocess

    v = {}
    for p in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy", "pillow",
              "accelerate", "torchvision"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    sh = lambda *c: subprocess.run(list(c), capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"versions": v, "python": sys.version.split()[0], "platform": platform.platform(),
            "macos_build": sh("sw_vers", "-buildVersion"), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                                                    "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE")},
            "torch_config": torch.__config__.show().splitlines()[:40], "pid": os.getpid()}


def run_dense(wv, rec, pkg, row, vlm_image, original, response, gen_ids, prompt_ids, prompt_hidden_calls) -> tuple[dict, dict]:
    """The package's decode_masks, step by step, with every tensor kept (see the module docstring)."""
    t0 = time.perf_counter()
    boxes = pkg["parse_seg_boxes"](response)
    rec_json: dict = {"boxes01": boxes, "n_boxes": len(boxes)}
    arrays: dict = {}
    if not boxes:
        rec_json["skipped"] = "no box before an <OBJ> in the response"
        return rec_json, arrays
    messages = wv._messages(vlm_image, row["prompt"]) + [{"role": "assistant", "content": response}]
    teacher = wv._inputs(messages, vlm_image, generation=False)
    t_ids = teacher["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)
    n_seg = sum(1 for t in t_ids if t == wv.seg_token_id)
    # response re-tokenized vs the generated ids (r2: is the generation-time hidden state enough?)
    gen_body = [t for t in gen_ids]
    if gen_body and gen_body[-1] == IM_END:
        gen_body = gen_body[:-1]
    t_body = t_ids[prompt_len:]
    if IM_END in t_body:
        t_body = t_body[:t_body.index(IM_END)]
    same_prefix = t_ids[:prompt_len] == list(prompt_ids)
    d = next((i for i in range(min(len(gen_body), len(t_body))) if gen_body[i] != t_body[i]), None)
    rec_json["retokenize"] = {"teacher_len": len(t_ids), "teacher_response_tokens": len(t_body),
                              "generated_tokens_wo_im_end": len(gen_body),
                              "equal": gen_body == t_body, "first_difference": d, "prompt_prefix_equal": same_prefix}
    canvas = original.convert("RGB").resize((wv.cnn_size, wv.cnn_size), resample=Image.Resampling.BICUBIC)
    cnn = pkg["assert_cnn_rgb"](pkg["_pil_to_float_rgb"](canvas).to(wv.device, dtype=wv.dtype),
                                cnn_image_size=wv.cnn_size)
    old = getattr(wv.model.config, "use_cache", True)
    wv.model.config.use_cache = False
    rec.on = False
    try:
        with torch.inference_mode():
            with pkg["VisualFeatureCapture"](wv.model, feature_layers=wv.cfg.vision_feature_layers) as vc:
                with pkg["LastHiddenCapture"](wv.model) as hc:
                    wv.model(**teacher, output_hidden_states=False, return_dict=True, use_cache=False)
    finally:
        wv.model.config.use_cache = old
    hidden = hc.hidden
    t_seg = pkg["extract_t_seg"](hidden, teacher["input_ids"], wv.seg_token_id)[0]
    grid_thw = teacher["image_grid_thw"]
    grid_hw = pkg["premerge_grid_hw"](grid_thw)[0]
    maps = pkg["per_sample_premerge_feature_maps"](vc.premerge_features, wv.cfg.vision_feature_layers, grid_thw,
                                                   wv.spatial_merge_size)[0]
    arrays["tap_block_major"] = np.stack([as_np(vc.premerge_features[int(i)]) for i in wv.cfg.vision_feature_layers])
    arrays["tap_maps"] = np.stack([as_np(m[0]) for m in maps])
    arrays["teacher_input_ids"] = np.asarray(t_ids, np.int64)
    arrays["t_seg"] = as_np(t_seg)
    obj_pos = [i for i, t in enumerate(t_ids) if t == wv.seg_token_id]
    arrays["obj_positions"] = np.asarray(obj_pos, np.int64)
    # generation-time hidden at each generated <OBJ>: its hidden state is computed on the call that feeds it back
    gen_hidden = []
    for k, t in enumerate(gen_ids):
        if t == OBJ:
            call = k + 1
            gen_hidden.append(prompt_hidden_calls[call][-1] if call < len(prompt_hidden_calls) else None)
    ok = [g for g in gen_hidden if g is not None]
    if ok and len(ok) == len(gen_hidden):
        gh = np.stack(ok)
        arrays["obj_hidden_generation"] = gh
        if gh.shape == arrays["t_seg"].shape:
            a, b = gh.astype(np.float64), arrays["t_seg"].astype(np.float64)
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
            rec_json["obj_hidden_generation_vs_teacher"] = {"n": int(gh.shape[0]), "cos_min": float(cos.min()),
                                                          "max_abs_diff": float(np.abs(a - b).max())}
    rec_json["obj_hidden_generation_available"] = f"{len(ok)}/{len(gen_hidden)}"
    count = min(len(boxes), int(t_seg.shape[0]))
    boxes_t = torch.tensor(boxes[:count], device=wv.device, dtype=wv.dtype)
    seg = t_seg[:count].to(wv.device, dtype=wv.dtype)
    outs = {k: [] for k in ("mask_logits", "mask_logits_aux", "mask_logits_coarse", "mask_logits_lowres")}
    alphas = []
    batches = []
    with torch.inference_mode():
        for start in range(0, count, wv.decoder_batch_size):
            end = min(count, start + wv.decoder_batch_size)
            o = wv.mask_decoder(image=cnn, visual_features_2d=[m.to(wv.device, dtype=wv.dtype) for m in maps],
                                t_seg=seg[start:end], boxes01=boxes_t[start:end], grid_hw=grid_hw)
            for k in outs:
                outs[k].append(as_np(o[k]))
            alphas.extend(wv._masks_from_logits(o["mask_logits"], original.height, original.width))
            batches.append([start, end])
    for k, v in outs.items():
        arrays[k] = np.concatenate(v, axis=0)
    arrays["alpha"] = np.stack(alphas).astype(np.float32)
    arrays["cnn_image"] = as_np(cnn[0])
    arrays["boxes01"] = np.asarray(boxes[:count], np.float32)
    rec_json.update({"n_obj_teacher": n_seg, "count_decoded": count, "grid_hw_premerge": list(grid_hw),
                     "decoder_batches": batches, "cnn_canvas": "original image, PIL BICUBIC 1024x1024",
                     "alpha_size_hw": [original.height, original.width],
                     "alpha_mean": [float(a.mean()) for a in alphas], "seconds": time.perf_counter() - t0})
    return rec_json, arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True, help="comma list of <form>:<fixture id>")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--hidden-ids", default="", help="fixture ids whose O2 prompt call keeps the 28 layer outputs")
    ap.add_argument("--check-package", action="store_true",
                    help="also run the package's generate_text on the first job and require the same response")
    ap.add_argument("--deadline-s", type=float, default=0.0,
                    help="start no new job after this many seconds from launch; exit 3 if jobs remain")
    args = ap.parse_args()
    t_launch = time.perf_counter()

    sys.path.append(str(Path(args.pkg).resolve()))
    from wevisone_vl import engine as E  # noqa: E402
    from wevisone_vl import feature_bridge as FB  # noqa: E402

    pkg = {"parse_seg_boxes": E.parse_seg_boxes, "_pil_to_float_rgb": E._pil_to_float_rgb,
           "assert_cnn_rgb": FB.assert_cnn_rgb, "VisualFeatureCapture": FB.VisualFeatureCapture,
           "LastHiddenCapture": FB.LastHiddenCapture, "extract_t_seg": FB.extract_t_seg,
           "premerge_grid_hw": FB.premerge_grid_hw, "per_sample_premerge_feature_maps": FB.per_sample_premerge_feature_maps}

    fx_path = Path(args.fixtures).resolve()
    fixtures = json.loads(fx_path.read_text())
    rows = {r["id"]: r for r in fixtures["rows"]}
    jobs = []
    for item in args.jobs.split(","):
        form, rid = item.strip().split(":", 1)
        if form not in FORMS or rid not in rows:
            raise SystemExit(f"unknown job {item}")
        if FORMS[form]["dtype"] != args.dtype:
            raise SystemExit(f"job {item} needs --dtype {FORMS[form]['dtype']}")
        jobs.append((form, rid))
    hidden_ids = {s for s in args.hidden_ids.split(",") if s}
    out_root = Path(args.out).resolve()
    todo = [(f, r) for f, r in jobs if not (out_root / f / f"{r}.json").exists()]
    print(f"[oracle] {len(jobs)} jobs, {len(todo)} to run, dtype {args.dtype}, threads {torch.get_num_threads()}",
          flush=True)
    if not todo:
        return 0

    t0 = time.perf_counter()
    wv = E.WeVisOneVL(args.model, device="cpu")
    load_s = time.perf_counter() - t0
    if args.dtype == "bfloat16":
        wv.model.to(torch.bfloat16)
        wv.dtype = torch.bfloat16
    pdt = {str(p.dtype) for p in wv.model.parameters()}
    print(f"[oracle] loaded in {load_s:.1f}s, model dtypes {pdt}, mask decoder "
          f"{sum(p.numel() for p in wv.mask_decoder.parameters()):,} params", flush=True)
    rec = Recorder(wv.model)
    env = env_record()
    env.update({"model_dir": str(Path(args.model).resolve()), "fixtures": str(fx_path),
                "fixtures_sha256": sha256_file(fx_path), "package_dir": str(Path(args.pkg).resolve()),
                "engine_py_sha256": sha256_file(Path(E.__file__)), "model_dtypes": sorted(pdt), "load_seconds": load_s,
                "script_sha256": sha256_file(Path(__file__))})

    checked = not args.check_package
    for n_done, (form, rid) in enumerate(todo):
        if args.deadline_s and time.perf_counter() - t_launch > args.deadline_s:
            print(f"[oracle] deadline {args.deadline_s:.0f}s reached: {len(todo) - n_done} jobs left", flush=True)
            return 3
        if (out_root / form / f"{rid}.json").exists():   # another process finished it meanwhile
            continue
        row = rows[rid]
        spec = FORMS[form]
        t_job = time.perf_counter()
        original = Image.open(fx_path.parent / row["image"]).convert("RGB")
        image = form_image(original, form)
        messages = wv._messages(image, row["prompt"])
        chat_text = wv.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = wv._inputs(messages, image, generation=True)
        ids = inputs["input_ids"][0].tolist()
        grid = inputs["image_grid_thw"].tolist()
        if spec["side"] is not None:
            g16 = spec["side"] // 16
            if grid != [[1, g16, g16]]:
                raise SystemExit(f"{form}:{rid}: image_grid_thw {grid} != [[1, {g16}, {g16}]]")
        n_img = sum(1 for t in ids if t == IMAGE_PAD)
        img_start = ids.index(IMAGE_PAD)
        gh, gw = grid[0][1] // 2, grid[0][2] // 2
        assert n_img == gh * gw, (n_img, gh, gw)
        pv = inputs["pixel_values"]
        host = None
        if spec["side"] is not None:
            ref = host_patches(image)
            host = {"max_abs_diff": float(np.abs(ref - pv.float().numpy()).max()),
                    "shape_equal": list(ref.shape) == list(pv.shape)}
        dense = row["task"] in DENSE_TASKS and form in DENSE_FORMS
        rec.clear()
        rec.on = True
        rec.layers_on = rid in hidden_ids and form in KEEP_VISION
        rec.last_on = dense
        rec.vision_on = form in KEEP_VISION
        t_gen = time.perf_counter()
        with torch.inference_mode():
            gen = wv.model.generate(**inputs, max_new_tokens=args.max_new, repetition_penalty=float(spec["rp"]),
                                    use_cache=True, do_sample=False, output_logits=True,
                                    return_dict_in_generate=True)
        gen_s = time.perf_counter() - t_gen
        rec.on = False
        seq = gen.sequences[0].tolist()
        gen_ids = seq[len(ids):]
        raw = wv.processor.batch_decode([gen_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        response = raw.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()
        stats = step_stats(list(gen.logits), gen_ids)
        hit_max = len(gen_ids) >= args.max_new and gen_ids[-1] not in (IM_END, 151643)
        pos = rec.pos
        if pos[0] is None:
            raise SystemExit(f"{form}:{rid}: the language model got no position_ids")
        pos_prompt = pos[0]
        pos_gen = np.stack([p[:, 0] for p in pos[1:]]) if len(pos) > 1 else np.zeros((0, pos_prompt.shape[0]), np.int64)
        arrays = {"input_ids": np.asarray(ids, np.int64), "pixel_values": as_np(pv),
                  "image_grid_thw": np.asarray(grid, np.int64), "position_ids_prompt": pos_prompt,
                  "position_ids_gen": pos_gen, "gen_ids": np.asarray(gen_ids, np.int64),
                  "prompt_last_logits": as_np(gen.logits[0][0]), "top5_ids": stats["top5_ids"],
                  "top5_logits": stats["top5_logits"], "gap_softmax": stats["gap_softmax"],
                  "gap_logit": stats["gap_logit"]}
        if form in KEEP_VISION:
            arrays["step_logits"] = np.stack([as_np(gen.logits[i][0]) for i in range(min(KEEP_STEPS, len(gen.logits)))])
            if rec.vision is None:
                raise SystemExit(f"{form}:{rid}: vision outputs not captured")
            arrays["image_embeds"] = rec.vision["image_embeds"]
            arrays["deepstack_embeds"] = rec.vision["deepstack_embeds"]
            # every prompt position's logits (gate A reads the last four): one full forward without the cache
            t_p = time.perf_counter()
            with torch.inference_mode():
                full = wv.model(**inputs, use_cache=False, return_dict=True).logits[0]
            ps = step_stats([full[i:i + 1] for i in range(full.shape[0])], [int(t) for t in full.argmax(-1)])
            arrays["prompt_logits_last4"] = as_np(full[-4:])
            arrays["prompt_top5_ids"] = ps["top5_ids"]
            arrays["prompt_top5_logits"] = ps["top5_logits"]
            arrays["prompt_gap_softmax"] = ps["gap_softmax"]
            out_prompt = {"positions": int(full.shape[0]), "seconds": time.perf_counter() - t_p,
                          "last_vs_generate_max_abs_diff": float(np.abs(as_np(full[-1]) - arrays["prompt_last_logits"]).max())}
            del full
        if rec.layers_on:
            arrays["hidden_embeds"] = rec.embeds
            arrays["hidden_layers"] = np.stack([rec.layer_out[i] for i in range(len(rec.layers))])
        out = {"id": rid, "form": form, "task": row["task"], "query": row["query"], "prompt": row["prompt"],
               "chat_text": chat_text, "prompt_len": len(ids), "image_size_wh": list(image.size),
               "original_size_wh": list(original.size), "image_grid_thw": grid, "merged_grid_hw": [gh, gw],
               "n_image_tokens": n_img, "image_start": img_start,
               "repetition_penalty": spec["rp"], "max_new_tokens": args.max_new, "dtype": args.dtype,
               "gen_len": len(gen_ids), "hit_max": hit_max, "ended_with": gen_ids[-1] if gen_ids else None,
               "raw_text": raw, "response": response, "gen_ids": gen_ids,
               "raw_argmax_differs_from_chosen": stats["raw_argmax_differs_from_chosen"],
               "min_gap_softmax_first16": float(stats["gap_softmax"][:16].min()) if len(gen_ids) else None,
               "position_rows": int(pos_prompt.shape[0]), "host_patches_vs_processor": host,
               "seconds": {"generate": gen_s}}
        if form in KEEP_VISION:
            out["prompt_logits_pass"] = out_prompt
        if dense:
            calls = rec.last
            d_json, d_arrays = run_dense(wv, rec, pkg, row, image, original, response, gen_ids, ids,
                                         [c for c in calls])
            out["dense"] = d_json
            arrays.update({f"dense_{k}": v for k, v in d_arrays.items()})
        if not checked:
            t_c = time.perf_counter()
            pkg_resp = wv.generate_text(image, row["prompt"], max_new_tokens=args.max_new,
                                        repetition_penalty=float(spec["rp"]))
            out["package_generate_text_check"] = {"equal": pkg_resp == response, "seconds": time.perf_counter() - t_c,
                                                  "package_response": pkg_resp}
            checked = True
            if pkg_resp != response:
                print(f"[oracle] WARNING package generate_text differs on {form}:{rid}", flush=True)
        out["seconds"]["total"] = time.perf_counter() - t_job
        out["environment"] = env
        out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        d = out_root / form
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{rid}.npz", **arrays)
        (d / f"{rid}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        print(f"[oracle] {form}:{rid} S={len(ids)} N={n_img} gen={len(gen_ids)}{' MAX' if hit_max else ''} "
              f"{gen_s:.1f}s/{out['seconds']['total']:.1f}s | {response[:110]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
