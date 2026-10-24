#!/usr/bin/env python3
"""fp32 torch ladder: the Core AI overlay for Qwen3-VL (decoder + fixed-grid vision) vs the WeVisOne-VL-2B oracle.

The overlay (`coreai_models.models.macos.qwen3_vl_pipelined`) is the code the exporter traces. This runs it eagerly
in fp32 on one CPU thread against the fixed-grid oracle (`oracle_wevisone.py`, form O2-448 for grid 14, O2-512 for
grid 16) on every fixture row, and records:

  positions  the (t, h, w) the overlay's forward hands to its attention (forward pre-hook on layer 0's attention)
             vs the position_ids the package handed to its language model, at every prompt position; and at every
             generation step the oracle took, the host rule p - amount vs the oracle's position (the 16 teacher-forced
             steps are also read from the hook)
  vision     the overlay tower on the oracle's own pixel_values (block-major, checked against a NumPy patchify of the
             S x S image by the oracle) vs the oracle's image_embeds and three deepstack rows: cosine of each matrix
  layers     on the `--hidden-ids` rows, the 28 decoder-layer outputs of the prompt call (and the input embeddings)
  prefill    the prompt-last logits: top-1 vs the oracle, max |d|, cosine
  decode     the oracle's generated ids teacher-forced one per call through the KV cache: argmax vs the oracle's next
             id at the first 16 steps, and the logits vs the oracle's (cosine, max |d|)
  e2e        overlay tower -> overlay decoder (no oracle tensors): prompt-last top-1
  red arms   run first on every row: rope_shift_amount + 1, and planes_1d (start = 1 << 30, amount = 0, image rows in
             place). RED = the position check fails AND the prompt-last max |d| exceeds 2x the green run's.

The decoder takes the host contract of the exporter: `<|image_pad|>` -> V + slot, rope_shift_start = first image
pad + N, rope_shift_amount = N - g. Stop line (ROUND1): any position mismatch, or any prefill top-1 mismatch.

    QW <coreai venv>/bin/python parity_torch_ladder.py --oracle <dir> --fixtures <fixtures.json> --grid 14 \
        --out <json> [--hidden-ids a,b] [--hf-id tencent/WeVisOne-VL-2B]
"""
from __future__ import annotations

import os

for _k, _v in (("OMP_NUM_THREADS", "1"), ("VECLIB_MAXIMUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("TOKENIZERS_PARALLELISM", "false"), ("HF_HUB_OFFLINE", "1")):
    os.environ[_k] = _v

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from coreai_models.models.macos.qwen3_vl_pipelined import (  # noqa: E402
    Qwen3VLPipelinedForCausalLM,
    Qwen3VLVisionEncoder,
)

V = 152680
IMAGE_PAD = 151655
TEXT_ONLY_START = 1 << 30
COS_BAR = 0.9999
N_DECODE = 16
FORM_FOR_GRID = {14: "O2-448", 16: "O2-512"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def row_cos_min(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return float(c.min())


def topk(x, k=5):
    x = np.asarray(x, np.float32)
    i = np.argsort(-x, kind="stable")[:k]
    return [int(t) for t in i], [float(x[t]) for t in i]


def host_inputs(ids: list[int], g: int) -> tuple[np.ndarray, int, int, int]:
    """The exporter's host contract: image pads -> V + slot; shift starts after the image, by N - g."""
    a = np.asarray(ids, np.int64)
    pads = np.flatnonzero(a == IMAGE_PAD)
    n = g * g
    if len(pads) != n or not np.array_equal(pads, np.arange(pads[0], pads[0] + n)):
        raise SystemExit(f"image pads {len(pads)} (contiguous {np.array_equal(pads, np.arange(pads[0], pads[0] + len(pads)))}) != {n}")
    mapped = a.copy()
    mapped[pads] = V + np.arange(n)
    return mapped, int(pads[0]), int(pads[0]) + n, n - g


class Probe:
    """Hooks on the overlay decoder: layer 0's attention inputs (positions), layer 0's input, layer outputs."""

    def __init__(self, model):
        self.on_layers = False
        self.pos: list[np.ndarray] = []
        self.embeds = None
        self.layer_out: dict[int, np.ndarray] = {}
        layers = model.model.layers
        layers[0].self_attn.register_forward_pre_hook(self._attn_pre)
        layers[0].register_forward_pre_hook(self._layer0_pre)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._out(i))

    def reset(self, layers: bool):
        self.on_layers = layers
        self.pos = []
        self.embeds = None
        self.layer_out = {}

    def _attn_pre(self, _m, args):
        _x, _position_ids, pos_t, pos_h, pos_w, _cache = args
        self.pos.append(torch.stack([pos_t[0], pos_h[0], pos_w[0]]).to(torch.int64).numpy().copy())

    def _layer0_pre(self, _m, args):
        if self.on_layers:
            self.embeds = args[0][0].detach().float().numpy().copy()

    def _out(self, i):
        def hook(_m, _a, out):
            if self.on_layers:
                self.layer_out[i] = out[0].detach().float().numpy().copy()
        return hook


class Runner:
    def __init__(self, model, max_len: int):
        self.model = model
        cfg = model.config
        self.shape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, max_len, cfg.head_dim)

    def fresh(self):
        return torch.zeros(self.shape, dtype=torch.float32), torch.zeros(self.shape, dtype=torch.float32)

    @torch.no_grad()
    def call(self, ids, total: int, emb, ds, start: int, amount: int, kc, vc) -> np.ndarray:
        out = self.model(torch.as_tensor(np.asarray(ids, np.int64)[None], dtype=torch.int32),
                         torch.arange(total, dtype=torch.int32)[None], emb, ds,
                         torch.tensor([start], dtype=torch.int32), torch.tensor([amount], dtype=torch.int32), kc, vc)
        return out[0].float().numpy()


def compare_positions(overlay: np.ndarray, oracle_thw: np.ndarray, mapped, g: int, start: int) -> dict:
    eq = overlay == oracle_thw
    bad = np.flatnonzero(~eq.all(0))
    rec = {"n": int(overlay.shape[1]), "mismatches": int(len(bad)), "equal": bool(len(bad) == 0)}
    if len(bad):
        rows = []
        for idx in bad[:12]:
            idx = int(idx)
            rows.append({"index": idx, "id": int(mapped[idx]), "oracle_thw": oracle_thw[:, idx].tolist(),
                         "overlay_thw": overlay[:, idx].tolist()})
        rec["first_mismatches"] = rows
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", required=True, help="oracle dir holding <form>/<id>.npz/json")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--grid", type=int, required=True, choices=[14, 16])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-ids", default="")
    ap.add_argument("--hf-id", default="tencent/WeVisOne-VL-2B")
    ap.add_argument("--only", default="", help="comma list of fixture ids (default: all)")
    args = ap.parse_args()

    g = args.grid
    form = FORM_FOR_GRID[g]
    odir = Path(args.oracle) / form
    rows = json.loads(Path(args.fixtures).read_text())["rows"]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    hidden_ids = {s for s in args.hidden_ids.split(",") if s}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    model = Qwen3VLPipelinedForCausalLM.from_hf(args.hf_id, target_dtype=torch.float32, grid_h=g, grid_w=g)
    vis = Qwen3VLVisionEncoder.from_hf(args.hf_id, target_dtype=torch.float32, grid_h=g, grid_w=g)
    load_s = time.perf_counter() - t0
    assert model.config.vocab_size == V, model.config.vocab_size
    assert not model.config.tie_word_embeddings
    probe = Probe(model)
    out = {"schema": "wevisone-torch-ladder/1", "grid": g, "form": form, "hf_id": args.hf_id,
           "oracle_dir": str(odir), "fixtures_sha256": sha256_file(Path(args.fixtures)),
           "bars": {"positions": "every prompt position and every generation step equal",
                    "vision_cos": COS_BAR, "layer_cos": COS_BAR, "prefill_top1": "all rows",
                    "decode": f"{N_DECODE}/{N_DECODE} teacher-forced argmax (or the whole answer when shorter)",
                    "red": "position check FAIL and prompt-last max|d| > 2x green"},
           "load_seconds": load_s, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "lm_head_untied": True, "rows": {}}
    stop_rows = []
    for row in rows:
        rid = row["id"]
        oj = odir / f"{rid}.json"
        if not oj.exists():
            out["rows"][rid] = {"skipped": "no oracle json"}
            continue
        t_row = time.perf_counter()
        ojs = json.loads(oj.read_text())
        z = np.load(odir / f"{rid}.npz")
        ids = z["input_ids"].tolist()
        S = len(ids)
        mapped, img_start, start, amount = host_inputs(ids, g)
        gen = z["gen_ids"].tolist()
        pos_o = z["position_ids_prompt"]          # [4, S]: text index, t, h, w
        oracle_thw = pos_o[1:4]
        rec: dict = {"S": S, "N": g * g, "image_start": img_start, "rope_shift_start": start,
                     "rope_shift_amount": amount, "gen_len": len(gen)}
        # the pixel order the overlay assumes (block-major) is what the oracle fed its own tower
        hp = ojs.get("host_patches_vs_processor") or {}
        pv = torch.from_numpy(z["pixel_values"].astype(np.float32))
        rec["pixel_values_block_major"] = {"shape": list(pv.shape), "numpy_patchify_max_abs_diff": hp.get("max_abs_diff")}
        if list(pv.shape) != [vis.n_patches, 1536] or hp.get("max_abs_diff") is None or hp["max_abs_diff"] > 1e-6:
            raise SystemExit(f"{rid}: pixel_values {list(pv.shape)} / patchify diff {hp.get('max_abs_diff')} — not block-major")

        # --- vision ---
        with torch.no_grad():
            v_img, v_ds = vis(pv)
        v_img = v_img.float().numpy()
        v_ds = v_ds.float().numpy().reshape(3, g * g, -1)
        o_img, o_ds = z["image_embeds"], z["deepstack_embeds"]
        rec["vision"] = {"image_embeds_cos": cos(v_img, o_img), "image_embeds_row_cos_min": row_cos_min(v_img, o_img),
                         "image_embeds_max_abs_d": float(np.abs(v_img - o_img).max()),
                         "deepstack_cos": [cos(v_ds[k], o_ds[k]) for k in range(3)],
                         "deepstack_row_cos_min": [row_cos_min(v_ds[k], o_ds[k]) for k in range(3)]}
        rec["vision"]["pass"] = bool(min([rec["vision"]["image_embeds_cos"]] + rec["vision"]["deepstack_cos"]) >= COS_BAR)

        emb = torch.from_numpy(o_img.astype(np.float32))
        ds = torch.from_numpy(o_ds.reshape(3 * g * g, -1).astype(np.float32))
        runner = Runner(model, S + N_DECODE + 2)
        ref_last = z["prompt_last_logits"]

        # --- red arms first ---
        red = {}
        for arm, (st, am) in {"amount_plus_1": (start, amount + 1), "planes_1d": (TEXT_ONLY_START, 0)}.items():
            probe.reset(False)
            kc, vc = runner.fresh()
            lg = runner.call(mapped, S, emb, ds, st, am, kc, vc)
            red[arm] = {"start": st, "amount": am, "positions": compare_positions(probe.pos[0], oracle_thw, mapped, g, start),
                        "last_max_abs_d": float(np.abs(lg[-1] - ref_last).max()),
                        "last_top1_equal": int(np.argmax(lg[-1])) == int(np.argmax(ref_last))}

        # --- green prefill (+ layers on the hidden rows) ---
        probe.reset(rid in hidden_ids and "hidden_layers" in z.files)
        kc, vc = runner.fresh()
        lg = runner.call(mapped, S, emb, ds, start, amount, kc, vc)
        pos_rec = compare_positions(probe.pos[0], oracle_thw, mapped, g, start)
        last = lg[-1]
        te, ve = topk(last)
        to, vo = topk(ref_last)
        rec["prefill"] = {"top1_equal": te[0] == to[0], "overlay_top5": te, "overlay_top5_logits": ve,
                          "oracle_top5": to, "oracle_top5_logits": vo, "max_abs_d": float(np.abs(last - ref_last).max()),
                          "cos": cos(last, ref_last)}
        if probe.on_layers:
            hl = z["hidden_layers"]
            lc = [cos(probe.layer_out[i], hl[i]) for i in range(hl.shape[0])]
            rec["layers"] = {"embeds_cos": cos(probe.embeds, z["hidden_embeds"]), "layer_cos": lc,
                             "layer_cos_min": min(lc), "pass": bool(min(lc) >= COS_BAR)}
        # --- teacher-forced decode through the cache ---
        n = min(N_DECODE, len(gen))
        steps = z["step_logits"] if "step_logits" in z.files else None
        argmax = [int(np.argmax(last))]
        dec_cos, dec_d = [cos(last, steps[0])] if steps is not None else [], [float(np.abs(last - steps[0]).max())] if steps is not None else []
        probe.reset(False)
        for j in range(1, n):
            lgj = runner.call([gen[j - 1]], S + j, emb, ds, start, amount, kc, vc)[-1]
            argmax.append(int(np.argmax(lgj)))
            if steps is not None and j < steps.shape[0]:
                dec_cos.append(cos(lgj, steps[j]))
                dec_d.append(float(np.abs(lgj - steps[j]).max()))
        tf_pos = np.stack([p[:, 0] for p in probe.pos], axis=1) if probe.pos else np.zeros((3, 0), np.int64)
        mism = [k for k in range(n) if argmax[k] != gen[k]]
        rec["decode"] = {"n": n, "equal": n - len(mism), "mismatch_steps": mism, "logits_cos_min": min(dec_cos) if dec_cos else None,
                         "logits_max_abs_d": max(dec_d) if dec_d else None, "pass": not mism}
        # generation-step positions: the hook (teacher-forced steps) and the host rule p - amount (every step)
        pg = z["position_ids_gen"]                 # [steps, 4]
        rule = np.array([[S + k - amount] * 3 for k in range(pg.shape[0])], np.int64).T if pg.shape[0] else np.zeros((3, 0), np.int64)
        gen_rule_eq = bool(np.array_equal(rule, pg[:, 1:4].T)) if pg.shape[0] else True
        hook_eq = bool(np.array_equal(tf_pos, pg[:tf_pos.shape[1], 1:4].T)) if tf_pos.shape[1] else True
        pos_rec["generation_steps"] = {"n": int(pg.shape[0]), "host_rule_equal": gen_rule_eq,
                                       "teacher_forced_hook_steps": int(tf_pos.shape[1]), "hook_equal": hook_eq,
                                       "text_index_row_equals_sequence_index": bool(np.array_equal(
                                           pos_o[0], np.arange(S)))}
        pos_rec["all_equal"] = bool(pos_rec["equal"] and gen_rule_eq and hook_eq)
        rec["positions"] = pos_rec
        # --- e2e: overlay tower -> overlay decoder ---
        kc, vc = runner.fresh()
        probe.reset(False)
        lg_e = runner.call(mapped, S, torch.from_numpy(v_img), torch.from_numpy(v_ds.reshape(3 * g * g, -1)), start,
                           amount, kc, vc)
        rec["e2e_prefill"] = {"top1_equal": int(np.argmax(lg_e[-1])) == to[0], "max_abs_d": float(np.abs(lg_e[-1] - ref_last).max())}
        green_d = rec["prefill"]["max_abs_d"]
        for arm in red.values():
            arm["position_check_failed"] = not arm["positions"]["equal"]
            arm["over_2x_green"] = arm["last_max_abs_d"] > 2 * green_d
            arm["result"] = "RED" if arm["position_check_failed"] and arm["over_2x_green"] else "NOT RED"
        rec["red"] = red
        rec["seconds"] = time.perf_counter() - t_row
        out["rows"][rid] = rec
        if not pos_rec["all_equal"] or not rec["prefill"]["top1_equal"]:
            stop_rows.append(rid)
        print(f"[ladder g{g}] {rid}: pos {'OK' if pos_rec['all_equal'] else 'MISMATCH ' + str(pos_rec['mismatches'])} | "
              f"vision cos {rec['vision']['image_embeds_cos']:.6f}/{min(rec['vision']['deepstack_cos']):.6f} | "
              f"prefill top1 {rec['prefill']['top1_equal']} d {rec['prefill']['max_abs_d']:.2e} | "
              f"decode {rec['decode']['equal']}/{n} | e2e {rec['e2e_prefill']['top1_equal']} | "
              f"red {red['amount_plus_1']['result']}/{red['planes_1d']['result']} | "
              f"{'layers min ' + format(rec['layers']['layer_cos_min'], '.6f') + ' | ' if 'layers' in rec else ''}"
              f"{rec['seconds']:.1f}s", flush=True)

    done = {k: v for k, v in out["rows"].items() if "skipped" not in v}
    summ = {"rows": len(done), "skipped": [k for k, v in out["rows"].items() if "skipped" in v],
            "positions_all_equal": all(v["positions"]["all_equal"] for v in done.values()),
            "vision_cos_min": min(min([v["vision"]["image_embeds_cos"]] + v["vision"]["deepstack_cos"]) for v in done.values()) if done else None,
            "prefill_top1": f"{sum(v['prefill']['top1_equal'] for v in done.values())}/{len(done)}",
            "decode": f"{sum(v['decode']['equal'] for v in done.values())}/{sum(v['decode']['n'] for v in done.values())}",
            "e2e_prefill_top1": f"{sum(v['e2e_prefill']['top1_equal'] for v in done.values())}/{len(done)}",
            "layers": {k: v["layers"]["layer_cos_min"] for k, v in done.items() if "layers" in v},
            "red": {arm: f"{sum(v['red'][arm]['result'] == 'RED' for v in done.values())}/{len(done)} RED"
                    for arm in ("amount_plus_1", "planes_1d")},
            "green_prefill_max_abs_d_max": max(v["prefill"]["max_abs_d"] for v in done.values()) if done else None}
    summ["stop_line"] = {"triggered": bool(stop_rows), "rows": stop_rows}
    ok = (done and summ["positions_all_equal"] and summ["vision_cos_min"] >= COS_BAR
          and all(v["prefill"]["top1_equal"] and v["decode"]["pass"] for v in done.values())
          and all(v.get("layers", {}).get("pass", True) for v in done.values())
          and all(v["red"][a]["result"] == "RED" for v in done.values() for a in ("amount_plus_1", "planes_1d")))
    summ["verdict"] = "PASS" if ok else "FAIL"
    out["summary"] = summ
    out["wall_seconds"] = time.perf_counter() - t_all
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script_sha256"] = sha256_file(Path(__file__))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[ladder g{g}] {summ['verdict']}: {json.dumps({k: summ[k] for k in ('positions_all_equal', 'vision_cos_min', 'prefill_top1', 'decode', 'e2e_prefill_top1', 'red', 'stop_line')})}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
