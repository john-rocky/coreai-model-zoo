#!/usr/bin/env python3
"""fp32 oracle for d1-3B from the provider's own code: the checkpoint's `modeling_d1.py` and the files it imports.

LiquidAI/d1-3B @ da1fe36a (LFM Open License v1.0) ships its runtime as remote code: `D1Model` (LFM2-VL with the
provider's hybrid language model, `lfm2_vl.py` / `hybrid.py`), `SystemOne` (`runner.py`), the prompt and the readout
(`prompt.py`), the answers (`api.py`). This script imports them unchanged and runs every fixture record on the CPU in
fp32 (the card: bf16 changes answers):

    model = AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, dtype=torch.float32)    # D1Model
    engine = model.engine                                                                          # runner.SystemOne

  (a) the API path: engine.run([(state, questions, ())]) — what `probabilities()` / `system_one()` compute: a request of
      one question is its row (`_logz_ids([row])`, a plain causal pass), a request of several is one Tree (`_request`:
      trunk = the prefix's ids, a branch per question)
  (b) the row form, per question: engine._logz_ids([row_ids])[0] -> the log-probabilities of the question's group ids,
      prompt.readout's probabilities, the argmax key, the top-2 margin, api.answer
  (c) per record, max |dp| between (a) and (b)
  (d) `--hidden <ids>`: a forward hook on model.model.language_model.embedding_norm (the final norm) keeps each row's
      hidden states [T, 2048] fp32 (npz) and the slot's row; asserted on them: the model's own lm_head on the slot row,
      minus its log-sum-exp, equals (b)'s log-probabilities bit for bit (the hook point), and the host's float64 gather
      h . E[id] over the tied rows lands within 1e-3 of the fp32 logits (recorded)
  (e) record 0 again after the loop: bit-equal log-probabilities
  (f) results/oracle_summary.json: per source, row lengths, near-ties (top-2 <= 0.02), gold agreement on our fixture
      subset (not the provider's benchmark numbers), Tree against row per record, seconds, and the load report
      (`load_report`: transformers' missing / unexpected keys, and every language tensor of the checkpoint equal to
      the loaded parameter)

A question the provider refuses (no `instructions`: `own_email_03` / `next_step`) is recorded with the error; the
record's other questions still run their rows, and its API path is the error the provider raises.

`--images` (round 5b): the picture records of `fixtures/image_records.json`, each picture opened with
`PIL.Image.open(path)` (a URL is not fetched: the record is listed as skipped), through the provider's image path:
  (a) the API path: engine.run([(state, questions, images)]) — `_request`: one question = the whole prompt in one plain
      pass (`_image_inputs(prefix + suffix)` -> `_one_pass(**inputs, logits_to_keep=1)`), several = one Tree (trunk =
      the processor's ids of the prefix with the pictures, `model.answer(trunk, packed, lengths, **vision)` =
      `get_image_features(**vision)` scattered into the `<image>` slots); its probabilities and input_tokens
  (b) the row form, per question: `_request`'s one-question branch line for line (cap_pixels, prefix_text with
      `_image_markup`, suffix_text, `_image_inputs`, `_one_pass`, log-softmax, readout, answer): the processor's ids,
      the pixel inputs' shapes, the hook-point assert of (d), the image features (a forward hook on the projector)
  (c) per record, max |dp| between (a) and (b) (one question: bit-equal; several: the Tree against the plain passes)
  (d) `--hidden`: per question the final norm's hidden states [T, 2048] fp32, the slot's row, the ids and the image
      features [n_image_tokens, 2048] fp32 (npz under <out-dir>/images_hidden/)
  (e) record 0 again after the loop: bit-equal (rows and the API path)
  -> <out-dir>/records_oracle_images.json and `--summary` (per path, near-ties, gold on our subset, Tree vs plain,
     seconds per record).

    HF_HUB_OFFLINE=1 python oracle_d1.py --threads 1                        # round 2: the model on the CPU, fp32
    HF_HUB_OFFLINE=1 python oracle_d1.py --threads 1 --hidden tv4_000,card_refund,long_34k
    HF_HUB_OFFLINE=1 python oracle_d1.py --images --threads 1 --hidden img01_shapes_384x384,img06_grid_1024x768,\\
        img12_small_300x300 --summary $ZOO_WORK_ROOT/_d1_3b/results/<json>                   # round 5b: the pictures

`--dry-run` (round 1) loads no model: the provider's API path runs on a stand-in backbone that records the ids it is
handed and returns seeded random fp32 logits over the 128,000 ids, so SystemOne.run / _request / _logz_ids / _plan,
prompt.render / aliases / readout_ids / readout and api.answer all run as shipped. `test_host.py` calls `dry_run()`
and checks host.py against it (ids, slot, groups, keys, the Tree's trunk and branches, the probabilities and answers).

-> $ZOO_WORK_ROOT/_d1_3b/oracle/records_oracle.json, oracle/hidden/<id>.npz, results/oracle_summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
IMAGE_FIXTURES = LANE / "fixtures" / "image_records.json"
IMAGE_ID = 124907                    # <image>
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
PROVIDER_SRC = LANE / "src"          # K/src/d1 = the six .py files of the snapshot, verbatim (+ __init__.py)
VOCAB = 128000
NEAR_TIE = 0.02
STANDIN_SCALE = 3.0


def snapshot() -> Path:
    return Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))


PROVIDER_FILES = ("api.py", "hybrid.py", "lfm2_vl.py", "modeling_d1.py", "prompt.py", "runner.py")


def provider(eng=None):
    """The provider's prompt / api modules. With an engine: its own package's (`--loader auto` runs the remote-code copy
    under ~/.cache/huggingface/modules, whose `render` accepts only its own question classes); without one, K/src/d1's
    (the package needs no model to import prompt / api)."""
    if eng is not None:
        import importlib

        pkg = type(eng).__module__.rsplit(".", 1)[0]
        return importlib.import_module(f"{pkg}.prompt"), importlib.import_module(f"{pkg}.api")
    if str(PROVIDER_SRC) not in sys.path:
        sys.path.insert(0, str(PROVIDER_SRC))
    import d1.api as api
    import d1.prompt as prompt
    return prompt, api


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- the stand-in backbone (--dry-run)
def standin_logits(ids: list[int], seed: int = 0):
    """Seeded random fp32 logits over the vocabulary, a function of the row's ids (so a row reads the same logits
    through the Tree and alone)."""
    import torch

    h = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little") ^ seed)
    return torch.randn(VOCAB, generator=g) * STANDIN_SCALE


class _Out:
    def __init__(self, logits):
        self.logits = logits


class StandIn:
    """The two backbone entry points SystemOne calls (`_one_pass(input_ids=..., logits_to_keep=1)` and
    `model.answer(trunk, packed, lengths)`), recording the ids they are handed."""

    def __init__(self, seed: int = 0):
        self.seed, self.calls = seed, []
        self.config = SimpleNamespace(model_type="lfm2_vl", _name_or_path=str(snapshot()))

    def one_pass(self, input_ids, logits_to_keep=1, **_):
        import torch

        assert input_ids.shape[0] == 1 and logits_to_keep == 1
        row = [int(x) for x in input_ids[0].tolist()]
        self.calls.append({"kind": "row", "ids": row})
        return _Out(standin_logits(row, self.seed).to(torch.float32)[None, None, :])

    def answer(self, trunk, questions, lengths, **vision):
        import torch

        assert not vision
        trunk_ids = [int(x) for x in trunk[0].tolist()]
        packed, out, branches, i = [int(x) for x in questions.tolist()], [], [], 0
        for n in [int(x) for x in lengths.tolist()]:
            branches.append(packed[i:i + n])
            i += n
        self.calls.append({"kind": "tree", "trunk": trunk_ids, "branches": branches})
        for b in branches:
            out.append(standin_logits(trunk_ids + b, self.seed))
        return torch.stack(out)


def standin_engine(tokenizer, backbone: StandIn):
    """runner.SystemOne without a model: the attributes its __init__ sets, the stand-in's entry points."""
    import torch

    if str(PROVIDER_SRC) not in sys.path:
        sys.path.insert(0, str(PROVIDER_SRC))
    from d1.prompt import DEFAULT_STATE_STYLE, DEFAULT_SYSTEM, default_lead
    from d1.runner import SystemOne

    eng = SystemOne.__new__(SystemOne)
    eng.model, eng.tokenizer, eng.model_id = backbone, tokenizer, backbone.config._name_or_path
    eng.lead = default_lead(backbone.config.model_type)
    eng.device = torch.device("cpu")
    bos = getattr(tokenizer, "bos_token", None)
    eng.bos = bos if isinstance(bos, str) else ""
    eng.calibration, eng.state_style, eng.system, eng.option_style = None, DEFAULT_STATE_STYLE, DEFAULT_SYSTEM, "desc"
    eng.token_budget, eng.processor = 65536, None
    eng._one_pass = backbone.one_pass
    return eng


# --------------------------------------------------------------------------- one record through the provider's code
def _keys(api, prompt, q) -> list[str]:
    """The keys of the provider's answer: choice / score from api.answer's probabilities; noul [yes, no] (readout_ids'
    order; api.answer reports probs[0] as P(yes))."""
    if isinstance(q, prompt.Noul):
        probe = api.answer(q, [0.75, 0.25])
        assert probe == {"type": "noul", "noul": 0.75}
        return list(host.NOUL_KEYS)
    n = len(q.criteria)
    return list(api.answer(q, [1.0 / n] * n)["probabilities"])


def provider_record(eng, r: dict, logz_fn) -> dict:
    """A fixture record through the provider's code: per question the row (render -> encode -> logz_fn -> readout ->
    answer), and the API path (engine.run) with its probabilities and input_tokens. `logz_fn(row_ids)` returns the
    row's log-probabilities over the vocabulary (the model, or the stand-in through `_logz_ids`)."""
    prompt, api = provider(eng)
    tok = eng.tokenizer
    state, qdict = r["request"]["state"], r["request"]["questions"]
    out: dict = {"id": r["id"], "source": r["source"], "questions": [], "refused": []}
    qs = []
    for name, qd in qdict.items():
        try:
            qs.append((name, prompt.as_question(qd)))
        except Exception as e:  # noqa: BLE001 — the provider's own refusal
            out["refused"].append({"name": name, "error": f"{type(e).__name__}: {e}"})
            qs.append((name, None))
    for name, q in qs:
        if q is None:
            continue
        text = eng.render(state, q)
        row = tok.encode(text, add_special_tokens=False)
        groups = prompt.readout_ids(tok, q)
        codes = ([c for c, _ in prompt.aliases(tok, list(q.criteria.keys()))]
                 if isinstance(q, prompt.Choice) else None)
        t0 = time.perf_counter()
        logz = logz_fn(row)
        secs = time.perf_counter() - t0
        probs = prompt.readout(tok, q, logz)
        keys = _keys(api, prompt, q)
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        margin = probs[order[0]] - probs[order[1]] if len(probs) > 1 else 1.0
        gold = host.gold_key({"type": q.type}, (r.get("gold") or {}).get(name))
        argmax = keys[max(range(len(probs)), key=probs.__getitem__)]
        out["questions"].append({
            "name": name, "type": q.type, "text": text, "row_ids": row, "row_len": len(row), "slot": len(row) - 1,
            "codes": codes, "groups": groups, "keys": keys,
            "group_logz": {str(i): float(logz[i]) for g in groups for i in g},
            "probs": probs, "argmax": argmax, "top2_margin": margin, "near_tie": margin <= NEAR_TIE,
            "gold": gold, "correct": None if gold is None else argmax == gold,
            "answer": api.answer(q, probs), "seconds": round(secs, 4)})
    if out["refused"]:
        try:
            eng.run([(state, [prompt.as_question(qd) for qd in qdict.values()], ())])
            out["api"] = {"error": None}
        except Exception as e:  # noqa: BLE001
            out["api"] = {"error": f"{type(e).__name__}: {e}"}
        return out
    t0 = time.perf_counter()
    [(probs_api, read)] = eng.run([(state, [q for _, q in qs], ())])
    secs = time.perf_counter() - t0
    resp = {"answers": {n: api.answer(q, p) for (n, q), p in zip(qs, probs_api)},
            "usage": {"input_tokens": read, "output_tokens": 0}}
    dp = max(abs(a - b) for p, e in zip(probs_api, out["questions"]) for a, b in zip(p, e["probs"]))
    out["api"] = {"path": "tree" if len(qs) > 1 else "row", "probs": probs_api, "input_tokens": read, "response": resp,
                  "max_abs_dp_vs_rows": dp, "seconds": round(secs, 4)}
    return out


def dry_run(records: list[dict], tokenizer, seed: int = 0) -> list[dict]:
    """Every record through the provider's code on the stand-in backbone. Per record: provider_record's entry plus the
    ids the API path handed the backbone (`calls`: a row, or the Tree's trunk and branches)."""
    out = []
    for r in records:
        backbone = StandIn(seed)
        eng = standin_engine(tokenizer, backbone)

        def logz_fn(row, eng=eng, backbone=backbone):
            n0 = len(backbone.calls)
            z = eng._logz_ids([row])[0]
            del backbone.calls[n0:]           # the row form's own calls are not the API path's
            return z

        e = provider_record(eng, r, logz_fn)
        e["calls"] = backbone.calls
        out.append(e)
    return out


# --------------------------------------------------------------------------- round 2: the model
class FinalNormHook:
    """Forward hook on the final norm: the last call's output [T, d] (fp32)."""

    def __init__(self, module):
        self.out = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, output):
        self.out = output[0].detach().clone()


def _key_part(k: str) -> str:
    return ("language" if "language_model." in k else "vision" if "vision" in k else
            "projector" if "projector" in k else "other")


def load_report(model, loading_info: dict, snap: Path) -> dict:
    """What transformers reports for the load (missing / unexpected / mismatched keys, by part), and every
    `model.language_model.*` tensor of the checkpoint against the loaded parameter of that name: bf16 widened to fp32
    is exact, so each must be torch.equal."""
    import torch
    from safetensors import safe_open

    def parts(keys) -> dict:
        out: dict = {}
        for k in keys:
            out.setdefault(_key_part(k), []).append(k)
        return {p: sorted(v) for p, v in sorted(out.items())}

    missing = sorted(loading_info.get("missing_keys") or [])
    unexpected = sorted(loading_info.get("unexpected_keys") or [])
    mismatched = sorted([str(x) for x in loading_info.get("mismatched_keys") or []])
    params = dict(model.named_parameters())
    ck = snap / "model.safetensors"
    lang, absent, differ = [], [], []
    with safe_open(str(ck), framework="pt", device="cpu") as f:
        keys = list(f.keys())  # noqa: SIM118
        for k in keys:
            if not k.startswith("model.language_model."):
                continue
            lang.append(k)
            p = params.get(k)
            if p is None:
                absent.append(k)
                continue
            t = f.get_tensor(k)
            if not (p.dtype == torch.float32 and tuple(p.shape) == tuple(t.shape) and torch.equal(p.detach(), t.float())):
                differ.append(k)
    counts: dict = {}
    for k in keys:
        counts[_key_part(k)] = counts.get(_key_part(k), 0) + 1
    return {"transformers": {"missing_keys": parts(missing), "unexpected_keys": parts(unexpected),
                             "mismatched_keys": mismatched, "error_msgs": [str(e) for e in loading_info.get("error_msgs") or []],
                             "n_missing": len(missing), "n_unexpected": len(unexpected), "n_mismatched": len(mismatched)},
            "checkpoint": {"file": str(ck), "keys": len(keys), "keys_by_part": dict(sorted(counts.items()))},
            "language_keys": len(lang), "language_keys_missing_in_model": absent,
            "language_keys_not_bit_equal": differ,
            "language_bit_equal": not absent and not differ,
            "check": "every model.language_model.* checkpoint tensor (bf16) widened to fp32 == the loaded parameter "
                     "of the same name (torch.equal)"}


def load_model(loader: str, threads: int):
    import torch

    if threads:
        torch.set_num_threads(threads)
    snap = snapshot()
    t0 = time.perf_counter()
    if loader == "package":     # the verbatim copies in K/src/d1, no remote-code cache
        if str(PROVIDER_SRC) not in sys.path:
            sys.path.insert(0, str(PROVIDER_SRC))
        from d1.modeling_d1 import D1Model
        model, linfo = D1Model.from_pretrained(str(snap), dtype=torch.float32, output_loading_info=True)
    else:                       # the card's way (copies the remote code under ~/.cache/huggingface/modules)
        from transformers import AutoModel
        model, linfo = AutoModel.from_pretrained(str(snap), trust_remote_code=True, dtype=torch.float32,
                                                 output_loading_info=True)
    model.eval()
    load_s = time.perf_counter() - t0
    engine = model.engine
    info = {"loader": loader, "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "engine": f"{type(engine).__module__}.{type(engine).__qualname__}", "dtype": str(next(model.parameters()).dtype),
            "device": str(engine.device), "bos": engine.bos, "lead": engine.lead, "state_style": engine.state_style,
            "system": engine.system, "option_style": engine.option_style, "torch_threads": torch.get_num_threads(),
            "load_seconds": round(load_s, 1),
            "tied": bool(model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()),
            "load_report": load_report(model, linfo, snap)}
    pkg_dir = Path(sys.modules[type(engine).__module__].__file__).parent
    code = {f: sha256_file(pkg_dir / f) for f in PROVIDER_FILES}
    info["provider_code"] = {"package": type(engine).__module__.rsplit(".", 1)[0], "dir": str(pkg_dir), "sha256": code,
                             "equal_to_snapshot": all(code[f] == sha256_file(snap / f) for f in PROVIDER_FILES)}
    return model, engine, info


def run_model(args) -> int:
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fx = Path(args.fixtures)
    records = json.loads(fx.read_text())["records"]
    if args.records:
        want = set(args.records.split(","))
        records = [r for r in records if r["id"] in want]
    hidden_ids = set(args.hidden.split(",")) if args.hidden else set()
    out_dir, res_dir = Path(args.out_dir), Path(args.results_dir)
    (out_dir / "hidden").mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    model, engine, info = load_model(args.loader, args.threads)
    hook = FinalNormHook(model.model.language_model.embedding_norm)
    E = model.lm_head.weight.detach()
    entries, hidden_index = [], {}
    t_all = time.perf_counter()

    for n, r in enumerate(records):
        keep = r["id"] in hidden_ids
        rows_hidden = []

        def logz_fn(row, keep=keep, rows_hidden=rows_hidden):
            z = engine._logz_ids([row])[0]
            h = hook.out                                     # [T, d] of this row's pass
            assert h is not None and h.shape[0] == len(row), "the hook did not see this row"
            raw = model.lm_head(h[-1][None, None, :]).float()[0, -1]
            assert torch.equal(raw - torch.logsumexp(raw, dim=-1), z), "lm_head(hook hidden) != _logz_ids"
            if keep:
                rows_hidden.append(h.float().numpy())
            return z

        with torch.inference_mode():
            e = provider_record(engine, r, logz_fn)
        if keep:
            arrays = {}
            for k, (q, h) in enumerate(zip(e["questions"], rows_hidden)):
                ids = host.group_ids(q["groups"])
                z64 = host.option_logits(h[-1], E[ids].float().numpy(), ids)
                with torch.no_grad():
                    z32 = model.lm_head(torch.from_numpy(h[-1])[None, None, :]).float()[0, -1]
                q["host_gather_max_abs_dlogit"] = max(abs(z64[i] - float(z32[i])) for i in ids)
                assert q["host_gather_max_abs_dlogit"] <= 1e-3, q["host_gather_max_abs_dlogit"]
                arrays[f"q{k}_hidden"] = h.astype(np.float32)
                arrays[f"q{k}_ids"] = np.asarray(q["row_ids"], np.int32)
                arrays[f"q{k}_slot_hidden"] = h[-1].astype(np.float32)
            tmp = out_dir / "hidden" / f"{r['id']}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, out_dir / "hidden" / f"{r['id']}.npz")
            hidden_index[r["id"]] = {k: list(v.shape) for k, v in arrays.items() if k.endswith("_hidden")}
        entries.append(e)
        if n % 20 == 0:
            print(f"[{n + 1}/{len(records)}] {r['id']} rows={[q['row_len'] for q in e['questions']]} "
                  f"api={e['api'].get('path')} dp={e['api'].get('max_abs_dp_vs_rows')}", flush=True)
    loop_s = time.perf_counter() - t_all

    with torch.inference_mode():   # (e) record 0 again
        again = provider_record(engine, records[0], lambda row: engine._logz_ids([row])[0])
    det = all(a["group_logz"] == b["group_logz"] for a, b in zip(again["questions"], entries[0]["questions"]))
    assert det, "record 0 re-run differs"
    header = {"schema": "d1-oracle/1", "model": {**MODEL, **info}, "fixtures": {"path": str(fx), "sha256": sha256_file(fx)},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           **{m: __import__(m).__version__ for m in ("transformers", "tokenizers", "numpy")}},
              "platform": platform.platform(), "loop_seconds": round(loop_s, 1),
              "determinism": {"record": records[0]["id"], "group_logz_bit_equal": det},
              "hidden_records": {"dir": str(out_dir / "hidden"), "shapes": hidden_index},
              "checks": ("per question: lm_head(final-norm hook row at the slot) - logsumexp == _logz_ids (torch.equal); "
                         "--hidden rows: host float64 gather within 1e-3 of the fp32 logits")}
    doc = {**header, "records": entries}
    write_atomic(out_dir / "records_oracle.json", (json.dumps(doc) + "\n").encode())
    if not args.records:
        summ = summarize(entries)
        summ.update({"oracle_sha256": sha256_file(out_dir / "records_oracle.json"), "determinism": header["determinism"],
                     "load_seconds": info["load_seconds"], "loop_seconds": header["loop_seconds"],
                     "threads": info["torch_threads"], "load_report": info["load_report"]})
        write_atomic(res_dir / "oracle_summary.json", (json.dumps(summ, indent=1) + "\n").encode())
        print(json.dumps({k: v for k, v in summ.items() if k != "by_source"}, indent=1))
    return 0


# --------------------------------------------------------------------------- round 5b: the pictures
class ProjectorHook:
    """Forward hook on the multimodal projector: each crop's image features [h w / 4, d] since the last take(), in
    call order (get_image_features projects the crops one by one, in crop order)."""

    def __init__(self, module):
        self.rows = []
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, output):
        self.rows.append(output.detach().reshape(-1, output.shape[-1]).float().clone())

    def take(self) -> list:
        out, self.rows = self.rows, []
        return out


def provider_image_record(eng, model, r: dict, images: list, norm_hook, proj_hook, keep: bool) -> tuple[dict, dict]:
    """A picture record through the provider's code: per question the row form (`_request`'s one-question branch line
    for line), then the API path engine.run([(state, questions, images)]). -> (entry, the --hidden arrays)."""
    import torch

    prompt, api = provider(eng)
    runner = sys.modules[type(eng).__module__]
    tok = eng.tokenizer
    state, qdict = r["request"]["state"], r["request"]["questions"]
    out: dict = {"id": r["id"], "source": r["source"], "images": r["images"], "questions": [], "refused": []}
    qs = []
    for name, qd in qdict.items():
        try:
            qs.append((name, prompt.as_question(qd)))
        except Exception as e:  # noqa: BLE001 — the provider's own refusal
            out["refused"].append({"name": name, "error": f"{type(e).__name__}: {e}"})
            qs.append((name, None))
    pics = [runner.cap_pixels(im) for im in images]
    out["pictures"] = [{"size": list(im.size), "mode": im.mode, "capped": list(p.size)} for im, p in zip(images, pics)]
    prefix = prompt.prefix_text(tok, state, eng.bos, eng.state_style, eng.system, eng._image_markup(len(pics)))
    arrays: dict = {}
    feats_first = None
    E = model.lm_head.weight.detach()
    for k, (name, q) in enumerate(qs):
        if q is None:
            continue
        suffix = prompt.suffix_text(tok, q, eng.lead, eng.option_style)
        inputs = eng._image_inputs(prefix + suffix, pics)
        row = [int(x) for x in inputs["input_ids"][0].tolist()]
        proj_hook.take()
        t0 = time.perf_counter()
        logits = eng._one_pass(**inputs, logits_to_keep=1).logits[0, -1].float()
        logz = logits - torch.logsumexp(logits, dim=-1)
        secs = time.perf_counter() - t0
        h = norm_hook.out                                    # [T, d] of this pass
        assert h is not None and h.shape[0] == len(row), "the hook did not see this row"
        raw = model.lm_head(h[-1][None, None, :]).float()[0, -1]
        assert torch.equal(raw - torch.logsumexp(raw, dim=-1), logz), "lm_head(hook hidden) != the pass's log-softmax"
        feats = torch.cat(proj_hook.take()).numpy().astype(np.float32)
        slots = [i for i, t in enumerate(row) if t == IMAGE_ID]
        if feats.shape[0] != len(slots):
            raise SystemExit(f"{r['id']}/{name}: {feats.shape[0]} image features for {len(slots)} <image> slots")
        if feats_first is None:
            feats_first = feats
        groups = prompt.readout_ids(tok, q)
        codes = ([c for c, _ in prompt.aliases(tok, list(q.criteria.keys()))]
                 if isinstance(q, prompt.Choice) else None)
        probs = prompt.readout(tok, q, logz)
        keys = _keys(api, prompt, q)
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        margin = probs[order[0]] - probs[order[1]] if len(probs) > 1 else 1.0
        gold = host.gold_key({"type": q.type}, (r.get("gold") or {}).get(name))
        argmax = keys[max(range(len(probs)), key=probs.__getitem__)]
        entry = {
            "name": name, "type": q.type, "text": prefix + suffix, "row_ids": row, "row_len": len(row),
            "slot": len(row) - 1, "codes": codes, "groups": groups, "keys": keys,
            "group_logz": {str(i): float(logz[i]) for g in groups for i in g},
            "probs": probs, "argmax": argmax, "top2_margin": margin, "near_tie": margin <= NEAR_TIE,
            "gold": gold, "correct": None if gold is None else argmax == gold, "answer": api.answer(q, probs),
            "seconds": round(secs, 4), "n_image_tokens": len(slots), "image_positions": [slots[0], slots[-1]],
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "spatial_shapes": [list(map(int, x)) for x in inputs["spatial_shapes"].tolist()],
            "image_features_sha256": hashlib.sha256(feats.tobytes()).hexdigest(),
            "image_features_absmax": float(np.abs(feats).max()),
            "image_features_equal_first_question": bool(np.array_equal(feats, feats_first))}
        if keep:
            ids = host.group_ids(groups)
            hn = h.float().numpy()
            z64 = host.option_logits(hn[-1], E[ids].float().numpy(), ids)
            with torch.no_grad():
                z32 = model.lm_head(h[-1][None, None, :]).float()[0, -1]
            entry["host_gather_max_abs_dlogit"] = max(abs(z64[i] - float(z32[i])) for i in ids)
            assert entry["host_gather_max_abs_dlogit"] <= 1e-3, entry["host_gather_max_abs_dlogit"]
            arrays[f"q{k}_hidden"] = hn.astype(np.float32)
            arrays[f"q{k}_ids"] = np.asarray(row, np.int32)
            arrays[f"q{k}_slot_hidden"] = hn[-1].astype(np.float32)
            if "image_features" not in arrays:
                arrays["image_features"] = feats
        out["questions"].append(entry)
    if out["refused"]:
        try:
            eng.run([(state, [prompt.as_question(qd) for qd in qdict.values()], images)])
            out["api"] = {"error": None}
        except Exception as e:  # noqa: BLE001
            out["api"] = {"error": f"{type(e).__name__}: {e}"}
        return out, arrays
    t0 = time.perf_counter()
    [(probs_api, read)] = eng.run([(state, [q for _, q in qs], images)])
    secs = time.perf_counter() - t0
    resp = {"answers": {n: api.answer(q, p) for (n, q), p in zip(qs, probs_api)},
            "usage": {"input_tokens": read, "output_tokens": 0}}
    dp = max(abs(a - b) for p, e in zip(probs_api, out["questions"]) for a, b in zip(p, e["probs"]))
    api_rec = {"path": "tree" if len(qs) > 1 else "plain", "probs": probs_api, "input_tokens": read, "response": resp,
               "max_abs_dp_vs_rows": dp,
               "bit_equal_rows": all(list(p) == list(e["probs"]) for p, e in zip(probs_api, out["questions"])),
               "seconds": round(secs, 4)}
    if len(qs) > 1:   # the Tree's trunk and branches as `_request` makes them
        trunk = [int(x) for x in eng._image_inputs(prefix, pics)["input_ids"][0].tolist()]
        branches = [tok.encode(prompt.suffix_text(tok, q, eng.lead, eng.option_style), add_special_tokens=False)
                    for _, q in qs]
        api_rec.update({"trunk_len": len(trunk), "branch_lens": [len(b) for b in branches],
                        "trunk_plus_branch_equals_row": [trunk + list(b) == e["row_ids"]
                                                         for b, e in zip(branches, out["questions"])],
                        "input_tokens_equal_trunk_plus_branches": read == len(trunk) + sum(map(len, branches))})
    else:
        api_rec["input_tokens_equal_row"] = read == out["questions"][0]["row_len"]
    out["api"] = api_rec
    return out, arrays


def run_images(args) -> int:
    import torch
    from PIL import Image

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    fx = Path(args.fixtures)
    records = json.loads(fx.read_text())["records"]
    if args.records:
        want = set(args.records.split(","))
        records = [r for r in records if r["id"] in want]
    hidden_ids = set(args.hidden.split(",")) if args.hidden else set()
    out_dir = Path(args.out_dir)
    hidden_dir = out_dir / "images_hidden"
    oracle_path = out_dir / "records_oracle_images.json"
    summary_path = Path(args.summary) if args.summary else Path(args.results_dir) / "oracle_images_summary.json"
    for p in (oracle_path, summary_path):
        if p.exists():
            raise SystemExit(f"{p} exists: the oracle is never overwritten")
    hidden_dir.mkdir(parents=True, exist_ok=True)
    model, engine, info = load_model(args.loader, args.threads)
    norm_hook = FinalNormHook(model.model.language_model.embedding_norm)
    proj_hook = ProjectorHook(model.model.multi_modal_projector)
    entries, skipped, hidden_index, first = [], [], {}, None
    t_all = time.perf_counter()
    for r in records:
        if any("://" in x for x in r["images"]):
            skipped.append({"id": r["id"], "images": r["images"],
                            "why": "a URL, not fetched: material that needs attribution is not used, even to measure"})
            continue
        paths = [fx.parent / x for x in r["images"]]
        images = [Image.open(p) for p in paths]
        t0 = time.perf_counter()
        with torch.inference_mode():
            e, arrays = provider_image_record(engine, model, r, images, norm_hook, proj_hook, r["id"] in hidden_ids)
        e["seconds_record"] = round(time.perf_counter() - t0, 2)
        e["image_files"] = [{"path": str(p), "sha256": sha256_file(p)} for p in paths]
        if arrays:
            tmp = hidden_dir / f"{r['id']}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, hidden_dir / f"{r['id']}.npz")
            hidden_index[r["id"]] = {k: list(v.shape) for k, v in arrays.items() if k.endswith("hidden") or k == "image_features"}
        if first is None:
            first = (r, paths, e)
        entries.append(e)
        print(f"{r['id']}: rows {[q['row_len'] for q in e['questions']]} images {[q['n_image_tokens'] for q in e['questions']]}"
              f" api={e['api'].get('path')} dp={e['api'].get('max_abs_dp_vs_rows')} {e['seconds_record']} s", flush=True)
    loop_s = time.perf_counter() - t_all
    r0, paths0, e0 = first
    with torch.inference_mode():   # (e) record 0 again
        again, _ = provider_image_record(engine, model, r0, [Image.open(p) for p in paths0], norm_hook, proj_hook, False)
    det = (all(a["group_logz"] == b["group_logz"] for a, b in zip(again["questions"], e0["questions"]))
           and again["api"]["probs"] == e0["api"]["probs"])
    assert det, "record 0 re-run differs"
    header = {"schema": "d1-oracle-images/1", "model": {**MODEL, **info}, "fixtures": {"path": str(fx), "sha256": sha256_file(fx)},
              "images_opened": "PIL.Image.open(path) (the provider's cap_pixels converts to RGB); a URL is not fetched",
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           **{m: __import__(m).__version__ for m in ("transformers", "tokenizers", "numpy", "PIL")}},
              "platform": platform.platform(), "loop_seconds": round(loop_s, 1),
              "determinism": {"record": r0["id"], "rows_group_logz_and_api_probs_bit_equal": det},
              "hidden_records": {"dir": str(hidden_dir), "shapes": hidden_index}, "skipped": skipped,
              "checks": ("per question: lm_head(final-norm hook row at the slot) - logsumexp == the plain pass's log-softmax "
                         "(torch.equal); the projector hook's features = one row per <image> slot; --hidden rows: host "
                         "float64 gather within 1e-3 of the fp32 logits")}
    doc = {**header, "records": entries}
    write_atomic(oracle_path, (json.dumps(doc) + "\n").encode())
    summ = summarize_images(entries)
    summ.update({"oracle": str(oracle_path), "oracle_sha256": sha256_file(oracle_path), "skipped": skipped,
                 "determinism": header["determinism"], "load_seconds": info["load_seconds"], "loop_seconds": header["loop_seconds"],
                 "threads": info["torch_threads"], "load_report": info["load_report"], "provider_code": info["provider_code"]})
    write_atomic(summary_path, (json.dumps(summ, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in summ.items() if k not in ("load_report", "provider_code", "by_record")}, indent=1))
    return 0


def summarize_images(entries: list[dict]) -> dict:
    qs = [(e, q) for e in entries for q in e["questions"]]
    by_path: dict = {}
    for e in entries:
        p = e["api"].get("path")
        g = by_path.setdefault(p, {"records": 0, "questions": 0})
        g["records"] += 1
        g["questions"] += len(e["questions"])
    gold = [q for _, q in qs if q["gold"] is not None]
    tree = {e["id"]: e["api"]["max_abs_dp_vs_rows"] for e in entries if e["api"].get("path") == "tree"}
    secs = [e["seconds_record"] for e in entries]
    return {"records": len(entries), "questions": len(qs), "by_path": by_path,
            "types": {t: sum(1 for _, q in qs if q["type"] == t) for t in ("choice", "noul", "score")},
            "near_ties": {"threshold_top2": NEAR_TIE, "n": sum(q["near_tie"] for _, q in qs),
                          "ids": [f"{e['id']}/{q['name']}" for e, q in qs if q["near_tie"]],
                          "smallest_top2": min(q["top2_margin"] for _, q in qs)},
            "gold_agreement_our_subset": f"{sum(bool(q['correct']) for q in gold)}/{len(gold)}",
            "gold_note": "gold on our own drawn pictures only, not the provider's benchmark numbers",
            "tree_vs_plain_max_abs_dp": max(tree.values()) if tree else None, "tree_vs_plain_by_record": tree,
            "plain_api_bit_equal_single_question": sum(1 for e in entries if e["api"].get("path") == "plain"
                                                       and e["api"]["bit_equal_rows"]),
            "plain_records_single_question": sum(1 for e in entries if e["api"].get("path") == "plain"),
            "trunk_plus_branch_equals_row": sum(all(e["api"]["trunk_plus_branch_equals_row"]) for e in entries
                                                if e["api"].get("path") == "tree"),
            "image_features_equal_across_questions": all(q["image_features_equal_first_question"] for _, q in qs),
            "row_len": {"min": min(q["row_len"] for _, q in qs), "max": max(q["row_len"] for _, q in qs)},
            "n_image_tokens": {e["id"]: e["questions"][0]["n_image_tokens"] for e in entries},
            "seconds_per_record": {"p50": round(pct(secs, 0.5), 2), "max": round(max(secs), 2), "total": round(sum(secs), 1),
                                   "what": "every question's plain pass + the API pass (+ the --hidden copies)"},
            "seconds_per_plain_pass": {"p50": round(pct([q["seconds"] for _, q in qs], 0.5), 2),
                                       "max": round(max(q["seconds"] for _, q in qs), 2)},
            "by_record": [{"id": e["id"], "path": e["api"].get("path"), "questions": len(e["questions"]),
                           "input_tokens": e["api"].get("input_tokens"), "row_len": [q["row_len"] for q in e["questions"]],
                           "n_image_tokens": e["questions"][0]["n_image_tokens"],
                           "argmax": {q["name"]: q["argmax"] for q in e["questions"]},
                           "gold": {q["name"]: q["gold"] for q in e["questions"]},
                           "top2": {q["name"]: q["top2_margin"] for q in e["questions"]},
                           "max_abs_dp_tree_vs_plain": e["api"]["max_abs_dp_vs_rows"] if e["api"].get("path") == "tree" else None,
                           "seconds": e["seconds_record"]} for e in entries]}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarize(entries: list[dict]) -> dict:
    by: dict = {}
    for e in entries:
        g = by.setdefault(e["source"], {"records": 0, "questions": 0, "types": {}, "gold": 0, "correct": 0,
                                        "near_ties": [], "refused": 0, "row_len": []})
        g["records"] += 1
        g["refused"] += len(e["refused"])
        for q in e["questions"]:
            g["questions"] += 1
            g["types"][q["type"]] = g["types"].get(q["type"], 0) + 1
            g["row_len"].append(q["row_len"])
            if q["gold"] is not None:
                g["gold"] += 1
                g["correct"] += bool(q["correct"])
            if q["near_tie"]:
                g["near_ties"].append(f"{e['id']}/{q['name']}")
    for g in by.values():
        rl = g.pop("row_len")
        g["row_len"] = {"p50": pct(rl, 0.5), "p99": pct(rl, 0.99), "max": max(rl)}
        g["gold_agreement_our_subset"] = f"{g['correct']}/{g['gold']}" if g["gold"] else None
    rows = [q["row_len"] for e in entries for q in e["questions"]]
    secs = [q["seconds"] for e in entries for q in e["questions"]]
    tree = [e["api"]["max_abs_dp_vs_rows"] for e in entries if e["api"].get("path") == "tree"]
    by_rec = {e["id"]: e["api"]["max_abs_dp_vs_rows"] for e in entries if e["api"].get("path") == "tree"}
    rec_secs = [sum(q["seconds"] for q in e["questions"]) + float(e["api"].get("seconds") or 0.0) for e in entries]
    near = [f"{e['id']}/{q['name']}" for e in entries for q in e["questions"] if q["near_tie"]]
    return {"records": len(entries), "questions": len(rows), "by_source": by,
            "row_len": {"min": min(rows), "p50": pct(rows, 0.5), "p99": pct(rows, 0.99), "max": max(rows)},
            "near_ties": {"threshold_top2": NEAR_TIE, "n": len(near), "ids": near},
            "tree_vs_row_max_abs_dp": max(tree) if tree else None, "tree_records": len(tree),
            "tree_vs_row_by_record": by_rec,
            "tree_vs_row_distribution": ({"min": min(tree), "p50": pct(tree, 0.5), "p90": pct(tree, 0.9), "max": max(tree)}
                                         if tree else None),
            "row_api_bit_equal_single_question": sum(1 for e in entries if e["api"].get("path") == "row"
                                                     and e["api"]["max_abs_dp_vs_rows"] == 0.0),
            "row_api_records_single_question": sum(1 for e in entries if e["api"].get("path") == "row"),
            "refused": [f"{e['id']}/{x['name']}: {x['error']}" for e in entries for x in e["refused"]],
            "gold_note": "gold agreement on our fixture subset only, not the provider's benchmark numbers",
            "seconds_per_row": {"p50": round(pct(secs, 0.5), 3), "max": round(max(secs), 3), "total": round(sum(secs), 1)},
            "seconds_per_record": {"what": "the record's row-form passes plus its API pass",
                                   "p50": round(pct(rec_secs, 0.5), 3), "max": round(max(rec_secs), 3),
                                   "total": round(sum(rec_secs), 1)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=str(LANE / "fixtures" / "records.json"))
    ap.add_argument("--out-dir", default=str(LANE / "oracle"))
    ap.add_argument("--results-dir", default=str(LANE / "results"))
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch's default)")
    ap.add_argument("--records", default=None, help="comma-separated record ids (no summary written)")
    ap.add_argument("--hidden", default=None, help="comma-separated record ids whose hidden rows are kept (npz)")
    ap.add_argument("--loader", choices=["auto", "package"], default="auto",
                    help="auto = AutoModel + trust_remote_code (the card); package = K/src/d1's verbatim copies")
    ap.add_argument("--dry-run", action="store_true", help="no model: the stand-in backbone (round 1; test_host.py)")
    ap.add_argument("--images", action="store_true",
                    help="the picture records (default fixtures/image_records.json) through the provider's image path")
    ap.add_argument("--summary", help="--images: the summary JSON (default <results-dir>/oracle_images_summary.json)")
    args = ap.parse_args()
    if args.images:
        if args.fixtures == str(LANE / "fixtures" / "records.json"):
            args.fixtures = str(IMAGE_FIXTURES)
        return run_images(args)
    if args.dry_run:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(snapshot()))
        recs = json.loads(Path(args.fixtures).read_text())["records"]
        res = dry_run(recs, tok)
        print(json.dumps({"records": len(res), "questions": sum(len(e["questions"]) for e in res),
                          "refused": [f"{e['id']}/{x['name']}" for e in res for x in e["refused"]]}))
        return 0
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
