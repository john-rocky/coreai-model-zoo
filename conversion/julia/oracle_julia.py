#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.0",
#     "transformers==5.0.0",
#     "safetensors==0.8.0",
#     "numpy==2.5.3",
#     "pyarrow==25.0.1",
# ]
# ///
"""Stage 0: the oracle — the publisher's own `julia` package, CPU fp32; never imports the re-authored graph.

    uv run --python 3.12 conversion/julia/oracle_julia.py

Loads the publisher's resident engine exactly as its typed-decisions reproduction does
(`julia.router.engine.FastEngine(<pinned snapshot>, device='cpu', transformer_backend='torch',
strict_encoding=True, max_length=1024, head_length=512, marker_only_head=False)`; transformers 5.0.x,
SDPA attention, 4 threads) and records, in order:

  1. the checkpoint: sha256 of the weights, tokenizer, configs and the runtime files the engine imports
  2. the 2,000 typed-decisions test questions (pinned parquet), one per forward — the publisher's protocol:
     token ids, marker positions and raw logits (`engine.logits`); the accuracy per type from the argmax
     must equal the publisher's CPU FP32 reproduction, 426/600 choice, 542/800 score, 483/600 noul, and
     every answer must equal the one the publisher's scripts/reproduce_typed.py wrote on this machine
     (when its predictions file is present in the work dir)
  3. the publisher's 100 Julia-1-ONNX parity requests (head 256, as the publisher's parity.py built them):
     our raw logits against the publisher's recorded PyTorch logits
  4. window-filling rows: the 35 typed questions longer than 512 tokens, cut to exactly 512 by the
     publisher's non-strict builder (head 256), and 20 synthetic long states (consecutive test states
     in one JSON list) cut to exactly 1024 (head 512) — rows with no padding at all, for the graph only
  5. per export window S in {512, 1024}: every row that fits S run again alone, batch 1, right-padded to S
     — the tensor reference at the export's own input shape — and, for the SUBSET rows, every hidden
     state (embeddings, 22 encoder layers, final norm, after the type embedding, both head layers, the
     scorer at every position), through SDPA (the engine's path) and through eager attention (diagnostic)

Writes oracle/rows.json, oracle/hidden/s{S}/*.npz, oracle/hidden_eager/s{S}/*.npz and results/oracle.json
in the work dir.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the snapshot is local and pinned; never fetch

import collections  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CLS_ID, HEAD_LENGTH, HIDDEN, MASK_ID, MODEL_ID, MODEL_SHA, PAD_ID,  # noqa: E402
                     PUBLISHED_TYPED_CPU, QTYPES, SEP_ID, SOURCE_SHA256, WINDOWS, environment, hashes,
                     oracle_dir, parity_path, parity_requests, results_dir, dataset_path, typed_requests,
                     verify_source, work_dir, write_json)

THREADS = 4
FILL_SYNTHETIC = 20
STATES = ["embeddings", *[f"layer_{i:02d}" for i in range(22)], "final_norm", "head_input", "head_0", "head_1"]


class Recorder:
    """Forward hooks on the publisher's modules; records only while `enabled`."""

    def __init__(self, model):
        self.enabled, self.store, self.handles = False, {}, []
        encoder = model.encoder

        def keep(name):
            def hook(module, args, output):
                if self.enabled:
                    self.store[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
            return hook

        def head_input(module, args, kwargs):
            if self.enabled:
                self.store["head_input"] = args[0].detach().clone()

        self.handles.append(encoder.embeddings.register_forward_hook(keep("embeddings")))
        for i, layer in enumerate(encoder.layers):
            self.handles.append(layer.register_forward_hook(keep(f"layer_{i:02d}")))
        self.handles.append(encoder.layers[1].attn.register_forward_hook(keep("layer_01_attention_output")))
        self.handles.append(encoder.final_norm.register_forward_hook(keep("final_norm")))
        self.handles.append(model.head.layers[0].register_forward_pre_hook(head_input, with_kwargs=True))
        for j, layer in enumerate(model.head.layers):
            self.handles.append(layer.register_forward_hook(keep(f"head_{j}")))

    def remove(self):
        for handle in self.handles:
            handle.remove()


def padded_forward(model, ids, markers, qtype, window):
    """The publisher's model on one row alone, batch 1, right-padded with PAD to `window`."""
    n = len(ids)
    input_ids = torch.full((1, window), PAD_ID, dtype=torch.long)
    input_ids[0, :n] = torch.tensor(ids)
    mask = torch.zeros((1, window), dtype=torch.long)
    mask[0, :n] = 1
    with torch.inference_mode():
        logits = model(input_ids, mask, torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool),
                       torch.tensor([QTYPES.index(qtype)]))
    return logits[0].float().numpy().astype(np.float64)


def fill_rows(tok, typed, sequence):
    """Rows that fill a window exactly: state cut by the publisher's non-strict builder."""
    out = []
    for row in typed:
        encoded = sequence(tok, row["request"], 1024, 512, strict=True)
        if len(encoded["ids"]) > 512:
            cut = sequence(tok, row["request"], 512, HEAD_LENGTH[512], strict=False)
            assert len(cut["ids"]) == 512 and cut["truncated"], row["row_id"]
            out.append({**{k: row[k] for k in ("type", "keys", "request")}, "row_id": "fill512:" + row["row_id"].split(":", 1)[1],
                        "set": "fill", "gold": None, "fill_window": 512, "strict": False})
    by_case = collections.OrderedDict()
    for row in typed:
        by_case.setdefault(row["row_id"].split(":")[1], row)
    cases = list(by_case.values())
    for i in range(FILL_SYNTHETIC):
        base = cases[(i * 7) % len(cases)]
        states = []
        while True:  # consecutive test states in one JSON list, until the row is longer than the window
            states.append(cases[(i * 7 + len(states)) % len(cases)]["request"]["state"])
            request = dict(base["request"], state=states)
            if len(sequence(tok, request, 8192, HEAD_LENGTH[1024], strict=False)["ids"]) > 1024 + 64 * (i % 4):
                break
        encoded = sequence(tok, request, 1024, HEAD_LENGTH[1024], strict=False)
        assert len(encoded["ids"]) == 1024 and encoded["truncated"], (i, len(encoded["ids"]))
        out.append({"row_id": f"fill1024:{i:02d}:{base['row_id'].split(':', 1)[1]}", "set": "fill", "type": base["type"],
                    "keys": base["keys"], "gold": None, "request": request, "fill_window": 1024, "strict": False})
    return out


def pick_subset(rows, window):
    """Rows whose every hidden state the oracle saves (by rule, so both windows get their own):
    shortest and longest typed rows that fit, each type at the median length, K = 2 / 4 / 5, a parity
    row, the two longest parity rows, and two window-filling rows."""
    fits = [r for r in rows if str(window) in r["windows"]]
    typed = [r for r in fits if r["set"] == "typed"]
    by_len = sorted(typed, key=lambda r: r["tokens"])
    chosen = [by_len[0]["row_id"], by_len[-1]["row_id"]]
    for kind in QTYPES:
        of_kind = sorted((r for r in typed if r["type"] == kind), key=lambda r: r["tokens"])
        chosen.append(of_kind[len(of_kind) // 2]["row_id"])
    for k in (2, 4, 5):
        chosen.append(next(r["row_id"] for r in by_len[len(by_len) // 3:] if len(r["markers"]) == k))
    parity = sorted((r for r in fits if r["set"] == "parity"), key=lambda r: r["tokens"])
    chosen += [parity[0]["row_id"], parity[-1]["row_id"], parity[-2]["row_id"]]
    fills = [r for r in fits if r["set"] == "fill" and r["fill_window"] == window]
    chosen += [r["row_id"] for r in fills[:2]]
    return list(dict.fromkeys(chosen))


def main():
    started = time.perf_counter()
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    source = verify_source()
    sys.path.insert(0, str(source))
    import transformers
    from julia.data import sequence
    from julia.model import JuliaDecisionModel
    from julia.router.engine import FastEngine

    strict_engine = FastEngine(source, device="cpu", transformer_backend="torch", strict_encoding=True,
                               max_length=1024, head_length=512, batch_size=16, marker_only_head=False)
    torch.set_num_threads(THREADS)
    model = strict_engine.model
    tok = strict_engine.tokenizer
    facts = {
        "attn_implementation": model.encoder.config._attn_implementation,
        "encoder_specialized": strict_engine.encoder_specialized,
        "marker_only_head": model.marker_only_head,
        "tokenizer": {"class": type(tok).__name__, "cls": tok.cls_token_id, "sep": tok.sep_token_id,
                      "pad": tok.pad_token_id, "mask": tok.mask_token_id, "mask_text": tok.mask_token, "vocab": len(tok)},
        "encoder_config_cls_token_id": model.encoder.config.cls_token_id,
        "temperature_buffer": model.temperature.tolist(),
        "head_activation": [getattr(l.activation, "__name__", str(l.activation)) for l in model.head.layers],
        "head_norm_first": [bool(l.norm_first) for l in model.head.layers],
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    }
    assert (facts["tokenizer"]["cls"], facts["tokenizer"]["sep"], facts["tokenizer"]["pad"], facts["tokenizer"]["mask"]) == \
        (CLS_ID, SEP_ID, PAD_ID, MASK_ID), facts["tokenizer"]
    assert facts["temperature_buffer"] == [1.0, 1.0, 1.0] and not model.marker_only_head
    print("facts", json.dumps(facts), flush=True)

    # 2. typed-decisions: one question per forward, the publisher's protocol.
    typed = typed_requests()
    counts = collections.Counter(r["type"] for r in typed)
    assert len(typed) == 2000 and counts == {"choice": 600, "score": 800, "noul": 600}, counts
    stats = {k: {"count": 0, "correct": 0} for k in QTYPES}
    t0 = time.perf_counter()
    for i, row in enumerate(typed):
        encoded = strict_engine._encode([row["request"]])[0]
        logits = strict_engine.logits([row["request"]])[0]
        row.update(ids=list(encoded["ids"]), markers=list(encoded["markers"]), tokens=len(encoded["ids"]),
                   publisher_logits=[float(v) for v in logits])
        prediction = row["keys"][int(np.argmax(logits))]
        row["prediction"] = prediction
        stats[row["type"]]["count"] += 1
        stats[row["type"]]["correct"] += int(prediction == row["gold"])
        if (i + 1) % 500 == 0:
            print(f"typed {i + 1}/2000 {time.perf_counter() - t0:.0f} s", flush=True)
    typed_seconds = time.perf_counter() - t0
    accuracy = {k: {**v, "published": PUBLISHED_TYPED_CPU[k][0]} for k, v in stats.items()}
    reproduced = all(stats[k]["correct"] == PUBLISHED_TYPED_CPU[k][0] and stats[k]["count"] == PUBLISHED_TYPED_CPU[k][1]
                     for k in QTYPES)
    print("typed accuracy", json.dumps(accuracy), "reproduced" if reproduced else "NOT REPRODUCED", flush=True)
    author_file = work_dir() / "author-repro" / "original-criteria-predictions.jsonl"
    author_check = None
    if author_file.exists():
        author = [json.loads(line) for line in author_file.read_text().splitlines() if line.strip()]
        same = sum(a["prediction"] == r["prediction"] and a["id"] == r["row_id"].split(":", 1)[1] for a, r in zip(author, typed))
        author_check = {"file": str(author_file), "rows": len(author), "identical_predictions": same}
        print("vs the publisher's reproduce_typed.py predictions", author_check, flush=True)

    # 3. the publisher's parity requests, built the way its parity.py built them (head 256, strict).
    parity = parity_requests()
    parity_engine = FastEngine(source, device="cpu", transformer_backend="torch", strict_encoding=True,
                               max_length=1024, head_length=256, batch_size=16, marker_only_head=False)
    torch.set_num_threads(THREADS)
    parity_errors, parity_argmax = [], 0
    for row in parity:
        encoded = parity_engine._encode([row["request"]])[0]
        logits = np.asarray(parity_engine.logits([row["request"]])[0], dtype=np.float64)
        row.update(ids=list(encoded["ids"]), markers=list(encoded["markers"]), tokens=len(encoded["ids"]),
                   publisher_logits=[float(v) for v in logits])
        recorded = np.asarray(row["recorded_pytorch_logits"], dtype=np.float64)
        parity_errors.append(float(np.max(np.abs(logits - recorded))))
        parity_argmax += int(logits.argmax() == recorded.argmax())
    parity_summary = {"rows": len(parity), "argmax_identical": parity_argmax, "max_abs_logit_vs_recorded": max(parity_errors),
                      "note": "recorded = the publisher's parity.py, batches of 4 padded to a multiple of 8; ours = one per forward"}
    print("parity", parity_summary, flush=True)

    # 4. window-filling rows (non-strict builder, the graph's boundary; never a host row).
    fills = fill_rows(tok, typed, sequence)
    engines = {w: FastEngine(source, device="cpu", transformer_backend="torch", strict_encoding=False,
                             max_length=w, head_length=HEAD_LENGTH[w], batch_size=16, marker_only_head=False)
               for w in WINDOWS}
    torch.set_num_threads(THREADS)
    for row in fills:
        engine = engines[row["fill_window"]]
        encoded = engine._encode([row["request"]])[0]
        logits = engine.logits([row["request"]])[0]
        assert len(encoded["ids"]) == row["fill_window"]
        row.update(ids=list(encoded["ids"]), markers=list(encoded["markers"]), tokens=len(encoded["ids"]),
                   publisher_logits=[float(v) for v in logits])
    print("fill rows", collections.Counter(r["fill_window"] for r in fills), flush=True)

    # The same ids at both budgets for every strict row that fits the 512 window (HEAD_LENGTH note).
    budget_check = {"rows_fitting_512": 0, "identical_at_head_256": 0, "refused_at_head_256": []}
    for row in typed + parity:
        if row["tokens"] > 512:
            continue
        budget_check["rows_fitting_512"] += 1
        try:
            again = sequence(tok, row["request"], 512, HEAD_LENGTH[512], strict=True)
        except ValueError:
            budget_check["refused_at_head_256"].append(row["row_id"])
            continue
        budget_check["identical_at_head_256"] += int(again["ids"] == row["ids"] and again["markers"] == row["markers"])
    print("budget check", {k: (v if not isinstance(v, list) else len(v)) for k, v in budget_check.items()}, flush=True)

    # 5. per window: the tensor reference at the export's input shape, and the SUBSET hidden states.
    rows = typed + parity + fills
    refused_512 = set(budget_check["refused_at_head_256"])
    for row in rows:
        row["windows"] = {}
    diagnostics, pad_rows = {}, {}
    recorder = Recorder(model)
    eager = JuliaDecisionModel.from_pretrained(source, memory_map=False).eval()
    # transformers 5.0 ModernBERT picks its attention function from the shared config at every forward
    # (MODERNBERT_ATTENTION_FUNCTION[config._attn_implementation]); set_attn_implementation() refuses it.
    eager.encoder.config._attn_implementation = "eager"
    eager_recorder = Recorder(eager)
    subsets = {}
    try:
        for window in WINDOWS:
            fits = [r for r in rows if r["tokens"] <= window and r["row_id"] not in (refused_512 if window == 512 else ())
                    and (r["set"] != "fill" or r["fill_window"] == window)]
            for row in fits:
                row["windows"][str(window)] = None
            subsets[window] = pick_subset(rows, window)
            hidden_dir, eager_dir = oracle_dir() / "hidden" / f"s{window}", oracle_dir() / "hidden_eager" / f"s{window}"
            hidden_dir.mkdir(parents=True, exist_ok=True)
            eager_dir.mkdir(parents=True, exist_ok=True)
            spread, pads = {name: 0.0 for name in STATES}, []
            relative = {name: 0.0 for name in STATES}
            t0 = time.perf_counter()
            for i, row in enumerate(fits):
                keep = row["row_id"] in subsets[window]
                recorder.enabled, recorder.store = keep, {}
                padded = padded_forward(model, row["ids"], row["markers"], row["type"], window)
                recorder.enabled = False
                ref = np.asarray(row["publisher_logits"], dtype=np.float64)
                row["windows"][str(window)] = {"marker_logits": [float(v) for v in padded],
                                               "padded_vs_engine_max_abs": float(np.max(np.abs(padded - ref)))}
                if keep:
                    store = recorder.store
                    with torch.inference_mode():
                        token_logits = model.scorer(store["head_1"]).squeeze(-1)[0].float().numpy()
                    n = row["tokens"]
                    dump = {name: store[name][0].float().numpy() for name in STATES}
                    np.savez(hidden_dir / f"{subsets[window].index(row['row_id']):03d}.npz", token_logits=token_logits,
                             marker_logits=padded.astype(np.float32), **dump)
                    if n + 64 < window:
                        attention = store["layer_01_attention_output"][0].float().numpy()
                        pads.append({"row_id": row["row_id"], "fully_masked_rows_zero": bool(np.all(attention[n + 64:] == 0.0)),
                                     "radius_edge_row_nonzero": bool(np.any(attention[n + 63] != 0.0))})
                    eager_recorder.enabled, eager_recorder.store = True, {}
                    eager_logits = padded_forward(eager, row["ids"], row["markers"], row["type"], window)
                    eager_recorder.enabled = False
                    eager_dump = {name: eager_recorder.store[name][0].float().numpy() for name in STATES}
                    np.savez(eager_dir / f"{subsets[window].index(row['row_id']):03d}.npz",
                             marker_logits=eager_logits.astype(np.float32), **eager_dump)
                    row["windows"][str(window)]["eager_marker_max_abs"] = float(np.max(np.abs(eager_logits - padded)))
                    for name in STATES:
                        delta = float(np.max(np.abs(dump[name][:n].astype(np.float64) - eager_dump[name][:n])))
                        spread[name] = max(spread[name], delta)
                        relative[name] = max(relative[name], delta / float(np.max(np.abs(dump[name][:n]))))
                if (i + 1) % 500 == 0:
                    print(f"s{window} {i + 1}/{len(fits)} {time.perf_counter() - t0:.0f} s", flush=True)
            diagnostics[str(window)] = {
                "rows": len(fits), "seconds": time.perf_counter() - t0,
                "max_padded_vs_engine": max(r["windows"][str(window)]["padded_vs_engine_max_abs"] for r in fits),
                "sdpa_vs_eager_max_abs_real": spread, "sdpa_vs_eager_relative_real": relative,
                "sdpa_vs_eager_marker_max_abs": max(r["windows"][str(window)].get("eager_marker_max_abs", 0.0) for r in fits)}
            pad_rows[str(window)] = pads
            print(f"s{window}: {len(fits)} rows, padded vs engine {diagnostics[str(window)]['max_padded_vs_engine']:.2e}, "
                  f"SDPA vs eager marker {diagnostics[str(window)]['sdpa_vs_eager_marker_max_abs']:.2e}", flush=True)
    finally:
        recorder.remove()
        eager_recorder.remove()
    for row in rows:
        row["windows"] = {w: v for w, v in row["windows"].items() if v is not None}

    determinism = {}
    for window in WINDOWS:
        probe = [r for r in rows if str(window) in r["windows"]][:3]
        determinism[str(window)] = max(float(np.max(np.abs(padded_forward(model, r["ids"], r["markers"], r["type"], window)
                                                           - np.asarray(r["windows"][str(window)]["marker_logits"]))))
                                       for r in probe)

    gaps, tops = [], []
    for row in typed:
        z = np.asarray(row["publisher_logits"], dtype=np.float64)
        p = np.exp(z - z.max())
        p /= p.sum()
        top = np.sort(p)[::-1]
        gaps.append(float(top[0] - top[1]))
        tops.append(float(top[0]))
    gaps, tops = np.asarray(gaps), np.asarray(tops)
    boundary = {"typed_rows": len(gaps), "top_two_gap_below_0.05": int((gaps < 0.05).sum()),
                "top_two_gap_below_0.2": int((gaps < 0.2).sum()), "top_probability_above_0.99": int((tops > 0.99).sum()),
                "top_probability_below_0.6": int((tops < 0.6).sum()), "min_top_two_gap": float(gaps.min())}

    failures = []
    if not reproduced:
        failures.append("the publisher's 426 / 542 / 483 not reproduced")
    if author_check and author_check["identical_predictions"] != 2000:
        failures.append("answers differ from the publisher's reproduce_typed.py run")
    if parity_summary["argmax_identical"] != len(parity):
        failures.append("parity requests: argmax differs from the publisher's recorded logits")
    output_files = [oracle_dir() / "rows.json", *sorted((oracle_dir() / "hidden").rglob("*.npz")),
                    *sorted((oracle_dir() / "hidden_eager").rglob("*.npz"))]
    document = {"schema": "julia-oracle-rows/1", "model": MODEL_ID, "model_sha": MODEL_SHA,
                "subset": {str(w): subsets[w] for w in WINDOWS}, "head_length": {str(w): HEAD_LENGTH[w] for w in WINDOWS},
                "rows": rows}
    write_json(oracle_dir() / "rows.json", document)
    report = {
        "status": "FAIL" if failures else "PASS", "failures": failures, "stage": "oracle", "model": MODEL_ID,
        "model_sha": MODEL_SHA, "reference": ("julia.router.engine.FastEngine(snapshot, device='cpu', transformer_backend='torch', "
                                             "strict_encoding=True, max_length=1024, head_length=512, marker_only_head=False), "
                                             "one question per forward"),
        "precision": "fp32", "threads": THREADS, "facts": facts, "typed_accuracy": accuracy, "typed_seconds": typed_seconds,
        "vs_publisher_reproduce_typed": author_check, "parity": parity_summary, "budget_check": budget_check,
        "rows_per_window": {str(w): sum(str(w) in r["windows"] for r in rows) for w in WINDOWS},
        "rows_by_set": dict(collections.Counter(r["set"] for r in rows)), "boundary": boundary,
        "tokens": {"max_typed": max(r["tokens"] for r in typed), "max_parity": max(r["tokens"] for r in parity)},
        "code_path_diagnostics": diagnostics, "sliding_pad_rows": pad_rows, "determinism_max_abs": determinism,
        "environment": environment(("torch", "transformers", "tokenizers", "safetensors", "numpy", "pyarrow")),
        "transformers": transformers.__version__, "source_sha256": SOURCE_SHA256,
        "data": {"dataset": str(dataset_path()), "parity": str(parity_path())},
        "output_hashes": hashes(output_files), "seconds": time.perf_counter() - started,
    }
    write_json(results_dir() / "oracle.json", report)
    print(report["status"], failures, f"{report['seconds']:.0f} s", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
