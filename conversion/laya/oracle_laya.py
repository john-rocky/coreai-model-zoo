#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.12.1",
#     "transformers==5.17.0",
#     "laya==0.3.4",
#     "safetensors==0.8.0",
#     "numpy==2.5.3",
#     "tokenizers==0.23.2",
#     "huggingface-hub==1.32.0",
# ]
# ///
"""Stage 0: the oracle — the publisher's own laya 0.3.4 model, CPU fp32; never imports the re-authored graph.

    uv run --python 3.12 conversion/laya/oracle_laya.py

Loads `laya.load(<pinned snapshot>, subfolder="multilingual", device="cpu")` exactly as the package
does (transformers 5.x ModernBERT with SDPA attention; transformers 4.x cannot read this encoder
config) and checks, in order:

  1. the checkpoint: sha256 of every source file, the safetensors tensors equal to the loaded
     state dict, 321,908,998 parameters; the package's in-place tokenizer_config.json fix-up is a no-op
  2. the fixture is the official builder's output: `build_sequence(tok, state, q, max_len=window,
     head_max_len=256)` reproduces every row's ids and marker positions (201 x 2)
  3. the fixture is the official answer: `Agent.predict` batched per fixture (max_len = window)
     reproduces every frozen answer dictionary, raw marker logits and act logits
  4. the per-row reference this port gates against: `agent.model(...)` on each row alone, batch 1,
     right-padded to ITS window (256 or 512) — marker logits, act logits, pooled_cls and act features
     for all 402 rows; every hidden state (embeddings, 22 residual layers, final_norm, after type_emb,
     both head layers, token logits at every position) for the SUBSET rows
  5. that reference vs the frozen fixture (argmax, |dp| <= 1e-3 at T=1, act probability) and vs the
     LiteRT lane's captures of the same official model padded to 512 (marker |d| <= 1e-3, act
     relative <= 1e-4)

Diagnostics (recorded, not gated): the head layers through PyTorch's non-fast path and the encoder
through eager attention, to measure how far the official model's own code paths sit apart; whether
fully masked sliding-window query rows (pad positions more than 64 past the last real token) produce
an exactly-zero attention output under SDPA.

Writes oracle/outputs_s{S}.npz, oracle/hidden/s{S}/NNN.npz (SDPA, the official path),
oracle/hidden_eager/s{S}/NNN.npz (eager attention, diagnostic) and results/oracle.json in the work dir.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the snapshot is local and pinned; never fetch

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (FIXTURE_SHA256, HEAD_MAX_LEN, HIDDEN, MODEL_ID, MODEL_SHA, SOURCE_SHA256,  # noqa: E402
                     SUBFOLDER, SUBSET, WINDOWS, environment, fixtures_dir, hashes, load_capture,
                     load_fixture_states, load_rows, oracle_dir, results_dir, row_inputs, sha256_of,
                     snapshot_root, verify_source, write_json)
from _gate_metrics import POLICY, evaluate_row, summarize  # noqa: E402

PARAMETER_COUNT = 321_908_998
THREADS = 4


class Recorder:
    """Forward hooks on the official modules; records only while `enabled`."""

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

        def act_input(module, args):
            if self.enabled:
                self.store["act_input"] = args[0].detach().clone()

        self.handles.append(encoder.embeddings.register_forward_hook(keep("embeddings")))
        for i, layer in enumerate(encoder.layers):
            self.handles.append(layer.register_forward_hook(keep(f"layer_{i:02d}")))
        self.handles.append(encoder.layers[1].attn.register_forward_hook(keep("layer_01_attention_output")))
        self.handles.append(encoder.final_norm.register_forward_hook(keep("final_norm")))
        self.handles.append(model.head.layers[0].register_forward_pre_hook(head_input, with_kwargs=True))
        for j, layer in enumerate(model.head.layers):
            self.handles.append(layer.register_forward_hook(keep(f"head_{j}")))
        self.handles.append(model.act_head.register_forward_pre_hook(act_input))

    def remove(self):
        for handle in self.handles:
            handle.remove()


def official_forward(model, row, window):
    x = row_inputs(row, window)
    ids = torch.from_numpy(x["input_ids"]).long()
    mask = torch.from_numpy(x["attention_mask"]).long()
    marker_pos = torch.tensor([row["marker_positions"]], dtype=torch.long)
    marker_mask = torch.ones(1, row["K"], dtype=torch.bool)
    qtype = torch.tensor([row["qtype"]], dtype=torch.long)
    with torch.no_grad():
        logits, act = model(ids, mask, marker_pos, marker_mask, qtype)
    return logits[0].numpy().copy(), act[0].numpy().copy()


def check_weights(model, source: Path) -> dict:
    from safetensors.torch import load_file

    weights = load_file(str(source / "model.safetensors"))
    state = model.state_dict()
    assert set(weights) == set(state), sorted(set(weights) ^ set(state))
    assert all(torch.equal(state[k], v.float()) for k, v in weights.items()), "loaded weights differ from the file"
    dtypes = sorted({str(v.dtype) for k, v in weights.items() if k != "temperature"})
    count = sum(v.numel() for v in weights.values())
    assert count == PARAMETER_COUNT, count
    return {"tensor_count": len(weights), "parameter_count": count, "file_dtypes_except_temperature": dtypes,
            "temperature_buffer": weights["temperature"].float().tolist()}


def check_sequences(agent, rows_by_window, states) -> dict:
    from laya.common import build_sequence, render_options

    out = {}
    for window, rows in rows_by_window.items():
        mismatches = []
        for row in rows:
            q = agent._to_internal(row["question"])
            ids, markers = build_sequence(agent.tok, states[row["fixture_id"]]["state"], q,
                                          max_len=window, head_max_len=HEAD_MAX_LEN)
            if ids != row["sequence_ids"] or markers != row["marker_positions"] or len(markers) != len(render_options(q)):
                mismatches.append(row["row_id"])
        out[str(window)] = {"rows": len(rows), "identical": len(rows) - len(mismatches), "mismatches": mismatches}
    return out


def rerun_official_batches(agent, rows_by_window, states) -> dict:
    """`Agent.predict` per fixture with max_len = window: the frozen answers, bit for bit if the stack is the same."""
    captured = []
    handle = agent.model.register_forward_hook(
        lambda module, args, output: captured.append((args[0].shape, output[0].detach().clone(), output[1].detach().clone())))
    out = {}
    try:
        for window, rows in rows_by_window.items():
            agent.cfg["max_len"] = window
            groups = OrderedDict()
            for row in rows:
                groups.setdefault(row["fixture_id"], []).append(row)
            exact, shape_ok, marker_error, act_error, differing = 0, 0, 0.0, 0.0, []
            for fixture_id, members in groups.items():
                fixture = states[fixture_id]
                assert [r["question_id"] for r in members] == list(fixture["questions"])
                captured.clear()
                result = agent.predict(fixture["state"], fixture["questions"])
                assert len(captured) == 1
                shape, logits, act = captured[0]
                for b, row in enumerate(members):
                    same = result["answers"][row["question_id"]] == row["official_answer"]
                    exact += same
                    shape_ok += list(shape) == row["official_batch_shape"]
                    m = float(np.max(np.abs(logits[b, :row["K"]].numpy().astype(np.float64) - row["raw_logits"])))
                    a = float(np.max(np.abs(act[b].numpy().astype(np.float64) - row["raw_act_logits"])))
                    marker_error, act_error = max(marker_error, m), max(act_error, a)
                    if not same:
                        differing.append(row["row_id"])
            out[str(window)] = {"rows": len(rows), "exact_answer_dicts": exact, "batch_shape_identical": shape_ok,
                                "max_marker_abs_vs_fixture_raw": marker_error, "max_act_abs_vs_fixture_raw": act_error,
                                "differing_rows": differing}
    finally:
        handle.remove()
        agent.cfg["max_len"] = json.loads((snapshot_root() / SUBFOLDER / "rl_agent_config.json").read_text())["max_len"]
    return out


def main():
    started = time.perf_counter()
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    source = verify_source()
    tokenizer_config = source / "tokenizer" / "tokenizer_config.json"
    import laya

    agent = laya.load(str(snapshot_root()), subfolder=SUBFOLDER, device="cpu")
    # laya's _fix_tokenizer_config rewrites this file in place when it needs fixing; at this revision it must not.
    assert sha256_of(tokenizer_config) == SOURCE_SHA256["tokenizer/tokenizer_config.json"], "laya rewrote tokenizer_config.json"
    model = agent.model.float().eval().requires_grad_(False)
    encoder = model.encoder
    facts = {
        "attn_implementation": encoder.config._attn_implementation,
        "device": str(agent.device), "dtype": str(agent.dtype),
        "tokenizer": {"cls": agent.tok.cls_token_id, "sep": agent.tok.sep_token_id, "pad": agent.tok.pad_token_id,
                      "mask": agent.tok.mask_token_id, "unk": agent.tok.unk_token_id, "vocab": len(agent.tok),
                      "class": type(agent.tok).__name__},
        "config": {k: agent.cfg[k] for k in ("max_len", "head_max_len", "temperature", "temperature_by_options", "act_costs")},
        "encoder_layer_types": list(encoder.config.layer_types), "sliding_window": encoder.config.sliding_window,
        "head_layer_norm_first": [bool(l.norm_first) for l in model.head.layers],
        "head_activation": [getattr(l.activation, "__name__", str(l.activation)) for l in model.head.layers],
    }
    assert facts["tokenizer"]["cls"] == 2 and facts["tokenizer"]["sep"] == 1 and facts["tokenizer"]["pad"] == 0
    assert facts["tokenizer"]["mask"] == 4 and agent.cfg["temperature"] == [1.0, 1.0, 1.0] and not agent.cfg["temperature_by_options"]
    weights = check_weights(model, source)
    print("weights", weights["parameter_count"], "params; attention", facts["attn_implementation"], flush=True)

    states = load_fixture_states()
    rows_by_window = {w: load_rows(w) for w in WINDOWS}
    sequences = check_sequences(agent, rows_by_window, states)
    print("sequence identity", json.dumps(sequences), flush=True)
    batches = rerun_official_batches(agent, rows_by_window, states)
    print("official batched re-run", json.dumps({w: {k: v for k, v in b.items() if k != "differing_rows"} for w, b in batches.items()}), flush=True)

    config = {"temperature": agent.cfg["temperature"], "temperature_by_options": agent.cfg["temperature_by_options"]}
    recorder = Recorder(model)
    windows, output_files = {}, []
    try:
        for window, rows in rows_by_window.items():
            hidden_dir = oracle_dir() / "hidden" / f"s{window}"
            hidden_dir.mkdir(parents=True, exist_ok=True)
            n_rows = len(rows)
            arrays = {"token_logits": np.zeros((n_rows, window), np.float32), "pooled_cls": np.zeros((n_rows, HIDDEN), np.float32),
                      "act_logits": np.zeros((n_rows, 2), np.float32), "feats": np.zeros((n_rows, 4), np.float32),
                      "marker_logits": np.full((n_rows, 20), np.nan, np.float32), "K": np.zeros(n_rows, np.int32)}
            records, litert_records, pad_rows = [], [], []
            for index, row in enumerate(rows):
                recorder.enabled, recorder.store = True, {}
                t0 = time.perf_counter()
                marker, act = official_forward(model, row, window)
                ms = (time.perf_counter() - t0) * 1000
                recorder.enabled = False
                store = recorder.store
                with torch.no_grad():
                    token_logits = model.scorer(store["head_1"]).squeeze(-1)[0].numpy()
                act_input = store["act_input"][0].numpy()
                assert np.isfinite(act_input).all() and np.isfinite(token_logits).all(), row["row_id"]
                k, n = row["K"], row["sequence_length"]
                arrays["token_logits"][index] = token_logits
                arrays["pooled_cls"][index] = act_input[:HIDDEN]
                arrays["feats"][index] = act_input[HIDDEN:]
                arrays["act_logits"][index] = act
                arrays["marker_logits"][index, :k] = marker
                arrays["K"][index] = k
                capture = load_capture(window, index, row)
                litert = evaluate_row(row, marker, act, config, tensor_reference={
                    "marker_logits": capture["original_logits"], "act_logits": capture["original_act"]})
                litert_records.append(litert)
                pooled_delta = float(np.max(np.abs(act_input[:HIDDEN].astype(np.float64) - capture["original_pooled_cls"][0])))
                record = {**evaluate_row(row, marker, act, config), "ms": ms,
                          "token_logits_at_markers_max_abs_vs_marker_logits": float(np.max(np.abs(
                              token_logits[row["marker_positions"]].astype(np.float64) - marker))),
                          "vs_fixture_batched_raw_marker_max_abs": float(np.max(np.abs(marker.astype(np.float64) - row["raw_logits"]))),
                          "vs_fixture_batched_raw_act_max_abs": float(np.max(np.abs(act.astype(np.float64) - row["raw_act_logits"]))),
                          "vs_litert_original_marker_max_abs": litert["marker_max_abs_error"],
                          "vs_litert_original_act_relative": litert["act_relative_error"],
                          "pooled_cls_max_abs_delta_vs_litert_original": pooled_delta,
                          "pooled_cls_relative_vs_litert_original": pooled_delta / float(np.max(np.abs(capture["original_pooled_cls"])))}
                if n + 64 < window:
                    # Sliding layer 1: a pad query i sees a real key only while i - 64 <= n - 1 (inclusive radius 64).
                    attention = store["layer_01_attention_output"][0].numpy()
                    record["sliding_fully_masked_rows_zero"] = bool(np.all(attention[n + 64:] == 0.0))
                    record["sliding_radius_edge_row_nonzero"] = bool(np.any(attention[n + 63] != 0.0))
                    pad_rows.append((record["sliding_fully_masked_rows_zero"], record["sliding_radius_edge_row_nonzero"]))
                if index in SUBSET:
                    names = ["embeddings", *[f"layer_{i:02d}" for i in range(len(encoder.layers))], "final_norm",
                             "head_input", "head_0", "head_1"]
                    dump = {name: store[name][0].numpy() for name in names}
                    inputs = row_inputs(row, window)
                    dump.update(token_logits=token_logits, marker_logits=marker, act_logits=act,
                                pooled_cls=act_input[:HIDDEN], feats=act_input[HIDDEN:],
                                input_ids=inputs["input_ids"][0], attention_mask=inputs["attention_mask"][0])
                    np.savez(hidden_dir / f"{index:03d}.npz", **dump)
                    record["hidden_abs_max_real_positions"] = {name: float(np.max(np.abs(dump[name][:n]))) for name in names}
                    record["hidden_abs_max_all_positions"] = {name: float(np.max(np.abs(dump[name]))) for name in names}
                records.append(record)
                if index % 25 == 0 or not (record["answer_pass"] and litert["tensor_pass"]):
                    print(f"s{window} {index:3d} {row['row_id']:<28} n={n:3d} K={k:2d} {ms:6.1f} ms "
                          f"dp={record['max_probability_error']:.2e} marker-vs-litert={litert['marker_max_abs_error']:.2e} "
                          f"act-rel-vs-litert={litert['act_relative_error']:.2e}", flush=True)
            path = oracle_dir() / f"outputs_s{window}.npz"
            np.savez(path, **arrays)
            output_files.append(path)
            output_files.extend(sorted(hidden_dir.glob("*.npz")))
            fixture_summary = summarize(records)
            litert_summary = {k: v for k, v in summarize(litert_records).items()
                              if k in ("tensor_status", "max_marker_abs_error", "max_act_relative_error",
                                       "max_act_abs_error", "max_act_reference_scale", "tensor_failures")}
            windows[str(window)] = {
                "vs_fixture": fixture_summary, "vs_litert_original": litert_summary,
                "max_vs_fixture_batched_raw_marker_abs": max(r["vs_fixture_batched_raw_marker_max_abs"] for r in records),
                "max_vs_fixture_batched_raw_act_abs": max(r["vs_fixture_batched_raw_act_max_abs"] for r in records),
                "max_pooled_cls_relative_vs_litert_original": max(r["pooled_cls_relative_vs_litert_original"] for r in records),
                "max_token_logits_at_markers_vs_marker_logits": max(r["token_logits_at_markers_max_abs_vs_marker_logits"] for r in records),
                "sliding_pad_rows": {"rows_checked": len(pad_rows), "fully_masked_rows_zero": sum(a for a, _ in pad_rows),
                                     "radius_edge_row_nonzero": sum(b for _, b in pad_rows)},
                "rows": records}
            print(f"s{window}: vs fixture {fixture_summary['answer_status']} (argmax {fixture_summary['argmax_identical']}/"
                  f"{fixture_summary['choice_score_rows']}, max dp {fixture_summary['max_probability_error']:.2e}); "
                  f"vs LiteRT original {litert_summary['tensor_status']} (marker {litert_summary['max_marker_abs_error']:.2e}, "
                  f"act rel {litert_summary['max_act_relative_error']:.2e})", flush=True)
    finally:
        recorder.remove()

    diagnostics = code_path_diagnostics(model, rows_by_window, output_files)
    determinism = {}
    for window, rows in rows_by_window.items():
        first = [official_forward(model, rows[i], window) for i in (0, 59, 181)]
        second = [official_forward(model, rows[i], window) for i in (0, 59, 181)]
        determinism[str(window)] = max(float(np.max(np.abs(a[0] - b[0]))) + float(np.max(np.abs(a[1] - b[1])))
                                       for a, b in zip(first, second))

    failures = []
    for window in WINDOWS:
        w = windows[str(window)]
        if w["vs_fixture"]["answer_status"] != "PASS":
            failures.append(f"s{window}: official batch-1 answers differ from the frozen fixture beyond the bar")
        if w["vs_litert_original"]["tensor_status"] != "PASS":
            failures.append(f"s{window}: official batch-1 tensors differ from the LiteRT capture beyond the bar")
        if sequences[str(window)]["mismatches"]:
            failures.append(f"s{window}: build_sequence does not reproduce the fixture ids")
    report = {
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "stage": "oracle", "model": MODEL_ID, "model_sha": MODEL_SHA, "subfolder": SUBFOLDER,
        "reference": "laya.load(snapshot, subfolder='multilingual', device='cpu'); agent.model(...) batch 1, right-padded to the row's window",
        "precision": "fp32", "threads": THREADS, "facts": facts, "weights": weights, "policy": POLICY,
        "environment": environment(("torch", "transformers", "laya", "tokenizers", "safetensors", "numpy", "huggingface-hub")),
        "source_sha256": SOURCE_SHA256, "fixture_sha256": FIXTURE_SHA256, "subset": list(SUBSET),
        "sequence_identity": sequences, "official_batched_rerun": batches, "determinism_max_abs": determinism,
        "code_path_diagnostics": diagnostics, "windows": windows,
        "output_hashes": hashes(output_files), "seconds": time.perf_counter() - started,
    }
    write_json(results_dir() / "oracle.json", report)
    print(report["status"], failures, f"{report['seconds']:.0f} s", flush=True)
    return 0 if report["status"] == "PASS" else 1


def code_path_diagnostics(model, rows_by_window, saved_files: list) -> dict:
    """How far the official model's own code paths sit apart on the subset rows. Informational.

    Per variant: marker / act distance and, per saved hidden state, the max |difference| over real
    positions (and the same excluding position 0, which carries the ~1.4e4 activation from layer 11 on,
    where one fp32 ulp is ~1e-3). This is the noise floor a different-but-correct implementation sits in.
    The eager-attention hidden states are saved too (oracle/hidden_eager/s{S}/NNN.npz): the same
    official weights and transformers code with the matmul-softmax-matmul attention the port authors.
    """
    names = ["embeddings", *[f"layer_{i:02d}" for i in range(len(model.encoder.layers))], "final_norm",
             "head_input", "head_0", "head_1"]

    def run_all(save_as: str | None = None):
        recorder = Recorder(model)
        out = {}
        try:
            for window, rows in rows_by_window.items():
                out[window] = []
                for index in SUBSET:
                    recorder.enabled, recorder.store = True, {}
                    marker, act = official_forward(model, rows[index], window)
                    recorder.enabled = False
                    states = {n: recorder.store[n][0].numpy() for n in names}
                    out[window].append((marker, act, states, rows[index]["sequence_length"]))
                    if save_as:
                        directory = oracle_dir() / save_as / f"s{window}"
                        directory.mkdir(parents=True, exist_ok=True)
                        np.savez(directory / f"{index:03d}.npz", marker_logits=marker, act_logits=act, **states)
                        saved_files.append(directory / f"{index:03d}.npz")
        finally:
            recorder.remove()
        return out

    base = run_all()
    variants = {}
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        variants["head_layers_without_fast_path"] = run_all()
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    encoder = model.encoder
    previous = encoder.config._attn_implementation
    try:
        if hasattr(encoder, "set_attn_implementation"):
            encoder.set_attn_implementation("eager")
        else:
            encoder.config._attn_implementation = "eager"
        eager_name = encoder.config._attn_implementation
        variants["encoder_eager_attention"] = run_all(save_as="hidden_eager")
    finally:
        if hasattr(encoder, "set_attn_implementation"):
            encoder.set_attn_implementation(previous)
        else:
            encoder.config._attn_implementation = previous
    assert encoder.config._attn_implementation == previous
    out = {}
    for name, result in variants.items():
        out[name] = {}
        for window in rows_by_window:
            pairs = list(zip(base[window], result[window]))
            layers = {}
            for n in names:
                layers[n] = {
                    "max_abs_real": max(float(np.max(np.abs(a[2][n][:a[3]].astype(np.float64) - b[2][n][:a[3]]))) for a, b in pairs),
                    "max_abs_real_excluding_position_0": max(float(np.max(np.abs(
                        a[2][n][1:a[3]].astype(np.float64) - b[2][n][1:a[3]]))) for a, b in pairs)}
            out[name][str(window)] = {
                "max_marker_abs": max(float(np.max(np.abs(a[0].astype(np.float64) - b[0]))) for a, b in pairs),
                "max_act_relative": max(float(np.max(np.abs(a[1].astype(np.float64) - b[1])) / np.max(np.abs(a[1]))) for a, b in pairs),
                "hidden_states": layers}
    out["encoder_eager_attention"]["attn_implementation_used"] = eager_name
    return out


if __name__ == "__main__":
    raise SystemExit(main())
