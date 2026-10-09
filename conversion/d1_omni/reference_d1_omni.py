#!/usr/bin/env python3
"""d1-omni-600M reference: the publisher's own model on every fixture request, CPU fp32 — what every later stage
of this port is compared against.

    python3 conversion/d1_omni/reference_d1_omni.py
      -> $ZOO_WORK_ROOT/_d1_omni/ref/records_ref.json, ref/npz/<name>.npz, results/ref_summary.json

The model is the checkpoint's own code at the pinned revision (`modeling_d1.D1OmniModel`, loaded by
`AutoModel.from_pretrained(<snapshot>, trust_remote_code=True, dtype=torch.float32)` under transformers 5.19), run
on the CPU in fp32 through its public API, unchanged: `probabilities(state, questions, images, audio)` per request,
which batches the request's questions in `_run` (one batch per request at these lengths). bf16 and MPS are never
the reference. Nothing in the publisher's files is edited; the numbers are read by hooks:

  - `_run` / `_forward` (instance-attribute wrappers): the publisher's rows (prefix, ids, markers, question,
    calibrate) of each batch and the probabilities it returns;
  - forward hooks on `encoder` (the trunk: embeddings in, `embedding_norm` output out), `head` (its five inputs
    and the raw marker logits [B, Kmax], -1e4 past each row's K), `vision`, `vision.tower` (the NaFlex inputs
    and `last_hidden_state`), `audio` and `audio.frontend` (the normalised log-mel and the frame count).

Per question it keeps the row's ids and markers (asserted equal to host.request_rows over the raw tokenizer), the
raw logits, the temperature (text rows), softmax(raw logits) before the noul flip, the publisher's probabilities,
the argmax key, the top-2 margin, the gold key and whether they agree. Per request: `system_one` run again and its
response asserted equal to host.response(rows, the probabilities that run returned) (answers and usage), the
seconds. Asserted besides: the loaded state dict equals model.safetensors tensor for tensor; the media prefix
lengths equal host.py's arithmetic; for every batch the head's input is the trunk output cut per row (the hidden
states saved here) and the head re-run on it returns the logits bit for bit (and from the reloaded npz files for
the single-question rows); record 0 run again at the end is bit-identical. Recorded (not asserted): every
request's second run (system_one) against the first, the host's numpy softmax against the publisher's, ten
multi-question records with each question alone against the batch, and the red arm (red_arm_000 against tv4_000).

npz (round 3's graph references): the trunk output `hidden [P+n, 1024]` with ids / markers / raw logits for 12
text rows (bucket 256: 4, 512: 3, 2048: 3, 4096: 2) and every row of the prefix records (card_cats, card_audio,
aud_01..03) with `prefix [P, 1024]`, the NaFlex inputs and tower output (image) or the mel and frames (audio).

The image rows (round 5) are a second file, made the same way from the fixtures' current version:

    python3 conversion/d1_omni/reference_d1_omni.py --only img_01 img_02 img_03 'imgm_*' \
        --out $ZOO_WORK_ROOT/_d1_omni/ref/records_ref_images.json

--only takes record ids or fnmatch patterns; each record runs in text mode and with its media, as above, and every
selected record with media writes ref/npz/<id>.npz (never over an existing file) with, besides the arrays above, the
tower's position table after the resize (`pos_resized [m, 1024, 768]` for the m distinct crop shapes `pos_shapes`,
`pos_index [crops]` = each crop's table; read by an instance wrapper on the embeddings' resize_positional_embeddings,
padded the publisher's way). The determinism check runs the first record in its media mode; the batch invariance
runs every selected media record's questions alone; the red arm asks img_01's questions over imgm_11's picture
(a suitcase vs none) instead of red_arm_000. The summary goes to results/ref_images_summary.json. That run read the
fixtures' version 2 (`--fixtures-sha256 07ea38a2…`); --only reads version 3 (round 6) unless told otherwise.

The audio rows (round 6) are a third file, made the same way:

    python3 conversion/d1_omni/reference_d1_omni.py --only 'aud_0[4-9]' 'aud_1[0-5]' \
        --out $ZOO_WORK_ROOT/_d1_omni/ref/records_ref_audio.json

Each audio npz holds, besides the arrays above, the conformer's output before the adapter (`encoder_out [T, 512]`,
all T steps of the clip's STFT columns, and `encoder_lengths` = the valid steps, P) and the subsampling's output
(`pre_encode_out [T, 512]`, `pre_encode_lengths`), read by forward hooks on audio.encoder and audio.encoder.pre_encode.
The red arm asks aud_04's questions over aud_07's clip (a password reset right away vs thanks for the support), and the
summary goes to results/ref_audio_summary.json (results/ref_<out stem without records_ref_>_summary.json).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _fixtures import fixtures_path as fixtures_for  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

WORK = work_path("_d1_omni")
os.environ.setdefault("HF_HUB_OFFLINE", "1")                               # the snapshot is local and pinned
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_MODULES_CACHE", str(WORK / "hf_modules"))       # trust_remote_code's module copy

import numpy as np  # noqa: E402
import torch  # noqa: E402

import d1_omni_model as dm  # noqa: E402
import host  # noqa: E402

FIXTURES_SHA256 = "c0781185cf9b86210024f94fa645f047af843bfaaf7d80c703c4272634030123"   # round 1: records_ref.json
FIXTURES_IMAGES_SHA256 = "07ea38a2c0f3d8da4afe4dd4ae79e083c46404cb8ab0ba150b9c0a3fca83ae62"  # round 5: the image rows
FIXTURES_AUDIO_SHA256 = "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b"   # round 6: aud_04..aud_15
RED_ARM_IMAGE = ("img_01", "imgm_11")  # --only: img_01's questions over imgm_11's picture (no suitcase)
RED_ARM_AUDIO = ("aud_04", "aud_07")   # --only, audio: aud_04's questions over aud_07's clip (thanks, no request)
NEAR_TIE = 0.02
# npz rows: (record id, question id) for the text rows, record ids for the prefix records (every row).
NPZ_TEXT = [("tv4_000", "answer"), ("semif_FIRST", "decision"), ("own_t01", "several_actions"), ("tv4s_00", "decision"),
            ("tv4_053", "answer"), ("tv4_009", "answer"), ("tv4_028", "answer"),
            ("own_L01", "stoppage"), ("own_L03", "fee_motion"), ("own_L02", "subcontract"),
            ("long_3400", "stoppage"), ("long_3400", "component")]
NPZ_PREFIX = ["card_cats", "card_audio", "aud_01", "aud_02", "aud_03"]
BATCH_SPLIT = ["own_t01", "own_t04", "own_t05", "own_j01", "own_j02", "own_a01", "own_m01", "own_L01", "own_L02",
               "own_L03", "aud_01"]
VISION_RENAME = ("vision.tower.vision_model.", "vision.tower.")
GROUPS = ("tv4x", "tv4s", "tv4", "semif", "own", "long", "img", "aud", "card", "red")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def group_of(record_id: str) -> str:
    return next((g for g in GROUPS if record_id.startswith(g + "_") or record_id.startswith(g)), "other")


def floats(t) -> list[float]:
    return [float(x) for x in torch.as_tensor(t).detach().reshape(-1).tolist()]


# --------------------------------------------------------------------------- hooks
class Taps:
    """Reads the publisher's model without changing it: wrappers on `_run` / `_forward`, forward hooks on the
    trunk, the head and the media towers. `calls` holds one dict per `_forward` (batch)."""

    def __init__(self, model):
        self.model = model
        self.flat, self.cur, self.calls = None, None, []
        self.media: dict[str, list] = {"vision": [], "tower": [], "audio": [], "frontend": [], "pos": [], "encoder": [],
                                       "pre_encode": []}
        orig_run, orig_forward = model._run, model._forward

        def run(rows, *args, **kwargs):
            self.flat = rows
            return orig_run(rows, *args, **kwargs)

        def forward(rows):
            self.cur = {"rows": rows}
            out = orig_forward(rows)
            self.cur["probs"] = out
            self.calls.append(self.cur)
            self.cur = None
            return out

        model._run, model._forward = run, forward
        # the NaFlex position table after the resize: an instance wrapper on the staticmethod the embeddings call as
        # self.resize_positional_embeddings(...) (transformers 5.19 loads the tower without its vision_model level)
        embeddings = model.vision.tower.embeddings
        resize = type(embeddings).resize_positional_embeddings

        def resize_and_keep(positional_embeddings, spatial_shapes, max_length):
            out = resize(positional_embeddings, spatial_shapes, max_length)
            self.media["pos"].append((spatial_shapes.detach().clone(), out.detach().clone()))
            return out

        embeddings.resize_positional_embeddings = resize_and_keep
        self.handles = [
            model.encoder.register_forward_hook(self._trunk),
            model.head.register_forward_hook(self._head),
            model.vision.register_forward_hook(lambda m, a, o: self.media["vision"].append(o.detach().clone())),
            model.vision.tower.register_forward_hook(self._tower, with_kwargs=True),
            model.audio.register_forward_hook(lambda m, a, o: self.media["audio"].append(o.detach().clone())),
            model.audio.frontend.register_forward_hook(
                lambda m, a, o: self.media["frontend"].append((o[0].detach().clone(), o[1].detach().clone()))),
            model.audio.encoder.register_forward_hook(
                lambda m, a, o: self.media["encoder"].append((o[0].detach().clone(), o[1].detach().clone()))),
            model.audio.encoder.pre_encode.register_forward_hook(
                lambda m, a, o: self.media["pre_encode"].append((o[0].detach().clone(), o[1].detach().clone()))),
        ]

    def _trunk(self, module, args, output):
        if self.cur is not None:
            h, pad, prefix = args
            self.cur["trunk_in"] = h.detach().clone()
            self.cur["trunk_pad"], self.cur["trunk_prefix"] = pad.detach().clone(), prefix.detach().clone()
            self.cur["trunk_out"] = output.detach().clone()

    def _head(self, module, args, output):
        if self.cur is not None:
            self.cur["head_in"] = tuple(a.detach().clone() for a in args)
            self.cur["head_out"] = output.detach().clone()

    def _tower(self, module, args, kwargs, output):
        self.media["tower"].append({k: v.detach().clone() for k, v in kwargs.items()}
                                   | {"last_hidden_state": output.last_hidden_state.detach().clone()})

    def reset(self):
        self.flat, self.cur, self.calls = None, None, []
        for v in self.media.values():
            v.clear()


# --------------------------------------------------------------------------- media
def load_media(record: dict) -> tuple[str | None, dict, dict]:
    """(mode, kwargs for probabilities(), info) for the record's media; ('missing', {}, info) when a file is absent."""
    media = record["media"]
    if media is None:
        return None, {}, {}
    prov = record["provenance"]
    if "images" in media:
        from PIL import Image

        paths = [WORK / p for p in media["images"]]
        missing = [str(p) for p in paths if not p.exists()]
        info = {"files": media["images"], "missing": missing}
        if missing:
            return "missing", {}, info
        expected = (prov.get("image") or {}).get("sha256")
        info["sha256"] = [sha256_file(p) for p in paths]
        if expected:
            assert info["sha256"] == [expected], (record["id"], info["sha256"], expected)
        images = [Image.open(p) for p in paths]
        info["px"] = [list(im.size) for im in images]
        info["pil_mode"] = [im.mode for im in images]
        return "image", {"images": images}, info
    import soundfile as sf

    path = WORK / media["audio"]
    if not path.exists():
        return "missing", {}, {"files": [media["audio"]], "missing": [str(path)]}
    audio, rate = sf.read(str(path), dtype="int16")
    info = {"files": [media["audio"]], "sha256": sha256_file(path), "read_dtype": str(audio.dtype),
            "read_shape": list(audio.shape), "sample_rate": rate}
    expected = (prov.get("audio") or {}).get("sha256") or (prov.get("clip") or {}).get("sha256")
    if expected:
        assert info["sha256"] == expected, (record["id"], info["sha256"], expected)
    return "audio", {"audio": audio}, info


def host_prefix_length(mode: str, info: dict) -> int:
    if mode == "image":
        return host.image_prefix_length([tuple(px) for px in info["px"]])
    return host.audio_prefix_length(info["read_shape"][0])


# --------------------------------------------------------------------------- one request
def argmax_key(q: host.Question, probs: list[float]) -> tuple[int, str]:
    best = max(range(len(probs)), key=probs.__getitem__)
    if q.type == "noul":
        return best, ("true" if best == 0 else "false")      # probabilities are [yes, no]
    if q.type == "choice":
        return best, list(q.criteria)[best]
    return best, str(best)


def run_request(model, taps: Taps, tok, record: dict, mode: str, media_kwargs: dict, prefix_expected: int | None):
    """One request through probabilities() (the reference) and system_one() (the response check)."""
    req = record["request"]
    state, qdict = req["state"], req["questions"]
    names = list(qdict)
    taps.reset()
    t0 = time.perf_counter()
    probs = model.probabilities(state, list(qdict.values()), images=media_kwargs.get("images"),
                                audio=media_kwargs.get("audio"))
    seconds = time.perf_counter() - t0
    flat, calls = taps.flat, list(taps.calls)
    media = {k: list(v) for k, v in taps.media.items()}
    assert len(flat) == len(names) == len(probs), (record["id"], len(flat), len(names), len(probs))
    index = {id(row): j for j, row in enumerate(flat)}
    per_q: list[dict] = [None] * len(names)
    for c, call in enumerate(calls):
        rows = call["rows"]
        offsets = call["trunk_prefix"].tolist()
        for b, row in enumerate(rows):
            j = index[id(row)]
            prefix, ids, markers, q, calibrate = row
            p = 0 if prefix is None else int(prefix.shape[1])
            assert offsets[b] == p
            per_q[j] = {"call": c, "b": b, "B": len(rows), "prefix": p, "ids": list(ids), "markers": list(markers),
                        "question": q, "calibrate": calibrate,
                        "logits_full": call["head_out"][b].clone(),
                        "hidden": call["trunk_out"][b, :p + len(ids)].clone(),
                        "probs_returned": list(call["probs"][b])}
    assert all(x is not None for x in per_q)
    for j, x in enumerate(per_q):
        assert x["probs_returned"] == list(probs[j]), (record["id"], j)

    # hook point: the head's input is the trunk output cut per row (what is saved), and the head re-run on it
    # returns the logits bit for bit
    proof = []
    for call in calls:
        rows, out = call["rows"], call["trunk_out"]
        offs, lens = [0 if r[0] is None else int(r[0].shape[1]) for r in rows], [len(r[1]) for r in rows]
        text = torch.nn.utils.rnn.pad_sequence([out[i, o:o + n] for i, (o, n) in enumerate(zip(offs, lens))],
                                               batch_first=True)
        head_in = call["head_in"]
        same_input = torch.equal(text, head_in[0])
        with torch.no_grad():
            again = model.head(text, *head_in[1:])
        proof.append({"B": len(rows), "head_input_is_trunk_rows": same_input,
                      "head_rerun_bit_equal": torch.equal(again, call["head_out"])})
        assert same_input and torch.equal(again, call["head_out"]), (record["id"], mode)

    # host rows from the raw tokenizer: the same ids and markers, the same prefix length
    p_seen = per_q[0]["prefix"]
    if mode != "text":
        assert p_seen == prefix_expected, (record["id"], mode, p_seen, prefix_expected)
    rows_host = host.request_rows(tok, state, qdict, mode, p_seen)
    for j, (r, x) in enumerate(zip(rows_host, per_q)):
        assert r.qid == names[j] and r.ids == x["ids"] and r.markers == x["markers"], (record["id"], mode, names[j])
        assert r.calibrate == x["calibrate"] and r.prefix_len == x["prefix"]

    # system_one: the response body against host.response on the probabilities that run returned
    taps.reset()
    t1 = time.perf_counter()
    resp = model.system_one(state, qdict, media_kwargs.get("images"), media_kwargs.get("audio"))
    seconds_system_one = time.perf_counter() - t1
    flat2, calls2 = taps.flat, list(taps.calls)
    index2 = {id(row): j for j, row in enumerate(flat2)}
    probs2: list = [None] * len(names)
    logits2: list = [None] * len(names)
    for call in calls2:
        for b, row in enumerate(call["rows"]):
            probs2[index2[id(row)]] = list(call["probs"][b])
            logits2[index2[id(row)]] = call["head_out"][b].clone()
    mine = host.response(rows_host, probs2)
    assert mine == resp, (record["id"], mode, mine, resp)
    rerun_bit_equal = all(torch.equal(a, x["logits_full"]) for a, x in zip(logits2, per_q))
    rerun_probs_equal = all(a == list(b) for a, b in zip(probs2, probs))

    questions = []
    for j, (name, x, r) in enumerate(zip(names, per_q, rows_host)):
        q = x["question"]
        k = q.options
        z = x["logits_full"][:k]
        assert torch.all(x["logits_full"][k:] == -1e4)
        T = host.temperature(q) if x["calibrate"] else None
        p_raw = torch.softmax(z, -1)
        p_pub = list(probs[j])
        p_host = host.probabilities_from_logits(z.numpy(), q, x["calibrate"])
        best, key = argmax_key(q, p_pub)
        order = sorted(p_pub, reverse=True)
        margin = float(order[0] - order[1])
        gold = (record.get("gold") or {}).get(name) if mode == (record_mode(record) or "text") else None
        positions = x["prefix"] + len(x["ids"])
        questions.append({
            "qid": name, "type": q.type, "K": k, "calibrate": x["calibrate"],
            "temperature_key": host.temperature_key(q) if x["calibrate"] else None, "T": T,
            "prefix": x["prefix"], "n_ids": len(x["ids"]), "positions": positions, "bucket": host.bucket_for(positions),
            "ids": x["ids"], "markers": x["markers"], "logits_raw": floats(z),
            "probs_raw": floats(p_raw), "probs": [float(v) for v in p_pub],
            "argmax_index": best, "argmax": key, "top2_margin": margin, "near_tie": margin <= NEAR_TIE,
            "gold": gold, "correct": None if gold is None else key == gold,
            "host_softmax_max_abs_dp": float(max(abs(a - b) for a, b in zip(p_host, p_pub))),
            "batch": {"call": x["call"], "index": x["b"], "B": x["B"]},
        })
    entry = {"id": record["id"], "source": record["source"], "public": record["public"], "mode": mode,
             "native": mode == (record_mode(record) or "text"), "prefix": p_seen, "seconds": round(seconds, 4),
             "seconds_system_one": round(seconds_system_one, 4), "batches": [len(c["rows"]) for c in calls],
             "questions": questions, "response": resp, "response_equal_host": True,
             "usage_input_tokens": resp["usage"]["input_tokens"], "hook_proof": proof,
             "second_run_logits_bit_equal": rerun_bit_equal, "second_run_probs_equal": rerun_probs_equal}
    return entry, per_q, media


def record_mode(record: dict) -> str | None:
    media = record["media"]
    if media is None:
        return None
    return "image" if "images" in media else "audio"


# --------------------------------------------------------------------------- npz
def npz_arrays(entry: dict, per_q: list[dict], media: dict, keep: list[int]) -> dict:
    arrays = {"qids": np.array([entry["questions"][j]["qid"] for j in keep]), "mode": np.array(entry["mode"]),
              "prefix_len": np.int32(entry["prefix"])}
    for k, j in enumerate(keep):
        q, x = entry["questions"][j], per_q[j]
        arrays[f"q{k}_hidden"] = x["hidden"].numpy().astype(np.float32)
        arrays[f"q{k}_ids"] = np.asarray(x["ids"], dtype=np.int32)
        arrays[f"q{k}_markers"] = np.asarray(x["markers"], dtype=np.int32)
        arrays[f"q{k}_logits_raw"] = np.asarray(q["logits_raw"], dtype=np.float32)
        arrays[f"q{k}_qtype"] = np.int32(host.QTYPES[q["type"]])
        arrays[f"q{k}_K"] = np.int32(q["K"])
        arrays[f"q{k}_batch"] = np.asarray([q["batch"]["index"], q["batch"]["B"]], dtype=np.int32)
    if entry["mode"] == "image":
        assert len(media["vision"]) == 1 and len(media["tower"]) == 1, (len(media["vision"]), len(media["tower"]))
        arrays["prefix"] = media["vision"][0][0].numpy().astype(np.float32)
        tower = media["tower"][0]
        arrays["pixel_values"] = tower["pixel_values"].numpy().astype(np.float32)
        arrays["spatial_shapes"] = tower["spatial_shapes"].numpy().astype(np.int64)
        arrays["pixel_attention_mask"] = tower["pixel_attention_mask"].numpy().astype(np.int32)
        arrays["tower_last_hidden_state"] = tower["last_hidden_state"].numpy().astype(np.float32)
        if media.get("pos"):  # the image run (round 5): the resized position table per distinct crop shape
            assert len(media["pos"]) == 1, len(media["pos"])
            shapes, table = media["pos"][0]
            assert torch.equal(shapes, tower["spatial_shapes"])
            distinct = sorted({tuple(x) for x in shapes.tolist()})
            arrays["pos_shapes"] = np.asarray(distinct, dtype=np.int64)
            arrays["pos_index"] = np.asarray([distinct.index(tuple(x)) for x in shapes.tolist()], dtype=np.int32)
            first = [shapes.tolist().index(list(d)) for d in distinct]
            arrays["pos_resized"] = table[first].numpy().astype(np.float32)
            for i, j in enumerate(arrays["pos_index"]):  # crops of one shape share one table, bit for bit
                assert torch.equal(table[i], table[first[j]])
    elif entry["mode"] == "audio":
        assert len(media["audio"]) == 1 and len(media["frontend"]) == 1
        arrays["prefix"] = media["audio"][0][0].numpy().astype(np.float32)
        mel, frames = media["frontend"][0]
        arrays["mel"] = mel[0].numpy().astype(np.float32)
        arrays["frames"] = frames.numpy().astype(np.int64)
        if media.get("encoder"):  # the audio run (round 6): the conformer and subsampling outputs, all T steps
            assert len(media["encoder"]) == 1 and len(media["pre_encode"]) == 1
            for key in ("encoder", "pre_encode"):
                x, lengths = media[key][0]
                arrays[f"{key}_out"] = x[0].numpy().astype(np.float32)
                arrays[f"{key}_lengths"] = lengths.numpy().astype(np.int64)
            assert int(arrays["encoder_lengths"][0]) == entry["prefix"] == int(arrays["pre_encode_lengths"][0])
            assert np.array_equal(arrays["encoder_out"].shape, arrays["pre_encode_out"].shape)
    if entry["mode"] != "text":
        for k, j in enumerate(keep):  # the trunk's prefix positions are the tower output, as fed
            assert np.array_equal(arrays[f"q{k}_hidden"].shape, (entry["prefix"] + len(per_q[j]["ids"]), 1024))
        assert arrays["prefix"].shape == (entry["prefix"], 1024)
    return arrays


def npz_head_check(model, path: Path, entry: dict, per_q: list[dict], keep: list[int]) -> list[dict]:
    """Reload the npz and compare it with the run; for a single-question request, feed its saved hidden state to
    model.head directly and require the run's logits bit for bit."""
    out = []
    with np.load(path) as z:
        for k, j in enumerate(keep):
            x = per_q[j]
            hidden = torch.from_numpy(z[f"q{k}_hidden"])
            same = torch.equal(hidden, x["hidden"])
            row = {"qid": entry["questions"][j]["qid"], "saved_equals_run": same, "B": x["B"]}
            if x["B"] == 1:
                p, n = x["prefix"], len(x["ids"])
                K = len(x["markers"])
                with torch.no_grad():
                    logits = model.head(hidden[None, p:p + n], torch.ones(1, n, dtype=torch.bool),
                                        torch.tensor([x["markers"]]), torch.ones(1, K, dtype=torch.bool),
                                        torch.tensor([host.QTYPES[x["question"].type]]))
                row["head_on_saved_hidden_bit_equal"] = torch.equal(logits[0], x["logits_full"])
                assert row["head_on_saved_hidden_bit_equal"], (path, k)
            assert same, (path, k)
            out.append(row)
    return out


# --------------------------------------------------------------------------- summary
def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarize(entries: list[dict], skipped: list[dict], info: dict) -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    gold = defaultdict(lambda: {"gold": 0, "correct": 0, "questions": 0})
    ties = []
    for e in entries:
        for q in e["questions"]:
            counts[f"{e['source']}/{e['mode']}"][q["type"]] += 1
            if e["native"]:
                g = gold[group_of(e["id"])]
                g["questions"] += 1
                if q["gold"] is not None:
                    g["gold"] += 1
                    g["correct"] += bool(q["correct"])
            if q["near_tie"]:
                ties.append(f"{e['id']}/{q['qid']}/{e['mode']} ({q['top2_margin']:.4f})")
    for g in gold.values():
        g["agreement_our_subset"] = round(g["correct"] / g["gold"], 4) if g["gold"] else None
    secs = [e["seconds"] for e in entries]
    by_bucket = defaultdict(list)
    for e in entries:
        for q in e["questions"]:
            by_bucket[str(q["bucket"])].append(e["seconds"] / len(e["questions"]))
    return {
        "requests": len(entries), "rows": sum(len(e["questions"]) for e in entries),
        "records": len({e["id"] for e in entries}), "native_requests": sum(e["native"] for e in entries),
        "skipped": skipped,
        "rows_by_source_mode_type": {k: dict(v) for k, v in sorted(counts.items())},
        "gold_agreement_by_group": dict(sorted(gold.items())),
        "gold_note": "argmax key == the fixture's gold key on the native-mode rows of our fixture subset only; "
                     "not the publisher's benchmark numbers",
        "near_ties": {"threshold_top2": NEAR_TIE, "count": len(ties), "ids": ties},
        "seconds_per_request": {"p50": round(pct(secs, 0.5), 4), "max": round(max(secs), 4),
                                "max_id": max(entries, key=lambda e: e["seconds"])["id"],
                                "total": round(sum(secs), 1)},
        "seconds_per_row_by_bucket": {b: {"rows": len(v), "p50": round(pct(v, 0.5), 4), "max": round(max(v), 4)}
                                      for b, v in sorted(by_bucket.items(), key=lambda kv: int(kv[0]))},
        "threads": info["torch_threads"],
    }


# --------------------------------------------------------------------------- main
def load_model(snapshot: Path) -> tuple:
    import transformers
    from transformers import AutoModel

    t0 = time.perf_counter()
    model, loading = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, dtype=torch.float32,
                                               output_loading_info=True)
    model = model.eval()
    seconds = time.perf_counter() - t0
    loading = {k: sorted(map(str, v)) for k, v in loading.items()}
    assert not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), loading
    from safetensors import safe_open

    # transformers 5.19's Siglip2VisionModel has no `vision_model` level: it loads the checkpoint's
    # vision.tower.vision_model.* under vision.tower.* (a rename, checked here tensor for tensor)
    state = model.state_dict()
    renamed = {}
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            if k.startswith(VISION_RENAME[0]):
                renamed[k] = VISION_RENAME[1] + k[len(VISION_RENAME[0]):]
        mapped = [renamed.get(k, k) for k in keys]
        assert set(mapped) == set(state) and len(mapped) == len(state), sorted(set(mapped) ^ set(state))[:10]
        unequal = [k for k in keys if not torch.equal(f.get_tensor(k), state[renamed.get(k, k)])]
    assert not unequal, unequal[:10]
    dtypes = Counter(str(v.dtype) for v in state.values())
    info = {"class": type(model).__name__, "module": type(model).__module__, "model_dtype": str(model.dtype),
            "state_dtypes": dict(dtypes), "tensors": len(state), "state_equals_safetensors": True,
            "checkpoint_keys_renamed_on_load": {"from": VISION_RENAME[0], "to": VISION_RENAME[1], "tensors": len(renamed)},
            "loading_info": loading, "load_seconds": round(seconds, 2), "transformers": transformers.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
            "training": model.training, "device": str(model.device),
            "temperatures_equal_host": dict(model.config.temperatures) == host.CONFIG["temperatures"],
            "hf_modules_cache": os.environ["HF_MODULES_CACHE"]}
    assert info["temperatures_equal_host"] and info["model_dtype"] == "torch.float32" and not model.training
    return model, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch's default)")
    ap.add_argument("--records", default=None, help="comma-separated record ids: a timing look, nothing written")
    ap.add_argument("--only", nargs="+", default=None,
                    help="record ids or fnmatch patterns: the image run (round 5) on the fixtures' current version")
    ap.add_argument("--out", type=Path, default=None, help="with --only: the reference file to write (a new file)")
    ap.add_argument("--npz-dir", type=Path, default=None,
                    help="with --only: where the npz files go (default ref/npz; staged, moved there at the end)")
    ap.add_argument("--fixtures-sha256", default=None,
                    help="with --only: the fixtures version to read (default round 6's, FIXTURES_AUDIO_SHA256; the "
                         "round 5 image file read FIXTURES_IMAGES_SHA256)")
    args = ap.parse_args()
    if bool(args.only) != bool(args.out):
        ap.error("--only and --out go together")
    started = time.perf_counter()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if args.threads:
        torch.set_num_threads(args.threads)
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    source = {name: sha256_file(snapshot / name) for name in dm.SOURCE_SHA256}
    assert source == dm.SOURCE_SHA256, source
    fixtures_sha256 = (args.fixtures_sha256 or FIXTURES_AUDIO_SHA256) if args.only else FIXTURES_SHA256
    fixtures_path = fixtures_for(fixtures_sha256)  # records.json, or the kept copy of that version (_fixtures.py)
    records = json.loads(fixtures_path.read_text())["records"]
    semif_first = next(r["id"] for r in records if r["id"].startswith("semif_"))
    npz_text = [(semif_first if rid == "semif_FIRST" else rid, qid) for rid, qid in NPZ_TEXT]
    named = {(r["id"], qid) for r in records for qid in r["request"]["questions"]}
    assert all(pair in named for pair in npz_text), [pair for pair in npz_text if pair not in named]
    assert all(any(r["id"] == rid for r in records) for rid in NPZ_PREFIX + BATCH_SPLIT)
    out_path = args.out.resolve() if args.out else WORK / "ref" / "records_ref.json"
    npz_prefix, batch_split = NPZ_PREFIX, BATCH_SPLIT
    if args.only:
        if out_path.exists():
            raise SystemExit(f"{out_path} exists: a reference file is never replaced")
        picked = [r for r in records if any(fnmatch.fnmatchcase(r["id"], pattern) for pattern in args.only)]
        unmatched = [pattern for pattern in args.only if not any(fnmatch.fnmatchcase(r["id"], pattern) for r in picked)]
        assert picked and not unmatched, unmatched
        records = picked
        npz_text = []
        npz_prefix = [r["id"] for r in records if r["media"] is not None]
        batch_split = list(npz_prefix)
        npz_final = (args.npz_dir or WORK / "ref" / "npz").resolve()
        existing = [rid for rid in npz_prefix if (npz_final / f"{rid}.npz").exists()]
        if existing:
            raise SystemExit(f"npz files exist for {existing}: never replaced")
    if args.records:
        want = args.records.split(",")
        records = [r for r in records if r["id"] in want]
        assert len(records) == len(want), want

    model, info = load_model(snapshot)
    tok = host.RawTokenizer(snapshot / "tokenizer.json")
    host.check_token_ids(tok), host.check_token_ids(model.tokenizer)
    info["tokenizer_class"] = type(model.tokenizer).__name__
    print(json.dumps({k: v for k, v in info.items() if k != "loading_info"}), flush=True)
    taps = Taps(model)
    out_ref = WORK / "ref"
    (out_ref / "npz").mkdir(parents=True, exist_ok=True)
    # the image run stages its npz files and moves them in only once everything has passed
    npz_dir = (npz_final / f".staging-{os.getpid()}") if args.only else out_ref / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    entries, skipped, npz_index, npz_checks = [], [], {}, {}
    partial_path = out_path.with_name(out_path.stem + ".partial.jsonl")
    partial = None if args.records else open(partial_path, "w")
    t_loop = time.perf_counter()
    for n, record in enumerate(records):
        media_mode, media_kwargs, media_info = load_media(record)
        modes = ["text"] + ([media_mode] if media_mode else [])
        for mode in modes:
            if mode == "missing":
                skipped.append({"id": record["id"], "mode": record_mode(record), "reason": "media file absent",
                                "missing": media_info["missing"], "rows": len(record["request"]["questions"])})
                continue
            kwargs = media_kwargs if mode != "text" else {}
            expected = host_prefix_length(mode, media_info) if mode != "text" else None
            entry, per_q, media = run_request(model, taps, tok, record, mode, kwargs, expected)
            if mode != "text":
                entry["media"] = media_info
            entries.append(entry)
            files = []  # (npz name, question indices): one file per text row, one per prefix record (all its rows)
            if mode == "text":
                files = [(f"{record['id']}__{q['qid']}", [j]) for j, q in enumerate(entry["questions"])
                         if (record["id"], q["qid"]) in npz_text]
            elif record["id"] in npz_prefix:
                files = [(record["id"], list(range(len(entry["questions"]))))]
            for name, keep in ([] if args.records else files):
                path = npz_dir / f"{name}.npz"
                arrays = npz_arrays(entry, per_q, media, keep)
                save_npz(path, arrays)
                npz_checks[name] = npz_head_check(model, path, entry, per_q, keep)
                npz_index[name] = {"id": record["id"], "mode": mode, "qids": [entry["questions"][j]["qid"] for j in keep],
                                   "rows": len(keep), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                                   "arrays": {k: list(v.shape) for k, v in arrays.items()},
                                   "buckets": [entry["questions"][j]["bucket"] for j in keep]}
            if partial:
                partial.write(json.dumps(entry) + "\n")
                partial.flush()
            slow = entry["seconds"] / len(entry["questions"]) > 5
            if n % 25 == 0 or mode != "text" or slow or args.records:
                print(f"[{n + 1}/{len(records)}] {record['id']} {mode} q={len(entry['questions'])} "
                      f"positions={[q['positions'] for q in entry['questions']]} batches={entry['batches']} "
                      f"{entry['seconds']:.3f}s (system_one {entry['seconds_system_one']:.3f}s)"
                      f"{' SLOW' if slow else ''}", flush=True)
    loop_seconds = time.perf_counter() - t_loop
    if partial:
        partial.close()
    if args.records:
        return 0

    by_key = {(e["id"], e["mode"]): e for e in entries}
    # determinism: record 0 again (the image run: in its media mode), logits bit-equal
    r0 = records[0]
    mode0, kwargs0, info0 = load_media(r0) if args.only else ("text", {}, {})
    mode0 = mode0 or "text"
    e0, _, _ = run_request(model, taps, tok, r0, mode0, kwargs0 if mode0 != "text" else {},
                           host_prefix_length(mode0, info0) if mode0 != "text" else None)
    first = by_key[(r0["id"], mode0)]
    determinism = {"record": r0["id"], "mode": mode0,
                   "logits_bit_equal": all(a["logits_raw"] == b["logits_raw"] for a, b in
                                           zip(e0["questions"], first["questions"])),
                   "probs_equal": all(a["probs"] == b["probs"] for a, b in zip(e0["questions"], first["questions"]))}
    assert determinism["logits_bit_equal"], determinism

    # batch invariance: each question alone vs the request's batch
    split = []
    for rid in batch_split:
        record = next(r for r in records if r["id"] == rid)
        mode, media_kwargs, media_info = load_media(record)
        mode = mode or "text"
        e = by_key[(rid, mode)]
        for j, (name, qd) in enumerate(record["request"]["questions"].items()):
            taps.reset()
            alone = model.probabilities(record["request"]["state"], [qd], images=media_kwargs.get("images"),
                                        audio=media_kwargs.get("audio"))[0]
            ref = e["questions"][j]
            split.append({"id": rid, "qid": name, "mode": mode, "B_in_request": ref["batch"]["B"],
                          "positions": ref["positions"],
                          "max_abs_dp": float(max(abs(a - b) for a, b in zip(alone, ref["probs"]))),
                          "argmax_equal": int(np.argmax(alone)) == ref["argmax_index"]})
    batch_invariance = {"records": batch_split, "rows": len(split),
                        "max_abs_dp": max(s["max_abs_dp"] for s in split),
                        "max_abs_dp_text_rows": max((s["max_abs_dp"] for s in split if s["mode"] == "text"),
                                                    default=None),
                        "argmax_equal": sum(s["argmax_equal"] for s in split),
                        "expected_le": 1e-6, "rows_detail": split}

    if args.only:
        # red arm (media): img_01's questions over imgm_11's picture, or aud_04's over aud_07's clip, must move the
        # answer distributions
        kind = record_mode(records[0])
        own_id, other_id = RED_ARM_IMAGE if kind == "image" else RED_ARM_AUDIO
        own = next(r for r in records if r["id"] == own_id)
        _, other_kwargs, _ = load_media(next(r for r in records if r["id"] == other_id))
        taps.reset()
        swapped = model.probabilities(own["request"]["state"], list(own["request"]["questions"].values()),
                                      images=other_kwargs.get("images"), audio=other_kwargs.get("audio"))
        mine = by_key[(own_id, kind)]["questions"]
        per_q = [float(max(abs(x - y) for x, y in zip(p, q["probs"]))) for p, q in zip(swapped, mine)]
        # the sweep (round 6): every selected media record's questions over the next selected record's media (cyclic)
        # = the donor the end-to-end controls use; how far that moves the answers (recorded, not asserted)
        sweep = []
        media_records = [r for r in records if r["media"] is not None]
        for i, rec in enumerate(media_records):
            donor = media_records[(i + 1) % len(media_records)]
            _, donor_kwargs, _ = load_media(donor)
            taps.reset()
            moved = model.probabilities(rec["request"]["state"], list(rec["request"]["questions"].values()),
                                        images=donor_kwargs.get("images"), audio=donor_kwargs.get("audio"))
            base = by_key[(rec["id"], kind)]["questions"]
            dps = {q["qid"]: float(max(abs(x - y) for x, y in zip(p, q["probs"]))) for p, q in zip(moved, base)}
            flips = [q["qid"] for p, q in zip(moved, base) if int(np.argmax(p)) != q["argmax_index"]]
            sweep.append({"id": rec["id"], "donor": donor["id"], "max_abs_dp": max(dps.values()), "by_question": dps,
                          "argmax_moved": flips})
        sweep_summary = {"rule": "each record's questions over the next selected record's media (cyclic)",
                         "records": len(sweep), "rows": sum(len(s["by_question"]) for s in sweep),
                         "rows_over_0.02": sum(v > 0.02 for s in sweep for v in s["by_question"].values()),
                         "rows_argmax_moved": sum(len(s["argmax_moved"]) for s in sweep),
                         "mean_row_max_abs_dp": float(np.mean([v for s in sweep for v in s["by_question"].values()])),
                         "detail": sweep}
        red_arm = {"pair": [f"{own_id} questions over {other_id}'s {'picture' if kind == 'image' else 'clip'}", own_id],
                   "sweep_next_record": sweep_summary, "max_abs_dp": max(per_q),
                   "max_abs_dp_by_question": dict(zip([q["qid"] for q in mine], per_q)),
                   "probs_red": [list(map(float, p)) for p in swapped], "probs_own": [q["probs"] for q in mine],
                   "expected_gt": 0.02}
    else:
        # red arm: one word changed in tv4_000's state must move the answer distribution
        a, b = by_key[("red_arm_000", "text")]["questions"][0], by_key[("tv4_000", "text")]["questions"][0]
        red_arm = {"pair": ["red_arm_000", "tv4_000"],
                   "max_abs_dp": float(max(abs(x - y) for x, y in zip(a["probs"], b["probs"]))),
                   "probs_red": a["probs"], "probs_tv4_000": b["probs"], "argmax": [a["argmax"], b["argmax"]],
                   "expected_gt": 0.02}

    second_run = {"requests": len(entries), "logits_bit_equal": sum(e["second_run_logits_bit_equal"] for e in entries),
                  "probs_equal": sum(e["second_run_probs_equal"] for e in entries),
                  "differing": [f"{e['id']}/{e['mode']}" for e in entries if not e["second_run_logits_bit_equal"]]}
    host_softmax = max(q["host_softmax_max_abs_dp"] for e in entries for q in e["questions"])
    header = {
        "schema": "d1-omni-reference/1", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference": "AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.float32) "
                     "(modeling_d1.D1OmniModel), CPU fp32, probabilities(state, questions, images, audio) per request",
        "model": {"hf_id": host.MODEL_ID, "revision": host.MODEL_SHA, "snapshot": str(snapshot),
                  "source_sha256": source, "weights_sha256": dm.WEIGHTS["sha256"]},
        "load": info, "fixtures": {"path": str(fixtures_path), "sha256": fixtures_sha256},
        "only": args.only,
        "versions": {"python": platform.python_version(), "platform": platform.platform(),
                     **{m: __import__(m).__version__ for m in ("torch", "transformers", "numpy", "tokenizers",
                                                                "safetensors", "huggingface_hub", "soundfile")},
                     "torchvision": __import__("torchvision").__version__, "pillow": __import__("PIL").__version__},
        "checks": {
            "ids_markers_equal_host_request_rows": sum(len(e["questions"]) for e in entries),
            "response_equal_host_response": len(entries),
            "prefix_length_equal_host": [f"{e['id']}/{e['mode']}: {e['prefix']}" for e in entries if e["mode"] != "text"],
            "hook_point": "every batch: head input == trunk output cut per row (torch.equal) and head re-run on it "
                          "== the run's logits (torch.equal); npz rows reloaded == run; single-question npz rows: "
                          "model.head(saved hidden) == the run's logits (torch.equal)",
            "hook_point_batches": sum(len(e["hook_proof"]) for e in entries),
            "npz_head_checks": npz_checks,
            "determinism": determinism, "second_run_system_one": second_run,
            "host_softmax_max_abs_dp": host_softmax,
            "batch_invariance": batch_invariance, "red_arm": red_arm,
        },
        "npz": npz_index, "loop_seconds": round(loop_seconds, 1),
    }
    doc = {**header, "records": entries}
    if args.only:  # every check above passed: move the staged npz files in (never over an existing file)
        for name in npz_index:
            target = npz_final / f"{name}.npz"
            assert not target.exists(), target
            os.rename(npz_dir / f"{name}.npz", target)
            assert sha256_file(target) == npz_index[name]["sha256"]
        npz_dir.rmdir()
        doc["npz_dir"] = str(npz_final)
    write_atomic(out_path, (json.dumps(doc) + "\n").encode())
    os.remove(partial_path)
    summary = summarize(entries, skipped, info)
    summary.update({
        "records_ref_file": str(out_path), "records_ref_sha256": sha256_file(out_path),
        "load_seconds": info["load_seconds"],
        "loop_seconds": round(loop_seconds, 1), "total_seconds": round(time.perf_counter() - started, 1),
        "determinism": determinism, "second_run_system_one": {k: v for k, v in second_run.items()},
        "batch_invariance": {k: v for k, v in batch_invariance.items() if k != "rows_detail"},
        "red_arm": {k: red_arm[k] for k in ("pair", "max_abs_dp", "argmax", "expected_gt") if k in red_arm} | (
            {"sweep_next_record": {k: v for k, v in red_arm["sweep_next_record"].items() if k != "detail"}}
            if "sweep_next_record" in red_arm else {}),
        "host_softmax_max_abs_dp": host_softmax, "npz_files": len(npz_index),
        "npz_rows": sum(v["rows"] for v in npz_index.values()),
        "npz_bytes": sum(v["bytes"] for v in npz_index.values()),
    })
    summary_path = WORK / "results" / (f"ref_{out_path.stem.removeprefix('records_ref_')}_summary.json" if args.only
                                       else "ref_summary.json")
    if args.only and out_path.parent != (WORK / "ref").resolve():  # a trial elsewhere keeps its summary beside it
        summary_path = out_path.with_name(out_path.stem + "_summary.json")
    if args.only and summary_path.exists():
        raise SystemExit(f"{summary_path} exists: not replaced")
    write_atomic(summary_path, (json.dumps(summary, indent=1) + "\n").encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows_by_source_mode_type", "near_ties")}, indent=1))
    print("near ties:", summary["near_ties"]["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
