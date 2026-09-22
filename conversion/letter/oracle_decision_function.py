# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "mlx-lm==0.31.3",
#   "mlx==0.32.2",
#   "torch==2.9.0",
#   "transformers==5.17.0",
#   "safetensors==0.8.0",
#   "huggingface_hub==1.32.0",
#   "numpy==2.3.5",
#   "tokenizers==0.23.2",
# ]
# ///
"""Build the pinned Jev-Style Qwen3.5-2B decision-function fixtures.

Embeds 22 synthetic requests / 58 rows. The author's unchanged jev_style_mlx.py
produces p_author on the published BF16 MLX checkpoint. Native transformers
Qwen3_5ForCausalLM produces p_oracle on the inverse-converted artifact in CPU FP32.
The latter only remaps model.language_model.X -> model.X: it never repeats the
convolution transpose or normalization offset. Fresh model processes keep MLX
GPU and Torch CPU resources separate. No prompt template, BOS or generation is
used: each prompt ends at Answer:, with space-prefixed letter tokens and T=1.

From the zoo root on Apple silicon:
  uv run --python 3.11 conversion/letter/oracle_decision_function.py \
      --out fixtures-decision.json --work-dir .cache/decision-oracle

--snapshot reuses an already downloaded original snapshot. --converted reuses a
converted directory, with --conversion-report when its receipt is elsewhere.
Otherwise the sibling mlx_to_hf_qwen3_5.convert_snapshot creates it under work-dir.
--author-python and --oracle-python can select separate pinned environments;
their actual versions are recorded. --prepare-only writes inputs without model
execution. The normal path evaluates both oracles and six actual typed APIs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HF_ID = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16"
REVISION = "69a91b6778df7d11b9e13e0f7d4c6a5aa6ad1bc5"
SOURCE_SHA256 = "95fc80a9c0bb7dbecd29896d67b6de1ba3306870dc7018bb4a240c9d63d00649"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
AUTHOR_SHA256 = "fa1d7c8d8571da9344752fc5ddb4f386897001b2becae1ea219c5e2eecfb93f0"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = ("You are a decision function. Read the state, then answer the question "
          "by choosing exactly one option.\n\n")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt(state, question, options):
    lines = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(options))
    return f"{HEADER}[State]\n{state}\n\n[Question]\n{question}\n\n[Options]\n{lines}\n\nAnswer:"


# Embedded deterministic cases; no imports from a conversion-run directory.
# EMBEDDED_REQUESTS_BEGIN
def definitions():
    requests = []

    def add(key, state, decisions, **flags):
        requests.append({"id": key, "state": state, "language": "en", "evidence_only": False,
                         "zoo_only": False, "decisions": decisions, **flags})

    def decision(kind, question, options):
        return {"kind": kind, "question": question, "options": options}

    def triple(key, state, cq, opts, bq, sq, levels):
        add(key, state, [decision("choice", cq, opts), decision("bool", bq, ["yes", "no"]),
                         decision("score", sq, levels)])

    sentiment5 = ["very negative", "negative", "neutral", "positive", "very positive"]
    triple("garden_play", "The small garden performance began late because a curtain jammed. Once it started, the actors spoke clearly, the gentle jokes worked, and the final scene moved the audience. I walked home smiling and would happily attend another performance.",
           "What is the overall sentiment of this review?", ["negative", "positive"],
           "Did the reviewer enjoy the performance overall?", "Rate the review's sentiment on the ordered scale.", sentiment5)
    triple("repair_visit", "A repair worker arrived two hours after the agreed time and left muddy footprints along the hallway. The leaking tap still dripped after the visit. The resident paid the bill but described the service as disappointing and said another repair would be needed.",
           "Which description best matches the resident's opinion?", ["satisfied", "disappointed", "undecided"],
           "Was the leaking tap fully repaired?", "How satisfied was the resident with the service?",
           ["extremely dissatisfied", "mostly dissatisfied", "neither satisfied nor dissatisfied", "mostly satisfied", "extremely satisfied"])
    triple("river_report", "After three days of heavy rain, a river overflowed onto nearby fields. Volunteers moved sacks of grain to dry storage while the local council opened a temporary shelter. No injuries were reported, and the bridge remained closed until an inspection could be completed.",
           "Which topic best describes this report?", ["sports", "weather and flooding", "music", "space research"],
           "Was the bridge open immediately after the flood?", "How much disruption does this report describe?",
           ["none", "minor", "moderate", "major"])
    triple("seed_trial", "A classroom planted two trays with the same number of seeds in the same soil. One tray received water every morning; the other stayed dry. After a week, many seeds in the watered tray sprouted, while none in the dry tray did. Both trays stood beside the same window.",
           "Which condition differed between the two trays?", ["the soil", "the window", "the amount of water", "the number of seeds", "the length of the week"],
           "Did the watered tray contain sprouted seeds?", "How strongly do the observations support that watering helped these seeds sprout?",
           ["no support", "weak support", "moderate support", "strong support", "very strong support"])
    triple("bakery_line", "At a village stall, a baker offered warm bread before sunrise. A short queue formed, and the first batch sold within an hour. Several visitors praised the crisp crust, although one thought it too dark. The baker planned a second batch for the afternoon.",
           "What was being sold at the stall?", ["pottery", "flowers", "books", "bread", "candles", "scarves"],
           "Did the first batch remain unsold all morning?", "How favorable is the reported response to the bread?", sentiment5)
    triple("path_notice", "A notice at a woodland entrance states that the lower path is closed because a fallen tree blocks it. Walkers may use the upper loop, which is clear and returns to the same entrance. The notice asks everyone to avoid the closed path until workers remove the tree.",
           "Which route does the notice allow walkers to use?", ["the closed lower path", "the upper loop", "the river bed", "the workers' yard", "a route through the blocked tree", "a private garden", "a route not mentioned"],
           "Does the upper loop return to the woodland entrance?", "How clear is the notice about which route is available?",
           ["completely unclear", "partly clear", "fully clear"])
    triple("quiet_gallery", "The gallery's new display contains simple clay bowls and a few pencil sketches. The visitor found the arrangement calm and the explanations useful, but no piece felt especially memorable. The visit was pleasant enough for a spare hour, though it did not inspire a return trip.",
           "Which evaluation best summarizes the visit?", ["a terrifying experience", "a thrilling discovery", "a complete disaster", "a useful but ordinary visit", "an exhausting competition", "a confusing argument", "a crowded celebration", "a costly emergency"],
           "Did the visitor describe any displayed piece as especially memorable?", "Rate the visitor's overall enthusiasm.",
           ["none at all", "very low", "low", "somewhat low", "modest", "moderate", "somewhat high", "high", "very high", "exceptional"])
    triple("evening_rehearsal", "A group of amateur musicians met in a hall on a rainy evening. They rehearsed a slow piece twice, stopped to discuss the ending, and then played it again without interruption. Chairs were empty because the session was private, and no audience had been invited.",
           "What was the group doing in the hall?", ["holding an election", "repairing the roof", "selling vegetables", "teaching swimming", "practicing music", "sorting mail", "cooking dinner", "painting a fence", "watching a match"],
           "Was an invited audience watching the rehearsal?", "How certain is it that the group was rehearsing rather than giving a public concert?",
           ["not certain at all", "slightly certain", "moderately certain", "very certain", "fully certain"])
    triple("package_log", "The parcel log records a box arriving at the north desk in the morning. A clerk checked its label, signed the log, and put it in the locked cupboard. The afternoon note says the box was still there and that no collection had taken place that day.",
           "Where was the box at the end of the recorded day?", ["on the delivery cart", "outside the gate", "at the north desk", "inside the locked cupboard", "in the kitchen", "on a garden bench", "under a bridge", "at the repair counter", "inside a mail sack", "in the meeting hall"],
           "Was the box collected during the recorded day?", "How strongly does the record support that the box remained in storage?",
           ["no support", "very weak support", "weak support", "limited support", "some support", "moderate support", "fairly strong support", "strong support", "very strong support", "explicit confirmation"])
    triple("water_check", "A caretaker inspected a drinking fountain after receiving a report of cloudy water. The first sample looked cloudy, but later samples became clear after the pipe was flushed. The caretaker marked the fountain out of use until a laboratory result could confirm that the water was suitable to drink.",
           "Why was the fountain kept out of use?", ["Its color was unfashionable", "It had already been removed", "The laboratory confirmation was still pending", "The garden had closed permanently", "The caretaker had lost every sample", "The pipe could not hold water", "No one had reported a concern", "Its sign had been stolen", "The building had no doors", "A concert was scheduled nearby", "It was reserved for decoration"],
           "Had the laboratory already confirmed suitability when the fountain was marked out of use?", "How cautious was the caretaker's response?",
           ["not cautious", "somewhat cautious", "very cautious"])
    triple("late_bus", "A traveler waited at a rural bus stop through a cold drizzle. The bus arrived forty minutes late, but its driver explained that a fallen branch had blocked the road and apologized. The traveler was relieved to get home, yet called the journey tiring and inconvenient.",
           "Which feeling best describes the traveler's account?", ["pure delight", "mild inconvenience mixed with relief", "complete indifference", "pride in winning", "fear of a sea voyage", "excitement about a promotion", "anger about a missing meal", "curiosity about a painting", "gratitude for a gift", "jealousy of a neighbor", "surprise at a snowfall", "regret over a broken cup"],
           "Did an obstruction on the road contribute to the delay?", "How inconvenient was the journey according to the account?",
           ["not inconvenient", "slightly inconvenient", "moderately inconvenient", "very inconvenient", "extremely inconvenient"])
    triple("shared_shed", "Two neighbors share a shed for garden tools. On Saturday morning, one borrowed the rake and wrote a note promising to return it before dusk. The other later found the note and used a broom instead. By sunset the rake was back on its usual hook inside the shed.",
           "What happened to the rake by sunset?", ["It was sold at a market", "It was left beside a river", "It was broken beyond repair", "It was hidden beneath the floor", "It was returned to its usual hook", "It was given away permanently", "It was buried in a field", "It was painted bright blue", "It was taken to a workshop", "It was missing without any note", "It was stored on a roof", "It was replaced with a shovel", "It was locked in a distant garage"],
           "Was the promise to return the rake before dusk fulfilled?", "How well was the shared-tool arrangement followed in this account?",
           ["not followed", "partly followed", "fully followed"])

    long_options = [
        "Move the dry supplies into the sheltered room",
        "Leave the dry supplies outside beside the gate",
        "Mix the dry supplies with the wet waste",
        "Send every remaining volunteer home before noon",
        "Close the shelter while rain is still falling",
        "Place the paper records in an uncovered cart",
        "Wait until the supplies have become completely soaked",
        "Block the only clear route into the shelter",
        "Discard all of the supplies without checking them",
        "Store the supplies directly beneath the leaking roof",
        "Ask visitors to carry the supplies into the stream",
        "Move the supplies farther from every available building",
        "Replace the protective covers with torn paper sheets",
        "Open all the storage boxes during the heaviest rain",
        "Scatter the supplies across the muddy playing field",
        "Refuse the offer of a clean and empty room",
    ]
    for index, count in enumerate([10, 11, 12, 13, 14, 16]):
        # Rotate so the obvious option occupies a different letter, avoiding an A-only gate.
        choices = long_options[:count]
        shift = index + 1
        choices = choices[-shift:] + choices[:-shift]
        state = (f"At collection station {index + 1}, volunteers have several boxes of dry paper supplies. "
                 "Rain is forecast to begin soon, and the outside table has no cover. An empty room nearby is clean, "
                 "dry, and available for storage. The coordinator wants the supplies to remain usable for the next day's lessons.")
        add(f"supply_station_{index + 1}", state, [
            decision("choice", "Which proposed action best protects the dry supplies from the rain?", choices),
            decision("choice", "What is the coordinator trying to prevent?", ["the paper supplies becoming wet", "the empty room staying dry", "the lessons receiving usable supplies"]),
            decision("choice", "Which detail makes sheltered storage possible?", ["The table has no cover", "An empty dry room is available", "Rain is forecast soon", "The supplies are made of paper"]),
        ])

    add("spanish_notice", "El aviso de la biblioteca dice que la sala de lectura abrirá después del mediodía porque el suelo se está limpiando. Los libros se pueden devolver en el buzón exterior durante toda la mañana. Una persona llega temprano con dos libros que necesita devolver y lee el aviso junto a la puerta.",
        [decision("choice", "¿Dónde puede devolver los libros durante la mañana?", ["en el buzón exterior", "en la sala cerrada", "en el jardín privado"])], language="es", evidence_only=True)
    add("spanish_review", "La visitante esperó unos minutos antes de entrar en el pequeño teatro. La función le pareció cálida y divertida, y comentó que los actores hablaban con claridad. Aunque el asiento era un poco duro, salió contenta y dijo que volvería para ver otra obra la próxima semana.",
        [decision("choice", "¿Cuál es el sentimiento general de la visitante?", ["negativo", "positivo", "neutral"])], language="es", evidence_only=True)

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "white", "black", "gray", "brown", "pink", "silver", "gold", "striped", "plain", "wooden"]
    rooms = ["east room", "west room", "upper room", "courtyard", "north room", "south room", "small room", "large room", "front room", "rear room", "round room", "square room", "lower room", "corner room", "middle room", "side room"]
    add("long_store_log", "The storekeeper checked the map containers after moving several unrelated boxes of blank paper. The floor remained dry and the labels were easy to read. The final entry says that all dry maps are in the green container; every other container is empty. This final entry replaces earlier provisional arrangements.",
        [decision("choice", "Which option correctly states where the dry maps are stored according to the final entry?", [f"The dry maps are stored in the {color} container." for color in colors])], zoo_only=True)
    add("long_room_log", "The hall keeper inspected the chairs, checked the curtains, and then issued a final room assignment. The final notice says the group must use the west room. Every other room is closed for cleaning, and the courtyard is wet. Earlier suggestions were provisional and should not determine the group's destination.",
        [decision("choice", "Which option gives the group's assigned destination according to the final notice?", [f"The group is assigned to use the {room}." for room in rooms])], zoo_only=True)
    return requests


def build_fixtures(tok):
    requests = definitions()
    rows = []
    all_label_ids = []
    for c in LETTERS:
        encoded = tok.encode(" " + c, add_special_tokens=False)
        assert len(encoded) == 1, (c, encoded)
        assert tok.decode(encoded) == " " + c, (c, tok.decode(encoded))
        all_label_ids.append(encoded[0])
    for req in requests:
        if req["zoo_only"]:
            decision = req["decisions"][0]
            filler = (" This option refers to the location named in its first sentence and makes no additional claim about "
                      "earlier plans, unrelated supplies, or the order of the routine inspection.")
            counter = 0
            while True:
                options = decision["options"].copy()
                options[counter % len(options)] += filler
                n = len(tok.encode(prompt(req["state"], decision["question"], options), add_special_tokens=False))
                if n > 1750:
                    break
                decision["options"] = options
                counter += 1
        state_tokens = len(tok.encode(req["state"], add_special_tokens=False))
        assert 30 <= state_tokens <= 500, (req["id"], state_tokens)
        req["state_tokens"] = state_tokens
        req["row_ids"] = []
        for j, dec in enumerate(req["decisions"]):
            rendered = prompt(req["state"], dec["question"], dec["options"])
            ids = tok.encode(rendered, add_special_tokens=False)
            assert rendered.endswith("Answer:")
            assert tok.decode(ids).endswith("Answer:")
            if req["zoo_only"]:
                assert 1500 <= len(ids) <= 2000
            else:
                assert len(ids) <= 1024
            count = len(dec["options"])
            assert 2 <= count <= (10 if dec["kind"] == "score" else 16)
            row_id = f"{req['id']}__{j + 1}_{dec['kind']}"
            req["row_ids"].append(row_id)
            rows.append({"id": row_id, "request_id": req["id"], "kind": dec["kind"],
                         "primitive": dec["kind"], "state": req["state"], "question": dec["question"],
                         "options": dec["options"], "prompt": rendered, "ids": ids, "slot": len(ids) - 1,
                         "label_ids": all_label_ids[:count], "labels": [" " + c for c in LETTERS[:count]],
                         "nopts": count, "tokens": len(ids), "state_tokens": state_tokens,
                         "language": req["language"], "evidence_only": req["evidence_only"],
                         "zoo_only": req["zoo_only"], "label_boundary_valid": True})
    counts = Counter(row["kind"] for row in rows)
    wide = [r["id"] for r in rows if r["kind"] == "choice" and 10 <= r["nopts"] <= 16]
    descriptive = [r["id"] for r in rows if r["kind"] == "choice" and all(len(o.split()) > 5 for o in r["options"])]
    score10 = [r["id"] for r in rows if r["kind"] == "score" and r["nopts"] == 10]
    assert len(requests) >= 18 and len(rows) >= 50
    assert counts["choice"] >= 26 and counts["bool"] >= 12 and counts["score"] >= 10
    assert len(wide) >= 6 and len(descriptive) >= 6 and len(score10) >= 2
    assert sum(r["evidence_only"] for r in rows) == 2
    assert sum(r["zoo_only"] for r in rows) == 2
    assert all(r["options"] == ["yes", "no"] for r in rows if r["kind"] == "bool")
    summary = {"requests": len(requests), "total_rows": len(rows), "by_kind": dict(counts),
               "choice_10_to_16_options": len(wide), "choice_descriptions_over_five_words": len(descriptive),
               "score_10_levels": len(score10), "evidence_only_rows": 2, "zoo_only_rows": 2,
               "kit_token_min": min(r["tokens"] for r in rows if not r["zoo_only"]),
               "kit_token_max": max(r["tokens"] for r in rows if not r["zoo_only"]),
               "ordinary_state_tokens_min": min(r["state_tokens"] for r in rows if not r["zoo_only"]),
               "ordinary_state_tokens_max": max(r["state_tokens"] for r in rows if not r["zoo_only"]),
               "zoo_only_token_lengths": [r["tokens"] for r in rows if r["zoo_only"]],
               "total_tokens": sum(r["tokens"] for r in rows), "coverage_status": "PASS",
               "wide_choice_ids": wide, "long_description_choice_ids": descriptive, "score_10_ids": score10,
               "constructed_examples": "Deterministic synthetic states; no real persons, companies, brands or products."}
    # Six separate representative requests, two API calls per primitive.
    api_checks = {"garden_play__1_choice": "decide", "supply_station_6__1_choice": "decide",
                  "river_report__2_bool": "decide_bool", "package_log__2_bool": "decide_bool",
                  "quiet_gallery__3_score": "decide_score", "late_bus__3_score": "decide_score"}
    return {"schema": "coreai-letter-fixtures/1", "source": {"hf_id": HF_ID, "revision": REVISION},
            "letters": list(LETTERS), "all_label_ids": all_label_ids, "label_style": "space-prefixed",
            "temperature": 1.0, "header": HEADER, "requests": requests, "rows": rows,
            "api_checks": api_checks, "summary": summary}

# EMBEDDED_REQUESTS_END


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("oracle wall-clock deadline reached")


def arm_deadline(deadline):
    check_deadline(deadline)
    signal.alarm(max(1, math.ceil(deadline - time.time())))


def environment():
    return {"python": sys.version, "interpreter": sys.executable,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "environment": {key: os.environ.get(key) for key in (
                "HF_HOME", "HF_HUB_DISABLE_XET", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")},
            "timing": "correctness only; may be contended"}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_worker(inputs, work, deadline):
    import mlx.core as mx
    source = Path(inputs["snapshot"])
    author_file = source / "jev_style_mlx.py"
    assert sha256(author_file) == AUTHOR_SHA256
    upstream = import_file("unchanged_jev_style_mlx", author_file)
    assert upstream.HEADER == HEADER and upstream.LETTERS == LETTERS
    record = {"status": "RUNNING", "source_file": str(author_file), "source_sha256": AUTHOR_SHA256,
              "source_unchanged": True, "environment": environment(), "device": "Apple silicon MLX GPU",
              "dtype": "BF16 weights/projection, FP32 softmax, A_log remains FP32", "temperature": 1.0,
              "rows": [], "api_checks": []}
    output = work / "author.json"
    atomic_json(output, record)
    try:
        started = time.monotonic()
        jev = upstream.JevStyle(str(source))
        record["load_seconds"] = time.monotonic() - started
        inner = jev.model.language_model.model
        assert inner.embed_tokens.weight.dtype == inner.norm.weight.dtype == mx.bfloat16
        assert inner.layers[0].linear_attn.A_log.dtype == mx.float32
        assert list(inner.embed_tokens.weight.shape) == [248320, 2048]
        for index, row in enumerate(inputs["rows"]):
            check_deadline(deadline)
            start = time.monotonic()
            assert jev.tok.encode(row["prompt"], add_special_tokens=False) == row["ids"]
            assert jev.label_ids[:row["nopts"]].tolist() == row["label_ids"]
            last = inner(mx.array([row["ids"]]))[0, -1]
            scores = inner.embed_tokens.weight[mx.array(row["label_ids"])] @ last
            assert scores.dtype == mx.bfloat16
            probs = mx.softmax(scores.astype(mx.float32))
            mx.eval(scores, probs, last)
            assert bool(mx.all(mx.isfinite(scores)).item()) and bool(mx.all(mx.isfinite(last)).item())
            p, raw = probs.tolist(), scores.astype(mx.float32).tolist()
            result = {"id": row["id"], "p_author": p, "author_raw_logits": raw,
                      "author_argmax": int(mx.argmax(probs).item()), "finite": True,
                      "last_hidden_abs_max": float(mx.max(mx.abs(last)).item()),
                      "raw_logit_min": min(raw), "raw_logit_max": max(raw),
                      "raw_logit_abs_max": max(abs(v) for v in raw), "wall_seconds": time.monotonic()-start}
            if row["id"] in inputs["api_checks"]:
                method = inputs["api_checks"][row["id"]]
                if method == "decide":
                    actual = jev.decide(row["state"], row["question"], row["options"])
                    expected = sorted(zip(row["options"], p), key=lambda pair: -pair[1])
                elif method == "decide_bool":
                    actual = jev.decide_bool(row["state"], row["question"])
                    expected = p[0]
                else:
                    actual = jev.decide_score(row["state"], row["question"], row["options"])
                    expected = (sum(i*v for i, v in enumerate(p)), list(zip(row["options"], p)))
                assert actual == expected, (row["id"], actual, expected)
                record["api_checks"].append({"row_id": row["id"], "request_id": row["request_id"],
                                             "method": method, "api_result": actual,
                                             "independent_row_result": expected, "exactly_equal": True})
            record["rows"].append(result)
            atomic_json(output, record)
            print(f"AUTHOR {index+1}/{len(inputs['rows'])} {row['id']} argmax={result['author_argmax']}", flush=True)
        assert len(record["api_checks"]) == 6 and len({c["request_id"] for c in record["api_checks"]}) == 6
        record.update(status="PASS", completed_rows=len(record["rows"]))
        atomic_json(output, record)
    except Exception as exc:
        record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        atomic_json(output, record)
        raise


def torch_worker(inputs, work, deadline, threads):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    assert torch.__version__ == "2.9.0" and transformers.__version__ == "5.17.0"
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    converted = Path(inputs["converted"])
    conversion = json.loads(Path(inputs["conversion_report"]).read_text())
    assert conversion["status"] == "PASS"
    assert sha256(converted / "model.safetensors") == conversion["output_sha256"]
    assert sha256(converted / "config.json") == conversion["config_sha256"]
    tok = AutoTokenizer.from_pretrained(inputs["snapshot"], local_files_only=True, trust_remote_code=False)
    assert build_fixtures(tok)["rows"] == inputs["rows"]
    record = {"status": "RUNNING", "environment": environment(), "device": "CPU", "dtype": "float32",
              "temperature": 1.0, "rows": [], "converted_artifact": {
                  "file": str(converted / "model.safetensors"), "sha256": conversion["output_sha256"],
                  "config": str(converted / "config.json"), "config_sha256": conversion["config_sha256"],
                  "conversion_evidence": inputs["conversion_report"]},
              "method": "Native Qwen3_5ForCausalLM on CPU FP32, actual converted artifact, strict loading with tied head. Only key prefix remap; no repeated inverse transforms. Fresh prefill/use_cache=False; last-position full-head logits restricted to space-letter IDs; T=1."}
    output = work / "torch.json"
    atomic_json(output, record)
    try:
        start = time.monotonic()
        cfg = json.loads((converted / "config.json").read_text())
        assert cfg["model_type"] == "qwen3_5_text" and cfg["tie_word_embeddings"]
        config = Qwen3_5TextConfig.from_dict(cfg)
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).float().cpu().eval()
        mapped, dtypes, shapes = {}, Counter(), []
        with safe_open(converted / "model.safetensors", framework="pt", device="cpu") as handle:
            for key in handle.keys():
                assert key.startswith("model.language_model.")
                target = "model." + key.removeprefix("model.language_model.")
                original = handle.get_tensor(key)
                dtypes[str(original.dtype)] += 1
                value = original.float()
                assert tuple(value.shape) == tuple(model.state_dict()[target].shape), key
                if key.endswith(".conv1d.weight"):
                    assert tuple(value.shape) == (6144, 1, 4) and original.dtype == torch.float32
                shapes.append({"stored_key": key, "loaded_key": target, "shape": list(value.shape),
                               "stored_dtype": str(original.dtype), "loaded_dtype": str(value.dtype)})
                mapped[target] = value
        assert len(mapped) == 320 and dtypes == {"torch.float32": 97, "torch.bfloat16": 223}
        mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"]
        loaded = model.load_state_dict(mapped, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        del mapped, original, value
        model.tie_weights()
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
        record["load_seconds"] = time.monotonic()-start
        record["load_state_dict"] = {"strict": True, "missing_keys": [], "unexpected_keys": [], "tied_head": True}
        atomic_json(work / "torch-layout.json", {"status": "PASS", "converted_artifact": record["converted_artifact"],
                    "tensor_layout_changes_in_oracle": [], "normalization_offsets_in_oracle": [],
                    "stored_dtype_counts": dict(dtypes), "tensors": shapes, "load_state_dict": record["load_state_dict"]})
        with torch.inference_mode():
            for index, row in enumerate(inputs["rows"]):
                check_deadline(deadline)
                start = time.monotonic()
                full = model(input_ids=torch.tensor([row["ids"]], dtype=torch.long), use_cache=False, logits_to_keep=1).logits
                assert tuple(full.shape) == (1, 1, 248320) and full.dtype == torch.float32 and bool(torch.isfinite(full).all())
                raw = full[0, -1].index_select(0, torch.tensor(row["label_ids"], dtype=torch.long))
                probs = raw.softmax(-1)
                ordered = torch.sort(probs, descending=True).values
                argmax, full_id = int(probs.argmax()), int(full[0, -1].argmax())
                result = {"id": row["id"], "p_oracle": probs.tolist(), "raw_logits": raw.tolist(),
                          "argmax": argmax, "top2_margin": float(ordered[0]-ordered[1]),
                          "argmax_label": row["labels"][argmax], "chosen_option": row["options"][argmax],
                          "fp32_full_vocab_argmax_id": full_id, "fp32_full_vocab_argmax_text": tok.decode([full_id]),
                          "fp32_full_vocab_argmax_is_label": full_id in row["label_ids"], "finite": True,
                          "raw_logit_min": float(raw.min()), "raw_logit_max": float(raw.max()),
                          "raw_logit_abs_max": float(raw.abs().max()), "full_vocab_logit_min": float(full.min()),
                          "full_vocab_logit_max": float(full.max()), "probability_sum": float(probs.sum()),
                          "wall_seconds": time.monotonic()-start}
                if row["kind"] == "bool":
                    result["yes_probability"] = float(probs[0])
                elif row["kind"] == "score":
                    result["expected_index"] = sum(i*p for i, p in enumerate(probs.tolist()))
                record["rows"].append(result)
                atomic_json(output, record)
                print(f"TORCH {index+1}/{len(inputs['rows'])} {row['id']} argmax={argmax} margin={result['top2_margin']:.7f}", flush=True)
        record.update(status="PASS", completed_rows=len(record["rows"]))
        atomic_json(output, record)
    except Exception as exc:
        record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        atomic_json(output, record)
        raise


def assemble(inputs, work, out):
    author = json.loads((work / "author.json").read_text())
    oracle = json.loads((work / "torch.json").read_text())
    assert author["status"] == oracle["status"] == "PASS"
    aa, bb = {r["id"]: r for r in author["rows"]}, {r["id"]: r for r in oracle["rows"]}
    assert set(aa) == set(bb) == {r["id"] for r in inputs["rows"]}
    result = {k: v for k, v in inputs.items() if k not in ("rows", "api_checks", "deadline_utc")}
    result["source"]["oracle"] = {
        "author": {"implementation": "pinned unchanged jev_style_mlx.JevStyle", "dtype": author["dtype"],
                   "device": author["device"], "source_sha256": AUTHOR_SHA256, "evidence": str(work / "author.json")},
        "fp32": {"implementation": "transformers5.17.0 native Qwen3_5ForCausalLM text tower", "dtype": "float32",
                 "device": "CPU", "evidence": str(work / "torch.json"), "layout_evidence": str(work / "torch-layout.json"),
                 "converted_artifact": oracle["converted_artifact"]}}
    result["api_checks"] = author["api_checks"]
    result["rows"] = []
    for row in inputs["rows"]:
        a, b = aa[row["id"]], bb[row["id"]]
        delta = [abs(x-y) for x, y in zip(a["p_author"], b["p_oracle"])]
        merged = {**row, **b, "p_author": a["p_author"], "author_raw_logits": a["author_raw_logits"],
                  "author_argmax": a["author_argmax"], "author_vs_oracle_max_abs_delta_p": max(delta),
                  "author_vs_oracle_mean_abs_delta_p": sum(delta)/len(delta),
                  "author_vs_oracle_argmax_equal": a["author_argmax"] == b["argmax"],
                  "author_chosen_option": row["options"][a["author_argmax"]],
                  "author_numeric": {k: a[k] for k in ("finite", "last_hidden_abs_max", "raw_logit_min", "raw_logit_max", "raw_logit_abs_max")},
                  "wall_seconds": {"author": a["wall_seconds"], "fp32_cpu": b["wall_seconds"]}}
        if row["kind"] == "bool":
            merged["author_yes_probability"] = a["p_author"][0]
        elif row["kind"] == "score":
            merged["author_expected_index"] = sum(i*p for i, p in enumerate(a["p_author"]))
        result["rows"].append(merged)
    rows = result["rows"]
    eligible = [r for r in rows if r["top2_margin"] >= 0.02]
    failed = [r["id"] for r in eligible if not r["author_vs_oracle_argmax_equal"]]
    disagreements = [{"id": r["id"], "author_argmax": r["author_argmax"], "fp32_argmax": r["argmax"],
                      "fp32_margin": r["top2_margin"], "max_abs_delta_p": r["author_vs_oracle_max_abs_delta_p"]}
                     for r in rows if not r["author_vs_oracle_argmax_equal"]]
    result["status"] = result["result"] = "FAIL" if failed else "PASS"
    gate = {"status": result["status"], "rule": "A/B argmax equality on every row with B top2 probability margin >=0.02",
            "eligible_rows": len(eligible), "matching_eligible_rows": len(eligible)-len(failed),
            "eligible_argmax_agreement": (len(eligible)-len(failed))/len(eligible), "failed_eligible_rows": failed,
            "probability_error_policy": "Record max and mean deltas; no extra hard probability tolerance."}
    result["inverse_conversion_gate"] = gate
    result["summary"].update({"completed_rows": len(rows), "author_api_checks": len(author["api_checks"]),
        "author_api_exact_equal": all(c["exactly_equal"] for c in author["api_checks"]), "both_oracles_all_rows": True,
        "oracle_argmax_matches": len(rows)-len(disagreements), "oracle_argmax_agreement": (len(rows)-len(disagreements))/len(rows),
        "oracle_disagreements": disagreements, "max_abs_p_author_minus_p_oracle": max(r["author_vs_oracle_max_abs_delta_p"] for r in rows),
        "mean_of_row_mean_abs_p_author_minus_p_oracle": sum(r["author_vs_oracle_mean_abs_delta_p"] for r in rows)/len(rows),
        "fp32_near_tie_rows_margin_lt_0_02": [r["id"] for r in rows if r["top2_margin"] < 0.02],
        "all_finite": all(r["finite"] and r["author_numeric"]["finite"] for r in rows), "inverse_conversion_gate": gate,
        "worst5_author_fp32": [{"id": r["id"], "max_abs_delta_p": r["author_vs_oracle_max_abs_delta_p"], "fp32_margin": r["top2_margin"]}
                               for r in sorted(rows, key=lambda r: -r["author_vs_oracle_max_abs_delta_p"])[:5]]})
    atomic_json(out, result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    if failed:
        raise RuntimeError(f"inverse conversion gate failed on {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--converted", type=Path)
    ap.add_argument("--conversion-report", type=Path)
    ap.add_argument("--hf-home", type=Path)
    ap.add_argument("--author-python", default=sys.executable)
    ap.add_argument("--oracle-python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=10800)
    ap.add_argument("--deadline-utc", help="Optional ISO UTC deadline, used by supervised runs")
    ap.add_argument("--worker", choices=("author", "torch"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = args.out.resolve()
    work = (args.work_dir or out.parent / ".decision-function-work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    deadline_utc = args.deadline_utc or (datetime.now(timezone.utc)+timedelta(seconds=args.timeout)).isoformat()
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00")).timestamp()
    arm_deadline(deadline)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HOME"] = str((args.hf_home or Path(os.environ.get("HF_HOME", work / "hf"))).resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    if args.worker:
        inputs = json.loads((work / "inputs.json").read_text())
        if args.worker == "author":
            author_worker(inputs, work, deadline)
        else:
            torch_worker(inputs, work, deadline, args.threads)
        return
    assert args.hf_id == HF_ID and args.revision == REVISION, "This fixture builder and source hashes are pinned to the documented model revision."
    if args.snapshot:
        source = args.snapshot.resolve()
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(args.hf_id, revision=args.revision, max_workers=2))
    source_hashes = {}
    for name, expected in (("model.safetensors", SOURCE_SHA256), ("tokenizer.json", TOKENIZER_SHA256), ("jev_style_mlx.py", AUTHOR_SHA256)):
        actual = sha256(source / name)
        assert actual == expected, f"source integrity mismatch: {name}"
        source_hashes[name] = actual
    atomic_json(work / "source-integrity.json", {"status": "PASS", "hf_id": args.hf_id,
                "revision": args.revision, "snapshot": str(source), "sha256": source_hashes})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    inputs = build_fixtures(tok)
    inputs.update(snapshot=str(source), deadline_utc=deadline_utc)
    if args.prepare_only:
        inputs["status"] = "PREPARED"
        atomic_json(out, inputs)
        atomic_json(work / "inputs.json", inputs)
        print(json.dumps(inputs["summary"], indent=2))
        return
    if args.converted:
        converted = args.converted.resolve()
        receipt = (args.conversion_report or converted / "mlx_to_hf.json").resolve()
    else:
        tool = Path(__file__).resolve().parents[1] / "mlx_to_hf_qwen3_5.py"
        converter = import_file("mlx_to_hf_qwen3_5", tool)
        converted = work / "converted"
        conversion = converter.convert_snapshot(source, converted)
        assert conversion["status"] == "PASS"
        receipt = converted / "mlx_to_hf.json"
    conversion = json.loads(receipt.read_text())
    assert conversion["status"] == "PASS"
    inputs.update(converted=str(converted), conversion_report=str(receipt))
    atomic_json(work / "inputs.json", inputs)
    atomic_json(out, {"status": "RUNNING", "source": inputs["source"], "work_dir": str(work)})
    commands = []
    for worker, python in (("author", args.author_python), ("torch", args.oracle_python)):
        check_deadline(deadline)
        # Preserve a venv's executable symlink: resolving it would discard the
        # venv and launch its bare base interpreter without installed packages.
        command = [os.path.abspath(os.path.expanduser(python)), "-B", str(Path(__file__).resolve()), "--worker", worker,
                   "--out", str(out), "--work-dir", str(work), "--deadline-utc", deadline_utc,
                   "--threads", str(args.threads)]
        start = time.monotonic()
        log_path = work / f"{worker}.log"
        with log_path.open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            entry = {"worker": worker, "command": command, "pid": child.pid, "log": str(log_path)}
            commands.append(entry)
            atomic_json(work / "commands.json", commands)
            entry.update(returncode=child.wait(), wall_seconds=time.monotonic()-start)
        atomic_json(work / "commands.json", commands)
        if entry["returncode"]:
            atomic_json(out, {"status": "FAIL", "worker": worker, "returncode": entry["returncode"], "log": str(log_path)})
            raise RuntimeError(f"{worker} worker failed; see {log_path}")
    assemble(inputs, work, out)


if __name__ == "__main__":
    main()
