# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gliner2==2.0.0",
#     "torch==2.12.1",
#     "transformers==4.57.6",
#     "tokenizers==0.22.2",
#     "safetensors==0.8.0",
#     "sentencepiece==0.2.2",
#     "numpy==2.5.3",
#     "huggingface_hub==0.36.2",
#     "peft==0.21.0",        # gliner2 2.0.0 imports peft unconditionally (gliner2/training/trainer.py)
#     "accelerate==1.15.0",
# ]
# ///
# GLiNER2.5-Decide (fastino/GLiNER2.5-Decide) — fp32 oracle + fixture builder for the Core AI port.
#
# The oracle is the official gliner2 2.0.0 package on CPU in fp32, one case per forward (no batch
# padding). For every case it saves what the Core AI graph and the host need to be gated against:
#   text (after gliner2's own "." append), tasks, input_ids, attention_mask, schema_tokens_list,
#   subword_list, schema_special_indices ([P, L1, L2, ...] per task), per-task fp32 logits, probs,
#   the decision under gliner2's rule, and model.classify_text(...)'s real output.
# The oracle path is the one classify_text runs internally:
#   model.encoder(input_ids, attention_mask).last_hidden_state
#   -> processor.extract_embeddings_from_batch -> classifier(stack(embs)[1:]).squeeze(-1)
# and every case asserts that its decision equals classify_text's and that the probabilities agree to
# < 1e-5 (the oracle's self-consistency gate).
#
# Outputs (under --work-dir):
#   fixtures/readme21.json            every classify_text example of the model card (rev-pinned), verbatim
#   fixtures/fast_decisions_s256.json fastino/fast-decisions: first --per-domain rows per domain (file order)
#                                     whose collated input_ids fit in S and whose label count fits in MMAX
#   fixtures/fast_decisions_long.json with -S 512 --min-len 257 --per-domain 10: up to 10 rows per domain whose
#                                     collated length is in [257, 512] (the S=512 shape's own range)
#   results/lengths.json              collated length + label count of all rows (S=256 / S=512 coverage)
#   results/own_subset_accuracy.json  exact match vs gold on the fixture subset (dev split; NOT the
#                                     publisher's held-out benchmark)
#
# Run with the private oracle venv (the shared zoo venv has gliner2 1.3.2, which cannot read Decide):
#   HF_HOME=<cache> HF_HUB_OFFLINE=1 <venv-oracle>/bin/python conversion/gliner25_decide_oracle.py
import argparse
import ast
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path

from _paths import work_path

MODEL_ID = "fastino/GLiNER2.5-Decide"
MODEL_REV = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
MODEL_SHA256 = "40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997"
DATASET_ID = "fastino/fast-decisions"
DATASET_REV = "1a33070cabf94ce2e29105482dd2ef6c157ad7f2"
# Dataset card order (the `configs:` list of the fast-decisions README).
DOMAINS = [
    "support_intent", "support_topic", "document_type", "review_sentiment", "agent_handoff",
    "email_triage", "ticket_route", "product_feedback", "banking_intent", "clinic_request",
    "travel_request", "news_topic", "paper_field", "sports_recap", "restaurant_review",
    "benefits_request", "screen_tags",
]
MARKERS = ["[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]",
           "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]", "[PAD]"]

# Every classify_text example of the model card at MODEL_REV, in card order, verbatim:
# (title, text, tasks, "Potential output"). check_readme() re-parses the card and asserts equality.
README_EXAMPLES = [
    ("Customer support intent",
     "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?",
     {"intent": ["order_status", "refund_request", "cancel_subscription", "update_payment",
                 "login_problem", "shipping_delay", "bug_report", "speak_to_human", "other"]},
     {"intent": "refund_request"}),
    ("Banking request",
     "The transfer I sent this morning is still pending, and I think I used the wrong sort code. Can you stop it and add Emily as the beneficiary instead?",
     {"intent": ["transfer_pending", "transfer_cancel", "beneficiary_add", "card_lost",
                 "balance_inquiry", "fraud_report", "mortgage_application", "fee_explanation"]},
     {"intent": "transfer_cancel"}),
    ("Travel request",
     "I need to move my Friday flight to Paris to Saturday morning, same cabin, and keep the aisle seat if you can.",
     {"request": ["book", "change", "cancel", "status", "seat_change", "refund", "baggage"]},
     {"request": "change"}),
    ("Clinic request",
     "The rash came back after the antibiotics finished. Can I get a same-week appointment with dermatology, or should I just refill the cream?",
     {"request": ["book_appointment", "refill_prescription", "test_results",
                  "referral", "billing_question", "cancel_appointment"]},
     {"request": "book_appointment"}),
    ("Review sentiment",
     "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
     {"sentiment": ["positive", "negative", "mixed", "neutral"]},
     {"sentiment": "mixed"}),
    ("Product aspects",
     "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
     {"aspects": {"labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
                  "multi_label": True, "cls_threshold": 0.4}},
     {"aspects": ["battery", "keyboard", "screen"]}),
    ("News topic",
     "The central bank held rates and said inflation is still above target, pushing bank stocks lower in afternoon trading.",
     {"topic": ["politics", "business", "sports", "science", "entertainment", "world"]},
     {"topic": "business"}),
    ("Document type",
     "INVOICE 1842\nBill to: Northstar QA\nAmount due: 2,400 USD\nDue: 30 April 2026\nWire instructions are on page 2.",
     {"document_type": ["invoice", "receipt", "contract", "resume", "support_email", "meeting_notes"]},
     {"document_type": "invoice"}),
    ("Email triage",
     "From: compliance@group.example\nSubject: Protocol update — action required today\n\nPlease confirm the new retention rule is applied before Friday's audit.",
     {"intent": ["fyi", "request", "approval", "complaint", "newsletter", "security_alert"],
      "urgency": ["low", "normal", "high", "critical"],
      "route": ["support", "billing", "legal", "security", "finance", "archive"]},
     {"intent": "request", "urgency": "high", "route": "legal"}),
    ("Ticket routing",
     "[subject] 401k deduction missing from this paystub\n[body] Last month's contribution posted. This month the line is gone and HR told me to open a ticket.",
     {"queue": ["payroll", "benefits", "it_access", "facilities",
                "expense_reimbursement", "manager_approval"]},
     {"queue": "benefits"}),
    ("Handoff to a person",
     "This is the third time I have explained the same missing refund. Stop the bot and get me a person.",
     {"handoff": ["yes", "no"]},
     {"handoff": "yes"}),
    ("Did the agent finish?",
     "Goal: email the Q4 summary to every partner.\nLast action: draft saved in the hub.\nSend button is still disabled because two partners have no address.",
     {"finished": ["yes", "no"]},
     {"finished": "no"}),
    ("Moderation",
     "Post the customer's home address in the public thread so everyone can see where the package actually went.",
     {"policy": ["allow", "personal_data", "harassment", "scam", "violence", "spam"]},
     {"policy": "personal_data"}),
    ("Incident severity",
     "The deploy left resource tags inconsistent across staging. Production checkout is unaffected. No customer reports yet.",
     {"severity": ["info", "low", "medium", "high", "critical"]},
     {"severity": "low"}),
    ("Urgency score",
     "Payroll file has to be corrected before the 5pm cutoff or the whole company is paid late.",
     {"urgency": ["0", "1", "2", "3", "4", "5"]},
     {"urgency": "5"}),
    ("Spam or not",
     "Your mailbox is almost full. Click here in the next hour or we will delete every message.",
     {"label": ["spam", "ham"]},
     {"label": "spam"}),
    ("Several decisions at once",
     "Guest in room 1408 says the AC has been out since yesterday and they want to move tonight or leave. They also asked for the incidentals hold to be released.",
     {"intent": ["maintenance", "room_change", "checkout", "billing", "complaint", "amenity_request"],
      "priority": ["low", "normal", "high", "urgent"],
      "needs_human": ["yes", "no"],
      "topics": {"labels": ["hvac", "billing", "housekeeping", "noise", "safety"],
                 "multi_label": True, "cls_threshold": 0.4}},
     {"intent": "room_change", "priority": "high", "needs_human": "yes", "topics": ["hvac", "billing"]}),
    ("Question over a passage",
     "The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.",
     {"answer": {"labels": ["yes", "no"], "prompt": "Did the treaty enter into force in 1992?"}},
     {"answer": "no"}),
    ("Book",
     "She closed the ledger, blew out the lamp, and listened for the stair. The house had been empty since the winter the river took the bridge.",
     {"genre": ["mystery", "romance", "history", "science_fiction", "literary_fiction", "cookbook"]},
     {"genre": "literary_fiction"}),
    ("Labels with a description",
     "Please reset the card PIN. The new one never arrived and the old one is locked after three tries.",
     {"intent": {"labels": {
         "card_pin_change": "The customer wants a new PIN or the current PIN replaced",
         "card_lost": "The physical card is missing",
         "balance_inquiry": "The customer wants the current balance"}}},
     {"intent": "card_pin_change"}),
    ("Ordinal score",
     "I finished it in two nights. The ending is earned, the middle drags, and I would still hand it to a friend.",
     {"rating": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]},
     {"rating": "7"}),
]


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def check_readme(readme_path):
    """Re-parse the card's classify_text calls and "Potential output" blocks; they must equal README_EXAMPLES."""
    md = Path(readme_path).read_text(encoding="utf-8")
    calls = []
    for block in re.findall(r"```python\n(.*?)```", md, re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "classify_text":
                calls.append((ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1])))
    outputs = [json.loads(o) for o in re.findall(r"Potential output:\s*```text\n(.*?)```", md, re.S)]
    assert len(calls) == len(outputs) == len(README_EXAMPLES), (len(calls), len(outputs), len(README_EXAMPLES))
    for (title, text, tasks, out), (c_text, c_tasks), c_out in zip(README_EXAMPLES, calls, outputs):
        assert (text, tasks, out) == (c_text, c_tasks, c_out), f"README example drifted: {title}"
    return len(calls)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ oracle
class Oracle:
    def __init__(self, snapshot_dir):
        import torch
        from gliner2 import AutoExtractor
        from gliner2.training.trainer import ExtractorCollator

        self.torch = torch
        self.model = AutoExtractor.from_pretrained(snapshot_dir)
        self.model.eval()
        self.model.processor.change_mode(is_training=False)
        assert not any(m.training for m in self.model.modules())
        assert next(self.model.parameters()).device.type == "cpu"
        assert next(self.model.parameters()).dtype == torch.float32
        # The collator batch_extract builds for max_len=None (gliner2/inference/runtime.py batch_extract).
        self.collator = ExtractorCollator(self.model.processor, is_training=False,
                                          architecture=self.model.architecture)
        tok = self.model.processor.tokenizer
        self.marker_ids = {m: tok.convert_tokens_to_ids(m) for m in MARKERS}

    def collate(self, text_raw, tasks):
        """The exact (text, schema) -> PreprocessedBatch path of classify_text, for one case."""
        schema = self.model._classification_schema(tasks)
        schema_dicts, _ = self.model._build_schema_dicts_and_metadata([schema])
        batch = self.collator([(text_raw, schema_dicts[0])])
        assert len(batch) == 1
        return batch, schema_dicts[0]

    def run_case(self, text_raw, tasks):
        torch = self.torch
        m = self.model
        batch, schema_dict = self.collate(text_raw, tasks)
        ids = batch.input_ids[0].tolist()
        assert batch.attention_mask[0].tolist() == [1] * len(ids)
        stl = batch.schema_tokens_list[0]
        ssi = [list(p) for p in batch.schema_special_indices[0]]
        assert all(t == "classifications" for t in batch.task_types[0])
        fmt = m.processor._format_input_with_mapping(stl, batch.text_tokens[0])
        assert fmt["input_ids"] == ids
        cls_cfgs = schema_dict["classifications"]
        assert len(stl) == len(cls_cfgs) == len(ssi) == len(tasks)

        with torch.inference_mode():
            hidden = m.encoder(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask).last_hidden_state
            _, schema_embs = m.processor.extract_embeddings_from_batch(hidden, batch.input_ids, batch)
            task_results = []
            for j, (tokens, cfg) in enumerate(zip(stl, cls_cfgs)):
                resolved = m._resolve_classification_config(tokens[2], cls_cfgs)
                assert resolved is cfg and cfg["task"] == list(tasks)[j], (tokens[2], cfg["task"])
                embs = torch.stack(schema_embs[0][j])
                assert embs.shape[0] == len(cfg["labels"]) + 1 == len(ssi[j])
                logits = m.classifier(embs[1:]).squeeze(-1)
                act = cfg.get("class_act", "auto")
                multi = bool(cfg.get("multi_label", False))
                if act == "sigmoid" or (act == "auto" and multi):
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=-1)
                labels = cfg["labels"]
                thr = cfg.get("cls_threshold", 0.5)
                p = probs.tolist()
                if multi:
                    chosen = [j2 for j2 in range(len(labels)) if p[j2] >= thr]
                    if not chosen:
                        chosen = [int(torch.argmax(probs).item())]
                    decision = [labels[k] for k in chosen]
                    decision_probs = [p[k] for k in chosen]
                else:
                    best = int(torch.argmax(probs).item())
                    decision, decision_probs = labels[best], p[best]
                task_results.append({
                    "task": cfg["task"], "labels": labels, "multi_label": multi,
                    "cls_threshold": thr, "class_act": act,
                    "prompt_str": tokens[2],
                    "logits": logits.tolist(), "probs": p,
                    "decision": decision, "decision_probs": decision_probs,
                })

        ct = m.classify_text(text_raw, tasks, include_confidence=True)
        max_dp = 0.0
        for tr in task_results:
            got = ct[tr["task"]]
            if tr["multi_label"]:
                ct_labels = [g["label"] for g in got]
                ct_probs = [g["confidence"] for g in got]
                assert ct_labels == tr["decision"], (tr["task"], ct_labels, tr["decision"])
            else:
                ct_labels, ct_probs = got["label"], [got["confidence"]]
                assert ct_labels == tr["decision"], (tr["task"], ct_labels, tr["decision"])
            mine = tr["decision_probs"] if tr["multi_label"] else [tr["decision_probs"]]
            max_dp = max([max_dp] + [abs(a - b) for a, b in zip(mine, ct_probs)])
        assert max_dp < 1e-5, max_dp

        return {
            "text_raw": text_raw,
            "text": batch.original_texts[0],
            "tasks": tasks,
            "seq_len": len(ids),
            "n_labels": sum(len(c["labels"]) for c in cls_cfgs),
            "n_words": len(batch.text_tokens[0]),
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "schema_tokens_list": stl,
            "subword_list": fmt["subword_list"],
            "schema_special_indices": ssi,
            "task_results": task_results,
            "classify_text": ct,
            "self_check": {"decisions_equal": True, "max_abs_prob_diff": max_dp},
        }


def versions():
    import gliner2
    import huggingface_hub
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "gliner2": getattr(gliner2, "__version__", "?"),
            "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__,
            "numpy": numpy.__version__, "huggingface_hub": huggingface_hub.__version__,
            "torch_threads": torch.get_num_threads(), "machine": platform.machine(),
            "platform": platform.platform()}


def header(oracle, kind, extra, snapshot_dir, model_sha):
    now = datetime.datetime.now().astimezone()
    return {
        "kind": kind,
        "created": now.isoformat(timespec="seconds"),
        "script": "conversion/gliner25_decide_oracle.py",
        "model_id": MODEL_ID, "model_rev": MODEL_REV, "model_safetensors_sha256": model_sha,
        "snapshot_dir": str(snapshot_dir),
        "oracle": {
            "package": "gliner2 2.0.0", "device": "cpu", "dtype": "float32", "batch": 1,
            "encoder": type(oracle.model.encoder).__name__,
            "attn_implementation": getattr(oracle.model.encoder.config, "_attn_implementation", None),
            "path": "model.encoder(input_ids, attention_mask).last_hidden_state -> "
                    "processor.extract_embeddings_from_batch -> classifier(stack(embs)[1:]).squeeze(-1)",
            "decision_rule": "class_act auto: multi_label -> sigmoid, prob >= cls_threshold (none -> argmax 1); "
                             "single -> softmax, argmax",
        },
        "marker_ids": oracle.marker_ids,
        "versions": versions(),
        **extra,
    }


def summarize_self_check(cases):
    n_tasks = sum(len(c["task_results"]) for c in cases)
    return {"cases": len(cases), "tasks": n_tasks,
            "decisions_equal_tasks": sum(len(c["task_results"]) for c in cases if c["self_check"]["decisions_equal"]),
            "max_abs_prob_diff": max(c["self_check"]["max_abs_prob_diff"] for c in cases)}


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(work_path("_gliner25_decide")))
    ap.add_argument("-S", type=int, default=256)
    ap.add_argument("--mmax", type=int, default=32)
    ap.add_argument("--per-domain", type=int, default=20)
    ap.add_argument("--sets", default="readme21,fast_decisions")
    ap.add_argument("--min-len", type=int, default=0,
                    help="fast_decisions: keep only rows whose collated length is >= this (the long set); "
                         "domains may then have fewer than --per-domain rows")
    ap.add_argument("--fd-name", default=None,
                    help="fast_decisions fixture stem; default fast_decisions_s<S> "
                         "(fast_decisions_long with --min-len)")
    args = ap.parse_args()
    S, MMAX = args.S, args.mmax
    work = Path(args.work_dir).expanduser()
    (work / "fixtures").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)
    sets = set(args.sets.split(","))
    t_start = time.time()

    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL_ID, revision=MODEL_REV, local_files_only=True))
    model_sha = sha256_file(snap / "model.safetensors")
    assert model_sha == MODEL_SHA256, model_sha
    print(f"[INFO] snapshot {snap} sha256 ok", flush=True)
    n_readme = check_readme(snap / "README.md")
    print(f"[INFO] README examples verified against the card: {n_readme}", flush=True)

    oracle = Oracle(str(snap))
    assert oracle.marker_ids["[L]"] == 128007 and oracle.marker_ids["[P]"] == 128003
    print(f"[INFO] oracle loaded ({time.time() - t_start:.1f}s) markers={oracle.marker_ids}", flush=True)

    # ---------------------------------------------------------- README examples
    if "readme21" in sets:
        t0 = time.time()
        cases = []
        for i, (title, text, tasks, out) in enumerate(README_EXAMPLES, 1):
            c = oracle.run_case(text, tasks)
            assert c["seq_len"] <= S and c["n_labels"] <= MMAX, (title, c["seq_len"], c["n_labels"])
            decided = {tr["task"]: tr["decision"] for tr in c["task_results"]}
            cases.append({"id": f"readme/{i:02d}-{slug(title)}",
                          "source": {"set": "readme21", "readme_index": i, "title": title},
                          **c, "readme_output": out, "matches_readme_output": decided == out})
            print(f"   readme {i:02d} len={c['seq_len']:3d} labels={c['n_labels']:2d} "
                  f"{'=' if decided == out else '≠'} card  {decided}", flush=True)
        hdr = header(oracle, "gliner25-decide-oracle-fixture", {
            "set": "readme21", "S": S, "MMAX": MMAX,
            "selection": f"all {len(README_EXAMPLES)} classify_text examples of the model card at model_rev, verbatim",
            "self_check": summarize_self_check(cases),
            "matches_card_potential_output": sum(c["matches_readme_output"] for c in cases),
            "wall_s": round(time.time() - t0, 1),
        }, snap, model_sha)
        (work / "fixtures" / "readme21.json").write_text(
            json.dumps({"header": hdr, "cases": cases}, ensure_ascii=False, indent=1))
        print(f"[PASS] readme21: {hdr['self_check']}  card-output matches "
              f"{hdr['matches_card_potential_output']}/{len(cases)}  ({hdr['wall_s']}s)", flush=True)

    # ---------------------------------------------------------- fast-decisions
    if "fast_decisions" in sets:
        t0 = time.time()
        long_set = args.min_len > 0
        fd_name = args.fd_name or ("fast_decisions_long" if long_set else f"fast_decisions_s{S}")
        picked_per_domain = {}
        dsnap = Path(snapshot_download(DATASET_ID, repo_type="dataset", revision=DATASET_REV,
                                       local_files_only=True))
        rows_len, per_domain_len, cases = [], {}, []
        for domain in DOMAINS:
            rows = [json.loads(line) for line in (dsnap / f"{domain}.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            picked = 0
            lens = []
            for r_idx, row in enumerate(rows):
                heads = row["output"]["classifications"]
                tasks = {h["task"]: {"labels": h["labels"], "multi_label": h["multi_label"]} for h in heads}
                assert len(tasks) == len(heads), (domain, r_idx)
                batch, _ = oracle.collate(row["input"], tasks)
                seq_len = batch.input_ids.shape[1]
                n_labels = sum(len(h["labels"]) for h in heads)
                fits = seq_len <= S and n_labels <= MMAX
                lens.append(seq_len)
                rows_len.append({"domain": domain, "row": r_idx, "seq_len": seq_len, "n_labels": n_labels,
                                 "n_tasks": len(heads), "fits_S": fits, "le_256": seq_len <= 256,
                                 "le_512": seq_len <= 512})
                if fits and seq_len >= args.min_len and picked < args.per_domain:
                    c = oracle.run_case(row["input"], tasks)
                    assert c["seq_len"] == seq_len
                    gold = {h["task"]: h["true_label"] for h in heads}
                    cases.append({"id": f"fd/{domain}/{r_idx:03d}",
                                  "source": {"set": "fast_decisions", "domain": domain, "row": r_idx},
                                  **c, "gold": gold})
                    picked += 1
            assert picked == args.per_domain or (long_set and picked < args.per_domain), (domain, picked)
            picked_per_domain[domain] = picked
            n = len(lens)
            per_domain_len[domain] = {
                "rows": n, "min": min(lens), "median": statistics.median(lens),
                "p90": sorted(lens)[int(0.9 * (n - 1))], "max": max(lens),
                "le_128": sum(x <= 128 for x in lens), "le_256": sum(x <= 256 for x in lens),
                "le_384": sum(x <= 384 for x in lens), "le_512": sum(x <= 512 for x in lens),
                "max_labels": max(r["n_labels"] for r in rows_len if r["domain"] == domain),
                "fits_S_and_MMAX": sum(r["fits_S"] for r in rows_len if r["domain"] == domain),
            }
            d = per_domain_len[domain]
            print(f"   {domain:18s} picked {picked}  len min/med/max {d['min']}/{d['median']}/{d['max']}  "
                  f"<=256 {d['le_256']}/{n}  <=512 {d['le_512']}/{n}  ({time.time() - t0:.0f}s)", flush=True)

        hdr_extra = {"dataset_id": DATASET_ID, "dataset_rev": DATASET_REV, "S": S, "MMAX": MMAX}
        total = len(rows_len)
        lengths = {
            "header": header(oracle, "gliner25-decide-collated-lengths", {
                **hdr_extra,
                "measure": "len(input_ids) from gliner2's own collate of (row input, all heads of the row in one "
                           "call, multi_label per row, cls_threshold default 0.5); no CLS/SEP exist in this layout",
            }, snap, model_sha),
            "overall": {
                "rows": total, "le_256": sum(r["le_256"] for r in rows_len),
                "le_512": sum(r["le_512"] for r in rows_len),
                "max_seq_len": max(r["seq_len"] for r in rows_len),
                "max_labels": max(r["n_labels"] for r in rows_len),
                "fits_S_and_MMAX": sum(r["fits_S"] for r in rows_len),
            },
            "per_domain": per_domain_len,
            "rows": rows_len,
        }
        if not long_set:                                   # all rows; the long run leaves it as written
            (work / "results" / "lengths.json").write_text(json.dumps(lengths, ensure_ascii=False, indent=1))

        length_rule = (f"{args.min_len} <= collated input_ids length <= {S}" if long_set
                       else f"collated input_ids length <= {S}")
        hdr = header(oracle, "gliner25-decide-oracle-fixture", {
            **hdr_extra, "set": fd_name,
            "selection": f"per domain (dataset card order), rows in file order; keep a row when its {length_rule} "
                         f"and its total label count <= {MMAX}; {'up to' if long_set else 'first'} "
                         f"{args.per_domain} kept rows. tasks = {{task: {{labels, multi_label}}}} for all heads of "
                         f"the row, cls_threshold default 0.5",
            "picked_per_domain": picked_per_domain,
            "self_check": summarize_self_check(cases),
            "wall_s": round(time.time() - t0, 1),
        }, snap, model_sha)
        (work / "fixtures" / f"{fd_name}.json").write_text(
            json.dumps({"header": hdr, "cases": cases}, ensure_ascii=False, indent=1))

        # own-subset exact match vs gold
        per_domain, rows_acc = {}, []
        for c in cases:
            dom = c["source"]["domain"]
            heads = []
            for tr in c["task_results"]:
                gold = c["gold"][tr["task"]]
                if tr["multi_label"]:
                    exact = sorted(tr["decision"]) == sorted(gold)
                else:
                    exact = tr["decision"] == gold[0]
                heads.append({"task": tr["task"], "multi_label": tr["multi_label"],
                              "decision": tr["decision"], "gold": gold, "exact": exact})
            all_exact = all(h["exact"] for h in heads)
            rows_acc.append({"id": c["id"], "domain": dom, "row": c["source"]["row"],
                             "heads": heads, "all_heads_exact": all_exact})
            d = per_domain.setdefault(dom, {"rows": 0, "rows_all_exact": 0, "heads": 0, "heads_exact": 0,
                                            "per_task": {}})
            d["rows"] += 1
            d["rows_all_exact"] += all_exact
            for h in heads:
                d["heads"] += 1
                d["heads_exact"] += h["exact"]
                t = d["per_task"].setdefault(h["task"], {"n": 0, "exact": 0, "multi_label": h["multi_label"]})
                t["n"] += 1
                t["exact"] += h["exact"]
        for d in per_domain.values():
            d["head_acc_pct"] = pct(d["heads_exact"], d["heads"])
            d["row_acc_pct"] = pct(d["rows_all_exact"], d["rows"])
        n_heads = sum(d["heads"] for d in per_domain.values())
        acc = {
            "header": header(oracle, "gliner25-decide-own-subset-accuracy", {
                **hdr_extra,
                "scope": f"OWN SUBSET of the published development split: {len(cases)} rows = "
                         f"{'up to' if long_set else 'first'} {args.per_domain} rows per domain with {length_rule} "
                         f"and labels <= {MMAX}. NOT the publisher's "
                         "benchmark (60.2% is fastino's number on the held-out 300-per-domain test split); "
                         "do not mix the two.",
                "protocol": "one classify_text-equivalent call per row with all heads of the row; single-label "
                            "exact = decision == true_label[0]; multi-label exact = set(decision) == set(true_label) "
                            "at cls_threshold 0.5. The dataset card's own loop scores each head in a separate call "
                            "with labels only (multi_label False) — a different protocol.",
            }, snap, model_sha),
            "overall": {
                "rows": len(rows_acc), "heads": n_heads,
                "micro_head_acc_pct": pct(sum(d["heads_exact"] for d in per_domain.values()), n_heads),
                "macro_head_acc_pct": round(statistics.mean(d["head_acc_pct"] for d in per_domain.values()), 2),
                "micro_row_acc_pct": pct(sum(r["all_heads_exact"] for r in rows_acc), len(rows_acc)),
                "macro_row_acc_pct": round(statistics.mean(d["row_acc_pct"] for d in per_domain.values()), 2),
            },
            "per_domain": per_domain,
            "rows": rows_acc,
        }
        acc_name = ("own_subset_accuracy.json" if fd_name == "fast_decisions_s256"
                    else f"own_subset_accuracy_{fd_name.removeprefix('fast_decisions_')}.json")
        (work / "results" / acc_name).write_text(json.dumps(acc, ensure_ascii=False, indent=1))
        print(f"[PASS] {fd_name}: {hdr['self_check']} picked {picked_per_domain}  ({hdr['wall_s']}s)", flush=True)
        if not long_set:
            print(f"[INFO] lengths: {lengths['overall']}", flush=True)
        print(f"[INFO] own-subset accuracy: {acc['overall']}", flush=True)

    print(f"[DONE] total {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
