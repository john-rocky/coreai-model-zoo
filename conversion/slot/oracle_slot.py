#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.9.0",
#     "transformers==5.17.0",
#     "safetensors==0.8.0",
#     "huggingface_hub==1.32.0",
#     "numpy==2.3.5",
#     "tokenizers==0.23.2",
#     "pydantic==2.13.5",
# ]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# ///
"""Author's OpenThai-SystemOne CPU/fp32 oracle; adapted from zoo 082fe55 decider gate.

Slot-head readout replaces the predecessor's option-letter vocabulary readout. Each
fixture is one unpermuted question encoded by the pinned author's Formatter. The script
also executes every public API request independently and in the one-pass form.

    uv run conversion/slot/oracle_slot.py --hf-id iapp/OpenThai-SystemOne \
      --revision f3709948b5e3cc9606a57e74ba62b7a639d17dd3 --out fixtures.json

The source package is copied unchanged from the pinned snapshot; client.py is
pinned to upstream commit 5d04bcc. Non-CUDA reference kernels are selected by
the author client. No flash-linear-attention or causal-conv1d is needed.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

"""Synthetic deterministic fixture requests; no sampling and no external entities."""

def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions,
            "criteria": dict.fromkeys(criteria) if isinstance(criteria, list) else criteria}


def noul(instructions, criteria=None):
    return {"type": "noul", "instructions": instructions, "criteria": criteria or {}}


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


def request(id, language, state, questions, zoo_only=False):
    return {"id": id, "language": language, "state": state, "questions": questions,
            "zoo_only": zoo_only}


REQUESTS = [
    request("r01", "th", "กล่องมีตัวล็อกแตก แต่สิ่งของภายในสมบูรณ์ ผู้ส่งต้องการเปลี่ยนตัวล็อก ไม่ได้ขอคืนเงิน", {
        "route": choice("ควรส่งคำขอนี้ให้ฝ่ายใด", {"ซ่อมแซม": "เปลี่ยนชิ้นส่วนที่ชำรุด", "การเงิน": "จัดการการชำระเงิน", "ขนส่ง": "ติดตามกล่องที่สูญหาย"}),
        "condition": choice("สิ่งของภายในมีสภาพอย่างไร", ["สูญหาย", "สมบูรณ์", "เปียก", "แตก"]),
        "refund": noul("ผู้ส่งขอคืนเงินหรือไม่", {"false": "ไม่ได้ขอเงินคืน", "true": "ขอเงินคืน"})}),
    request("r02", "th", {"ห้อง": {"อุณหภูมิ": "เย็น", "โคมไฟ": "ปิด", "หน้าต่าง": "เปิด"}, "คำขอ": "ปิดหน้าต่าง"}, {
        "action": choice("ควรทำตามคำขอใด", {"ปิดหน้าต่าง": "ปิดช่องหน้าต่าง", "เปิดไฟ": "เปิดโคมไฟ", "เปิดหน้าต่าง": "เปิดช่องหน้าต่างเพิ่ม"}),
        "lamp": choice("ขณะนี้โคมไฟอยู่ในสถานะใด", ["เปิด", "กะพริบ", "ปิด"]),
        "cold": noul("ห้องนี้เย็นหรือไม่")}),
    request("r03", "th", "กระเบื้องที่ทำเครื่องหมายเป็นรูปหกเหลี่ยมสีเขียว พื้นผิวแห้งและไม่มีขอบ", {
        "color": choice("กระเบื้องมีสีอะไร", {"สีแดง": "สีแดง", "สีเขียว": "สีเขียว", "สีน้ำเงิน": "สีน้ำเงิน"}),
        "shape": choice("กระเบื้องเป็นรูปอะไร", ["วงกลม", "สี่เหลี่ยม", "สามเหลี่ยม", "ห้าเหลี่ยม", "หกเหลี่ยม", "วงรี", "ดาว", "หัวใจ", "เสี้ยววงเดือน", "แปดเหลี่ยม", "ลูกศร", "กากบาท"]),
        "wet": noul("กระเบื้องเปียกหรือไม่", {"false": "กระเบื้องแห้ง", "true": "มีน้ำบนกระเบื้อง"})}),
    request("r04", "th", ["ตู้ทำจากไม้", "ลิ้นชักที่เลือกคือหมายเลข 7", "ลิ้นชักล็อกอยู่", "มีกุญแจ"], {
        "status": choice("ลิ้นชักอยู่ในสถานะใด", {"ล็อก": "เปิดไม่ได้จนกว่าจะใช้กุญแจ", "เปิด": "เปิดอยู่แล้ว", "เสีย": "ใช้งานไม่ได้เพราะชำรุด"}),
        "drawer": choice("เลือกหมายเลขลิ้นชักใด", [str(i) for i in range(1, 12)]),
        "key": noul("มีกุญแจอยู่หรือไม่")}),
    request("r05", "th", "แปลงผักแห้ง ต้นกล้ามีใบสีเขียว งานถัดไปคือรดน้ำ ยังไม่ถึงเวลาเก็บเกี่ยว", {
        "task": choice("งานถัดไปคืออะไร", {"รดน้ำ": "เติมน้ำให้แปลงผัก", "เก็บเกี่ยว": "เก็บผักที่โตเต็มที่", "ทาสี": "ทาสีรั้ว"}),
        "leaves": choice("ใบของต้นกล้ามีสีอะไร", ["น้ำตาล", "เหลือง", "เขียว", "แดง"]),
        "dry": noul("แปลงผักแห้งหรือไม่", {"false": "ดินยังเปียก", "true": "ดินแห้ง"})}),
    request("r06", "th", {"งาน": {"ขั้นตอน": "ตรวจทาน", "ผล": "รอผล", "ความสำคัญ": "ต่ำ"}, "อนุมัติ": False}, {
        "phase": choice("งานอยู่ในขั้นตอนใด", {"ร่าง": "กำลังเขียนร่าง", "ตรวจทาน": "กำลังตรวจสอบเนื้อหา", "เสร็จ": "งานเสร็จสมบูรณ์"}),
        "priority": choice("งานมีความสำคัญระดับใด", ["สูง", "กลาง", "ต่ำ"]),
        "approved": noul("งานได้รับอนุมัติแล้วหรือไม่", {"false": "ยังไม่ได้รับอนุมัติ", "true": "ได้รับอนุมัติแล้ว"})}),
    request("r07", "en", "The parcel belongs to team 12. The contents are intact. No refund is requested.", {
        "team": choice("Which team owns the parcel?", [str(i) for i in range(1, 17)]),
        "condition": choice("How are the contents described?", ["missing", "intact", "wet", "broken"]),
        "refund": noul("Is a refund requested?", {"false": "No money back is requested", "true": "Money back is requested"})}),
    request("r08", "en", "The selected storage bin is violet. Its lid is closed and the lamp is off.", {
        "bin": choice("Which bin is selected?", ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "black", "white", "gray", "brown", "pink"]),
        "lamp": choice("What is the lamp state?", {"on": "The lamp emits light", "off": "The lamp emits no light", "flashing": "The lamp blinks"}),
        "closed": noul("Is the selected bin lid closed?")}),
    request("r09", "en", {"tank": {"contents": "clean water", "fill": "half full", "valve": "closed"}}, {
        "liquid": choice("What liquid is in the tank?", ["oil", "water", "ink"]),
        "valve": noul("Is the valve open?", {"false": "The valve is closed", "true": "The valve is open"}),
        "fill": score("How full is the tank?", ["empty", "one quarter full", "half full", "three quarters full", "completely full"])}),
    request("r10", "en", ["Four objects were inspected.", "Their surfaces are smooth and white.", "No damage is present."], {
        "surface": choice("How is the inspected surface described?", ["rough", "smooth", "cracked", "sticky"]),
        "white": noul("Are the inspected objects white?"),
        "damage": score("How much damage is present?", ["no damage", "minor scratches", "shallow dents", "large cracks", "completely broken"])}),
    request("r11", "en", "A task is paused for inspection. Restart is forbidden. Its urgency is low.", {
        "status": choice("What is the task status?", ["running", "paused", "finished"]),
        "restart": noul("Is restarting allowed?", {"false": "Restarting is forbidden", "true": "Restarting is allowed"}),
        "urgency": score("How urgent is the task?", ["low", "medium", "high"])}),
    request("r12", "en", "The path leads west and is covered in gravel. It is closed. The slope is flat.", {
        "direction": choice("Which direction does the path lead?", ["north", "south", "east", "west"]),
        "open": noul("Is the path open?"),
        "slope": score("What is the slope?", ["flat", "slightly sloped", "steep"])}),
    request("r13", "mixed", {"ห้อง": "storage", "door": "ปิด", "light": "on", "temperature": "อุ่น"}, {
        "door": choice("ประตูอยู่ในสถานะใด? Choose its state.", ["เปิด / open", "ปิด / closed", "หาย / missing"]),
        "light": noul("Is the light on? ไฟเปิดอยู่หรือไม่", {"false": "ปิด / off", "true": "เปิด / on"}),
        "temperature": score("อุณหภูมิห้องเป็นอย่างไร? Rate the temperature.", ["cold / เย็น", "warm / อุ่น", "hot / ร้อน"])}),
    request("r14", "mixed", ["The basket contains แอปเปิลสีแดง.", "ตะกร้าแห้งและว่างไปครึ่งหนึ่ง / half empty."], {
        "color": choice("ผลไม้สีอะไร? Choose the fruit color.", ["แดง / red", "เขียว / green", "เหลือง / yellow"]),
        "wet": noul("Is the basket wet? ตะกร้าเปียกหรือไม่"),
        "fill": score("ตะกร้าเต็มเพียงใด? Rate its fill.", ["empty / ว่าง", "half full / ครึ่งหนึ่ง", "full / เต็ม"])}),
    request("r15", "en", "A dial points to level 7 on a scale from 0 to 9. Its color is blue. The switch is in the lower of two positions.", {
        "color": choice("What is the dial color?", ["red", "blue", "green"]),
        "level": score("Which level does the dial indicate?", [f"level {i}" for i in range(10)]),
        "position": score("Which position is the switch in?", ["lower position", "upper position"])}),
    request("r16", "th", "มาตรวัดชี้ระดับ 3 จากระดับ 0 ถึง 9 หน้าปัดสีขาว สวิตช์อยู่ตำแหน่งบนจากสองตำแหน่ง", {
        "color": choice("หน้าปัดมีสีอะไร", ["ดำ", "ขาว", "เทา"]),
        "level": score("มาตรวัดชี้ระดับใด", [f"ระดับ {i}" for i in range(10)]),
        "position": score("สวิตช์อยู่ตำแหน่งใด", ["ตำแหน่งล่าง", "ตำแหน่งบน"])}),
    request("r17", "en", "The requested archive slot is 27. Retrieve slot 27.", {
        "slot": choice("Which archive slot is requested?", [str(i) for i in range(40)])}, zoo_only=True),
    request("r18", "en", "The requested archive slot is 173. Retrieve slot 173.", {
        "slot": choice("Which archive slot is requested?", [str(i) for i in range(255)])}, zoo_only=True),
]


DEFAULT_HF_ID = 'iapp/OpenThai-SystemOne'
DEFAULT_REVISION = 'f3709948b5e3cc9606a57e74ba62b7a639d17dd3'
WORK = Path.cwd()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


def floors():
    kit = [r for r in REQUESTS if not r['zoo_only']]
    qs = [q for req in kit for q in req['questions'].values()]
    choices = [q for q in qs if q['type'] == 'choice']
    scores = [q for q in qs if q['type'] == 'score']
    nouls = [q for q in qs if q['type'] == 'noul']
    result = {
        'requests': len(REQUESTS), 'kit_requests': len(kit), 'kit_rows': len(qs),
        'wide_rows': sum(len(r['questions']) for r in REQUESTS if r['zoo_only']),
        'languages': dict(collections.Counter(r['language'] for r in kit)),
        'dict_states': sum(isinstance(r['state'], dict) for r in kit),
        'list_states': sum(isinstance(r['state'], list) for r in kit),
        'types': dict(collections.Counter(q['type'] for q in qs)),
        'choice_all_descriptions': sum(all(v is not None for v in q['criteria'].values()) for q in choices),
        'choice_all_null': sum(all(v is None for v in q['criteria'].values()) for q in choices),
        'choice_11_16': sum(11 <= len(q['criteria']) <= 16 for q in choices),
        'noul_descriptions': sum(set(q['criteria']) == {'true', 'false'} for q in nouls),
        'score_10_levels': sum(len(q['criteria']) == 10 for q in scores),
    }
    assert result['kit_requests'] >= 16 and result['kit_rows'] >= 48 and result['wide_rows'] == 2
    assert all(result['languages'][k] >= v for k, v in {'th': 6, 'en': 6, 'mixed': 2}.items())
    assert result['dict_states'] >= 3 and result['list_states'] >= 2
    assert result['types']['choice'] >= 22 and result['types']['noul'] >= 10 and result['types']['score'] >= 8
    assert result['choice_all_descriptions'] >= 6 and result['choice_all_null'] >= 6
    assert result['choice_11_16'] >= 4 and result['noul_descriptions'] >= 4
    assert result['score_10_levels'] >= 2
    assert all(2 <= len(q['criteria']) <= 16 for q in choices)
    assert all(2 <= len(q['criteria']) <= 10 for q in scores)
    return result


def main():
    global WORK
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hf-id', default=DEFAULT_HF_ID)
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--work-dir', type=Path, help='author package and progress evidence; default beside output')
    parser.add_argument('--out', type=Path, default=Path('fixtures-openthai-systemone.json'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.out = args.out.resolve()
    WORK = args.work_dir.resolve() if args.work_dir else args.out.parent / ('.' + args.out.stem + '-work')
    WORK.mkdir(parents=True, exist_ok=True)
    import os
    import shutil
    import urllib.request
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    args.snapshot = Path(snapshot_download(args.hf_id, revision=args.revision,
                                          ignore_patterns=['assets/*'], max_workers=8))
    package = WORK / 'author/openthai_systemone'
    package.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').write_text('')
    sources = []
    for name in ('modeling.py', 'formatting.py', 'types.py', 'configuration.py'):
        source = args.snapshot / name
        shutil.copyfile(source, package / name)
        data = source.read_bytes()
        sources.append({'name': name, 'source': str(source), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(), 'unmodified': True})
    client_url = ('https://raw.githubusercontent.com/iapp-technology/openthai-systemone/'
                  '5d04bcc/openthai_systemone/client.py')
    with urllib.request.urlopen(client_url, timeout=90) as response:
        data = response.read()
    (package / 'client.py').write_bytes(data)
    sources.append({'name': 'client.py', 'source': client_url, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(), 'unmodified': True})
    write_json(WORK / 'author_sources.json', {'hf_id': args.hf_id, 'revision': args.revision,
               'resolved_revision': args.snapshot.name, 'files': sources})
    import torch
    import transformers
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(WORK / 'author'))
    from openthai_systemone.client import SystemOneClient
    from openthai_systemone.formatting import collate, sanitize, spec_to_text, SPECIAL_TOKENS
    from openthai_systemone.modeling import QTYPE_INDEX, confidence_from_probs
    from openthai_systemone.types import parse_question

    start = time.monotonic()
    composition = floors()
    client = SystemOneClient(str(args.snapshot), device='cpu', dtype=torch.float32)
    model = client.model
    assert model.config.answer_token_id == 248082
    assert model.config.n_slots == 256 and model.config.abstain_slot == 255
    assert model.slot_head.weight.shape == (256, 1024)
    assert model.slot_head.bias.shape == (256,)
    assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).device.type == 'cpu'
    temperatures = model.log_temperature.exp().detach().cpu().tolist()
    temperature_exact = {name: temperatures[idx] for name, idx in QTYPE_INDEX.items()}
    temperature_6 = {name: round(value, 6) for name, value in temperature_exact.items()}
    tokenizer = client.tok
    formatter = client.fmt
    control_ids = {name: 248077 + i for i, name in enumerate(SPECIAL_TOKENS[:6])}
    control_ids.update({f'<|ts_opt_{i}|>': 248083 + i for i in range(256)})
    assert set(control_ids) == set(SPECIAL_TOKENS)
    for name, expected_id in control_ids.items():
        assert tokenizer.encode(name, add_special_tokens=False) == [expected_id], name
    assert len(tokenizer) == 248339
    model_info = {
        'snapshot': str(args.snapshot.resolve()), 'temperature_by_type': temperature_6,
        'temperature_exact_by_type': temperature_exact,
        'temperature_6_decimal_strings': {k: f'{v:.6f}' for k, v in temperature_exact.items()},
        'log_temperature': model.log_temperature.detach().cpu().tolist(),
        'dtype': str(next(model.parameters()).dtype), 'device': 'cpu',
        'parameters': sum(p.numel() for p in model.parameters()),
        'answer_token_id': 248082, 'n_slots': 256, 'abstain_slot': 255,
        'control_token_assertions': len(control_ids), 'tokenizer_length': len(tokenizer),
        'transformers': transformers.__version__, 'torch': torch.__version__,
    }
    write_json(WORK / 'oracle_model.json', model_info)
    print(json.dumps(model_info), flush=True)

    def guard():
        if args.deadline_unix and time.time() >= args.deadline_unix:
            raise TimeoutError('Oracle deadline reached between CPU calls')

    def direct(encoded, qtype_index):
        guard()
        batch = collate([encoded], tokenizer.pad_token_id)
        batch.pop('labels')
        batch['qtypes'] = torch.tensor([[qtype_index]], dtype=torch.long)
        with torch.inference_mode():
            output = model(**batch)
            raw = model.slot_head(output.hidden_states.to(model.slot_head.weight.dtype)).float()[0, 0]
        assert torch.isfinite(raw).all()
        assert torch.isfinite(output.probs).all()
        return raw, output.probs[0, 0]

    rows, assemblies = [], []
    for req in REQUESTS:
        typed = {qid: parse_question(q) for qid, q in req['questions'].items()}
        request_rows, single_answers = [], {}
        for qid, q in typed.items():
            encoded = formatter.encode(req['state'], {qid: q})
            assert not encoded.truncated_state
            assert len(encoded.specs) == len(encoded.answer_positions) == 1
            spec = encoded.specs[0]
            full_ids = encoded.input_ids
            slot = encoded.answer_positions[0]
            assert slot == len(full_ids) - 2 and full_ids[slot] == 248082
            assert tokenizer.decode(full_ids[-1:]) == '\n'
            ids = full_ids[:slot + 1]
            k = encoded.option_counts[0]
            assert spec.perm == list(range(k)), 'Formatter option order must remain untouched'
            limit = 2048 if req['zoo_only'] else 1024
            assert len(ids) <= limit, (req['id'], qid, len(ids), limit)
            qtext_lines = spec_to_text(spec).splitlines()
            option_lines = qtext_lines[1:-1]
            options = []
            for i, line in enumerate(option_lines):
                prefix = f'<|ts_opt_{i}|> '
                assert line.startswith(prefix)
                options.append(line[len(prefix):])
            assert len(options) == k
            qt = QTYPE_INDEX[spec.qtype]
            raw_full, p_full = direct(encoded, qt)
            truncated = dataclasses.replace(encoded, input_ids=ids)
            raw_truncated, _ = direct(truncated, qt)
            causal_delta = float((raw_full - raw_truncated).abs().max())
            assert causal_delta <= 1e-4, (req['id'], qid, causal_delta)
            p = p_full[:k] / p_full[:k].sum()
            abstain = float(p_full[255])
            nonzero_count = int(torch.count_nonzero(p_full))
            assert torch.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-5
            # Independent expression of the author's masking/temperature contract.
            scaled = raw_full / temperature_exact[spec.qtype]
            masked = scaled.clone()
            masked[k:255] = float('-inf')
            p_explicit = masked.softmax(-1)
            assert torch.equal(p_full, p_explicit)
            top2 = torch.topk(p, 2).values
            record = {
                'id': f"{req['id']}-{qid}", 'request_id': req['id'], 'question_id': qid,
                'type': spec.qtype, 'qtype_index': qt, 'kind': 'slot',
                'question': sanitize(spec.instructions).strip(), 'options': options,
                'option_names': spec.option_names, 'ids_full': full_ids, 'slot': slot,
                'ids': ids, 'nopts': k, 'label_ids': list(range(k)),
                'temperature': temperature_exact[spec.qtype], 'raw_logits': raw_full.tolist(),
                'p_full': p_full.tolist(), 'p_full_nonzero_count': nonzero_count,
                'p_full_expected_nonzero_count': k + 1, 'p_full_nonzero_count_matches': nonzero_count == k + 1,
                'p_oracle': p.tolist(), 'abstain': abstain,
                'confidence': confidence_from_probs(p, k), 'argmax': int(p.argmax()),
                'top2_margin': float(top2[0] - top2[1]), 'tokens': len(ids),
                'zoo_only': req['zoo_only'], 'language': req['language'],
                'causality_max_abs_diff': causal_delta,
                'raw_logits_min': float(raw_full.min()), 'raw_logits_max': float(raw_full.max()),
                'raw_logits_absmax': float(raw_full.abs().max()), 'all_finite': True,
            }
            guard()
            api = client.system_one(req['state'], {qid: q}, permutations=1)
            assembly = check_api(record, encoded, api, client, torch)
            record['api_assembly_exact_equal'] = assembly['exact_equal']
            assert assembly['exact_equal'], (record['id'], assembly)
            single_answers[qid] = assembly
            rows.append(record)
            request_rows.append(record)
            write_json(WORK / 'oracle_progress.json', {
                'completed_rows': len(rows), 'last_row': record['id'],
                'causality_max_abs_diff': max(r['causality_max_abs_diff'] for r in rows),
                'wall_seconds': time.monotonic() - start,
            })
            write_json(WORK / 'oracle_rows.partial.json', rows)
            print(f"{record['id']}: tokens={len(ids)} k={k} causality={causal_delta:.9g} API exact=True", flush=True)
        guard()
        one_pass = client.system_one(req['state'], typed, permutations=1)
        one_pass = as_dict(one_pass)
        compared = []
        for r in request_rows:
            answer = one_pass['answers'][r['question_id']]
            vector = answer_vector(answer, r)
            independent_api = single_answers[r['question_id']]['actual_answer']
            single_vector = answer_vector(independent_api, r)
            differences = [abs(x-y) for x,y in zip(vector, single_vector)]
            compared.append({'row_id': r['id'], 'one_pass_answer': answer,
                             'independent_answer': independent_api,
                             'one_pass_probabilities': vector, 'independent_probabilities': single_vector,
                             'max_abs_delta_p': max(differences)})
        assemblies.append({'request_id': req['id'],
                           'exact_equal': all(a['exact_equal'] for a in single_answers.values()),
                           'independent_questions': single_answers, 'one_pass': one_pass,
                           'one_pass_comparison': compared,
                           'one_pass_max_abs_delta_p': max(c['max_abs_delta_p'] for c in compared),
                           'one_pass_is_gate': False})
        write_json(WORK / 'oracle_api.partial.json', assemblies)
        print(f"{req['id']}: one-pass max |delta p|={assemblies[-1]['one_pass_max_abs_delta_p']:.9g}", flush=True)
    summary = dict(composition)
    summary.update({
        'rows': len(rows), 'min_top2_margin': min(r['top2_margin'] for r in rows),
        'near_ties_below_0_02': [r['id'] for r in rows if r['top2_margin'] < 0.02],
        'max_tokens': max(r['tokens'] for r in rows),
        'max_kit_tokens': max(r['tokens'] for r in rows if not r['zoo_only']),
        'wide_tokens': {r['id']: r['tokens'] for r in rows if r['zoo_only']},
        'total_tokens': sum(r['tokens'] for r in rows),
        'causality_max_abs_diff': max(r['causality_max_abs_diff'] for r in rows),
        'causality_tolerance': 1e-4, 'causality_all_pass': True,
        'single_question_api_all_exact_equal': all(r['api_assembly_exact_equal'] for r in rows),
        'p_full_nonzero_counts_all_match': all(r['p_full_nonzero_count_matches'] for r in rows),
        'one_pass_max_abs_delta_p': max(a['one_pass_max_abs_delta_p'] for a in assemblies),
        'wall_seconds': time.monotonic() - start, 'all_finite': True,
        'raw_logits_absmax': max(r['raw_logits_absmax'] for r in rows),
        'status': 'PASS',
    })
    result = {
        'schema': 'coreai-slot-fixtures/1',
        'source': {'hf_id': args.hf_id, 'revision': args.revision,
                   'oracle': "author's modeling.py slot_logits, fp32, CPU",
                   'transformers': transformers.__version__, 'torch': torch.__version__,
                   'client_revision': '5d04bcc', 'snapshot': str(args.snapshot.resolve())},
        'temperature_by_type': temperature_6, 'temperature_exact_by_type': temperature_exact,
        'layout': 'openthai_systemone', 'independent_rows': True, 'permutations': 1,
        'n_slots': 256, 'abstain_slot': 255, 'answer_token_id': 248082,
        'control_ids': control_ids, 'requests': REQUESTS, 'rows': rows,
        'api_assembly': assemblies, 'summary': summary,
        'api_contract_note': 'Only choice exposes abstain. Noul exposes noul only; score exposes probabilities/confidence but not abstain. Every row records all three quantities directly from the author model. The pinned client does not round; choice/score probabilities undergo a second fp32 renormalization in _decode_named.',
        'adapted_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(json.dumps(REQUESTS, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'author_sources': sources,
    }
    write_json(args.out, result)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f'wrote {args.out}', flush=True)


def as_dict(value):
    return value.model_dump() if hasattr(value, 'model_dump') else value


def answer_vector(answer, row):
    if row['type'] == 'noul':
        return [1.0 - answer['noul'], answer['noul']]
    return [answer['probabilities'][name] for name in row['option_names']]


def check_api(row, encoded, api, client, torch):
    """Reassemble the direct-model row through the author's unchanged decoder.

    client.py 126 renormalizes once; lines 142-143 normalize a second time for
    choice and score outputs. There is no decimal rounding in this pinned client.
    """
    qid = row['question_id']
    spec = encoded.specs[0]
    per_q = {qid: dict(zip(spec.option_names, row['p_oracle']))}
    expected = as_dict(client._decode_named(encoded, per_q, {qid: row['abstain']}, 1))
    actual = as_dict(api)
    expected_answer = expected['answers'][qid]
    actual_answer = actual['answers'][qid]
    fields = {name: actual_answer.get(name) == value for name, value in expected_answer.items()}
    expected_vector = answer_vector(expected_answer, row)
    actual_vector = answer_vector(actual_answer, row)
    return {
        'row_id': row['id'], 'exact_equal': actual == expected,
        'actual_answer': actual_answer, 'expected_answer': expected_answer,
        'actual_response': actual, 'expected_response': expected,
        'fields_exact_equal': fields,
        'max_abs_delta_p_after_client_normalization': max(abs(x-y) for x,y in zip(actual_vector, expected_vector)),
        'max_abs_delta_p_vs_single_renormalization': max(abs(x-y) for x,y in zip(actual_vector, row['p_oracle'])),
        'normalization': 'Author _decode_named: second fp32 renormalization for choice/score; noul directly reads yes. No rounding.',
        'abstain_exposed': 'abstain' in actual_answer,
        'confidence_exposed': 'confidence' in actual_answer,
    }


if __name__ == '__main__':
    main()
