"""Turn an AneGateRunner device log + its fixture into a JSON gate transcript for a zoo card.
Usage: python transcript.py --log _device_2b_clean.log --fixture fixtures/minicpm5_2b/fixture.json \
           --bundle <name> --device "iPhone 17 Pro / iOS 27.0 24A435" --out gate.json [--red-log ...]"""
import argparse, json, re, datetime
ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True); ap.add_argument("--fixture", required=True)
ap.add_argument("--bundle", required=True); ap.add_argument("--device", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--red-log")
ap.add_argument("--note", default="")
a = ap.parse_args()
fx = json.load(open(a.fixture))
def parse(path):
    tf, fr, gate, summary = {}, {}, {}, None
    for line in open(path):
        m = re.match(r"TF (\w+) k=(\d+) exp=(\d+) got=(\d+) second=(\d+) oracle_margin=([\d.]+) dev_gap=([\d.-]+) (ok|KNIFE|FAIL)", line)
        if m:
            tf.setdefault(m.group(1), []).append({"k": int(m.group(2)), "expected": int(m.group(3)), "got": int(m.group(4)),
                "device_second": int(m.group(5)), "oracle_margin": float(m.group(6)), "device_fp16_gap": float(m.group(7)), "status": m.group(8)})
        m = re.match(r"FR (\w+) got\((\d+)\)=\[([\d, ]*)\]", line)
        if m: fr.setdefault(m.group(1), {})["got"] = [int(x) for x in m.group(3).split(",") if x.strip()]
        m = re.match(r"FR (\w+) (PASS|FAIL) (.*)", line)
        if m: fr.setdefault(m.group(1), {}).update({"verdict": m.group(2), "reason": m.group(3).strip()})
        m = re.match(r"GATE prompt=(\w+) tf=(\d+)/(\d+) knife=(\d+) fail=(\d+) fr=(PASS|FAIL) tf_s=([\d.]+) fr_s=([\d.]+) verdict=(PASS|FAIL)", line)
        if m: gate[m.group(1)] = {"tf_ok": int(m.group(2)), "tf_n": int(m.group(3)), "knife": int(m.group(4)), "tf_fail": int(m.group(5)),
                                  "fr": m.group(6), "tf_seconds": float(m.group(7)), "fr_seconds": float(m.group(8)), "verdict": m.group(9)}
        m = re.match(r"GATE_SUMMARY (.*) VERDICT=(PASS|FAIL)", line)
        if m: summary = {"fields": m.group(1), "verdict": m.group(2)}
        m = re.match(r"engine loaded in ([\d.]+)s", line)
        if m: load = float(m.group(1))
    return tf, fr, gate, summary, load
tf, fr, gate, summary, load = parse(a.log)
out = {
    "schema": "ane-gate-device-transcript/1",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "bundle": a.bundle, "device": a.device, "engine": "Apple main 7359dbc StaticShapeEngine via EngineFactory (unmodified, AneGateRunner)",
    "oracle_model": fx["hf_id"], "oracle_dtype": fx["dtype"], "margin_floor": fx["floor"], "margin_definition": fx["margin_definition"],
    "rules": "TF: teacher-forced single-step argmax vs oracle at every step; mismatch on a step with oracle margin >= floor = FAIL, below = knife-edge (excluded). FR: free-running greedy judged at the first divergence by the oracle's margin there; the stop (EOS) is a step like any other. PASS needs both.",
    "engine_load_seconds": load, "note": a.note,
    "prompts": [{"name": p["name"], "prompt_text": p["prompt_text"], "chat": p["chat"], "prompt_ids": p["prompt_ids"],
                 "expected_ids": p["expected_ids"], "expected_text": p["expected_text"], "margins": p["margins"],
                 "stopped_on_eos": p["stopped_on_eos"], "tf": tf.get(p["name"], []), "fr": fr.get(p["name"], {}), "gate": gate.get(p["name"], {})}
                for p in fx["prompts"]],
    "summary": summary,
}
if a.red_log:
    rtf, rfr, rgate, rsum, _ = parse(a.red_log)
    rfx = json.load(open(a.fixture.replace("fixture.json", "fixture_red.json")))
    out["red_test"] = {"poison": rfx["poison"], "summary": rsum, "gate": rgate,
                       "purpose": "the same tool on a deliberately wrong fixture must go RED"}
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out, "verdict", summary and summary["verdict"], "red:", a.red_log and out["red_test"]["summary"]["verdict"])
