#!/usr/bin/env python3
"""d1-omni-600M decision latency on the Mac (Core AI Python runtime, AOT .aimodelc): every form of every workload in
one process, the forms of a workload alternated, two measurement windows -> the speed ladder's numbers.

    PY=~/code/coreai/coreai-models/.venv/bin/python
    ~/code/standup/tools/quiet/quiet_hold.py d1d-r7-run1 -- $PY conversion/d1_omni/timing_run.py window --run run1
    ~/code/standup/tools/quiet/quiet_hold.py d1d-r7-run2 -- $PY conversion/d1_omni/timing_run.py window --run run2
    $PY conversion/d1_omni/timing_run.py summarize                          # run1 + run2 -> ranking.json, r7_table.md
    $PY conversion/d1_omni/timing_run.py main --label dry --workloads W1 --forms dec-wfp16-L256-gpu --rounds 3 --warmup 1
    $PY conversion/d1_omni/timing_run.py main --label prewarm --rounds 1 --warmup 1      # every bundle once, no numbers

The rule is fixed before the first window in <work>/results/timing/ranking_rule.md (sha256 recorded in every output):
the workloads, the forms, what one decision counts and how the forms are ranked. In short:

  workloads  W1 one question (card_text/refund, 47 positions, L256; the chunk rows pad it to L512..4096), W2 three
             questions in one pass (card_text refund/team/urgency, 3 calls), W3 a 3.4k-token state (long_3400/
             tension_10, 3,454 positions, L4096), W4 a 384 px image (img_01/room: 1 crop -> 144 prefix rows, L256),
             W5 10 s of audio (aud_01/topic: 9.6 s -> the 10 s bucket, 121 prefix rows, L256; the chunk rows pad the
             clip to the 20 / 30 s buckets), W5s a 2.8 s clip (aud_04/topic: 5 s bucket, and padded to 10 s), W5L a
             28 s clip (aud_12/topic: 30 s bucket, 413 positions -> L512)
  decision   graph inputs (host.graph_inputs) -> rt.NDArray -> await the graph -> copy the output -> host gather +
             temperature + fp32 softmax; a media workload first runs its media graph (inputs prepared beforehand)
             and cuts the P prefix rows. Preprocessing (tokenize, crop resize + patchify + position table, mel) is
             timed apart, in the same process, never inside a decision
  protocol   every bundle loaded once in one process (AOT .aimodelc, explicit SpecializationOptions: gpu or
             neural_engine; never default() or cpu_only()); each workload: 35 rounds over its forms, forward on
             even rounds and reversed on odd ones, the first 5 are warm-up; 30 decisions per form; then its
             preprocessing 5 + 30 times. Every decision's probabilities are checked against the oracle row
  window     `window` is the measurement window's driver (run it under quiet_hold.py): snapshots of the lock (+ lsof),
             uptime, swap, the other GPU jobs and the busiest processes; up to 10 min of waiting for the other lanes'
             GPU jobs (the exception rule of memory #9 for jobs that do not read the lock); the main process; one
             fresh process per bundle (its load seconds and footprint); snapshots again
  ranking    `summarize`: score = the mean of the two windows' medians; ship candidates = forms whose every bundle
             passed its parity gate and whose every timed decision matched the oracle row (argmax, |dp| <= 0.02);
             every candidate within 3 % of the fastest is a co-candidate; a form whose two medians differ by >= 30 %
             is flagged

Outputs (never replaced: a .runN name is used): results/timing/<label>_main.json (+ logs/r7_<label>_main.trace.jsonl,
one line per decision, flushed), <label>_solo.json, <label>_window.json; summarize: results/timing/ranking.json and
r7_table.md. The numbers are this Mac's in these windows and are not compared with any other runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime
import gc
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import gpu_lock, hf_snapshot, work_path  # noqa: E402

import host  # noqa: E402

WORK = work_path("_d1_omni")
OUT = WORK / "results" / "timing"
RULE = OUT / "ranking_rule.md"
REFERENCES = {  # the oracle rows (rounds 2, 5, 6), pinned
    "main": ("ref/records_ref.json", "e7dba6f44d0452a6aea3e401746d417503056d6f468ff72b56c5511883436c5f"),
    "images": ("ref/records_ref_images.json", "681d2d10eef35ecb8980a3caf93d922a77a17e88ee58795468e78b915ac91b08"),
    "audio": ("ref/records_ref_audio.json", "c483f8c511cfcdb05a8211e495a7476b11f9318899a1125c387c9e722a6d8fbc"),
}
FIXTURES_SHA256 = "18b4ddff73a8044b63f949c69245210e1af8a26c48c5fdb785fd91a7807ead5b"  # v3 (round 6)
DEC_INPUTS = ("input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot")
ROUNDS, WARMUP = 35, 5
GPU_JOB_WAIT_S = 600
GPU_JOBS = re.compile(r"readout_gate|coreai_gate|/gate_[a-z0-9_]*\.py|export_[a-z0-9_]*\.py|runtime_check\.py|"
                      r"vision_check\.py|audio_check\.py|timing_run\.py|/timing\.py|llm-runner|llm-benchmark|"
                      r"dashboard_job\.py|decide-cli|coreai-build|coreai_verify|llm-bench|yardstick|mlx_lm|"
                      r"--accel gpu|litert_parity|xcodebuild")
WRAPPERS = re.compile(r"quiet_wait\.py|quiet_hold\.py|\bgrep\b|\bpgrep\b"
                      r"|--platform (android|ios)\b|/android/|\badb\b|devicectl|BENCH_ANDROID")  # + phone jobs (exempt)
# A GPU job is a process that can drive the GPU: an interpreter or a tool binary. A log viewer, a shell or a timeout
# wrapper whose arguments only mention such a path is not (round 7's first window waited on `tail -f …llm-bench…/x.log`).
GPU_CAPABLE = re.compile(r"^(python[\d.]*|Python|swift.*|coreai-build|xcodebuild|llm-runner|llm-benchmark|decide-cli|"
                         r"litert.*|.*_gate|.*-gate|.*bench.*|mlx.*)$", re.I)


# =========================================================================== registry: bundles and workloads
def bundle_registry() -> dict[str, dict]:
    """Every bundle the round times, by id: its compiled folder (AOT, rounds 3 and 7) and the compute unit it runs on.
    The two Neural Engine probes of round 7 join only when their compile kept a .aimodelc (regions > 0)."""
    reg: dict[str, dict] = {}

    def add(bid: str, folder: str, compute: str, kind: str, **extra):
        reg[bid] = {"id": bid, "folder": WORK / "compiled" / folder, "compute": compute, "kind": kind, **extra}

    add("dec-wfp16-L256-gpu", "wfp16-L256-h16c", "gpu", "decide", precision="wfp16", L=256)
    add("dec-fp16-L256-gpu", "fp16-L256-h16c", "gpu", "decide", precision="fp16", L=256)
    add("dec-fp32-L256-gpu", "fp32-L256-h16c-r7", "gpu", "decide", precision="fp32", L=256)
    add("dec-int8-L256-gpu", "int8-L256-h16c-r7", "gpu", "decide", precision="int8", L=256)
    add("dec-fp16-L256-ane", "fp16-L256-ane", "neural_engine", "decide", precision="fp16", L=256)
    add("dec-wfp16-L256-ane", "wfp16-L256-ane", "neural_engine", "decide", precision="wfp16", L=256)
    for p in ("wfp16", "fp16"):
        for L in (512, 1024, 2048, 4096):
            add(f"dec-{p}-L{L}-gpu", f"{p}-L{L}-h16c-r7", "gpu", "decide", precision=p, L=L)
    for p in ("fp32", "wfp16", "fp16"):
        add(f"vis-{p}-gpu", f"vision-{p}-h16c-r7", "gpu", "vision", precision=p)
    add("vis-fp16-ane", "vision-fp16-ane-r7", "neural_engine", "vision", precision="fp16", probe=True)
    add("aud-fp32-10s-gpu", "audio-fp32-10s-h16c-r7", "gpu", "audio", precision="fp32", sec=10)
    for p in ("wfp16", "fp16"):
        for s in (5, 10, 20, 30):
            add(f"aud-{p}-{s}s-gpu", f"audio-{p}-{s}s-h16c-r7", "gpu", "audio", precision=p, sec=s)
    add("aud-fp16-10s-ane", "audio-fp16-10s-ane-r7", "neural_engine", "audio", precision="fp16", sec=10, probe=True)
    # round 12: the stripped ship AOTs (aot_ship.py, compiled/ship-h16c/), the small decision buckets among them
    for L in (64, 128, 256):
        add(f"dec-fp16-L{L}-ship-gpu", f"ship-h16c/fp16-L{L}", "gpu", "decide", precision="fp16", L=L, stripped=True)
    for s in (5, 10):
        add(f"aud-fp16-{s}s-ship-gpu", f"ship-h16c/audio-fp16-{s}s", "gpu", "audio", precision="fp16", sec=s,
            stripped=True)
    return reg


def probe_kept(b: dict) -> bool:
    path = b["folder"] / "provenance" / "aot-manifest.json"
    if not path.exists():
        return False
    aot = json.loads(path.read_text())
    return aot.get("status") == "COMPILED" and bool(aot.get("aimodelc_kept"))


def workload_registry(reg: dict) -> dict[str, dict]:
    """The rule's workloads (ranking_rule.md §2, §3): the oracle row(s), the media, and the forms (tuples of bundle
    ids) alternated in one loop. A form whose Neural Engine probe kept no .aimodelc is left out (no-go, not timed)."""
    ane_vis = probe_kept(reg["vis-fp16-ane"])
    ane_aud = probe_kept(reg["aud-fp16-10s-ane"])
    dec256 = ["dec-wfp16-L256-gpu", "dec-fp16-L256-gpu", "dec-fp32-L256-gpu", "dec-int8-L256-gpu",
              "dec-fp16-L256-ane", "dec-wfp16-L256-ane"]
    chunk = [f"dec-{p}-L{L}-gpu" for L in (512, 1024, 2048, 4096) for p in ("wfp16", "fp16")]
    vis = ["vis-fp32-gpu", "vis-wfp16-gpu", "vis-fp16-gpu"] + (["vis-fp16-ane"] if ane_vis else [])
    aud10 = ["aud-fp32-10s-gpu", "aud-wfp16-10s-gpu", "aud-fp16-10s-gpu"] + (["aud-fp16-10s-ane"] if ane_aud else [])
    two = ["dec-wfp16-L256-gpu", "dec-fp16-L256-gpu"]
    return {
        "W1": {"what": "one question, short state", "kind": "text", "ref": "main", "record": "card_text",
               "mode": "text", "qids": ["refund"], "forms": [(d,) for d in dec256 + chunk]},
        "W2": {"what": "three questions in one pass (3 calls)", "kind": "text", "ref": "main", "record": "card_text",
               "mode": "text", "qids": ["refund", "team", "urgency"], "forms": [(d,) for d in dec256]},
        "W3": {"what": "one question over a 3.4k-token state", "kind": "text", "ref": "main", "record": "long_3400",
               "mode": "text", "qids": ["tension_10"], "forms": [("dec-wfp16-L4096-gpu",), ("dec-fp16-L4096-gpu",)]},
        "W4": {"what": "one question about a 384 px image (1 crop)", "kind": "image", "ref": "images",
               "record": "img_01", "mode": "image", "qids": ["room"],
               "forms": [(v, d) for v in vis for d in two]},
        "W5": {"what": "one question about 10 s of audio", "kind": "audio", "ref": "main", "record": "aud_01",
               "mode": "audio", "qids": ["topic"],
               "forms": [(a, d) for a in aud10 for d in two]
               + [(f"aud-{p}-{s}s-gpu", "dec-wfp16-L256-gpu") for s in (20, 30) for p in ("wfp16", "fp16")]},
        "W5s": {"what": "one question about a 2.8 s clip (the 5 s bucket, and padded to 10 s)", "kind": "audio",
                "ref": "audio", "record": "aud_04", "mode": "audio", "qids": ["topic"],
                "forms": [(f"aud-{p}-{s}s-gpu", "dec-wfp16-L256-gpu") for s in (5, 10) for p in ("wfp16", "fp16")]},
        "W5L": {"what": "one question about a 28 s clip (the 30 s bucket, L512)", "kind": "audio", "ref": "audio",
                "record": "aud_12", "mode": "audio", "qids": ["topic"],
                "forms": [(f"aud-{p}-30s-gpu", f"dec-{d}-L512-gpu") for p in ("wfp16", "fp16") for d in ("wfp16", "fp16")]},
    }


def workload_registry_r12(reg: dict) -> dict[str, dict]:
    """Round 12's workloads (ranking_rule_r12.md §2): round 7's rows on the stripped fp16 ship AOTs, the decision graph
    at L64 / L128 / L256 where the row fits (W5s's 98 positions not at L64; W5's 184 only at L256)."""
    dec = [f"dec-fp16-L{L}-ship-gpu" for L in (64, 128, 256)]
    return {
        "W1": {"what": "one question, short state", "kind": "text", "ref": "main", "record": "card_text",
               "mode": "text", "qids": ["refund"], "forms": [(d,) for d in dec]},
        "W2": {"what": "three questions in one pass (3 calls)", "kind": "text", "ref": "main", "record": "card_text",
               "mode": "text", "qids": ["refund", "team", "urgency"], "forms": [(d,) for d in dec]},
        "W5s": {"what": "one question about a 2.8 s clip (the 5 s bucket)", "kind": "audio", "ref": "audio",
                "record": "aud_04", "mode": "audio", "qids": ["topic"],
                "forms": [("aud-fp16-5s-ship-gpu", f"dec-fp16-L{L}-ship-gpu") for L in (128, 256)]},
        "W5": {"what": "one question about 10 s of audio", "kind": "audio", "ref": "main", "record": "aud_01",
               "mode": "audio", "qids": ["topic"], "forms": [("aud-fp16-10s-ship-gpu", "dec-fp16-L256-ship-gpu")]},
    }


# the form sets a run can time (--set): their workloads and the rule file fixed before their windows
SETS = {"r7": (workload_registry, OUT / "ranking_rule.md"), "r12": (workload_registry_r12, OUT / "ranking_rule_r12.md")}
SET = "r7"


def use_set(name: str) -> None:
    global SET, RULE
    SET, RULE = name, SETS[name][1]


def workloads_of(reg: dict) -> dict[str, dict]:
    return SETS[SET][0](reg)


def form_id(form: tuple) -> str:
    return "+".join(form)


# =========================================================================== small helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def free_name(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 100):
        candidate = path.with_name(f"{path.stem}.run{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"no free name next to {path}")


def write_new(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = free_name(path)
    out.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    return out


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _command(argv: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable ({type(error).__name__})"


def environment() -> dict:
    from importlib import metadata

    versions = {}
    for name in ("coreai-core", "numpy", "torch", "torchvision", "pillow", "soundfile", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"date": now(), "machine": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_bytes": int(_command(["sysctl", "-n", "hw.memsize"]) or 0),
            "macos_version": _command(["sw_vers", "-productVersion"]),
            "macos_build": _command(["sw_vers", "-buildVersion"]), "python": platform.python_version(),
            "executable": sys.executable, "packages": versions, "argv": sys.argv, "pid": os.getpid()}


def rule_info() -> dict:
    st = RULE.stat()
    return {"path": str(RULE), "sha256": sha256_file(RULE),
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")}


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


# --------------------------------------------------------------------------- memory
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint() -> dict:
    """ru_maxrss (bytes on macOS) and proc_pid_rusage(RUSAGE_INFO_V4): resident, phys_footprint, lifetime max."""
    out = {"ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(os.getpid(), 4, buf) == 0:
        u64 = lambda offset: int.from_bytes(buf.raw[offset:offset + 8], "little")  # noqa: E731
        out.update(resident_bytes=u64(64), phys_footprint_bytes=u64(72), lifetime_max_phys_footprint_bytes=u64(240))
    return out


# --------------------------------------------------------------------------- the machine around the numbers
def lock_state() -> dict:
    path = gpu_lock()
    try:
        content = path.read_text().strip()
    except OSError:
        content = None
    holders = _command(["lsof", "-t", str(path)]).split()
    return {"path": str(path), "content": content, "lsof_pids": holders}


def _ancestors(pid: int) -> set[int]:
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        parent = _command(["ps", "-o", "ppid=", "-p", str(pid)])
        pid = int(parent) if parent.strip().isdigit() else 1
    return seen


def process_table() -> list[dict]:
    rows = []
    for line in _command(["ps", "-Ao", "pid=,ppid=,stat=,%cpu=,rss=,etime=,command="]).splitlines():
        parts = line.split(None, 6)
        if len(parts) == 7 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "stat": parts[2], "cpu": float(parts[3]),
                         "rss_kb": int(parts[4]), "etime": parts[5], "command": parts[6]})
    return rows


def gpu_jobs(table: list[dict] | None = None, stopped: list | None = None) -> list[str]:
    """Other lanes' GPU-capable jobs by command line (read only): this process's ancestors and descendants, agent
    CLIs, shells and the lock wrappers (a waiter is not a job, memory #9) are left out. With `stopped` (a list), a job
    in the stopped state (ps STAT T: paused by its lane's wrapper while a window is open) goes there instead
    (ranking_rule_r8_addendum_1.md: it cannot drive the GPU while the window lasts)."""
    table = table if table is not None else process_table()
    mine = _ancestors(os.getpid())
    children = {r["pid"]: r["ppid"] for r in table}

    def descends(pid: int) -> bool:
        seen = set()
        while pid > 1 and pid not in seen:
            if pid == os.getpid():
                return True
            seen.add(pid)
            pid = children.get(pid, 1)
        return False

    out = []
    for r in table:
        cmd = r["command"]
        name = Path(cmd.split()[0]).name.lstrip("-") if cmd else ""
        if (r["pid"] in mine or descends(r["pid"]) or "claude" in name.lower() or name in ("zsh", "bash", "sh", "fish")
                or not GPU_CAPABLE.match(name) or WRAPPERS.search(cmd) or not GPU_JOBS.search(cmd)):
            continue
        line = f"{r['pid']} {r['etime']} cpu {r['cpu']:.0f}% {' '.join(cmd.split())[:220]}"
        if stopped is not None and r["stat"].startswith("T"):
            stopped.append(f"{line} (stat {r['stat']})")
            continue
        out.append(line)
    return out


def snapshot() -> dict:
    table = process_table()
    busiest = sorted(table, key=lambda r: -r["cpu"])[:8]
    return {"at": now(), "lock": lock_state(), "loadavg": list(os.getloadavg()), "uptime": _command(["uptime"]),
            "swapusage": _command(["sysctl", "-n", "vm.swapusage"]),
            "memory_pressure": _command(["memory_pressure"]).splitlines()[-1:],
            "disk_free": _command(["df", "-h", "/System/Volumes/Data"]).splitlines()[-1:],
            "gpu_jobs": gpu_jobs(table),
            "busiest": [f"{r['pid']} cpu {r['cpu']:.0f}% rss {r['rss_kb'] // 1024} MB {' '.join(r['command'].split())[:160]}"
                        for r in busiest]}


# =========================================================================== inputs: rows, oracle, media
def snapshot_dir() -> Path:
    return Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))


def load_references() -> dict:
    refs = {}
    for key, (rel, sha) in REFERENCES.items():
        path = WORK / rel
        if sha256_file(path) != sha:
            raise SystemExit(f"{path} is not the pinned reference ({sha[:8]}…)")
        refs[key] = json.loads(path.read_text())
    return refs


def oracle_row(entry: dict, q: dict, question: host.Question) -> dict:
    """A row in the shape _metrics.row_record reads."""
    return {"id": entry["id"], "qid": q["qid"], "mode": entry["mode"], "source": entry.get("source"), "type": q["type"],
            "K": q["K"], "positions": q["positions"], "near_tie": q["near_tie"], "top2_margin": q["top2_margin"],
            "argmax_index": q["argmax_index"], "calibrate": q["calibrate"], "bucket": q["bucket"],
            "question": {"type": question.type, "instructions": question.instructions, "criteria": question.criteria},
            "oracle": {"logits_raw": q["logits_raw"], "probs": q["probs"]}}


class Case:
    """One workload's inputs: the host rows built from the fixture request (asserted equal to the oracle's ids and
    markers), the oracle rows, the media and its prepared graph inputs, and the preprocessing callables."""

    def __init__(self, wid: str, spec: dict, refs: dict, fixtures: dict, tok, table, reg: dict):
        from PIL import Image

        self.wid, self.spec = wid, spec
        record = fixtures[spec["record"]]
        request = record["request"]
        entry = next(e for e in refs[spec["ref"]]["records"] if e["id"] == spec["record"] and e["mode"] == spec["mode"])
        qs = {qid: request["questions"][qid] for qid in spec["qids"]}
        self.state, self.questions = request["state"], qs
        self.media_info: dict = {}
        if spec["kind"] == "text":
            self.rows = host.request_rows(tok, self.state, qs, "text")
        elif spec["kind"] == "image":
            path = WORK / record["media"]["images"][0]
            expected = (record["provenance"].get("image") or {}).get("sha256")
            digest = sha256_file(path)
            assert expected in (None, digest), (path, digest, expected)
            with Image.open(path) as im:
                im.load()
                self.image = im.copy()
            self.rows = host.image_request_rows(tok, self.state, qs, [self.image])
            self.crops = host.image_crops_inputs(self.image, table)
            self.media_info = {"file": str(path.relative_to(WORK)), "sha256": digest, "px": list(self.image.size),
                               "crops": len(self.crops), "tokens": [c["tokens"] for c in self.crops]}
        else:
            import soundfile as sf

            path = WORK / record["media"]["audio"]
            prov = record["provenance"]
            expected = (prov.get("clip") or prov.get("audio") or {}).get("sha256")
            digest = sha256_file(path)
            assert expected in (None, digest), (path, digest, expected)
            samples, rate = sf.read(str(path), dtype="int16")
            assert rate == host.SAMPLE_RATE and samples.ndim == 1, (path, rate, samples.shape)
            self.samples = samples
            self.rows = host.audio_request_rows(tok, self.state, qs, samples)
            secs = sorted({reg[b]["sec"] for form in spec["forms"] for b in form if reg[b]["kind"] == "audio"})
            self.audio = {sec: host.audio_inputs(samples, sec) for sec in secs}
            self.media_info = {"file": str(path.relative_to(WORK)), "sha256": digest, "samples": int(len(samples)),
                               "seconds": len(samples) / host.SAMPLE_RATE, "own_bucket_s": host.audio_bucket_for(len(samples)),
                               "buckets_timed": secs, "prefix_rows": int(self.audio[secs[0]]["prefix_rows"])}
        by_qid = {q["qid"]: q for q in entry["questions"]}
        self.oracle = []
        for row in self.rows:
            q = by_qid[row.qid]
            if row.ids != q["ids"] or row.markers != q["markers"] or row.positions != q["positions"]:
                raise SystemExit(f"{wid}: host row {row.qid} differs from the oracle's ids / markers / positions")
            self.oracle.append(oracle_row(entry, q, row.question))
        self.describe = {"record": spec["record"], "mode": spec["mode"], "qids": spec["qids"],
                         "positions": [r.positions for r in self.rows], "prefix": [r.prefix_len for r in self.rows],
                         "native_bucket": [host.bucket_for(r.positions) for r in self.rows], **self.media_info}
        if SET != "r7":  # round 12: the bucket each row takes when the host also ships L64 / L128
            self.describe["native_bucket_with_small"] = [host.bucket_for(r.positions, host.ALL_BUCKETS) for r in self.rows]

    # ---- preprocessing (timed apart, never inside a decision)
    def preprocess_steps(self, tok, table) -> dict:
        """name -> zero-argument callable: what a host does before its first graph call."""
        steps = {}
        if self.spec["kind"] == "text":
            steps["tokenize"] = lambda: host.request_rows(tok, self.state, self.questions, "text")
        elif self.spec["kind"] == "image":
            steps["tokenize"] = lambda: host.image_request_rows(tok, self.state, self.questions, [self.image])
            steps["crops_patchify_positions"] = lambda: host.image_crops_inputs(self.image, table)
            path = WORK / self.media_info["file"]

            def decode():
                from PIL import Image

                with Image.open(path) as im:
                    im.load()
                    return im.size
            steps["file_decode_png"] = decode
        else:
            sec = self.media_info["own_bucket_s"]
            steps["tokenize"] = lambda: host.audio_request_rows(tok, self.state, self.questions, self.samples)
            steps["mel_torch_and_masks"] = lambda: host.audio_inputs(self.samples, sec)
            steps["mel_numpy_and_masks"] = lambda: host.audio_inputs(self.samples, sec, numpy_mel=True)
            path = WORK / self.media_info["file"]

            def decode():
                import soundfile as sf

                return len(sf.read(str(path), dtype="int16")[0])
            steps["file_decode_wav"] = decode
        return steps


# =========================================================================== one decision
async def decide_rows(dfn, rows, L, prefix, parts) -> list[np.ndarray]:
    """The decision graph over each row (one call per row): inputs, the call, the host. Returns each row's marker
    logits; adds the part timings (ms) into `parts`."""
    out = []
    for row in rows:
        t0 = time.perf_counter()
        inputs, markers = host.graph_inputs(row, L, prefix)
        t1 = time.perf_counter()
        res = await dfn({k: rt.NDArray(inputs[k]) for k in DEC_INPUTS})
        scores = np.array(res["scores"].numpy(), copy=True).reshape(-1)
        t2 = time.perf_counter()
        logits = scores[markers]
        host.probabilities_from_logits(logits, row.question, row.calibrate)
        t3 = time.perf_counter()
        parts["inputs_ms"] += (t1 - t0) * 1e3
        parts["graph_ms"] += (t2 - t1) * 1e3
        parts["host_ms"] += (t3 - t2) * 1e3
        out.append(logits)
    return out


async def decide(case: Case, form: tuple, fns: dict, reg: dict) -> tuple[float, dict, list[np.ndarray]]:
    """One decision of the workload with this form: (total ms, parts, marker logits per row)."""
    parts = {"media_ms": 0.0, "inputs_ms": 0.0, "graph_ms": 0.0, "host_ms": 0.0}
    t0 = time.perf_counter()
    if case.spec["kind"] == "text":
        b = reg[form[0]]
        logits = await decide_rows(fns[form[0]], case.rows, b["L"], None, parts)
    elif case.spec["kind"] == "image":
        vfn, b = fns[form[0]], reg[form[1]]
        pieces = []
        for crop in case.crops:
            res = await vfn({k: rt.NDArray(crop[k]) for k in host.VISION_INPUT_NAMES})
            pieces.append(np.array(res["prefix"].numpy(), copy=True).reshape(host.VISION_TOKENS, -1)[:crop["tokens"]])
        prefix = np.concatenate(pieces, axis=0)
        parts["media_ms"] = (time.perf_counter() - t0) * 1e3
        logits = await decide_rows(fns[form[1]], case.rows, b["L"], prefix, parts)
    else:
        afn, a, b = fns[form[0]], reg[form[0]], reg[form[1]]
        inputs = case.audio[a["sec"]]
        res = await afn({k: rt.NDArray(inputs[k]) for k in host.AUDIO_INPUT_NAMES})
        prefix = np.array(res["prefix"].numpy(), copy=True).reshape(inputs["steps"], -1)[:inputs["prefix_rows"]]
        parts["media_ms"] = (time.perf_counter() - t0) * 1e3
        logits = await decide_rows(fns[form[1]], case.rows, b["L"], prefix, parts)
    return (time.perf_counter() - t0) * 1e3, parts, logits


def output_check(case: Case, calls: list[list[np.ndarray]]) -> dict:
    """Every timed decision's probabilities against the oracle row(s): argmax equal and max |dp| over all calls; drift
    = the largest marker-logit change from the first call."""
    from _metrics import SHIP_BAR, row_record

    worst_dp, worst_dl, all_argmax, drift = 0.0, 0.0, True, 0.0
    first = calls[0]
    for call in calls:
        for row, z, z0 in zip(case.oracle, call, first):
            rec = row_record(row, z)
            worst_dp = max(worst_dp, rec["max_abs_dp"])
            worst_dl = max(worst_dl, rec["max_abs_dlogit"])
            all_argmax = all_argmax and rec["argmax_equal"]
            drift = max(drift, float(np.max(np.abs(np.asarray(z, np.float64) - np.asarray(z0, np.float64)))))
    probs_first = [row_record(row, z)["probs"] for row, z in zip(case.oracle, first)]
    ok = all_argmax and worst_dp <= SHIP_BAR["max_abs_dp"]
    return {"calls": len(calls), "rows_per_call": len(case.oracle), "argmax_equal_every_call": all_argmax,
            "max_abs_dp": worst_dp, "max_abs_dlogit": worst_dl, "drift_marker_logits": drift,
            "status": "PASS" if ok else "FAIL", "bar": "argmax equal and |dp| <= 0.02 on every timed call",
            "first_call_probs": probs_first, "oracle_probs": [r["oracle"]["probs"] for r in case.oracle]}


# =========================================================================== the main process
def verify_bundle(b: dict) -> tuple[Path, dict]:
    aot = json.loads((b["folder"] / "provenance" / "aot-manifest.json").read_text())
    if aot.get("status") != "COMPILED" or not aot.get("aimodelc_kept", True):
        raise SystemExit(f"{b['id']}: no compiled bundle in {b['folder']}")
    aimodelc = b["folder"] / aot["aimodelc"]
    for item in aot["files"]:
        file = aimodelc / item["path"]
        if file.stat().st_size != item["bytes"] or sha256_file(file) != item["sha256"]:
            raise SystemExit(f"{file} changed since its compile")
    return aimodelc, aot


def options_for(compute: str):
    kind = rt.ComputeUnitKind.gpu() if compute == "gpu" else rt.ComputeUnitKind.neural_engine()
    return rt.SpecializationOptions.from_preferred_compute_unit_kind(kind)


def select(args, reg: dict) -> tuple[dict, list[str]]:
    works = workloads_of(reg)
    wanted = [w.strip() for w in args.workloads.split(",")] if args.workloads else list(works)
    forms_filter = {f.strip() for f in args.forms.split(",")} if args.forms else None
    chosen = {}
    for wid in wanted:
        spec = dict(works[wid])
        forms = spec["forms"]
        if forms_filter is not None:  # a form id, or the bundle id of a one-bundle form
            forms = [f for f in forms if form_id(f) in forms_filter or (len(f) == 1 and f[0] in forms_filter)]
        if forms:
            spec["forms"] = forms
            chosen[wid] = spec
    needed = [bid for bid in reg if any(bid in f for spec in chosen.values() for f in spec["forms"])]
    return chosen, needed


async def main_async(args) -> dict:
    t_start = time.perf_counter()
    reg = bundle_registry()
    works, needed = select(args, reg)
    trace = open(args.trace, "a", buffering=1) if args.trace else None

    def note(**event):
        if trace is not None:
            trace.write(json.dumps({"t": now(), "label": args.label, **event}, ensure_ascii=False) + "\n")

    note(event="start", pid=os.getpid(), workloads=list(works), bundles=needed)
    doc = {"schema": "d1-omni-timing/1", "label": args.label, "rule": rule_info(), "rounds": args.rounds,
           "warmup": args.warmup, "started": now(), "environment": environment(),
           "code_sha256": {f: sha256_file(HERE / f) for f in ("timing_run.py", "host.py", "_metrics.py", "mel_host.py")},
           "memory_start": footprint()}
    # bundles: verify against the compile manifest, then load each once with its explicit compute unit
    t0 = time.perf_counter()
    verified = {bid: verify_bundle(reg[bid]) for bid in needed}
    doc["verify_seconds"] = time.perf_counter() - t0
    fns, models, bundles = {}, {}, {}
    for bid in needed:
        b, (aimodelc, aot) = reg[bid], verified[bid]
        opts = options_for(b["compute"])
        before = footprint()
        t0 = time.perf_counter()
        model = await rt.AIModel.load(aimodelc, opts)
        load_s = time.perf_counter() - t0
        if list(model.function_names) != ["main"]:
            raise SystemExit(f"{bid}: functions {model.function_names}")
        models[bid], fns[bid] = model, model.load_function("main")
        after = footprint()
        bundles[bid] = {"folder": str(b["folder"]), "aimodelc": aimodelc.name, "compute": b["compute"],
                        "specialization_options": " ".join(str(opts).split()), "kind": b["kind"],
                        "precision": b["precision"], "L": b.get("L"), "sec": b.get("sec"), "bytes": aot["bytes"],
                        "resources_bin_bytes": aot["resources_bin_bytes"], "main_hash": aot["hashes"].get("main.hash"),
                        "ane_regions": aot["ane"]["regions"], "load_s": load_s,
                        "phys_footprint_delta_bytes": after.get("phys_footprint_bytes", 0) - before.get("phys_footprint_bytes", 0),
                        "ru_maxrss_after_bytes": after["ru_maxrss_bytes"]}
        note(event="loaded", bundle=bid, load_s=load_s)
        print(f"loaded {bid} in {load_s:.2f} s", flush=True)
    doc["bundles"] = bundles
    doc["memory_after_load"] = footprint()
    # inputs
    snap = snapshot_dir()
    tok = host.RawTokenizer(snap / "tokenizer.json")
    host.check_token_ids(tok)
    table = host.load_position_table(snap / "model.safetensors")
    from _fixtures import fixtures_path

    fixtures = {r["id"]: r for r in json.loads(fixtures_path(FIXTURES_SHA256).read_text())["records"]}
    refs = load_references()
    doc["workloads"] = {}
    for wid, spec in works.items():
        case = Case(wid, spec, refs, fixtures, tok, table, reg)
        forms = spec["forms"]
        samples = {form_id(f): [] for f in forms}
        warm = {form_id(f): [] for f in forms}
        part_samples = {form_id(f): [] for f in forms}
        calls = {form_id(f): [] for f in forms}
        order_log = []
        gc.collect()
        gc.disable()  # no cycle collection inside a timed decision (reference counting still frees every array)
        t_w = time.perf_counter()
        for r in range(args.rounds):
            order = forms if r % 2 == 0 else list(reversed(forms))
            order_log.append([form_id(f) for f in order][:1])
            for f in order:
                fid = form_id(f)
                ms, parts, logits = await decide(case, f, fns, reg)
                calls[fid].append(logits)
                if r < args.warmup:
                    warm[fid].append(ms)
                else:
                    samples[fid].append(ms)
                    part_samples[fid].append(parts)
                note(event="decision", w=wid, form=fid, round=r, warm=r < args.warmup, ms=ms,
                     **{k: round(v, 4) for k, v in parts.items()})
        loop_s = time.perf_counter() - t_w
        gc.enable()
        result_forms = {}
        for f in forms:
            fid = form_id(f)
            ps = part_samples[fid]
            result_forms[fid] = {
                "bundles": list(f), "compute": [reg[b]["compute"] for b in f],
                "stats_ms": stats(samples[fid]) if samples[fid] else None, "samples_ms": samples[fid],
                "warmup_ms": warm[fid],
                "parts_median_ms": {k: float(np.median([p[k] for p in ps])) for k in ps[0]} if ps else None,
                "output_check": output_check(case, calls[fid])}
        # preprocessing, timed apart
        pre = {}
        for name, fn in case.preprocess_steps(tok, table).items():
            times = []
            for i in range(args.warmup + max(args.rounds - args.warmup, 1)):
                t0 = time.perf_counter()
                fn()
                if i >= args.warmup or args.rounds <= args.warmup:
                    times.append((time.perf_counter() - t0) * 1e3)
            pre[name] = stats(times)
        doc["workloads"][wid] = {"what": spec["what"], "case": case.describe, "forms": result_forms,
                                 "form_order_even_rounds": [form_id(f) for f in forms],
                                 "preprocess_ms": pre, "loop_seconds": loop_s, "snapshot_after": snapshot(),
                                 "memory_after": footprint()}
        medians = [(v["stats_ms"]["median"], k) for k, v in result_forms.items() if v["stats_ms"]]
        print(f"{wid}: {len(forms)} forms x {args.rounds} rounds in {loop_s:.1f} s"
              + (f"; fastest median {min(medians)[0]:.2f} ms ({min(medians)[1]})" if medians else ""), flush=True)
        note(event="workload_done", w=wid, loop_s=loop_s)
    doc["memory_end"] = footprint()
    doc["seconds"] = time.perf_counter() - t_start
    doc["finished"] = now()
    assert models  # every model stays alive until the last output is copied
    note(event="end", seconds=doc["seconds"])
    return doc


def cmd_main(args) -> int:
    doc = asyncio.run(main_async(args))
    out = write_new(Path(args.out) if args.out else OUT / f"{args.label}_main.json", doc)
    print("->", out, flush=True)
    return 0


# =========================================================================== one bundle in a fresh process
async def solo_async(args) -> dict:
    reg = bundle_registry()
    b = reg[args.bundle]
    t_proc = time.perf_counter()
    aimodelc, aot = verify_bundle(b)
    mem0 = footprint()
    opts = options_for(b["compute"])
    t0 = time.perf_counter()
    model = await rt.AIModel.load(aimodelc, opts)
    load_s = time.perf_counter() - t0
    fn = model.load_function("main")
    mem_load = footprint()
    snap = snapshot_dir()
    tok = host.RawTokenizer(snap / "tokenizer.json")
    table = host.load_position_table(snap / "model.safetensors")
    from _fixtures import fixtures_path

    fixtures = {r["id"]: r for r in json.loads(fixtures_path(FIXTURES_SHA256).read_text())["records"]}
    refs = load_references()
    works = workloads_of(reg)
    if b["kind"] == "decide":
        spec = dict(works["W1"], forms=[(args.bundle,)])
    elif b["kind"] == "vision":
        spec = dict(works["W4"], forms=[(args.bundle, "dec-wfp16-L256-gpu")])
    else:
        spec = dict(works["W5"] if b["sec"] >= 10 else works["W5s"], forms=[(args.bundle, "dec-wfp16-L256-gpu")])
    case = Case("solo", spec, refs, fixtures, tok, table, reg)

    async def media_or_decision() -> float:
        t = time.perf_counter()
        if b["kind"] == "decide":
            await decide_rows(fn, case.rows, b["L"], None, {"inputs_ms": 0.0, "graph_ms": 0.0, "host_ms": 0.0})
        elif b["kind"] == "vision":
            for crop in case.crops:
                res = await fn({k: rt.NDArray(crop[k]) for k in host.VISION_INPUT_NAMES})
                np.array(res["prefix"].numpy(), copy=True)
        else:
            res = await fn({k: rt.NDArray(case.audio[b["sec"]][k]) for k in host.AUDIO_INPUT_NAMES})
            np.array(res["prefix"].numpy(), copy=True)
        return (time.perf_counter() - t) * 1e3

    first_ms = await media_or_decision()
    calls_ms = [await media_or_decision() for _ in range(5)]
    mem_end = footprint()
    assert model is not None
    return {"bundle": args.bundle, "kind": b["kind"], "compute": b["compute"], "aimodelc": str(aimodelc),
            "bytes": aot["bytes"], "resources_bin_bytes": aot["resources_bin_bytes"], "load_s": load_s,
            "first_call_ms": first_ms, "next_5_calls_ms": calls_ms,
            "what_is_called": "decide: W1's row at the bundle's L; vision: img_01's crop; audio: aud_01 (aud_04 for 5 s) "
                              "at the bundle's bucket (the media graph alone)",
            "memory": {"before_load": mem0, "after_load": mem_load, "end": mem_end},
            "process_seconds": time.perf_counter() - t_proc, "pid": os.getpid(), "at": now()}


def cmd_solo(args) -> int:
    doc = asyncio.run(solo_async(args))
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{args.bundle}: load {doc['load_s']:.2f} s, footprint end "
          f"{doc['memory']['end'].get('phys_footprint_bytes', 0) / 2**20:.0f} MiB", flush=True)
    return 0


# =========================================================================== the window driver
def wait_gpu_quiet(max_s: float, skip_stopped: bool = False) -> dict:
    """Wait up to max_s for the other lanes' GPU jobs to end; skip_stopped (round 12, ranking_rule_r8_addendum_1.md):
    a stopped job is recorded in stopped_jobs, not waited on."""
    t0 = time.monotonic()
    seen: list[str] = []
    paused: list[str] = []
    while True:
        stopped = [] if skip_stopped else None
        jobs = gpu_jobs(stopped=stopped)
        paused = sorted(set(paused) | set(stopped or []))
        extra = {"stopped_jobs": paused} if skip_stopped else {}
        if not jobs:
            return {"waited_s": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": True, **extra}
        seen = sorted(set(seen) | set(jobs))
        if time.monotonic() - t0 > max_s:
            return {"waited_s": round(time.monotonic() - t0, 1), "jobs_seen": seen, "quiet": False, "left": jobs,
                    **extra}
        print(f"waiting for {len(jobs)} GPU job(s): {jobs[0][:140]}", flush=True)
        time.sleep(15)


def cmd_window(args) -> int:
    label = args.run
    lock_label = args.lock_label or f"d1d-r7-{label}"
    sub_set = ["--set", SET] if SET != "r7" else []  # the form set, passed on to every process of the window
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "d1-omni-timing-window/1", "run": label, "set": SET, "started": now(), "rule": rule_info(),
           "expected_lock_label": lock_label, "lock_at_start": lock_state(), "snapshots": {}}
    content = doc["lock_at_start"]["content"] or ""
    doc["lock_is_ours"] = content.startswith(f"{lock_label} timing")
    if not doc["lock_is_ours"] and not args.no_lock:
        print(f"the lock holds {content!r}, not {lock_label}: run this under quiet_hold.py", flush=True)
        return 2
    doc["snapshots"]["before"] = snapshot()
    # round 12: a stopped job of another lane is recorded, not waited on (ranking_rule_r8_addendum_1.md)
    doc["gpu_quiet_wait"] = wait_gpu_quiet(GPU_JOB_WAIT_S if not args.no_wait else 0, skip_stopped=SET != "r7")
    doc["contended"] = not doc["gpu_quiet_wait"]["quiet"]
    doc["snapshots"]["after_wait"] = snapshot()
    if not args.no_smoke:
        # every form once (1 round, its numbers kept apart and never ranked): the code paths run and every bundle's
        # compile-cache entry exists before the measured process loads them (ranking_rule_addendum_1.md)
        smoke_out = free_name(WORK / "logs" / f"r7_{label}_smoke_main.json")
        argv = [sys.executable, str(Path(__file__).resolve()), "main", "--label", f"{label}-smoke", "--out", str(smoke_out),
                "--rounds", "1", "--warmup", "1"] + (["--workloads", args.workloads] if args.workloads else []) + sub_set
        t0 = time.monotonic()
        with open(WORK / "logs" / f"r7_{label}_smoke_main.log", "a") as fh:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT).returncode
        doc["smoke"] = {"rc": rc, "wall_s": time.monotonic() - t0, "out": str(smoke_out), "argv": argv,
                        "note": "every form once; not a measurement (never ranked)"}
        print(f"smoke: rc {rc}, {doc['smoke']['wall_s']:.0f} s", flush=True)
        if rc != 0:
            doc["snapshots"]["after_smoke"] = snapshot()
            doc["lock_at_end"] = lock_state()
            doc["finished"] = now()
            out = write_new(out_dir / f"{label}_window.json", doc)
            print("smoke failed: no measurement ->", out, flush=True)
            return rc
        doc["snapshots"]["after_smoke"] = snapshot()
    main_out = free_name(out_dir / f"{label}_main.json")
    trace = WORK / "logs" / f"r7_{label}_main.trace.jsonl"
    log = WORK / "logs" / f"r7_{label}_main.log"
    argv = [sys.executable, str(Path(__file__).resolve()), "main", "--label", label, "--out", str(main_out),
            "--trace", str(trace), "--rounds", str(args.rounds), "--warmup", str(args.warmup)] + sub_set
    if args.workloads:
        argv += ["--workloads", args.workloads]
    t0 = time.monotonic()
    with open(log, "a") as fh:
        rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT).returncode
    doc["main"] = {"rc": rc, "wall_s": time.monotonic() - t0, "out": str(main_out), "trace": str(trace),
                   "log": str(log), "argv": argv}
    print(f"main: rc {rc}, {doc['main']['wall_s']:.0f} s", flush=True)
    doc["snapshots"]["after_main"] = snapshot()
    solo = {}
    if rc == 0 and not args.no_solo:
        loaded = list(json.loads(main_out.read_text())["bundles"])
        tmp_dir = WORK / "logs" / f"r7_{label}_solo"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for bid in loaded:
            out = tmp_dir / f"{bid}.json"
            t1 = time.monotonic()
            with open(tmp_dir / f"{bid}.log", "a") as fh:
                rc_s = subprocess.run([sys.executable, str(Path(__file__).resolve()), "solo", "--bundle", bid, "--out",
                                       str(out)] + sub_set, stdout=fh, stderr=subprocess.STDOUT).returncode
            solo[bid] = json.loads(out.read_text()) if rc_s == 0 and out.exists() else {"rc": rc_s}
            solo[bid]["process_wall_s"] = time.monotonic() - t1
            print(f"solo {bid}: rc {rc_s}", flush=True)
        doc["solo_out"] = str(write_new(out_dir / f"{label}_solo.json", {"run": label, "rule": rule_info(), "bundles": solo}))
    doc["snapshots"]["after_solo"] = snapshot()
    doc["lock_at_end"] = lock_state()
    doc["finished"] = now()
    out = write_new(out_dir / f"{label}_window.json", doc)
    print("->", out, flush=True)
    if rc == 0 and args.summarize:
        rc_z = subprocess.run([sys.executable, str(Path(__file__).resolve()), "summarize"]).returncode
        print(f"summarize: rc {rc_z}", flush=True)
    return rc


# =========================================================================== the ranking (ranking_rule.md §5)
def _gate_doc(paths: list[Path], want: dict) -> tuple[Path | None, dict | None]:
    for path in paths:
        d = json.loads(path.read_text())
        dec = {v["precision"] for v in d.get("decide", {}).values()}
        if (d.get("compute") == want["compute"] and d.get("decide_compute", d.get("compute")) == "gpu"
                and dec == {want["decide"]}):
            return path, d
    return None, None


def parity_of(form: tuple, reg: dict) -> dict:
    """The parity verdict of a form = its bundles' gate JSONs (ranking_rule.md §6)."""
    res = WORK / "results"
    if len(form) == 1:
        b = reg[form[0]]
        unit = "gpu" if b["compute"] == "gpu" else "neural_engine"
        path = res / f"runtime_{b['precision']}_L{b['L']}_{unit}.json"
        d = json.loads(path.read_text())
        s = d["summary"]
        return {"status": s["ship_bar"]["status"], "json": str(path.relative_to(WORK)),
                "text": f"{s['ship_bar']['status']} {s['argmax_equal']}/{s['rows']}, max|Δp| {s['max_abs_dp']:.3g}, "
                        f"mean {s['mean_row_max_abs_dp']:.3g}" + (f", drift {d['repeat_drift']['max_abs_scores']:.3g}"
                                                               if d['repeat_drift']['max_abs_scores'] else "")}
    media, dec = reg[form[0]], reg[form[1]]
    kind = media["kind"]
    want = {"compute": media["compute"], "decide": dec["precision"]}
    paths = sorted(res.glob(f"{kind}_e2e_{media['precision']}*.json"))
    path, d = _gate_doc(paths, want)
    if d is None:
        return {"status": "MISSING", "json": None, "text": f"no {kind} end-to-end gate with {want}"}
    arm = "vision_bundle_prefix" if kind == "vision" else "audio_bundle_prefix_torch_mel"
    s = d["summaries"][arm]
    covered = ""
    if kind == "audio":
        buckets = sorted(int(k) for k in d["audio"])
        covered = f" (audio buckets {buckets} s)"
        if media["sec"] not in buckets:
            return {"status": "MISSING", "json": str(path.relative_to(WORK)), "text": f"bucket {media['sec']} s not gated"}
    return {"status": d["status"], "json": str(path.relative_to(WORK)),
            "text": f"{d['status']} end to end {s['argmax_equal']}/{s['rows']}, max|Δp| {s['max_abs_dp']:.3g}, "
                    f"mean {s['mean_row_max_abs_dp']:.3g}{covered}"}


def cmd_summarize(args) -> int:
    reg = bundle_registry()
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    runs = {}
    for label in ("run1", "run2"):
        path = Path(getattr(args, label)) if getattr(args, label) else OUT / f"{label}_main.json"
        runs[label] = (path, json.loads(path.read_text()))
    solos = {}
    for label in ("run1", "run2"):
        main_path = runs[label][0]
        path = main_path.with_name(main_path.name.replace("_main", "_solo"))
        if path.exists():
            solos[label] = json.loads(path.read_text())["bundles"]
    rules = {k: v["rule"]["sha256"] for k, (p, v) in runs.items()}
    if len(set(rules.values()) | {sha256_file(RULE)}) != 1:
        raise SystemExit(f"the rule file changed between the windows: {rules}")
    out = {"schema": "d1-omni-ranking/1", "rule": rule_info(), "runs": {k: str(p) for k, (p, _) in runs.items()},
           "workloads": {}}
    for wid in runs["run1"][1]["workloads"]:
        w1, w2 = runs["run1"][1]["workloads"][wid], runs["run2"][1]["workloads"].get(wid)
        rows = []
        for fid, f1 in w1["forms"].items():
            f2 = w2["forms"].get(fid) if w2 else None
            form = tuple(f1["bundles"])
            m1 = f1["stats_ms"]["median"]
            m2 = f2["stats_ms"]["median"] if f2 else None
            score = (m1 + m2) / 2 if m2 is not None else None
            parity = parity_of(form, reg)
            checks = [f1["output_check"]["status"]] + ([f2["output_check"]["status"]] if f2 else [])
            eligible = parity["status"] == "PASS" and all(c == "PASS" for c in checks) and m2 is not None
            spread = abs(m1 - m2) / min(m1, m2) if m2 is not None else None
            load = {k: [solos[k][b].get("load_s") for b in form] if k in solos else None for k in ("run1", "run2")}
            foot = {k: [solos[k][b]["memory"]["end"].get("phys_footprint_bytes") if "memory" in solos[k][b] else None
                        for b in form] if k in solos else None for k in ("run1", "run2")}
            rows.append({"form": fid, "bundles": list(form), "compute": f1["compute"], "run1": f1["stats_ms"],
                         "run2": f2["stats_ms"] if f2 else None, "score_ms": score,
                         "window_spread": spread, "spread_flag_30pct": spread is not None and spread >= 0.30,
                         "parity": parity, "output_check": checks,
                         "output_check_max_abs_dp": [f1["output_check"]["max_abs_dp"]]
                         + ([f2["output_check"]["max_abs_dp"]] if f2 else []),
                         "parts_median_ms": {"run1": f1["parts_median_ms"], "run2": f2["parts_median_ms"] if f2 else None},
                         "eligible": eligible, "solo_load_s": load, "solo_phys_footprint_end_bytes": foot,
                         "bytes": [json.loads((reg[b]["folder"] / "provenance" / "aot-manifest.json").read_text())["bytes"]
                                   for b in form]})
        cands = [r for r in rows if r["eligible"]]
        best = min((r["score_ms"] for r in cands), default=None)
        for r in rows:
            r["rank_note"] = None
            if best is not None and r["eligible"] and r["score_ms"] <= best * 1.03:
                r["ship_candidate"] = True
            else:
                r["ship_candidate"] = False
            if best is not None and not r["eligible"] and r["score_ms"] is not None and r["score_ms"] < best * 0.97:
                r["rank_note"] = "faster than the fastest candidate by > 3 % but not eligible (parity or output check)"
        order = sorted(rows, key=lambda r: (r["score_ms"] is None, r["score_ms"] or 0))
        for i, r in enumerate(order, 1):
            r["order_by_score"] = i
        out["workloads"][wid] = {"what": w1["what"], "case": w1["case"], "best_candidate_score_ms": best,
                                 "preprocess_ms": {"run1": w1["preprocess_ms"], "run2": w2["preprocess_ms"] if w2 else None},
                                 "rows": order}
    path = write_new(out_dir / "ranking.json", out)
    lines = ["# round 7 timing — per workload (score = mean of the two windows' medians, ms per decision)", "",
             f"rule: {out['rule']['path']} (sha256 {out['rule']['sha256'][:12]}…, mtime {out['rule']['mtime']})", ""]
    for wid, w in out["workloads"].items():
        lines += [f"## {wid} — {w['what']}", "",
                  "| # | form | score ms | run1 median (p10–p90) | run2 median (p10–p90) | spread | parity | timed check | ship candidate | solo load s (run1/run2) |",
                  "|---:|---|---:|---|---|---:|---|---|---|---|"]
        for r in w["rows"]:
            fmt = lambda s: f"{s['median']:.2f} ({s['p10']:.2f}–{s['p90']:.2f})" if s else "—"  # noqa: E731
            num = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
            ld = "/".join(",".join(num(x) for x in v) if v else "—" for v in r["solo_load_s"].values())
            lines.append(f"| {r['order_by_score']} | {r['form']} | {num(r['score_ms'])} | {fmt(r['run1'])} | {fmt(r['run2'])} | "
                         f"{(r['window_spread'] or 0) * 100:.1f} %{' ⚑' if r['spread_flag_30pct'] else ''} | "
                         f"{r['parity']['text']} | {'/'.join(r['output_check'])} (max\\|Δp\\| "
                         f"{max(r['output_check_max_abs_dp']):.3g}) | {'yes' if r['ship_candidate'] else ''} | {ld} |")
        pre = w["preprocess_ms"]
        lines += ["", "preprocess (ms, median run1 / run2): " + "; ".join(
            f"{k} {pre['run1'][k]['median']:.2f} / {pre['run2'][k]['median']:.2f}" if pre["run2"] else k
            for k in pre["run1"]), ""]
    table = free_name(out_dir / "r7_table.md")
    table.write_text("\n".join(lines) + "\n")
    print("->", path, table)
    return 0


# =========================================================================== the ranking of round 12 (ranking_rule_r12.md)
def parity_r12(form: tuple, reg: dict) -> dict:
    """The parity verdict of a round-12 form (ranking_rule_r12.md §5): the gates of the stripped bundles. A decision
    form = its bucket's runtime gate; an audio form = the end-to-end gate whose routing put rows on the form's decision
    bucket (round 12's small_runtime_audio_e2e_fp16.json for L64 / L128, round 10's ship_runtime_audio_e2e_fp16.json
    for L256), with the rows it ran at that bucket."""
    res = WORK / "results"
    dec = reg[form[-1]]
    if len(form) == 1:
        path = res / f"ship_runtime_decide_fp16_L{dec['L']}_gpu.json"
        if not path.exists():
            return {"status": "MISSING", "json": None, "text": f"no runtime gate {path.name}"}
        d = json.loads(path.read_text())
        s = d["summary"]
        return {"status": d["status"], "json": str(path.relative_to(WORK)),
                "text": f"{d['status']} {s['argmax_equal']}/{s['rows']}, max|Δp| {s['max_abs_dp']:.3g}, "
                        f"mean {s['mean_row_max_abs_dp']:.3g}"}
    media = reg[form[0]]
    small = dec["L"] in host.SMALL_BUCKETS
    path = res / ("small_runtime_audio_e2e_fp16.json" if small else "ship_runtime_audio_e2e_fp16.json")
    if not path.exists():
        return {"status": "MISSING", "json": None, "text": f"no end-to-end gate {path.name}"}
    d = json.loads(path.read_text())
    s = d["summaries"]["audio_bundle_prefix_torch_mel"]
    at = s.get("by_bucket", {}).get(str(dec["L"]))
    covered = str(media["sec"]) in d["audio"] and at is not None
    status = d["status"] if covered else "MISSING"
    text = (f"{d['status']} end to end {s['argmax_equal']}/{s['rows']}, max|Δp| {s['max_abs_dp']:.3g}, "
            f"mean {s['mean_row_max_abs_dp']:.3g}")
    if at is not None:
        text += f"; rows at L{dec['L']}: {at['argmax_equal']}/{at['rows']}, max|Δp| {at['max_abs_dp']:.3g}"
    return {"status": status, "json": str(path.relative_to(WORK)), "text": text,
            "rows_at_bucket": at, "audio_buckets": sorted(int(k) for k in d["audio"])}


def cmd_summarize_r12(args) -> int:
    """One window (ranking_rule_r12.md §5): score = the window's median; candidates = the forms whose gate passed and
    whose every timed decision matched the oracle row; every candidate within 3 % of the fastest is a co-candidate."""
    reg = bundle_registry()
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    main_path = Path(args.main) if args.main else OUT / "run5_r12_main.json"
    run = json.loads(main_path.read_text())
    if run["rule"]["sha256"] != sha256_file(RULE):
        raise SystemExit(f"the rule file changed after the window: {run['rule']['sha256']} != {sha256_file(RULE)}")
    solo_path = main_path.with_name(main_path.name.replace("_main", "_solo"))
    solo = json.loads(solo_path.read_text())["bundles"] if solo_path.exists() else {}
    window_path = main_path.with_name(main_path.name.replace("_main", "_window"))
    window = json.loads(window_path.read_text()) if window_path.exists() else {}
    out = {"schema": "d1-omni-ranking-r12/1", "rule": rule_info(), "run": str(main_path), "solo": str(solo_path),
           "window": {k: window.get(k) for k in ("run", "set", "started", "finished", "expected_lock_label",
                                                  "lock_is_ours", "contended", "gpu_quiet_wait", "lock_at_start",
                                                  "lock_at_end")},
           "main_started": run["started"], "main_finished": run.get("finished"), "workloads": {}}
    for wid, w in run["workloads"].items():
        rows = []
        for fid, f in w["forms"].items():
            form = tuple(f["bundles"])
            parity = parity_r12(form, reg)
            check = f["output_check"]["status"]
            st = f["stats_ms"]
            rows.append({"form": fid, "bundles": list(form), "compute": f["compute"],
                         "L": reg[form[-1]]["L"], "score_ms": st["median"], "stats_ms": st,
                         "parts_median_ms": f["parts_median_ms"], "parity": parity, "output_check": check,
                         "output_check_max_abs_dp": f["output_check"]["max_abs_dp"],
                         "output_check_drift": f["output_check"]["drift_marker_logits"],
                         "eligible": parity["status"] == "PASS" and check == "PASS",
                         "solo_load_s": [solo.get(b, {}).get("load_s") for b in form],
                         "solo_phys_footprint_end_bytes": [solo.get(b, {}).get("memory", {}).get("end", {})
                                                           .get("phys_footprint_bytes") for b in form],
                         "bytes": [json.loads((reg[b]["folder"] / "provenance" / "aot-manifest.json").read_text())["bytes"]
                                   for b in form]})
        best = min((r["score_ms"] for r in rows if r["eligible"]), default=None)
        for r in rows:
            r["ship_candidate"] = best is not None and r["eligible"] and r["score_ms"] <= best * 1.03
            r["rank_note"] = ("faster than the fastest candidate by > 3 % but not eligible (parity or output check)"
                              if best is not None and not r["eligible"] and r["score_ms"] < best * 0.97 else None)
            r["vs_best_pct"] = (r["score_ms"] / best - 1) * 100 if best else None
        order = sorted(rows, key=lambda r: r["score_ms"])
        for i, r in enumerate(order, 1):
            r["order_by_score"] = i
        out["workloads"][wid] = {"what": w["what"], "case": w["case"], "best_candidate_score_ms": best,
                                 "preprocess_ms": w["preprocess_ms"], "loop_seconds": w["loop_seconds"], "rows": order}
    path = write_new(out_dir / "ranking_r12.json", out)
    lines = ["# round 12 timing — per workload (one window: score = the window's median, ms per decision)", "",
             f"rule: {out['rule']['path']} (sha256 {out['rule']['sha256'][:12]}…, mtime {out['rule']['mtime']}); window "
             f"{out['window'].get('expected_lock_label')} {out['window'].get('started')} – {out['window'].get('finished')}", ""]
    for wid, w in out["workloads"].items():
        lines += [f"## {wid} — {w['what']}", "",
                  "| # | form | median ms (p10–p90) | vs best | decision graph ms | media graph ms | parity | timed check | ship candidate | solo load s |",
                  "|---:|---|---|---:|---:|---:|---|---|---|---|"]
        for r in w["rows"]:
            st, parts = r["stats_ms"], r["parts_median_ms"] or {}
            vs = "—" if r["vs_best_pct"] is None else f"{r['vs_best_pct']:+.1f} %"
            load = ", ".join("—" if x is None else f"{x:.2f}" for x in r["solo_load_s"])
            lines.append(f"| {r['order_by_score']} | {r['form']} | {st['median']:.2f} ({st['p10']:.2f}–{st['p90']:.2f}) | "
                         f"{vs} | {parts.get('graph_ms', 0):.2f} | {parts.get('media_ms', 0):.2f} | {r['parity']['text']} | "
                         f"{r['output_check']} (max\\|Δp\\| {r['output_check_max_abs_dp']:.3g}) | "
                         f"{'yes' if r['ship_candidate'] else ''} | {load} |")
        lines += ["", "preprocess (ms, median): " + "; ".join(f"{k} {v['median']:.2f}" for k, v in w["preprocess_ms"].items()), ""]
    table = free_name(out_dir / "r12_table.md")
    table.write_text("\n".join(lines) + "\n")
    print("->", path, table)
    return 0


# =========================================================================== CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("main", help="one process: every selected form, alternated")
    m.add_argument("--label", required=True)
    m.add_argument("--workloads", default=None, help="comma list (default: all)")
    m.add_argument("--forms", default=None, help="comma list of form ids or decision bundle ids (default: all)")
    m.add_argument("--rounds", type=int, default=ROUNDS)
    m.add_argument("--warmup", type=int, default=WARMUP)
    m.add_argument("--out", default=None)
    m.add_argument("--trace", default=None)
    s = sub.add_parser("solo", help="one bundle in a fresh process: load seconds and footprint")
    s.add_argument("--bundle", required=True)
    s.add_argument("--out", required=True)
    w = sub.add_parser("window", help="the measurement window's protocol (run under quiet_hold.py)")
    w.add_argument("--run", required=True, choices=["run1", "run2", "dry", "run5_r12", "dry_r12"])
    w.add_argument("--lock-label", default=None, help="the window's label in the lock (default d1d-r7-<run>; round 12: "
                   "d1d-r12)")
    w.add_argument("--workloads", default=None)
    w.add_argument("--rounds", type=int, default=ROUNDS)
    w.add_argument("--warmup", type=int, default=WARMUP)
    w.add_argument("--no-lock", action="store_true", help="dry run only: do not require our label in the lock")
    w.add_argument("--no-wait", action="store_true", help="dry run only: do not wait for other GPU jobs")
    w.add_argument("--no-solo", action="store_true")
    w.add_argument("--no-smoke", action="store_true", help="skip the in-window pass of every form once")
    w.add_argument("--summarize", action="store_true", help="after the window's work: summarize run1 + run2")
    w.add_argument("--out-dir", default=None, help="default: <work>/results/timing (a dry run writes elsewhere)")
    z = sub.add_parser("summarize", help="run1 + run2 -> ranking.json and r7_table.md (--set r12: one window -> "
                       "ranking_r12.json and r12_table.md)")
    z.add_argument("--run1", default=None)
    z.add_argument("--run2", default=None)
    z.add_argument("--main", default=None, help="--set r12: the window's main JSON (default results/timing/run5_r12_main.json)")
    z.add_argument("--out-dir", default=None)
    for p in (m, s, w, z):
        p.add_argument("--set", choices=sorted(SETS), default="r7", help="the form set and its rule file (r7: "
                       "ranking_rule.md; r12: ranking_rule_r12.md, the stripped fp16 L64 / L128 / L256)")
    args = ap.parse_args()
    use_set(args.set)
    if args.cmd == "main":
        return cmd_main(args)
    if args.cmd == "solo":
        return cmd_solo(args)
    if args.cmd == "window":
        return cmd_window(args)
    return cmd_summarize_r12(args) if SET == "r12" else cmd_summarize(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("main", "solo"):
        import coreai.runtime as rt  # noqa: E402  (the runtime only where a model is loaded)
    raise SystemExit(main())
