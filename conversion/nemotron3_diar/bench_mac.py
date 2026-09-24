"""Mac timing of the Nemotron-3-Diarization streaming bundle (low_latency, T=541) on this Mac's GPU and
Neural Engine, through the same host loop the gates use. Export venv:

    ~/code/coreai/coreai-models/.venv/bin/python bench_mac.py

Per engine: load time; warmup 5 calls, then 100 timed graph calls on captured packed inputs (97.6 s
fixture steps, cycled; the graph is static, so every call costs the same) -> ms/chunk median and p90;
then the whole 97.6 s recording through host_loop.run (NumPy mel per chunk, embed, packing, graph,
speaker cache) -> wall, RTF = wall / 97.6 s, and the host share (wall minus the graph calls).

GPU lock: if ~/code/coreai/_GPU_LOCK is absent it is created here and removed at the end; if it is
present (another session holds it) it is left alone and every number is marked contended.
Numbers go to _work/bench_mac.json.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import host_loop as hl  # noqa: E402
from gate_reauthor import pack  # noqa: E402

WORK = HERE / "_work"
ART = WORK / "artifacts"
LOCK = Path.home() / "code" / "coreai" / "_GPU_LOCK"
AUDIO_S = 97.6
CHUNK_S = 0.72          # low_latency: 9 encoder frames x 80 ms emitted per step
ENGINES = [
    ("GPU (JIT, preferred gpu)", ART / "n3d_streaming_float16.aimodel", "gpu"),
    ("ANE (JIT, preferred neural_engine)", ART / "n3d_streaming_float16.aimodel", "ane"),
]


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def bench(label: str, bundle: Path, unit: str, io, assets, audio) -> dict:
    t0 = time.perf_counter()
    eng = hl.CoreAIEngine(bundle, unit)
    load_s = time.perf_counter() - t0
    n_steps = int(io["n_steps"])
    inputs = [pack(io[f"s{i:03d}_inputs_embeds"], 541, "noise", i) for i in range(n_steps)]
    inputs = [(p.numpy(), v.numpy()) for p, v in inputs]
    for k in range(5):
        eng(*inputs[k % n_steps])
    ts = []
    for k in range(100):
        p, v = inputs[(k * 7) % n_steps]
        t1 = time.perf_counter()
        eng(p, v)
        ts.append(time.perf_counter() - t1)
    ts_ms = np.array(ts) * 1e3
    graph_s = []

    def on_step(_i, info):
        graph_s.append(info["graph_s"])
    steps, n_frames = hl.build_steps("streaming", "low_latency", "host", assets, audio, None)
    t2 = time.perf_counter()
    logits, _ = hl.run(eng, steps, "streaming", assets["silence"], n_frames, on_step=on_step)
    wall = time.perf_counter() - t2
    rec = {"engine": label, "bundle": str(bundle.relative_to(HERE)), "unit": unit, "load_s": round(load_s, 2),
           "ms_per_chunk_median": round(float(np.median(ts_ms)), 2), "ms_per_chunk_p90": round(float(np.percentile(ts_ms, 90)), 2),
           "ms_per_chunk_min": round(float(ts_ms.min()), 2), "chunk_audio_ms": CHUNK_S * 1e3,
           "loop_steps": len(graph_s), "loop_wall_s": round(wall, 3), "loop_rtf": round(wall / AUDIO_S, 4),
           "loop_graph_s": round(float(sum(graph_s)), 3), "loop_host_s": round(wall - float(sum(graph_s)), 3),
           "loop_frames": int(logits.shape[0])}
    print(f"[bench] {label}: load {load_s:.2f} s | graph {rec['ms_per_chunk_median']} ms/chunk median, p90 "
          f"{rec['ms_per_chunk_p90']} (100 calls after 5 warmup; a chunk = {CHUNK_S * 1e3:.0f} ms of audio) | "
          f"97.6 s loop: {len(graph_s)} steps, wall {wall:.2f} s, RTF {rec['loop_rtf']}, graph {rec['loop_graph_s']} s, "
          f"host {rec['loop_host_s']} s", flush=True)
    return rec


def main():
    owned = False
    contended = LOCK.exists()
    if not contended:
        LOCK.touch()
        owned = True
    try:
        env = {"machine": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
               "macos": f"{platform.mac_ver()[0]} ({sh(['sw_vers', '-buildVersion'])})",
               "gpu_lock": "held by another session (contended)" if contended else "taken by this run",
               "load_avg_start": os.getloadavg()}
        print(f"[bench] {env}", flush=True)
        io = np.load(WORK / "chunk_io_diarization_example_ll.npz")
        assets = hl.load_assets()
        audio = hl.load_audio(hl.FIXTURE_DIR / hl.FIXTURES["diarization_example"])
        rows = [bench(label, bundle, unit, io, assets, audio) for label, bundle, unit in ENGINES]
        env["load_avg_end"] = os.getloadavg()
        for r in rows:
            r["contended"] = contended
        (WORK / "bench_mac.json").write_text(json.dumps({"env": env, "results": rows}, indent=1))
        print(f"[bench] load avg end {env['load_avg_end']}; -> _work/bench_mac.json", flush=True)
    finally:
        if owned:
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
