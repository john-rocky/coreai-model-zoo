"""Shared pieces of the round-3 engine gates: the Core AI python runtime on AOT h16c `.aimodelc` assets.

Every gate here loads an AOT asset with `SpecializationOptions.default()` (never a JIT `.aimodel` on the
GPU, never `cpu_only()`), feeds the S=1 decoder one token per call from fresh zero states with the
full-length position ramp (as `decider_vision/readout_gate_vision.py`), and judges a free-run greedy
against the oracle by `conversion/coreai_gate.py`'s rule: the first divergence passes only where the
fp32 oracle's softmax top-2 gap there is < 0.1 (`oracle/margins.npz`).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _paths import exports_dir, work_path  # noqa: E402

V = 248094
STOP_IDS = (248044, 248046)
KNIFE_EDGE = 0.1
N_FREE = 32
LANE = work_path("_minicpmv47")
ORACLE = LANE / "oracle"
R3 = LANE / "r3"
AOT_DIR = exports_dir() / "minicpmv47_aotc"
PROMPT_IMAGES = {"P1": "earth", "P2": "saturn_v", "P3": "synthetic", "P4": "earth"}
DECODER_CONTRACT = {"input_ids": ([1, 1], "int32"), "position_ids": ([1, -1], "int32"),
                    "image_embeds": ([64, 1024], "float16"), "rope_shift_start": ([1], "int32"),
                    "rope_shift_amount": ([1], "int32")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> dict:
    files = sorted(p for p in Path(path).rglob("*") if p.is_file())
    per = {str(p.relative_to(path)): sha256_file(p) for p in files}
    tree = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"path": str(path), "bytes": sum(p.stat().st_size for p in files), "tree_sha256": tree}


def aimodelc_for(bundle: Path) -> Path:
    p = AOT_DIR / f"{Path(bundle).name}.h16c.aimodelc"
    if not p.is_dir():
        raise SystemExit(f"no AOT asset {p} (compile it with aot_compile.py first)")
    return p


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def dsc(d) -> tuple[list[int], str]:
    return [int(x) for x in d.shape], str(d.dtype).split(".")[-1]


def fn_desc(fn) -> dict:
    d = fn.desc
    out = {"inputs": {n: dsc(d.input_descriptor(n)) for n in d.input_names},
           "outputs": {n: dsc(d.output_descriptor(n)) for n in d.output_names}}
    try:
        out["states"] = {n: dsc(d.state_descriptor(n)) for n in d.state_names}
    except Exception:  # noqa: BLE001 - a stateless function (the tower)
        out["states"] = {}
    return out


def nd(rt, a):
    return rt.NDArray(np.ascontiguousarray(a))


def softmax_gap(logits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[n, V] -> (top-1 ids, top-2 ids, p1 - p2), float64 softmax over the full vocabulary."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    order = np.argsort(-p, axis=-1, kind="stable")[:, :2]
    rows = np.arange(p.shape[0])
    return order[:, 0], order[:, 1], p[rows, order[:, 0]] - p[rows, order[:, 1]]


def topk(x: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
    x = np.asarray(x, dtype=np.float32)
    i = np.argsort(-x, kind="stable")[:k]
    return [int(t) for t in i], [float(x[t]) for t in i]


def first_divergence(engine: list[int], oracle: list[int], margins: np.ndarray) -> dict:
    """coreai_gate.py's rule on two greedy id lists (the engine's ends at its stop token or at N_FREE)."""
    n = min(len(engine), len(oracle))
    d = next((i for i in range(n) if engine[i] != oracle[i]), None)
    if d is None:
        return {"first_divergence": None, "exact_prefix": n, "compared": n, "verdict": "PASS",
                "rule": "no divergence over the compared steps"}
    m = float(margins[d]) if d < len(margins) else None
    tie = m is not None and m < KNIFE_EDGE
    return {"first_divergence": d, "exact_prefix": d, "compared": n, "engine_id": int(engine[d]),
            "oracle_id": int(oracle[d]), "oracle_margin_at_divergence": m, "knife_edge": tie,
            "verdict": "PASS_TIE" if tie else "FAIL",
            "rule": f"first divergence passes only where the oracle's softmax top-2 gap < {KNIFE_EDGE}"}


class Decoder:
    """One loaded S=1 decoder function; fresh zero states per sequence."""

    def __init__(self, rt, model, fn, max_ctx: int):
        self.rt, self.model, self.fn, self.max_ctx = rt, model, fn, max_ctx
        self.desc = fn_desc(fn)

    @classmethod
    async def load(cls, rt, aimodelc: Path, max_ctx: int):
        t0 = time.perf_counter()
        model = await maybe(rt.AIModel.load(str(aimodelc), rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        fn = await maybe(model.load_function("main"))
        self = cls(rt, model, fn, max_ctx)
        self.load_seconds = {"model": t1 - t0, "main": time.perf_counter() - t1}
        return self

    def check_contract(self) -> None:
        d = self.desc
        bad = {n: (d["inputs"].get(n), w) for n, w in DECODER_CONTRACT.items()
               if d["inputs"].get(n) is None or tuple(d["inputs"][n][0]) != tuple(w[0]) or d["inputs"][n][1] != w[1]}
        if bad or set(d["inputs"]) != set(DECODER_CONTRACT):
            raise SystemExit(f"descriptor differs from the contract: {bad} {sorted(d['inputs'])}")
        if [tuple(v[0]) for v in d["outputs"].values()] != [(1, 1, V)]:
            raise SystemExit(f"outputs {d['outputs']} != logits [1, 1, {V}]")
        if len(d["states"]) != 4:
            raise SystemExit(f"states {d['states']}")

    def fresh_state(self) -> dict:
        return {n: nd(self.rt, np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                for n, (shape, dt) in self.desc["states"].items()}

    def static(self, image_embeds, start: int, amount: int) -> dict:
        rt = self.rt
        return {"image_embeds": nd(rt, np.asarray(image_embeds, np.float16)),
                "rope_shift_start": nd(rt, np.array([start], np.int32)),
                "rope_shift_amount": nd(rt, np.array([amount], np.int32))}

    async def step(self, token: int, t: int, static: dict, state: dict) -> np.ndarray:
        rt = self.rt
        res = await maybe(self.fn(inputs={"input_ids": nd(rt, np.array([[token]], np.int32)),
                                          "position_ids": nd(rt, np.arange(t + 1, dtype=np.int32)[None]),
                                          **static}, state=state))
        lg = np.asarray(res["logits"].numpy())
        assert lg.shape == (1, 1, V), lg.shape
        return lg[0, 0]

    async def run(self, ids, static: dict, keep: set | None = None, free: int = 0) -> dict:
        """Feed `ids` one token per call from fresh states (logits kept at positions in `keep`, argmax at
        every position), then free-run greedy for up to `free` tokens (stop ids end it, included)."""
        state = self.fresh_state()
        keep = keep or set()
        argmax, kept, ms = [], {}, []
        lg = None
        for t, tk in enumerate(ids):
            t0 = time.perf_counter()
            lg = await self.step(int(tk), t, static, state)
            ms.append((time.perf_counter() - t0) * 1e3)
            argmax.append(int(np.argmax(lg.astype(np.float32))))
            if t in keep:
                kept[t] = lg.copy()
        gen, gen_ms, t = [], [], len(ids)
        if free:
            nxt = argmax[-1]
            while True:
                gen.append(nxt)
                if nxt in STOP_IDS or len(gen) >= free:
                    break
                t0 = time.perf_counter()
                lg = await self.step(nxt, t, static, state)
                gen_ms.append((time.perf_counter() - t0) * 1e3)
                t += 1
                nxt = int(np.argmax(lg.astype(np.float32)))
        return {"argmax": argmax, "kept": kept, "last": lg, "step_ms": ms, "gen": gen, "gen_ms": gen_ms}


def env_record() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "numpy": np.__version__}
    for p in ("coreai-core", "coreai-torch", "coreai-opt", "coreai-models", "torch", "tokenizers", "pillow"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    sh = lambda *c: subprocess.run(list(c), capture_output=True, text=True).stdout.strip()  # noqa: E731
    lock = Path(os.path.expanduser("~/code/coreai/_GPU_LOCK"))
    return {"versions": v, "platform": platform.platform(), "macos_build": sh("sw_vers", "-buildVersion"),
            "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "pid": os.getpid(),
            "runtime": "coreai python runtime, AOT h16c GPU .aimodelc, SpecializationOptions.default(), no JIT",
            "gpu_lock_at_start": lock.read_text(errors="replace").strip() if lock.exists() else None,
            "quiet_ok_env": os.environ.get("QUIET_OK")}


def gpu_util() -> int | None:
    """IOAccelerator 'Device Utilization %' (decider_vision/gate_tower.py's probe)."""
    txt = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                         text=True).stdout
    key = '"Device Utilization %"='
    try:
        return int(txt.split(key)[1].split(",")[0].split("}")[0]) if key in txt else None
    except ValueError:
        return None


class GpuSampler:
    """Samples the GPU utilization every `period` s on a thread while a gate runs: evidence that the
    work ran on the GPU delegate and not on a CPU fallback (machine-wide, so read it next to the window)."""

    def __init__(self, period: float = 0.25):
        import threading

        self.period, self.samples, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.period):
            u = gpu_util()
            if u is not None:
                self.samples.append(u)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)

    def summary(self) -> dict:
        s = np.asarray(self.samples, dtype=np.float64)
        return {"n": int(s.size), "period_s": self.period, "max": float(s.max()) if s.size else None,
                "median": float(np.median(s)) if s.size else None,
                "share_ge_50": float((s >= 50).mean()) if s.size else None}


def _sh(*cmd) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True).stdout.strip()


GPU_SUSPECTS = ("python", "llm-runner", "coreai", "xctest", "PipelinedBench", "mlx", "ollama", "litert",
                "Simulator", "ffmpeg", "screencapture", "QuickTime", "Xcode", "metal")


def window_record(label: str) -> dict:
    lock = Path(os.path.expanduser("~/code/coreai/_GPU_LOCK"))
    ps = _sh("ps", "-axo", "pid,ppid,pcpu,rss,etime,comm")
    me = {os.getpid(), os.getppid()}
    rows = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, pcpu, rss, etime, comm = parts
        if int(pid) in me:
            continue
        if any(s.lower() in comm.lower() for s in GPU_SUSPECTS) or float(pcpu) >= 20.0:
            rows.append({"pid": int(pid), "ppid": int(ppid), "pcpu": float(pcpu), "rss_kb": int(rss), "etime": etime,
                         "comm": comm})
    return {"label": label, "taken_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gpu_lock": lock.read_text(errors="replace").strip() if lock.exists() else None,
            "lock_lsof": _sh("lsof", str(lock)).splitlines(),
            "loadavg": _sh("sysctl", "-n", "vm.loadavg"), "swapusage": _sh("sysctl", "vm.swapusage"),
            "other_processes_cpu20_or_gpu_suspects": rows, "driver_pid": os.getpid()}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
