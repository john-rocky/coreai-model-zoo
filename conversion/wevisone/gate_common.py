"""Shared pieces of the WeVisOne-VL-2B engine gates: the Core AI python runtime on AOT h16c `.aimodelc` assets.

Every gate loads an AOT asset with `SpecializationOptions.default()` (never a JIT `.aimodel` on the GPU, never
`cpu_only()`), feeds the S=1 "main" function one token per call from fresh zero states with the full-length position
ramp, and judges a free-run greedy against the oracle by `conversion/coreai_gate.py`'s rule: the first divergence
passes only where the fp32 oracle's softmax top-2 gap there is < 0.1. (From `conversion/minicpmv47/gate_common.py`;
the decoder here also takes the three deepstack rows, and can rebuild its AIModel every `reload_every` calls.)
"""
from __future__ import annotations

import glob
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

V = 152680
IMAGE_PAD = 151655
STOP_IDS = (151645, 151643)
KNIFE_EDGE = 0.1
TEXT_ONLY_START = 1 << 30
HIDDEN = 2048


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


def host_inputs(ids, g: int) -> tuple[np.ndarray, int, int]:
    """The exporter's host contract: `<|image_pad|>` -> V + slot; rope shift starts after the image, by N - g."""
    a = np.asarray(ids, np.int64)
    pads = np.flatnonzero(a == IMAGE_PAD)
    n = g * g
    if len(pads) != n or not np.array_equal(pads, np.arange(pads[0], pads[0] + n)):
        raise SystemExit(f"image pads {len(pads)} != {n} or not contiguous")
    mapped = a.copy()
    mapped[pads] = V + np.arange(n)
    return mapped, int(pads[0]) + n, n - g


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


def topk(x: np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
    x = np.asarray(x, dtype=np.float32)
    i = np.argsort(-x, kind="stable")[:k]
    return [int(t) for t in i], [float(x[t]) for t in i]


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def first_divergence(engine: list[int], oracle: list[int], gaps) -> dict:
    """coreai_gate.py's rule on two greedy id lists (the engine's ends at its stop token or at its cap)."""
    n = min(len(engine), len(oracle))
    d = next((i for i in range(n) if engine[i] != oracle[i]), None)
    if d is None:
        return {"first_divergence": None, "exact_prefix": n, "compared": n, "verdict": "PASS",
                "rule": "no divergence over the compared steps"}
    m = float(gaps[d]) if d < len(gaps) else None
    tie = m is not None and m < KNIFE_EDGE
    return {"first_divergence": d, "exact_prefix": d, "compared": n, "engine_id": int(engine[d]),
            "oracle_id": int(oracle[d]), "oracle_gap_at_divergence": m, "knife_edge": tie,
            "verdict": "PASS_TIE" if tie else "FAIL",
            "rule": f"first divergence passes only where the oracle's softmax top-2 gap < {KNIFE_EDGE}"}


class Decoder:
    """One loaded S=1 decoder function on an AOT asset; fresh zero states per sequence; optional periodic reload."""

    def __init__(self, rt, aimodelc: Path, max_ctx: int, n_img: int, reload_every: int, function: str = "main"):
        self.rt, self.aimodelc, self.max_ctx, self.n_img = rt, aimodelc, max_ctx, n_img
        self.reload_every, self.function = reload_every, function
        self.calls_since_load = 0
        self.reloads = 0
        self.reload_seconds = 0.0

    async def _load(self):
        t0 = time.perf_counter()
        self.model = await maybe(self.rt.AIModel.load(str(self.aimodelc), self.rt.SpecializationOptions.default()))
        t1 = time.perf_counter()
        self.fn = await maybe(self.model.load_function(self.function))
        return t1 - t0, time.perf_counter() - t1

    @classmethod
    async def load(cls, rt, aimodelc: Path, max_ctx: int, n_img: int, reload_every: int = 0, function: str = "main"):
        self = cls(rt, aimodelc, max_ctx, n_img, reload_every, function)
        m, f = await self._load()
        self.load_seconds = {"model": m, "function": f}
        self.desc = fn_desc(self.fn)
        return self

    def check_contract(self) -> None:
        # the image inputs are fp16 for fp16 / int8 bundles and fp32 for an fp32-compute bundle (fp16w32)
        self.io_dtype = self.desc["inputs"].get("image_embeds", [None, "float16"])[1]
        if self.io_dtype not in ("float16", "float32"):
            raise SystemExit(f"image_embeds dtype {self.io_dtype}")
        want = {"input_ids": ([1, 1], "int32"), "position_ids": ([1, -1], "int32"),
                "image_embeds": ([self.n_img, HIDDEN], self.io_dtype),
                "deepstack_embeds": ([3 * self.n_img, HIDDEN], self.io_dtype),
                "rope_shift_start": ([1], "int32"), "rope_shift_amount": ([1], "int32")}
        d = self.desc
        bad = {n: (d["inputs"].get(n), w) for n, w in want.items()
               if d["inputs"].get(n) is None or tuple(d["inputs"][n][0]) != tuple(w[0]) or d["inputs"][n][1] != w[1]}
        if bad or set(d["inputs"]) != set(want):
            raise SystemExit(f"descriptor differs from the contract: {bad} {sorted(d['inputs'])}")
        if [tuple(v[0]) for v in d["outputs"].values()] != [(1, 1, V)]:
            raise SystemExit(f"outputs {d['outputs']} != logits [1, 1, {V}]")

    def fresh_state(self) -> dict:
        return {n: nd(self.rt, np.zeros([self.max_ctx if s < 0 else s for s in shape], np.dtype(dt)))
                for n, (shape, dt) in self.desc["states"].items()}

    def static(self, image_embeds, deepstack, start: int, amount: int) -> dict:
        rt = self.rt
        dt = np.dtype(getattr(self, "io_dtype", "float16"))
        return {"image_embeds": nd(rt, np.asarray(image_embeds, dt)),
                "deepstack_embeds": nd(rt, np.asarray(deepstack, dt).reshape(3 * self.n_img, HIDDEN)),
                "rope_shift_start": nd(rt, np.array([start], np.int32)),
                "rope_shift_amount": nd(rt, np.array([amount], np.int32))}

    async def step(self, token: int, t: int, static: dict, state: dict) -> np.ndarray:
        if self.reload_every and self.calls_since_load >= self.reload_every:
            t0 = time.perf_counter()
            await self._load()
            self.reload_seconds += time.perf_counter() - t0
            self.reloads += 1
            self.calls_since_load = 0
        rt = self.rt
        res = await maybe(self.fn(inputs={"input_ids": nd(rt, np.array([[token]], np.int32)),
                                          "position_ids": nd(rt, np.arange(t + 1, dtype=np.int32)[None]),
                                          **static}, state=state))
        self.calls_since_load += 1
        lg = np.asarray(res["logits"].numpy())
        assert lg.shape == (1, 1, V), lg.shape
        return lg[0, 0]

    async def run(self, ids, static: dict, keep: set | None = None, free: int = 0) -> dict:
        """Feed `ids` one token per call from fresh states (logits kept at positions in `keep`, argmax at every
        position), then free-run greedy for up to `free` tokens (a stop id ends it, included)."""
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
    for p in ("coreai-core", "coreai-torch", "coreai-opt", "coreai-models", "torch", "tokenizers"):
        try:
            v[p] = md.version(p)
        except Exception as e:  # noqa: BLE001
            v[p] = repr(e)
    sh = lambda *c: subprocess.run(list(c), capture_output=True, text=True).stdout.strip()  # noqa: E731
    lock = Path(os.path.expanduser("~/code/coreai/_GPU_LOCK"))
    return {"versions": v, "platform": platform.platform(), "macos_build": sh("sw_vers", "-buildVersion"),
            "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "pid": os.getpid(),
            "runtime": "coreai python runtime, AOT h16c .aimodelc, SpecializationOptions.default(), no JIT",
            "gpu_lock_at_start": lock.read_text(errors="replace").strip() if lock.exists() else None,
            "quiet_ok_env": os.environ.get("QUIET_OK")}


def gpu_util() -> int | None:
    txt = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True,
                         text=True).stdout
    key = '"Device Utilization %"='
    try:
        return int(txt.split(key)[1].split(",")[0].split("}")[0]) if key in txt else None
    except ValueError:
        return None


class GpuSampler:
    """Samples the machine-wide GPU utilization every `period` s on a thread while a gate runs."""

    def __init__(self, period: float = 0.5):
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


def window_record(label: str) -> dict:
    sh = lambda *c: subprocess.run(list(c), capture_output=True, text=True).stdout.strip()  # noqa: E731
    lock = Path(os.path.expanduser("~/code/coreai/_GPU_LOCK"))
    return {"label": label, "taken_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "gpu_lock": lock.read_text(errors="replace").strip() if lock.exists() else None,
            "loadavg": sh("sysctl", "-n", "vm.loadavg"), "swapusage": sh("sysctl", "vm.swapusage")}


def clean_own_mpsgraph_scratch() -> dict:
    """Remove this process's MPSGraph scratch dirs (named mpsgraph-<pid>-…); other sessions' entries stay."""
    pid = os.getpid()
    roots = set(glob.glob("/private/var/folders/*/*/T/com.apple.MetalPerformanceShadersGraph"))
    roots.add(os.path.join(tempfile.gettempdir(), "com.apple.MetalPerformanceShadersGraph"))
    removed, freed = [], 0
    for root in roots:
        for d in glob.glob(os.path.join(root, f"mpsgraph-{pid}-*")):
            try:
                size = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())
                shutil.rmtree(d)
                removed.append(d)
                freed += size
            except OSError:
                pass
    return {"pid": pid, "removed": removed, "bytes": freed}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
