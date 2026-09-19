"""Per-op profile of one decode step through coreai.runtime's Profiler hook.

    python _smoke/bonsai/profile_step.py <bundle>/aot_mac/<name>.<arch>.aimodelc [--steps 4]

Loads `main` with a Profiler, runs a few S=1 steps on the four zero states, and prints the
events of the last step grouped by event id with their wall time — the runtime's own view of
where a token goes, to set against the Metal System Trace command-buffer picture.
"""
import argparse, asyncio, collections, sys, time
from pathlib import Path
import numpy as np


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--kv", type=int, default=2048)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    import coreai.runtime as rt
    opts = rt.SpecializationOptions.default() if args.asset.endswith(".aimodelc") \
        else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = await rt.AIModel.load(str(args.asset), opts)
    events = []          # (step, event_id, phase, data, metadata, t_begin, t_end)
    step = [0]
    open_ = {}
    counter = [0]
    def begin(ev):
        counter[0] += 1
        open_[counter[0]] = (step[0], ev.event_id, ev.phase, ev.data, dict(ev.metadata), time.perf_counter_ns())
        return counter[0]
    def end(ev, iid):
        s, eid, ph, data, md, t0 = open_.pop(iid)
        events.append((s, eid, ph, data, md, t0, time.perf_counter_ns()))
    def single(ev):
        events.append((step[0], ev.event_id, ev.phase, ev.data, dict(ev.metadata), time.perf_counter_ns(), None))
    prof = rt.Profiler(on_log_event=single, on_log_event_begin=begin, on_log_event_end=end)
    fn = m.load_function("main", profiler=prof)
    d = fn.desc
    print(d, flush=True)
    st = {}
    for name in ("keyCache", "valueCache", "convState", "recState"):
        shape = None
        # shapes from the inspect output of this bundle
    st = {"keyCache": rt.NDArray(np.zeros((16, 1, 4, args.kv, 256), np.float16)),
          "valueCache": rt.NDArray(np.zeros((16, 1, 4, args.kv, 256), np.float16)),
          "convState": rt.NDArray(np.zeros((48, 1, 10240, 3), np.float16)),
          "recState": rt.NDArray(np.zeros((48, 1, 48, 128, 128), np.float16))}
    for s in range(args.steps):
        step[0] = s
        feed = {"input_ids": rt.NDArray(np.array([[1000 + s]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(s + 1, dtype=np.int32)[None])}
        t0 = time.perf_counter()
        await fn(inputs=feed, state=st)
        print(f"step {s}: {(time.perf_counter() - t0) * 1e3:.1f} ms, events so far {len(events)}", flush=True)
    last = [e for e in events if e[0] == args.steps - 1]
    print(f"\nlast step: {len(last)} events; phases {collections.Counter(e[2] for e in last)}")
    print("sample:", [(e[1], e[2], e[3][:60], e[4]) for e in last[:5]])
    tot = collections.defaultdict(float); cnt = collections.Counter()
    for s, eid, ph, data, md, t0, t1 in last:
        if t1 is None: continue
        key = eid
        tot[key] += (t1 - t0) / 1e6; cnt[key] += 1
    print(f"\n{'ms':>9} {'n':>5}  event id")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"{v:9.2f} {cnt[k]:5d}  {k}")
    # ids that carry metadata keys
    keys = collections.Counter(k for e in last for k in e[4])
    print("metadata keys:", keys.most_common(10))
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
