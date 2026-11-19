# Community port — NOT an Apple model.
"""Engine gate: run the exported base_lm prefill+decode .aimodel bundles on the Core AI engine and
compare hidden states to the oracle. Shared keyCache/valueCache NDArrays threaded prefill->decode."""
from __future__ import annotations

import asyncio
import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parent / "artifacts"
SCRATCH = os.environ.get("SCRATCH") or os.path.join(tempfile.gettempdir(), "voxcpm_scratch")
REF = np.load(os.path.join(SCRATCH, "oracle_ref.npz"))
NL, NKV, HD, CL = 24, 2, 64, 512


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


async def main():
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    pdir = ART / "voxcpm_base_fp16_prefill_t9"
    ddir = ART / "voxcpm_base_fp16_decode_cl512"
    pm = await rt.AIModel.load(str(pdir / f"{pdir.name}.aimodel"), gpu)
    dm = await rt.AIModel.load(str(ddir / f"{ddir.name}.aimodel"), gpu)
    pfn, dfn = pm.load_function("main"), dm.load_function("main")

    state = {
        "keyCache": rt.NDArray(np.zeros((NL, 1, NKV, CL, HD), dtype=np.float16)),
        "valueCache": rt.NDArray(np.zeros((NL, 1, NKV, CL, HD), dtype=np.float16)),
    }

    # prefill
    emb = REF["prefill_base.in_inputs_embeds__0"].astype(np.float16)
    pres = await pfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb))}, state=state)
    hid = pres["hidden"].numpy()
    cs = [cos(hid, REF["prefill_base.out__0"])]
    print(f"  prefill q={emb.shape[1]} cos={cs[0]:.6f} {'OK' if cs[0]>=0.99 else 'FAIL'}")

    n = sum(1 for k in REF.files if k.startswith("base_step.out__"))
    for i in range(n):
        e = REF[f"base_step.in_0__{i}"].astype(np.float16).reshape(1, 1, 1024)
        p = REF[f"base_step.in_1__{i}"].astype(np.int32).reshape(1)
        res = await dfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(e)),
                                "pos": rt.NDArray(np.ascontiguousarray(p))}, state=state)
        c = cos(res["hidden"].numpy(), REF[f"base_step.out__{i}"])
        cs.append(c)
        print(f"  decode[{i}] pos={int(p[0])} cos={c:.6f} {'OK' if c>=0.99 else 'FAIL'}")

    lo = min(cs)
    print(f"\n>>> engine vs oracle: min cos = {lo:.6f} (fp16 engine vs fp32 oracle) -> "
          f"{'GATE PASS' if lo>=0.99 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
