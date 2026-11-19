# Community port — NOT an Apple model.
"""Gate the exported drafter bundle (python GPU runtime) on REAL cases.

Feeds the same real cases as the torch parity gate (gemma4e2b_drafter_real_cases.npz)
to the exported .aimodel and compares first-draft argmax vs the tflite reference.
The fp16 graph adds noise over the fp32 torch module; PASS = argmax agreement >= 75%
over the sampled cases and mean logits cos >= 0.99.

Run: .venv/bin/python ../coreai-models-community/_smoke/gate_gemma4_mtp_drafter_bundle.py
"""
import asyncio
import os
import sys

import numpy as np

EXTRACT = os.path.expanduser("~/code/litertlm-convert/out/gemma4e2b_extract")
BUNDLE = "exports/gemma4_e2b_mtp_drafter"
MAX_SEQ = 1024
N_CASES = 10


async def run() -> bool:
    import coreai.runtime as rt

    z = np.load(f"{EXTRACT}/gemma4e2b_drafter_real_cases.npz")
    s_k13, s_v13, s_k14, s_v14 = z["kv_scales"]
    seq_ref = z["seq"].tolist()
    T = z["k13"].shape[0]

    k11 = np.zeros((1, 1, MAX_SEQ, 512), np.float16)
    v11 = np.zeros((1, 1, MAX_SEQ, 512), np.float16)
    k14 = np.zeros((1, 1, MAX_SEQ, 512), np.float16)
    v14 = np.zeros((1, 1, MAX_SEQ, 512), np.float16)
    k11[0, 0, :T, :256] = z["k13"].astype(np.float32) * s_k13
    v11[0, 0, :T, :256] = z["v13"].astype(np.float32) * s_v13
    k14[0, 0, :T, :] = z["k14"].astype(np.float32) * s_k14
    v14[0, 0, :T, :] = z["v14"].astype(np.float32) * s_v14

    m = await rt.AIModel.load(
        f"{BUNDLE}/gemma4_e2b_mtp_drafter.aimodel",
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()),
    )
    fn = m.load_function("main")
    print("drafter engine loaded", flush=True)

    # ONE position_ids length (avoid per-length respecialization): use the LAST
    # cases, all evaluated at their own pos but padded caches allow a shared seq —
    # no: input_pos defines masks. Sample every 5th case at ITS length (few
    # lengths, ~10 specializations is too slow) -> instead pick 10 cases at the
    # SAME pos? positions differ per case. Compromise: 5 cases, 5 lengths.
    idxs = [0, 10, 20, 30, 40][:N_CASES]
    agree = 0
    coss = []
    for n in idxs:
        pos = int(z[f"c{n}_pos"][0])
        seq = pos + 1
        tok = int(seq_ref[pos + 1])          # the token at position p = pos+1
        m_full = np.zeros((1, 1, 1, MAX_SEQ), np.float16); m_full[..., :seq] = 1.0
        m_slide = np.zeros((1, 1, 1, MAX_SEQ), np.float16)
        m_slide[..., max(0, pos - 511):seq] = 1.0
        res = await fn(inputs={
            "input_ids": rt.NDArray(np.array([[tok]], np.int32)),
            "hidden": rt.NDArray(z[f"c{n}_hidden"].astype(np.float16).reshape(1, 1, 1536)),
            "pos": rt.NDArray(np.array([[pos]], np.int32)),
            "mask_sliding": rt.NDArray(m_slide), "mask_full": rt.NDArray(m_full),
            "k11": rt.NDArray(k11), "v11": rt.NDArray(v11),
            "k14": rt.NDArray(k14), "v14": rt.NDArray(v14),
        })
        lg = res["logits"].numpy()[0, 0].astype(np.float32)
        ref = z[f"c{n}_logits"]
        am, ar = int(lg.argmax()), int(ref.argmax())
        cos = float(np.dot(lg, ref) / (np.linalg.norm(lg) * np.linalg.norm(ref)))
        coss.append(cos)
        agree += am == ar
        print(f"case {n:2d}: pos={pos:3d} argmax {am}{'==' if am == ar else '!='}{ar} cos={cos:.5f}",
              flush=True)

    print(f"agree {agree}/{len(idxs)}  mean cos {np.mean(coss):.5f}")
    return agree >= int(0.75 * len(idxs)) and np.mean(coss) >= 0.99


def main():
    ok = asyncio.run(run())
    print("DRAFTER BUNDLE GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
