"""Split the Core AI VAE bundle's distance to a Qwen-Image-2.1-Turbo oracle into the bundle's own error and the
band the Turbo checkpoint's bf16 VAE file puts between the two weight sets.

The Turbo ``vae/`` is the bf16 round-to-nearest-even copy of the base fp32 weights; the oracle decoded with it
(upcast to fp32), the Core AI VAE bundles were exported from the base fp32 weights. Four decodes of the SAME
``final_latents`` (all ``[1,4,H,W]`` fp32 in [-1,1], clamped):

  O   the oracle's ``vae_out[:, :, 0]``                              (diffusers 85c9fa4, Turbo vae, fp32 CPU)
  Tt  ``parity_vae_torch.py --synthetic --snapshot <Turbo>``         (same weights as O, this machine)
  Tb  ``parity_vae_torch.py --synthetic --snapshot <base>``          (the weights the bundle holds)
  E   the bundle's output saved by ``engine_parity_vae.py``          (GPU, AOT efr, default())

  (i)   O  vs Tt   the reference reproduces                         bar max|d| <= 1e-5
  (ii)  Tb vs Tt   the bf16 rounding band of the Turbo VAE file     (measured, no bar)
  (iii) E  vs Tb   the bundle against its own weights               bar per channel (R, G, B, A): corr >= 0.9999,
                                                                    max|d| <= 1e-2 = the engine VAE gate's bar
                                                                    (measured: 09-25 1.35e-5..2.33e-5, 10-11
                                                                    3.0e-5..4.2e-5 max|d|; values, not the bar)
  (iv)  E  vs O    what ``engine_parity_vae.py`` scores             (context)
  (v)   E  vs Tt                                                     (context)

Elementwise |E - O| <= |E - Tb| + |Tb - Tt| + |Tt - O|, so with (i) and (iii) measured at the 1e-5 level, (iv) is
(ii) to within their sum: a (iv) that breaks the engine bar while (i) and (iii) hold is the weight file, not the
bundle.
Per pair: every channel (R, G, B, A) corr / max|d| / mean|d| / rel L2, and the uint8 RGBA images the pipeline
would write (PSNR, max byte difference).

``--fp64-base <dir>`` (``parity_vae_torch.py --synthetic --fp64 --snapshot <base>``): the fp32 floor of (iii) —
Tb and E each against the float64 decode of the base weights (F). When (iii) misses its bar by a small factor,
this says whether the bundle sits farther from the function than the CPU fp32 reference does, or the two fp32
decodes just round differently.

Run (any venv with numpy + pillow, from conversion/qwenimage21/):
  python vae_band.py --oracle oracle/turbo256 --torch-turbo _work/vae_ref_turbo256_wturbo \\
      --torch-base _work/vae_ref_turbo256_wbase --engine _work/engine_vae/qi21_vae_256_fp32_default/image_turbo256.f32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = ("R", "G", "B", "A")
BAR_I = 1e-5
BAR_III_CORR, BAR_III_MAXD = 0.9999, 1e-2            # per channel, as engine_parity_vae.py


def to_u8(img: np.ndarray) -> np.ndarray:
    """[1,4,H,W] in [-1,1] -> [H,W,4] uint8, as ``VaeImageProcessor.postprocess`` + ``numpy_to_pil``."""
    x = np.clip(img[0].astype(np.float32) * 0.5 + 0.5, 0.0, 1.0).transpose(1, 2, 0)
    return (x * 255).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.abs(x - y)
    return dict(nan=int(np.isnan(x).sum() + np.isnan(y).sum()), corr=float(np.corrcoef(x, y)[0, 1]),
                maxd=float(d.max()), mean_abs=float(d.mean()), rel_l2=float(np.linalg.norm(x - y) / np.linalg.norm(y)),
                bit_exact=bool(np.array_equal(a, b)))


def pair(a: np.ndarray, b: np.ndarray) -> dict:
    r = dict(all=stats(a, b), channels={n: stats(a[:, c], b[:, c]) for c, n in enumerate(CH)})
    ua, ub = to_u8(a), to_u8(b)
    du = np.abs(ua.astype(np.int16) - ub.astype(np.int16))
    r["u8"] = dict(psnr_rgba=psnr(ua, ub), psnr_rgb=psnr(ua[..., :3], ub[..., :3]), max_byte_diff=int(du.max()),
                   alpha_max_byte_diff=int(du[..., 3].max()), frac_bytes_diff=float((du > 0).mean()))
    return r


def load_ref(d: Path) -> np.ndarray:
    m = json.load(open(d / "meta.json"))
    return np.fromfile(d / "vae_out.f32", "<f4").reshape(m["vae_out_shape"])[:, :, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--torch-turbo", required=True, help="_work/vae_ref_<name>/ decoded with the Turbo vae")
    ap.add_argument("--torch-base", required=True, help="_work/vae_ref_<name>/ decoded with the base vae")
    ap.add_argument("--engine", required=True, help="the bundle output engine_parity_vae.py saved (image_<oracle>.f32)")
    ap.add_argument("--fp64-base", default=None, help="_work/vae_ref_<name>/ decoded in float64 with the base vae")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    od, tt_d, tb_d = Path(args.oracle), Path(args.torch_turbo), Path(args.torch_base)
    o = load_ref(od)
    tt, tb = load_ref(tt_d), load_ref(tb_d)
    e = np.fromfile(args.engine, "<f4").reshape(o.shape)
    fl = np.fromfile(od / "final_latents.f32", "<f4")
    for d in (tt_d, tb_d):                                   # the four decodes must share their input
        assert np.array_equal(np.fromfile(d / "final_latents.f32", "<f4"), fl), f"{d} decoded other latents"
    mt, mb = (json.load(open(d / "meta.json")) for d in (tt_d, tb_d))
    res = dict(oracle=str(od), torch_turbo=str(tt_d), torch_base=str(tb_d), engine=str(args.engine),
               torch_turbo_snapshot=mt.get("vae_snapshot"), torch_base_snapshot=mb.get("vae_snapshot"),
               torch_turbo_param_dtypes=mt.get("vae_param_dtypes"), torch_base_param_dtypes=mb.get("vae_param_dtypes"),
               shape=list(o.shape))
    rows = (("i_oracle_vs_torch_turbo", o, tt), ("ii_torch_base_vs_torch_turbo", tb, tt),
            ("iii_engine_vs_torch_base", e, tb), ("iv_engine_vs_oracle", e, o), ("v_engine_vs_torch_turbo", e, tt))
    print(f"[vae-band] {od.name}: shape {list(o.shape)}; Tt = {mt.get('vae_snapshot')} {mt.get('vae_param_dtypes')}; "
          f"Tb = {mb.get('vae_snapshot')} {mb.get('vae_param_dtypes')}", flush=True)
    for key, a, b in rows:
        r = pair(a, b)
        res[key] = r
        print(f"  {key:<30} all: corr {r['all']['corr']:.9f}  max|d| {r['all']['maxd']:.3e}  mean|d| "
              f"{r['all']['mean_abs']:.3e}  rel {r['all']['rel_l2']:.3e}  bit-exact {r['all']['bit_exact']} | u8 PSNR "
              f"{r['u8']['psnr_rgba']:.2f} dB, max byte {r['u8']['max_byte_diff']} (alpha {r['u8']['alpha_max_byte_diff']})",
              flush=True)
        print("  " + " " * 30 + "  ".join(f"{n}: corr {r['channels'][n]['corr']:.7f} max|d| {r['channels'][n]['maxd']:.2e}"
                                          for n in CH), flush=True)
    if args.fp64_base:
        fd = Path(args.fp64_base)
        assert np.array_equal(np.fromfile(fd / "final_latents.f32", "<f4"), fl), f"{fd} decoded other latents"
        mf = json.load(open(fd / "meta.json"))
        assert mf.get("vae_param_dtypes") == ["torch.float64"], mf.get("vae_param_dtypes")
        f64 = np.fromfile(fd / "vae_out.f64", "<f8").reshape(mf["vae_out_shape"])[:, :, 0]
        res["fp64_base"] = dict(dir=str(fd), snapshot=mf.get("vae_snapshot"))
        print(f"[vae-band] fp32 floor vs the float64 decode of the base weights ({fd.name}):", flush=True)
        for key, a in (("floor_torch_base_fp32_vs_fp64", tb), ("floor_engine_vs_fp64", e)):
            s = dict(all=stats(a, f64), channels={n: stats(a[:, c], f64[:, c]) for c, n in enumerate(CH)})
            res["fp64_base"][key] = s
            print(f"  {key:<30} all: max|d| {s['all']['maxd']:.3e}  mean|d| {s['all']['mean_abs']:.3e}  rel "
                  f"{s['all']['rel_l2']:.3e}  | " + "  ".join(f"{n} {s['channels'][n]['maxd']:.2e}" for n in CH),
                  flush=True)
    i_ok = res["i_oracle_vs_torch_turbo"]["all"]["maxd"] <= BAR_I
    iii_ch = res["iii_engine_vs_torch_base"]["channels"].values()
    iii_ok = all(c["nan"] == 0 and c["corr"] >= BAR_III_CORR and c["maxd"] <= BAR_III_MAXD for c in iii_ch)
    bound = (res["iii_engine_vs_torch_base"]["all"]["maxd"] + res["ii_torch_base_vs_torch_turbo"]["all"]["maxd"]
             + res["i_oracle_vs_torch_turbo"]["all"]["maxd"])
    res["triangle"] = dict(iv_maxd=res["iv_engine_vs_oracle"]["all"]["maxd"], bound_iii_plus_ii_plus_i=bound,
                           iv_minus_ii=res["iv_engine_vs_oracle"]["all"]["maxd"]
                           - res["ii_torch_base_vs_torch_turbo"]["all"]["maxd"])
    res["i_ok"], res["iii_ok"] = bool(i_ok), bool(iii_ok)
    res["nan"] = sum(res[k]["all"]["nan"] for k, _, _ in rows)
    res["ok"] = bool(i_ok and iii_ok and res["nan"] == 0)
    print(f"[vae-band] (i) O vs Tt max|d| {res['i_oracle_vs_torch_turbo']['all']['maxd']:.3e} <= {BAR_I:g}: {i_ok}; "
          f"(iii) E vs Tb per channel min corr {min(c['corr'] for c in iii_ch):.7f} >= {BAR_III_CORR:g}, max|d| "
          f"{max(c['maxd'] for c in iii_ch):.3e} <= {BAR_III_MAXD:g}: {iii_ok}; "
          f"(iv) {res['triangle']['iv_maxd']:.3e} vs (ii) {res['ii_torch_base_vs_torch_turbo']['all']['maxd']:.3e} "
          f"(bound (iii)+(ii)+(i) {bound:.3e})  {'PASS' if res['ok'] else 'FAIL'}", flush=True)
    js = Path(args.json or HERE / "_work" / f"vae_band_{od.name}.json")
    js.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(js, "w"), indent=2)
    print(f"[vae-band] -> {js}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
