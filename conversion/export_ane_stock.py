#!/usr/bin/env python3
"""Neural Engine lane for any checkpoint Apple's STOCK iOS builders accept (mistral / olmo2 /
qwen2 / qwen3 + the overlay's plain-Llama remap): stock static iOS export -> AOT for the ANE ->
region count -> a loadable device bundle -> HF staging. Generalizes `export_minicpm5.py
--ios-ane` (2026-09-15) so a new repo is one command, not a new script.

    python conversion/export_ane_stock.py <hf-id | Apple preset short-name> \\
        [--compression 4bit_weight_palettized_group32 | --qconfig <kmeans yaml>] \\
        [--max-context-length 4096] [--output-dir DIR] [--author .. --license .. --description ..] \\
        [--chat-eos <token>] [--devbundle NAME] [--stage <HF repo name>]

    # AOT + gate an IR that already exists (e.g. the `ios/` subtree of an official repo):
    python conversion/export_ane_stock.py --aot-only <dir with metadata.json + *.aimodel + tokenizer/> \\
        --output-dir DIR [--devbundle NAME] [--stage <HF repo name>]

What it does, in order (each step is the same command the MiniCPM5 lane ran by hand):

1. `uv run coreai.llm.export <model> --platform iOS --compression <preset> --max-context-length N`
   in the coreai-models checkout (default `../../coreai-models-rebase`: coreai-torch 0.4.2 with the
   zoo overlay applied — the overlay adds the `llama -> mistral` remap and metadata entries, the
   builders and the k-means palettizer are Apple's). A raw hf id needs `--experimental
   --compute-precision float16`; an Apple preset short-name (`qwen3-0.6b`) resolves them itself.
   ALWAYS pass `--max-context-length` for an unregistered id: otherwise the static graph set is built
   from config.max_position_embeddings (32k-131k). The iOS CLI couples the scheme to the platform:
   `--compression` is a palettization preset, `--qconfig` a `kmeans_palettization_config` yaml
   (8-bit rescue = the `minicpm5_pal8_g32.yaml` shape). Linear int4 cannot reach the ANE.
2. Patches author/license/description into `<name>.aimodel/metadata.json` when given (the exporter
   only warns for an id missing from export/metadata.py; the fields ride into the .aimodelc).
3. Optionally rewrites the tokenizer eos (`--chat-eos <|im_end|>`) for checkpoints whose base eos
   is not the chat turn terminator (MiniCPM5). Qwen ships the right one already.
4. `xcrun coreai-build compile <aimodel> --output <out>/aot_h18p_ane --platform iOS
   --preferred-compute neural-engine --architecture h18p` (Xcode 27.0 RC, coreai-build 3600.83.1)
   then counts `*ANE_region*` entries in the .aimodelc. **0 regions = silent GPU fallback = FAIL**;
   the standard prompt_opt/extend
   {256..4096} x {8,16,64} + load_embeddings set gives 31.
5. Writes `<out>/aot_h18p_ane/metadata.json` (assets.main -> the .aimodelc, compilation.targets) and
   copies `tokenizer/` next to it: that directory IS a loadable device bundle (EngineFactory ->
   StaticShapeEngine) — `--devbundle NAME` clones it to `exports/apple_bench/devbundles/NAME` for
   `ondevice/_ane_gate` (`BUNDLE=... ./_build.sh`) and the bench app.
6. `--stage <Repo>` clones the two HF subtrees to `exports/hf_stage/<Repo>/ios-ane-h18p/` (AOT) and
   `ios-static/` (the IR before AOT — portable, compile it for another chip).
   Upload: `_ane_stock_hf_upload.py`. Clones are APFS `cp -c` (no extra bytes).

A summary lands in `<out>/_ane_stock.json` (sizes, region count, paths). Gate the device bundle
with `ondevice/_ane_gate` (red first, then clean) before anything ships; 4-bit FAIL -> re-run with
`--qconfig conversion/minicpm5_pal8_g32.yaml` (8-bit) -> FAIL again = do not ship, put the FAIL
transcript on the card (cards never lie).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                               # ~/code/coreai
DEFAULT_CHECKOUT = ROOT / "coreai-models-rebase"        # coreai-torch 0.4.2 + zoo overlay (do not edit)
EXPORTS = ROOT / "coreai-models" / "exports"            # where every lane's artifacts live
DEVBUNDLES = EXPORTS / "apple_bench" / "devbundles"
HF_STAGE = EXPORTS / "hf_stage"
IOS_COMPRESSION = "4bit_weight_palettized_group32"      # Apple's iOS default preset (embeddings int8)
IOS_MAX_CONTEXT = 4096
AOT_ARCH = "h18p"                                       # iPhone 17-class
AOT_XCODE = "/Applications/Xcode-27.0.0-RC.app/Contents/Developer"   # coreai-build 3600.83.1
STANDARD_REGIONS = 31                                   # 5 contexts x 3 query x {prompt_opt, extend} + load_embeddings


def sh(cmd: list[str], cwd: Path | None = None, log: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if log:
        with log.open("a") as f:
            f.write("+ " + " ".join(str(c) for c in cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(str(c) for c in cmd)}"
                         + (f"\n  log: {log}" if log else ""))


def du_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def clone_tree(src: Path, dst: Path) -> None:
    """APFS clone (cp -c): instant, no extra bytes. Replaces dst."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True)


def export(model: str, out_dir: Path, checkout: Path, compression: str | None, qconfig: Path | None,
           max_context: int, overwrite: bool, extra: list[str]) -> Path:
    """Apple's stock static iOS export. Returns the `<name>.aimodel` directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "coreai.llm.export", model, "--platform", "iOS",
           "--max-context-length", str(max_context), "--output-dir", str(out_dir)]
    if "/" in model:  # raw hf id, not an Apple preset short-name
        cmd += ["--experimental", "--compute-precision", "float16"]
    if qconfig is not None:
        cmd += ["--compression-config", str(qconfig.resolve())]   # absolute: the exporter's cwd is the checkout
    elif compression is not None:
        cmd += ["--compression", compression]
    if overwrite:
        cmd.append("--overwrite")
    cmd += extra
    log = out_dir / "_export.log"
    with log.open("a") as f:
        f.write(f"export start {time.strftime('%Y-%m-%d %H:%M:%S')} checkout={checkout}\n")
    t0 = time.time()
    sh(cmd, cwd=checkout, log=log, env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    with log.open("a") as f:
        f.write(f"export end {time.strftime('%Y-%m-%d %H:%M:%S')} ({time.time() - t0:.0f} s)\n")
    aimodels = sorted(out_dir.rglob("*.aimodel"))
    if not aimodels:
        raise SystemExit(f"no *.aimodel under {out_dir} (see {log})")
    return aimodels[-1]


def patch_metadata(aimodel: Path, author: str | None, license_: str | None, description: str | None) -> None:
    meta_path = aimodel / "metadata.json"
    meta = json.loads(meta_path.read_text())
    changed = False
    for key, val in (("author", author), ("license", license_), ("description", description)):
        if val and meta.get(key) != val:
            meta[key] = val
            changed = True
    if changed:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"patched {meta_path.name}: author/license/description")
    missing = [k for k in ("author", "license", "description") if not meta.get(k)]
    if missing:
        print(f"WARNING {meta_path}: missing {missing} (unregistered hf id; pass --author/--license/--description)")


def fix_chat_eos(tok_dir: Path, eos: str) -> None:
    """The engine stops on the single eosTokenId; a chat bundle must declare the chat turn
    terminator (MiniCPM5: <|im_end|>, base eos </s>) or it never halts."""
    cfg = tok_dir / "tokenizer_config.json"
    d = json.loads(cfg.read_text())
    d["eos_token"] = eos
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    stm = tok_dir / "special_tokens_map.json"
    if stm.exists():
        m = json.loads(stm.read_text())
        if isinstance(m.get("eos_token"), dict):
            m["eos_token"]["content"] = eos
        else:
            m["eos_token"] = eos
        stm.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"chat eos -> {eos} in {tok_dir}")


def aot(aimodel: Path, aot_dir: Path, arch: str) -> tuple[Path, int]:
    """AOT for the Neural Engine. Returns (aimodelc, ANE region count)."""
    if aot_dir.exists():
        shutil.rmtree(aot_dir)
    aot_dir.mkdir(parents=True)
    log = aot_dir.parent / f"_aot_{arch}_ane.log"
    t0 = time.time()
    sh(["xcrun", "coreai-build", "compile", str(aimodel), "--output", str(aot_dir),
        "--platform", "iOS", "--preferred-compute", "neural-engine", "--architecture", arch],
       log=log, env={**os.environ, "DEVELOPER_DIR": AOT_XCODE})
    print(f"AOT took {time.time() - t0:.0f} s")
    aimodelc = next(aot_dir.glob("*.aimodelc"))
    regions = sum(1 for _ in aimodelc.rglob("*ANE_region*"))
    print(f"AOT {aimodelc.name}: {regions} ANE regions (standard graph set = {STANDARD_REGIONS})")
    return aimodelc, regions


def make_device_bundle(ir_dir: Path, aimodel: Path, aimodelc: Path, aot_dir: Path, arch: str) -> None:
    """metadata.json next to the .aimodelc (assets.main -> it) + tokenizer/ = loadable bundle."""
    meta = json.loads((ir_dir / "metadata.json").read_text())
    meta["assets"]["main"] = aimodelc.name
    meta.setdefault("compilation", {})["targets"] = [
        f"{arch} neural-engine (xcrun coreai-build compile --platform iOS --preferred-compute neural-engine "
        f"--architecture {arch})"]
    (aot_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    tok = ir_dir / "tokenizer"
    if tok.is_dir():
        shutil.copytree(tok, aot_dir / "tokenizer", dirs_exist_ok=True)
    else:
        print(f"WARNING no tokenizer/ next to {aimodel.name}; the bundle needs one to load")


def stage_hf(repo: str, ir_dir: Path, aimodel: Path, aot_dir: Path, arch: str) -> dict:
    stage = HF_STAGE / repo
    ane = stage / f"ios-ane-{arch}"
    static = stage / "ios-static"
    clone_tree(aot_dir, ane)
    if static.exists():
        shutil.rmtree(static)
    static.mkdir(parents=True)
    for name in ("metadata.json", "tokenizer", aimodel.name):
        src = ir_dir / name
        if src.exists():
            clone_tree(src, static / name)
    # the static metadata must point at the IR, not the aimodelc
    meta = json.loads((static / "metadata.json").read_text())
    meta["assets"]["main"] = aimodel.name
    (static / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"ios_ane": str(ane), "ios_static": str(static)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="hf id (org/name) or an Apple registry preset short-name (qwen3-0.6b)")
    ap.add_argument("--aot-only", metavar="IR_DIR",
                    help="skip the export: AOT this existing bundle dir (metadata.json + *.aimodel + tokenizer/)")
    ap.add_argument("--compression", default=None,
                    help=f"iOS palettization preset (default for a raw hf id: {IOS_COMPRESSION}; "
                         "a preset short-name keeps Apple's registry recipe unless this is given)")
    ap.add_argument("--qconfig", default=None, help="kmeans_palettization_config yaml (e.g. minicpm5_pal8_g32.yaml = 8-bit)")
    ap.add_argument("--max-context-length", type=int, default=IOS_MAX_CONTEXT)
    ap.add_argument("--output-dir", default=None, help="default: exports/<name>_ios_ane[_<qconfig stem>]")
    ap.add_argument("--checkout", default=str(DEFAULT_CHECKOUT), help="coreai-models checkout to run the exporter in")
    ap.add_argument("--arch", default=AOT_ARCH)
    ap.add_argument("--author"); ap.add_argument("--license"); ap.add_argument("--description")
    ap.add_argument("--chat-eos", default=None, help="rewrite the tokenizer eos to this token (e.g. <|im_end|>)")
    ap.add_argument("--devbundle", default=None, metavar="NAME",
                    help=f"clone the device bundle to {DEVBUNDLES}/NAME (for ondevice/_ane_gate)")
    ap.add_argument("--stage", default=None, metavar="REPO",
                    help=f"clone ios-ane-<arch>/ + ios-static/ to {HF_STAGE}/REPO")
    ap.add_argument("--overwrite", action="store_true", help="pass --overwrite to the exporter")
    ap.add_argument("--skip-export", action="store_true", help="reuse an existing export under --output-dir")
    ap.add_argument("--export-extra", default="", help="extra args for coreai.llm.export, space separated")
    args = ap.parse_args()

    if not args.model and not args.aot_only:
        ap.error("give a model or --aot-only IR_DIR")
    if args.compression and args.qconfig:
        ap.error("--compression and --qconfig are mutually exclusive")
    qconfig = Path(args.qconfig) if args.qconfig else None
    if qconfig and not qconfig.exists():
        ap.error(f"--qconfig not found: {qconfig}")

    summary: dict = {"model": args.model, "arch": args.arch, "checkout": args.checkout,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    def under_exports(p: str) -> Path:
        """A relative path is tried against the cwd first, then against exports/ (recipes use the latter)."""
        cand = Path(p)
        return cand.resolve() if cand.is_absolute() or cand.exists() else (EXPORTS / p).resolve()

    if args.aot_only:
        ir_dir = under_exports(args.aot_only)
        aimodel = next(ir_dir.glob("*.aimodel"))
        out_dir = under_exports(args.output_dir) if args.output_dir else ir_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        summary["export"] = "skipped (--aot-only)"
    else:
        name = args.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        suffix = f"_{qconfig.stem}" if qconfig else ""
        out_dir = under_exports(args.output_dir) if args.output_dir else EXPORTS / f"{name}_ios_ane{suffix}"
        if args.skip_export:
            aimodel = sorted(out_dir.rglob("*.aimodel"))[-1]
            summary["export"] = "skipped (--skip-export)"
        else:
            compression = args.compression
            if compression is None and qconfig is None and "/" in args.model:
                compression = IOS_COMPRESSION
            aimodel = export(args.model, out_dir, Path(args.checkout).resolve(), compression, qconfig,
                             args.max_context_length, args.overwrite, args.export_extra.split())
            summary["export"] = {"compression": qconfig.name if qconfig else (compression or "registry preset"),
                                 "max_context_length": args.max_context_length}
        ir_dir = aimodel.parent
    print(f"IR: {aimodel} ({du_bytes(aimodel) / 1e9:.2f} GB)")
    summary["aimodel"] = str(aimodel)
    summary["aimodel_gb"] = round(du_bytes(aimodel) / 1e9, 3)

    patch_metadata(aimodel, args.author, args.license, args.description)
    if args.chat_eos:
        for tok in sorted(ir_dir.rglob("tokenizer")):
            fix_chat_eos(tok, args.chat_eos)

    aot_dir = out_dir / f"aot_{args.arch}_ane"
    aimodelc, regions = aot(aimodel, aot_dir, args.arch)
    make_device_bundle(ir_dir, aimodel, aimodelc, aot_dir, args.arch)
    summary.update({"aimodelc": str(aimodelc), "aimodelc_gb": round(du_bytes(aimodelc) / 1e9, 3),
                    "ane_regions": regions, "device_bundle": str(aot_dir)})

    if args.devbundle:
        dst = DEVBUNDLES / args.devbundle
        clone_tree(aot_dir, dst)
        summary["devbundle"] = str(dst)
        print(f"devbundle: {dst}")
    if args.stage:
        summary["hf_stage"] = stage_hf(args.stage, ir_dir, aimodel, aot_dir, args.arch)
        print(f"hf stage: {summary['hf_stage']}")

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "_ane_stock.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if regions == 0:
        raise SystemExit("FAIL: 0 ANE regions — coreai-build fell back to the GPU (linear int4 or a non-palettized graph?)")


if __name__ == "__main__":
    main()
