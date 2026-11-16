"""Upload the MiniCPM5 (1B or 2B) Neural Engine bundles to HF.

Adds two subtrees to mlboydaisuke/MiniCPM5-<size>-CoreAI next to the existing `int8/` (GPU, pipelined):
  ios-ane-h18p/   metadata.json + <name>.h18p.aimodelc + tokenizer/   — AOT for iPhone 17-class (h18p), ANE
  ios-static/     metadata.json + <name>.aimodel      + tokenizer/   — the same stock static export before AOT
and inserts a "Neural Engine bundle" section into the card (README.md) if it is not there yet.
The staging dir is what `export_minicpm5.py --ios-ane` + `_ane_gate` produced and gated on 2026-09-15.

    coreai-models/.venv/bin/python conversion/_minicpm5_ane_hf_upload.py --size 2b|1b [--stage-only]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

SUBTREES = ["ios-ane-h18p", "ios-static"]
MARK = "## Neural Engine bundle"
SIZES = {
    "2b": dict(
        repo="mlboydaisuke/MiniCPM5-2B-CoreAI",
        stage=Path.home() / "code/coreai/coreai-models/exports/hf_stage/MiniCPM5-2B-CoreAI",
        quant="k-means **4-bit palettized, group 32** (Apple's iOS default preset)", size_gb="1.4 GB",
        gate="teacher-forced single-step sweep 24/24 + 8/8 + 16/16 and free-run greedy token-exact including the stop",
        transcript="models/minicpm5-2b/gate-minicpm5-2b-ane-device.json",
        speed="decode **48.0 / 38.2 tok/s**, prefill 1858 / 1494 tok/s; footprint 2.0 GB",
        extra="",
    ),
    "1b": dict(
        repo="mlboydaisuke/MiniCPM5-1B-CoreAI",
        stage=Path.home() / "code/coreai/coreai-models/exports/hf_stage/MiniCPM5-1B-CoreAI",
        quant="k-means **8-bit palettized, group 32** (`conversion/minicpm5_pal8_g32.yaml`)", size_gb="1.3 GB",
        gate="teacher-forced single-step sweep 24/24 + 10/10 + 16/16 and free-run greedy token-exact including the stop",
        transcript="models/minicpm5-1b/gate-minicpm5-1b-ane-device.json",
        speed="decode **69.6 / 58.6 tok/s**, prefill 2710 / 2364 tok/s; footprint 1.3 GB (p128 / g256: decode 62.3)",
        extra=("Apple's default 4-bit preset **fails** the same gate on this checkpoint (two margin-clear flips on a "
               "chat turn, unchanged with fp16 embeddings; `gate-minicpm5-1b-ane-device-4bit-FAIL.json`), so the "
               "1B ships at 8 bits. "),
    ),
}
SECTION = """## Neural Engine bundle (iOS static export, AOT h18p) — 2026-09-15

`ios-ane-h18p/` is Apple's stock `coreai.llm.export --platform iOS` static export of this checkpoint
({quant}; embeddings int8; static graphs `prompt_opt`/`extend` ×
contexts {{256, 512, 1024, 2048, 4096}} × query {{8, 16, 64}}), AOT-compiled with
`xcrun coreai-build compile --platform iOS --preferred-compute neural-engine --architecture h18p`
(31/31 ANE regions, {size_gb}). It loads through Apple's `EngineFactory` → `StaticShapeEngine` unchanged
(iPhone 17-class devices). `ios-static/` is the same export before AOT — the portable IR; compile it
for another chip yourself. {extra}

Gated on an iPhone 17 Pro (iOS 27.0) against the fp32 HF oracle: {gate} — transcript in the zoo,
`{transcript}`. Speed on the same phone (Apple llm-benchmark
method, 512-token prompt, 1024 generated, 5 trials, two back-to-back runs): {speed}. The Neural Engine is the power-efficient lane; these are
not a GPU comparison (different protocol from the `int8/` row above).

"""


def main() -> None:
    stage_only = "--stage-only" in sys.argv
    size = sys.argv[sys.argv.index("--size") + 1] if "--size" in sys.argv else "2b"
    cfg = SIZES[size]
    REPO, STAGE = cfg["repo"], cfg["stage"]
    section = SECTION.format(**cfg)
    assert "__SPEED" not in section, "fill in the measured speed for this size first"
    for sub in SUBTREES:
        d = STAGE / sub
        assert (d / "metadata.json").exists() and (d / "tokenizer").is_dir(), f"stage incomplete: {d}"
    readme = Path(hf_hub_download(REPO, "README.md"))
    card = readme.read_text()
    if MARK not in card:
        anchor = "## Measured"
        assert anchor in card, "card layout changed; insert the section by hand"
        card = card.replace(anchor, section + anchor, 1)
        (STAGE / "README.md").write_text(card)
        print("card: section inserted ->", STAGE / "README.md")
    else:
        print("card: section already present")
    if stage_only:
        print("stage only; nothing uploaded"); return
    api = HfApi()
    for sub in SUBTREES:
        print("uploading", sub)
        api.upload_folder(repo_id=REPO, folder_path=str(STAGE / sub), path_in_repo=sub,
                          commit_message=f"{sub}: stock static iOS export, AOT h18p neural-engine (device gate PASS 3/3)")
    if (STAGE / "README.md").exists():
        api.upload_file(repo_id=REPO, path_or_fileobj=str(STAGE / "README.md"), path_in_repo="README.md",
                        commit_message="card: Neural Engine bundle section")
    print("done")


if __name__ == "__main__":
    main()
