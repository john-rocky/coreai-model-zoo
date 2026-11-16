"""Replace the MiniCPM5-2B Neural Engine bundles on HF with the 6-bit export (S5, 2026-09-16).

mlboydaisuke/MiniCPM5-2B-CoreAI:
  ios-ane-h18p/   metadata.json + minicpm5_2b_minicpm5_pal6_g8_static.h18p.aimodelc + tokenizer/   (AOT h18p, ANE, 2.5 GB)
  ios-static/     metadata.json + minicpm5_2b_minicpm5_pal6_g8_static.aimodel      + tokenizer/   (the IR before AOT, 1.9 GB)
The previous files in those two subtrees (the 2026-09-15 4-bit g32 export) are deleted in the same commits, and the card's
"Neural Engine bundle" section is rewritten. Staging = coreai-models/exports/hf_stage/MiniCPM5-2B-CoreAI/ (built from
export_minicpm5.py --ios-ane --qconfig minicpm5_pal6_g8.yaml == ondevice/_ane_gate/_ane_export_s1.py, gated on the phone).

    HF_HUB_DISABLE_XET=1 coreai-models/.venv/bin/python conversion/_minicpm5_2b_ane6_hf_upload.py [--stage-only]
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/MiniCPM5-2B-CoreAI"
STAGE = Path.home() / "code/coreai/coreai-models/exports/hf_stage/MiniCPM5-2B-CoreAI"
SUBTREES = ["ios-ane-h18p", "ios-static"]
MARK = "## Neural Engine bundle"
SECTION = """## Neural Engine bundle (iOS static export, AOT h18p) — 2026-09-16

`ios-ane-h18p/` is Apple's stock `coreai.llm.export --platform iOS` static export of this checkpoint at
**6-bit k-means palettization, group 8** (the shape of Apple's own iOS preset for Qwen3-1.7B;
`conversion/minicpm5_pal6_g8.yaml` in the zoo; embeddings int8; static graphs `prompt_opt`/`extend` ×
contexts {256, 512, 1024, 2048, 4096} × query {8, 16, 64}), AOT-compiled with
`xcrun coreai-build compile --platform iOS --preferred-compute neural-engine --architecture h18p`
(31/31 ANE regions, 2.5 GB). It loads through Apple's `EngineFactory` → `StaticShapeEngine` unchanged
(iPhone 17-class devices). `ios-static/` is the same export before AOT — the portable IR; compile it
for another chip yourself.

Why 6-bit: the 2026-09-15 bundle was Apple's iOS default (4-bit, group 32). It passed the three short
gate prompts but diverged from fp32 at the first token of a 109-token free-form answer (and got the
physics wrong). At 6-bit the same phone matches fp32 token-for-token on the three short prompts (24/24,
8/8 including the stop, 16/16) and for 33 tokens of the long answer, then diverges on a step where fp32
itself is nearly tied (top-2 margin 0.17); 5 of its 109 teacher-forced steps differ — transcript
`models/minicpm5-2b/gate-minicpm5-2b-ane-6bit-device.json` in the zoo. An 8-bit shape is exact in an
fp32 simulation of the recipe, but the 2B at 8 bits (2.16 GB of weights) loads on the ANE and never
returns its first token, and 4-bit group 8 does not compile for the ANE at all — zoo
`knowledge/ane-vs-gpu-iphone-2026-09.md` §9.

Speed on an iPhone 17 Pro (iOS 27.0), same day, engine released and idle until the thermal state was
back at `fair`, then 60 s of 128-token-prompt / 256-token trials: **38.5 tok/s decode** (flat over the
minute), prefill 1679 tok/s, footprint 2.6 GB. In the same run the `int8/` GPU bundle decoded 23.8 tok/s;
the previous 4-bit ANE bundle did ~55 (`models/minicpm5-2b/bench-iphone-ane-6bit-vs-gpu-2026-09-16.json`).
**The first launch on a phone builds the ANE programs: about 23 minutes** (0.2 s afterwards; the cache
lives in the app's container and is invalidated by an iOS update).

Task accuracy, GSM8K test (first 200 questions, 0-shot CoT, greedy, no-think, max 640 new tokens; same
prompt and scoring as the litertlm-convert evals) — `models/minicpm5-2b/gsm8k-200-2026-09-17.json` in the zoo:

| model | correct / 200 |
|---|---:|
| fp32 checkpoint (bf16 on a Mac) | 172 (86.0 %) |
| **`ios-ane-h18p/` 6-bit, on the iPhone 17 Pro** | **173 (86.5 %)** |
| the same 6-bit recipe applied to the fp32 weights (Mac) | 172 (86.0 %) |
| the `int8/` recipe applied to the fp32 weights (Mac) | 171 (85.5 %) |
| the replaced 4-bit bundle, on the iPhone 17 Pro | 131 (65.5 %) |

"""


def main() -> None:
    stage_only = "--stage-only" in sys.argv
    for sub in SUBTREES:
        d = STAGE / sub
        assert (d / "metadata.json").exists() and (d / "tokenizer").is_dir(), f"stage incomplete: {d}"
        m = json.load(open(d / "metadata.json"))
        assert m["compression"] == "minicpm5_pal6_g8" and (d / m["assets"]["main"]).exists(), d
    # author/license/description: the int8 bundle's metadata carries them (exported with the overlay's registry entry)
    try:
        src = json.load(open(hf_hub_download(REPO, "int8/metadata.json")))
        for sub in SUBTREES:
            p = STAGE / sub / "metadata.json"; m = json.load(open(p)); changed = False
            for k in ("author", "license", "description"):
                if not m.get(k) and src.get(k): m[k] = src[k]; changed = True
            if changed: json.dump(m, open(p, "w"), indent=2); print("metadata fields filled from int8/:", sub)
    except Exception as e:  # noqa: BLE001
        print("metadata fields not filled:", e)
    readme = Path(hf_hub_download(REPO, "README.md")); card = readme.read_text()
    assert MARK in card, "card has no Neural Engine section; insert by hand"
    i = card.index(MARK); j = card.index("\n## ", i + len(MARK))
    card = card[:i] + SECTION + card[j + 1:]
    (STAGE / "README.md").write_text(card); print("card: section rewritten ->", STAGE / "README.md")
    if stage_only:
        print("stage only; nothing uploaded"); return
    api = HfApi()
    files = api.list_repo_files(REPO)
    # 2026-09-17 completion: the ios-ane-h18p/ 6-bit files landed on 09-16 (metadata.json already points at them); the 4-bit
    # aimodelc survived because delete_patterns are relative to path_in_repo — remove it explicitly, then replace ios-static/.
    old_aot = "ios-ane-h18p/minicpm5_2b_4bit_weight_palettized_group32_static.h18p.aimodelc"
    if any(f.startswith(old_aot + "/") for f in files):
        print("deleting", old_aot)
        api.delete_folder(repo_id=REPO, path_in_repo=old_aot, commit_message="ios-ane-h18p: drop the 4-bit g32 aimodelc (replaced by the 6-bit export; metadata.json points at the 6-bit one)")
    if not any(f.startswith("ios-ane-h18p/minicpm5_2b_minicpm5_pal6_g8_static.h18p.aimodelc/") for f in files):
        print("uploading ios-ane-h18p")
        api.upload_folder(repo_id=REPO, folder_path=str(STAGE / "ios-ane-h18p"), path_in_repo="ios-ane-h18p", delete_patterns=["**"],
                          commit_message="ios-ane-h18p: 6-bit k-means g8 static iOS export, AOT h18p neural-engine (replaces the 4-bit g32 export of 2026-09-15)")
    print("uploading ios-static (replacing the 4-bit IR)")
    api.upload_folder(repo_id=REPO, folder_path=str(STAGE / "ios-static"), path_in_repo="ios-static", delete_patterns=["**"],
                      commit_message="ios-static: the 6-bit k-means g8 static iOS export before AOT (replaces the 4-bit g32 IR of 2026-09-15)")
    api.upload_file(repo_id=REPO, path_or_fileobj=str(STAGE / "README.md"), path_in_repo="README.md",
                    commit_message="card: Neural Engine bundle section — 6-bit, why, speed, first-load cost, GSM8K 200")
    print("done:", [f for f in api.list_repo_files(REPO) if f.startswith(("ios-ane-h18p/", "ios-static/")) and f.count("/") <= 2])


if __name__ == "__main__":
    main()
