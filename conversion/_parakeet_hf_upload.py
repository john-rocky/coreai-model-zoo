"""Stage + upload a Parakeet-TDT-0.6B Core AI bundle to HF.

Stages the 30 s encoder (fp16) + predict/joint (fp32) `.aimodel`s, the tokenizer assets
(`parakeet/stage_bundle_assets.py`, which retags `tokenizer_class` for swift-transformers),
the librosa-slaney mel filterbank and a cc-by-4.0 model card, then uploads the lot.

Two checkpoints share this script because they share an architecture — v3 (25 European
languages, vocab 8193) and v2 (English-only, vocab 1025, better English WER). Their bundle
*filenames are identical*, so they cannot share a Hugging Face repo: `--repo` picks the
target and `--artifacts` picks which conversion's bundles go into it. `--variant` just
supplies the defaults for one of the two.

    coreai-models/.venv/bin/python conversion/_parakeet_hf_upload.py --variant v3
    coreai-models/.venv/bin/python conversion/_parakeet_hf_upload.py --variant v2 --stage-only
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "parakeet"))
from stage_bundle_assets import MEL_FILE, stage  # noqa: E402

BUNDLES = ["parakeet_encoder_float16_L2885.aimodel", "parakeet_predict_float32.aimodel",
           "parakeet_joint_float32.aimodel"]

VARIANTS = {
    "v3": {
        "repo": "mlboydaisuke/Parakeet-TDT-0.6B-CoreAI",
        "artifacts": "artifacts",
        "hf_id": "nvidia/parakeet-tdt-0.6b-v3",
        "title": "Parakeet-TDT-0.6B — Core AI",
        "vocab": 8193, "blank": 8192, "tokens": 77,
        "languages": "25 European languages",
        "extra": "the first **transducer / TDT (RNN-T family)** ASR in the zoo",
        "swift_note": ", and again token-exact through the Swift **CoreAIKit** "
                      "`KitParakeetModel`",
        "use_init": "try await KitParakeetModel(model: .parakeetTDT)",
    },
    "v2": {
        "repo": "rahulrachuri/parakeet-tdt-0.6b-v2-coreai",
        "artifacts": "artifacts_v2",
        "hf_id": "nvidia/parakeet-tdt-0.6b-v2",
        "title": "Parakeet-TDT-0.6B-v2 — Core AI",
        "vocab": 1025, "blank": 1024, "tokens": 82,
        "languages": "English",
        "extra": "the **English-specialist** sibling of the v3 port (lower English WER, "
                 "1025-entry vocabulary)",
    },
}

CARD = """---
license: cc-by-4.0
library_name: coreai
pipeline_tag: automatic-speech-recognition
base_model: {hf_id}
tags: [core-ai, coreaikit, parakeet, tdt, rnn-t, transducer, asr, on-device, apple]
---

# {title}

[`{hf_id}`](https://huggingface.co/{hf_id}) (cc-by-4.0, 600M) converted to **Apple Core AI**
`.aimodel` — {extra}
([zoo](https://github.com/john-rocky/coreai-models-community)). Transcribes ≤~29 s clips in
{languages} as **three stateless graphs + a host greedy loop** (no LLM runtime).

- `parakeet_encoder_float16_L2885.aimodel` — FastConformer encoder + projector (fp16, ~1.2 GB),
  `mel[1,128,2885] → enc_proj[1,361,640]`.
- `parakeet_predict_float32.aimodel` — embedding → 2-layer LSTM → projector (fp32),
  `token[1,1],h,c[2,1,640] → dec_out[1,640],h',c'`.
- `parakeet_joint_float32.aimodel` — `head(relu(enc_frame+dec_out))` (fp32),
  `→ token_logits[1,{vocab}], dur_logits[1,5]`.
- `tokenizer.json` (+ `tokenizer_config.json`), `{mel_file}` (librosa-slaney).

Gated **{tokens}/{tokens} token-exact** end-to-end vs the HF `ParakeetForTDT` reference.
blank {blank} · durations [0,1,2,3,4] · 16 kHz.

## Use (CoreAIKit)

```swift
let parakeet = try await KitParakeetModel(bundleAt: bundleDirectory)
let result = try await parakeet.transcribe(samples: pcm16kMono)   // 16 kHz mono Float
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="v3",
                    help="which checkpoint's bundle to publish (supplies the defaults below)")
    ap.add_argument("--repo", help="target HF repo id (default: the variant's)")
    ap.add_argument("--artifacts", help="bundle dir under conversion/parakeet/ (default: the variant's)")
    ap.add_argument("--hf-id", help="source checkpoint, or a local dir in HF layout, for the "
                                    "tokenizer assets (default: the variant's)")
    ap.add_argument("--stage", default=None, help="staging directory (default: /tmp/parakeet_hf_<variant>)")
    ap.add_argument("--stage-only", action="store_true", help="build the upload directory, do not upload")
    args = ap.parse_args()

    v = VARIANTS[args.variant]
    repo = args.repo or v["repo"]
    art = HERE / "parakeet" / (args.artifacts or v["artifacts"])
    hf_id = args.hf_id or v["hf_id"]
    stage_dir = Path(args.stage) if args.stage else Path(f"/tmp/parakeet_hf_{args.variant}")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for name in BUNDLES:
        shutil.copytree(art / name, stage_dir / name)
    stage(hf_id, stage_dir)
    (stage_dir / "README.md").write_text(CARD.format(mel_file=MEL_FILE, **v, hf_id=hf_id))
    print(f"[stage] {stage_dir} -> {repo}")

    if args.stage_only:
        print("[stage-only] not uploading")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(stage_dir), repo_id=repo, repo_type="model",
                      commit_message=f"{v['title']}: encoder + predict + joint + tokenizer")
    print("uploaded", repo)


if __name__ == "__main__":
    main()
