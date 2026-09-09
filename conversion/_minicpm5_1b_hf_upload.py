"""Stage + upload the MiniCPM5-1B Core AI int8 (per-block-32) bundle as a NEW REVISION of the
published repo. USER-GATED — run only when asked.

Replaces the per-channel bundle of rev 5ad650f (2026-07-20), whose LM head has dead rows from
vocab id ~65024 up (<|im_end|> included — it never halted a chat turn). Layout is unchanged so the
kit only needs a revision bump: `int8/` holds the FM-format dynamic bundle (metadata.json +
minicpm5_1b_int8.aimodel + tokenizer/), LICENSE + config.json at the root, README.md is the card.
The tokenizer files are byte-identical to the published ones (checked 2026-09-09); LICENSE and
config.json are re-used from the repo. Only the asset, its metadata and the card change.

The card keeps the gen-cards managed blocks (devicemark, use-it) and the funnel block verbatim
from the published README — they belong to scripts/gen-cards, not to this script.

    coreai-models/.venv/bin/python conversion/_minicpm5_1b_hf_upload.py [--stage-only]
"""
import json
import os
import re
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "mlboydaisuke/MiniCPM5-1B-CoreAI"
SRC = "openbmb/MiniCPM5-1B"
BUNDLE = (Path.home() / "code/coreai/coreai-models/exports/minicpm5-1b-b32-20260909"
          / "minicpm5_1b_minicpm5_int8sym_b32_dynamic")            # what coreai.llm.export wrote
PUBLISHED_NAME = "minicpm5_1b_int8"                                # what the repo calls it (unchanged)
STAGE = Path("/private/tmp/claude-501/-Users-majimadaisuke-code-coreai-coreai-models-community"
             "/f2dbfa48-1d05-4cdd-a4eb-3302f5e4fb03/scratchpad/minicpm5_1b_hf_stage")
COMMIT = ("Re-export int8 per-block-32: the per-channel bundle had dead LM-head rows from vocab "
          "id ~65024 up (never emitted <|im_end|>)")

# ---- card: everything outside the managed blocks ------------------------------------------
TITLE_AND_INTRO = """# MiniCPM5-1B — Core AI (int8 block-32, runs on iPhone)

Apple **Core AI** (`.aimodel`) conversion of [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) —
OpenBMB's 1.08B on-device LLM with **hybrid Think / No-Think reasoning** and **128K** context, reaching
1B-class open-source SOTA. Runs fully on-device on **iPhone** and Apple Silicon Macs (GPU, pipelined engine).

> **Revision note (2026-09-09).** This revision replaces the per-channel int8 bundle published as
> `5ad650f`. That bundle's LM head had **dead rows from vocab id ~65024 up** — every token there
> scored ~0 in the engine, `<|im_end|>` (130073) included — so a chat turn **never halted** (it ran
> to the token cap) and any answer needing a high-id token lost it. The card's "halts cleanly" line
> was wrong. Details and the measurements are under *Why per-block-32* below; pin this revision or
> newer.

Part of the community Core AI model zoo: **https://github.com/john-rocky/coreai-model-zoo**

"""

BODY = """## Measured

| | decode | prefill | numerics | size |
|---|---:|---:|---|---:|
| **iPhone 17 Pro** (A19 Pro, `PipelinedBench`, Release) | **61.7 tok/s** | 65.6 tok/s | **24/24 token-exact** vs HF fp32 on the margin-clean alphabet prompt (min fp32 top-2 margin 0.841) **+ 6/6 including the stop** on the no-think turn `1+1=?` (`1+1=2` then `<|im_end|>`); engine ready 7.3 s cold | **1.1 GB** |
| **M4 Max** (macOS 27, `llm-benchmark`, 512p/1024g) | **246.6 tok/s** | 6649 tok/s | **16/16 token-exact** vs the fp32 oracle (margin-aware gate, min margin 0.913) + the same 6/6 stop gate | |

Halt check through the engine on the same phone-shaped bundle: the Think-mode turn `1+1=?` stops on
its own after 171 tokens (cap 400); the no-think turn after 6.

⚠️ **iPhone context cap: prompt + generated tokens must stay under 1024.** The bundle declares a 131072
dynamic KV, and the shipped `CoreAIPipelinedEngine` caps iOS growing-KV capacity at 1024 (its guard
against the iOS compiler miscompiling growing-KV specializations at seq ≥ 2048). Chunk or trim the
history on iOS; macOS has no cap.

## Why per-block-32 (what was wrong with the previous revision)

The previous bundle (`5ad650f`, int8 **per-channel** absmax) passed a 24-token greedy parity check and
still never ended a chat turn. Teacher-forcing it through the engine against the fp32 reference
located the defect in the LM head, by vocab id:

| token (id) | fp32 P | per-channel bundle | per-block-32 bundle (this revision) |
|---|---:|---:|---:|
| `.lineTo` (65023) | 0.978 | 0.993 | — |
| `粒子` (65039) | 0.977 | **0.0000** | 0.98 |
| ` chromosomal` (65528) | 0.383 | **0.0000** | 0.380 |
| ` OpenAI` (130051) | 0.929 | **0.0000** | 0.923 |
| `\\n\\n` after `</think>` (130063) | 0.9999 | **0.0000** | 0.9999 |
| `<|im_end|>` after `1+1=2` (130073) | 0.873 | **0.0000** (top-1 `.`) | 0.867 |

Every probed row at id ≥ 65039 is dead in the per-channel bundle and every probed row ≤ 65023 is
healthy (16 probes; the boundary lies in that 16-id window, which contains 65024 = 127 × 512). Rows
below it match fp32 to ~0.01 in probability, which is why free-running English prose looked fine.
A fresh per-channel export on the same toolchain (coreai-torch 0.4.1 / coreai-opt 0.2.1 /
coreai-core 1.0.0b2) reproduces the shipped bundle's logits to four decimals, so this is a property
of the per-channel int8 path for this 130560-row head, not a one-off; per-block-32 (this bundle),
the CLI's default int4 preset and fp16 (`--compression none`) are all clean on the same probes.
Which component owns the row cut-off (quantizer, converter or the runtime's per-channel int8 matmul)
is not established here.

The gate that catches it is now part of the zoo's `cli/coreai_verify.py`: `--chat no-think --prompt
"1+1=?"` makes the fp32 oracle's `<|im_end|>` a gated step — a bundle that runs past a stop the oracle
takes at margin 0.80 fails — and `--must-stop-within N` is the plain halt check. The old bundle is red
on both, this one green.

## Quantization

Weight-only **symmetric int8, per-block-32** (a scale per 32-wide block along the input dim; no
clipping), applied as a torch pre-export pass via `coreai-opt`; SDPA / RoPE / RMSNorm stay full
precision — the same YAML that ships the [2B](https://huggingface.co/mlboydaisuke/MiniCPM5-2B-CoreAI).

```bash
uv run coreai.llm.export openbmb/MiniCPM5-1B --experimental --compute-precision float16 \\
  --compression-config minicpm5_int8sym_b32.yaml
# minicpm5_int8sym_b32.yaml: quantization_config → op_state_spec.weight = {dtype: int8,
#   qscheme: symmetric, granularity: {type: per_block, block_size: 32}}
```

## Conversion notes

- **`llama → mistral` remap.** MiniCPM5-1B's `model_type` is `llama`; the stock exporter has no
  `llama` graph family, but Mistral's builder is architecturally identical for this config (GQA,
  no qkv bias, no qk-norm, explicit `head_dim` honored). One-line remap in the model registry.
- **Chat EOS.** Base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>`
  (id 130073). The bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships). Checked
  through the engine with the chat template applied: the no-think turn `1+1=?` answers `1+1=2` and
  stops at step 5, where the fp32 reference stops (margin 0.80).
- **Dynamic-shape bundle** → the Core AI pipelined engine (the iPhone path); a static iOS export
  routes to the static-shape engine instead, which this FM-format bundle doesn't target.
- **Thinking.** The model thinks by default (`<think>…</think>` before the answer); pass
  `enable_thinking=False` through the chat template for a direct answer. Give generation a
  generous budget (the kit caps at 4096) — the think trace alone can run a few hundred tokens.

## Run

```swift
// iOS / macOS, via Foundation Models
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: modelURL)   // int8/ bundle
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```

## Reproduce

Exporter, gate, card and port notes live in the
[Core AI model zoo](https://github.com/john-rocky/coreai-model-zoo):
[`models/minicpm5-1b/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/minicpm5-1b),
[`conversion/export_minicpm5.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_minicpm5.py),
[`knowledge/minicpm5-1b.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/minicpm5-1b.md).

```bash
python3 conversion/zoo_convert.py show minicpm5-1b
python3 conversion/zoo_convert.py run  minicpm5-1b
python3 cli/coreai_verify.py <bundle> --chat no-think --prompt "1+1=?" -n 16 --must-stop-within 16
```

## License

Apache-2.0 (upstream MiniCPM5 license). Model © OpenBMB — see
https://huggingface.co/openbmb/MiniCPM5-1B. Conversion: community.

"""

USE_IT_BEGIN = "<!-- gen-cards:use-it begin"
USE_IT_END = "<!-- gen-cards:use-it end -->"
FUNNEL_BEGIN = "<!-- funnel:v1 -->"
TITLE = "# MiniCPM5-1B — Core AI"
FIRST_SECTION_AFTER_USE_IT = "## On-device numbers"


def build_card(published: str) -> str:
    """Published card with the three unmanaged regions replaced; managed blocks untouched."""
    i_title = published.index(TITLE)
    i_use_begin = published.index(USE_IT_BEGIN)
    i_use_end = published.index(USE_IT_END) + len(USE_IT_END)
    i_funnel = published.index(FUNNEL_BEGIN)
    head = published[:i_title]                      # frontmatter + Core AI blurb + devicemark block
    use_it = published[i_use_begin:i_use_end]
    tail = published[i_funnel:]                     # funnel block
    assert FIRST_SECTION_AFTER_USE_IT in published[i_use_end:i_funnel], "card layout changed; re-read it"
    return head + TITLE_AND_INTRO + use_it + "\n\n" + BODY + tail


def stage() -> None:
    card_src = Path(hf_hub_download(REPO, "README.md")).read_text()
    card = build_card(card_src)
    assert not re.search(r"__[A-Z0-9_]+__", card), "fill the measured numbers before staging"
    if STAGE.exists():
        shutil.rmtree(STAGE)
    dest = STAGE / "int8"
    dest.mkdir(parents=True)
    src_asset = next(BUNDLE.glob("*.aimodel"))
    shutil.copytree(src_asset, dest / f"{PUBLISHED_NAME}.aimodel")
    shutil.copytree(BUNDLE / "tokenizer", dest / "tokenizer")
    meta = json.loads((BUNDLE / "metadata.json").read_text())
    meta["name"] = PUBLISHED_NAME
    meta["assets"]["main"] = f"{PUBLISHED_NAME}.aimodel"
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    inner = dest / f"{PUBLISHED_NAME}.aimodel" / "metadata.json"
    if inner.exists():
        im = json.loads(inner.read_text())
        if "name" in im:
            im["name"] = PUBLISHED_NAME
            inner.write_text(json.dumps(im, indent=2) + "\n")
    (STAGE / "README.md").write_text(card)
    shutil.copy(hf_hub_download(REPO, "config.json"), STAGE / "config.json")
    shutil.copy(hf_hub_download(REPO, "LICENSE"), STAGE / "LICENSE")
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {STAGE}  ({total / 1e6:.0f} MB)")
    for f in sorted(STAGE.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(STAGE)}  {f.stat().st_size / 1e6:.1f} MB")


def upload() -> None:
    api = HfApi()
    info = api.upload_folder(repo_id=REPO, folder_path=str(STAGE), repo_type="model",
                             commit_message=COMMIT)
    print(f"uploaded -> {info.commit_url if hasattr(info, 'commit_url') else info}")
    print("new revision:", api.model_info(REPO).sha)


if __name__ == "__main__":
    stage()
    if "--stage-only" not in sys.argv:
        upload()
