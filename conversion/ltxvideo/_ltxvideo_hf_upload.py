"""Upload the LTX-Video 2B Core AI bundles + README to HF. Needs HF auth.

Ships the half-size bundles (DiT fp16 + VAE fp16 + T5 bf16) — the ship recipe. The bundles are
resolution-specific for DiT/VAE; upload the demo-resolution set + the recipe README.
Run after `huggingface-cli login`.
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/LTX-Video-2B-CoreAI"
SRC = Path(__file__).parent.parent.parent.parent  # adjust to wherever coreai_out lives
SRC = Path(os.environ.get("LTXV_COREAI_OUT", str(SRC / "scratch" / "LTX-Video" / "coreai_out")))
HERE = Path(__file__).parent
STAGE = Path("/tmp/ltxvideo_hf")

# DiT/VAE fp16 are demo-resolution (512x768x49); T5 bf16 is resolution-independent.
BUNDLES = ["dit_fp16.aimodel", "vae_fp16.aimodel", "t5_bf16.aimodel"]

shutil.rmtree(STAGE, ignore_errors=True)
STAGE.mkdir(parents=True)
for b in BUNDLES:
    src = SRC / b
    if src.is_dir():
        shutil.copytree(src, STAGE / b)
    else:
        print(f"WARN missing {src} (run _conv_fp16.py first; set LTXV_COREAI_OUT)")
shutil.copy(HERE / "README.md", STAGE / "README.md")

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                  commit_message="LTX-Video 2B -> Core AI: text->video; DiT fp16 + VAE fp16 + T5 bf16 + recipe")
print("uploaded", REPO)
