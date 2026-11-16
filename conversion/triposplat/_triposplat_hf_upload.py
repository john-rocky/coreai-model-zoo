"""Upload the TripoSplat Core AI fp16 bundles + README to HF. Needs HF auth.

Stages the 4 fp16 .aimodel bundles (the on-device .aimodelc are too big / device-only — ship the
portable fp16 .aimodel) + the decode/octree nets + the recipe README, then uploads.
Run after `huggingface-cli login`.
"""
import os, shutil
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
from huggingface_hub import HfApi

REPO = "mlboydaisuke/TripoSplat-CoreAI"
# fp16 .aimodel bundles produced by _conv_fp16.py / _conv_decoder.py / _conv_octree.py
SRC = Path.home() / "TripoSplatRuntime" / "coreai_out"
HERE = Path(__file__).parent
STAGE = Path("/tmp/triposplat_hf")

BUNDLES = ["dinov3_fp16.aimodel", "vae_fp16.aimodel", "dit_fp16.aimodel",
           "gs_fp16.aimodel", "octree_fp32.aimodel", "decode_fp32.aimodel"]

shutil.rmtree(STAGE, ignore_errors=True)
STAGE.mkdir(parents=True)
for b in BUNDLES:
    src = SRC / b
    if src.is_dir():
        shutil.copytree(src, STAGE / b)
    else:
        print(f"WARN missing {src} (run the convert scripts first)")
shutil.copy(HERE / "README.md", STAGE / "README.md")

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                  commit_message="TripoSplat -> Core AI: image->3D Gaussian splats; fp16 5-net pipeline + recipe")
print("uploaded", REPO)
