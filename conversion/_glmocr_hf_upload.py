"""Upload the GLM-OCR Core AI bundles (vision fp16 + decoder int8hu + tokenizer) to HF."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "mlboydaisuke/GLM-OCR-CoreAI"
STAGE = "/tmp/glm_ocr_hf"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=STAGE, repo_id=REPO, repo_type="model",
    commit_message="GLM-OCR -> Core AI: vision (fp16) + decoder (int8hu S=1) + tokenizer")
print("uploaded", REPO)
