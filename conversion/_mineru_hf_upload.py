"""Upload the MinerU2.5-Pro Core AI bundles (vision fp16 + decoder int8lin S=1 + tokenizer) to HF."""
import os
import tempfile

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "mlboydaisuke/MinerU2.5-Pro-CoreAI"
STAGE = os.environ.get("MINERU_HF_STAGE") or os.path.join(tempfile.gettempdir(), "mineru_hf_stage")

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=STAGE, repo_id=REPO, repo_type="model",
    commit_message="MinerU2.5-Pro -> Core AI: vision (fp16) + decoder (int8lin S=1) + tokenizer")
print("uploaded", REPO)
