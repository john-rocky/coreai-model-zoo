"""Upload the MinerU2.5-Pro Core AI bundles (vision fp16 + decoder int8lin S=1 + tokenizer) to HF."""
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "mlboydaisuke/MinerU2.5-Pro-CoreAI"
STAGE = "/private/tmp/claude-501/-Users-majimadaisuke-code-coreai/3440de1a-a495-4a00-b8a4-4733849f64bd/scratchpad/mineru_hf"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=STAGE, repo_id=REPO, repo_type="model",
    commit_message="MinerU2.5-Pro -> Core AI: vision (fp16) + decoder (int8lin S=1) + tokenizer")
print("uploaded", REPO)
