"""Upload the Kokoro-82M Core AI bundles + English voices to HF."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "mlboydaisuke/Kokoro-82M-CoreAI"
STAGE = "/tmp/kokoro_hf"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=STAGE, repo_id=REPO, repo_type="model",
                  commit_message="Kokoro-82M -> Core AI: predictor/prosody/vocoder bundles + 28 EN voices")
print("uploaded", REPO)
