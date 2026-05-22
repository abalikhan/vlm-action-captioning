"""Pre-download model weights into the Docker image at build time.

Uses snapshot_download so no model class (and its import-time deps) is needed.
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR = os.environ.get("HF_HOME", "/app/.cache/huggingface")

print(f"Downloading {MODEL_ID} to {CACHE_DIR}...")
snapshot_download(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    token=os.environ.get("HF_TOKEN") or None,
)
print("Model download complete.")
