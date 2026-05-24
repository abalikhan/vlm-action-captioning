import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# define the container image Modal will build and run
# this mirrors your Dockerfile.serve but uses Modal's image builder
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1",
    )
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "qwen-vl-utils>=0.0.8",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "pillow>=10.0.0",
        "decord>=0.6.0",
        "av>=12.0.0",
        "python-multipart>=0.0.12",
        "huggingface_hub>=0.24.0",
    )
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(
        local_path="serving",
        remote_path="/app/serving",
    )
)

# Modal volume to cache model weights across cold starts
# without this every cold start re-downloads 4.5GB
model_cache = modal.Volume.from_name(
    "vlm-action-captioning-weights",
    create_if_missing=True,
)

MODEL_CACHE_PATH = "/vol/model_weights"

app = modal.App(
    name="vlm-action-captioning",
    image=image,
)


@app.cls(
    gpu="T4",
    volumes={MODEL_CACHE_PATH: model_cache},
    scaledown_window=300,  # keep warm for 5 min after last request
    timeout=120,           # max 2 min per request
)
@modal.concurrent(max_inputs=1)
class ActionCaptioningService:

    @modal.enter()
    def load_model(self):
        """Runs once when the container starts. Loads model into GPU memory."""
        import os
        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

        from serving.app.inference import ActionCaptioner
        self.captioner = ActionCaptioner(device="auto")
        self.captioner.load()

    @modal.method()
    def caption_video_from_bytes(
        self,
        video_bytes: bytes,
        filename: str,
        prompt: str = "Describe the action happening in this video in one sentence.",
        num_frames: int = 8,
    ) -> dict:
        """Takes raw video bytes and returns a caption."""
        import tempfile
        from pathlib import Path

        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            result = self.captioner.caption_video(
                video_path=tmp_path,
                prompt=prompt,
                num_frames=num_frames,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result

    @modal.method()
    def caption_video_from_url(
        self,
        video_url: str,
        prompt: str = "Describe the action happening in this video in one sentence.",
        num_frames: int = 8,
    ) -> dict:
        """Downloads video from URL and returns a caption."""
        import urllib.request
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve(video_url, tmp_path)

        try:
            result = self.captioner.caption_video(
                video_path=tmp_path,
                prompt=prompt,
                num_frames=num_frames,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Public health check endpoint."""
        return {
            "status": "ok",
            "model_loaded": self.captioner is not None,
        }


web_app = FastAPI(
    title="VLM Action Captioning",
    description="Video action understanding via Qwen2-VL-2B-Instruct on Modal",
    version="0.1.0",
)

service = ActionCaptioningService()


class URLRequest(BaseModel):
    video_url: str
    prompt: str = "Describe the action happening in this video in one sentence."
    num_frames: int = 8


@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.post("/caption/upload")
async def caption_upload(
    file: UploadFile = File(...),
    prompt: str = Form(default="Describe the action happening in this video in one sentence."),
    num_frames: int = Form(default=8),
):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported format.")

    video_bytes = await file.read()
    result = await service.caption_video_from_bytes.remote.aio(
        video_bytes=video_bytes,
        filename=file.filename,
        prompt=prompt,
        num_frames=num_frames,
    )
    return result


@web_app.post("/caption/url")
async def caption_url(request: URLRequest):
    result = await service.caption_video_from_url.remote.aio(
        video_url=request.video_url,
        prompt=request.prompt,
        num_frames=request.num_frames,
    )
    return result


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("wandb-secret")],
    scaledown_window=300,
    timeout=120,
)
@modal.asgi_app(label="api")
def fastapi_app():
    return web_app


@app.function(
    gpu="T4",
    volumes={MODEL_CACHE_PATH: model_cache},
)
def download_model():
    """
    One-time function to pre-download model weights into the Modal volume.
    Run this once with: modal run serving/modal/app.py::download_model
    """
    import os
    from huggingface_hub import snapshot_download

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Downloading {MODEL_ID} to {MODEL_CACHE_PATH}...")
    snapshot_download(
        MODEL_ID,
        cache_dir=MODEL_CACHE_PATH,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("Done.")
    model_cache.commit()