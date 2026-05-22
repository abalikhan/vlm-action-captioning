import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from serving.app.inference import ActionCaptioner

# global model instance, loaded once at startup
captioner: ActionCaptioner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load model into memory once
    global captioner
    captioner = ActionCaptioner(device="auto")
    captioner.load()
    yield
    # shutdown: nothing to clean up explicitly
    del captioner


app = FastAPI(
    title="VLM Action Captioning API",
    description="Video action understanding using Qwen2-VL-2B-Instruct",
    version="0.1.0",
    lifespan=lifespan,
)


class CaptionRequest(BaseModel):
    video_url: str = Field(
        ...,
        description="URL of the video to caption",
        examples=["https://example.com/video.mp4"],
    )
    prompt: str = Field(
        default="Describe the action happening in this video in one sentence.",
        description="Instruction prompt for the model",
    )
    num_frames: int = Field(
        default=8,
        ge=4,
        le=32,
        description="Number of frames to sample from the video",
    )


class CaptionResponse(BaseModel):
    caption: str
    latency_ms: float
    num_frames: int
    model_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


START_TIME = time.time()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=captioner is not None and captioner.model is not None,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/caption/url", response_model=CaptionResponse)
async def caption_from_url(request: CaptionRequest):
    """Caption a video given a public URL."""
    import urllib.request

    if captioner is None or captioner.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            urllib.request.urlretrieve(request.video_url, tmp_path)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download video: {str(e)}"
        )

    try:
        result = captioner.caption_video(
            video_path=tmp_path,
            prompt=request.prompt,
            num_frames=request.num_frames,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return CaptionResponse(**result)


@app.post("/caption/upload", response_model=CaptionResponse)
async def caption_from_upload(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="Describe the action happening in this video in one sentence."
    ),
    num_frames: int = Form(default=8),
):
    """Caption a video uploaded directly as a file."""
    if captioner is None or captioner.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use mp4, avi, mov, or mkv.",
        )

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename).suffix, delete=False
    ) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        result = captioner.caption_video(
            video_path=tmp_path,
            prompt=prompt,
            num_frames=num_frames,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return CaptionResponse(**result)
