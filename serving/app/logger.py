import os
import time
from typing import Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class InferenceLogger:
    """
    Logs inference requests to Weights and Biases.
    Gracefully no-ops if W&B is not configured.
    """

    def __init__(self):
        self.enabled = False
        self.run = None

    def setup(
        self,
        project: str = "vlm-action-captioning",
        job_type: str = "inference",
    ):
        if not WANDB_AVAILABLE:
            print("W&B not installed, logging disabled.")
            return

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            print("WANDB_API_KEY not set, logging disabled.")
            return

        try:
            self.run = wandb.init(
                project=project,
                entity="abalikhan-personal",
                job_type=job_type,
                resume="allow",
                settings=wandb.Settings(silent=True),
            )
            self.enabled = True
            print(f"W&B logging enabled: {self.run.url}")
        except Exception as e:
            print(f"W&B init failed, logging disabled: {e}")

    def log_inference(
        self,
        prompt: str,
        caption: str,
        latency_ms: float,
        num_frames: int,
        video_filename: Optional[str] = None,
        error: Optional[str] = None,
    ):
        if not self.enabled or self.run is None:
            return

        try:
            self.run.log(
                {
                    "latency_ms": latency_ms,
                    "num_frames": num_frames,
                    "prompt_length": len(prompt),
                    "caption_length": len(caption) if caption else 0,
                    "has_error": error is not None,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()
