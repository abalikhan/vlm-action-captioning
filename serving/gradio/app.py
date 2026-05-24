import gradio as gr
import requests
import tempfile
import os
from pathlib import Path

MODAL_ENDPOINT = os.environ.get(
    "MODAL_ENDPOINT",
    "https://abalikhan--vlm-action-captioning-fastapi-app.modal.run",
)

EXAMPLE_PROMPTS = [
    "Describe the action happening in this video in one sentence.",
    "What sport or physical activity is being performed?",
    "Describe the setting and the main action in this video.",
    "What are the people in this video doing?",
]


def caption_video(video_path: str, prompt: str, num_frames: int) -> tuple:
    if video_path is None:
        return "Please upload a video first.", ""

    try:
        with open(video_path, "rb") as f:
            filename = Path(video_path).name
            response = requests.post(
                f"{MODAL_ENDPOINT}/caption/upload",
                files={"file": (filename, f, "video/mp4")},
                data={"prompt": prompt, "num_frames": int(num_frames)},
                timeout=300,
            )
        response.raise_for_status()
        result = response.json()

        caption = result["caption"]
        metadata = (
            f"Latency: {result['latency_ms']}ms   "
            f"Frames sampled: {result['num_frames']}   "
            f"Model: {result['model_id']}"
        )
        return caption, metadata

    except requests.exceptions.Timeout:
        return (
            "Request timed out. The model may be cold starting (30-60s). Please try again.",
            "",
        )
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}", ""
    except Exception as e:
        return f"Unexpected error: {str(e)}", ""


def check_endpoint_health() -> str:
    try:
        response = requests.get(f"{MODAL_ENDPOINT}/health", timeout=10)
        if response.status_code == 200:
            return "Endpoint status: online"
        return "Endpoint status: degraded"
    except Exception:
        return "Endpoint status: offline or cold (first request may take 30-60s)"


with gr.Blocks(
    title="VLM Action Captioning",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # Video Action Captioning
        ### Powered by Qwen2-VL-2B-Instruct, deployed on Modal serverless GPU

        Upload a short video clip and the model will describe the action in natural language.
        The first request may take 30-60 seconds due to cold start. Subsequent requests are fast.

        **Model:** [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
        **Inference:** Modal serverless T4 GPU
        **Code:** [GitHub](https://github.com/your-username/vlm-action-captioning)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Upload Video",
                sources=["upload"],
            )
            prompt_input = gr.Dropdown(
                choices=EXAMPLE_PROMPTS,
                value=EXAMPLE_PROMPTS[0],
                label="Prompt",
                allow_custom_value=True,
            )
            num_frames_input = gr.Slider(
                minimum=4,
                maximum=32,
                value=8,
                step=4,
                label="Number of frames to sample",
            )
            submit_btn = gr.Button(
                "Generate Caption",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Generated Caption",
                lines=4,
                show_copy_button=True,
            )
            metadata_output = gr.Textbox(
                label="Inference Metadata",
                lines=2,
                interactive=False,
            )
            health_output = gr.Textbox(
                label="Endpoint Status",
                lines=1,
                interactive=False,
            )
            check_health_btn = gr.Button(
                "Check Endpoint Status",
                variant=