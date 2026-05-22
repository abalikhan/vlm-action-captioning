import os
import time
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-2B-Instruct")

# these are the generation defaults we will tune later
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}


class ActionCaptioner:
    def __init__(self, device: str = "auto"):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading {MODEL_ID}...")
        start = time.time()

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        print(f"Model loaded in {time.time() - start:.1f}s")
        return self

    def caption_video(
        self,
        video_path: str,
        prompt: str = "Describe the action happening in this video in one sentence.",
        num_frames: int = 8,
    ) -> dict:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(path),
                        "max_pixels": 360 * 420,
                        "nframes": num_frames,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **GENERATION_CONFIG,
            )

        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]

        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        latency_ms = (time.time() - start) * 1000

        return {
            "caption": caption.strip(),
            "latency_ms": round(latency_ms, 1),
            "num_frames": num_frames,
            "model_id": MODEL_ID,
        }