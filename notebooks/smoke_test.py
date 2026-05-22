"""
Quick local test before wrapping in FastAPI.
Download any short video to test with, for example:
wget -O test_video.mp4 "https://www.pexels.com/download/video/854671"
or use any .mp4 you have locally.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.app.inference import ActionCaptioner

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    
    captioner = ActionCaptioner(device="auto")
    captioner.load()
    
    result = captioner.caption_video(
        video_path=video_path,
        prompt="Describe the action happening in this video in one sentence.",
        num_frames=8,
    )
    
    print("\nResult:")
    print(f"  Caption : {result['caption']}")
    print(f"  Latency : {result['latency_ms']}ms")
    print(f"  Frames  : {result['num_frames']}")

if __name__ == "__main__":
    main()