from unittest.mock import MagicMock, patch
import pytest


def test_action_captioner_init():
    from serving.app.inference import ActionCaptioner
    captioner = ActionCaptioner(device="cpu")
    assert captioner.model is None
    assert captioner.processor is None


def test_caption_video_missing_file():
    from serving.app.inference import ActionCaptioner
    captioner = ActionCaptioner()
    captioner.model = MagicMock()
    captioner.processor = MagicMock()
    with pytest.raises(FileNotFoundError):
        captioner.caption_video("nonexistent_video.mp4")