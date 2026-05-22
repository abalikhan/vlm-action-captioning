import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_captioner():
    mock = MagicMock()
    mock.model = MagicMock()
    mock.caption_video.return_value = {
        "caption": "A person is running on a track.",
        "latency_ms": 1200.0,
        "num_frames": 8,
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    }
    return mock


@pytest.fixture
def client(mock_captioner):
    with patch("serving.app.main.ActionCaptioner", return_value=mock_captioner):
        from serving.app.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_caption_upload_invalid_format(client):
    response = client.post(
        "/caption/upload",
        files={"file": ("test.txt", b"not a video", "text/plain")},
    )
    assert response.status_code == 400


def test_caption_upload_valid(client, tmp_path):
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake video content")
    
    with open(fake_video, "rb") as f:
        response = client.post(
            "/caption/upload",
            files={"file": ("test.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert "latency_ms" in data