from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_chat_endpoint_generates_from_local_model() -> None:
    _require_smoke_artifacts()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "Bebin AI",
            "max_new_tokens": 4,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Bebin AI"
    assert body["response"]
    assert body["new_token_ids"]


def test_chat_stream_endpoint_emits_sse_events() -> None:
    _require_smoke_artifacts()
    client = TestClient(app)

    with client.stream(
        "POST",
        "/chat/stream",
        json={
            "message": "Bebin AI",
            "max_new_tokens": 4,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
        },
    ) as response:
        content = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: start" in content
    assert "event: token" in content
    assert "event: done" in content


def _require_smoke_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / settings.model_tokenizer_path).exists()
    assert (repo_root / settings.model_checkpoint_path).exists()

