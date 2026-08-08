from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_chat_endpoint_generates_from_local_model() -> None:
    _require_smoke_artifacts()
    client = TestClient(app)
    token = _register(client, "chat@example.com")

    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
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
    assert body["conversation_id"]
    assert body["response"]
    assert body["new_token_ids"]


def test_chat_stream_endpoint_emits_sse_events() -> None:
    _require_smoke_artifacts()
    client = TestClient(app)
    token = _register(client, "stream@example.com")

    with client.stream(
        "POST",
        "/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
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


def test_chat_endpoint_reports_tool_results() -> None:
    _require_smoke_artifacts()
    client = TestClient(app)
    token = _register(client, "tools@example.com")

    response = client.post(
        "/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": "Calculate 12 / 3",
            "max_new_tokens": 4,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "use_tools": True,
        },
    )

    assert response.status_code == 200
    tools = response.json()["tool_results"]
    assert tools[0]["name"] == "calculator"
    assert tools[0]["metadata"]["value"] == 4


def _require_smoke_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / settings.model_tokenizer_path).exists()
    assert (repo_root / settings.model_checkpoint_path).exists()


def _register(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "password123"})
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    return response.json()["token"]
