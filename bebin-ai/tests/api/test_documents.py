from fastapi.testclient import TestClient

from app.main import app


def test_document_upload_search_and_chat_rag_context() -> None:
    client = TestClient(app)
    token = _register_or_login(client, "rag@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    upload = client.post(
        "/documents",
        headers=headers,
        files={
            "file": (
                "notes.txt",
                b"Bebin AI retrieval should find this uploaded document about vector search and local context.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    assert upload.json()["chunk_count"] == 1

    search = client.get("/documents/search?query=vector%20search", headers=headers)
    assert search.status_code == 200
    assert search.json()
    assert "vector search" in search.json()[0]["text"]

    chat = client.post(
        "/chat",
        headers=headers,
        json={
            "message": "What mentions vector search?",
            "max_new_tokens": 4,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "use_rag": True,
        },
    )
    assert chat.status_code == 200
    assert chat.json()["rag_context"]
    assert "vector search" in chat.json()["rag_context"]


def _register_or_login(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "password123"})
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    return response.json()["token"]

