from fastapi.testclient import TestClient

from app.main import app


def test_register_login_and_conversation_persistence() -> None:
    client = TestClient(app)
    email = "phase9@example.com"
    token = _register_or_login(client, email)

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == email

    created = client.post(
        "/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "Persistent chat"},
    )
    assert created.status_code == 200
    conversation_id = created.json()["id"]

    listed = client.get("/conversations", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    assert any(item["id"] == conversation_id for item in listed.json())


def test_protected_routes_require_token() -> None:
    client = TestClient(app)

    response = client.get("/conversations")

    assert response.status_code == 401


def _register_or_login(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "password123"})
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    return response.json()["token"]

