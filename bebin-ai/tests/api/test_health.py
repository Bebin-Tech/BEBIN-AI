from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "Bebin AI API"


def test_request_id_header_is_returned() -> None:
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.headers["X-Request-ID"] == "test-request-id"


def test_liveness_check() -> None:
    client = TestClient(app)

    response = client.get("/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_check_reports_checks() -> None:
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code in {200, 503}
    payload = response.json()
    assert "database" in payload["checks"]
    assert "tokenizer" in payload["checks"]
    assert "checkpoint" in payload["checks"]
    assert "upload_dir" in payload["checks"]
