from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import resume_rag_service


def test_bulk_resume_upload_search_and_chat(monkeypatch) -> None:
    client = TestClient(app)
    token = _register_or_login(client, "resume-rag@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    resume_text_by_filename = {
        "priya-sharma.pdf": """
Priya Sharma
Email: priya@example.com
Phone: +91 98765 43210
Skills: Python, LangChain, FAISS, FastAPI, RAG
Experience: Built resume retrieval chatbots and candidate search systems.
""",
        "arjun-mehta.pdf": """
Arjun Mehta
Email: arjun@example.com
Skills: React, TypeScript, CSS, Design Systems
Experience: Frontend engineer for dashboard and chat interfaces.
""",
    }

    def fake_extract_pdf_text(path: Path) -> str:
        for filename, text in resume_text_by_filename.items():
            if str(path).endswith(filename):
                return text
        return ""

    monkeypatch.setattr(resume_rag_service, "extract_pdf_text", fake_extract_pdf_text)

    upload = client.post(
        "/resumes/bulk",
        headers=headers,
        files=[
            ("files", ("priya-sharma.pdf", b"%PDF fake priya", "application/pdf")),
            ("files", ("arjun-mehta.pdf", b"%PDF fake arjun", "application/pdf")),
        ],
    )

    assert upload.status_code == 200
    uploaded = upload.json()
    assert len(uploaded) == 2
    assert {resume["candidate_name"] for resume in uploaded} == {"Priya Sharma", "Arjun Mehta"}

    search = client.get(
        "/resumes/search?query=LangChain%20FAISS&candidate_name=Priya&top_k=3",
        headers=headers,
    )

    assert search.status_code == 200
    results = search.json()
    assert results
    assert results[0]["candidate_name"] == "Priya Sharma"
    assert "LangChain" in results[0]["text"]

    chat = client.post(
        "/resumes/chat",
        headers=headers,
        json={"question": "Find candidate with LangChain and FAISS experience", "candidate_name": "Priya", "top_k": 3},
    )

    assert chat.status_code == 200
    body = chat.json()
    assert "Priya Sharma" in body["answer"]
    assert body["matches"]
    assert body["matches"][0]["candidate_name"] == "Priya Sharma"


def _register_or_login(client: TestClient, email: str) -> str:
    response = client.post("/auth/register", json={"email": email, "password": "password123"})
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert response.status_code == 200
    return response.json()["token"]
