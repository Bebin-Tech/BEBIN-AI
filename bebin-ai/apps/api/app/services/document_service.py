from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, DocumentChunk, User
from rag.chunking import chunk_text
from rag.pdf import extract_pdf_text
from rag.retriever import FaissRetriever, RetrievedChunk


SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/plain", "text/markdown"}


class DocumentServiceError(RuntimeError):
    pass


def ingest_document(db: Session, user: User, upload: UploadFile) -> Document:
    if upload.content_type not in SUPPORTED_CONTENT_TYPES:
        raise DocumentServiceError(f"Unsupported document type: {upload.content_type}")

    upload_dir = _user_upload_dir(user)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename or "document").name
    storage_path = upload_dir / f"{uuid4()}-{safe_name}"
    storage_path.write_bytes(upload.file.read())

    text = _extract_text(storage_path, upload.content_type)
    chunks = chunk_text(text)
    if not chunks:
        raise DocumentServiceError("Document contains no extractable text")

    document = Document(
        user_id=user.id,
        filename=safe_name,
        content_type=upload.content_type or "application/octet-stream",
        storage_path=str(storage_path),
    )
    db.add(document)
    db.flush()

    for position, chunk in enumerate(chunks):
        db.add(
            DocumentChunk(
                document_id=document.id,
                user_id=user.id,
                position=position,
                text=chunk,
            )
        )

    db.commit()
    db.refresh(document)
    return document


def search_user_documents(db: Session, user: User, query: str, top_k: int = 4) -> list[tuple[RetrievedChunk, str]]:
    rows = db.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.user_id == user.id)
        .order_by(Document.created_at.desc(), DocumentChunk.position.asc())
    ).all()
    retriever = FaissRetriever()
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=0.0,
        )
        for chunk, _filename in rows
    ]
    retriever.add(retrieved_chunks)
    filenames = {chunk.id: filename for chunk, filename in rows}
    return [(chunk, filenames.get(chunk.chunk_id, "")) for chunk in retriever.search(query, top_k=top_k)]


def format_rag_context(results: list[tuple[RetrievedChunk, str]]) -> str:
    if not results:
        return ""

    sections = []
    for index, (chunk, filename) in enumerate(results, start=1):
        sections.append(f"[Document {index}: {filename}]\n{chunk.text}")
    return "\n\n".join(sections)


def _extract_text(path: Path, content_type: str | None) -> str:
    if content_type == "application/pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8-sig")


def _user_upload_dir(user: User) -> Path:
    base = settings.upload_dir
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[4] / base
    return base / user.id

