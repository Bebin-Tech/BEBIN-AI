from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import re

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Resume, ResumeChunk, User
from rag.embeddings import HashingEmbeddingModel
from rag.pdf import extract_pdf_text
from rag.retriever import FaissRetriever, RetrievedChunk

try:
    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - keeps local tests usable before dependencies are installed.
    @dataclass(frozen=True)
    class LangChainDocument:
        page_content: str
        metadata: dict[str, object]

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents: list[LangChainDocument]) -> list[LangChainDocument]:
            chunks: list[LangChainDocument] = []
            for document in documents:
                text = document.page_content
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    content = text[start:end].strip()
                    if content:
                        chunks.append(LangChainDocument(page_content=content, metadata=document.metadata))
                    if end == len(text):
                        break
                    start = max(0, end - self.chunk_overlap)
            return chunks


PDF_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


class ResumeRagError(RuntimeError):
    pass


def ingest_resume_pdfs(db: Session, user: User, uploads: list[UploadFile]) -> list[Resume]:
    if not uploads:
        raise ResumeRagError("Upload at least one resume PDF")

    resumes: list[Resume] = []
    for upload in uploads:
        resumes.append(_ingest_one_resume(db, user, upload))

    db.commit()
    for resume in resumes:
        db.refresh(resume)
    return resumes


def list_user_resumes(db: Session, user: User) -> list[Resume]:
    return (
        db.query(Resume)
        .filter(Resume.user_id == user.id)
        .order_by(Resume.created_at.desc(), Resume.candidate_name.asc())
        .all()
    )


def search_resumes(
    db: Session,
    user: User,
    query: str,
    candidate_name: str | None = None,
    top_k: int = 5,
) -> list[tuple[RetrievedChunk, Resume]]:
    rows = _resume_chunk_rows(db, user, candidate_name)
    retriever = FaissRetriever(HashingEmbeddingModel())
    chunks = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.resume_id,
            text=_retrieval_text(resume, chunk.text),
            score=0.0,
        )
        for chunk, resume in rows
    ]
    retriever.add(chunks)
    resumes_by_id = {chunk.id: resume for chunk, resume in rows}
    return [
        (chunk, resumes_by_id[chunk.chunk_id])
        for chunk in retriever.search(query, top_k=top_k)
        if chunk.chunk_id in resumes_by_id
    ]


def answer_resume_question(
    db: Session,
    user: User,
    question: str,
    candidate_name: str | None = None,
    top_k: int = 5,
) -> tuple[str, list[tuple[RetrievedChunk, Resume]]]:
    results = search_resumes(db, user, question, candidate_name=candidate_name, top_k=top_k)
    if not results:
        target = f" for {candidate_name}" if candidate_name else ""
        return f"No matching resume evidence found{target}.", []

    grouped: dict[str, list[tuple[RetrievedChunk, Resume]]] = {}
    for chunk, resume in results:
        grouped.setdefault(resume.id, []).append((chunk, resume))

    sections = []
    for index, matches in enumerate(grouped.values(), start=1):
        _chunk, resume = matches[0]
        snippets = [_shorten(match.text) for match, _resume in matches[:2]]
        profile = _profile_line(resume)
        sections.append(
            f"{index}. {resume.candidate_name} ({resume.filename})\n"
            f"   {profile}\n"
            f"   Evidence: {' | '.join(snippets)}"
        )

    return "Found matching resume evidence:\n" + "\n".join(sections), results


def _ingest_one_resume(db: Session, user: User, upload: UploadFile) -> Resume:
    if upload.content_type not in PDF_CONTENT_TYPES and not (upload.filename or "").lower().endswith(".pdf"):
        raise ResumeRagError(f"Unsupported resume type for {upload.filename}: upload PDF files only")

    upload_dir = _resume_upload_dir(user)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename or "resume.pdf").name
    storage_path = upload_dir / f"{uuid4()}-{safe_name}"
    storage_path.write_bytes(upload.file.read())

    text = extract_pdf_text(storage_path)
    normalized = _normalize_text(text)
    if not normalized:
        raise ResumeRagError(f"Resume has no extractable text: {safe_name}")

    candidate_name = _extract_candidate_name(normalized, safe_name)
    resume = Resume(
        user_id=user.id,
        filename=safe_name,
        candidate_name=candidate_name,
        email=_extract_email(normalized),
        phone=_extract_phone(normalized),
        skills=_extract_skills(normalized),
        storage_path=str(storage_path),
    )
    db.add(resume)
    db.flush()

    document = LangChainDocument(
        page_content=normalized,
        metadata={"resume_id": resume.id, "candidate_name": candidate_name, "filename": safe_name},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents([document])
    if not chunks:
        raise ResumeRagError(f"Resume has no indexable chunks: {safe_name}")

    for position, chunk in enumerate(chunks):
        db.add(
            ResumeChunk(
                resume_id=resume.id,
                user_id=user.id,
                position=position,
                text=chunk.page_content,
            )
        )
    return resume


def _resume_chunk_rows(
    db: Session,
    user: User,
    candidate_name: str | None,
) -> list[tuple[ResumeChunk, Resume]]:
    statement = (
        select(ResumeChunk, Resume)
        .join(Resume, Resume.id == ResumeChunk.resume_id)
        .where(ResumeChunk.user_id == user.id)
        .order_by(Resume.created_at.desc(), ResumeChunk.position.asc())
    )
    rows = list(db.execute(statement).all())
    if not candidate_name:
        return rows

    needle = candidate_name.lower()
    filtered = [
        (chunk, resume)
        for chunk, resume in rows
        if needle in resume.candidate_name.lower() or needle in resume.filename.lower()
    ]
    return filtered if filtered else rows


def _retrieval_text(resume: Resume, chunk_text: str) -> str:
    metadata = " ".join(
        value
        for value in [resume.candidate_name, resume.filename, resume.email or "", resume.skills or ""]
        if value
    )
    return f"{metadata}\n{chunk_text}"


def _profile_line(resume: Resume) -> str:
    parts = []
    if resume.email:
        parts.append(f"Email: {resume.email}")
    if resume.phone:
        parts.append(f"Phone: {resume.phone}")
    if resume.skills:
        parts.append(f"Skills: {_shorten(resume.skills, limit=160)}")
    return "; ".join(parts) if parts else "Profile details extracted from resume text."


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_candidate_name(text: str, filename: str) -> str:
    for line in text.splitlines()[:8]:
        candidate = line.strip(" -:\t")
        if _looks_like_name(candidate):
            return candidate[:180]
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()[:180] or "Unknown Candidate"


def _looks_like_name(value: str) -> bool:
    if not 2 <= len(value) <= 80:
        return False
    if "@" in value or re.search(r"\d", value):
        return False
    lowered = value.lower()
    blocked = {"resume", "curriculum vitae", "cv", "profile", "summary"}
    return lowered not in blocked and len(value.split()) <= 5


def _extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None


def _extract_phone(text: str) -> str | None:
    match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None


def _extract_skills(text: str) -> str | None:
    match = re.search(r"(?im)^skills?\s*[:\-]\s*(.+)$", text)
    if match:
        return match.group(1).strip()[:1000]
    return None


def _shorten(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _resume_upload_dir(user: User) -> Path:
    base = settings.upload_dir
    if not base.is_absolute():
        base = Path(__file__).resolve().parents[4] / base
    return base / "resumes" / user.id
