from datetime import datetime

from pydantic import BaseModel


class ResumeResponse(BaseModel):
    id: str
    filename: str
    candidate_name: str
    email: str | None
    phone: str | None
    skills: str | None
    chunk_count: int
    created_at: datetime


class ResumeSearchResult(BaseModel):
    resume_id: str
    chunk_id: str
    candidate_name: str
    filename: str
    text: str
    score: float


class ResumeChatRequest(BaseModel):
    question: str
    candidate_name: str | None = None
    top_k: int = 5


class ResumeChatResponse(BaseModel):
    answer: str
    matches: list[ResumeSearchResult]
