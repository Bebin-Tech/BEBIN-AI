from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    chunk_count: int
    created_at: datetime


class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    score: float

