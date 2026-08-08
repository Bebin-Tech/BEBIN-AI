from datetime import datetime

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    position: int
    created_at: datetime


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=160)


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []

