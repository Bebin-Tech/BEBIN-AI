from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8_000)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=50, ge=0)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, gt=0.0, le=5.0)
    stop_on_eos: bool = True
    seed: int | None = None


class ChatResponse(BaseModel):
    message: str
    response: str
    token_ids: list[int]
    new_token_ids: list[int]
    stopped_on_eos: bool

