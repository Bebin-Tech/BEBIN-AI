from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatServiceError, get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def create_chat_completion(request: ChatRequest) -> ChatResponse:
    try:
        result = get_chat_service().generate(request)
    except ChatServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


@router.post("/stream")
def stream_chat_completion(request: ChatRequest) -> StreamingResponse:
    def events() -> str:
        try:
            yield from get_chat_service().stream(request)
        except ChatServiceError as exc:
            yield f"event: error\ndata: {str(exc)}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {str(exc)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

