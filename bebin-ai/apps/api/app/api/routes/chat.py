from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatServiceError, get_chat_service
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def create_chat_completion(
    request: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    try:
        result = get_chat_service().generate(request, db=db, user=user)
    except ChatServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


@router.post("/stream")
def stream_chat_completion(
    request: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    def events() -> str:
        try:
            yield from get_chat_service().stream(request, db=db, user=user)
        except ChatServiceError as exc:
            yield f"event: error\ndata: {str(exc)}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {str(exc)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
