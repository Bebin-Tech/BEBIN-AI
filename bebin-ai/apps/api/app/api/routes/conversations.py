from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.api.deps import DbSession, get_current_user
from app.db.models import Conversation, Message, User, now_utc
from app.schemas.conversation import ConversationCreateRequest, ConversationResponse, MessageResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    db: DbSession,
    user: User = Depends(get_current_user),
) -> list[ConversationResponse]:
    conversations = db.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    return [_conversation_response(conversation) for conversation in conversations]


@router.post("", response_model=ConversationResponse)
def create_conversation(
    request: ConversationCreateRequest,
    db: DbSession,
    user: User = Depends(get_current_user),
) -> ConversationResponse:
    conversation = Conversation(user_id=user.id, title=request.title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return _conversation_response(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    db: DbSession,
    user: User = Depends(get_current_user),
) -> ConversationResponse:
    conversation = _get_user_conversation(db, user, conversation_id)
    return _conversation_response(conversation)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    db: DbSession,
    user: User = Depends(get_current_user),
) -> None:
    conversation = _get_user_conversation(db, user, conversation_id)
    db.delete(conversation)
    db.commit()


def append_message(
    db,
    conversation: Conversation,
    role: str,
    content: str,
) -> Message:
    next_position = len(conversation.messages)
    message = Message(
        conversation_id=conversation.id,
        role=role,
        content=content,
        position=next_position,
    )
    conversation.updated_at = now_utc()
    db.add(message)
    db.flush()
    db.refresh(conversation)
    return message


def _get_user_conversation(db, user: User, conversation_id: str) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                position=message.position,
                created_at=message.created_at,
            )
            for message in conversation.messages
        ],
    )

