from typing import Annotated

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_token
from app.db.models import User, UserSession
from app.db.session import get_db


DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    session = db.scalar(
        select(UserSession).where(UserSession.token_hash == hash_token(token))
    )
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid session token")

    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid session user")
    return user

