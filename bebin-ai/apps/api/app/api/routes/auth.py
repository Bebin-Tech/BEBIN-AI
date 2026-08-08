from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.security import create_session_token, hash_password, hash_token, verify_password
from app.db.models import User, UserSession
from app.db.session import get_db
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = request.email.lower()
    existing = db.scalar(select(User).where(User.email == email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email is already registered")

    user = User(email=email, password_hash=hash_password(request.password))
    db.add(user)
    db.flush()
    token = _create_session(db, user)
    db.commit()
    return _auth_response(token, user)


@router.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    email = request.email.lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_session(db, user)
    db.commit()
    return _auth_response(token, user)


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=user.id, email=user.email)


def _create_session(db: Session, user: User) -> str:
    token = create_session_token()
    db.add(UserSession(user_id=user.id, token_hash=hash_token(token)))
    return token


def _auth_response(token: str, user: User) -> AuthResponse:
    return AuthResponse(token=token, user=UserResponse(id=user.id, email=user.email))

