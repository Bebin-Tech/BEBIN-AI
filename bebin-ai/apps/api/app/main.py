from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api.routes.chat import router as chat_router
from app.api.routes.auth import router as auth_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.init_db import init_db


def create_app() -> FastAPI:
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.enable_request_logging:
        app.add_middleware(RequestContextMiddleware)

    init_db()
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(conversations_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    return app


app = create_app()
