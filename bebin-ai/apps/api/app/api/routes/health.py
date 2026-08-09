from pathlib import Path

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessCheck(BaseModel):
    ok: bool
    detail: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, ReadinessCheck]


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/live", response_model=HealthResponse)
def liveness_check() -> HealthResponse:
    return health_check()


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check(response: Response) -> ReadinessResponse:
    checks = {
        "database": _database_check(),
        "tokenizer": _path_check(settings.model_tokenizer_path),
        "checkpoint": _path_check(settings.model_checkpoint_path),
        "upload_dir": _upload_dir_check(settings.upload_dir),
    }
    ready = all(check.ok for check in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)


def _database_check() -> ReadinessCheck:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        return ReadinessCheck(ok=False, detail=str(exc))
    return ReadinessCheck(ok=True, detail="connected")


def _path_check(path: Path) -> ReadinessCheck:
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        return ReadinessCheck(ok=False, detail=f"missing: {resolved}")
    return ReadinessCheck(ok=True, detail=str(resolved))


def _upload_dir_check(path: Path) -> ReadinessCheck:
    resolved = _resolve_repo_path(path)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ReadinessCheck(ok=False, detail=str(exc))
    return ReadinessCheck(ok=True, detail=str(resolved))


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[5] / path
