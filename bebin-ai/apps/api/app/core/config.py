from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bebin AI API"
    app_version: str = "0.1.0"
    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    database_url: str = Field(default="sqlite:///./bebin_ai.db", alias="DATABASE_URL")
    model_tokenizer_path: Path = Field(
        default=Path("artifacts/tokenizer/tokenizer.json"),
        alias="MODEL_TOKENIZER_PATH",
    )
    model_checkpoint_path: Path = Field(
        default=Path("artifacts/runs/smoke/last.pt"),
        alias="MODEL_CHECKPOINT_PATH",
    )
    upload_dir: Path = Field(default=Path("uploads"), alias="UPLOAD_DIR")
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    enable_docs: bool = True
    enable_request_logging: bool = Field(default=True, alias="ENABLE_REQUEST_LOGGING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
