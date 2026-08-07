from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bebin AI API"
    app_version: str = "0.1.0"
    app_env: str = Field(default="local", alias="APP_ENV")
    database_url: str = Field(default="sqlite:///./bebin_ai.db", alias="DATABASE_URL")
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    enable_docs: bool = True

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

