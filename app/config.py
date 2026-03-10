"""Configuration management for TurboBrain."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_agent_id: str = ""
    google_drive_api: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./turbobrain.db"

    @property
    def effective_database_url(self) -> str:
        """Return the database URL, converting postgres:// to async driver format."""
        url = self.database_url
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql://") and "+asyncpg" not in url:
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"

    # CORS origins (comma-separated; use "*" to allow all origins)
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
