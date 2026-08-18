import logging
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"
    MAX_UPLOAD_SIZE_MB: int = 10
    DATABASE_URL: str = "sqlite+aiosqlite:///./resume_assessment.db"
    QDRANT_URL: str = ":memory:"
    QDRANT_API_KEY: str = ""
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_resume_assessment")
