"""Application settings loaded from environment variables / backend/.env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "2000"))
    reply_max_tokens: int = int(os.getenv("REPLY_MAX_TOKENS", "700"))
    # Week 3: conversation memory
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    max_history_tokens: int = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))
    session_ttl_minutes: int = int(os.getenv("SESSION_TTL_MINUTES", "60"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "500"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    )


settings = Settings()
