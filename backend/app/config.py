"""Application settings loaded from environment variables / backend/.env.

Locally these come from backend/.env. On Render they are set in the
service's Environment tab - the .env file is never committed or deployed.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _origins(raw: str) -> tuple[str, ...]:
    # A trailing slash is a common copy-paste mistake ("https://x.vercel.app/")
    # and would never match the browser's Origin header, so strip it.
    return tuple(o.strip().rstrip("/") for o in raw.split(",") if o.strip())


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
    # Week 4: production
    # Empty -> chats are kept in memory (local dev). Set -> stored in Postgres/Supabase.
    database_url: str = os.getenv("DATABASE_URL", "").strip()
    # Comma-separated list of frontend URLs allowed to call this API.
    cors_origins: tuple[str, ...] = _origins(os.getenv("CORS_ORIGINS", "http://localhost:5173"))
    # Optional regex, e.g. to also allow Vercel preview deployments:
    # https://indigo-ai-.*\.vercel\.app
    cors_origin_regex: str | None = os.getenv("CORS_ORIGIN_REGEX") or None


settings = Settings()
