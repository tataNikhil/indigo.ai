"""Schemas for the chat endpoints (Week 2 replies, Week 3 sessions)."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.ingestion import IngestionRequest, IngestionResult


class ChatRequest(IngestionRequest):
    # leave session_id empty to start a new chat
    session_id: UUID | None = None


class ReplyMetadata(BaseModel):
    model: str
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    history_messages_used: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    session_id: str
    reply_markdown: str = Field(min_length=1)
    analysis: IngestionResult
    reply_metadata: ReplyMetadata


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    analysis: IngestionResult | None = None
    reply_metadata: ReplyMetadata | None = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    messages: list[ChatMessage] = Field(default_factory=list)
