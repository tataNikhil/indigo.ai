"""Schemas for the Week 2 chat endpoint."""

from pydantic import BaseModel, Field

from app.schemas.ingestion import IngestionRequest, IngestionResult


class ChatRequest(IngestionRequest):
    """Same validation as the ingestion request (non-empty, max length, channel)."""


class ReplyMetadata(BaseModel):
    model: str
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)


class ChatResponse(BaseModel):
    reply_markdown: str = Field(min_length=1)
    analysis: IngestionResult
    reply_metadata: ReplyMetadata
