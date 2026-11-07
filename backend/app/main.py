"""FastAPI entry point.

Week 1: /api/v1/ingest      -> structured classification (the firewall)
Week 2: /api/v1/chat        -> classification + context-aware Markdown reply
"""

import logging
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from app.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, ReplyMetadata
from app.schemas.ingestion import (
    BatchIngestionRequest,
    BatchIngestionResult,
    IngestionRequest,
    IngestionResult,
)
from app.services.firewall import Classifier, TokenBudgetExceededError, ingest
from app.services.llm_service import LLMNotConfiguredError, classify_message
from app.services.response_service import GeneratedReply, generate_reply

logger = logging.getLogger("indigo")

ReplyGenerator = Callable[[str, IngestionResult], GeneratedReply]

app = FastAPI(
    title="Indigo.ai - Enterprise Structured Ingestion Firewall",
    description=(
        "Turns raw support / financial text into strictly typed JSON "
        "(intent, risk profile, priority, token metadata) and generates a "
        "context-aware Markdown reply based on that analysis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Dependencies (overridden in tests so no real API calls are made)
def get_classifier() -> Classifier:
    return classify_message


def get_reply_generator() -> ReplyGenerator:
    return generate_reply


def _run(request: IngestionRequest, classifier: Classifier) -> IngestionResult:
    try:
        return ingest(request, classifier)
    except TokenBudgetExceededError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except (InstructorRetryException, ValidationError) as exc:
        logger.warning("LLM output rejected by schema: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "The model's output failed schema validation and was rejected.",
        ) from exc
    except Exception as exc:  # network errors, rate limits, bad model name...
        logger.exception("LLM call failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "LLM provider request failed. Check the API key, model name and rate limits.",
        ) from exc


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.groq_model,
        "llm_configured": bool(settings.groq_api_key),
    }


@app.post("/api/v1/ingest", response_model=IngestionResult)
def ingest_one(
    request: IngestionRequest, classifier: Classifier = Depends(get_classifier)
) -> IngestionResult:
    return _run(request, classifier)


@app.post("/api/v1/ingest/batch", response_model=BatchIngestionResult)
def ingest_batch(
    batch: BatchIngestionRequest, classifier: Classifier = Depends(get_classifier)
) -> BatchIngestionResult:
    return BatchIngestionResult([_run(item, classifier) for item in batch.root])


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    classifier: Classifier = Depends(get_classifier),
    reply_generator: ReplyGenerator = Depends(get_reply_generator),
) -> ChatResponse:
    # Step 1: same firewall as Week 1 - detect intent, risk and priority.
    analysis = _run(request, classifier)

    # Step 2: use that analysis as context for the actual reply.
    try:
        reply = reply_generator(request.message, analysis)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Reply generation failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not generate a reply right now. Please try again.",
        ) from exc

    return ChatResponse(
        reply_markdown=reply.markdown,
        analysis=analysis,
        reply_metadata=ReplyMetadata(
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            latency_ms=reply.latency_ms,
        ),
    )
