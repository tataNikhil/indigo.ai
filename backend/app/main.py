"""FastAPI entry point."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from app.config import settings
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ReplyMetadata,
    SessionInfo,
)
from app.schemas.ingestion import (
    BatchIngestionRequest,
    BatchIngestionResult,
    IngestionRequest,
    IngestionResult,
)
from app.services.firewall import Classifier, TokenBudgetExceededError, ingest
from app.services.llm_service import LLMNotConfiguredError, classify_message
from app.services.memory_store import Session, SessionNotFoundError, SessionStore
from app.services.postgres_store import DatabaseUnavailableError, PostgresSessionStore
from app.services.response_service import GeneratedReply, generate_reply

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indigo")

ReplyGenerator = Callable[[str, IngestionResult, list[dict]], GeneratedReply]

CLASSIFIER_CONTEXT_MESSAGES = 3

def build_session_store() -> SessionStore | PostgresSessionStore:
    if settings.database_url:
        logger.info("Chat history: Postgres")
        return PostgresSessionStore(
            settings.database_url,
            max_messages=settings.max_history_messages,
            ttl_seconds=settings.session_ttl_minutes * 60,
        )
    logger.info("Chat history: in memory (DATABASE_URL not set)")
    return SessionStore(
        max_messages=settings.max_history_messages,
        ttl_seconds=settings.session_ttl_minutes * 60,
        max_sessions=settings.max_sessions,
    )


session_store = build_session_store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Allowed CORS origins: %s", ", ".join(settings.cors_origins) or "none")
    yield
    session_store.close()


app = FastAPI(
    title="Indigo.ai - Enterprise Structured Ingestion Firewall",
    description=(
        "Turns raw support / financial text into strictly typed JSON "
        "(intent, risk profile, priority, token metadata), generates a "
        "context-aware Markdown reply, and remembers each conversation by session ID."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_origin_regex=settings.cors_origin_regex,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(DatabaseUnavailableError)
def database_unavailable(request: Request, exc: DatabaseUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Chat history is temporarily unavailable. Please try again shortly."},
    )


# overridden in tests
def get_classifier() -> Classifier:
    return classify_message


def get_reply_generator() -> ReplyGenerator:
    return generate_reply


def get_session_store() -> SessionStore:
    return session_store


def _run(
    request: IngestionRequest,
    classifier: Classifier,
    earlier_messages: list[str] | None = None,
) -> IngestionResult:
    try:
        return ingest(request, classifier, earlier_messages)
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


def _session_not_found() -> HTTPException:
    return HTTPException(
        status.HTTP_404_NOT_FOUND,
        "Session not found. It may have expired or the server restarted - start a new chat.",
    )


def _to_session_info(session: Session) -> SessionInfo:
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        messages=[
            ChatMessage(role=m.role, content=m.content, created_at=m.created_at, **m.extra)
            for m in session.messages
        ],
    )


@app.get("/health")
def health(store: SessionStore = Depends(get_session_store)) -> dict:
    database_ok = store.ping()
    return {
        "status": "ok" if database_ok else "degraded",
        "model": settings.groq_model,
        "llm_configured": bool(settings.groq_api_key),
        "storage": store.kind,
        "database": "ok" if database_ok else "unreachable",
        "active_sessions": len(store) if database_ok else None,
    }


# Week 1: ingestion firewall


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


# Week 3: sessions


@app.post("/api/v1/sessions", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
def create_session(store: SessionStore = Depends(get_session_store)) -> SessionInfo:
    return _to_session_info(store.create())


@app.get("/api/v1/sessions/{session_id}", response_model=SessionInfo)
def get_session(
    session_id: UUID, store: SessionStore = Depends(get_session_store)
) -> SessionInfo:
    try:
        return _to_session_info(store.get(str(session_id)))
    except SessionNotFoundError as exc:
        raise _session_not_found() from exc


@app.delete("/api/v1/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: UUID, store: SessionStore = Depends(get_session_store)):
    if not store.delete(str(session_id)):
        raise _session_not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Week 2 + 3: chat with memory


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    classifier: Classifier = Depends(get_classifier),
    reply_generator: ReplyGenerator = Depends(get_reply_generator),
    store: SessionStore = Depends(get_session_store),
) -> ChatResponse:
    session_id = str(request.session_id) if request.session_id else None
    history: list[dict] = []
    if session_id:
        try:
            history = store.history_for_llm(session_id, settings.max_history_tokens)
        except SessionNotFoundError as exc:
            raise _session_not_found() from exc

    earlier = [m["content"] for m in history if m["role"] == "user"]
    analysis = _run(request, classifier, earlier[-CLASSIFIER_CONTEXT_MESSAGES:])

    try:
        reply = reply_generator(request.message, analysis, history)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except Exception as exc:
        logger.exception("Reply generation failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not generate a reply right now. Please try again.",
        ) from exc

    metadata = ReplyMetadata(
        model=reply.model,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        latency_ms=reply.latency_ms,
        history_messages_used=len(history),
    )

    # save only after a successful reply
    if session_id is None:
        session_id = store.create().session_id
    try:
        store.append_turn(
            session_id,
            request.message,
            reply.markdown,
            {"analysis": analysis.model_dump(), "reply_metadata": metadata.model_dump()},
        )
    except SessionNotFoundError as exc:  # deleted while the LLM was answering
        raise _session_not_found() from exc

    return ChatResponse(
        session_id=session_id,
        reply_markdown=reply.markdown,
        analysis=analysis,
        reply_metadata=metadata,
    )
