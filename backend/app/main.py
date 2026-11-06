"""FastAPI entry point for the Enterprise Structured Ingestion Firewall."""

import logging

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from instructor.core import InstructorRetryException
from pydantic import ValidationError

from app.config import settings
from app.schemas.ingestion import (
    BatchIngestionRequest,
    BatchIngestionResult,
    IngestionRequest,
    IngestionResult,
)
from app.services.firewall import Classifier, TokenBudgetExceededError, ingest
from app.services.llm_service import LLMNotConfiguredError, classify_message

logger = logging.getLogger("ingestion_firewall")

app = FastAPI(
    title="Enterprise Structured Ingestion Firewall",
    description=(
        "Turns raw support / financial text into strictly typed JSON "
        "(intent, risk profile, priority, token metadata). "
        "Anything that fails validation is rejected before it reaches downstream systems."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_classifier() -> Classifier:
    """Dependency so tests can swap in a fake classifier."""
    return classify_message


def _run(request: IngestionRequest, classifier: Classifier) -> IngestionResult:
    try:
        return ingest(request, classifier)
    except TokenBudgetExceededError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc)) from exc
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
