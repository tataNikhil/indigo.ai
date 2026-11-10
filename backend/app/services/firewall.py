"""Ingestion pipeline: budget check -> LLM classification -> guard rules -> result."""

import time
from collections.abc import Callable

from app.config import settings
from app.schemas.ingestion import (
    RISK_ORDER,
    SLA_HOURS,
    IngestionRequest,
    IngestionResult,
    PriorityParameters,
    RiskProfile,
    TokenMetadata,
)
from app.services.injection_guard import detect_injection, detect_pii
from app.services.llm_service import ClassificationOutcome
from app.utils.token_counter import count_tokens

# (message, channel, earlier user messages in the same chat) -> outcome
Classifier = Callable[[str, str, list[str]], ClassificationOutcome]


class TokenBudgetExceededError(ValueError):
    def __init__(self, tokens: int, limit: int):
        super().__init__(f"Message is {tokens} tokens; the limit is {limit}.")
        self.tokens = tokens
        self.limit = limit


def _raise_risk(risk: RiskProfile, minimum: str, flag: str, note: str) -> RiskProfile:
    level = risk.level if RISK_ORDER[risk.level] >= RISK_ORDER[minimum] else minimum
    reason = risk.reason if flag in risk.flags else f"{risk.reason} {note}"[:300]
    return RiskProfile(level=level, flags=[*risk.flags, flag], reason=reason)


def ingest(
    request: IngestionRequest,
    classifier: Classifier,
    earlier_messages: list[str] | None = None,
) -> IngestionResult:
    input_tokens = count_tokens(request.message)
    if input_tokens > settings.max_input_tokens:
        raise TokenBudgetExceededError(input_tokens, settings.max_input_tokens)

    started = time.perf_counter()
    outcome = classifier(request.message, request.channel, earlier_messages or [])
    latency_ms = int((time.perf_counter() - started) * 1000)

    result = outcome.classification
    risk = result.risk
    priority = result.priority

    # Deterministic rules the LLM cannot talk its way out of.
    if detect_injection(request.message):
        risk = _raise_risk(risk, "medium", "prompt_injection", "Message attempts to steer the classifier.")
        priority = priority.model_copy(update={"requires_human": True})
    if detect_pii(request.message):
        risk = _raise_risk(risk, risk.level, "pii_present", "Sensitive identifiers detected.")

    return IngestionResult(
        intent=result.intent,
        confidence=result.confidence,
        risk=risk,
        priority=PriorityParameters(
            level=priority.level,
            requires_human=priority.requires_human,
            sla_hours=SLA_HOURS[priority.level],
        ),
        summary=result.summary,
        token_metadata=TokenMetadata(
            estimated_input_tokens=input_tokens,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            model=outcome.model,
            latency_ms=latency_ms,
        ),
    )
