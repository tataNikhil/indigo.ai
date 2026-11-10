"""Generates the customer-facing Markdown reply with a plain Groq chat call."""

import time
from dataclasses import dataclass
from functools import lru_cache

from groq import Groq

from app.config import settings
from app.prompts.templates import build_system_prompt
from app.schemas.ingestion import IngestionResult
from app.services.llm_service import LLMNotConfiguredError


@dataclass
class GeneratedReply:
    markdown: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class EmptyReplyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> Groq:
    if not settings.groq_api_key:
        raise LLMNotConfiguredError("GROQ_API_KEY is not set in backend/.env")
    return Groq(api_key=settings.groq_api_key)


def build_messages(system_prompt: str, history: list[dict], message: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": message},
    ]


def generate_reply(
    message: str, analysis: IngestionResult, history: list[dict] | None = None
) -> GeneratedReply:
    system_prompt = build_system_prompt(
        intent=analysis.intent,
        risk_level=analysis.risk.level,
        risk_flags=analysis.risk.flags,
        priority=analysis.priority.level,
        sla_hours=analysis.priority.sla_hours,
        requires_human=analysis.priority.requires_human,
        summary=analysis.summary,
    )

    started = time.perf_counter()
    completion = _client().chat.completions.create(
        model=settings.groq_model,
        temperature=0.3,
        max_tokens=settings.reply_max_tokens,
        messages=build_messages(system_prompt, history or [], message),
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise EmptyReplyError("The model returned an empty reply.")

    usage = completion.usage
    return GeneratedReply(
        markdown=text,
        model=completion.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        latency_ms=latency_ms,
    )
