"""Week 2 tests: intent detection -> reply generation, all with fake LLMs."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier, get_reply_generator
from app.prompts.templates import INTENT_GUIDANCE, build_system_prompt
from app.services.response_service import GeneratedReply
from tests.test_firewall import make_outcome

FAKE_MARKDOWN = "**Sorry about the double debit.**\n\n1. Keep your UTR handy\n2. Wait 5-7 days"


def fake_reply(message, analysis, history=None):
    return GeneratedReply(
        markdown=FAKE_MARKDOWN,
        model="fake-model",
        prompt_tokens=300,
        completion_tokens=80,
        latency_ms=12,
    )


@pytest.fixture
def client():
    app.dependency_overrides[get_classifier] = lambda: (lambda m, c, e: make_outcome())
    app.dependency_overrides[get_reply_generator] = lambda: fake_reply
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_returns_markdown_and_analysis(client):
    response = client.post("/api/v1/chat", json={"message": "Money debited twice"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply_markdown"] == FAKE_MARKDOWN
    assert body["analysis"]["intent"] == "payment_issue"
    assert body["reply_metadata"]["completion_tokens"] == 80


def test_reply_generator_receives_the_analysis():
    seen = {}

    def spy(message, analysis, history):
        seen["message"] = message
        seen["intent"] = analysis.intent
        return fake_reply(message, analysis)

    app.dependency_overrides[get_classifier] = lambda: (lambda m, c, e: make_outcome())
    app.dependency_overrides[get_reply_generator] = lambda: spy
    try:
        TestClient(app).post("/api/v1/chat", json={"message": "  refund pls  "})
    finally:
        app.dependency_overrides.clear()

    assert seen == {"message": "refund pls", "intent": "payment_issue"}


def test_chat_rejects_blank_message(client):
    assert client.post("/api/v1/chat", json={"message": "   "}).status_code == 422


def test_reply_failure_returns_502():
    def broken(message, analysis, history):
        raise RuntimeError("timeout talking to provider")

    app.dependency_overrides[get_classifier] = lambda: (lambda m, c, e: make_outcome())
    app.dependency_overrides[get_reply_generator] = lambda: broken
    try:
        response = TestClient(app).post("/api/v1/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "timeout" not in response.text


def test_classifier_failure_stops_before_reply():
    called = []

    def boom(m, c, e):
        raise RuntimeError("down")

    app.dependency_overrides[get_classifier] = lambda: boom
    app.dependency_overrides[get_reply_generator] = lambda: (lambda m, a, h: called.append(1))
    try:
        response = TestClient(app).post("/api/v1/chat", json={"message": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert called == []


# ---------- prompt templates ----------


def test_every_intent_has_guidance():
    from typing import get_args

    from app.schemas.ingestion import Intent

    assert set(get_args(Intent)) == set(INTENT_GUIDANCE)


def test_system_prompt_includes_detected_context():
    prompt = build_system_prompt(
        intent="fraud_report",
        risk_level="critical",
        risk_flags=["fraud_suspected", "financial_loss"],
        priority="urgent",
        sla_hours=1,
        requires_human=True,
        summary="Unknown UPI debit after a scam call.",
    )
    assert "Intent: fraud_report" in prompt
    assert "fraud_suspected, financial_loss" in prompt
    assert "1930" in prompt
    assert "human review needed: yes" in prompt
    assert "$" not in prompt  # every placeholder was filled
