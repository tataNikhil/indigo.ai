from typing import get_args

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier, get_reply_generator
from app.prompts.templates import INTENT_GUIDANCE
from app.schemas.ingestion import Intent
from app.services.response_service import GeneratedReply
from tests.test_firewall import make_outcome


def fake_reply(message, analysis, history=None):
    return GeneratedReply("**Sorry about the double debit.**", "fake-model", 300, 80, 12)


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
    assert body["reply_markdown"].startswith("**Sorry")
    assert body["analysis"]["intent"] == "payment_issue"


def test_reply_failure_returns_502(client):
    def broken(message, analysis, history):
        raise RuntimeError("timeout talking to provider")

    app.dependency_overrides[get_reply_generator] = lambda: broken
    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 502
    assert "timeout" not in response.text


def test_every_intent_has_guidance():
    assert set(get_args(Intent)) == set(INTENT_GUIDANCE)
