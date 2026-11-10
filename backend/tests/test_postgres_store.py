# Postgres tests run only when TEST_DATABASE_URL points at a throwaway database.

import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier, get_reply_generator, get_session_store
from app.services.memory_store import SessionNotFoundError
from app.services.postgres_store import DatabaseUnavailableError, PostgresSessionStore
from app.services.response_service import GeneratedReply
from tests.test_firewall import make_outcome

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")


def fake_reply(message, analysis, history):
    return GeneratedReply(f"Answer to: {message} (saw {len(history)})", "fake-model", 10, 5, 3)


def api_client(store):
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_classifier] = lambda: (lambda m, c, e: make_outcome())
    app.dependency_overrides[get_reply_generator] = lambda: fake_reply
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def store():
    s = PostgresSessionStore(TEST_DATABASE_URL, max_messages=4)
    s.ping()
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("truncate chat_messages, chat_sessions")
    yield s
    s.close()


@needs_db
def test_turns_round_trip(store):
    sid = store.create().session_id
    store.append_turn(sid, "hi", "hello!", {"analysis": {"intent": "greeting"}})
    messages = store.get(sid).messages
    assert [m.content for m in messages] == ["hi", "hello!"]
    assert messages[1].extra == {"analysis": {"intent": "greeting"}}


@needs_db
def test_expired_session_is_not_found(store):
    sid = store.create().session_id
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute(
            "update chat_sessions set last_active = now() - interval '2 hours' where id = %s",
            (uuid.UUID(sid),),
        )
    with pytest.raises(SessionNotFoundError):
        store.get(sid)


@needs_db
def test_chat_survives_a_restart(store):
    sid = api_client(store).post("/api/v1/chat", json={"message": "Rs 499 debited twice"}).json()[
        "session_id"
    ]

    restarted = PostgresSessionStore(TEST_DATABASE_URL, max_messages=4)
    try:
        body = api_client(restarted).post(
            "/api/v1/chat", json={"message": "how long?", "session_id": sid}
        ).json()
        assert body["reply_metadata"]["history_messages_used"] == 2
    finally:
        restarted.close()


def test_database_down_returns_503():
    dead = PostgresSessionStore("postgresql://nobody:nothing@127.0.0.1:1/none", connect_timeout=1)
    try:
        assert dead.ping() is False
        with pytest.raises(DatabaseUnavailableError):
            dead.create()
        response = api_client(dead).post("/api/v1/chat", json={"message": "hello"})
        assert response.status_code == 503
        assert "127.0.0.1" not in response.text
    finally:
        dead.close()
