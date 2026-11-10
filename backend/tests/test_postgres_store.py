"""Week 4 tests: chat history in Postgres (Supabase in production).

Tests that need a real database run only when TEST_DATABASE_URL is set, e.g.
    TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/postgres pytest -q
Use a throwaway database - the tables are emptied between tests.

The "database is down" tests always run; they point at a port nothing listens on.
"""

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

DEAD_DATABASE_URL = "postgresql://nobody:nothing@127.0.0.1:1/none"


def fake_classifier(message, channel, earlier):
    return make_outcome()


def fake_reply(message, analysis, history):
    return GeneratedReply(f"Answer to: {message} (saw {len(history)})", "fake-model", 10, 5, 3)


@pytest.fixture
def store():
    s = PostgresSessionStore(TEST_DATABASE_URL, max_messages=4, ttl_seconds=3600)
    s.ping()  # creates the tables on first connection
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("truncate chat_messages, chat_sessions")
    yield s
    s.close()


def api_client(store):
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    app.dependency_overrides[get_reply_generator] = lambda: fake_reply
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


# ---------- store behaviour against a real Postgres ----------


@needs_db
def test_turns_round_trip_in_order(store):
    sid = store.create().session_id
    store.append_turn(sid, "hi", "hello!", {"analysis": {"intent": "greeting"}})
    store.append_turn(sid, "refund?", "Sure.")
    messages = store.get(sid).messages
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hi"),
        ("assistant", "hello!"),
        ("user", "refund?"),
        ("assistant", "Sure."),
    ]
    assert messages[1].extra == {"analysis": {"intent": "greeting"}}
    assert messages[0].extra == {}


@needs_db
def test_sessions_are_isolated(store):
    a, b = store.create().session_id, store.create().session_id
    store.append_turn(a, "order #A1", "noted")
    assert store.get(b).messages == []


@needs_db
def test_only_newest_messages_are_kept(store):
    sid = store.create().session_id
    for i in range(5):
        store.append_turn(sid, f"q{i}", f"a{i}")
    assert [m.content for m in store.get(sid).messages] == ["q3", "a3", "q4", "a4"]


@needs_db
def test_history_for_llm(store):
    sid = store.create().session_id
    store.append_turn(sid, "old " * 300, "old " * 300)
    store.append_turn(sid, "recent q", "recent a")
    assert store.history_for_llm(sid, max_tokens=50) == [
        {"role": "user", "content": "recent q"},
        {"role": "assistant", "content": "recent a"},
    ]


@needs_db
def test_expired_sessions_are_not_found(store):
    sid = store.create().session_id
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute(
            "update chat_sessions set last_active = now() - interval '2 hours' where id = %s",
            (uuid.UUID(sid),),
        )
    with pytest.raises(SessionNotFoundError):
        store.get(sid)
    assert len(store) == 0


@needs_db
def test_delete_removes_messages_too(store):
    sid = store.create().session_id
    store.append_turn(sid, "q", "a")
    assert store.delete(sid) is True
    assert store.delete(sid) is False
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        left = conn.execute("select count(*) from chat_messages").fetchone()[0]
    assert left == 0


@needs_db
def test_unknown_and_malformed_ids(store):
    with pytest.raises(SessionNotFoundError):
        store.get(str(uuid.uuid4()))
    with pytest.raises(SessionNotFoundError):
        store.get("not-a-uuid")
    assert store.delete("not-a-uuid") is False


@needs_db
def test_rls_is_enabled(store):
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        rows = conn.execute(
            "select relname, relrowsecurity from pg_class "
            "where relname in ('chat_sessions', 'chat_messages')"
        ).fetchall()
    assert dict(rows) == {"chat_sessions": True, "chat_messages": True}


# ---------- API + Postgres ----------


@needs_db
def test_chat_survives_a_server_restart(store):
    client = api_client(store)
    sid = client.post("/api/v1/chat", json={"message": "Rs 499 debited twice"}).json()[
        "session_id"
    ]

    # A brand-new store object = what the app sees after Render restarts it.
    restarted = PostgresSessionStore(TEST_DATABASE_URL, max_messages=4)
    try:
        client = api_client(restarted)
        body = client.post(
            "/api/v1/chat", json={"message": "how long?", "session_id": sid}
        ).json()
        assert body["reply_metadata"]["history_messages_used"] == 2
        assert body["reply_markdown"] == "Answer to: how long? (saw 2)"

        history = client.get(f"/api/v1/sessions/{sid}").json()["messages"]
        assert [m["content"] for m in history][0] == "Rs 499 debited twice"
        assert history[1]["analysis"]["intent"] == "payment_issue"
    finally:
        restarted.close()


@needs_db
def test_health_with_database(store):
    body = api_client(store).get("/health").json()
    assert body["storage"] == "postgres"
    assert body["database"] == "ok"
    assert body["status"] == "ok"


# ---------- database down (always runs) ----------


@pytest.fixture
def dead_store():
    s = PostgresSessionStore(DEAD_DATABASE_URL, connect_timeout=1)
    yield s
    s.close()


def test_store_reports_database_down(dead_store):
    assert dead_store.ping() is False
    with pytest.raises(DatabaseUnavailableError):
        dead_store.create()


def test_api_returns_503_without_leaking_details(dead_store):
    client = api_client(dead_store)

    response = client.post("/api/v1/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "nobody" not in response.text and "127.0.0.1" not in response.text

    response = client.post(
        "/api/v1/chat", json={"message": "hello", "session_id": str(uuid.uuid4())}
    )
    assert response.status_code == 503


def test_health_stays_up_when_database_is_down(dead_store):
    response = api_client(dead_store).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "unreachable"
