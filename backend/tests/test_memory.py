import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier, get_reply_generator, get_session_store
from app.services.memory_store import SessionNotFoundError, SessionStore
from app.services.response_service import GeneratedReply
from tests.test_firewall import make_outcome


def test_turns_are_stored_in_order():
    store = SessionStore()
    sid = store.create().session_id
    store.append_turn(sid, "hi", "hello!")
    store.append_turn(sid, "refund?", "Sure.")
    assert [m.content for m in store.get(sid).messages] == ["hi", "hello!", "refund?", "Sure."]


def test_old_turns_are_trimmed():
    store = SessionStore(max_messages=4)
    sid = store.create().session_id
    for i in range(5):
        store.append_turn(sid, f"q{i}", f"a{i}")
    assert [m.content for m in store.get(sid).messages] == ["q3", "a3", "q4", "a4"]


def test_sessions_expire():
    now = [1000.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: now[0])
    sid = store.create().session_id
    now[0] += 61
    with pytest.raises(SessionNotFoundError):
        store.get(sid)


@pytest.fixture
def client():
    store = SessionStore()
    seen = []

    def fake_reply(message, analysis, history):
        seen.append(history)
        return GeneratedReply(f"Answer to: {message}", "fake-model", 10, 5, 3)

    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_classifier] = lambda: (lambda m, c, e: make_outcome())
    app.dependency_overrides[get_reply_generator] = lambda: fake_reply
    client = TestClient(app)
    client.seen = seen
    yield client
    app.dependency_overrides.clear()


def chat(client, message, session_id=None):
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/api/v1/chat", json=payload)


def test_follow_up_gets_previous_turns(client):
    sid = chat(client, "Rs 499 was debited twice").json()["session_id"]
    body = chat(client, "How long will the refund take?", sid).json()

    assert body["session_id"] == sid
    assert body["reply_metadata"]["history_messages_used"] == 2
    assert client.seen[-1][0] == {"role": "user", "content": "Rs 499 was debited twice"}


def test_sessions_do_not_share_history(client):
    chat(client, "Order #A1 is late")
    b = chat(client, "I forgot my password").json()["session_id"]
    chat(client, "any update?", b)
    assert client.seen[-1][0]["content"] == "I forgot my password"


def test_delete_and_unknown_session(client):
    sid = chat(client, "hello").json()["session_id"]
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
    assert chat(client, "hi", str(uuid.uuid4())).status_code == 404


def test_failed_reply_is_not_saved(client):
    sid = chat(client, "first").json()["session_id"]

    def broken(message, analysis, history):
        raise RuntimeError("provider down")

    app.dependency_overrides[get_reply_generator] = lambda: broken
    assert chat(client, "second", sid).status_code == 502
    assert len(client.get(f"/api/v1/sessions/{sid}").json()["messages"]) == 2
