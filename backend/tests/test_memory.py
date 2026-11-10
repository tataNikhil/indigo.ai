"""Week 3 tests: session IDs, chat history, follow-ups and isolation (fake LLMs)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_classifier, get_reply_generator, get_session_store
from app.services.memory_store import SessionNotFoundError, SessionStore
from app.services.response_service import GeneratedReply, build_messages
from tests.test_firewall import make_outcome


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ---------- SessionStore unit tests ----------


def test_create_gives_unique_ids():
    store = SessionStore()
    ids = {store.create().session_id for _ in range(50)}
    assert len(ids) == 50
    assert all(uuid.UUID(i) for i in ids)


def test_turns_are_stored_in_order():
    store = SessionStore()
    sid = store.create().session_id
    store.append_turn(sid, "hi", "hello!")
    store.append_turn(sid, "refund?", "Sure.")
    roles = [(m.role, m.content) for m in store.get(sid).messages]
    assert roles == [
        ("user", "hi"),
        ("assistant", "hello!"),
        ("user", "refund?"),
        ("assistant", "Sure."),
    ]


def test_sessions_are_isolated():
    store = SessionStore()
    a, b = store.create().session_id, store.create().session_id
    store.append_turn(a, "my order is #A1", "noted")
    assert store.get(b).messages == []


def test_old_turns_are_trimmed_in_pairs():
    store = SessionStore(max_messages=4)
    sid = store.create().session_id
    for i in range(5):
        store.append_turn(sid, f"q{i}", f"a{i}")
    contents = [m.content for m in store.get(sid).messages]
    assert contents == ["q3", "a3", "q4", "a4"]


def test_history_for_llm_respects_token_budget():
    store = SessionStore()
    sid = store.create().session_id
    store.append_turn(sid, "old question " * 200, "old answer " * 200)
    store.append_turn(sid, "recent question", "recent answer")
    history = store.history_for_llm(sid, max_tokens=50)
    assert history == [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]


def test_history_never_starts_with_assistant():
    store = SessionStore()
    sid = store.create().session_id
    store.append_turn(sid, "a long question " * 100, "short")
    history = store.history_for_llm(sid, max_tokens=20)
    assert history == []


def test_sessions_expire_after_ttl():
    clock = FakeClock()
    store = SessionStore(ttl_seconds=60, clock=clock)
    sid = store.create().session_id
    clock.now += 30
    store.get(sid)  # activity resets the timer
    clock.now += 59
    store.get(sid)
    clock.now += 61
    with pytest.raises(SessionNotFoundError):
        store.get(sid)


def test_least_recently_used_session_is_evicted():
    store = SessionStore(max_sessions=2)
    first = store.create().session_id
    second = store.create().session_id
    store.get(first)  # first is now the most recently used
    store.create()
    assert store.delete(first) is True
    assert store.delete(second) is False


def test_get_returns_a_copy():
    store = SessionStore()
    sid = store.create().session_id
    store.get(sid).messages.append("junk")
    assert store.get(sid).messages == []


def test_build_messages_order():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    messages = build_messages("SYSTEM", history, "q2")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "q2"


# ---------- API tests ----------


@pytest.fixture
def calls():
    """Records what the fake LLMs were given, for every request."""
    return {"classifier": [], "reply": []}


@pytest.fixture
def client(calls):
    store = SessionStore()

    def fake_classifier(message, channel, earlier):
        calls["classifier"].append(list(earlier))
        return make_outcome()

    def fake_reply(message, analysis, history):
        calls["reply"].append(list(history))
        return GeneratedReply(f"Answer to: {message}", "fake-model", 10, 5, 3)

    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_classifier] = lambda: fake_classifier
    app.dependency_overrides[get_reply_generator] = lambda: fake_reply
    yield TestClient(app)
    app.dependency_overrides.clear()


def chat(client, message, session_id=None):
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/api/v1/chat", json=payload)


def test_first_message_creates_a_session(client):
    body = chat(client, "My refund is late").json()
    assert uuid.UUID(body["session_id"])
    assert body["reply_metadata"]["history_messages_used"] == 0


def test_follow_up_gets_previous_turns(client, calls):
    sid = chat(client, "Rs 499 was debited twice").json()["session_id"]
    body = chat(client, "How long will the refund take?", sid).json()

    assert body["session_id"] == sid
    assert body["reply_metadata"]["history_messages_used"] == 2
    assert calls["reply"][-1] == [
        {"role": "user", "content": "Rs 499 was debited twice"},
        {"role": "assistant", "content": "Answer to: Rs 499 was debited twice"},
    ]
    # the classifier sees earlier user messages so it can resolve the follow-up
    assert calls["classifier"][-1] == ["Rs 499 was debited twice"]


def test_two_sessions_do_not_share_history(client, calls):
    a = chat(client, "Order #A1 is late").json()["session_id"]
    b = chat(client, "I forgot my password").json()["session_id"]
    assert a != b

    chat(client, "any update?", b)
    assert calls["reply"][-1][0]["content"] == "I forgot my password"


def test_get_session_returns_history_with_analysis(client):
    sid = chat(client, "Where is my order?").json()["session_id"]
    body = client.get(f"/api/v1/sessions/{sid}").json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["analysis"]["intent"] == "payment_issue"
    assert body["messages"][0]["analysis"] is None


def test_new_chat_and_delete(client):
    created = client.post("/api/v1/sessions")
    assert created.status_code == 201
    sid = created.json()["session_id"]

    assert chat(client, "hello", sid).status_code == 200
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
    assert chat(client, "still there?", sid).status_code == 404


def test_unknown_or_invalid_session_id(client):
    assert chat(client, "hi", str(uuid.uuid4())).status_code == 404
    assert chat(client, "hi", "not-a-uuid").status_code == 422
    assert client.get("/api/v1/sessions/not-a-uuid").status_code == 422


def test_failed_reply_is_not_saved(client):
    sid = chat(client, "first").json()["session_id"]

    def broken(message, analysis, history):
        raise RuntimeError("provider down")

    app.dependency_overrides[get_reply_generator] = lambda: broken
    assert chat(client, "second", sid).status_code == 502

    contents = [m["content"] for m in client.get(f"/api/v1/sessions/{sid}").json()["messages"]]
    assert "second" not in contents
    assert len(contents) == 2


def test_failed_first_message_does_not_create_a_session(client):
    store = app.dependency_overrides[get_session_store]()

    def broken(message, analysis, history):
        raise RuntimeError("provider down")

    app.dependency_overrides[get_reply_generator] = lambda: broken
    assert chat(client, "hello").status_code == 502
    assert len(store) == 0


def test_health_reports_active_sessions(client):
    chat(client, "one")
    chat(client, "two")
    assert client.get("/health").json()["active_sessions"] == 2
