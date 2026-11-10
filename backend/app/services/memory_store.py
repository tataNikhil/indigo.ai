"""In-memory chat history, one conversation per session ID."""

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.utils.token_counter import count_tokens


class SessionNotFoundError(KeyError):
    pass


@dataclass
class StoredMessage:
    role: str  # "user" or "assistant"
    content: str
    created_at: str
    extra: dict = field(default_factory=dict)  # analysis / reply metadata for the UI


@dataclass
class Session:
    session_id: str
    created_at: str
    last_active: float
    messages: list[StoredMessage] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def trim_for_llm(messages: list[StoredMessage], max_tokens: int) -> list[dict]:
    selected: list[dict] = []
    used = 0
    for msg in reversed(messages):
        cost = count_tokens(msg.content)
        if used + cost > max_tokens:
            break
        selected.append({"role": msg.role, "content": msg.content})
        used += cost
    selected.reverse()
    if selected and selected[0]["role"] == "assistant":
        selected.pop(0)
    return selected


class SessionStore:
    kind = "memory"

    def __init__(
        self,
        max_messages: int = 20,
        ttl_seconds: int = 3600,
        max_sessions: int = 500,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def _get(self, session_id: str) -> Session:
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        session.last_active = self._clock()
        self._sessions.move_to_end(session_id)  # mark as most recently used
        return session

    def create(self) -> Session:
        with self._lock:
            self._purge_expired()
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)  # drop least recently used
            session = Session(
                session_id=str(uuid.uuid4()),
                created_at=_now_iso(),
                last_active=self._clock(),
            )
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._get(session_id)
            return Session(
                session_id=session.session_id,
                created_at=session.created_at,
                last_active=session.last_active,
                messages=list(session.messages),
            )

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        assistant_extra: dict | None = None,
    ) -> None:
        with self._lock:
            session = self._get(session_id)
            session.messages.append(StoredMessage("user", user_text, _now_iso()))
            session.messages.append(
                StoredMessage("assistant", assistant_text, _now_iso(), assistant_extra or {})
            )
            overflow = len(session.messages) - self.max_messages
            if overflow > 0:
                drop = overflow + (overflow % 2)
                del session.messages[:drop]

    def history_for_llm(self, session_id: str, max_tokens: int) -> list[dict]:
        return trim_for_llm(self.get(session_id).messages, max_tokens)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._sessions)
