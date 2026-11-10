"""Chat history stored in Postgres (Supabase in production)."""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from app.services.memory_store import (
    Session,
    SessionNotFoundError,
    StoredMessage,
    trim_for_llm,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
create table if not exists chat_sessions (
    id uuid primary key,
    created_at timestamptz not null default now(),
    last_active timestamptz not null default now()
);

create table if not exists chat_messages (
    id bigserial primary key,
    session_id uuid not null references chat_sessions(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    extra jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists chat_messages_session_idx on chat_messages (session_id, id);
create index if not exists chat_sessions_last_active_idx on chat_sessions (last_active);

alter table chat_sessions enable row level security;
alter table chat_messages enable row level security;
"""


class DatabaseUnavailableError(RuntimeError):
    pass


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


class PostgresSessionStore:
    kind = "postgres"

    def __init__(
        self,
        database_url: str,
        max_messages: int = 20,
        ttl_seconds: int = 3600,
        connect_timeout: int = 10,
    ):
        self.max_messages = max_messages - (max_messages % 2)
        self.ttl_seconds = ttl_seconds
        self._schema_ready = False
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=5,
            timeout=connect_timeout,
            kwargs={"prepare_threshold": None, "connect_timeout": connect_timeout},
            open=False,
        )
        self._pool.open(wait=False)

    @contextmanager
    def _connection(self):
        try:
            with self._pool.connection() as conn:
                conn.row_factory = dict_row
                if not self._schema_ready:
                    conn.execute(SCHEMA_SQL)
                    self._schema_ready = True
                yield conn
        except (PoolTimeout, psycopg.OperationalError) as exc:
            logger.error("Database unavailable: %s", exc)
            raise DatabaseUnavailableError("Database is unavailable") from exc

    def _touch(self, conn, session_id: str) -> dict:
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise SessionNotFoundError(session_id) from exc
        row = conn.execute(
            """
            update chat_sessions set last_active = now()
            where id = %s and last_active > now() - make_interval(secs => %s)
            returning id, created_at
            """,
            (sid, self.ttl_seconds),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return row

    def _purge_expired(self, conn) -> None:
        conn.execute(
            "delete from chat_sessions where last_active < now() - make_interval(secs => %s)",
            (self.ttl_seconds,),
        )

    def create(self) -> Session:
        with self._connection() as conn:
            self._purge_expired(conn)
            row = conn.execute(
                "insert into chat_sessions (id) values (%s) returning id, created_at",
                (uuid.uuid4(),),
            ).fetchone()
        return Session(
            session_id=str(row["id"]),
            created_at=_iso(row["created_at"]),
            last_active=0.0,
        )

    def get(self, session_id: str) -> Session:
        with self._connection() as conn:
            row = self._touch(conn, session_id)
            rows = conn.execute(
                """
                select role, content, extra, created_at from chat_messages
                where session_id = %s order by id
                """,
                (row["id"],),
            ).fetchall()
        return Session(
            session_id=str(row["id"]),
            created_at=_iso(row["created_at"]),
            last_active=0.0,
            messages=[
                StoredMessage(r["role"], r["content"], _iso(r["created_at"]), r["extra"] or {})
                for r in rows
            ],
        )

    def delete(self, session_id: str) -> bool:
        try:
            sid = uuid.UUID(session_id)
        except ValueError:
            return False
        with self._connection() as conn:
            deleted = conn.execute("delete from chat_sessions where id = %s", (sid,)).rowcount
        return deleted > 0

    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        assistant_extra: dict | None = None,
    ) -> None:
        with self._connection() as conn, conn.transaction():
            row = self._touch(conn, session_id)
            sid = row["id"]
            conn.execute(
                "insert into chat_messages (session_id, role, content) values (%s, 'user', %s)",
                (sid, user_text),
            )
            conn.execute(
                """
                insert into chat_messages (session_id, role, content, extra)
                values (%s, 'assistant', %s, %s)
                """,
                (sid, assistant_text, Jsonb(assistant_extra or {})),
            )
            conn.execute(
                """
                delete from chat_messages
                where session_id = %s and id not in (
                    select id from chat_messages where session_id = %s
                    order by id desc limit %s
                )
                """,
                (sid, sid, self.max_messages),
            )

    def history_for_llm(self, session_id: str, max_tokens: int) -> list[dict]:
        return trim_for_llm(self.get(session_id).messages, max_tokens)

    def ping(self) -> bool:
        try:
            with self._connection() as conn:
                conn.execute("select 1")
            return True
        except DatabaseUnavailableError:
            return False

    def close(self) -> None:
        self._pool.close()

    def __len__(self) -> int:
        with self._connection() as conn:
            row = conn.execute(
                """
                select count(*) as n from chat_sessions
                where last_active > now() - make_interval(secs => %s)
                """,
                (self.ttl_seconds,),
            ).fetchone()
        return row["n"]
