"""Chat persistence database helpers (ADR-058 / SPEC-041).

This module owns the Chat DB lifecycle:
- Session registry table (chat_session)
- Safe path validation (default under settings.data_dir)
- Purge helpers for LangGraph checkpoint tables (checkpoints/writes)

LangGraph checkpointer/storage primitives create their own tables; this module
creates DocMind-owned tables only.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from src.config.settings import DocMindSettings, settings
from src.persistence.checkpoint_identity import (
    checkpoint_thread_id,
    memory_namespace_prefix,
)
from src.persistence.path_utils import resolve_path_under_data_dir
from src.utils.time import now_ms

CHAT_SESSION_TABLE = "chat_session"

STORE_TABLE = "store"
STORE_VECTORS_TABLE = "store_vectors"
CHECKPOINT_IDENTITY_TABLES = ("checkpoints", "writes")
LEGACY_CHECKPOINT_IDENTITY_WHERE = """
    thread_id IS NULL
    OR length(thread_id) != 72
    OR substr(thread_id, 1, 8) != 'docmind:'
"""
INCOMPATIBLE_CHECKPOINT_DB_MESSAGE = (
    "Chat checkpoint tables are incompatible with DocMind v2. Stop DocMind, "
    "archive chat.db with its -wal and -shm sidecars, then start v2 with a fresh "
    "Chat DB."
)
LEGACY_CHECKPOINT_IDENTITY_MESSAGE = (
    "Legacy v1 checkpoint identities detected. Stop DocMind, archive chat.db with "
    "its -wal and -shm sidecars, then start v2 with a fresh Chat DB."
)


# Python 3.12 deprecated sqlite3's implicit date/datetime adapters. LangGraph's
# native TTL store writes datetime parameters, so register the same SQLite-sortable
# ISO representation explicitly instead of relying on the deprecated defaults.
sqlite3.register_adapter(date, lambda value: value.isoformat())
sqlite3.register_adapter(datetime, lambda value: value.isoformat(" "))


class LegacyCheckpointIdentityError(RuntimeError):
    """Raised when a v1 raw thread identifier prevents safe v2 startup."""


@dataclass(frozen=True, slots=True)
class ChatSession:
    """Session registry entry for a LangGraph thread."""

    thread_id: str
    title: str
    created_at_ms: int
    updated_at_ms: int
    deleted_at_ms: int | None = None
    last_checkpoint_id: str | None = None


def _validate_chat_db_path(path: Path, cfg: DocMindSettings = settings) -> Path:
    return resolve_path_under_data_dir(
        path=path,
        data_dir=cfg.data_dir,
        label="Chat DB path",
    )


def open_chat_db(path: Path, *, cfg: DocMindSettings = settings) -> sqlite3.Connection:
    """Open the Chat DB connection (WAL, busy_timeout, safe location)."""
    resolved = _validate_chat_db_path(path, cfg)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Native LangGraph SqliteStore owns explicit BEGIN/COMMIT boundaries and
    # therefore requires an autocommit connection.
    conn = sqlite3.connect(
        str(resolved),
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Avoid SQLITE_BUSY explosions under Streamlit reruns
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def ensure_session_registry(conn: sqlite3.Connection) -> None:
    """Create the chat_session registry table (SPEC-041)."""
    reject_legacy_checkpoint_identities(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_session (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            last_checkpoint_id TEXT,
            deleted_at_ms INTEGER
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_session_updated "
        "ON chat_session(updated_at_ms);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_session_deleted "
        "ON chat_session(deleted_at_ms);"
    )
    conn.commit()


def reject_legacy_checkpoint_identities(conn: sqlite3.Connection) -> None:
    """Fail before startup when checkpoint tables contain pre-v2 raw IDs.

    V2 intentionally has no ambiguous raw-ID migration. Operators must archive
    the SQLite database (including WAL/SHM sidecars) and start with a fresh file.
    """
    for table in CHECKPOINT_IDENTITY_TABLES:
        if not _table_exists(conn, table):
            continue
        try:
            incompatible = conn.execute(
                f"""
                SELECT 1
                FROM {table}
                WHERE {LEGACY_CHECKPOINT_IDENTITY_WHERE}
                LIMIT 1;
                """,  # noqa: S608 - static internal table allowlist
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise LegacyCheckpointIdentityError(
                INCOMPATIBLE_CHECKPOINT_DB_MESSAGE
            ) from exc
        if incompatible is not None:
            raise LegacyCheckpointIdentityError(LEGACY_CHECKPOINT_IDENTITY_MESSAGE)


def create_session(*, title: str, conn: sqlite3.Connection) -> ChatSession:
    """Create a new chat session and return its registry record."""
    thread_id = str(uuid.uuid4())
    now = now_ms()
    conn.execute(
        """
        INSERT INTO chat_session
            (
                thread_id,
                title,
                created_at_ms,
                updated_at_ms,
                last_checkpoint_id,
                deleted_at_ms
            )
        VALUES (?, ?, ?, ?, NULL, NULL);
        """,
        (thread_id, title, now, now),
    )
    conn.commit()
    return ChatSession(
        thread_id=thread_id,
        title=title,
        created_at_ms=now,
        updated_at_ms=now,
        deleted_at_ms=None,
        last_checkpoint_id=None,
    )


def list_sessions(
    conn: sqlite3.Connection,
    *,
    include_deleted: bool = False,
    limit: int = 200,
) -> list[ChatSession]:
    """Return chat sessions ordered by most recently updated."""
    cur = conn.execute(
        """
        SELECT
            thread_id,
            title,
            created_at_ms,
            updated_at_ms,
            last_checkpoint_id,
            deleted_at_ms
        FROM chat_session
        WHERE (? = 1 OR deleted_at_ms IS NULL)
        ORDER BY updated_at_ms DESC
        LIMIT ?;
        """,
        (1 if include_deleted else 0, int(limit)),
    )
    rows = cur.fetchall()
    return [
        ChatSession(
            thread_id=str(r["thread_id"]),
            title=str(r["title"]),
            created_at_ms=int(r["created_at_ms"]),
            updated_at_ms=int(r["updated_at_ms"]),
            last_checkpoint_id=(
                str(r["last_checkpoint_id"]) if r["last_checkpoint_id"] else None
            ),
            deleted_at_ms=int(r["deleted_at_ms"]) if r["deleted_at_ms"] else None,
        )
        for r in rows
    ]


def upsert_session(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    title: str,
    last_checkpoint_id: str | None = None,
) -> None:
    """Insert or update a session registry entry."""
    now = now_ms()
    conn.execute(
        """
        INSERT INTO chat_session
            (
                thread_id,
                title,
                created_at_ms,
                updated_at_ms,
                last_checkpoint_id,
                deleted_at_ms
            )
        VALUES (?, ?, ?, ?, ?, NULL)
        ON CONFLICT(thread_id) DO UPDATE SET
            title=excluded.title,
            updated_at_ms=excluded.updated_at_ms,
            last_checkpoint_id=COALESCE(
                excluded.last_checkpoint_id,
                chat_session.last_checkpoint_id
            ),
            deleted_at_ms=NULL;
        """,
        (str(thread_id), str(title), now, now, last_checkpoint_id),
    )
    conn.commit()


def rename_session(conn: sqlite3.Connection, *, thread_id: str, title: str) -> None:
    """Rename a session in the registry."""
    now = now_ms()
    conn.execute(
        """
        UPDATE chat_session
        SET title=?, updated_at_ms=?
        WHERE thread_id=?;
        """,
        (str(title), now, str(thread_id)),
    )
    conn.commit()


def touch_session(
    conn: sqlite3.Connection, *, thread_id: str, last_checkpoint_id: str | None = None
) -> None:
    """Update a session's updated_at timestamp (and optionally checkpoint id)."""
    now = now_ms()
    conn.execute(
        """
        UPDATE chat_session
        SET updated_at_ms=?, last_checkpoint_id=COALESCE(?, last_checkpoint_id)
        WHERE thread_id=?;
        """,
        (now, last_checkpoint_id, str(thread_id)),
    )
    conn.commit()


def soft_delete_session(conn: sqlite3.Connection, *, thread_id: str) -> None:
    """Soft delete a session registry entry."""
    now = now_ms()
    conn.execute(
        """
        UPDATE chat_session
        SET deleted_at_ms=?, updated_at_ms=?
        WHERE thread_id=?;
        """,
        (now, now, str(thread_id)),
    )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return row is not None


def purge_session(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    user_id: str,
) -> dict[str, int]:
    """Hard delete a session's persisted data (checkpoints + writes + memories)."""
    tid = str(thread_id)
    persistence_tid = checkpoint_thread_id(thread_id=tid, user_id=user_id)
    deleted = {"langgraph": 0, "memory_store": 0, "session": 0}
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE;")
        # LangGraph tables may not exist before their primitive's first setup.
        langgraph_deletes = (
            ("writes", "DELETE FROM writes WHERE thread_id=?;"),
            ("checkpoints", "DELETE FROM checkpoints WHERE thread_id=?;"),
        )
        for table, stmt in langgraph_deletes:
            if not _table_exists(conn, table):
                continue
            cur = conn.execute(stmt, (persistence_tid,))
            if cur.rowcount >= 0:
                deleted["langgraph"] += cur.rowcount

        # Native LangGraph store tables may not exist yet. Exact equality is
        # intentional: user-scope memories and sibling sessions stay.
        memory_prefix = memory_namespace_prefix(user_id=user_id, thread_id=tid)
        memory_deletes = (
            (
                STORE_VECTORS_TABLE,
                f"DELETE FROM {STORE_VECTORS_TABLE} WHERE prefix=?;",  # noqa: S608
            ),
            (
                STORE_TABLE,
                f"DELETE FROM {STORE_TABLE} WHERE prefix=?;",  # noqa: S608
            ),
        )
        for table, stmt in memory_deletes:
            if not _table_exists(conn, table):
                continue
            cur = conn.execute(stmt, (memory_prefix,))
            if cur.rowcount >= 0:
                deleted["memory_store"] += cur.rowcount

        cur = conn.execute(
            "DELETE FROM chat_session WHERE thread_id=?;",
            (tid,),
        )
        if cur.rowcount >= 0:
            deleted["session"] += cur.rowcount
    logger.debug("purge_session deleted counts: {}", deleted)
    return deleted
