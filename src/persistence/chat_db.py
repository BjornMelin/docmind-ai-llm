"""Chat persistence database helpers (ADR-058 / SPEC-041).

This module owns the Chat DB lifecycle:
- Session registry table (chat_session)
- Safe path validation (default under settings.data_dir)
- Purge helpers for LangGraph checkpoint tables (checkpoints/writes)

LangGraph checkpointer/storage primitives create their own tables; this module
creates DocMind-owned tables only.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import DocMindSettings, settings
from src.persistence.memory_store import NAMESPACE_THREAD_INDEX
from src.persistence.path_utils import resolve_path_under_data_dir
from src.utils.time import now_ms

logger = logging.getLogger(__name__)

CHAT_SESSION_TABLE = "chat_session"

# DocMind store tables (memory store)
STORE_ITEMS_TABLE = "docmind_store_items"
STORE_VEC_TABLE = "docmind_store_vec"


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
    conn = sqlite3.connect(str(resolved), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Avoid SQLITE_BUSY explosions under Streamlit reruns
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def ensure_session_registry(conn: sqlite3.Connection) -> None:
    """Create the chat_session registry table (SPEC-041)."""
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


def purge_session(conn: sqlite3.Connection, *, thread_id: str) -> None:
    """Hard delete a session's persisted data (checkpoints + writes + memories)."""
    tid = str(thread_id)
    with conn:
        # LangGraph SqliteSaver tables (best-effort; may not exist yet)
        langgraph_deletes = (
            "DELETE FROM writes WHERE thread_id=?;",
            "DELETE FROM checkpoints WHERE thread_id=?;",
        )
        for stmt in langgraph_deletes:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(stmt, (tid,))

        # DocMind memory store tables (best-effort; may not exist yet).
        # Namespace layout is positional (SPEC-041):
        # `namespace=(user_id, session_type, thread_id, ...)` so `thread_id` maps to
        # `ns{NAMESPACE_THREAD_INDEX}` (e.g. ns2).
        try:
            if NAMESPACE_THREAD_INDEX == 2:
                conn.execute(
                    "DELETE FROM docmind_store_vec WHERE ns2=?;",
                    (tid,),
                )
                conn.execute(
                    "DELETE FROM docmind_store_items WHERE ns2=?;",
                    (tid,),
                )
            else:
                logger.debug(
                    "Skipping memory store cleanup; namespace layout differs "
                    "(NAMESPACE_THREAD_INDEX=%s)",
                    NAMESPACE_THREAD_INDEX,
                )
        except sqlite3.OperationalError:
            # Tables not created yet; fail gracefully.
            pass

        conn.execute(
            "DELETE FROM chat_session WHERE thread_id=?;",
            (tid,),
        )
