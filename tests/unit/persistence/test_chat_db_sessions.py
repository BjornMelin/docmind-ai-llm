"""Unit tests for chat session persistence and exact native-store purge."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.embeddings import Embeddings

from src.agents.tools import memory as memory_tools
from src.agents.tools.memory import (
    MemoryCandidate,
    consolidate_and_apply_memory_candidates,
)
from src.config.settings import DocMindSettings
from src.persistence.chat_db import (
    LegacyCheckpointIdentityError,
    create_session,
    ensure_session_registry,
    list_sessions,
    purge_session,
    rename_session,
    soft_delete_session,
    touch_session,
)
from src.persistence.checkpoint_identity import (
    checkpoint_thread_id,
    memory_namespace,
)
from src.persistence.memory_store import close_memory_store, open_memory_store

pytestmark = pytest.mark.unit


class _Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if "apple" in str(text).lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def _settings(tmp_path: Path) -> DocMindSettings:
    return cast(DocMindSettings, SimpleNamespace(data_dir=tmp_path))


def _open_store(tmp_path: Path):  # type: ignore[no-untyped-def]
    return open_memory_store(
        Path("chat.db"),
        index={"dims": 2, "embed": _Embeddings(), "fields": ["content"]},
        cfg=_settings(tmp_path),
    )


def test_startup_rejects_raw_v1_checkpoint_identities(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "legacy-chat.db")
    try:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT NOT NULL);")
        conn.execute("INSERT INTO checkpoints (thread_id) VALUES ('raw-thread-id');")

        with pytest.raises(
            LegacyCheckpointIdentityError,
            match=r"archive chat\.db",
        ):
            ensure_session_registry(conn)

        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='chat_session';"
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_startup_accepts_v2_hashed_checkpoint_identity(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "v2-chat.db")
    try:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT NOT NULL);")
        conn.execute(
            "INSERT INTO checkpoints (thread_id) VALUES (?);",
            (checkpoint_thread_id(thread_id="thread", user_id="user"),),
        )

        ensure_session_registry(conn)

        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='chat_session';"
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def test_session_crud_and_exact_native_store_purge(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    conn = store.conn
    try:
        ensure_session_registry(conn)
        created = create_session(title="First", conn=conn)

        rename_session(conn, thread_id=created.thread_id, title="Renamed")
        assert list_sessions(conn)[0].title == "Renamed"
        touch_session(conn, thread_id=created.thread_id, last_checkpoint_id="c1")
        assert list_sessions(conn)[0].last_checkpoint_id == "c1"
        soft_delete_session(conn, thread_id=created.thread_id)
        assert list_sessions(conn) == []
        assert list_sessions(conn, include_deleted=True)[0].deleted_at_ms is not None

        conn.execute("CREATE TABLE IF NOT EXISTS writes (thread_id TEXT);")
        conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT);")
        persistence_thread = checkpoint_thread_id(
            thread_id=created.thread_id,
            user_id="local-user",
        )
        other_persistence_thread = checkpoint_thread_id(
            thread_id=created.thread_id,
            user_id="other-user",
        )
        conn.executemany(
            "INSERT INTO writes (thread_id) VALUES (?);",
            ((persistence_thread,), (other_persistence_thread,)),
        )

        session_ns = memory_namespace(user_id="local-user", thread_id=created.thread_id)
        other_user_ns = memory_namespace(
            user_id="other-user", thread_id=created.thread_id
        )
        sibling_ns = memory_namespace(user_id="local-user", thread_id="sibling")
        user_ns = memory_namespace(user_id="local-user")
        for namespace, key in (
            (session_ns, "delete"),
            (other_user_ns, "other-user"),
            (sibling_ns, "sibling"),
            (user_ns, "user-scope"),
        ):
            store.put(namespace, key, {"content": f"apple {key}"})

        deleted = purge_session(
            conn,
            thread_id=created.thread_id,
            user_id="local-user",
        )

        assert deleted["memory_store"] == 2  # item plus vector
        assert list_sessions(conn, include_deleted=True) == []
        assert conn.execute("SELECT thread_id FROM writes").fetchone()[0] == (
            other_persistence_thread
        )
        assert store.get(session_ns, "delete") is None
        assert store.get(other_user_ns, "other-user") is not None
        assert store.get(sibling_ns, "sibling") is not None
        assert store.get(user_ns, "user-scope") is not None
    finally:
        close_memory_store(store)


def test_purge_rolls_back_when_existing_langgraph_table_is_malformed(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    conn = store.conn
    try:
        ensure_session_registry(conn)
        created = create_session(title="Keep on failure", conn=conn)
        persistence_thread = checkpoint_thread_id(
            thread_id=created.thread_id,
            user_id="local-user",
        )
        conn.execute("CREATE TABLE writes (thread_id TEXT);")
        conn.execute("CREATE TABLE checkpoints (wrong_column TEXT);")
        conn.execute(
            "INSERT INTO writes (thread_id) VALUES (?);",
            (persistence_thread,),
        )

        with pytest.raises(sqlite3.OperationalError, match="thread_id"):
            purge_session(
                conn,
                thread_id=created.thread_id,
                user_id="local-user",
            )

        assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 1
        assert list_sessions(conn)[0].thread_id == created.thread_id
    finally:
        close_memory_store(store)


def test_real_tools_consolidation_and_purge_use_public_thread_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_tools, "log_jsonl", lambda _event: None)
    store = _open_store(tmp_path)
    public_thread_id = "persistence-public-thread"
    user_id = "persistence-tool-user"
    opaque_thread_id = checkpoint_thread_id(
        thread_id=public_thread_id,
        user_id=user_id,
    )
    runtime = SimpleNamespace(
        store=store,
        config={
            "configurable": {
                "thread_id": opaque_thread_id,
                "public_thread_id": public_thread_id,
                "user_id": user_id,
            }
        },
    )
    namespace = memory_namespace(user_id=user_id, thread_id=public_thread_id)
    try:
        ensure_session_registry(store.conn)
        remembered = json.loads(
            memory_tools.remember.func(  # type: ignore[attr-defined]
                "apple preference",
                kind="preference",
                state={
                    "deadline_ts": time.monotonic() + 60,
                    "memory_generations": (
                        memory_tools.capture_memory_namespace_generations(
                            user_id=user_id,
                            thread_id=public_thread_id,
                        )
                    ),
                },
                runtime=runtime,
            )
        )
        assert remembered["ok"] is True

        changed = consolidate_and_apply_memory_candidates(
            [
                MemoryCandidate(
                    content="banana fact",
                    kind="fact",
                    importance=0.8,
                    source_checkpoint_id="checkpoint-1",
                )
            ],
            store,
            namespace,
            deadline_ts=time.monotonic() + 60,
        )
        assert changed == 1

        recalled = json.loads(
            memory_tools.recall_memories.func(  # type: ignore[attr-defined]
                "apple",
                state={"deadline_ts": time.monotonic() + 60},
                runtime=runtime,
                limit=10,
            )
        )
        assert recalled["ok"] is True
        assert {memory["content"] for memory in recalled["memories"]} == {
            "apple preference",
            "banana fact",
        }
        assert (
            store.search(memory_namespace(user_id=user_id, thread_id=opaque_thread_id))
            == []
        )

        store.put(
            memory_namespace(user_id=user_id),
            "keep-user",
            {"content": "apple user memory"},
        )
        purge_session(
            store.conn,
            thread_id=public_thread_id,
            user_id=user_id,
        )

        assert store.search(namespace) == []
        assert store.get(memory_namespace(user_id=user_id), "keep-user") is not None
    finally:
        close_memory_store(store)
