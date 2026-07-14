"""Unit tests for native LangGraph SQLite memory-store lifecycle."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.sqlite import SqliteStore

from src.config.settings import DocMindSettings
from src.persistence.chat_db import open_chat_db
from src.persistence.checkpoint_identity import (
    memory_id,
    memory_namespace,
    memory_namespace_prefix,
)
from src.persistence.memory_store import (
    close_memory_store,
    migrate_legacy_memory_store,
    open_memory_store,
)

pytestmark = pytest.mark.unit


class _TwoDimensionalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _WrongDimensionEmbeddings(_TwoDimensionalEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0, 0.0]


def _settings(tmp_path: Path) -> DocMindSettings:
    return cast(DocMindSettings, SimpleNamespace(data_dir=tmp_path))


def _create_legacy_tables(tmp_path: Path) -> sqlite3.Connection:
    conn = open_chat_db(Path("chat.db"), cfg=_settings(tmp_path))
    conn.execute(
        """
        CREATE TABLE docmind_store_items (
            item_id INTEGER PRIMARY KEY,
            ns_key TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            expires_at_ms INTEGER
        );
        """
    )
    conn.execute("CREATE TABLE docmind_store_vec (item_id INTEGER PRIMARY KEY);")
    return conn


def test_native_put_get_search_delete_roundtrip(tmp_path: Path) -> None:
    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    try:
        namespace = ("memories", "u1", "t1")
        store.put(
            namespace,
            "k1",
            {"content": "hello", "kind": "fact"},
            index=False,
        )

        item = store.get(namespace, "k1")
        assert item is not None
        assert item.value["content"] == "hello"
        assert [
            result.key
            for result in store.search(("memories", "u1"), filter={"kind": "fact"})
        ] == ["k1"]
        assert namespace in store.list_namespaces(prefix=("memories", "u1"))

        store.delete(namespace, "k1")
        assert store.get(namespace, "k1") is None
    finally:
        close_memory_store(store)


def test_native_namespaces_isolate_user_and_sessions(tmp_path: Path) -> None:
    """Native prefix search cannot cross canonical scope boundaries."""
    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    user_namespace = memory_namespace(user_id="isolation-user")
    session_a = memory_namespace(
        user_id="isolation-user",
        thread_id="thread-a",
    )
    session_b = memory_namespace(
        user_id="isolation-user",
        thread_id="thread-b",
    )
    try:
        for namespace, key in (
            (user_namespace, "user"),
            (session_a, "a"),
            (session_b, "b"),
        ):
            store.put(
                namespace,
                key,
                {"content": key, "kind": "fact"},
                index=False,
            )

        assert [item.key for item in store.search(user_namespace)] == ["user"]
        assert [item.key for item in store.search(session_a)] == ["a"]
        assert [item.key for item in store.search(session_b)] == ["b"]
    finally:
        close_memory_store(store)


def test_memory_store_starts_and_stops_configured_ttl_sweeper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[SqliteStore] = []
    stopped: list[tuple[SqliteStore, float | None]] = []

    monkeypatch.setattr(
        SqliteStore,
        "start_ttl_sweeper",
        lambda self: started.append(self),
    )
    monkeypatch.setattr(
        SqliteStore,
        "stop_ttl_sweeper",
        lambda self, timeout=None: stopped.append((self, timeout)) or True,
    )

    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    assert store.ttl_config == {
        "refresh_on_read": False,
        "sweep_interval_minutes": 1,
    }
    assert started == [store]

    close_memory_store(store)
    assert stopped == [(store, 1.0)]


def test_expired_memory_is_removed_by_native_ttl_sweep(tmp_path: Path) -> None:
    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    store.stop_ttl_sweeper(timeout=1.0)
    namespace = memory_namespace(user_id="ttl-user", thread_id="ttl-thread")
    try:
        store.put(
            namespace,
            "expires",
            {"content": "temporary", "kind": "fact"},
            index=False,
            ttl=1,
        )
        with store.conn:
            store.conn.execute(
                "UPDATE store SET expires_at=datetime('now', '-1 minute') "
                "WHERE prefix=? AND key=?",
                (".".join(namespace), "expires"),
            )

        assert store.sweep_ttl() == 1
        assert store.get(namespace, "expires", refresh_ttl=False) is None
    finally:
        close_memory_store(store)


def test_legacy_rows_migrate_once_with_canonical_origin(tmp_path: Path) -> None:
    conn = _create_legacy_tables(tmp_path)
    now_ms = int(time.time() * 1000)
    rows = (
        (
            "memories\x1fu1",
            "user-global",
            json.dumps({"content": "global", "kind": "fact"}),
            None,
        ),
        (
            "memories\x1fu1\x1ft1",
            "explicit",
            json.dumps({"content": "keep me", "kind": "fact"}),
            now_ms + 600_000,
        ),
        (
            "memories\x1fu1\x1ft1",
            "consolidated",
            json.dumps(
                {
                    "content": "derived",
                    "kind": "fact",
                    "source_checkpoint_id": "checkpoint-1",
                }
            ),
            None,
        ),
        (
            "memories\x1fu1\x1ft1",
            "expired",
            json.dumps({"content": "expired", "kind": "fact"}),
            now_ms - 1,
        ),
    )
    conn.executemany(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, ?);
        """,
        rows,
    )
    conn.commit()
    conn.close()

    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    namespace = memory_namespace(user_id="u1", thread_id="t1")
    user_namespace = memory_namespace(user_id="u1")
    user_key = memory_id("global", "fact")
    explicit_key = memory_id("keep me", "fact")
    consolidated_key = memory_id("derived", "fact")
    try:
        explicit = store.get(namespace, explicit_key, refresh_ttl=False)
        consolidated = store.get(namespace, consolidated_key, refresh_ttl=False)
        assert explicit is not None
        assert explicit.value["origin"] == "explicit"
        ttl_minutes = store.conn.execute(
            "SELECT ttl_minutes FROM store WHERE prefix=? AND key=?;",
            (
                memory_namespace_prefix(user_id="u1", thread_id="t1"),
                explicit_key,
            ),
        ).fetchone()[0]
        assert ttl_minutes is not None
        assert float(ttl_minutes) > 0
        assert consolidated is not None
        assert consolidated.value["origin"] == "consolidation"
        assert store.get(user_namespace, user_key) is not None
        assert [item.key for item in store.search(user_namespace)] == [user_key]
        assert {item.key for item in store.search(namespace)} == {
            explicit_key,
            consolidated_key,
        }
        assert (
            store.get(namespace, memory_id("expired", "fact"), refresh_ttl=False)
            is None
        )
        legacy_tables = store.conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE name IN ('docmind_store_items', 'docmind_store_vec');
            """
        ).fetchall()
        assert legacy_tables == []
    finally:
        close_memory_store(store)

    reopened = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    try:
        assert reopened.get(namespace, explicit_key) is not None
        assert len(reopened.search(namespace)) == 2
        assert [item.key for item in reopened.search(user_namespace)] == [user_key]
    finally:
        close_memory_store(reopened)


def test_legacy_migration_canonicalizes_keys_and_resolves_collisions(
    tmp_path: Path,
) -> None:
    conn = _create_legacy_tables(tmp_path)
    now_ms = int(time.time() * 1000)
    conn.executemany(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, ?);
        """,
        (
            (
                "memories\x1fu%_\x1ft%_",
                "legacy-explicit-key",
                json.dumps(
                    {
                        "content": "Same fact",
                        "kind": "fact",
                        "origin": "explicit",
                        "tags": ["explicit"],
                    }
                ),
                now_ms + 60_000,
            ),
            (
                "memories\x1fu%_\x1ft%_",
                "legacy-consolidation-key",
                json.dumps(
                    {
                        "content": "same FACT",
                        "kind": "fact",
                        "origin": "consolidation",
                        "source_checkpoint_id": "checkpoint-1",
                        "tags": ["derived"],
                    }
                ),
                None,
            ),
        ),
    )
    conn.commit()
    conn.close()

    store = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    namespace = memory_namespace(user_id="u%_", thread_id="t%_")
    canonical_key = memory_id("same fact", "fact")
    try:
        items = store.search(namespace)
        assert [item.key for item in items] == [canonical_key]
        assert items[0].value == {
            "content": "Same fact",
            "kind": "fact",
            "origin": "explicit",
            "tags": ["explicit", "derived"],
        }
        ttl_minutes = store.conn.execute(
            "SELECT ttl_minutes FROM store WHERE prefix=? AND key=?;",
            (".".join(namespace), canonical_key),
        ).fetchone()[0]
        assert ttl_minutes is None
    finally:
        close_memory_store(store)


def test_legacy_migration_preserves_existing_native_explicit_record(
    tmp_path: Path,
) -> None:
    conn = _create_legacy_tables(tmp_path)
    namespace = memory_namespace(user_id="collision-user", thread_id="collision-thread")
    canonical_key = memory_id("same fact", "fact")
    conn.execute(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, NULL);
        """,
        (
            "memories\x1fcollision-user\x1fcollision-thread",
            "legacy-derived",
            json.dumps(
                {
                    "content": "same fact",
                    "kind": "fact",
                    "importance": 0.9,
                    "origin": "consolidation",
                    "source_checkpoint_id": "legacy-checkpoint",
                    "tags": ["legacy"],
                }
            ),
        ),
    )
    store = SqliteStore(conn)
    store.setup()
    native_value = {
        "content": "same fact",
        "kind": "fact",
        "importance": 0.4,
        "origin": "explicit",
        "tags": ["native"],
    }
    store.put(namespace, canonical_key, native_value, index=["content"])

    try:
        assert migrate_legacy_memory_store(store) == 1
        migrated = store.get(namespace, canonical_key, refresh_ttl=False)
        assert migrated is not None
        assert migrated.value == native_value
        assert not conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE name IN ('docmind_store_items', 'docmind_store_vec');
            """
        ).fetchall()
    finally:
        close_memory_store(store)


def test_legacy_explicit_collision_preserves_later_finite_native_expiry(
    tmp_path: Path,
) -> None:
    """An explicit winner cannot turn two finite collision records permanent."""
    conn = _create_legacy_tables(tmp_path)
    now_ms = int(time.time() * 1000)
    legacy_expiry_ms = now_ms + 600_000
    conn.execute(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, ?);
        """,
        (
            "memories\x1fcollision-user\x1fcollision-thread",
            "legacy-explicit",
            json.dumps(
                {
                    "content": "same finite fact",
                    "kind": "fact",
                    "origin": "explicit",
                }
            ),
            legacy_expiry_ms,
        ),
    )
    store = SqliteStore(conn)
    store.setup()
    namespace = memory_namespace(
        user_id="collision-user",
        thread_id="collision-thread",
    )
    canonical_key = memory_id("same finite fact", "fact")
    prefix = ".".join(namespace)
    store.put(
        namespace,
        canonical_key,
        {
            "content": "same finite fact",
            "kind": "fact",
            "origin": "consolidation",
            "source_checkpoint_id": "native-checkpoint",
        },
        index=False,
        ttl=20.0,
    )
    native_expiry_ms = int(
        conn.execute(
            """
            SELECT ROUND((julianday(expires_at) - 2440587.5) * 86400000.0)
            FROM store WHERE prefix=? AND key=?;
            """,
            (prefix, canonical_key),
        ).fetchone()[0]
    )
    assert native_expiry_ms > legacy_expiry_ms

    try:
        assert migrate_legacy_memory_store(store) == 1
        migrated = store.get(namespace, canonical_key, refresh_ttl=False)
        assert migrated is not None
        assert migrated.value["origin"] == "explicit"
        expiry_row = conn.execute(
            """
            SELECT
                ttl_minutes,
                ROUND((julianday(expires_at) - 2440587.5) * 86400000.0)
            FROM store WHERE prefix=? AND key=?;
            """,
            (prefix, canonical_key),
        ).fetchone()
        assert expiry_row[0] is not None
        assert abs(int(expiry_row[1]) - native_expiry_ms) <= 100
        assert not conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE name IN ('docmind_store_items', 'docmind_store_vec');
            """
        ).fetchall()
    finally:
        close_memory_store(store)


def test_invalid_legacy_value_retains_source_tables(tmp_path: Path) -> None:
    conn = _create_legacy_tables(tmp_path)
    conn.execute(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, NULL);
        """,
        ("memories\x1fu1\x1ft1", "broken", "not-json"),
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="not valid JSON"):
        open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))

    check = open_chat_db(Path("chat.db"), cfg=_settings(tmp_path))
    try:
        tables = {
            str(row[0])
            for row in check.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name IN ('docmind_store_items', 'docmind_store_vec');
                """
            ).fetchall()
        }
        assert tables == {"docmind_store_items", "docmind_store_vec"}
    finally:
        check.close()


def test_failed_copy_validation_retains_sources_and_retries_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _create_legacy_tables(tmp_path)
    conn.execute(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, NULL);
        """,
        (
            "memories\x1fu1\x1ft1",
            "memory",
            json.dumps({"content": "valid", "kind": "fact"}),
        ),
    )
    store = SqliteStore(conn)
    store.setup()
    monkeypatch.setattr(store, "get", lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(RuntimeError, match="validation failed"):
            migrate_legacy_memory_store(store)
        tables = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name IN ('docmind_store_items', 'docmind_store_vec');
                """
            ).fetchall()
        }
        assert tables == {"docmind_store_items", "docmind_store_vec"}
    finally:
        close_memory_store(store)

    retried = open_memory_store(Path("chat.db"), cfg=_settings(tmp_path))
    namespace = memory_namespace(user_id="u1", thread_id="t1")
    key = memory_id("valid", "fact")
    try:
        assert retried.get(namespace, key) is not None
        assert [item.key for item in retried.search(namespace)] == [key]
        assert not any(
            retried.conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name IN ('docmind_store_items', 'docmind_store_vec');
                """
            ).fetchall()
        )
    finally:
        close_memory_store(retried)


def test_legacy_migration_repairs_interrupted_vector_write_before_hard_cut(
    tmp_path: Path,
) -> None:
    """A partial native item cannot make retry drop its unindexed legacy source."""
    conn = _create_legacy_tables(tmp_path)
    expires_at_ms = int(time.time() * 1000) + 600_000
    conn.execute(
        """
        INSERT INTO docmind_store_items (ns_key, key, value_json, expires_at_ms)
        VALUES (?, ?, ?, ?);
        """,
        (
            "memories\x1fvector-user\x1fvector-thread",
            "legacy-key",
            json.dumps(
                {
                    "content": "apple fact",
                    "kind": "fact",
                    "origin": "explicit",
                }
            ),
            expires_at_ms,
        ),
    )
    conn.commit()
    store = SqliteStore(
        conn,
        index={
            "dims": 2,
            "embed": _TwoDimensionalEmbeddings(),
            "text_fields": ["content"],
        },
    )
    store.setup()
    namespace = memory_namespace(
        user_id="vector-user",
        thread_id="vector-thread",
    )
    key = memory_id("apple fact", "fact")
    prefix = ".".join(namespace)
    store.conn.execute(
        """
        CREATE TRIGGER block_legacy_vector
        BEFORE INSERT ON store_vectors
        BEGIN
            SELECT RAISE(ABORT, 'blocked vector write');
        END;
        """
    )

    try:
        with pytest.raises(sqlite3.IntegrityError, match="blocked vector write"):
            migrate_legacy_memory_store(store)

        native_metadata = tuple(
            store.conn.execute(
                """
                SELECT created_at, updated_at, expires_at, ttl_minutes
                FROM store WHERE prefix=? AND key=?;
                """,
                (prefix, key),
            ).fetchone()
        )
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM store_vectors WHERE prefix=? AND key=?;",
                (prefix, key),
            ).fetchone()[0]
            == 0
        )
        assert store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='docmind_store_items';"
        ).fetchone()

        # A retry that fails inside the vector statement must durably restore the
        # authoritative metadata before startup closes the failed connection.
        with pytest.raises(sqlite3.IntegrityError, match="blocked vector write"):
            migrate_legacy_memory_store(store)
        close_memory_store(store)
        conn = open_chat_db(Path("chat.db"), cfg=_settings(tmp_path))
        assert (
            tuple(
                conn.execute(
                    """
                SELECT created_at, updated_at, expires_at, ttl_minutes
                FROM store WHERE prefix=? AND key=?;
                """,
                    (prefix, key),
                ).fetchone()
            )
            == native_metadata
        )
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='docmind_store_items';"
        ).fetchone()
        conn.execute("DROP TRIGGER block_legacy_vector;")
        store = SqliteStore(
            conn,
            index={
                "dims": 2,
                "embed": _WrongDimensionEmbeddings(),
                "text_fields": ["content"],
            },
        )
        store.setup()
        with pytest.raises(RuntimeError, match="vector validation failed"):
            migrate_legacy_memory_store(store)
        assert (
            store.conn.execute(
                """
            SELECT vec_length(embedding) FROM store_vectors
            WHERE prefix=? AND key=? AND field_name='content';
            """,
                (prefix, key),
            ).fetchone()[0]
            == 3
        )
        assert store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='docmind_store_items';"
        ).fetchone()

        store.embeddings = _TwoDimensionalEmbeddings()
        assert migrate_legacy_memory_store(store) == 1
        assert (
            tuple(
                store.conn.execute(
                    """
                SELECT created_at, updated_at, expires_at, ttl_minutes
                FROM store WHERE prefix=? AND key=?;
                """,
                    (prefix, key),
                ).fetchone()
            )
            == native_metadata
        )
        assert (
            store.conn.execute(
                """
            SELECT vec_length(embedding) FROM store_vectors
            WHERE prefix=? AND key=? AND field_name='content';
            """,
                (prefix, key),
            ).fetchone()[0]
            == 2
        )
        assert [item.key for item in store.search(namespace, query="apple")] == [key]
        assert not store.conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE name IN ('docmind_store_items', 'docmind_store_vec');
            """
        ).fetchall()
    finally:
        close_memory_store(store)
