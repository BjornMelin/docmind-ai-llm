"""Native LangGraph SQLite memory-store lifecycle and legacy migration."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

from langgraph.store.base import IndexConfig, TTLConfig
from langgraph.store.sqlite import SqliteStore
from loguru import logger

from src.config.settings import DocMindSettings, settings
from src.persistence.chat_db import open_chat_db
from src.persistence.checkpoint_identity import memory_id, memory_namespace

LEGACY_ITEMS_TABLE = "docmind_store_items"
LEGACY_VEC_TABLE = "docmind_store_vec"
_LEGACY_NAMESPACE_DELIMITER = "\x1f"
_MEMORY_INDEX_FIELD = "content"
_MEMORY_TTL_CONFIG: TTLConfig = {
    "refresh_on_read": False,
    "sweep_interval_minutes": 1,
}


def open_memory_store(
    path: Path,
    *,
    index: IndexConfig | None = None,
    cfg: DocMindSettings = settings,
) -> SqliteStore:
    """Open and initialize the canonical native LangGraph SQLite store."""
    conn = open_chat_db(path, cfg=cfg)
    try:
        # langgraph-checkpoint-sqlite 3.1 uses ``text_fields`` internally even
        # though the public BaseStore IndexConfig calls this key ``fields``.
        sqlite_index: dict[str, Any] | None = dict(index) if index else None
        if sqlite_index and "fields" in sqlite_index:
            sqlite_index["text_fields"] = sqlite_index.pop("fields")
        store = SqliteStore(
            conn,
            index=cast(Any, sqlite_index),
            ttl=_MEMORY_TTL_CONFIG,
        )
        store.setup()
        migrate_legacy_memory_store(store)
        store.start_ttl_sweeper()
        return store
    except BaseException:
        with contextlib.suppress(Exception):
            conn.close()
        raise


def close_memory_store(store: SqliteStore) -> None:
    """Stop native background work and close the store connection safely."""
    with contextlib.suppress(Exception):
        store.stop_ttl_sweeper(timeout=1.0)
    with contextlib.suppress(Exception):
        store.conn.commit()
    with contextlib.suppress(Exception):
        store.conn.close()


def _native_memory_vector_is_valid(
    store: SqliteStore,
    namespace: tuple[str, ...],
    key: str,
) -> bool:
    """Return whether the canonical content vector exists at the configured size."""
    if not store.index_config:
        return True
    dimensions = store.index_config.get("dims")
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
    ):
        return False
    try:
        row = store.conn.execute(
            """
            SELECT COUNT(*), MIN(vec_length(embedding)), MAX(vec_length(embedding))
            FROM store_vectors
            WHERE prefix=? AND key=? AND field_name=?;
            """,
            (".".join(namespace), key, _MEMORY_INDEX_FIELD),
        ).fetchone()
    except sqlite3.DatabaseError:
        return False
    try:
        return bool(
            row is not None
            and int(row[0]) == 1
            and int(row[1]) == dimensions
            and int(row[2]) == dimensions
        )
    except (TypeError, ValueError):
        return False


def _native_memory_expiry_ms(
    store: SqliteStore,
    namespace: tuple[str, ...],
    key: str,
) -> int | None:
    """Return one native record's absolute expiry without conflating TTL with None."""
    row = store.conn.execute(
        """
        SELECT
            expires_at,
            CAST(
                ROUND((julianday(expires_at) - 2440587.5) * 86400000.0)
                AS INTEGER
            )
        FROM store
        WHERE prefix=? AND key=?;
        """,
        (".".join(namespace), key),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Legacy memory migration native record disappeared; source tables retained"
        )
    if row[0] is None:
        return None
    if row[1] is None:
        raise RuntimeError(
            "Legacy memory migration native expiry is invalid; source tables retained"
        )
    return int(row[1])


def _reindex_native_memory_preserving_metadata(
    store: SqliteStore,
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
) -> None:
    """Repair one partial native copy without changing its value or TTL authority."""
    prefix = ".".join(namespace)
    metadata = store.conn.execute(
        """
        SELECT
            created_at,
            updated_at,
            expires_at,
            ttl_minutes,
            CASE
                WHEN expires_at IS NULL THEN NULL
                ELSE MAX(
                    0.000001,
                    (julianday(expires_at) - julianday('now')) * 1440.0
                )
            END
        FROM store
        WHERE prefix=? AND key=?;
        """,
        (prefix, key),
    ).fetchone()
    if metadata is None:
        raise RuntimeError(
            "Legacy memory migration native record disappeared; source tables retained"
        )

    remaining_ttl = float(metadata[4]) if metadata[4] is not None else None
    try:
        store.put(
            namespace,
            key,
            value,
            index=[_MEMORY_INDEX_FIELD],
            ttl=remaining_ttl,
        )
    finally:
        # SqliteStore commits its item upsert even when the following vector write
        # raises. Restore the authoritative native metadata durably on both paths
        # without committing an unrelated caller-owned transaction.
        if store.conn.in_transaction:
            raise RuntimeError(
                "Legacy memory migration metadata repair found an open transaction; "
                "source tables retained"
            )
        with store.conn:
            store.conn.execute("BEGIN IMMEDIATE;")
            store.conn.execute(
                """
                UPDATE store
                SET created_at=?, updated_at=?, expires_at=?, ttl_minutes=?
                WHERE prefix=? AND key=?;
                """,
                (*metadata[:4], prefix, key),
            )

    if not _native_memory_vector_is_valid(store, namespace, key):
        raise RuntimeError(
            "Legacy memory migration vector validation failed; source tables retained"
        )


def migrate_legacy_memory_store(store: SqliteStore) -> int:
    """Copy active custom-store rows into the native schema, then hard-cut.

    The migration is intentionally retryable. Native writes are deterministic
    upserts; legacy tables remain intact until every active row has been read
    back with its canonicalized value.

    Returns:
        Number of canonical active legacy records copied and validated.
    """
    conn = store.conn
    if not _table_exists(conn, LEGACY_ITEMS_TABLE):
        _drop_orphaned_legacy_vector_table(conn)
        return 0

    now_ms = int(time.time() * 1000)
    rows = conn.execute(
        f"""
        SELECT item_id, ns_key, key, value_json, expires_at_ms
        FROM {LEGACY_ITEMS_TABLE}
        WHERE expires_at_ms IS NULL OR expires_at_ms > ?
        ORDER BY item_id;
        """,  # noqa: S608 # table name is a module constant
        (now_ms,),
    ).fetchall()

    records: dict[
        tuple[tuple[str, ...], str],
        tuple[dict[str, Any], int | None],
    ] = {}
    for row in rows:
        namespace = _canonicalize_legacy_namespace(row["ns_key"])
        value = _canonicalize_legacy_value(row["value_json"])
        key = memory_id(value["content"], value["kind"])
        expires_at_ms = (
            int(row["expires_at_ms"]) if row["expires_at_ms"] is not None else None
        )
        identity = (namespace, key)
        existing = records.get(identity)
        records[identity] = (
            _resolve_legacy_collision(existing, value, expires_at_ms)
            if existing is not None
            else (value, expires_at_ms)
        )

    migrated: list[tuple[tuple[str, ...], str, dict[str, Any], bool]] = []
    for (namespace, key), (value, expires_at_ms) in records.items():
        existing = store.get(namespace, key, refresh_ttl=False)
        existing_value = (
            existing.value
            if existing is not None and isinstance(existing.value, dict)
            else None
        )
        if existing_value is not None and not (
            existing_value.get("origin") == "consolidation"
            and value.get("origin") == "explicit"
        ):
            # Existing native state is authoritative across interrupted/retried
            # migrations. In particular, never replace an explicit user write
            # with a legacy derived record or refresh its existing TTL.
            ttl_row = conn.execute(
                "SELECT ttl_minutes FROM store WHERE prefix=? AND key=?;",
                (".".join(namespace), key),
            ).fetchone()
            if not _native_memory_vector_is_valid(store, namespace, key):
                _reindex_native_memory_preserving_metadata(
                    store,
                    namespace,
                    key,
                    dict(existing_value),
                )
            migrated.append(
                (
                    namespace,
                    key,
                    dict(existing_value),
                    ttl_row is None or ttl_row[0] is None,
                )
            )
            continue
        if existing_value is not None:
            value, expires_at_ms = _resolve_legacy_collision(
                (
                    existing_value,
                    _native_memory_expiry_ms(store, namespace, key),
                ),
                value,
                expires_at_ms,
            )
        ttl_minutes = (
            max(0.0, (int(expires_at_ms) - int(time.time() * 1000)) / 60_000)
            if expires_at_ms is not None
            else None
        )
        if ttl_minutes is not None and ttl_minutes <= 0:
            continue
        store.put(
            namespace,
            key,
            value,
            index=[_MEMORY_INDEX_FIELD],
            ttl=ttl_minutes,
        )
        migrated.append((namespace, key, value, expires_at_ms is None))

    for namespace, key, expected_value, expected_without_ttl in migrated:
        copied = store.get(namespace, key, refresh_ttl=False)
        if copied is None or copied.value != expected_value:
            raise RuntimeError(
                "Legacy memory migration validation failed; source tables retained"
            )
        ttl_row = conn.execute(
            "SELECT ttl_minutes FROM store WHERE prefix=? AND key=?;",
            (".".join(namespace), key),
        ).fetchone()
        if ttl_row is None or (ttl_row[0] is None) != expected_without_ttl:
            raise RuntimeError(
                "Legacy memory migration TTL validation failed; source tables retained"
            )
        if not _native_memory_vector_is_valid(store, namespace, key):
            raise RuntimeError(
                "Legacy memory migration vector validation failed; "
                "source tables retained"
            )

    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE;")
        conn.execute(f"DROP TABLE IF EXISTS {LEGACY_VEC_TABLE};")
        conn.execute(f"DROP TABLE IF EXISTS {LEGACY_ITEMS_TABLE};")
    logger.info("Migrated {} legacy memory-store rows to LangGraph", len(migrated))
    return len(migrated)


def _canonicalize_legacy_namespace(raw_namespace: object) -> tuple[str, ...]:
    parts = tuple(str(raw_namespace).split(_LEGACY_NAMESPACE_DELIMITER))
    if len(parts) not in {2, 3} or parts[0] != "memories":
        raise ValueError("Legacy memory namespace is not recognized")
    return memory_namespace(
        user_id=parts[1],
        thread_id=parts[2] if len(parts) == 3 else None,
    )


def _canonicalize_legacy_value(raw_value: object) -> dict[str, Any]:
    """Decode one legacy value and make write ownership explicit."""
    try:
        value = json.loads(str(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Legacy memory value is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Legacy memory value must be a JSON object")
    canonical = dict(value)
    content = canonical.get("content")
    kind = canonical.get("kind")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Legacy memory content must be a non-empty string")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("Legacy memory kind must be a non-empty string")
    canonical["content"] = content.strip()
    canonical["kind"] = kind.strip()
    if canonical.get("origin") not in {"explicit", "consolidation"}:
        canonical["origin"] = (
            "consolidation" if canonical.get("source_checkpoint_id") else "explicit"
        )
    return canonical


def _resolve_legacy_collision(
    existing: tuple[dict[str, Any], int | None],
    incoming_value: dict[str, Any],
    incoming_expiry: int | None,
) -> tuple[dict[str, Any], int | None]:
    existing_value, existing_expiry = existing
    existing_is_explicit = existing_value.get("origin") == "explicit"
    incoming_is_explicit = incoming_value.get("origin") == "explicit"
    winner = (
        existing_value
        if existing_is_explicit and not incoming_is_explicit
        else incoming_value
    )
    merged = dict(winner)
    tags: list[str] = []
    for value in (existing_value, incoming_value):
        raw_tags = value.get("tags")
        if not isinstance(raw_tags, list):
            continue
        tags.extend(str(tag) for tag in raw_tags if tag is not None and str(tag))
    tags = list(dict.fromkeys(tags))
    if tags:
        merged["tags"] = tags
    expiry = (
        None
        if existing_expiry is None or incoming_expiry is None
        else max(existing_expiry, incoming_expiry)
    )
    return merged, expiry


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    ).fetchone()
    return row is not None


def _drop_orphaned_legacy_vector_table(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, LEGACY_VEC_TABLE):
        return
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE;")
        conn.execute(f"DROP TABLE IF EXISTS {LEGACY_VEC_TABLE};")
