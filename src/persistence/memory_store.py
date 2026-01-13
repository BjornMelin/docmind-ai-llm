"""SQLite-backed LangGraph BaseStore with optional semantic search (ADR-058 / SPEC-041).

This implements a persistent store compatible with LangGraph's InjectedStore
feature. It is designed for DocMind's offline-first Streamlit UX:

- Persistent across restarts (SQLite file under settings.data_dir by default)
- Namespaces support hierarchical prefixes, used for (user_id, thread_id) scoping
- Optional semantic search powered by sqlite-vec (vec0 virtual table)

Security-by-default:
- Parameterized SQL only
- sqlite-vec extension loading is enabled only briefly during initialization

TTL note:
- `PutOp.ttl` values are interpreted in **minutes** (converted to milliseconds).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import DocMindSettings, settings
from src.persistence.path_utils import resolve_path_under_data_dir
from src.utils.time import now_ms

try:  # pragma: no cover - optional native dependency
    import sqlite_vec
except (ImportError, ModuleNotFoundError):  # pragma: no cover - defensive
    sqlite_vec = None  # type: ignore[assignment]

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

StoreOp = GetOp | PutOp | SearchOp | ListNamespacesOp

# Store tables
ITEMS_TABLE = "docmind_store_items"
VEC_TABLE = "docmind_store_vec"

# Namespace representation
_NS_DELIM = "\x1f"
_MAX_NS_DEPTH = 8

# Namespace layout: (user_id, session_type, thread_id, ...)
# Used by chat session purge logic (SPEC-041).
NAMESPACE_THREAD_INDEX = 2

# Oversample to compensate for vec0 offset limitations.
VEC0_OVERSAMPLE = 256


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def _ns_parts(namespace: tuple[str, ...]) -> tuple[str, ...]:
    if len(namespace) > _MAX_NS_DEPTH:
        raise ValueError(f"namespace depth > {_MAX_NS_DEPTH}: {namespace!r}")
    return tuple(str(p) for p in namespace)


def _ns_key(namespace: tuple[str, ...]) -> str:
    parts = _ns_parts(namespace)
    return _NS_DELIM.join(parts)


def _get_by_dotted_path(obj: Any, dotted_path: str) -> Any:
    """Extract a value from a nested dict using dot notation.

    Safely navigates nested dictionaries using a dotted path (e.g., 'a.b.c').
    Returns None if any component is missing or if the current value is not a dict.

    Args:
        obj: Dictionary to navigate (or any object, treated as non-dict if not a dict).
        dotted_path: Dot-separated path string (e.g., 'metadata.user.id').

    Returns:
        The value at the dotted path, or None if the path is invalid or missing.

    Example:
        >>> _get_by_dotted_path({'a': {'b': 42}}, 'a.b')
        42
        >>> _get_by_dotted_path({'a': {'b': 42}}, 'a.c')
        None
    """
    cur: Any = obj
    for part in dotted_path.split("."):
        if not part:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _cmp_gt(a: Any, b: Any) -> bool:
    """Return True if a > b with type safety."""
    try:
        return a is not None and a > b
    except TypeError:
        return False


def _cmp_gte(a: Any, b: Any) -> bool:
    """Return True if a >= b with type safety."""
    try:
        return a is not None and a >= b
    except TypeError:
        return False


def _cmp_lt(a: Any, b: Any) -> bool:
    """Return True if a < b with type safety."""
    try:
        return a is not None and a < b
    except TypeError:
        return False


def _cmp_lte(a: Any, b: Any) -> bool:
    """Return True if a <= b with type safety."""
    try:
        return a is not None and a <= b
    except TypeError:
        return False


_FILTER_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": _cmp_gt,
    "$gte": _cmp_gte,
    "$lt": _cmp_lt,
    "$lte": _cmp_lte,
}


def _matches_filter_clause(extracted: Any, clause_value: Any) -> bool:
    """Evaluate a filter clause against extracted values.

    Handles three clause types:
    1. Operator dict (e.g., {'$gt': 10}): applies operators ($gt, $gte, $lt,
       $lte, $eq, $ne)
    2. Scalar value: direct equality check
    3. None: checks if value is None

    List values are handled via "any" semantics: if the extracted value is a
    list, the clause matches if it matches any element in the list.

    Args:
        extracted: Value extracted from object being filtered (may be None
            or list).
        clause_value: The filter clause from query (scalar, operator dict, or
            None).

    Returns:
        True if extracted value satisfies the clause, False otherwise.

    Raises:
        ValueError: If clause_value contains an unsupported operator.

    Example:
        >>> _matches_filter_clause(15, {'$gt': 10})
        True
        >>> _matches_filter_clause(5, {'$gt': 10})
        False
        >>> _matches_filter_clause([5, 15], {'$gt': 10})
        True  # Matches because 15 > 10
    """
    if isinstance(extracted, list):
        return any(_matches_filter_clause(v, clause_value) for v in extracted)

    if isinstance(clause_value, dict):
        for op, v in clause_value.items():
            handler = _FILTER_OPS.get(op)
            if handler is None:
                raise ValueError(f"Unsupported filter operator: {op!r}")
            if not handler(extracted, v):
                return False
        return True

    if clause_value is None:
        return extracted is None
    return extracted == clause_value


def _match_namespace(ns: tuple[str, ...], cond: MatchCondition) -> bool:
    """Check if namespace matches a condition with prefix/suffix matching.

    Supports wildcard matching using '*' to skip individual namespace
    components.
    - Prefix match: namespace must start with condition path (e.g.,
      ('user1', 'session1') matches prefix ('user1', '*'))
    - Suffix match: namespace must end with condition path (e.g.,
      ('org', 'user1', 'session1') matches suffix ('user1', 'session1'))

    Args:
        ns: The namespace tuple to test (e.g., ('user_id', 'session_type',
            'thread_id')).
        cond: MatchCondition with match_type ('prefix' or 'suffix') and
            path components.

    Returns:
        True if namespace matches condition, False otherwise (including
        unsupported match_type).

    Example:
        >>> cond_prefix = MatchCondition(path=('user1', '*'),
        ...     match_type='prefix')
        >>> _match_namespace(('user1', 'session1', 'thread1'), cond_prefix)
        True
        >>> cond_suffix = MatchCondition(path=('session1',),
        ...     match_type='suffix')
        >>> _match_namespace(('user1', 'session1'), cond_suffix)
        True
    """
    path = cond.path
    result = False
    if cond.match_type == "prefix" and len(path) <= len(ns):
        result = True
        for i, p in enumerate(path):
            if p == "*":
                continue
            if ns[i] != p:
                result = False
                break
    elif cond.match_type == "suffix" and len(path) <= len(ns):
        start = len(ns) - len(path)
        result = True
        for i, p in enumerate(path):
            if p == "*":
                continue
            if ns[start + i] != p:
                result = False
                break
    return result


def _matches_filter(value: dict[str, Any], filt: dict[str, Any] | None) -> bool:
    """Check if value object matches all clauses in a filter dict.

    All clauses must match (AND semantics). Each key in the filter dict is a
    dotted path used to extract a value from the object, then compared
    against the clause value using _matches_filter_clause.

    Args:
        value: The object (typically an item dict) to filter.
        filt: Filter dict mapping dotted paths to clause values, or None to
            skip filtering.

    Returns:
        True if all clauses match (or filt is None), False if any clause
        fails.

    Example:
        >>> obj = {'metadata': {'priority': 10, 'status': 'active'}}
        >>> filt = {'metadata.priority': {'$gte': 5},
        ...         'metadata.status': 'active'}
        >>> _matches_filter(obj, filt)
        True
    """
    if not filt:
        return True
    for k, clause in filt.items():
        extracted = _get_by_dotted_path(value, str(k))
        if not _matches_filter_clause(extracted, clause):
            return False
    return True


class DocMindSqliteStore(BaseStore):
    """SQLite-backed LangGraph BaseStore with vec0-powered semantic search.

    TTL values are interpreted as minutes.
    """

    supports_ttl: bool = True

    __slots__ = (
        "_closed",
        "_conn",
        "_filter_fetch_cap",
        "_lock",
        "_path",
        "_tokenized_fields",
        "_vec_enabled",
        "embeddings",
        "index_config",
    )

    def __init__(
        self,
        path: Path,
        *,
        index: IndexConfig | None = None,
        filter_fetch_cap: int = 5000,
        cfg: DocMindSettings = settings,
    ) -> None:
        """Create a persistent store bound to the given SQLite DB path.

        Args:
            path: SQLite database path (typically settings.chat.sqlite_path).
            index: Optional semantic search configuration (dims + embed + fields).
            filter_fetch_cap: Max rows fetched when filtering in Python.
            cfg: DocMind settings used for data_dir validation.
        """
        self._path = resolve_path_under_data_dir(
            path=path,
            data_dir=cfg.data_dir,
            label="Memory store",
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
            raise
        self._conn = conn
        self._lock = threading.Lock()
        self._closed = False
        self._filter_fetch_cap = max(1, int(filter_fetch_cap))

        self.index_config = index.copy() if index else None
        self.embeddings = ensure_embeddings(index.get("embed")) if index else None
        self._tokenized_fields: list[tuple[str, str | list[Any]]] = []
        if self.index_config:
            fields = self.index_config.get("fields") or ["$"]
            self._tokenized_fields = [
                (p, tokenize_path(p) if p != "$" else "$") for p in fields
            ]

        self._vec_enabled = False
        self._setup_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection (best-effort).

        Safe to call multiple times.
        """
        with self._lock:
            if self._closed:
                return

            with contextlib.suppress(Exception):
                self._conn.commit()
            with contextlib.suppress(Exception):
                self._conn.close()

            self._closed = True
            self._vec_enabled = False

    def __enter__(self) -> DocMindSqliteStore:
        """Return self for context manager use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Close the store on context manager exit."""
        self.close()

    # BaseStore API
    def batch(self, ops: Iterable[StoreOp]) -> list[Result]:
        """Execute a batch of store operations synchronously."""
        results: list[Result] = []
        with self._lock:
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._handle_get(op))
                elif isinstance(op, PutOp):
                    results.append(self._handle_put(op))
                elif isinstance(op, SearchOp):
                    results.append(self._handle_search(op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._handle_list_namespaces(op))
                else:
                    raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[StoreOp]) -> list[Result]:
        """Execute a batch of store operations asynchronously."""
        return await asyncio.to_thread(self.batch, ops)

    # Schema / initialization
    def _setup_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docmind_store_items (
                item_id INTEGER PRIMARY KEY,
                ns0 TEXT NOT NULL DEFAULT '',
                ns1 TEXT NOT NULL DEFAULT '',
                ns2 TEXT NOT NULL DEFAULT '',
                ns3 TEXT NOT NULL DEFAULT '',
                ns4 TEXT NOT NULL DEFAULT '',
                ns5 TEXT NOT NULL DEFAULT '',
                ns6 TEXT NOT NULL DEFAULT '',
                ns7 TEXT NOT NULL DEFAULT '',
                ns_key TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                accessed_at_ms INTEGER NOT NULL,
                expires_at_ms INTEGER,
                UNIQUE(ns_key, key)
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docmind_store_items_ns0 "
            "ON docmind_store_items(ns0);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docmind_store_items_ns_key "
            "ON docmind_store_items(ns_key);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_docmind_store_items_updated "
            "ON docmind_store_items(updated_at_ms);"
        )
        self._conn.commit()

        self._setup_vec0()

    def _setup_vec0(self) -> None:
        if not self.index_config or not self.embeddings:
            self._vec_enabled = False
            return
        if sqlite_vec is None:
            logger.warning("sqlite-vec unavailable; semantic store search disabled")
            self._vec_enabled = False
            return
        dims = int(self.index_config.get("dims") or 0)
        if dims <= 0:
            raise ValueError("index.dims must be set for semantic store search")
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        except Exception as exc:  # pragma: no cover - depends on platform
            logger.warning("sqlite-vec load failed; semantic search disabled: %s", exc)
            self._vec_enabled = False
            return
        finally:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._conn.enable_load_extension(False)

        # vec0 metadata constraints only support simple operators; we store namespace
        # parts in dedicated columns to support prefix matching via equality.
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE}
            USING vec0(
                item_id INTEGER PRIMARY KEY,
                ns0 TEXT,
                ns1 TEXT,
                ns2 TEXT,
                ns3 TEXT,
                ns4 TEXT,
                ns5 TEXT,
                ns6 TEXT,
                ns7 TEXT,
                embedding float[{dims}] distance_metric=cosine
            );
            """
        )
        self._conn.commit()
        self._vec_enabled = True

    # Op handlers
    def _handle_get(self, op: GetOp) -> Item | None:
        now = now_ms()
        ns_key = _ns_key(op.namespace)
        row = self._conn.execute(
            """
            SELECT
                item_id,
                value_json,
                created_at_ms,
                updated_at_ms,
                accessed_at_ms,
                expires_at_ms
            FROM docmind_store_items
            WHERE ns_key=? AND key=?
              AND (expires_at_ms IS NULL OR expires_at_ms > ?);
            """,
            (ns_key, str(op.key), now),
        ).fetchone()
        if row is None:
            return None

        value = json.loads(str(row["value_json"]))
        created = int(row["created_at_ms"])
        updated = int(row["updated_at_ms"])

        refresh = bool(op.refresh_ttl)
        if refresh and row["expires_at_ms"] is not None:
            # Sliding TTL: refresh access timestamp and extend expiry by the
            # same TTL delta.
            ttl_delta_ms = int(row["expires_at_ms"]) - int(row["accessed_at_ms"])
            if ttl_delta_ms < 0:
                logger.debug(
                    "Negative TTL delta detected (item_id={}, key={}, accessed={}, "
                    "expires={}, now={}, delta={}); clamping to zero",
                    row["item_id"],
                    op.key,
                    row["accessed_at_ms"],
                    row["expires_at_ms"],
                    now,
                    ttl_delta_ms,
                )
            expires_at_ms = now + max(0, ttl_delta_ms)
            self._conn.execute(
                """
                UPDATE docmind_store_items
                SET accessed_at_ms=?, expires_at_ms=?
                WHERE item_id=?;
                """,
                (now, expires_at_ms, int(row["item_id"])),
            )
            self._conn.commit()

        return Item(
            value=value,
            key=str(op.key),
            namespace=tuple(op.namespace),
            created_at=_ms_to_dt(created),
            updated_at=_ms_to_dt(updated),
        )

    def _handle_put(self, op: PutOp) -> Item | None:
        now = now_ms()
        ns_parts = _ns_parts(op.namespace)
        ns_key = _NS_DELIM.join(ns_parts)
        key = str(op.key)

        if op.value is None:
            row = self._conn.execute(
                "SELECT item_id FROM docmind_store_items WHERE ns_key=? AND key=?;",
                (ns_key, key),
            ).fetchone()
            if row is not None:
                item_id = int(row["item_id"])
                if self._vec_enabled:
                    with contextlib.suppress(sqlite3.OperationalError):
                        self._conn.execute(
                            "DELETE FROM docmind_store_vec WHERE item_id=?;",
                            (item_id,),
                        )
                self._conn.execute(
                    "DELETE FROM docmind_store_items WHERE item_id=?;",
                    (item_id,),
                )
                self._conn.commit()
            return None

        value_json = json.dumps(op.value, ensure_ascii=False, separators=(",", ":"))
        existing = self._conn.execute(
            """
            SELECT item_id, created_at_ms
            FROM docmind_store_items
            WHERE ns_key=? AND key=?;
            """,
            (ns_key, key),
        ).fetchone()

        expires_at_ms: int | None = None
        if op.ttl is not None:
            # op.ttl is specified in minutes.
            expires_at_ms = now + int(float(op.ttl) * 60_000.0)

        ns_cols = list(ns_parts) + [""] * (_MAX_NS_DEPTH - len(ns_parts))
        if existing is None:
            cur = self._conn.execute(
                """
                INSERT INTO docmind_store_items
                    (ns0, ns1, ns2, ns3, ns4, ns5, ns6, ns7, ns_key, key, value_json,
                     created_at_ms, updated_at_ms, accessed_at_ms, expires_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    *ns_cols[:_MAX_NS_DEPTH],
                    ns_key,
                    key,
                    value_json,
                    now,
                    now,
                    now,
                    expires_at_ms,
                ),
            )
            if cur.lastrowid is None:  # pragma: no cover - defensive
                raise RuntimeError("SQLite insert did not return lastrowid")
            item_id = int(cur.lastrowid)
            created_at_ms = now
        else:
            item_id = int(existing["item_id"])
            created_at_ms = int(existing["created_at_ms"])
            self._conn.execute(
                """
                UPDATE docmind_store_items
                SET value_json=?, updated_at_ms=?, accessed_at_ms=?, expires_at_ms=?,
                    ns0=?, ns1=?, ns2=?, ns3=?, ns4=?, ns5=?, ns6=?, ns7=?
                WHERE item_id=?;
                """,
                (
                    value_json,
                    now,
                    now,
                    expires_at_ms,
                    *ns_cols[:_MAX_NS_DEPTH],
                    item_id,
                ),
            )

        # Indexing control: False disables embedding updates for this item.
        if self._vec_enabled and op.index is not False:
            try:
                embedding = self._embed_value(op.value, override_fields=op.index)
                self._upsert_embedding(
                    item_id=item_id, ns_cols=ns_cols, embedding=embedding
                )
            except ValueError as exc:  # pragma: no cover - fail-open
                logger.warning("Memory embedding dimension mismatch: {}", exc)
            except Exception as exc:  # pragma: no cover - fail-open
                logger.warning("Memory embedding update failed; continuing: {}", exc)

        self._conn.commit()
        return Item(
            value=op.value,
            key=key,
            namespace=tuple(op.namespace),
            created_at=_ms_to_dt(created_at_ms),
            updated_at=_ms_to_dt(now),
        )

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        now = now_ms()
        prefix_parts = _ns_parts(op.namespace_prefix)
        prefix_key = _ns_key(prefix_parts) if prefix_parts else None
        prefix_like = f"{prefix_key}{_NS_DELIM}%" if prefix_key else None

        if op.query and self._vec_enabled and self.embeddings:
            return self._search_semantic(
                op,
                prefix_parts=prefix_parts,
                now=now,
            )

        requested_limit = int(op.limit)
        requested_offset = int(op.offset)

        # Non-semantic search: return most recently updated items.
        if op.filter:
            # Apply filters in Python to avoid dynamic SQL construction.
            # NOTE: _filter_fetch_cap (default 5000) bounds this work; raise the
            # cap or use DB-side filtering for heavy workloads.
            fetch_limit = min(
                self._filter_fetch_cap,
                requested_limit + requested_offset + 512,
            )
            sql_limit = fetch_limit
            sql_offset = 0
            if fetch_limit >= self._filter_fetch_cap:
                logger.warning(
                    "Filter fetch cap reached (cap={}, limit={}, offset={}, "
                    "fetch_limit={}); results may be incomplete",
                    self._filter_fetch_cap,
                    requested_limit,
                    requested_offset,
                    fetch_limit,
                )
        else:
            sql_limit = requested_limit
            sql_offset = requested_offset

        cur = self._conn.execute(
            """
            SELECT ns_key, key, value_json, created_at_ms, updated_at_ms
            FROM docmind_store_items
            WHERE (expires_at_ms IS NULL OR expires_at_ms > ?)
              AND (? IS NULL OR ns_key = ? OR ns_key LIKE ?)
            ORDER BY updated_at_ms DESC
            LIMIT ? OFFSET ?;
            """,
            (
                now,
                prefix_key,
                prefix_key,
                prefix_like,
                sql_limit,
                sql_offset,
            ),
        )
        rows = cur.fetchall()
        items: list[SearchItem] = []
        for r in rows:
            value = json.loads(str(r["value_json"]))
            if op.filter and not _matches_filter(value, op.filter):
                continue
            ns_tuple = tuple(str(r["ns_key"]).split(_NS_DELIM)) if r["ns_key"] else ()
            items.append(
                SearchItem(
                    namespace=ns_tuple,
                    key=str(r["key"]),
                    value=value,
                    created_at=_ms_to_dt(int(r["created_at_ms"])),
                    updated_at=_ms_to_dt(int(r["updated_at_ms"])),
                    score=None,
                )
            )

        if op.filter:
            return items[requested_offset : requested_offset + requested_limit]
        return items

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        cur = self._conn.execute(
            "SELECT DISTINCT ns_key FROM docmind_store_items "
            "ORDER BY ns_key LIMIT ? OFFSET ?;",
            (int(op.limit), int(op.offset)),
        )
        raw = [str(r["ns_key"]) for r in cur.fetchall()]
        namespaces = [tuple(x.split(_NS_DELIM)) if x else () for x in raw]
        if not op.match_conditions:
            return (
                namespaces
                if op.max_depth is None
                else [ns[: op.max_depth] for ns in namespaces]
            )

        filtered = [
            ns
            for ns in namespaces
            if all(_match_namespace(ns, cond) for cond in (op.match_conditions or ()))
        ]
        if op.max_depth is not None:
            filtered = [ns[: op.max_depth] for ns in filtered]
        return filtered

    # Query helpers
    def _vec_namespace_prefix_params(self, prefix_parts: tuple[str, ...]) -> list[Any]:
        values: list[Any] = [None] * _MAX_NS_DEPTH
        for i, p in enumerate(prefix_parts[:_MAX_NS_DEPTH]):
            values[i] = p
        # Predicate uses (? IS NULL OR nsX = ?) pairs.
        params: list[Any] = []
        for v in values:
            params.extend([v, v])
        return params

    def _embed_value(self, value: dict[str, Any], override_fields: Any) -> list[float]:
        if not self.embeddings or not self.index_config:
            raise RuntimeError("Embeddings not configured")
        # Use override fields if provided, else defaults.
        fields: list[tuple[str, str | list[Any]]] = self._tokenized_fields
        if isinstance(override_fields, list):
            fields = [
                (p, tokenize_path(p) if p != "$" else "$") for p in override_fields
            ]

        texts: list[str] = []
        for _, tokenized in fields:
            if tokenized == "$":
                texts.append(json.dumps(value, ensure_ascii=False))
                continue
            extracted = get_text_at_path(value, tokenized)
            if not extracted:
                continue
            if isinstance(extracted, list):
                texts.extend([str(x) for x in extracted if x])
            else:
                texts.append(str(extracted))
        joined = "\n".join([t for t in texts if t]).strip()
        if not joined:
            joined = json.dumps(value, ensure_ascii=False)
        vec = self.embeddings.embed_documents([joined])[0]
        return [float(x) for x in vec]

    def _upsert_embedding(
        self, *, item_id: int, ns_cols: list[str], embedding: list[float]
    ) -> None:
        if sqlite_vec is None:
            return
        if self.index_config:
            dims = int(self.index_config.get("dims") or 0)
            if dims and len(embedding) != dims:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {dims}, got {len(embedding)}"
                )
        blob = sqlite_vec.serialize_float32(embedding)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO docmind_store_vec
                (item_id, ns0, ns1, ns2, ns3, ns4, ns5, ns6, ns7, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (item_id, *ns_cols[:_MAX_NS_DEPTH], blob),
        )

    def _search_semantic(
        self,
        op: SearchOp,
        *,
        prefix_parts: tuple[str, ...],
        now: int,
    ) -> list[SearchItem]:
        if op.query is None or self.embeddings is None:
            return []
        if sqlite_vec is None:
            return []
        query_vec = [float(x) for x in self.embeddings.embed_query(str(op.query))]
        blob = sqlite_vec.serialize_float32(query_vec)

        requested_limit = int(op.limit)
        requested_offset = int(op.offset)

        # vec0 supports offset poorly; oversample and slice client-side.
        k = max(1, requested_limit + requested_offset + VEC0_OVERSAMPLE)

        ns_params = self._vec_namespace_prefix_params(prefix_parts)

        cur = self._conn.execute(
            """
            WITH knn AS (
                SELECT item_id, distance
                FROM docmind_store_vec
                WHERE embedding MATCH ?
                  AND k = ?
                  AND (
                    (? IS NULL OR ns0 = ?) AND
                    (? IS NULL OR ns1 = ?) AND
                    (? IS NULL OR ns2 = ?) AND
                    (? IS NULL OR ns3 = ?) AND
                    (? IS NULL OR ns4 = ?) AND
                    (? IS NULL OR ns5 = ?) AND
                    (? IS NULL OR ns6 = ?) AND
                    (? IS NULL OR ns7 = ?)
                  )
                ORDER BY distance
            )
            SELECT
                i.ns_key,
                i.key,
                i.value_json,
                i.created_at_ms,
                i.updated_at_ms,
                knn.distance
            FROM knn
            JOIN docmind_store_items i USING(item_id)
            WHERE (i.expires_at_ms IS NULL OR i.expires_at_ms > ?)
            ORDER BY knn.distance
            LIMIT ?;
            """,
            (
                blob,
                k,
                *ns_params,
                now,
                k,
            ),
        )
        rows = cur.fetchall()
        items: list[SearchItem] = []
        for r in rows:
            value = json.loads(str(r["value_json"]))
            if op.filter and not _matches_filter(value, op.filter):
                continue
            ns_tuple = tuple(str(r["ns_key"]).split(_NS_DELIM)) if r["ns_key"] else ()
            distance = float(r["distance"])
            items.append(
                SearchItem(
                    namespace=ns_tuple,
                    key=str(r["key"]),
                    value=value,
                    created_at=_ms_to_dt(int(r["created_at_ms"])),
                    updated_at=_ms_to_dt(int(r["updated_at_ms"])),
                    score=float((2.0 - distance) / 2.0),
                )
            )
        return items[requested_offset : requested_offset + requested_limit]
