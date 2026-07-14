"""Agent tools for long-term memory (ADR-058 / SPEC-041).

These tools use LangGraph's injected store (BaseStore) and the run config
(`user_id`, `thread_id`) to implement:
- remember: store a memory item
- recall: semantic search over memories
- forget: delete memory item

All tools avoid logging raw user content.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolRuntime
from langgraph.store.base import BaseStore
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.agents.models import AgentRuntimeContext, MemoryNamespaceGenerations
from src.config import settings
from src.persistence.checkpoint_identity import memory_id, memory_namespace
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

MAX_RECALL_LIMIT = 100
MemoryKind = Literal["fact", "preference", "todo", "project_state"]
MemoryOrigin = Literal["explicit", "consolidation"]

# A fixed stripe set bounds lock ownership while serializing every mutation for a
# namespace. Hash collisions only reduce concurrency; they cannot mix store keys.
# ponytail: process-local stripes; use DB-level CAS for multi-process writers.
_MEMORY_NAMESPACE_LOCKS = tuple(threading.RLock() for _ in range(64))
_MEMORY_NAMESPACE_GENERATIONS: dict[tuple[str, ...], int] = {}
_MEMORY_NAMESPACE_TOMBSTONES: set[tuple[str, ...]] = set()


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    """Canonical input for explicit and consolidation memory writes."""

    content: str
    kind: MemoryKind
    importance: float
    tags: list[str] | None
    origin: MemoryOrigin
    source_checkpoint_id: str | None = None
    ttl: int | None = None


class MemoryCapacityError(RuntimeError):
    """Raised when a new explicit memory would exceed its namespace cap."""


class MemoryCandidate(BaseModel):
    """A potential long-term memory extracted from conversation."""

    content: str
    kind: MemoryKind
    importance: float = Field(ge=0.0, le=1.0)
    source_checkpoint_id: str
    tags: list[str] | None = None


class ConsolidationAction(BaseModel):
    """Action to take during memory consolidation."""

    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    existing_id: str | None = None
    candidate: MemoryCandidate | None = None


class ExtractedMemory(BaseModel):
    """Schema used for LLM extraction output (no checkpoint id)."""

    content: str
    kind: MemoryKind
    importance: float = Field(ge=0.0, le=1.0)
    tags: list[str] | None = None


class MemoryExtractionResult(BaseModel):
    """Structured LLM output for memory extraction."""

    memories: list[ExtractedMemory] = Field(default_factory=list)


class MemoryConsolidationPolicy(BaseModel):
    """Policy knobs for memory consolidation and retention."""

    similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    low_importance_threshold: float = Field(ge=0.0, le=1.0, default=0.3)
    low_importance_ttl_minutes: int = Field(ge=0, default=14 * 24 * 60)
    max_items_per_namespace: int = Field(ge=1, default=200)
    max_candidates_per_turn: int = Field(ge=1, default=8)

    @classmethod
    def from_settings(cls) -> Self:
        """Build the canonical policy from current application settings."""
        chat = settings.chat
        return cls(
            similarity_threshold=float(chat.memory_similarity_threshold),
            low_importance_threshold=float(chat.memory_low_importance_threshold),
            low_importance_ttl_minutes=int(chat.memory_low_importance_ttl_days)
            * 24
            * 60,
            max_items_per_namespace=int(chat.memory_max_items_per_namespace),
            max_candidates_per_turn=int(chat.memory_max_candidates_per_turn),
        )


def _namespace_from_config(
    config: RunnableConfig | dict[str, Any] | None,
    *,
    scope: Literal["session", "user"] = "session",
) -> tuple[str, ...]:
    cfg = config.get("configurable") if isinstance(config, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    user_id = str(cfg.get("user_id") or "local")
    thread_id = str(cfg.get("public_thread_id") or cfg.get("thread_id") or "default")
    if scope == "user":
        return memory_namespace(user_id=user_id)
    return memory_namespace(user_id=user_id, thread_id=thread_id)


def _ids_from_config(config: RunnableConfig | dict[str, Any] | None) -> tuple[str, str]:
    cfg = config.get("configurable") if isinstance(config, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    user_id = str(cfg.get("user_id") or "local")
    thread_id = str(cfg.get("public_thread_id") or cfg.get("thread_id") or "default")
    return user_id, thread_id


def _candidate_ttl_minutes(
    candidate: MemoryCandidate, policy: MemoryConsolidationPolicy
) -> int | None:
    if candidate.importance < policy.low_importance_threshold:
        ttl = int(policy.low_importance_ttl_minutes)
        return ttl if ttl > 0 else None
    return None


def _merge_tags(
    existing: dict[str, Any] | None, candidate: MemoryCandidate
) -> list[str] | None:
    existing_tags = existing.get("tags") if isinstance(existing, dict) else None
    merged: list[str] = []
    for tag_list in (existing_tags, candidate.tags):
        if isinstance(tag_list, list):
            merged.extend(str(tag) for tag in tag_list if tag is not None)

    # Order-preserving deduplication using a set for O(n) performance
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in merged:
        if tag and tag not in seen:
            seen.add(tag)
            deduped.append(tag)

    return deduped or None


def _normalize_content(text: Any) -> str:
    return str(text).strip()


@contextmanager
def memory_namespace_lock(namespace: tuple[str, ...]) -> Iterator[None]:
    """Serialize process-local memory mutations for a bounded lock stripe."""
    lock = _MEMORY_NAMESPACE_LOCKS[hash(namespace) % len(_MEMORY_NAMESPACE_LOCKS)]
    with lock:
        yield


def memory_namespace_generation(namespace: tuple[str, ...]) -> int:
    """Return the current process-local mutation generation for a namespace."""
    with memory_namespace_lock(namespace):
        return _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0)


def capture_memory_namespace_generations(
    *, user_id: str, thread_id: str
) -> MemoryNamespaceGenerations:
    """Capture the mutation fences inherited by one admitted graph turn."""
    return {
        "session": memory_namespace_generation(
            memory_namespace(user_id=user_id, thread_id=thread_id)
        ),
        "user": memory_namespace_generation(memory_namespace(user_id=user_id)),
    }


def advance_memory_namespace_generation(namespace: tuple[str, ...]) -> int:
    """Invalidate already-scheduled work while allowing newly scheduled work."""
    with memory_namespace_lock(namespace):
        generation = _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0) + 1
        _MEMORY_NAMESPACE_GENERATIONS[namespace] = generation
        return generation


def try_advance_memory_namespace_generation(
    namespace: tuple[str, ...], *, timeout_s: float = 0.0
) -> int | None:
    """Advance a namespace generation without an unbounded lock wait."""
    lock = _MEMORY_NAMESPACE_LOCKS[hash(namespace) % len(_MEMORY_NAMESPACE_LOCKS)]
    if not lock.acquire(timeout=max(0.0, float(timeout_s))):
        return None
    try:
        generation = _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0) + 1
        _MEMORY_NAMESPACE_GENERATIONS[namespace] = generation
        return generation
    finally:
        lock.release()


def tombstone_memory_namespace(namespace: tuple[str, ...]) -> int:
    """Permanently reject future writes to a hard-purged namespace."""
    with memory_namespace_lock(namespace):
        generation = _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0) + 1
        _MEMORY_NAMESPACE_GENERATIONS[namespace] = generation
        _MEMORY_NAMESPACE_TOMBSTONES.add(namespace)
        return generation


def try_tombstone_memory_namespace(
    namespace: tuple[str, ...], *, timeout_s: float
) -> int | None:
    """Permanently fence a namespace within a caller-owned time budget."""
    lock = _MEMORY_NAMESPACE_LOCKS[hash(namespace) % len(_MEMORY_NAMESPACE_LOCKS)]
    if not lock.acquire(timeout=max(0.0, float(timeout_s))):
        return None
    try:
        generation = _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0) + 1
        _MEMORY_NAMESPACE_GENERATIONS[namespace] = generation
        _MEMORY_NAMESPACE_TOMBSTONES.add(namespace)
        return generation
    finally:
        lock.release()


def is_memory_namespace_tombstoned(namespace: tuple[str, ...]) -> bool:
    """Return whether a hard purge permanently fenced the namespace."""
    with memory_namespace_lock(namespace):
        return namespace in _MEMORY_NAMESPACE_TOMBSTONES


def _memory_write_is_current(
    namespace: tuple[str, ...], expected_generation: int | None
) -> bool:
    if namespace in _MEMORY_NAMESPACE_TOMBSTONES:
        return False
    return expected_generation is None or expected_generation == (
        _MEMORY_NAMESPACE_GENERATIONS.get(namespace, 0)
    )


def _state_deadline(state: dict[str, Any]) -> float | None:
    try:
        deadline_ts = float(state["deadline_ts"])
    except (KeyError, TypeError, ValueError):
        return None
    return deadline_ts if math.isfinite(deadline_ts) else None


def memory_generation_from_state(
    state: dict[str, Any], scope: Literal["session", "user"]
) -> int | None:
    """Return one validated admitted generation from graph state."""
    generations = state.get("memory_generations")
    if not isinstance(generations, dict):
        return None
    generation = generations.get(scope)
    if isinstance(generation, bool) or not isinstance(generation, int):
        return None
    return generation if generation >= 0 else None


def _has_mutation_budget(deadline_ts: float) -> bool:
    return math.isfinite(deadline_ts) and time.monotonic() < deadline_ts


def _item_value(item: Any) -> dict[str, Any] | None:
    value = getattr(item, "value", None)
    return value if isinstance(value, dict) else None


def _is_explicit_memory(item: Any) -> bool:
    value = _item_value(item)
    return value is not None and value.get("origin") == "explicit"


def _require_explicit_write_capacity(
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    canonical_id: str,
    replace_memory_id: str | None,
) -> None:
    """Reject only genuinely new explicit records once the namespace is full."""
    max_items = int(settings.chat.memory_max_items_per_namespace)
    if max_items <= 0:
        return
    if store.get(namespace, canonical_id) is not None:
        return
    if (
        replace_memory_id is not None
        and store.get(namespace, replace_memory_id) is not None
    ):
        return
    if len(store.search(namespace, query=None, limit=max_items)) >= max_items:
        raise MemoryCapacityError("memory namespace is at capacity")


def save_memory(  # noqa: PLR0911 - explicit mutation stop conditions
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    write: MemoryWrite,
    deadline_ts: float | None = None,
    replace_memory_id: str | None = None,
    expected_generation: int | None = None,
) -> str | None:
    """Write one canonical memory while preserving explicit user intent.

    ``None`` means no mutation occurred because the deadline/fence rejected it
    or a consolidation write yielded to an explicit record.
    """
    normalized_content = _normalize_content(write.content)
    canonical_id = memory_id(normalized_content, write.kind)
    payload: dict[str, Any] = {
        "content": normalized_content,
        "kind": str(write.kind),
        "importance": max(0.0, min(1.0, float(write.importance))),
        "tags": list(write.tags) if write.tags is not None else None,
        "origin": write.origin,
    }
    if write.source_checkpoint_id is not None:
        payload["source_checkpoint_id"] = str(write.source_checkpoint_id)

    with memory_namespace_lock(namespace):
        if not _memory_write_is_current(namespace, expected_generation):
            return None
        if deadline_ts is not None and not _has_mutation_budget(deadline_ts):
            return None
        if write.origin == "explicit":
            _require_explicit_write_capacity(
                store,
                namespace,
                canonical_id=canonical_id,
                replace_memory_id=replace_memory_id,
            )

        source = None
        if replace_memory_id is not None and replace_memory_id != canonical_id:
            source = store.get(namespace, replace_memory_id)
            if write.origin == "consolidation" and _is_explicit_memory(source):
                return None

        target = (
            store.get(namespace, canonical_id)
            if write.origin == "consolidation"
            else None
        )
        if write.origin == "consolidation" and _is_explicit_memory(target):
            if replace_memory_id is None or replace_memory_id == canonical_id:
                return None
            if deadline_ts is not None and not _has_mutation_budget(deadline_ts):
                return None
            store.delete(namespace, replace_memory_id)
            return canonical_id

        if deadline_ts is not None and not _has_mutation_budget(deadline_ts):
            return None
        if replace_memory_id is not None and replace_memory_id != canonical_id:
            # A content change also changes the deterministic key. Preserve the
            # old record until the target upsert succeeds, then always finish the
            # already-admitted logical rekey even if its deadline crosses. A
            # failed delete can leave a duplicate, but an identical retry
            # converges; deleting first could lose the only durable record.
            if write.ttl is None:
                store.put(namespace, canonical_id, payload, index=["content"])
            else:
                store.put(
                    namespace,
                    canonical_id,
                    payload,
                    index=["content"],
                    ttl=write.ttl,
                )
            store.delete(namespace, replace_memory_id)
        elif write.ttl is None:
            store.put(namespace, canonical_id, payload, index=["content"])
        else:
            store.put(
                namespace,
                canonical_id,
                payload,
                index=["content"],
                ttl=write.ttl,
            )
        return canonical_id


def delete_memory(
    store: BaseStore,
    namespace: tuple[str, ...],
    memory_id_value: str,
    *,
    deadline_ts: float | None = None,
    expected_generation: int | None = None,
) -> bool:
    """Delete one memory if its deadline and optional generation remain valid."""
    with memory_namespace_lock(namespace):
        if expected_generation is not None and not _memory_write_is_current(
            namespace, expected_generation
        ):
            return False
        if deadline_ts is not None and not _has_mutation_budget(deadline_ts):
            return False
        store.delete(namespace, str(memory_id_value))
        return True


def _blocked_memory_response(
    event_key: Literal[
        "chat.memory_saved",
        "chat.memory_searched",
        "chat.memory_deleted",
    ],
    *,
    scope: Literal["session", "user"],
    thread_redacted: str,
    user_redacted: str,
    error_type: str = "deadline_invalid_or_expired",
    error: str = "operation deadline exceeded",
) -> str:
    with _SuppressTelemetry():
        log_jsonl(
            {
                event_key: False,
                "scope": scope,
                "error_type": error_type,
                "thread_id": thread_redacted,
                "user_id": user_redacted,
            }
        )
    response: dict[str, Any] = {
        "ok": False,
        "error": error,
    }
    if event_key == "chat.memory_searched":
        response["memories"] = []
    return json.dumps(response)


def _content_matches(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.strip().casefold() == b.strip().casefold()


def _prepare_policy(
    policy: MemoryConsolidationPolicy | None,
) -> MemoryConsolidationPolicy:
    if policy is not None:
        return policy
    return MemoryConsolidationPolicy.from_settings()


@tool
def remember(  # noqa: PLR0911 - explicit fail-closed tool outcomes
    content: str,
    state: Annotated[dict[str, Any], InjectedState],
    runtime: ToolRuntime[AgentRuntimeContext, dict[str, Any]],
    kind: MemoryKind = "fact",
    importance: float = 0.7,
    tags: list[str] | None = None,
    scope: Literal["session", "user"] = "session",
) -> str:
    """Store a long-term memory item (explicit user intent)."""
    start = time.perf_counter()
    store = runtime.store
    config = runtime.config
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    user_id, thread_id = _ids_from_config(config)
    thread_redacted = build_pii_log_entry(
        str(thread_id), key_id="telemetry.thread_id"
    ).redacted
    user_redacted = build_pii_log_entry(
        str(user_id), key_id="telemetry.user_id"
    ).redacted
    tags_value: list[str] | None = None
    if (
        tags is not None
        and isinstance(tags, Iterable)
        and not isinstance(tags, (str, bytes))
    ):
        tags_value = list(tags)
    deadline_ts = _state_deadline(state)
    try:
        if deadline_ts is None or not _has_mutation_budget(deadline_ts):
            return _blocked_memory_response(
                "chat.memory_saved",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
            )
        expected_generation = memory_generation_from_state(state, scope)
        if expected_generation is None:
            return _blocked_memory_response(
                "chat.memory_saved",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
                error_type="memory_generation_invalid",
                error="memory mutation invalidated",
            )
        mem_id = save_memory(
            store,
            ns,
            write=MemoryWrite(
                content=content,
                kind=kind,
                importance=importance,
                tags=tags_value,
                origin="explicit",
            ),
            deadline_ts=deadline_ts,
            expected_generation=expected_generation,
        )
        if mem_id is None:
            return _blocked_memory_response(
                "chat.memory_saved",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
                error_type="memory_generation_stale_or_deadline_expired",
                error="memory mutation invalidated",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with _SuppressTelemetry():
            log_jsonl(
                {
                    "chat.memory_saved": True,
                    "scope": scope,
                    "count": 1,
                    "latency_ms": round(elapsed_ms, 2),
                    "thread_id": thread_redacted,
                    "user_id": user_redacted,
                }
            )
        return json.dumps({"ok": True, "memory_id": mem_id})
    except MemoryCapacityError:
        return _blocked_memory_response(
            "chat.memory_saved",
            scope=scope,
            thread_redacted=thread_redacted,
            user_redacted=user_redacted,
            error_type="memory_capacity_reached",
            error="memory capacity reached",
        )
    except Exception as e:
        redaction = build_pii_log_entry(str(e), key_id="agents.tools.memory.remember")
        logger.debug(
            "remember store.put failed (error_type={} error={})",
            type(e).__name__,
            redaction.redacted,
        )
        with _SuppressTelemetry():
            log_jsonl(
                {
                    "chat.memory_saved": False,
                    "scope": scope,
                    "error_type": type(e).__name__,
                    "error": redaction.redacted,
                    "thread_id": thread_redacted,
                    "user_id": user_redacted,
                }
            )
        return json.dumps({"ok": False, "error": "save failed"})


@tool
def recall_memories(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    runtime: ToolRuntime[AgentRuntimeContext, dict[str, Any]],
    limit: int = 5,
    scope: Literal["session", "user"] = "session",
) -> str:
    """Semantic search across stored memories."""
    start = time.perf_counter()
    store = runtime.store
    config = runtime.config
    if store is None:
        return json.dumps(
            {
                "ok": False,
                "error": "memory store unavailable",
                "memories": [],
            }
        )
    ns = _namespace_from_config(config, scope=scope)
    user_id, thread_id = _ids_from_config(config)
    thread_redacted = build_pii_log_entry(
        str(thread_id), key_id="telemetry.thread_id"
    ).redacted
    user_redacted = build_pii_log_entry(
        str(user_id), key_id="telemetry.user_id"
    ).redacted
    safe_limit = max(1, min(MAX_RECALL_LIMIT, int(limit)))
    deadline_ts = _state_deadline(state)
    if deadline_ts is None or not _has_mutation_budget(deadline_ts):
        return _blocked_memory_response(
            "chat.memory_searched",
            scope=scope,
            thread_redacted=thread_redacted,
            user_redacted=user_redacted,
        )
    try:
        results = store.search(ns, query=str(query), limit=safe_limit)
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="agents.tools.memory.recall")
        logger.debug(
            "recall_memories search failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        with _SuppressTelemetry():
            log_jsonl(
                {
                    "chat.memory_searched": False,
                    "scope": scope,
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                    "thread_id": thread_redacted,
                    "user_id": user_redacted,
                }
            )
        return json.dumps({"ok": False, "error": "search failed", "memories": []})
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _SuppressTelemetry():
        log_jsonl(
            {
                "chat.memory_searched": True,
                "scope": scope,
                "top_k": safe_limit,
                "latency_ms": round(elapsed_ms, 2),
                "result_count": len(results),
                "thread_id": thread_redacted,
                "user_id": user_redacted,
            }
        )
    # Return only structured memory content; do not include internal timings from store.
    out = []
    for item in results:
        value = getattr(item, "value", None)
        if not isinstance(value, dict):
            continue
        out.append(
            {
                "id": getattr(item, "key", None),
                "content": value.get("content"),
                "kind": value.get("kind"),
                "importance": value.get("importance"),
                "tags": value.get("tags"),
                "score": getattr(item, "score", None),
            }
        )
    return json.dumps({"ok": True, "memories": out})


@tool
def forget_memory(
    memory_id: str,
    state: Annotated[dict[str, Any], InjectedState],
    runtime: ToolRuntime[AgentRuntimeContext, dict[str, Any]],
    scope: Literal["session", "user"] = "session",
) -> str:
    """Delete a stored memory item by id."""
    store = runtime.store
    config = runtime.config
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    user_id, thread_id = _ids_from_config(config)
    thread_redacted = build_pii_log_entry(
        str(thread_id), key_id="telemetry.thread_id"
    ).redacted
    user_redacted = build_pii_log_entry(
        str(user_id), key_id="telemetry.user_id"
    ).redacted
    deadline_ts = _state_deadline(state)
    try:
        if deadline_ts is None or not _has_mutation_budget(deadline_ts):
            return _blocked_memory_response(
                "chat.memory_deleted",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
            )
        expected_generation = memory_generation_from_state(state, scope)
        if expected_generation is None:
            return _blocked_memory_response(
                "chat.memory_deleted",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
                error_type="memory_generation_invalid",
                error="memory mutation invalidated",
            )
        if not delete_memory(
            store,
            ns,
            memory_id,
            deadline_ts=deadline_ts,
            expected_generation=expected_generation,
        ):
            return _blocked_memory_response(
                "chat.memory_deleted",
                scope=scope,
                thread_redacted=thread_redacted,
                user_redacted=user_redacted,
                error_type="memory_generation_stale_or_deadline_expired",
                error="memory mutation invalidated",
            )
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="agents.tools.memory.forget")
        logger.debug(
            "forget_memory delete failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        with _SuppressTelemetry():
            log_jsonl(
                {
                    "chat.memory_deleted": False,
                    "scope": scope,
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                    "thread_id": thread_redacted,
                    "user_id": user_redacted,
                }
            )
        return json.dumps({"ok": False, "error": "delete failed"})

    with _SuppressTelemetry():
        log_jsonl(
            {
                "chat.memory_deleted": True,
                "scope": scope,
                "thread_id": thread_redacted,
                "user_id": user_redacted,
            }
        )
    return json.dumps({"ok": True})


class _SuppressTelemetry:
    """Defensive wrapper for telemetry logging.

    `log_jsonl` is intended to be best-effort, but keeping this context manager
    ensures telemetry failures (or test stubs that raise) never break tool
    execution.
    """

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        if exc_type is not None:
            is_expected = issubclass(exc_type, (OSError, RuntimeError, ValueError))
            log_func = logger.debug if is_expected else logger.warning
            redaction = build_pii_log_entry(
                str(exc), key_id="agents.tools.memory.telemetry"
            )
            log_func(
                "memory tool telemetry error (error_type={}, error={})",
                exc_type.__name__,
                redaction.redacted,
            )
            return True
        return False


def _recent_dialogue(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Return the most recent human/assistant dialogue slice.

    Excludes tool/system messages and starts at the last human message.
    """
    last_user_index: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        if getattr(messages[idx], "type", None) == "human":
            last_user_index = idx
            break
    if last_user_index is None:
        return []

    dialogue: list[AnyMessage] = []
    for msg in messages[last_user_index:]:
        if getattr(msg, "type", None) in {"human", "ai"}:
            dialogue.append(msg)
    return dialogue


def _conversation_text(dialogue: list[AnyMessage]) -> str:
    """Build a stable text representation for extraction prompts."""
    try:
        return get_buffer_string(dialogue, human_prefix="User", ai_prefix="Assistant")
    except Exception:
        return "\n".join(
            f"{'User' if getattr(m, 'type', None) == 'human' else 'Assistant'}: "
            f"{_normalize_content(getattr(m, 'content', m))}"
            for m in dialogue
        )


def _extract_prompt(text: str, policy: MemoryConsolidationPolicy) -> str:
    return (
        "You extract long-term user memories. Only use facts/preferences/"
        "todos/project state explicitly stated by the user. Ignore assistant "
        "suggestions or hallucinations. Return at most "
        f"{policy.max_candidates_per_turn} items.\n\n"
        "Conversation:\n"
        f"{text}\n\n"
        "Respond with structured JSON."
    )


def _strip_code_fences(content: str) -> str:
    if "```json" in content:
        return content.split("```json")[1].split("```", maxsplit=1)[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```", maxsplit=1)[0].strip()
    return content


def _invoke_extraction_llm(
    llm: Any,
    prompt: str,
    *,
    timeout_s: float | None = None,
) -> MemoryExtractionResult | None:
    invoke_kwargs = {"timeout": timeout_s} if timeout_s is not None else {}
    structured_llm: Any | None = None
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(MemoryExtractionResult)
        except Exception as exc:
            logger.debug(
                "Structured output unavailable for memory extraction: {}",
                type(exc).__name__,
            )
            structured_llm = None

    if structured_llm is not None:
        response = structured_llm.invoke(prompt, **invoke_kwargs)
        if isinstance(response, MemoryExtractionResult):
            return response
        if isinstance(response, dict):
            return MemoryExtractionResult.model_validate(response)
        return None

    response = llm.invoke(prompt, **invoke_kwargs)
    content = getattr(response, "content", str(response))
    try:
        payload = json.loads(_strip_code_fences(str(content)))
    except json.JSONDecodeError as exc:
        logger.debug("Memory extraction JSON parse failed: {}", type(exc).__name__)
        return None
    if isinstance(payload, dict):
        payload = payload.get("memories", [])
    return MemoryExtractionResult.model_validate({"memories": payload})


def _candidates_from_extracted(
    extracted: MemoryExtractionResult,
    checkpoint_id: str,
    policy: MemoryConsolidationPolicy,
) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    for item in extracted.memories[: policy.max_candidates_per_turn]:
        content = _normalize_content(item.content)
        if not content:
            continue
        try:
            candidates.append(
                MemoryCandidate(
                    content=content,
                    kind=item.kind,
                    importance=float(item.importance),
                    tags=item.tags,
                    source_checkpoint_id=str(checkpoint_id),
                )
            )
        except ValidationError as exc:
            logger.debug("Invalid memory candidate skipped: {}", type(exc).__name__)
    return candidates


def _dedupe_candidates(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    deduped: dict[tuple[str, str], MemoryCandidate] = {}
    for cand in candidates:
        key = (cand.kind, cand.content.casefold())
        existing = deduped.get(key)
        if existing is None or cand.importance > existing.importance:
            deduped[key] = cand
    return list(deduped.values())


def extract_memory_candidates(
    messages: Sequence[AnyMessage],
    checkpoint_id: str,
    llm: BaseChatModel | BaseLanguageModel | None = None,
    *,
    policy: MemoryConsolidationPolicy | None = None,
    deadline_ts: float | None = None,
) -> list[MemoryCandidate]:
    """Extract potential memories from the conversation turn."""
    if not llm or not messages:
        return []

    policy = _prepare_policy(policy)
    dialogue = _recent_dialogue(messages)
    if not dialogue:
        return []

    logger.debug("Extracting memory candidates from {} messages", len(dialogue))
    prompt = _extract_prompt(_conversation_text(dialogue), policy)
    try:
        timeout_s: float | None = None
        if deadline_ts is not None:
            timeout_s = float(deadline_ts) - time.monotonic()
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                return []
        extracted = _invoke_extraction_llm(llm, prompt, timeout_s=timeout_s)
        if extracted is None:
            return []
        candidates = _candidates_from_extracted(extracted, checkpoint_id, policy)
        return _dedupe_candidates(candidates)
    except Exception as exc:
        logger.warning("Memory extraction failed: {}", type(exc).__name__)
        return []


def consolidate_memory_candidates(
    candidates: list[MemoryCandidate],
    store: BaseStore,
    namespace: tuple[str, ...],
    similarity_threshold: float | None = None,
    *,
    deadline_ts: float | None = None,
    policy: MemoryConsolidationPolicy | None = None,
) -> list[ConsolidationAction]:
    """Compare candidates with existing memories and decide actions."""
    policy = _prepare_policy(policy)
    threshold = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else policy.similarity_threshold
    )
    actions: list[ConsolidationAction] = []

    def _with_merged_tags(
        candidate: MemoryCandidate, existing_value: dict[str, Any] | None
    ) -> MemoryCandidate:
        merged_tags = _merge_tags(existing_value, candidate)
        if merged_tags == candidate.tags:
            return candidate
        return candidate.model_copy(update={"tags": merged_tags})

    def _determine_consolidation_action(
        score: float | None,
        threshold_value: float,
        existing_value: dict[str, Any] | None,
        candidate: MemoryCandidate,
        existing_id: str,
    ) -> ConsolidationAction:
        existing_kind = (
            existing_value.get("kind") if isinstance(existing_value, dict) else None
        )
        existing_content = (
            existing_value.get("content") if isinstance(existing_value, dict) else None
        )
        existing_importance = (
            float(existing_value.get("importance", 0.5))
            if isinstance(existing_value, dict)
            else 0.5
        )

        existing_is_explicit = (
            isinstance(existing_value, dict)
            and existing_value.get("origin") == "explicit"
        )

        action: ConsolidationAction = ConsolidationAction(
            action="ADD", candidate=candidate
        )
        if score is None:
            if _content_matches(existing_content, candidate.content):
                if existing_is_explicit:
                    action = ConsolidationAction(action="NOOP", existing_id=existing_id)
                elif candidate.importance > existing_importance:
                    updated_candidate = _with_merged_tags(candidate, existing_value)
                    action = ConsolidationAction(
                        action="UPDATE",
                        existing_id=existing_id,
                        candidate=updated_candidate,
                    )
                else:
                    action = ConsolidationAction(action="NOOP", existing_id=existing_id)
        else:
            score_value = float(score)
            if score_value >= threshold_value and existing_kind == candidate.kind:
                if existing_is_explicit:
                    action = ConsolidationAction(action="NOOP", existing_id=existing_id)
                elif _content_matches(existing_content, candidate.content):
                    if candidate.importance > existing_importance:
                        updated_candidate = _with_merged_tags(candidate, existing_value)
                        action = ConsolidationAction(
                            action="UPDATE",
                            existing_id=existing_id,
                            candidate=updated_candidate,
                        )
                    else:
                        action = ConsolidationAction(
                            action="NOOP", existing_id=existing_id
                        )
                elif candidate.importance >= existing_importance:
                    updated_candidate = _with_merged_tags(candidate, existing_value)
                    action = ConsolidationAction(
                        action="UPDATE",
                        existing_id=existing_id,
                        candidate=updated_candidate,
                    )
                else:
                    action = ConsolidationAction(action="NOOP", existing_id=existing_id)
        return action

    for cand in candidates:
        if deadline_ts is not None and not _has_mutation_budget(deadline_ts):
            break
        # Vector search for similar memories in the same namespace
        results = store.search(
            namespace, query=cand.content, limit=1, filter={"kind": cand.kind}
        )
        if results:
            best_match = results[0]
            # Native SqliteStore returns cosine similarity (higher is closer).
            score = getattr(best_match, "score", 0.0)

            existing_value = (
                best_match.value if isinstance(best_match.value, dict) else None
            )
            action = _determine_consolidation_action(
                score=score,
                threshold_value=threshold,
                existing_value=existing_value,
                candidate=cand,
                existing_id=str(best_match.key),
            )
            actions.append(action)
            continue

        # No similar memory found
        actions.append(ConsolidationAction(action="ADD", candidate=cand))

    return actions


def consolidate_and_apply_memory_candidates(
    candidates: list[MemoryCandidate],
    store: BaseStore,
    namespace: tuple[str, ...],
    *,
    deadline_ts: float,
    policy: MemoryConsolidationPolicy | None = None,
    expected_generation: int | None = None,
) -> int:
    """Serialize consolidation and reject stale or tombstoned work."""
    with memory_namespace_lock(namespace):
        if not _memory_write_is_current(namespace, expected_generation):
            return 0
        if not _has_mutation_budget(deadline_ts):
            return 0
        actions = consolidate_memory_candidates(
            candidates,
            store,
            namespace,
            deadline_ts=deadline_ts,
            policy=policy,
        )
        if not _has_mutation_budget(deadline_ts):
            return 0
        return apply_consolidation_policy(
            store,
            namespace,
            actions,
            deadline_ts=deadline_ts,
            policy=policy,
            expected_generation=expected_generation,
        )


def apply_consolidation_policy(
    store: BaseStore,
    namespace: tuple[str, ...],
    actions: list[ConsolidationAction],
    *,
    deadline_ts: float,
    policy: MemoryConsolidationPolicy | None = None,
    expected_generation: int | None = None,
) -> int:
    """Apply consolidation actions to the durable store (SPEC-041)."""
    policy = _prepare_policy(policy)
    change_count = 0
    with memory_namespace_lock(namespace):
        if not _memory_write_is_current(namespace, expected_generation):
            return 0
        for op in actions:
            if not _has_mutation_budget(deadline_ts):
                break
            try:
                if op.action in {"ADD", "UPDATE"} and op.candidate:
                    ttl = _candidate_ttl_minutes(op.candidate, policy)
                    saved_id = save_memory(
                        store,
                        namespace,
                        write=MemoryWrite(
                            content=op.candidate.content,
                            kind=op.candidate.kind,
                            importance=op.candidate.importance,
                            tags=op.candidate.tags,
                            origin="consolidation",
                            source_checkpoint_id=op.candidate.source_checkpoint_id,
                            ttl=ttl,
                        ),
                        deadline_ts=deadline_ts,
                        replace_memory_id=(
                            op.existing_id if op.action == "UPDATE" else None
                        ),
                        expected_generation=expected_generation,
                    )
                    if saved_id is not None:
                        change_count += 1
                elif op.action == "DELETE" and op.existing_id:
                    if delete_memory(
                        store,
                        namespace,
                        op.existing_id,
                        deadline_ts=deadline_ts,
                        expected_generation=expected_generation,
                    ):
                        change_count += 1
            except Exception as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="agents.tools.memory.consolidation"
                )
                logger.warning(
                    "Memory consolidation mutation failed "
                    "(action={} error_type={} error={})",
                    op.action,
                    type(exc).__name__,
                    redaction.redacted,
                )

        if policy.max_items_per_namespace > 0 and _has_mutation_budget(deadline_ts):
            change_count += _enforce_namespace_limits(
                store,
                namespace,
                policy,
                deadline_ts=deadline_ts,
                expected_generation=expected_generation,
            )

    if change_count > 0:
        logger.info("Memory consolidation applied (changes={})", change_count)
    return change_count


def _enforce_namespace_limits(
    store: BaseStore,
    namespace: tuple[str, ...],
    policy: MemoryConsolidationPolicy,
    *,
    deadline_ts: float,
    expected_generation: int | None = None,
) -> int:
    if not _memory_write_is_current(namespace, expected_generation):
        return 0
    max_items = int(policy.max_items_per_namespace)
    if max_items <= 0:
        return 0
    items: list[Any] = []
    offset = 0
    batch_size = max(50, min(500, max_items * 2))
    while True:
        if not _has_mutation_budget(deadline_ts):
            return 0
        batch = store.search(namespace, limit=batch_size, offset=offset)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
        # Safeguard: cap total items to prevent unbounded memory usage
        if len(items) > max_items * 10:
            logger.warning(
                "Memory namespace has excessive items ({}); capping eviction scan",
                len(items),
            )
            break

    if len(items) <= max_items:
        return 0
    if len(items) > max_items * 2:
        logger.debug(
            "Memory namespace has {} items (limit {}, batch {}); eviction may be slow",
            len(items),
            max_items,
            batch_size,
        )

    def sort_key(item: Any) -> tuple[float, float]:
        value = getattr(item, "value", {}) if hasattr(item, "value") else {}
        importance = 0.5
        if isinstance(value, dict):
            try:
                importance = float(value.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
        updated_at = getattr(item, "updated_at", None)
        updated_ts = 0.0
        if isinstance(updated_at, datetime):
            updated_ts = updated_at.timestamp()
        return (importance, updated_ts)

    # The automatic retention cap never deletes explicit user intent. Derived
    # consolidation records are the only eviction candidates; when explicit
    # records already fill the cap, a newly derived record evicts itself.
    derived_items = [
        item
        for item in items
        if (_item_value(item) or {}).get("origin") == "consolidation"
    ]
    items_sorted = sorted(derived_items, key=sort_key)
    to_delete = items_sorted[: min(len(items) - max_items, len(items_sorted))]
    deleted = 0
    for item in to_delete:
        if not _has_mutation_budget(deadline_ts):
            break
        key = getattr(item, "key", None)
        if key is None:
            continue
        try:
            if delete_memory(
                store,
                namespace,
                str(key),
                deadline_ts=deadline_ts,
                expected_generation=expected_generation,
            ):
                deleted += 1
        except Exception as exc:
            logger.debug("Failed to evict memory item: {}", type(exc).__name__)
    return deleted
