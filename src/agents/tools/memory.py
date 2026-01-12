"""Agent tools for long-term memory (ADR-057 / SPEC-041).

These tools use LangGraph's injected store (BaseStore) and the run config
(`user_id`, `thread_id`) to implement:
- remember: store a memory item
- recall: semantic search over memories
- forget: delete memory item

All tools avoid logging raw user content.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from loguru import logger

from src.utils.telemetry import log_jsonl


def _namespace_from_config(
    config: RunnableConfig | dict[str, Any] | None,
    *,
    scope: Literal["session", "user"] = "session",
) -> tuple[str, ...]:
    cfg = config.get("configurable") if isinstance(config, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    user_id = str(cfg.get("user_id") or "local")
    thread_id = str(cfg.get("thread_id") or "default")
    if scope == "user":
        return ("memories", user_id)
    return ("memories", user_id, thread_id)


@tool
def remember(
    content: str,
    kind: Literal["fact", "preference", "todo", "project_state"] = "fact",
    importance: float = 0.7,
    tags: list[str] | None = None,
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> str:
    """Store a long-term memory item (explicit user intent)."""
    start = time.perf_counter()
    mem_id = str(uuid.uuid4())
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    tags_value: list[str] | None = None
    if (
        tags is not None
        and isinstance(tags, Iterable)
        and not isinstance(tags, (str, bytes))
    ):
        tags_value = list(tags)
    payload: dict[str, Any] = {
        "content": str(content),
        "kind": str(kind),
        "importance": float(importance),
        "tags": tags_value,
    }
    store.put(ns, mem_id, payload, index=["content"])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _SuppressTelemetry():
        log_jsonl({
            "chat.memory_saved": True,
            "scope": scope,
            "count": 1,
            "latency_ms": round(elapsed_ms, 2),
        })
    return json.dumps({"ok": True, "memory_id": mem_id})


@tool
def recall_memories(
    query: str,
    limit: int = 5,
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> str:
    """Semantic search across stored memories."""
    start = time.perf_counter()
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
    if store is None:
        return json.dumps({
            "ok": False,
            "error": "memory store unavailable",
            "memories": [],
        })
    ns = _namespace_from_config(config, scope=scope)
    results = store.search(ns, query=str(query), limit=int(limit))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _SuppressTelemetry():
        log_jsonl({
            "chat.memory_searched": True,
            "scope": scope,
            "top_k": int(limit),
            "latency_ms": round(elapsed_ms, 2),
            "result_count": len(results),
        })
    # Return only structured memory content; do not include internal timings from store.
    out = []
    for item in results:
        value = getattr(item, "value", None)
        if not isinstance(value, dict):
            continue
        out.append({
            "id": getattr(item, "key", None),
            "content": value.get("content"),
            "kind": value.get("kind"),
            "importance": value.get("importance"),
            "tags": value.get("tags"),
            "score": getattr(item, "score", None),
        })
    return json.dumps({"ok": True, "memories": out})


@tool
def forget_memory(
    memory_id: str,
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> str:
    """Delete a stored memory item by id."""
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    store.delete(ns, str(memory_id))
    with _SuppressTelemetry():
        log_jsonl({"chat.memory_deleted": True, "scope": scope})
    return json.dumps({"ok": True})


class _SuppressTelemetry:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> bool:
        if exc_type is not None:
            logger.debug(
                "memory tool telemetry suppressed: %s: %s", exc_type.__name__, exc
            )
        return exc_type is not None and issubclass(
            exc_type, (OSError, RuntimeError, ValueError)
        )
