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
import time
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.config import settings
from src.utils.telemetry import log_jsonl


class MemoryCandidate(BaseModel):
    """A potential long-term memory extracted from conversation."""

    content: str
    kind: str = Field(pattern="^(fact|preference|todo|project_state)$")
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
    kind: str = Field(pattern="^(fact|preference|todo|project_state)$")
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


def _ids_from_config(config: RunnableConfig | dict[str, Any] | None) -> tuple[str, str]:
    cfg = config.get("configurable") if isinstance(config, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    user_id = str(cfg.get("user_id") or "local")
    thread_id = str(cfg.get("thread_id") or "default")
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


def _content_matches(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return a.strip().casefold() == b.strip().casefold()


def _prepare_policy(
    policy: MemoryConsolidationPolicy | None,
) -> MemoryConsolidationPolicy:
    if policy is not None:
        return policy
    return MemoryConsolidationPolicy(
        similarity_threshold=float(settings.chat.memory_similarity_threshold),
        low_importance_threshold=float(settings.chat.memory_low_importance_threshold),
        low_importance_ttl_minutes=int(settings.chat.memory_low_importance_ttl_days)
        * 24
        * 60,
        max_items_per_namespace=int(settings.chat.memory_max_items_per_namespace),
        max_candidates_per_turn=int(settings.chat.memory_max_candidates_per_turn),
    )


@tool
def remember(
    content: str,
    kind: Literal["fact", "preference", "todo", "project_state"] = "fact",
    importance: float = 0.7,
    tags: list[str] | None = None,
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime | None = None,
) -> str:
    """Store a long-term memory item (explicit user intent)."""
    start = time.perf_counter()
    mem_id = str(uuid.uuid4())
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    user_id, thread_id = _ids_from_config(config)
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
        log_jsonl(
            {
                "chat.memory_saved": True,
                "scope": scope,
                "count": 1,
                "latency_ms": round(elapsed_ms, 2),
                "thread_id": thread_id,
                "user_id": user_id,
            }
        )
    return json.dumps({"ok": True, "memory_id": mem_id})


@tool
def recall_memories(
    query: str,
    limit: int = 5,
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime | None = None,
) -> str:
    """Semantic search across stored memories."""
    start = time.perf_counter()
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
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
    results = store.search(ns, query=str(query), limit=int(limit))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with _SuppressTelemetry():
        log_jsonl(
            {
                "chat.memory_searched": True,
                "scope": scope,
                "top_k": int(limit),
                "latency_ms": round(elapsed_ms, 2),
                "result_count": len(results),
                "thread_id": thread_id,
                "user_id": user_id,
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
    scope: Literal["session", "user"] = "session",
    runtime: ToolRuntime | None = None,
) -> str:
    """Delete a stored memory item by id."""
    store = runtime.store if runtime is not None else None
    config = runtime.config if runtime is not None else {}
    if store is None:
        return json.dumps({"ok": False, "error": "memory store unavailable"})
    ns = _namespace_from_config(config, scope=scope)
    user_id, thread_id = _ids_from_config(config)
    try:
        store.delete(ns, str(memory_id))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("forget_memory delete failed: {}", exc)
        with _SuppressTelemetry():
            log_jsonl(
                {
                    "chat.memory_deleted": False,
                    "scope": scope,
                    "error": str(exc),
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            )
        return json.dumps({"ok": False, "error": f"delete failed: {exc}"})

    with _SuppressTelemetry():
        log_jsonl(
            {
                "chat.memory_deleted": True,
                "scope": scope,
                "thread_id": thread_id,
                "user_id": user_id,
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
            log_func(
                "memory tool telemetry error: {exc_type}: {exc}",
                exc_type=exc_type.__name__,
                exc=exc,
            )
            return True
        return False


def _recent_dialogue(messages: list[AnyMessage]) -> list[AnyMessage]:
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
        return content.split("```json")[1].split("```")[0].strip()
    if "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content


def _invoke_extraction_llm(llm: Any, prompt: str) -> MemoryExtractionResult | None:
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
        response = structured_llm.invoke(prompt)
        if isinstance(response, MemoryExtractionResult):
            return response
        if isinstance(response, dict):
            return MemoryExtractionResult.model_validate(response)
        return None

    response = llm.invoke(prompt)
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
    messages: list[AnyMessage],
    checkpoint_id: str,
    llm: Any = None,
    *,
    policy: MemoryConsolidationPolicy | None = None,
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
        extracted = _invoke_extraction_llm(llm, prompt)
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

    for cand in candidates:
        # Vector search for similar memories in the same namespace
        results = store.search(
            namespace, query=cand.content, limit=1, filter={"kind": cand.kind}
        )
        if results:
            best_match = results[0]
            # Use score from sqlite-vec if available (higher is more similar in
            # some configs,
            # but usually distance-based). DocMindSqliteStore normalizes search results.
            score = getattr(best_match, "score", 0.0)

            existing_value = (
                best_match.value if isinstance(best_match.value, dict) else None
            )
            existing_kind = (
                existing_value.get("kind") if isinstance(existing_value, dict) else None
            )
            existing_content = (
                existing_value.get("content")
                if isinstance(existing_value, dict)
                else None
            )
            existing_importance = (
                float(existing_value.get("importance", 0.5))
                if isinstance(existing_value, dict)
                else 0.5
            )

            if score is None:
                # Fallback to exact match when semantic scores are unavailable.
                if _content_matches(existing_content, cand.content):
                    if cand.importance > existing_importance:
                        updated_candidate = _with_merged_tags(cand, existing_value)
                        actions.append(
                            ConsolidationAction(
                                action="UPDATE",
                                existing_id=best_match.key,
                                candidate=updated_candidate,
                            )
                        )
                    else:
                        actions.append(
                            ConsolidationAction(
                                action="NOOP", existing_id=best_match.key
                            )
                        )
                else:
                    actions.append(ConsolidationAction(action="ADD", candidate=cand))
                continue
            else:
                score_value = float(score)
                # Consolidation decision rule:
                # - Only consider UPDATE/NOOP when `score_value >= threshold` and
                #   `existing_kind == cand.kind`.
                # - If `_content_matches(existing_content, cand.content)`
                #   (exact/near-exact), require `cand.importance > existing_importance`
                #   before UPDATE; otherwise NOOP.
                #   Exact duplicates are stricter to avoid churn.
                # - If the match is semantic-only (the `else` branch), allow
                #   `cand.importance >= existing_importance` so equal-importance
                #   candidates can replace older entries.
                # - Updates merge tags via `_merge_tags(existing_value, cand)` and
                #   record a
                #   `ConsolidationAction(..., existing_id=best_match.key, ...)`.
                if score_value >= threshold and existing_kind == cand.kind:
                    if _content_matches(existing_content, cand.content):
                        if cand.importance > existing_importance:
                            updated_candidate = _with_merged_tags(cand, existing_value)
                            actions.append(
                                ConsolidationAction(
                                    action="UPDATE",
                                    existing_id=best_match.key,
                                    candidate=updated_candidate,
                                )
                            )
                        else:
                            actions.append(
                                ConsolidationAction(
                                    action="NOOP", existing_id=best_match.key
                                )
                            )
                    else:
                        if cand.importance >= existing_importance:
                            updated_candidate = _with_merged_tags(cand, existing_value)
                            actions.append(
                                ConsolidationAction(
                                    action="UPDATE",
                                    existing_id=best_match.key,
                                    candidate=updated_candidate,
                                )
                            )
                        else:
                            actions.append(
                                ConsolidationAction(
                                    action="NOOP", existing_id=best_match.key
                                )
                            )
                    continue

        # No similar memory found
        actions.append(ConsolidationAction(action="ADD", candidate=cand))

    return actions


def apply_consolidation_policy(
    store: BaseStore,
    namespace: tuple[str, ...],
    actions: list[ConsolidationAction],
    *,
    policy: MemoryConsolidationPolicy | None = None,
) -> int:
    """Apply consolidation actions to the durable store (SPEC-041)."""
    policy = _prepare_policy(policy)
    change_count = 0
    for op in actions:
        if op.action == "ADD" and op.candidate:
            mem_id = str(uuid.uuid4())
            ttl = _candidate_ttl_minutes(op.candidate, policy)
            store.put(
                namespace,
                mem_id,
                op.candidate.model_dump(),
                index=["content"],
                ttl=ttl,
            )
            change_count += 1
        elif op.action == "UPDATE" and op.existing_id and op.candidate:
            ttl = _candidate_ttl_minutes(op.candidate, policy)
            store.put(
                namespace,
                op.existing_id,
                op.candidate.model_dump(),
                index=["content"],
                ttl=ttl,
            )
            change_count += 1
        elif op.action == "DELETE" and op.existing_id:
            store.delete(namespace, op.existing_id)
            change_count += 1

    if policy.max_items_per_namespace > 0:
        change_count += _enforce_namespace_limits(store, namespace, policy)

    if change_count > 0:
        logger.info(
            "Memory consolidation applied: {} changes in namespace {}",
            change_count,
            namespace,
        )
    return change_count


def _enforce_namespace_limits(
    store: BaseStore,
    namespace: tuple[str, ...],
    policy: MemoryConsolidationPolicy,
) -> int:
    max_items = int(policy.max_items_per_namespace)
    if max_items <= 0:
        return 0
    items: list[Any] = []
    offset = 0
    batch_size = max(50, min(500, max_items * 2))
    while True:
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
                "Namespace {} has excessive items ({}); capping eviction scan",
                namespace,
                len(items),
            )
            break

    if len(items) <= max_items:
        return 0
    if len(items) > max_items * 2:
        logger.debug(
            "Namespace {} has {} items (limit {}, batch {}); eviction may be slow",
            namespace,
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

    items_sorted = sorted(items, key=sort_key)
    to_delete = items_sorted[: max(0, len(items) - max_items)]
    deleted = 0
    for item in to_delete:
        key = getattr(item, "key", None)
        if key is None:
            continue
        try:
            store.delete(namespace, str(key))
            deleted += 1
        except Exception as exc:
            logger.debug("Failed to evict memory item: {}", type(exc).__name__)
    return deleted
