"""Retrieval tools and helpers extracted from monolithic module."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Annotated, Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, ToolRuntime
from langgraph.types import Command
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from loguru import logger

from src.agents.deadlines import remaining_deadline_seconds
from src.agents.models import AgentRuntimeContext
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

from .constants import MAX_RETRIEVAL_RESULTS


def _resolve_router_engine(
    runtime: ToolRuntime[AgentRuntimeContext, dict[str, Any]],
) -> BaseQueryEngine | None:
    """Resolve the router engine from transient tool runtime context.

    Args:
        runtime: The LangGraph ToolRuntime providing caller context.

    Returns:
        The router engine instance if found, otherwise None.
    """
    runtime_ctx = runtime.context
    if isinstance(runtime_ctx, dict):
        router_engine = runtime_ctx.get("router_engine")
        if isinstance(router_engine, BaseQueryEngine):
            return cast(BaseQueryEngine, router_engine)

    return None


def _emit_router_event(*, outcome: str) -> None:
    """Log a PII-safe event for the canonical router boundary.

    Args:
        outcome: Stable router outcome label.
    """
    with contextlib.suppress(Exception):  # pragma: no cover - telemetry
        log_jsonl(
            {
                "retrieval.backend": "llama_index_router",
                "retrieval.outcome": outcome,
            }
        )


def _router_error_response(
    *,
    query: str,
    turn_id: str,
    message: str,
    error_type: str | None = None,
    error_fingerprint: str | None = None,
) -> str:
    """Return the stable JSON error contract for document retrieval.

    Args:
        query: User input query.
        turn_id: Non-model-authored identifier for the current coordinator run.
        message: User-safe error message.
        error_type: Optional exception class name.
        error_fingerprint: Optional redacted diagnostic fingerprint.

    Returns:
        JSON-encoded retrieval error.
    """
    payload: dict[str, Any] = {
        "documents": [],
        "error": message,
        "strategy_used": "router",
        "query_optimized": query,
        "turn_id": turn_id,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if error_fingerprint is not None:
        payload["error_fingerprint"] = error_fingerprint
    return json.dumps(payload, default=str)


def _router_response(
    *,
    query: str,
    turn_id: str,
    documents: list[dict[str, Any]],
    start_time: float,
) -> str:
    """Return the stable JSON success contract for document retrieval."""
    return json.dumps(
        {
            "documents": documents,
            "strategy_used": "router",
            "query_original": query,
            "query_optimized": query,
            "turn_id": turn_id,
            "document_count": len(documents),
            "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "router_used": True,
        },
        default=str,
    )


def _retrieval_command(content: str, *, tool_call_id: str) -> Command:
    """Persist one retrieval batch while returning its JSON to the model."""
    payload = cast(dict[str, Any], json.loads(content))
    # LangGraph injects this hidden argument. Pairing the provider's call ID with
    # the coordinator-owned turn ID gives reducer/delta reconciliation a stable
    # per-turn identity even when a local provider reuses call IDs.
    payload["retrieval_id"] = tool_call_id
    content = json.dumps(payload, default=str)
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "retrieval_results": [payload],
        }
    )


@tool
async def retrieve_documents(
    query: str,
    runtime: ToolRuntime[AgentRuntimeContext, dict[str, Any]],
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve documents through the prebuilt LlamaIndex router.

    The router is the sole owner of vector, hybrid, multimodal, keyword, and
    GraphRAG strategy selection. It is supplied through transient
    ``ToolRuntime.context`` and never persisted in graph state.

    Args:
        query: User input query string.
        runtime: LangGraph ToolRuntime containing the canonical router engine.
        state: Injected LangGraph state used only for conversational recall.
        tool_call_id: Injected identifier for the model-visible tool response.

    Returns:
        A state update containing the retrieval batch and model-visible JSON.
    """
    start_time = time.perf_counter()
    turn_id_value = state.get("turn_id")
    turn_id = turn_id_value.strip() if isinstance(turn_id_value, str) else ""
    if not turn_id:
        _emit_router_event(outcome="turn_missing")
        return _retrieval_command(
            _router_error_response(
                query=query,
                turn_id="",
                message="Retrieval turn unavailable",
            ),
            tool_call_id=tool_call_id,
        )

    router_engine = _resolve_router_engine(runtime)
    if router_engine is None:
        _emit_router_event(outcome="router_missing")
        return _retrieval_command(
            _router_error_response(
                query=query,
                turn_id=turn_id,
                message="Document router unavailable",
            ),
            tool_call_id=tool_call_id,
        )

    try:
        remaining = remaining_deadline_seconds(
            state,
            operation="Document retrieval",
        )
        async with asyncio.timeout(remaining):
            documents = _parse_tool_result(await router_engine.aquery(query))

        # Contextual recall: if user references prior context and retrieval returned
        # nothing, reuse most recent sources from persisted state. This enables
        # "what does that chart show?" flows across reloads without storing images.
        if not documents and _looks_contextual(query):
            recalled = _recall_recent_sources(state)
            if recalled:
                documents.extend(recalled)

        documents = _deduplicate_documents(documents)
        _emit_router_event(outcome="success")
        return _retrieval_command(
            _router_response(
                query=query,
                turn_id=turn_id,
                documents=documents,
                start_time=start_time,
            ),
            tool_call_id=tool_call_id,
        )

    except TimeoutError as exc:
        _emit_router_event(outcome="deadline_exceeded")
        return _retrieval_command(
            _router_error_response(
                query=query,
                turn_id=turn_id,
                message="Document retrieval deadline exceeded",
                error_type=type(exc).__name__,
            ),
            tool_call_id=tool_call_id,
        )
    except Exception as exc:  # external query engines expose heterogeneous errors
        redaction = build_pii_log_entry(
            str(exc), key_id="retrieve_documents.router_query"
        )
        logger.error(
            "Document retrieval failed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        _emit_router_event(outcome="query_failed")
        return _retrieval_command(
            _router_error_response(
                query=query,
                turn_id=turn_id,
                message="Document retrieval failed",
                error_type=type(exc).__name__,
                error_fingerprint=redaction.fingerprint[:12],
            ),
            tool_call_id=tool_call_id,
        )


def document_identity(document: dict[str, Any]) -> str | None:
    """Return a stable retrieved-node identity when one is available."""
    for key in ("id", "node_id"):
        value = document.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for key in ("chunk_id", "page_id"):
        value = metadata.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"
    return None


def _deduplicate_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate stable node identities while preserving native rank order.

    Args:
        documents: Raw list of retrieved document dictionaries.

    Returns:
        Documents in their native order, capped by ``MAX_RETRIEVAL_RESULTS``.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for document in documents:
        identity = document_identity(document)
        if identity is not None:
            if identity in seen:
                continue
            seen.add(identity)
        unique.append(document)
        if len(unique) >= MAX_RETRIEVAL_RESULTS:
            break
    return unique


def _looks_contextual(query: str) -> bool:
    """Detects if a query refers to prior context or visual elements.

    Args:
        query: User input query.

    Returns:
        True if contextual keywords or pronouns are detected.
    """
    # Require a deictic reference. Bare nouns such as "image" or "table" are
    # ordinary retrieval subjects and must not silently recall prior evidence.
    pattern = (
        r"\b(?:this|that|these|those|they|them|above|previous|earlier|mentioned)\b"
        r"|\b(?:this|that|the|above|previous|earlier)\s+"
        r"(?:chart|figure|diagram|table|image|photo)\b"
        r"|\bit\b(?=\s+(?:is|was|seems|refers|referring|about|in|on|of))"
    )
    return re.search(pattern, str(query), flags=re.IGNORECASE) is not None


def _recall_recent_sources(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extracts recent document sources from the graph state for contextual recall.

    Args:
        state: The persisted graph state.

    Returns:
        A list of sanitized document dictionaries from prior turns.
    """
    if not isinstance(state, dict):
        return []
    # Prefer synthesis_result documents (closest to what user saw).
    sr = state.get("synthesis_result")
    if isinstance(sr, dict):
        docs = sr.get("documents")
        if isinstance(docs, list):
            sanitized = [
                _sanitize_document_dict(d) for d in docs if isinstance(d, dict)
            ]
            if sanitized:
                return sanitized

    rr = state.get("retrieval_results")
    if isinstance(rr, list):
        for item in reversed(rr):
            if isinstance(item, dict):
                docs = item.get("documents")
                if isinstance(docs, list):
                    sanitized = [
                        _sanitize_document_dict(d) for d in docs if isinstance(d, dict)
                    ]
                    if sanitized:
                        return sanitized
    return []


def _sanitize_metadata(metadata: object | None) -> dict[str, Any]:
    """Prepares document metadata for visibility in the LLM context window.

    Removes sensitive paths, large binary blobs, and internal tracking fields.

    Args:
        metadata: Raw metadata object or dictionary.

    Returns:
        A sanitized dictionary containing only safe metadata fields.
    """
    sanitized = _sanitize_document_dict({"metadata": metadata}).get("metadata")
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_document_dict(doc: dict[str, Any]) -> dict[str, Any]:
    """Applies security and size constraints to a document data structure.

    Args:
        doc: The document dictionary to sanitize.

    Returns:
        The sanitized document dictionary.
    """

    def _sanitize_metadata_dict(meta: Any) -> dict[str, Any]:
        if not isinstance(meta, dict):
            return {}
        # Never persist raw paths or blobs in agent-visible sources.
        drop = {
            "image_base64",
            "thumbnail_base64",
            "image_path",
            "thumbnail_path",
            "source_path",
            "file_path",
            "path",
        }
        sanitized = {k: v for k, v in meta.items() if k not in drop}
        # Defensive: drop any future *_base64 keys.
        sanitized = {
            k: v for k, v in sanitized.items() if not str(k).endswith("_base64")
        }
        # `source` is frequently used by upstream libraries to carry a path/URI.
        # Preserve only a safe basename when it looks path-like.
        src = sanitized.get("source")
        if isinstance(src, str) and (
            "/" in src or "\\" in src or src.startswith("file:")
        ):
            sanitized["source"] = Path(src).name
        return sanitized

    cleaned = dict(doc)
    # Drop forbidden top-level keys too (some tool stacks return flat dicts).
    for key in list(cleaned.keys()):
        if key in {
            "image_base64",
            "thumbnail_base64",
            "image_path",
            "thumbnail_path",
            "source_path",
            "file_path",
            "path",
        } or str(key).endswith("_base64"):
            cleaned.pop(key, None)

    cleaned["metadata"] = _sanitize_metadata_dict(cleaned.get("metadata"))
    # If a tool returns a top-level `source`, apply the same policy.
    src = cleaned.get("source")
    if isinstance(src, str) and ("/" in src or "\\" in src or src.startswith("file:")):
        cleaned["source"] = Path(src).name
    return cleaned


def _parse_tool_result(result: RESPONSE_TYPE) -> list[dict[str, Any]]:
    """Normalize the canonical non-streaming LlamaIndex response.

    Args:
        result: Response from the configured LlamaIndex router.

    Returns:
        Sanitized source-node dictionaries in native rank order.

    Raises:
        TypeError: If the router returns a noncanonical response type.
    """
    if not isinstance(result, Response):
        raise TypeError("RouterQueryEngine must return a non-streaming Response")

    documents: list[dict[str, Any]] = []
    for node_with_score in result.source_nodes:
        node = node_with_score.node
        documents.append(
            {
                "id": node.node_id,
                "content": node.get_content(),
                "metadata": _sanitize_metadata(node.metadata),
                "score": float(node_with_score.score or 0.0),
            }
        )
    if documents:
        return documents
    return []
