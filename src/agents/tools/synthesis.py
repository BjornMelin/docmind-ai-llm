"""Synthesis helpers extracted from monolithic tools module."""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, TypeGuard, cast

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from loguru import logger

from src.utils.log_safety import build_pii_log_entry

from .constants import MAX_RETRIEVAL_RESULTS
from .retrieval import document_identity


def _synthesis_command(payload: dict[str, Any], *, tool_call_id: str) -> Command:
    """Persist synthesis output while returning its JSON to the model."""
    content = json.dumps(payload, default=str)
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "synthesis_result": payload,
        }
    )


def _invalid_input_command(*, tool_call_id: str, turn_id: str) -> Command:
    """Return the stable synthesis input-error contract."""
    return _synthesis_command(
        {
            "documents": [],
            "error": "Invalid input format",
            "synthesis_metadata": {},
            "turn_id": turn_id,
        },
        tool_call_id=tool_call_id,
    )


def _is_retrieval_batch(value: object) -> TypeGuard[dict[str, Any]]:
    """Return whether decoded JSON matches the current retrieval contract."""
    if not isinstance(value, dict):
        return False
    documents = value.get("documents")
    retrieval_id = value.get("retrieval_id")
    turn_id = value.get("turn_id")
    strategy = value.get("strategy_used")
    retrieval_time = value.get("processing_time_ms")
    return (
        isinstance(documents, list)
        and all(isinstance(document, dict) for document in documents)
        and isinstance(retrieval_id, str)
        and bool(retrieval_id.strip())
        and isinstance(turn_id, str)
        and bool(turn_id.strip())
        and (strategy is None or isinstance(strategy, str))
        and (
            retrieval_time is None
            or (
                not isinstance(retrieval_time, bool)
                and isinstance(retrieval_time, int | float)
            )
        )
    )


def current_retrieval_batches(
    state: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Return validated retrieval batches belonging to the current run only."""
    turn_id_value = state.get("turn_id")
    if not isinstance(turn_id_value, str) or not turn_id_value.strip():
        logger.error("Missing synthesis turn identifier")
        return None
    turn_id = turn_id_value.strip()

    batches = state.get("retrieval_results")
    if not isinstance(batches, list):
        logger.error("Invalid retrieval state shape")
        return None
    current_batches = [
        batch
        for batch in batches
        if isinstance(batch, dict) and batch.get("turn_id") == turn_id
    ]
    if not all(_is_retrieval_batch(batch) for batch in current_batches):
        logger.error("Invalid current-turn retrieval state shape")
        return None
    return [cast(dict[str, Any], batch) for batch in current_batches]


def interleave_retrieval_documents(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fairly merge independent ranked batches while preserving batch rank."""
    document_batches = [
        cast(list[dict[str, Any]], result["documents"]) for result in results
    ]
    unique_documents: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    max_batch_size = max((len(batch) for batch in document_batches), default=0)
    for rank in range(max_batch_size):
        for documents in document_batches:
            if rank >= len(documents):
                continue
            document = documents[rank]
            identity = document_identity(document)
            if identity is not None:
                if identity in seen_identities:
                    continue
                seen_identities.add(identity)
            unique_documents.append(document)
    return unique_documents


def retrieval_batch_watermark(results: list[dict[str, Any]]) -> list[list[str]]:
    """Return the ordered, turn-scoped identities consumed by synthesis."""
    return [
        [cast(str, result["turn_id"]), cast(str, result["retrieval_id"])]
        for result in results
    ]


def _latest_user_query(state: dict[str, Any]) -> str:
    """Return the latest plain-text user message from trusted graph state."""
    messages = state.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


@tool
def synthesize_results(
    state: Annotated[dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Combine current-turn retrieval results from injected graph state."""
    try:
        start_time = time.perf_counter()

        turn_id_value = state.get("turn_id")
        turn_id = turn_id_value.strip() if isinstance(turn_id_value, str) else ""
        results_list = current_retrieval_batches(state)
        if results_list is None:
            return _invalid_input_command(
                tool_call_id=tool_call_id,
                turn_id=turn_id,
            )
        original_query = _latest_user_query(state)

        all_documents: list[dict[str, Any]] = []
        strategies_used: list[str] = []
        total_processing_time = 0.0

        for result in results_list:
            all_documents.extend(cast(list[dict[str, Any]], result["documents"]))

            strategy = result.get("strategy_used")
            if strategy is not None:
                strategy_value = cast(str, strategy)
                if strategy_value not in strategies_used:
                    strategies_used.append(strategy_value)

            retrieval_time = result.get("processing_time_ms")
            if retrieval_time is not None:
                total_processing_time += float(retrieval_time)

        logger.info(
            "Synthesizing {} documents from {} sources",
            len(all_documents),
            len(results_list),
        )

        # Independent queries have no meaningful global score. Round-robin their
        # native rankings so one large batch cannot erase another query's evidence.
        unique_documents = interleave_retrieval_documents(results_list)

        final_documents = unique_documents[:MAX_RETRIEVAL_RESULTS]

        processing_time = time.perf_counter() - start_time

        synthesis_metadata = {
            "original_count": len(all_documents),
            "after_deduplication": len(unique_documents),
            "final_count": len(final_documents),
            "strategies_used": strategies_used,
            "deduplication_ratio": round(
                len(unique_documents) / max(len(all_documents), 1), 2
            ),
            "processing_time_ms": round(processing_time * 1000, 2),
            "total_retrieval_time_ms": total_processing_time,
        }

        result_data: dict[str, Any] = {
            "documents": final_documents,
            "synthesis_metadata": synthesis_metadata,
            "original_query": original_query,
            "retrieval_watermark": retrieval_batch_watermark(results_list),
            "turn_id": turn_id,
        }

        logger.info("Synthesis complete: {} final documents", len(final_documents))
        return _synthesis_command(result_data, tool_call_id=tool_call_id)

    except (RuntimeError, ValueError, AttributeError) as exc:
        redaction = build_pii_log_entry(
            str(exc),
            key_id="agents.tools.synthesis",
        )
        logger.error(
            "Result synthesis failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return _synthesis_command(
            {
                "documents": [],
                "error": "synthesis failed",
                "synthesis_metadata": {},
                "turn_id": (
                    state["turn_id"] if isinstance(state.get("turn_id"), str) else ""
                ),
            },
            tool_call_id=tool_call_id,
        )
