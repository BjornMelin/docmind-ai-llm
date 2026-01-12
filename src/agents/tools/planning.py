"""Routing and planning tools extracted from monolithic module."""

from __future__ import annotations

import json
import time
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from .constants import (
    COMPLEX_CONFIDENCE,
    COMPLEX_QUERY_WORD_THRESHOLD,
    CONFIDENCE_ADJUSTMENT_FACTOR,
    MEDIUM_CONFIDENCE,
    MEDIUM_QUERY_WORD_THRESHOLD,
    RECENT_CHAT_HISTORY_LIMIT,
    SIMPLE_CONFIDENCE,
)


@tool
def route_query(
    query: str,
    state: Annotated[dict | None, InjectedState] = None,
) -> str:
    """Analyze query and determine optimal processing strategy."""
    try:
        start_time = time.perf_counter()

        previous_queries = _extract_previous_queries_from_state(state)

        # Analyze query characteristics
        query_lower = query.lower().strip()
        word_count = len(query.split())

        # Determine base complexity and strategy
        complexity, strategy, needs_planning, confidence = _determine_complexity(
            query_lower, word_count
        )

        # Adjust confidence for context dependence
        context_indicators = ["this", "that", "it", "they", "them", "above", "previous"]
        if (
            any(indicator in query_lower for indicator in context_indicators)
            and not previous_queries
        ):
            confidence *= CONFIDENCE_ADJUSTMENT_FACTOR

        # GraphRAG strategy for relationship queries
        if any(
            pattern in query_lower
            for pattern in ["connect", "relationship", "network", "link"]
        ):
            strategy = "graphrag"

        processing_time = time.perf_counter() - start_time

        decision = {
            "strategy": strategy,
            "complexity": complexity,
            "needs_planning": needs_planning,
            "confidence": confidence,
            "processing_time_ms": round(processing_time * 1000, 2),
            "word_count": word_count,
            "context_dependent": bool(previous_queries),
        }

        logger.info("Query routed: {} complexity, {} strategy", complexity, strategy)
        return json.dumps(decision)

    except Exception as e:
        logger.error("Query routing failed: {}", e)
        raise


def _extract_previous_queries_from_state(state: dict | None) -> list[str]:
    # NOTE: Final-release persistence stores LangChain message objects in state
    # (via LangGraph checkpointer). We avoid LlamaIndex ChatMemoryBuffer here to
    # keep state fully serializable.
    if not isinstance(state, dict):
        return []

    # Reserved flags (no-op in final-release; tests may inject for future use).
    # context_recovery_enabled, reset_context_on_error

    messages = state.get("messages", [])
    if not isinstance(messages, list):
        return []

    previous: list[str] = []
    for msg in messages[:-1]:  # exclude current user query
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", None)
            if content:
                previous.append(str(content))

    return previous[-RECENT_CHAT_HISTORY_LIMIT:]


def _determine_complexity(query_lower: str, word_count: int):
    complex_patterns = [
        "compare",
        "contrast",
        "difference",
        "vs",
        "versus",
        "analyze",
        "breakdown",
        "explain how",
        "step by step",
        "process of",
        "relationship between",
        "impact of",
        "cause and effect",
    ]
    medium_patterns = [
        "find",
        "search",
        "look for",
        "tell me about",
        "what is",
        "how does",
        "why",
        "when",
        "where",
        "list",
        "show me",
    ]
    simple_patterns = ["define", "what is", "who is", "when was", "where is"]

    complexity = "simple"
    strategy = "vector"
    needs_planning = False
    confidence = SIMPLE_CONFIDENCE

    if any(p in query_lower for p in complex_patterns) or (
        word_count > COMPLEX_QUERY_WORD_THRESHOLD
    ):
        return "complex", "hybrid", True, COMPLEX_CONFIDENCE
    if any(p in query_lower for p in medium_patterns) or (
        word_count > MEDIUM_QUERY_WORD_THRESHOLD
    ):
        return "medium", "hybrid", False, MEDIUM_CONFIDENCE
    if any(p in query_lower for p in simple_patterns):
        return "simple", "vector", False, SIMPLE_CONFIDENCE

    return complexity, strategy, needs_planning, confidence


@tool
def plan_query(
    query: str,
    complexity: str,
    _state: Annotated[dict, InjectedState] | None = None,
) -> str:
    """Decompose complex queries into structured sub-tasks."""
    try:
        start_time = time.perf_counter()

        if complexity == "simple":
            # Simple queries don't need decomposition
            plan = {
                "original_query": query,
                "sub_tasks": [query],
                "execution_order": "sequential",
                "estimated_complexity": "low",
            }
            return json.dumps(plan)

        # Decomposition patterns for different query types
        query_lower = query.lower()
        sub_tasks: list[str] = []

        # Comparison queries
        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            entities = []
            for token in query.split():
                tok = token.strip(",.?!")
                if tok.lower() not in ("compare", "vs", "versus", "difference"):
                    entities.append(tok)
            if len(entities) >= 2:
                sub_tasks = [
                    f"Gather information about {entities[0]}",
                    f"Gather information about {entities[1]}",
                    f"Compare {entities[0]} vs {entities[1]} across key dimensions",
                ]

        # Analysis queries
        elif any(
            word in query_lower for word in ["analyze", "analysis", "impact", "effect"]
        ):
            sub_tasks = [
                "Identify key components and variables",
                "Gather background information and context",
                "Collect evidence and examples",
                "Synthesize findings and draw conclusions",
            ]

        # Process/how-to queries
        elif any(
            phrase in query_lower
            for phrase in ["explain how", "how does", "step by step", "process of"]
        ):
            sub_tasks = [
                "Define key terms and concepts",
                "Break down the process into steps",
                "Explain each step with examples",
                "Summarize the overall process and implications",
            ]

        # List/enumeration queries
        elif any(
            word in query_lower for word in ["list", "types", "kinds", "categories"]
        ):
            sub_tasks = [
                "List the main categories",
                "Categorize examples under each category",
                "Organize findings in a clear structure",
            ]

        # Default decomposition for complex queries
        else:
            sub_tasks = [
                "Identify key components and aspects",
                "Gather background information",
                "Collect supporting evidence",
                "Synthesize into a coherent answer",
            ]

        # Determine execution order based on dependencies
        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            execution_order = "parallel"
        else:
            execution_order = "sequential"

        processing_time = time.perf_counter() - start_time

        plan = {
            "original_query": query,
            "sub_tasks": sub_tasks,
            "execution_order": execution_order,
            "estimated_complexity": "high" if len(sub_tasks) >= 3 else "medium",
            "task_count": len(sub_tasks),
            "processing_time_ms": round(processing_time * 1000, 2),
        }

        logger.info(
            "Query planned: {} sub-tasks, {} execution", len(sub_tasks), execution_order
        )
        return json.dumps(plan)

    except (RuntimeError, ValueError, AttributeError) as e:
        logger.error("Query planning failed: {}", e)
        # Fallback plan
        fallback = {
            "original_query": query,
            "sub_tasks": [query],
            "execution_order": "sequential",
            "estimated_complexity": "medium",
            "error": str(e),
        }
        return json.dumps(fallback)
