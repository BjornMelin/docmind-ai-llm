"""Routing and planning tools extracted from monolithic module."""

from __future__ import annotations

import json
import time
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from src.utils.log_safety import log_error_with_context


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

    except Exception as exc:
        log_error_with_context(exc, operation="plan_query")
        # Fallback plan
        fallback = {
            "original_query": query,
            "sub_tasks": [query],
            "execution_order": "sequential",
            "estimated_complexity": "medium",
            "error_type": type(exc).__name__,
            "error": "planning failed",
        }
        return json.dumps(fallback)
