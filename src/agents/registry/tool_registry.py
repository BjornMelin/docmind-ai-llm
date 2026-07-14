"""Canonical worker tool-set construction for the LangGraph supervisor."""

from __future__ import annotations

from typing import Any

from src.config.settings import DocMindSettings, settings


def build_agent_tool_sets(
    app_settings: DocMindSettings = settings,
) -> dict[str, list[Any]]:
    """Build the four production worker tool sets."""
    from src.agents.tools.memory import forget_memory, recall_memories, remember
    from src.agents.tools.ollama_web_tools import get_langchain_web_tools
    from src.agents.tools.planning import plan_query
    from src.agents.tools.retrieval import retrieve_documents
    from src.agents.tools.synthesis import synthesize_results
    from src.agents.tools.validation import validate_response

    retrieval_tools = [
        retrieve_documents,
        recall_memories,
        remember,
        forget_memory,
        *get_langchain_web_tools(app_settings),
    ]
    return {
        "planner_agent": [plan_query],
        "retrieval_agent": retrieval_tools,
        "synthesis_agent": [synthesize_results],
        "validation_agent": [validate_response],
    }


__all__ = ["build_agent_tool_sets"]
