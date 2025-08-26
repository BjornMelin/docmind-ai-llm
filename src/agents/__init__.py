"""Agents Package for DocMind AI.

This package contains utilities for creating and managing agents for
the DocMind AI application, including the new multi-agent coordination system.

Modules:
    coordinator: Main multi-agent coordinator with LangGraph supervisor
    retrieval: Document retrieval with multi-strategy support
    tool_factory: Factory for creating agent tools with optimal configuration
    tools: Shared @tool functions for agents
           (routing, planning, retrieval, synthesis, validation)
"""  # noqa: N999

# Multi-Agent Coordination System imports
from .coordinator import (
    MultiAgentCoordinator,
    create_multi_agent_coordinator,
)
from .models import AgentResponse, MultiAgentState

# Retrieval Agent imports
from .retrieval import (
    RetrievalAgent,
    RetrievalResult,
    optimize_query_for_strategy,
    select_optimal_strategy,
)

# Tool Factory imports
from .tool_factory import ToolFactory

# Shared Tool Functions imports
from .tools import (
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)

__all__ = [
    # Multi-Agent Coordination System
    "AgentResponse",
    "MultiAgentState",
    "MultiAgentCoordinator",
    "create_multi_agent_coordinator",
    # Retrieval Agent
    "RetrievalAgent",
    "RetrievalResult",
    "optimize_query_for_strategy",
    "select_optimal_strategy",
    # Tool Factory
    "ToolFactory",
    # Shared Tool Functions
    "plan_query",
    "retrieve_documents",
    "route_query",
    "synthesize_results",
    "validate_response",
]
