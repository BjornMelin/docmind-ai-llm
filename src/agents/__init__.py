"""Agents Package for DocMind AI.

This package contains utilities for creating and managing agents for
the DocMind AI application, including the new multi-agent coordination system.

Modules:
    agent_utils: Utilities for creating and managing agents
    agent_factory: Simple agent factory for backward compatibility
    tool_factory: Factory for creating agent tools

    Multi-Agent Coordination System:
    coordinator: Main multi-agent coordinator with LangGraph supervisor
    router: Query routing and complexity analysis agent
    planner: Query planning and decomposition agent
    retrieval: Document retrieval with multi-strategy support
    synthesis: Result synthesis and deduplication agent
    validator: Response validation and quality scoring agent
    tools: Shared @tool functions for agents
"""  # noqa: N999

# Legacy compatibility imports
from .agent_factory import (
    create_agentic_rag_system,
    get_agent_system,
    process_query_with_agent_system,
)
from .agent_utils import (
    analyze_documents_agentic,
    chat_with_agent,
    create_agent_with_tools,
    create_tools_from_index,
)

# Multi-Agent Coordination System imports
from .coordinator import (
    AgentResponse,
    MultiAgentCoordinator,
    create_multi_agent_coordinator,
)
from .planner import (
    PlannerAgent,
    QueryPlan,
    create_planner_agent,
    decompose_comparison_query,
    detect_decomposition_strategy,
)
from .retrieval import (
    RetrievalAgent,
    RetrievalResult,
    create_retrieval_agent,
    optimize_query_for_strategy,
    select_optimal_strategy,
)
from .router import (
    RouterAgent,
    RoutingDecision,
    analyze_query_complexity,
    create_router_agent,
    detect_query_intent,
)
from .supervisor_graph import (
    AgentState,
    SupervisorConfig,
    SupervisorGraph,
    cleanup_supervisor_graph,
    get_supervisor_graph,
    initialize_supervisor_graph,
)
from .synthesis import (
    SynthesisAgent,
    SynthesisResult,
    calculate_content_similarity,
    create_synthesis_agent,
    merge_document_metadata,
)
from .tool_factory import ToolFactory

# Shared tools
from .tools import (
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)
from .validator import (
    ValidationAgent,
    ValidationIssue,
    ValidationResult,
    assess_response_completeness,
    calculate_source_coverage,
    create_validation_agent,
    detect_hallucinations,
)

__all__ = [
    # Legacy compatibility
    "analyze_documents_agentic",
    "chat_with_agent",
    "create_agent_with_tools",
    "create_tools_from_index",
    "create_agentic_rag_system",
    "get_agent_system",
    "process_query_with_agent_system",
    "ToolFactory",
    # Multi-Agent Coordination System
    "MultiAgentCoordinator",
    "AgentResponse",
    "create_multi_agent_coordinator",
    # LangGraph Supervisor System
    "SupervisorGraph",
    "SupervisorConfig",
    "AgentState",
    "get_supervisor_graph",
    "initialize_supervisor_graph",
    "cleanup_supervisor_graph",
    # Individual Agents
    "RouterAgent",
    "RoutingDecision",
    "create_router_agent",
    "analyze_query_complexity",
    "detect_query_intent",
    "PlannerAgent",
    "QueryPlan",
    "create_planner_agent",
    "decompose_comparison_query",
    "detect_decomposition_strategy",
    "RetrievalAgent",
    "RetrievalResult",
    "create_retrieval_agent",
    "optimize_query_for_strategy",
    "select_optimal_strategy",
    "SynthesisAgent",
    "SynthesisResult",
    "create_synthesis_agent",
    "calculate_content_similarity",
    "merge_document_metadata",
    "ValidationAgent",
    "ValidationResult",
    "ValidationIssue",
    "create_validation_agent",
    "detect_hallucinations",
    "calculate_source_coverage",
    "assess_response_completeness",
    # Shared Tools
    "route_query",
    "plan_query",
    "retrieve_documents",
    "synthesize_results",
    "validate_response",
]
