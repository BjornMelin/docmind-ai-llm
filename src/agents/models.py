"""Agent-specific Pydantic models for DocMind AI Multi-Agent System.

This module contains data models specifically designed for the multi-agent coordination
system, including state management and response schemas. These models are closely
coupled to LangGraph patterns and agent coordination logic.

HYBRID MODEL ORGANIZATION:
Following the hybrid model organization strategy, these models are colocated within
the agents domain because they are:
- Tightly coupled to agent coordination workflows
- Specific to LangGraph supervisor patterns
- Not used outside the agent coordination system
- Domain-specific to multi-agent orchestration

Models:
    AgentResponse: Response model from multi-agent coordination system
    MultiAgentState: Extended state for LangGraph multi-agent coordination

Architecture Decision:
    These models are placed within the agents module following LangGraph best practices
    and domain-driven design principles. They are domain-specific and tightly coupled
    to the agent coordination system, making this the most appropriate location per
    the hybrid organization strategy.
"""

from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Response from ADR-compliant multi-agent coordination system.

    This model represents the structured output from the multi-agent system,
    including content, sources, metadata, and performance metrics specific
    to the agent coordination workflow.

    Attributes:
        content: Generated response content from the multi-agent system
        sources: List of source documents used in generation
        metadata: Agent processing metadata and coordination information
        validation_score: Response validation confidence score (0.0-1.0)
        processing_time: Total processing time in seconds
        optimization_metrics: FP8 optimization and parallel execution metrics
        agent_decisions: Decisions made by agents during processing
        fallback_used: Whether fallback to basic RAG was used
    """

    content: str = Field(description="Generated response content")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source documents"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Agent processing metadata"
    )
    validation_score: float = Field(
        default=0.0, description="Response validation confidence", ge=0.0, le=1.0
    )
    processing_time: float = Field(
        default=0.0, description="Total processing time in seconds"
    )
    optimization_metrics: dict[str, Any] = Field(
        default_factory=dict, description="FP8 and parallel execution metrics"
    )
    agent_decisions: list[dict[str, Any]] = Field(
        default_factory=list, description="Decisions made by agents during processing"
    )
    fallback_used: bool = Field(
        default=False, description="Whether fallback to basic RAG was used"
    )


class MultiAgentGraphState(AgentState[Any]):
    """TypedDict state schema for LangChain/LangGraph agent graphs.

    ``langchain.agents.create_agent`` requires a ``TypedDict`` that extends
    ``AgentState``. The Pydantic ``MultiAgentState`` model remains the local
    validation/serialization helper used by DocMind code and tests.
    """

    tools_data: NotRequired[dict[str, Any]]
    routing_decision: NotRequired[dict[str, Any]]
    planning_output: NotRequired[dict[str, Any]]
    retrieval_results: NotRequired[list[dict[str, Any]]]
    synthesis_result: NotRequired[dict[str, Any]]
    validation_result: NotRequired[dict[str, Any]]
    agent_timings: NotRequired[dict[str, float]]
    total_start_time: NotRequired[float]
    parallel_execution_active: NotRequired[bool]
    token_reduction_achieved: NotRequired[float]
    context_trimmed: NotRequired[bool]
    tokens_trimmed: NotRequired[int]
    kv_cache_usage_gb: NotRequired[float]
    output_mode: NotRequired[str]
    errors: NotRequired[list[str]]
    fallback_used: NotRequired[bool]
    remaining_steps: NotRequired[int]
    timed_out: NotRequired[bool]
    deadline_s: NotRequired[float]


class MultiAgentState(BaseModel):
    """Extended state for ADR-compliant multi-agent coordination.

    This model extends LangGraph's MessagesState to provide comprehensive state
    management for the multi-agent system, including agent decisions, performance
    tracking, context management, and error handling.

    Inherits:
        MessagesState: Base LangGraph state with message handling capabilities

    Additional Fields:
        tools_data: Tool-specific data and configurations
        context: Chat memory buffer for conversation context
        routing_decision: Decision made by routing agent
        planning_output: Output from planning agent
        retrieval_results: Results from retrieval agent
        synthesis_result: Result from synthesis agent
        validation_result: Result from validation agent
        agent_timings: Performance timing for each agent
        total_start_time: Start time of the coordination workflow
        parallel_execution_active: Whether parallel execution is active
        token_reduction_achieved: Token reduction percentage achieved
        context_trimmed: Whether context was trimmed for token limits
        tokens_trimmed: Number of tokens trimmed
        kv_cache_usage_gb: KV cache usage in gigabytes
        output_mode: Output formatting mode
        errors: List of errors encountered during processing
        fallback_used: Whether fallback RAG mode was used
        remaining_steps: Remaining steps for LangGraph supervisor
    """

    # Core state
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    tools_data: dict[str, Any] = Field(default_factory=dict)

    # Agent decisions and results
    routing_decision: dict[str, Any] = Field(default_factory=dict)
    planning_output: dict[str, Any] = Field(default_factory=dict)
    retrieval_results: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_result: dict[str, Any] = Field(default_factory=dict)
    validation_result: dict[str, Any] = Field(default_factory=dict)

    # Performance tracking (ADR-011)
    agent_timings: dict[str, float] = Field(default_factory=dict)
    total_start_time: float = Field(default=0.0)
    parallel_execution_active: bool = Field(default=False)
    token_reduction_achieved: float = Field(default=0.0)

    # Context management (ADR-004, ADR-011)
    context_trimmed: bool = Field(default=False)
    tokens_trimmed: int = Field(default=0)
    kv_cache_usage_gb: float = Field(default=0.0)

    # Output mode configuration (ADR-011)
    output_mode: str = Field(default="structured")

    # Error handling
    errors: list[str] = Field(default_factory=list)
    fallback_used: bool = Field(default=False)

    # LangGraph supervisor requirements (ADR-011)
    remaining_steps: int = Field(
        default=10, description="Remaining steps for supervisor"
    )
